"""
The main user-facing module of ``edges-cal``.

This module contains wrappers around lower-level functions in other modules, providing
a one-stop interface for everything related to calibration.
"""

from __future__ import annotations

import copy
import warnings
from collections import deque
from collections.abc import Callable, Sequence
from functools import partial
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Self

import attrs
import h5py
import hickle
import numpy as np
from astropy import units as un
from astropy.convolution import Gaussian1DKernel, convolve
from astropy.io.misc import yaml as ayaml
from matplotlib import pyplot as plt
from scipy.interpolate import InterpolatedUnivariateSpline as Spline

from .. import types as tp
from ..averaging.averaging import bin_array_unweighted
from ..cached_property import cached_property, safe_property
from ..io import calobsdef, calobsdef3
from ..io.serialization import hickleable
from ..logging import logger
from ..tools import ComplexSpline
from . import loss
from .s11 import S11Model
from . import noise_waves as rcf
from . import reflection_coefficient as rc
from .loss import HotLoadCorrection
from .load_data import Load
from .calibrator import Calibrator

@hickleable
@attrs.define(slots=False)
class CalibrationObservation:
    """
    A composite object representing a full Calibration Observation.

    This includes spectra of all calibrators, and methods to find the calibration
    parameters. It strictly follows Monsalve et al. (2017) in its formalism.
    While by default the class uses the calibrator sources ("ambient", "hot_load",
    "open", "short"), it can be modified to take other sources by setting
    ``CalibrationObservation._sources`` to a new tuple of strings.

    Parameters
    ----------
    loads
        dictionary of load names to Loads
    receiver
        The object defining the reflection coefficient of the receiver.
    cterms
        The number of polynomial terms used for the scaling/offset functions
    wterms
        The number of polynomial terms used for the noise-wave parameters.
    metadata
        Metadata associated with the data.
    """

    loads: dict[str, Load] = attrs.field()
    receiver: S11Model = attrs.field()
    
    _metadata: dict[str, Any] = attrs.field(factory=dict, kw_only=True)

    @property
    def metadata(self):
        """Metadata associated with the object."""
        return self._metadata

    def __attrs_post_init__(self):
        """Set the loads as attributes directly."""
        for k, v in self.loads.items():
            setattr(self, k, v)

    @classmethod
    def from_edges2_caldef(
        cls,
        caldef: calobsdef.CalObsDefEDGES2,
        *,
        freq_bin_size: int = 1,
        spectrum_kwargs: dict[str, dict[str, Any]] | None = None,
        s11_kwargs: dict[str, dict[str, Any]] | None = None,
        internal_switch_kwargs: dict[str, Any] | None = None,
        f_low: tp.FreqType = 40.0 * un.MHz,
        f_high: tp.FreqType = np.inf * un.MHz,
        receiver_kwargs: dict[str, Any] | None = None,
        restrict_s11_model_freqs: bool = True,
        loss_models: dict[str, callable] | None = None,
        **kwargs,
    ) -> Self:
        """Create the object from an edges-io observation.

        Parameters
        ----------
        caldef
            A calibration definition object from which all the data can be read.
        semi_rigid_path : str or Path, optional
            Path to a file containing S11 measurements for the semi rigid cable. Used to
            correct the hot load S11. Found automatically if not given.
        freq_bin_size
            The size of each frequency bin (of the spectra) in units of the raw size.
        spectrum_kwargs
            Keyword arguments used to instantiate the calibrator :class:`LoadSpectrum`
            objects. See its documentation for relevant parameters. Parameters specified
            here are used for _all_ calibrator sources.
        s11_kwargs
            Keyword arguments used to instantiate the calibrator :class:`LoadS11`
            objects. See its documentation for relevant parameters. Parameters specified
            here are used for _all_ calibrator sources.
        internal_switch_kwargs
            Keyword arguments used to instantiate the :class:`~s11.InternalSwitch`
            objects. See its documentation for relevant parameters. The same internal
            switch is used to calibrate the S11 for each input source.
        f_low : float
            Minimum frequency to keep for all loads (and their S11's). If for some
            reason different frequency bounds are desired per-load, one can pass in
            full load objects through ``load_spectra``.
        f_high : float
            Maximum frequency to keep for all loads (and their S11's). If for some
            reason different frequency bounds are desired per-load, one can pass in
            full load objects through ``load_spectra``.
        sources
            A sequence of strings specifying which loads to actually use in the
            calibration. Default is all four standard calibrators.
        receiver_kwargs
            Keyword arguments used to instantiate the calibrator :class:`~s11.Receiver`
            objects. See its documentation for relevant parameters. ``lna_kwargs`` is a
            deprecated alias.
        restrict_s11_model_freqs
            Whether to restrict the S11 modelling (i.e. smoothing) to the given freq
            range. The final output will be calibrated only between the given freq
            range, but the S11 models themselves can be fit over a broader set of
            frequencies.
        """
        loss_models = loss_models or {}
        if "hot_load" not in loss_models and caldef.hot_load.sparams_file is not None:
            loss_models["hot_load"] = HotLoadCorrection.from_file(
                f_low=f_low, f_high=f_high, path=caldef.hot_load.sparams_file
            )

        if "calkit" not in receiver_kwargs:
            receiver_kwargs["calkit"] = rc.get_calkit(
                rc.AGILENT_85033E, resistance_of_match=caldef.receiver_female_resistance
            )

        return cls._from_caldef(
            caldef=caldef, 
            freq_bin_size=freq_bin_size,
            spectrum_kwargs=spectrum_kwargs,
            s11_kwargs=s11_kwargs,
            internal_switch_kwargs=internal_switch_kwargs,
            f_low=f_low,
            f_high=f_high,
            receiver_kwargs=receiver_kwargs,
            restrict_s11_model_freqs=restrict_s11_model_freqs,
            loss_models=loss_models,            
        )

    @classmethod
    def from_edges3_caldef(
        cls,
        caldef: calobsdef3.CalkitEdges3,
        *,
        freq_bin_size: int = 1,
        spectrum_kwargs: dict[str, dict[str, Any]] | None = None,
        s11_kwargs: dict[str, dict[str, Any]] | None = None,
        f_low: tp.FreqType = 40.0 * un.MHz,
        f_high: tp.FreqType = np.inf * un.MHz,
        receiver_kwargs: dict[str, Any] | None = None,
        restrict_s11_model_freqs: bool = True,
        loss_models: dict[str, callable] | None = None,
        **kwargs,
    ) -> Self:
        """Create the object from an edges-io observation.

        Parameters
        ----------
        io_obj
            An calibration observation object from which all the data can be read.
        freq_bin_size
            The size of each frequency bin (of the spectra) in units of the raw size.
        spectrum_kwargs
            Keyword arguments used to instantiate the calibrator :class:`LoadSpectrum`
            objects. See its documentation for relevant parameters. Parameters specified
            here are used for _all_ calibrator sources.
        s11_kwargs
            Keyword arguments used to instantiate the calibrator :class:`LoadS11`
            objects. See its documentation for relevant parameters. Parameters specified
            here are used for _all_ calibrator sources.
        internal_switch_kwargs
            Keyword arguments used to instantiate the :class:`~s11.InternalSwitch`
            objects. See its documentation for relevant parameters. The same internal
            switch is used to calibrate the S11 for each input source.
        f_low : float
            Minimum frequency to keep for all loads (and their S11's). If for some
            reason different frequency bounds are desired per-load, one can pass in
            full load objects through ``load_spectra``.
        f_high : float
            Maximum frequency to keep for all loads (and their S11's). If for some
            reason different frequency bounds are desired per-load, one can pass in
            full load objects through ``load_spectra``.
        sources
            A sequence of strings specifying which loads to actually use in the
            calibration. Default is all four standard calibrators.
        receiver_kwargs
            Keyword arguments used to instantiate the calibrator :class:`~s11.Receiver`
            objects. See its documentation for relevant parameters. ``lna_kwargs`` is a
            deprecated alias.
        restrict_s11_model_freqs
            Whether to restrict the S11 modelling (i.e. smoothing) to the given freq
            range. The final output will be calibrated only between the given freq
            range, but the S11 models themselves can be fit over a broader set of
            frequencies.
        loss_models
            A dictionary of loss models for each source. If a particular source has no
            loss its entry can be missing or None. By default, the only source with loss
            is the hot_load, which uses a 4" cable.
        """
        loss_models = loss_models or {}
        if "hot_load" not in loss_models:
            loss_models["hot_load"] = loss.get_cable_loss_model("UT-141C-SP")

        default_rcv_kw = {
            'calkit': rc.get_calkit(
                rc.AGILENT_ALAN,
                resistance_of_match=49.962 * un.Ohm,
            ),
            "cable_length": 4.26 * un.imperial.inch,
            "cable_loss":-91.5 * un.percent,
            "cable_dielectric": -1.24 * un.percent,
        }
        receiver_kwargs = default_rcv_kw | receiver_kwargs
        
        return cls._from_caldef(
            caldef=caldef, 
            freq_bin_size=freq_bin_size,
            spectrum_kwargs=spectrum_kwargs,
            s11_kwargs=s11_kwargs,
            f_low=f_low,
            f_high=f_high,
            receiver_kwargs=receiver_kwargs,
            restrict_s11_model_freqs=restrict_s11_model_freqs,
            loss_models=loss_models,            
        )


    @classmethod
    def _from_caldef(
        cls,
        caldef: calobsdef3.CalObsDefEDGES3 | calobsdef.CalObsDefEDGES2,
        *,
        freq_bin_size: int = 1,
        spectrum_kwargs: dict[str, dict[str, Any]] | None = None,
        s11_kwargs: dict[str, dict[str, Any]] | None = None,
        internal_switch_kwargs: dict[str, Any] | None = None,
        f_low: tp.FreqType = 40.0 * un.MHz,
        f_high: tp.FreqType = np.inf * un.MHz,
        receiver_kwargs: dict[str, Any] | None = None,
        restrict_s11_model_freqs: bool = True,
        loss_models: dict[str, callable] | None = None,
        **kwargs,
    ) -> Self:
        """Create a CalibrationObservation from a "definition" of all required paths.

        Parameters
        ----------
        caldef
            An calibration observation object from which all the data can be read.
        freq_bin_size
            The size of each frequency bin (of the spectra) in units of the raw size.
        spectrum_kwargs
            Keyword arguments used to instantiate the calibrator :class:`LoadSpectrum`
            objects. See its documentation for relevant parameters. Parameters specified
            here are used for _all_ calibrator sources.
        s11_kwargs
            Keyword arguments used to instantiate the calibrator :class:`LoadS11`
            objects. See its documentation for relevant parameters. Parameters specified
            here are used for _all_ calibrator sources.
        internal_switch_kwargs
            Keyword arguments used to instantiate the :class:`~s11.InternalSwitch`
            objects. See its documentation for relevant parameters. The same internal
            switch is used to calibrate the S11 for each input source.
        f_low : float
            Minimum frequency to keep for all loads (and their S11's). If for some
            reason different frequency bounds are desired per-load, one can pass in
            full load objects through ``load_spectra``.
        f_high : float
            Maximum frequency to keep for all loads (and their S11's). If for some
            reason different frequency bounds are desired per-load, one can pass in
            full load objects through ``load_spectra``.
        sources
            A sequence of strings specifying which loads to actually use in the
            calibration. Default is all four standard calibrators.
        receiver_kwargs
            Keyword arguments used to instantiate the calibrator :class:`~s11.Receiver`
            objects. See its documentation for relevant parameters. ``lna_kwargs`` is a
            deprecated alias.
        restrict_s11_model_freqs
            Whether to restrict the S11 modelling (i.e. smoothing) to the given freq
            range. The final output will be calibrated only between the given freq
            range, but the S11 models themselves can be fit over a broader set of
            frequencies.
        loss_models
            A dictionary of loss models for each source. If a particular source has no
            loss its entry can be missing or None. By default, the only source with loss
            is the hot_load, which uses a 4" cable.
        """
        if f_high < f_low:
            raise ValueError("f_high must be larger than f_low!")

        spectrum_kwargs = spectrum_kwargs or {}
        s11_kwargs = s11_kwargs or {}
        internal_switch_kwargs = internal_switch_kwargs or {}
        receiver_kwargs = receiver_kwargs or {}
        loss_models = loss_models or {}

        for v in [spectrum_kwargs, s11_kwargs, internal_switch_kwargs, receiver_kwargs]:
            assert isinstance(v, dict)

        f_low = f_low.to("MHz", copy=False)
        f_high = f_high.to("MHz", copy=False)

        receiver = S11Model.from_receiver_filespec(
            obs=caldef.receiver_s11,
            f_low=f_low if restrict_s11_model_freqs else 0 * un.MHz,
            f_high=f_high if restrict_s11_model_freqs else np.inf * un.MHz,
            **receiver_kwargs,
        ).with_model_delay()

        if "default" not in spectrum_kwargs:
            spectrum_kwargs["default"] = {}

        if "freq_bin_size" not in spectrum_kwargs["default"]:
            spectrum_kwargs["default"]["freq_bin_size"] = freq_bin_size

        def get_load(name, ambient_temperature=None):
            return Load.from_caldef(
                caldef=caldef,
                load_name=name,
                f_low=f_low,
                f_high=f_high,
                s11_kwargs=s11_kwargs,
                spec_kwargs={
                    **spectrum_kwargs["default"],
                    **spectrum_kwargs.get(name, {}),
                },
                ambient_temperature=ambient_temperature,
                restrict_s11_freqs=restrict_s11_model_freqs,
                loss_model=loss_models.get(name, None),
            )

        amb = get_load("ambient")
        loads = {'ambient': amb}
        
        loads = {
            src: get_load(src, ambient_temperature=amb.temp_ave) for src in ('hot_load', 'open', 'short')
        }

        return cls(
            loads=loads,
            receiver=receiver,
            **kwargs,
        )

    # def with_load_calkit(self, calkit, loads: Sequence[str] | None = None):
    #     """Return a new observation with loads having given calkit."""
    #     if loads is None:
    #         loads = self.load_names
    #     elif isinstance(loads, str):
    #         loads = [loads]

    #     loads = {
    #         name: load.with_calkit(calkit) if name in loads else load
    #         for name, load in self.loads.items()
    #     }

    #     return attr.evolve(self, loads=loads)

    # @safe_property
    # def t_load(self) -> float:
    #     """Assumed temperature of the load."""
    #     return self.loads[next(iter(self.loads.keys()))].t_load

    # @safe_property
    # def t_load_ns(self) -> float:
    #     """Assumed temperature of the load + noise source."""
    #     return self.loads[next(iter(self.loads.keys()))].t_load_ns

    @cached_property
    def freq(self) -> tp.FreqType:
        """The frequencies at which spectra were measured."""
        return self.loads[next(iter(self.loads.keys()))].freq

    # @safe_property
    # def internal_switch(self):
    #     """The S11 object representing the internal switch."""
    #     return self.loads[self.load_names[0]].reflections.internal_switch

    @safe_property
    def load_names(self) -> tuple[str]:
        """Names of the loads."""
        return tuple(self.loads.keys())

    def averaged_spectrum(self, load: Load):
        return load.spectrum.q.data.squeeze() * self.t_load_ns + self.t_load


    @cached_property
    def load_s11_models(self):
        """Dictionary of S11 correction models, one for each source."""
        try:
            return dict(self._injected_source_s11s)
        except (TypeError, AttributeError):
            return {
                name: source.s11_model(self.freq.to_value("MHz"))
                for name, source in self.loads.items()
            }

    @cached_property
    def source_thermistor_temps(self) -> dict[str, float | np.ndarray]:
        """Dictionary of input source thermistor temperatures."""
        if (
            hasattr(self, "_injected_source_temps")
            and self._injected_source_temps is not None
        ):
            return self._injected_source_temps
        return {k: source.temp_ave for k, source in self.loads.items()}


    @cached_property
    def receiver_s11(self):
        """The corrected S11 of the LNA evaluated at the data frequencies."""
        if hasattr(self, "_injected_lna_s11") and self._injected_lna_s11 is not None:
            return self._injected_lna_s11
        return self.receiver.s11_model(self.freq.to_value("MHz"))


    def _load_str_to_load(self, load: Load | str):
        if isinstance(load, str):
            try:
                load = self.loads[load]
            except (AttributeError, KeyError) as e:
                raise AttributeError(
                    f"load must be a Load object or a string (one of {self.load_names})"
                ) from e
        else:
            assert isinstance(load, Load), (
                f"load must be a Load instance, got the {load} {type(Load)}"
            )
        return load

    def get_K(
        self, freq: tp.FreqType | None = None
    ) -> dict[str, tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]]:
        """Get the source-S11-dependent factors of Monsalve (2017) Eq. 7."""
        if freq is None:
            freq = self.freq
            gamma_ants = self.load_s11_models
        else:
            gamma_ants = {
                name: source.s11_model(freq.to_value("MHz"))
                for name, source in self.loads.items()
            }

        lna_s11 = self.receiver.s11_model(freq.to_value("MHz"))
        return {
            name: rcf.get_K(gamma_rec=lna_s11, gamma_ant=gamma_ant)
            for name, gamma_ant in gamma_ants.items()
        }

    def get_calibration_residuals(self, calibrator: Calibrator) -> dict[str, tp.FloatArray]:
        """Get the residuals of calibrated spectra to the known temperatures."""
        return {
            name: calibrator.calibrate_Load(load) - load.temp_ave
            for name, load in self.loads.items()
        }
    
    def get_rms(self, calibrator: Calibrator, smooth: int = 4):
        """Return a dict of RMS values for each source.

        Parameters
        ----------
        smooth : int
            The number of bins over which to smooth residuals before taking the RMS.
        """
        
        resids = self.get_calibration_residuals(calibrator)
        out = {}
        for name, res in resids.items():
            if smooth > 1:
                res = convolve(res, Gaussian1DKernel(stddev=smooth), boundary="extend")
            out[name] = np.sqrt(np.nanmean(res**2))
        return out


    def clone(self, **kwargs):
        """Clone the instance, updating some parameters.

        Parameters
        ----------
        kwargs :
            All parameters to be updated.
        """
        return attrs.evolve(self, **kwargs)
    
    def inject(
        self,
        lna_s11: np.ndarray = None,
        source_s11s: dict[str, np.ndarray] | None = None,
        c1: np.ndarray = None,
        c2: np.ndarray = None,
        t_unc: np.ndarray = None,
        t_cos: np.ndarray = None,
        t_sin: np.ndarray = None,
        averaged_spectra: dict[str, np.ndarray] | None = None,
        thermistor_temp_ave: dict[str, np.ndarray] | None = None,
    ) -> Self:
        """Make a new :class:`CalibrationObservation` based on this, with injections.

        Parameters
        ----------
        lna_s11
            The LNA S11 as a function of frequency to inject.
        source_s11s
            Dictionary of ``{source: S11}`` for each source to inject.
        c1
            Scaling parameter as a function of frequency to inject.
        c2 : [type], optional
            Offset parameter to inject as a function of frequency.
        t_unc
            Uncorrelated temperature to inject (as function of frequency)
        t_cos
            Correlated temperature to inject (as function of frequency)
        t_sin
            Correlated temperature to inject (as function of frequency)
        averaged_spectra
            Dictionary of ``{source: spectrum}`` for each source to inject.

        Returns
        -------
        :class:`CalibrationObservation`
            A new observation object with the injected models.
        """
        new = self.clone()
        f = new.freq.to_value("MHz")
        if lna_s11 is not None:
            new._injected_lna_s11 = lna_s11
            new.receiver = SimpleNamespace(s11_model=ComplexSpline(f, lna_s11))

        if source_s11s is not None:
            new._injected_source_s11s = source_s11s
            new.loads = copy.deepcopy(self.loads)  # make a copy
            for name, s in source_s11s.items():
                new.loads[name].reflections = SimpleNamespace(
                    s11_model=ComplexSpline(f, s)
                )

        new._injected_c1 = c1
        new._injected_c2 = c2
        new._injected_t_unc = t_unc
        new._injected_t_cos = t_cos
        new._injected_t_sin = t_sin
        new._injected_averaged_spectra = averaged_spectra
        new._injected_source_temps = thermistor_temp_ave

        return new

    @classmethod
    def from_yaml(cls, config: tp.PathLike | dict, obs_path: tp.PathLike | None = None):
        """Create the calibration observation from a YAML configuration."""
        if not isinstance(config, dict):
            with open(config) as yml:
                config = ayaml.load(yml)

        iokw = config.pop("data", {})

        if not obs_path:
            obs_path = iokw.pop("path")

        from_def = iokw.pop("compile_from_def", False)

        if from_def:
            io_obs = calobsdef.CalibrationObservation.from_def(obs_path, **iokw)
        else:
            io_obs = calobsdef.CalibrationObservation(obs_path, **iokw)

        return cls.from_edges2_caldef(io_obs, **config)


def perform_term_sweep(
    calobs: CalibrationObservation,
    delta_rms_thresh: float = 0,
    max_cterms: int = 15,
    max_wterms: int = 15,
) -> CalibrationObservation:
    """For a given calibration definition, perform a sweep over number of terms.

    Parameters
    ----------
    calobs: :class:`CalibrationObservation` instance
        The definition calibration class. The `cterms` and `wterms` in this instance
        should define the *lowest* values of the parameters to sweep over.
    delta_rms_thresh : float
        The threshold in change in RMS between one set of parameters and the next that
        will define where to cut off. If zero, will run all sets of parameters up to
        the maximum terms specified.
    max_cterms : int
        The maximum number of cterms to trial.
    max_wterms : int
        The maximum number of wterms to trial.
    """
    cterms = range(calobs.cterms, max_cterms)
    wterms = range(calobs.wterms, max_wterms)

    winner = np.zeros(len(cterms), dtype=int)
    rms = np.ones((len(cterms), len(wterms))) * np.inf

    for i, c in enumerate(cterms):
        for j, w in enumerate(wterms):
            clb = calobs.clone(cterms=c, wterms=w)

            res = clb.get_load_residuals()
            dof = sum(len(r) for r in res.values()) - c - w

            rms[i, j] = np.sqrt(
                sum(np.nansum(np.square(x)) for x in res.values()) / dof
            )

            logger.info(f"Nc = {c:02}, Nw = {w:02}; RMS/dof = {rms[i, j]:1.3e}")

            # If we've decreased by more than the threshold, this wterms becomes
            # the new winner (for this number of cterms)
            if j > 0 and rms[i, j] >= rms[i, j - 1] - delta_rms_thresh:
                winner[i] = j - 1
                break

        if i > 0 and rms[i, winner[i]] >= rms[i - 1, winner[i - 1]] - delta_rms_thresh:
            break

    logger.info(
        f"Best parameters found for Nc={cterms[i - 1]}, "
        f"Nw={wterms[winner[i - 1]]}, "
        f"with RMS = {rms[i - 1, winner[i - 1]]}."
    )

    best = np.unravel_index(np.argmin(rms), rms.shape)
    return calobs.clone(
        cterms=cterms[best[0]],
        wterms=cterms[best[1]],
    )
