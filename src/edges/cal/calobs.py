"""
The main user-facing module of ``edges-cal``.

This module contains wrappers around lower-level functions in other modules, providing
a one-stop interface for everything related to calibration.
"""

import copy
from typing import Any, Self

import attrs
import numpy as np
from astropy import units as un
from astropy.convolution import Gaussian1DKernel, convolve

from edges.cal.s11.base import CalibratedSParams
from edges.cal.s11.s11model import S11ModelParams

from .. import types as tp
from ..cached_property import cached_property, safe_property
from ..io import calobsdef, calobsdef3
from ..io.serialization import hickleable
from . import loss
from . import noise_waves as rcf
from . import reflection_coefficient as rc
from .calibrator import Calibrator
from .load_data import Load
from .loss import LossFunctionGivenSparams
from .s11 import CalibratedS11


@hickleable
@attrs.define(slots=False, kw_only=True, frozen=True)
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
    metadata
        Metadata associated with the data.
    """

    loads: dict[str, Load] = attrs.field()
    receiver: CalibratedS11 = attrs.field()
    _raw_receiver: CalibratedS11 | None = attrs.field(default=None)

    @property
    def ambient(self) -> Load:
        """The ambient load."""
        return self.loads["ambient"]

    @property
    def hot_load(self) -> Load:
        """The hot load."""
        return self.loads["hot_load"]

    @property
    def open(self) -> Load:
        """The open load."""
        return self.loads["open"]

    @property
    def short(self) -> Load:
        """The short load."""
        return self.loads["short"]

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
        loss_model_params: S11ModelParams | None = None,
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
            Keyword arguments used to instantiate the :class:`~s11.CalibratedSParams`
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
        receiver_kwargs = receiver_kwargs or {}
        if "calkit" not in receiver_kwargs:
            receiver_kwargs["calkit"] = rc.get_calkit(
                rc.AGILENT_85033E, resistance_of_match=caldef.receiver_female_resistance
            )

        loss_models = loss_models or {}
        if "hot_load" not in loss_models and caldef.hot_load.sparams_file is not None:
            hot_load_cable_sparams = CalibratedSParams.from_hot_load_semi_rigid(
                caldef.hot_load.sparams_file, f_low=f_low, f_high=f_high
            )

            loss_models["hot_load"] = LossFunctionGivenSparams(hot_load_cable_sparams)

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
            loss_model_params=loss_model_params,
        )

    @classmethod
    def from_edges3_caldef(
        cls,
        caldef: calobsdef3.CalObsDefEDGES3,
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
            Keyword arguments used to instantiate the :class:`~s11.CalibratedSParams`
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

        receiver_kwargs = receiver_kwargs or {}

        default_rcv_kw = {
            "calkit": rc.get_calkit(
                rc.AGILENT_ALAN,
                resistance_of_match=49.962 * un.Ohm,
            ),
            "cable_length": 4.26 * un.imperial.inch,
            "cable_loss_percent": -91.5 * un.percent,
            "cable_dielectric_percent": -1.24 * un.percent,
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
        loss_model_params: S11ModelParams | None = None,
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
            Keyword arguments used to instantiate the :class:`~s11.CalibratedSParams`
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

        rcv_model_params = receiver_kwargs.pop(
            "model_params", S11ModelParams.from_receiver_defaults()
        )
        raw_receiver = CalibratedS11.from_receiver_filespec(
            pathspec=caldef.receiver_s11,
            f_low=f_low if restrict_s11_model_freqs else 0 * un.MHz,
            f_high=f_high if restrict_s11_model_freqs else np.inf * un.MHz,
            **receiver_kwargs,
        )

        if "default" not in spectrum_kwargs:
            spectrum_kwargs["default"] = {}

        if "freq_bin_size" not in spectrum_kwargs["default"]:
            spectrum_kwargs["default"]["freq_bin_size"] = freq_bin_size

        def get_load(name, ambient_temperature=298 * un.K):
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
                loss_model_params=loss_model_params,
            )

        amb = get_load("ambient")
        loads = {"ambient": amb}

        loads |= {
            src: get_load(src, ambient_temperature=amb.temp_ave)
            for src in ("hot_load", "open", "short")
        }

        # Smooth the receiver s11
        receiver = raw_receiver.smoothed(rcv_model_params, freqs=amb.freqs)

        # Smooth the loss models, if necessary:
        for name, loss_model in loss_models.items():
            if (
                isinstance(loss_model, LossFunctionGivenSparams)
                and loss_model.sparams.freqs.size != amb.freqs.size
            ):
                loss_models[name] = attrs.evolve(
                    loss_model,
                    sparams=loss_model.sparams.smoothed(
                        params=S11ModelParams.from_hot_load_cable_defaults(),
                        freqs=amb.freqs,
                    ),
                )

        return cls(
            loads=loads,
            receiver=receiver,
            raw_receiver=raw_receiver,
            **kwargs,
        )

    @cached_property
    def freqs(self) -> tp.FreqType:
        """The frequencies at which spectra were measured."""
        return self.loads[next(iter(self.loads.keys()))].freqs

    @safe_property
    def load_names(self) -> tuple[str]:
        """Names of the loads."""
        return tuple(self.loads.keys())

    def averaged_spectrum(self, load: Load, t_load_ns: float, t_load: float):
        """Compute a quick guess at the calibrated spectrum of a given load."""
        return load.spectrum.q.data.squeeze() * t_load_ns + t_load

    @cached_property
    def load_s11_models(self):
        """Dictionary of S11 correction models, one for each source."""
        return {name: source.s11.s11 for name, source in self.loads.items()}

    @cached_property
    def source_thermistor_temps(self) -> dict[str, tp.TemperatureType]:
        """Dictionary of input source thermistor temperatures."""
        return {k: source.temp_ave for k, source in self.loads.items()}

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
        self,
    ) -> dict[str, tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]]:
        """Get the source-S11-dependent factors of Monsalve (2017) Eq. 7."""
        gamma_ants = {name: load.s11.s11 for name, load in self.loads.items()}

        lna_s11 = self.receiver.s11
        return {
            name: rcf.get_K(gamma_rec=lna_s11, gamma_ant=gamma_ant)
            for name, gamma_ant in gamma_ants.items()
        }

    def get_calibration_residuals(
        self, calibrator: Calibrator
    ) -> dict[str, tp.FloatArray]:
        """Get the residuals of calibrated spectra to the known temperatures."""
        return {
            name: calibrator.calibrate_load(load) - load.temp_ave
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

    @property
    def receiver_s11(self) -> CalibratedS11:
        """The S11 of the receiver."""
        return self.receiver.s11

    def inject(
        self,
        receiver: np.ndarray = None,
        source_s11s: dict[str, np.ndarray] | None = None,
        averaged_q: dict[str, np.ndarray] | None = None,
        thermistor_temp_ave: dict[str, np.ndarray] | None = None,
    ) -> Self:
        """Make a new :class:`CalibrationObservation` based on this, with injections.

        Returns
        -------
        :class:`CalibrationObservation`
            A new observation object with the injected models.
        """
        self.freqs.to_value("MHz")

        kw = {}
        if receiver is not None:
            receiver = CalibratedS11(s11=receiver, freqs=self.freqs)
            kw["receiver"] = receiver

        if (
            source_s11s is not None
            or averaged_q is not None
            or thermistor_temp_ave is not None
        ):
            newloads = copy.deepcopy(self.loads)  # make a copy

            if source_s11s is not None:
                for name, s in source_s11s.items():
                    newloads[name] = attrs.evolve(
                        newloads[name], s11=CalibratedS11(freqs=self.freqs, s11=s)
                    )

            if averaged_q is not None or thermistor_temp_ave is not None:
                for name, s in averaged_q.items():
                    newloads[name] = attrs.evolve(
                        newloads[name],
                        spectrum=attrs.evolve(
                            newloads[name].spectrum,
                            q=newloads[name].spectrum.q.update(
                                data=s[None, None, None]
                            ),
                            temp_ave=thermistor_temp_ave.get(
                                name, newloads[name].temp_ave
                            ),
                        ),
                    )

            kw["loads"] = newloads
        return self.clone(**kw)
