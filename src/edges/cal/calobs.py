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
from . import loss, s11
from . import noise_waves as rcf
from . import reflection_coefficient as rc
from .loss import HotLoadCorrection
from .spectra import LoadSpectrum


@hickleable
@attrs.define(kw_only=True)
class Load:
    """Wrapper class containing all relevant information for a given load.

    Parameters
    ----------
    spectrum : :class:`LoadSpectrum`
        The spectrum for this particular load.
    reflections : :class:`SwitchCorrection`
        The S11 measurements for this particular load.
    hot_load_correction : :class:`HotLoadCorrection`
        If this is a hot load, provide a hot load correction.
    ambient : :class:`LoadSpectrum`
        If this is a hot load, need to provide an ambient spectrum to correct it.
    """

    spectrum: LoadSpectrum = attrs.field()
    reflections: s11.S11Model = attrs.field()
    _loss_model: Callable[[np.ndarray], np.ndarray] | HotLoadCorrection | None = (
        attrs.field(default=None)
    )
    ambient_temperature: float = attrs.field(default=298.0)

    @property
    def loss_model(self):
        """The loss model as a callable function of frequency."""
        if isinstance(self._loss_model, HotLoadCorrection):
            return self._loss_model.power_gain
        return self._loss_model

    @property
    def load_name(self) -> str:
        """The name of the load."""
        return self.reflections.load_name

    @classmethod
    def from_caldef2(
        cls,
        caldef: calobsdef.CalObsDefEDGES2,
        load_name: str,
        ambient_temperature: tp.TemperatureType | None = None,
        f_low: tp.FreqType = 40 * un.MHz,
        f_high: tp.FreqType = np.inf * un.MHz,
        reflection_kwargs: dict | None = None,
        spec_kwargs: dict | None = None,
        loss_kwargs: dict | None = None,
    ):
        """
        Define a full :class:`Load` from a path and name.

        Parameters
        ----------
        path : str or Path
            Path to the top-level calibration observation.
        load_name : str
            Name of a load to define.
        f_low, f_high : float
            Min/max frequencies to keep in measurements.
        reflection_kwargs : dict
            Extra arguments to pass through to :class:`SwitchCorrection`.
        spec_kwargs : dict
            Extra arguments to pass through to :class:`LoadSpectrum`.
        ambient_temperature
            The ambient temperature to use for the loss, if required (required for new
            hot loads). By default, read an ambient load's actual temperature reading
            from the io object.

        Returns
        -------
        load : :class:`Load`
            The load object, containing all info about spectra and S11's for that load.
        """
        if not spec_kwargs:
            spec_kwargs = {}
        if not reflection_kwargs:
            reflection_kwargs = {}
        loss_kwargs = loss_kwargs or {}
        # Fill up kwargs with keywords from this instance
        # TODO: here we only use the calkit defined for the FIRST switching_state,
        # instead of using each calkit for each switching_state. To fix this, we require
        # having meta information inside the S11/ directory.
        if "internal_switch_kwargs" not in reflection_kwargs:
            reflection_kwargs["internal_switch_kwargs"] = {}

        if "calkit" not in reflection_kwargs["internal_switch_kwargs"]:
            reflection_kwargs["internal_switch_kwargs"]["calkit"] = rc.get_calkit(
                rc.AGILENT_85033E, resistance_of_match=caldef.male_resistance
            )

        # For the LoadSpectrum, we can specify both f_low/f_high and f_range_keep.
        # The first pair is what defines what gets read in and smoothed/averaged.
        # The second pair then selects a part of this range to keep for doing
        # calibration with.
        if "f_low" not in spec_kwargs:
            spec_kwargs["f_low"] = f_low
        if "f_high" not in spec_kwargs:
            spec_kwargs["f_high"] = f_high

        loaddef = getattr(caldef, load_name)
        spec = LoadSpectrum.from_loaddef(
            loaddef=loaddef,
            f_range_keep=(f_low, f_high),
            **spec_kwargs,
        )

        refl = s11.LoadS11.from_caldef(
            loaddef.s11,
            caldef.switching_state,
            f_low=f_low,
            f_high=f_high,
            **reflection_kwargs,
        )

        if loaddef.name == "hot_load":
            hlc = HotLoadCorrection.from_file(f_low=f_low, f_high=f_high, **loss_kwargs)

            return cls(
                spectrum=spec,
                reflections=refl,
                loss_model=hlc,
                ambient_temperature=ambient_temperature,
            )
        return cls(spectrum=spec, reflections=refl)

    @classmethod
    def from_caldef3(
        cls,
        io_obj: calobsdef3.CalkitEdges3,
        load_name: str,
        f_low: tp.FreqType = 40 * un.MHz,
        f_high: tp.FreqType = np.inf * un.MHz,
        reflection_kwargs: dict | None = None,
        spec_kwargs: dict | None = None,
        loss_model: callable | None = None,
        ambient_temperature: float | None = None,
        restrict_s11_freqs: bool = False,
    ):
        """
        Define a full :class:`Load` from a path and name.

        Parameters
        ----------
        path : str or Path
            Path to the top-level calibration observation.
        load_name : str
            Name of a load to define.
        f_low, f_high : float
            Min/max frequencies to keep in measurements.
        reflection_kwargs : dict
            Extra arguments to pass through to :class:`SwitchCorrection`.
        spec_kwargs : dict
            Extra arguments to pass through to :class:`LoadSpectrum`.
        ambient_temperature
            The ambient temperature to use for the loss, if required (required for new
            hot loads). By default, read an ambient load's actual temperature reading
            from the io object.

        Returns
        -------
        load : :class:`Load`
            The load object, containing all info about spectra and S11's for that load.
        """
        if not spec_kwargs:
            spec_kwargs = {}
        if not reflection_kwargs:
            reflection_kwargs = {}

        # For the LoadSpectrum, we can specify both f_low/f_high and f_range_keep.
        # The first pair is what defines what gets read in and smoothed/averaged.
        # The second pair then selects a part of this range to keep for doing
        # calibration with.
        if "f_low" not in spec_kwargs:
            spec_kwargs["f_low"] = f_low
        if "f_high" not in spec_kwargs:
            spec_kwargs["f_high"] = f_high

        spec = LoadSpectrum.from_edges3(
            io_obs=io_obj,
            load_name=load_name,
            f_range_keep=(f_low, f_high),
            **spec_kwargs,
        )

        refl = s11.LoadS11.from_edges3(
            obs=io_obj,
            load_name=load_name,
            f_low=f_low if restrict_s11_freqs else 0 * un.MHz,
            f_high=f_high if restrict_s11_freqs else np.inf * un.MHz,
            **reflection_kwargs,
        )

        if refl.model_delay == 0 * un.s:
            refl = refl.with_model_delay()

        return cls(
            spectrum=spec,
            reflections=refl,
            ambient_temperature=ambient_temperature,
            loss_model=loss_model,
        )

    def loss(self, freq: tp.FreqType | None = None):
        """The loss of this load."""
        if freq is None:
            freq = self.freq

        if self.loss_model is None:
            return np.ones(len(freq))

        return self.loss_model(freq, self.reflections.s11_model(freq))

    def get_temp_with_loss(self, freq: tp.FreqType | None = None):
        """Calculate the temperature of the load accounting for loss."""
        if self.loss_model is None:
            return self.spectrum.temp_ave

        gain = self.loss(freq)
        return gain * self.spectrum.temp_ave + (1 - gain) * self.ambient_temperature

    @cached_property
    def temp_ave(self) -> np.ndarray:
        """The average temperature of the thermistor (over frequency and time)."""
        return self.get_temp_with_loss()

    @property
    def averaged_Q(self) -> np.ndarray:
        """The average spectrum power ratio, Q (over time)."""
        return self.spectrum.q

    # @property
    # def t_load(self) -> float:
    #     """The assumed temperature of the internal load."""
    #     return self.spectrum.t_load

    # @property
    # def t_load_ns(self) -> float:
    #     """The assumed temperature of the internal load + noise source."""
    #     return self.spectrum.t_load_ns

    @property
    def s11_model(self) -> Callable[[np.ndarray], np.ndarray]:
        """Callable S11 model as function of frequency."""
        return self.reflections.s11_model

    @property
    def freq(self) -> tp.FreqType:
        """Frequencies of the spectrum."""
        return self.spectrum.q.freqs

    def with_calkit(self, calkit: rc.Calkit):
        """Return a new Load with updated calkit."""
        if "calkit" not in self.reflections.metadata:
            raise RuntimeError(
                "Cannot clone with new calkit since calkit is unknown for the load"
            )

        loads11 = [
            attr.evolve(x, calkit=calkit)
            for x in self.reflections.metadata["load_s11s"]
        ]
        isw = self.reflections.internal_switch.with_new_calkit(calkit)

        return attr.evolve(
            self,
            reflections=s11.LoadS11.from_load_and_internal_switch(
                load_s11=loads11, internal_switch=isw, base=self.reflections
            ),
        )


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
    receiver: s11.Receiver = attrs.field()
    cterms: int = attrs.field(default=5, kw_only=True)
    wterms: int = attrs.field(default=7, kw_only=True)
    apply_loss_to_true_temp: bool = attrs.field(default=False, kw_only=True)
    smooth_scale_offset_within_loop: bool = attrs.field(default=False, kw_only=True)
    cable_delay_sweep: np.ndarray = attrs.field(default=np.array([0]))
    ncal_iter: int = attrs.field(default=4, kw_only=True)
    fit_method: str = attrs.field(default="lstsq")
    scale_offset_poly_spacing: float = attrs.field(default=1.0, converter=float)
    t_load: float = attrs.field(default=400)
    t_load_ns: float = attrs.field(default=300)

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
    def from_caldef2(
        cls,
        caldef: calobsdef.CalObsDefEDGES2,
        *,
        freq_bin_size: int = 1,
        spectrum_kwargs: dict[str, dict[str, Any]] | None = None,
        s11_kwargs: dict[str, dict[str, Any]] | None = None,
        internal_switch_kwargs: dict[str, Any] | None = None,
        f_low: tp.FreqType = 40.0 * un.MHz,
        f_high: tp.FreqType = np.inf * un.MHz,
        sources: tuple[str] = ("ambient", "hot_load", "open", "short"),
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

        receiver = s11.Receiver.from_io(
            pathspec=caldef.receiver_s11,
            f_low=f_low if restrict_s11_model_freqs else 0 * un.MHz,
            f_high=f_high if restrict_s11_model_freqs else np.inf * un.MHz,
            **receiver_kwargs,
        )

        if "default" not in spectrum_kwargs:
            spectrum_kwargs["default"] = {}

        if "freq_bin_size" not in spectrum_kwargs["default"]:
            spectrum_kwargs["default"]["freq_bin_size"] = freq_bin_size

        def get_load(name, ambient_temperature=None):
            return Load.from_caldef2(
                caldef=caldef,
                load_name=name,
                f_low=f_low,
                f_high=f_high,
                reflection_kwargs={
                    **s11_kwargs.get("default", {}),
                    **s11_kwargs.get(name, {}),
                    "internal_switch_kwargs": internal_switch_kwargs,
                },
                spec_kwargs={
                    **spectrum_kwargs["default"],
                    **spectrum_kwargs.get(name, {}),
                },
                loss_kwargs={**hot_load_loss_kwargs, "path": semi_rigid_path},
                ambient_temperature=ambient_temperature,
            )

        hlc = HotLoadCorrection.from_file(f_low=f_low, f_high=f_high, **loss_kwargs)

        loads = {}
        for src in sources:
            loads[src] = get_load(
                src,
                ambient_temperature=loads["ambient"].spectrum.temp_ave
                if src == "hot_load"
                else None,
            )

        return cls(
            loads=loads,
            receiver=receiver,
            # metadata={
            #     "s11_kwargs": s11_kwargs,
            #     "lna_kwargs": receiver_kwargs,
            #     "spectra": {
            #         name: load.spectrum.metadata for name, load in loads.items()
            #     },
            #     "io": caldef,
            # },
            **kwargs,
        )

    @classmethod
    def from_caldef3(
        cls,
        caldef: calobsdef3.CalkitEdges3,
        *,
        freq_bin_size: int = 1,
        spectrum_kwargs: dict[str, dict[str, Any]] | None = None,
        s11_kwargs: dict[str, dict[str, Any]] | None = None,
        internal_switch_kwargs: dict[str, Any] | None = None,
        f_low: tp.FreqType = 40.0 * un.MHz,
        f_high: tp.FreqType = np.inf * un.MHz,
        sources: tuple[str] = ("ambient", "hot_load", "open", "short"),
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

        if "calkit" not in receiver_kwargs:
            receiver_kwargs["calkit"] = rc.get_calkit(
                rc.AGILENT_ALAN,
                resistance_of_match=49.962 * un.Ohm,
            )

        receiver = s11.Receiver.from_edges3(
            obs=caldef,
            f_low=f_low if restrict_s11_model_freqs else 0 * un.MHz,
            f_high=f_high if restrict_s11_model_freqs else np.inf * un.MHz,
            **receiver_kwargs,
        ).with_model_delay()

        if "default" not in spectrum_kwargs:
            spectrum_kwargs["default"] = {}

        if "freq_bin_size" not in spectrum_kwargs["default"]:
            spectrum_kwargs["default"]["freq_bin_size"] = freq_bin_size

        def get_load(name, ambient_temperature=None):
            return Load.from_edges3(
                io_obj=caldef,
                load_name=name,
                f_low=f_low,
                f_high=f_high,
                reflection_kwargs={
                    "calkit": receiver_kwargs["calkit"],
                    **s11_kwargs.get("default", {}),
                    **s11_kwargs.get(name, {}),
                },
                spec_kwargs={
                    **spectrum_kwargs["default"],
                    **spectrum_kwargs.get(name, {}),
                },
                ambient_temperature=ambient_temperature,
                restrict_s11_freqs=restrict_s11_model_freqs,
                loss_model=loss_models.get(name, None),
            )

        amb = get_load("ambient")
        loads = {
            src: get_load(src, ambient_temperature=amb.temp_ave) for src in sources
        }

        return cls(
            loads=loads,
            receiver=receiver,
            # metadata={
            #     "path": caldef.temperature_file.parent.parent,
            #     "s11_kwargs": s11_kwargs,
            #     "lna_kwargs": receiver_kwargs,
            #     "spectra": {
            #         name: load.spectrum.metadata for name, load in loads.items()
            #     },
            #     "io": caldef,
            # },
            **kwargs,
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
        sources: tuple[str] = ("ambient", "hot_load", "open", "short"),
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

        receiver = s11.Receiver.from_edges3(
            obs=caldef,
            f_low=f_low if restrict_s11_model_freqs else 0 * un.MHz,
            f_high=f_high if restrict_s11_model_freqs else np.inf * un.MHz,
            **receiver_kwargs,
        ).with_model_delay()

        if "default" not in spectrum_kwargs:
            spectrum_kwargs["default"] = {}

        if "freq_bin_size" not in spectrum_kwargs["default"]:
            spectrum_kwargs["default"]["freq_bin_size"] = freq_bin_size

        def get_load(name, ambient_temperature=None):
            return Load.from_edges3(
                io_obj=caldef,
                load_name=name,
                f_low=f_low,
                f_high=f_high,
                reflection_kwargs=s11_kwargs,
                spec_kwargs={
                    **spectrum_kwargs["default"],
                    **spectrum_kwargs.get(name, {}),
                },
                ambient_temperature=ambient_temperature,
                restrict_s11_freqs=restrict_s11_model_freqs,
                loss_model=loss_models.get(name, None),
            )

        amb = get_load("ambient")
        loads = {
            src: get_load(src, ambient_temperature=amb.temp_ave) for src in sources
        }

        return cls(
            loads=loads,
            receiver=receiver,
            **kwargs,
        )

    def with_load_calkit(self, calkit, loads: Sequence[str] | None = None):
        """Return a new observation with loads having given calkit."""
        if loads is None:
            loads = self.load_names
        elif isinstance(loads, str):
            loads = [loads]

        loads = {
            name: load.with_calkit(calkit) if name in loads else load
            for name, load in self.loads.items()
        }

        return attr.evolve(self, loads=loads)

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

    @safe_property
    def internal_switch(self):
        """The S11 object representing the internal switch."""
        return self.loads[self.load_names[0]].reflections.internal_switch

    @safe_property
    def load_names(self) -> tuple[str]:
        """Names of the loads."""
        return tuple(self.loads.keys())

    # def new_load(
    #     self,
    #     load_name: str,
    #     io_obj: calobsdef.CalibrationObservation,
    #     reflection_kwargs: dict | None = None,
    #     spec_kwargs: dict | None = None,
    # ):
    #     """Create a new load with the given load name.

    #     Uses files inside the current observation.

    #     Parameters
    #     ----------
    #     load_name : str
    #         The name of the load
    #     run_num_spec : dict or int
    #         Run number to use for the spectrum.
    #     run_num_load : dict or int
    #         Run number to use for the load's S11.
    #     reflection_kwargs : dict
    #         Keyword arguments to construct the :class:`SwitchCorrection`.
    #     spec_kwargs : dict
    #         Keyword arguments to construct the :class:`LoadSpectrum`.
    #     """
    #     reflection_kwargs = reflection_kwargs or {}
    #     spec_kwargs = spec_kwargs or {}

    #     spec_kwargs["freq_bin_size"] = self.metadata.get("freq_bin_size", 1)
    #     spec_kwargs["t_load"] = self.open.spectrum.t_load
    #     spec_kwargs["t_load_ns"] = self.open.spectrum.t_load_ns

    #     if "frequency_smoothing" not in spec_kwargs:
    #         spec_kwargs["frequency_smoothing"] = self.open.spectrum.metadata[
    #             "frequency_smoothing"
    #         ]

    #     spec_kwargs["f_low"], spec_kwargs["f_high"] = self.open.spectrum.metadata[
    #         "pre_smooth_freq_range"
    #     ]

    #     return Load.from_io(
    #         io_obj=io_obj,
    #         load_name=load_name,
    #         f_low=self.freq._f_low,
    #         f_high=self.freq._f_high,
    #         reflection_kwargs=reflection_kwargs,
    #         spec_kwargs=spec_kwargs,
    #     )

    def plot_raw_spectra(self, fig=None, ax=None) -> plt.Figure:
        """
        Plot raw uncalibrated spectra for all calibrator sources.

        Parameters
        ----------
        fig : :class:`plt.Figure`
            A matplotlib figure on which to make the plot. By default creates a new one.
        ax : :class:`plt.Axes`
            A matplotlib Axes on which to make the plot. By default creates a new one.

        Returns
        -------
        fig : :class:`plt.Figure`
            The figure on which the plot was made.
        """
        if fig is None and ax is None:
            fig, ax = plt.subplots(
                len(self.loads), 1, sharex=True, gridspec_kw={"hspace": 0.05}
            )

        for i, (name, load) in enumerate(self.loads.items()):
            ax[i].plot(load.freq, self.averaged_spectrum(load))
            ax[i].set_ylabel("$T^*$ [K]")
            ax[i].set_title(name)
            ax[i].grid(True)
        ax[-1].set_xlabel("Frequency [MHz]")

        return fig

    def averaged_spectrum(self, load: Load):
        return load.spectrum.q.data.squeeze() * self.t_load_ns + self.t_load

    def plot_s11_models(self, **kwargs):
        """
        Plot residuals of S11 models for all sources.

        Returns
        -------
        dict:
            Each entry has a key of the source name, and the value is a matplotlib fig.
        """
        fig, ax = plt.subplots(
            4,
            len(self.loads) + 1,
            figsize=((len(self.loads) + 1) * 4, 6),
            sharex=True,
            gridspec_kw={"hspace": 0.05},
            layout="constrained",
        )

        for i, (name, source) in enumerate(self.loads.items()):
            source.reflections.plot_residuals(ax=ax[:, i], title=False, **kwargs)
            ax[0, i].set_title(name)

        self.receiver.plot_residuals(ax=ax[:, -1], title=False, **kwargs)
        ax[0, -1].set_title("Receiver")
        return ax

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
    def cal_coefficient_models(self):
        """The calibration coefficient models."""
        if (
            hasattr(self, "_injected_averaged_spectra")
            and self._injected_averaged_spectra is not None
        ):
            ave_spec = self._injected_averaged_spectra
        else:
            ave_spec = {
                k: self.averaged_spectrum(source) for k, source in self.loads.items()
            }

        if self.apply_loss_to_true_temp:
            temp_ant = self.source_thermistor_temps
            loss = None
        else:
            temp_ant = dict(self.source_thermistor_temps)
            temp_ant["hot_load"] = self.hot_load.spectrum.temp_ave
            loss = self.hot_load.loss()

        scale, off, nwp = deque(
            rcf.get_calibration_quantities_iterative(
                self.freq.to_value("MHz"),
                temp_raw=ave_spec,
                gamma_rec=self.receiver.s11_model,
                gamma_ant={name: load.s11_model for name, load in self.loads.items()},
                temp_ant=temp_ant,
                cterms=self.cterms,
                wterms=self.wterms,
                temp_amb_internal=self.t_load,
                hot_load_loss=loss,
                smooth_scale_offset_within_loop=self.smooth_scale_offset_within_loop,
                delays_to_fit=self.cable_delay_sweep,
                niter=self.ncal_iter,
                poly_spacing=self.scale_offset_poly_spacing,
            ),
            maxlen=1,
        ).pop()
        return {"C1": scale, "C2": off, "NW": nwp}

    def C1(self, f: tp.FreqType | None = None):
        """
        Scaling calibration parameter.

        Parameters
        ----------
        f : array-like
            The frequencies at which to evaluate C1. By default, the frequencies of this
            instance.
        """
        if hasattr(self, "_injected_c1") and self._injected_c1 is not None:
            return np.array(self._injected_c1)

        if f is None:
            f = self.freq.to_value("MHz")
        elif hasattr(f, "unit"):
            f = f.to_value("MHz")

        return self.cal_coefficient_models["C1"](f)

    def C2(self, f: tp.FreqType | None = None):
        """
        Offset calibration parameter.

        Parameters
        ----------
        f : array-like
            The frequencies at which to evaluate C2. By default, the frequencies of this
            instance.
        """
        if hasattr(self, "_injected_c2") and self._injected_c2 is not None:
            return np.array(self._injected_c2)

        if f is None:
            f = self.freq.to_value("MHz")
        elif hasattr(f, "unit"):
            f = f.to_value("MHz")

        return self.cal_coefficient_models["C2"](f)

    def Tunc(self, f: tp.FreqType | None = None):
        """
        Uncorrelated noise-wave parameter.

        Parameters
        ----------
        f : array-like
            The frequencies at which to evaluate Tunc. By default, the frequencies of
            thisinstance.
        """
        if hasattr(self, "_injected_t_unc") and self._injected_t_unc is not None:
            return np.array(self._injected_t_unc)

        if f is None:
            f = self.freq.to_value("MHz")
        elif hasattr(f, "unit"):
            f = f.to_value("MHz")

        return self.cal_coefficient_models["NW"].get_tunc(f)

    def Tcos(self, f: tp.FreqType | None = None):
        """
        Cosine noise-wave parameter.

        Parameters
        ----------
        f : array-like
            The frequencies at which to evaluate Tcos. By default, the frequencies of
            this instance.
        """
        if hasattr(self, "_injected_t_cos") and self._injected_t_cos is not None:
            return np.array(self._injected_t_cos)

        if f is None:
            f = self.freq.to_value("MHz")
        elif hasattr(f, "unit"):
            f = f.to_value("MHz")

        return self.cal_coefficient_models["NW"].get_tcos(f)

    def Tsin(self, f: tp.FreqType | None = None):
        """
        Sine noise-wave parameter.

        Parameters
        ----------
        f : array-like
            The frequencies at which to evaluate Tsin. By default, the frequencies of
            this instance.
        """
        if hasattr(self, "_injected_t_sin") and self._injected_t_sin is not None:
            return np.array(self._injected_t_sin)

        if f is None:
            f = self.freq.to_value("MHz")
        elif hasattr(f, "unit"):
            f = f.to_value("MHz")

        return self.cal_coefficient_models["NW"].get_tsin(f)

    @cached_property
    def receiver_s11(self):
        """The corrected S11 of the LNA evaluated at the data frequencies."""
        if hasattr(self, "_injected_lna_s11") and self._injected_lna_s11 is not None:
            return self._injected_lna_s11
        return self.receiver.s11_model(self.freq.to_value("MHz"))

    def get_linear_coefficients(self, load: Load | str):
        """
        Calibration coefficients a,b such that T = aT* + b (derived from Eq. 7).

        Parameters
        ----------
        load : str or :class:`Load`
            The load for which to get the linear coefficients.
        """
        if isinstance(load, str):
            load_s11 = self.load_s11_models[load]
        elif load.load_name in self.load_s11_models:
            load_s11 = self.load_s11_models[load.load_name]
        else:
            load_s11 = load.s11_model(self.freq.to_value("MHz"))

        return rcf.get_linear_coefficients(
            load_s11,
            self.receiver_s11,
            self.C1(self.freq),
            self.C2(self.freq),
            self.Tunc(self.freq),
            self.Tcos(self.freq),
            self.Tsin(self.freq),
            t_load=self.t_load,
        )

    def calibrate(self, load: Load | str, q=None, temp=None):
        """
        Calibrate the temperature of a given load.

        Parameters
        ----------
        load : :class:`Load` or str
            The load to calibrate.

        Returns
        -------
        array : calibrated antenna temperature in K, len(f).
        """
        load = self._load_str_to_load(load)
        a, b = self.get_linear_coefficients(load)

        if q is not None:
            temp = self.t_load_ns * q + self.t_load
        elif temp is None:
            temp = self.averaged_spectrum(load)

        return a * temp + b

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

    def decalibrate(self, temp: np.ndarray, load: Load | str, freq: np.ndarray = None):
        """
        Decalibrate a temperature spectrum, yielding uncalibrated T*.

        Parameters
        ----------
        temp : array_like
            A temperature spectrum, with the same length as `freq`.
        load : str or :class:`Load`
            The load to calibrate.
        freq : array-like
            The frequencies at which to decalibrate. By default, the frequencies of the
            instance.

        Returns
        -------
        array_like : T*, the normalised uncalibrated temperature.
        """
        if freq is None:
            freq = self.freq

        if freq.min() < self.freq.min():
            warnings.warn(
                "The minimum frequency is outside the calibrated range "
                f"({self.freq.min()} - {self.freq.max()} MHz)",
                stacklevel=2,
            )

        if freq.max() > self.freq.max():
            warnings.warn(
                "The maximum frequency is outside the calibrated range",
                stacklevel=2,
            )

        a, b = self.get_linear_coefficients(load)
        return (temp - b) / a

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

    def plot_calibrated_temp(
        self,
        load: Load | str,
        bins: int = 2,
        fig=None,
        ax=None,
        xlabel=True,
        ylabel=True,
        label: str = "",
        as_residuals: bool = False,
        load_in_title: bool = False,
        rms_in_label: bool = True,
    ):
        """
        Make a plot of calibrated temperature for a given source.

        Parameters
        ----------
        load : :class:`~LoadSpectrum` instance
            Source to plot.
        bins : int
            Number of bins to smooth over (std of Gaussian kernel)
        fig : Figure
            Optionally provide a matplotlib figure to add to.
        ax : Axis
            Optionally provide a matplotlib Axis to add to.
        xlabel : bool
            Whether to write the x-axis label
        ylabel : bool
            Whether to write the y-axis label

        Returns
        -------
        fig :
            The matplotlib figure that was created.
        """
        load = self._load_str_to_load(load)

        if fig is None and ax is None:
            fig, ax = plt.subplots(1, 1, facecolor="w")

        # binning
        temp_calibrated = self.calibrate(load)
        if bins > 0:
            freq_ave_cal = bin_array_unweighted(temp_calibrated, size=bins)
            f = bin_array_unweighted(self.freq.to_value("MHz"), size=bins)
        else:
            freq_ave_cal = temp_calibrated
            f = self.freq.to_value("MHz")

        freq_ave_cal[np.isinf(freq_ave_cal)] = np.nan

        rms = np.sqrt(np.mean((freq_ave_cal - np.mean(freq_ave_cal)) ** 2))

        ax.plot(
            f,
            freq_ave_cal,
            label=f"Calibrated {load.load_name} [RMS = {rms:.3f}]",
        )

        temp_ave = self.source_thermistor_temps.get(load.load_name, load.temp_ave)

        if np.isscalar(temp_ave):
            temp_ave = np.ones(self.freq.size) * temp_ave

        ax.plot(
            self.freq,
            temp_ave,
            color="C2",
            label="Average thermistor temp",
        )

        ax.set_ylim([np.nanmin(freq_ave_cal), np.nanmax(freq_ave_cal)])
        if xlabel:
            ax.set_xlabel("Frequency [MHz]")

        if ylabel:
            ax.set_ylabel("Temperature [K]")

        plt.ticklabel_format(useOffset=False)
        ax.grid()
        ax.legend()

        return plt.gcf()

    def get_load_residuals(self):
        """Get residuals of the calibrated temperature for a each load."""
        return {
            name: self.calibrate(load) - load.temp_ave
            for name, load in self.loads.items()
        }

    def get_rms(self, smooth: int = 4):
        """Return a dict of RMS values for each source.

        Parameters
        ----------
        smooth : int
            The number of bins over which to smooth residuals before taking the RMS.
        """
        resids = self.get_load_residuals()
        out = {}
        for name, res in resids.items():
            if smooth > 1:
                res = convolve(res, Gaussian1DKernel(stddev=smooth), boundary="extend")
            out[name] = np.sqrt(np.nanmean(res**2))
        return out

    def plot_calibrated_temps(self, bins: int = 64, fig=None, ax=None, **kwargs):
        """
        Plot all calibrated temperatures in a single figure.

        Parameters
        ----------
        bins : int
            Number of bins in the smoothed spectrum

        Returns
        -------
        fig :
            Matplotlib figure that was created.
        """
        if fig is None or ax is None or len(ax) != len(self.loads):
            fig, ax = plt.subplots(
                len(self.loads),
                1,
                sharex=True,
                gridspec_kw={"hspace": 0.05},
                figsize=(10, 12),
            )

        for i, source in enumerate(self.loads):
            self.plot_calibrated_temp(
                source,
                bins=bins,
                fig=fig,
                ax=ax[i],
                xlabel=i == (len(self.loads) - 1),
            )

        fig.suptitle("Calibrated Temperatures for Calibration Sources", fontsize=15)
        return fig

    def plot_coefficients(self, fig=None, ax=None):
        """
        Make a plot of the calibration models, C1, C2, Tunc, Tcos and Tsin.

        Parameters
        ----------
        fig : Figure
            Optionally pass a matplotlib figure to add to.
        ax : Axis
            Optionally pass a matplotlib axis to pass to. Must have 5 axes.
        """
        if fig is None or ax is None:
            fig, ax = plt.subplots(
                5, 1, facecolor="w", gridspec_kw={"hspace": 0.05}, figsize=(10, 9)
            )

        labels = [
            "Scale ($C_1$)",
            "Offset ($C_2$) [K]",
            r"$T_{\rm unc}$ [K]",
            r"$T_{\rm cos}$ [K]",
            r"$T_{\rm sin}$ [K]",
        ]
        for i, (kind, label) in enumerate(
            zip(["C1", "C2", "Tunc", "Tcos", "Tsin"], labels, strict=False)
        ):
            ax[i].plot(self.freq, getattr(self, kind)())
            ax[i].set_ylabel(label, fontsize=13)
            ax[i].grid()
            plt.ticklabel_format(useOffset=False)

            if i == 4:
                ax[i].set_xlabel("Frequency [MHz]", fontsize=13)

        fig.suptitle("Calibration Parameters", fontsize=15)
        return fig

    def clone(self, **kwargs):
        """Clone the instance, updating some parameters.

        Parameters
        ----------
        kwargs :
            All parameters to be updated.
        """
        return attr.evolve(self, **kwargs)

    def write(self, filename: Path):
        """Write the calibration observation to a file."""
        self.to_calibrator().write(filename=filename)

    def to_calibrator(self):
        """Directly create a :class:`Calibrator` object without writing to file."""
        return Calibrator(
            t_load=self.t_load,
            t_load_ns=self.t_load_ns,
            C1=self.cal_coefficient_models["C1"],
            C2=self.cal_coefficient_models["C2"],
            Tunc=self.cal_coefficient_models["NW"].get_tunc,
            Tcos=self.cal_coefficient_models["NW"].get_tcos,
            Tsin=self.cal_coefficient_models["NW"].get_tsin,
            freq=self.freq,
            receiver_s11=self.receiver.s11_model,
            internal_switch=self.internal_switch,
            metadata=self.metadata,
        )

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
    ) -> CalibrationObservation:
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

        return cls.from_caldef2(io_obs, **config)


class CalFileReadError(Exception):
    pass


@hickleable
@attrs.define(kw_only=True)
class Calibrator:
    freq: tp.FreqType = attrs.field()

    _C1: Callable[[np.ndarray], np.ndarray] = attrs.field()
    _C2: Callable[[np.ndarray], np.ndarray] = attrs.field()
    _Tunc: Callable[[np.ndarray], np.ndarray] = attrs.field()
    _Tcos: Callable[[np.ndarray], np.ndarray] = attrs.field()
    _Tsin: Callable[[np.ndarray], np.ndarray] = attrs.field()
    _receiver_s11: Callable[[np.ndarray], np.ndarray] = attrs.field()

    coefficient_freq_units: un.Unit = attrs.field(default=un.MHz)
    receiver_s11_freq_units: un.Unit = attrs.field(default=un.MHz)

    internal_switch: s11.InternalSwitch | None = attrs.field(default=None)
    t_load: float = attrs.field(default=300)
    t_load_ns: float = attrs.field(default=350)
    metadata: dict = attrs.field(factory=dict)

    def __attrs_post_init__(self):
        """Initialize properties of the class."""
        for key in ["C1", "C2", "Tunc", "Tcos", "Tsin"]:
            setattr(
                self,
                key,
                partial(self._call_func, key=key, unit=self.coefficient_freq_units),
            )
        for key in [
            "receiver_s11",
        ]:
            setattr(
                self,
                key,
                partial(self._call_func, key=key, unit=self.receiver_s11_freq_units),
            )

    def clone(self, **kwargs):
        """Clone the instance with new parameters."""
        new = copy.deepcopy(self)
        return attr.evolve(new, **kwargs)

    @internal_switch.validator
    def _isw_vld(self, att, val):
        if val is None:
            return

        if isinstance(val, s11.InternalSwitch):
            return

        for key in ("s11", "s12", "s22"):
            if not hasattr(val, f"{key}_model") or not callable(
                getattr(val, f"{key}_model")
            ):
                raise ValueError(f"internal_switch must provide {key}_model method")

    def _call_func(self, freq: tp.FreqType | None = None, *, key=None, unit="MHz"):
        if freq is None:
            freq = self.freq

        if not hasattr(freq, "unit"):
            raise ValueError("freq must have units of frequency")

        freq = freq.to_value(unit)
        return getattr(self, f"_{key}")(freq)

    @classmethod
    def from_calfile(cls, path: tp.PathLike) -> Calibrator:
        """Generate from calfile."""
        path = Path(path)

        if not path.exists():
            raise FileNotFoundError(f"calfile {path} not found!")

        if not h5py.is_hdf5(path):
            raise ValueError(
                "This file is not in HDF5 format. "
                "Perhaps you mean to read it as a specal object?"
            )
        with h5py.File(path, "r") as fl:
            version = fl.attrs.get("format.version", "1.0")
            if "C1" in fl:
                version = "0.0"

            try:
                return getattr(cls, f"_read_calfile_v{version.split('.')[0]}")(fl)
            except Exception as e:
                raise CalFileReadError(
                    f"Something went wrong reading the calfile: {path} which is "
                    f"version {version}"
                ) from e

    @classmethod
    def _read_calfile_v2(cls, fl: h5py.File):
        """Read calfile v2."""
        raise NotADirectoryError("v2 files are deprecated!")

    @classmethod
    def _read_calfile_v3(cls, fl: h5py.File):
        t_load = fl.attrs["t_load"]
        t_load_ns = fl.attrs["t_load_ns"]
        fixed_freq = fl.attrs["fixed_freqs"]

        freq = fl["frequencies"][...] * un.MHz

        if not fixed_freq:
            raise NotImplementedError("model-based calibration files not yet supported")

        cc = fl["cal_coefficients"]
        cc = {k: cc[k][...] for k in cc}

        rcv = fl["receiver_s11"]["s11"][()]
        rcv = ComplexSpline(freq, rcv)

        swg = fl["internal_switch"]
        if "s11" in swg:
            sws11 = ComplexSpline(freq, swg["s11"][()])
            sws12 = ComplexSpline(freq, swg["s12"][()])
            sws22 = ComplexSpline(freq, swg["s22"][()])
            sw = SimpleNamespace(s11_model=sws11, s12_model=sws12, s22_model=sws22)
        else:
            sw = None

        cc = {k: Spline(freq, v) for k, v in cc.items()}
        return cls(
            t_load=t_load,
            t_load_ns=t_load_ns,
            freq=freq,
            receiver_s11=rcv,
            internal_switch=sw,
            # metadata=meta,
            **cc,
        )

    def write(self, filename: str | Path, fixed_freqs: bool = True):
        """
        Write all information required to calibrate a new spectrum to file.

        Parameters
        ----------
        filename : path
            The filename to write to.
        """
        # TODO: this is *not* all the metadata available when using edges-io. We should
        # build a better system of maintaining metadata in subclasses to be used here.
        with h5py.File(filename, "w") as fl:
            # Write attributes

            fl.attrs["t_load"] = self.t_load
            fl.attrs["t_load_ns"] = self.t_load_ns
            fl.attrs["format.version"] = "2.0"
            fl.attrs["fixed_freqs"] = fixed_freqs

            # hickle is OK here because these are standard python objects.
            hickle.dump(self.freq, fl.create_group("frequencies"))
            hickle.dump(self.metadata, fl.create_group("metadata"))

            # Write calibration coefficients
            if fixed_freqs:
                self._write_fixed_freq_h5(fl)
            else:
                self._write_model_based_h5(fl)

    def _write_fixed_freq_h5(self, fl: h5py.File):
        ccgroup = fl.create_group("cal_coefficients")
        ccgroup["C1"] = self.C1()
        ccgroup["C2"] = self.C2()
        ccgroup["Tunc"] = self.Tunc()
        ccgroup["Tcos"] = self.Tcos()
        ccgroup["Tsin"] = self.Tsin()

        fq = self.freq.to_value("MHz")
        grp = fl.create_group("receiver_s11")
        grp["s11"] = self.receiver_s11(self.freq)

        sw_group = fl.create_group("internal_switch")
        if hasattr(self.internal_switch, "s11_model"):
            sw_group["s11"] = self.internal_switch.s11_model(fq)
            sw_group["s12"] = self.internal_switch.s12_model(fq)
            sw_group["s22"] = self.internal_switch.s22_model(fq)

    @classmethod
    def from_calobs(cls, calobs: CalibrationObservation) -> Calibrator:
        """Generate a :class:`Calibration` from an in-memory observation."""
        return calobs.to_calibrator()

    def _linear_coefficients(self, freq, ant_s11):
        return rcf.get_linear_coefficients(
            ant_s11,
            self.receiver_s11(freq),
            self.C1(freq),
            self.C2(freq),
            self.Tunc(freq),
            self.Tcos(freq),
            self.Tsin(freq),
            self.t_load,
        )

    def calibrate_temp(self, freq: np.ndarray, temp: np.ndarray, ant_s11: np.ndarray):
        """
        Calibrate given uncalibrated spectrum.

        Parameters
        ----------
        freq : np.ndarray
            The frequencies at which to calibrate
        temp :  np.ndarray
            The temperatures to calibrate (in K).
        ant_s11 : np.ndarray
            The antenna S11 for the load.

        Returns
        -------
        temp : np.ndarray
            The calibrated temperature.
        """
        a, b = self._linear_coefficients(freq, ant_s11)
        return temp * a + b

    def decalibrate_temp(self, freq, temp, ant_s11):
        """
        De-calibrate given calibrated spectrum.

        Parameters
        ----------
        freq : np.ndarray
            The frequencies at which to calibrate
        temp :  np.ndarray
            The temperatures to calibrate (in K).
        ant_s11 : np.ndarray
            The antenna S11 for the load.

        Returns
        -------
        temp : np.ndarray
            The calibrated temperature.

        Notes
        -----
        Using this and then :meth:`calibrate_temp` immediately should be an identity
        operation.
        """
        a, b = self._linear_coefficients(freq, ant_s11)
        return (temp - b) / a

    def calibrate_Q(
        self, freq: np.ndarray, q: np.ndarray, ant_s11: np.ndarray
    ) -> np.ndarray:
        """
        Calibrate given power ratio spectrum.

        Parameters
        ----------
        freq : np.ndarray
            The frequencies at which to calibrate
        q :  np.ndarray
            The power ratio to calibrate.
        ant_s11 : np.ndarray
            The antenna S11 for the load.

        Returns
        -------
        temp : np.ndarray
            The calibrated temperature.
        """
        uncal_temp = self.t_load_ns * q + self.t_load

        return self.calibrate_temp(freq, uncal_temp, ant_s11)


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
