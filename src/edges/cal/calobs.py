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

from .. import types as tp
from ..cached_property import cached_property, safe_property
from ..io import calobsdef, calobsdef3
from ..io.serialization import hickleable
from . import loss
from . import noise_waves as nw
from . import sparams as sp
from .calibrator import Calibrator
from .input_sources import InputSource
from .loss import LossFunctionGivenSparams


@hickleable
@attrs.define(slots=False, kw_only=True, frozen=True)
class CalibrationObservation:
    """
    An object representing a full Calibration Observation.

    Parameters
    ----------
    loads
        Dictionary of load names mapping to :class:`InputSource` objects.
    receiver
        The reflection coefficient of the receiver.
    """

    loads: dict[str, InputSource] = attrs.field()
    receiver: sp.ReflectionCoefficient = attrs.field(
        validator=attrs.validators.instance_of(sp.ReflectionCoefficient)
    )
    _raw_receiver: sp.ReflectionCoefficient | None = attrs.field(
        default=None,
        validator=attrs.validators.optional(
            attrs.validators.instance_of(sp.ReflectionCoefficient)
        ),
    )

    @property
    def ambient(self) -> InputSource:
        """The ambient load."""
        return self.loads["ambient"]

    @property
    def hot_load(self) -> InputSource:
        """The hot load."""
        return self.loads["hot_load"]

    @property
    def open(self) -> InputSource:
        """The open load."""
        return self.loads["open"]

    @property
    def short(self) -> InputSource:
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
        internal_calkit: sp.Calkit | None = None,
        external_calkit_internal_switch: sp.Calkit | None = None,
        f_low: tp.FreqType = 40.0 * un.MHz,
        f_high: tp.FreqType = np.inf * un.MHz,
        receiver_kwargs: dict[str, Any] | None = None,
        restrict_s11_model_freqs: bool = True,
        loss_models: dict[str, callable] | None = None,
        loss_model_params: sp.S11ModelParams | None = None,
        internal_switch_temperature: tp.TemperatureType | None = None,
        internal_switch_model_params: sp.S11ModelParams
        | None = sp.internal_switch_model_params(),
    ) -> Self:
        """Create the object from an edges-io observation.

        Parameters
        ----------
        caldef
            A calibration definition object from which all the data can be read.
        freq_bin_size
            The size of each frequency bin (of the spectra) in units of the raw size.
        spectrum_kwargs
            Keyword arguments used to instantiate the calibrator :class:`LoadSpectrum`
            objects. See its documentation for relevant parameters. Parameters under
            the key "default" are used for _all_ calibrator sources, and parameters
            under the name of a source are used for that source only.
        s11_kwargs
            Keyword arguments used to calibrate and model the reflection coefficients
            of all calibrator sources (see :meth:`InputSource.from_caldef`). The key
            "model_params" may hold the :class:`S11ModelParams` used for all sources.
        internal_calkit
            The calkit model of the internal standards used to calibrate the internal
            switch. By default, determined from ``caldef``.
        external_calkit_internal_switch
            The calkit model of the external standards used to calibrate the internal
            switch. By default, determined from ``caldef``.
        f_low : float
            Minimum frequency to keep for all loads (and their S11's).
        f_high : float
            Maximum frequency to keep for all loads (and their S11's).
        receiver_kwargs
            Keyword arguments used to compute the receiver reflection coefficient (see
            :func:`edges.cal.sparams.get_gamma_receiver_from_filespec`). The key
            "model_params" may hold the :class:`S11ModelParams` used to model it.
        restrict_s11_model_freqs
            Whether to restrict the S11 modelling (i.e. smoothing) to the given freq
            range. The final output will be calibrated only between the given freq
            range, but the S11 models themselves can be fit over a broader set of
            frequencies.
        loss_models
            A dictionary of loss models for each source. If a particular source has no
            loss its entry can be missing or None. By default, the hot load uses the
            cable S-parameters in ``caldef.hot_load.sparams_file``, if it is set.
        loss_model_params
            Parameters used to smooth the S-parameters of loss models onto the
            frequencies of the spectra. If None, use
            :func:`edges.cal.sparams.hot_load_cable_model_params`.
        internal_switch_temperature
            The measured temperature of the internal switch, used to choose the
            internal switch calkit model. By default, determined from ``caldef``.
        internal_switch_model_params
            Parameters used to smooth the internal switch S-parameters. If None, they
            are not smoothed.
        """
        receiver_kwargs = dict(receiver_kwargs or {})
        if "calkit" not in receiver_kwargs:
            receiver_kwargs["calkit"] = sp.get_calkit(
                sp.AGILENT_85033E,
                resistance_of_match=caldef.receiver_s11.calkit_match_resistance
                if hasattr(caldef.receiver_s11, "calkit_match_resistance")
                else caldef.receiver_s11[0].calkit_match_resistance,
            )

        loss_models = dict(loss_models or {})
        if "hot_load" not in loss_models and caldef.hot_load.sparams_file is not None:
            hot_load_cable_sparams = sp.read_semi_rigid_cable_sparams_file(
                caldef.hot_load.sparams_file, f_low=f_low, f_high=f_high
            )

            loss_models["hot_load"] = LossFunctionGivenSparams(hot_load_cable_sparams)

        internal_switch = sp.get_internal_switch_from_caldef(
            caldef,
            external_calkit=external_calkit_internal_switch,
            internal_calkit=internal_calkit,
            measured_temperature=internal_switch_temperature,
        )

        if internal_switch_model_params is not None:
            internal_switch = internal_switch.smoothed(
                internal_switch_model_params,
                freqs=internal_switch.freqs,
            )

        return cls._from_caldef(
            caldef=caldef,
            freq_bin_size=freq_bin_size,
            spectrum_kwargs=spectrum_kwargs,
            s11_kwargs=s11_kwargs,
            internal_switch=internal_switch,
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
        caldef
            A calibration definition object from which all the data can be read.
        freq_bin_size
            The size of each frequency bin (of the spectra) in units of the raw size.
        spectrum_kwargs
            Keyword arguments used to instantiate the calibrator :class:`LoadSpectrum`
            objects. See its documentation for relevant parameters. Parameters under
            the key "default" are used for _all_ calibrator sources, and parameters
            under the name of a source are used for that source only.
        s11_kwargs
            Keyword arguments used to calibrate and model the reflection coefficients
            of all calibrator sources (see :meth:`InputSource.from_caldef`). The key
            "model_params" may hold the :class:`S11ModelParams` used for all sources.
        f_low : float
            Minimum frequency to keep for all loads (and their S11's).
        f_high : float
            Maximum frequency to keep for all loads (and their S11's).
        receiver_kwargs
            Keyword arguments used to compute the receiver reflection coefficient (see
            :func:`edges.cal.sparams.get_gamma_receiver_from_filespec`). These update
            the EDGES-3 defaults. The key "model_params" may hold the
            :class:`S11ModelParams` used to model it.
        restrict_s11_model_freqs
            Whether to restrict the S11 modelling (i.e. smoothing) to the given freq
            range. The final output will be calibrated only between the given freq
            range, but the S11 models themselves can be fit over a broader set of
            frequencies.
        loss_models
            A dictionary of loss models for each source. If a particular source has no
            loss its entry can be missing or None. By default, the only source with loss
            is the hot_load, which uses a 4" cable.
        kwargs
            Passed through to :meth:`_from_caldef` (e.g. ``loss_model_params``).
            Unknown keywords raise a TypeError.
        """
        loss_models = dict(loss_models or {})
        if "hot_load" not in loss_models:
            loss_models["hot_load"] = loss.get_cable_loss_model("UT-141C-SP")

        receiver_kwargs = receiver_kwargs or {}

        default_rcv_kw = {
            "calkit": sp.get_calkit(
                sp.AGILENT_ALAN,
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
        internal_switch: sp.SParams | None = None,
        f_low: tp.FreqType = 40.0 * un.MHz,
        f_high: tp.FreqType = np.inf * un.MHz,
        receiver_kwargs: dict[str, Any] | None = None,
        restrict_s11_model_freqs: bool = True,
        loss_models: dict[str, callable] | None = None,
        loss_model_params: sp.S11ModelParams | None = None,
        **kwargs,
    ) -> Self:
        """Create a CalibrationObservation from a "definition" of all required paths.

        Parameters
        ----------
        caldef
            A calibration definition object from which all the data can be read.
        freq_bin_size
            The size of each frequency bin (of the spectra) in units of the raw size.
        spectrum_kwargs
            Keyword arguments used to instantiate the calibrator :class:`LoadSpectrum`
            objects. See its documentation for relevant parameters. Parameters under
            the key "default" are used for _all_ calibrator sources, and parameters
            under the name of a source are used for that source only.
        s11_kwargs
            Keyword arguments used to calibrate and model the reflection coefficients
            of all calibrator sources (see :meth:`InputSource.from_caldef`). The key
            "model_params" may hold the :class:`S11ModelParams` used for all sources.
        internal_switch
            The (smoothed) S-parameters of the internal switch, used to calibrate the
            S11 of each input source (EDGES-2 only).
        f_low : float
            Minimum frequency to keep for all loads (and their S11's).
        f_high : float
            Maximum frequency to keep for all loads (and their S11's).
        receiver_kwargs
            Keyword arguments used to compute the receiver reflection coefficient (see
            :func:`edges.cal.sparams.get_gamma_receiver_from_filespec`). The key
            "model_params" may hold the :class:`S11ModelParams` used to model it.
        restrict_s11_model_freqs
            Whether to restrict the S11 modelling (i.e. smoothing) to the given freq
            range. The final output will be calibrated only between the given freq
            range, but the S11 models themselves can be fit over a broader set of
            frequencies.
        loss_models
            A dictionary of loss models for each source. If a particular source has no
            loss its entry can be missing or None.
        loss_model_params
            Parameters used to smooth the S-parameters of loss models onto the
            frequencies of the spectra. If None, use
            :func:`edges.cal.sparams.hot_load_cable_model_params`.
        kwargs
            Passed through to the class constructor.
        """
        if f_high < f_low:
            raise ValueError("f_high must be larger than f_low!")

        spectrum_kwargs = spectrum_kwargs or {}
        s11_kwargs = s11_kwargs or {}
        receiver_kwargs = receiver_kwargs or {}
        loss_models = loss_models or {}

        for v in [spectrum_kwargs, s11_kwargs, receiver_kwargs]:
            assert isinstance(v, dict)

        f_low = f_low.to("MHz", copy=False)
        f_high = f_high.to("MHz", copy=False)

        # Never modify the caller's dictionaries.
        receiver_kwargs = dict(receiver_kwargs)
        rcv_model_params = receiver_kwargs.pop("model_params", None)
        if rcv_model_params is None:
            rcv_model_params = sp.receiver_model_params()

        raw_receiver = sp.get_gamma_receiver_from_filespec(caldef, **receiver_kwargs)

        spectrum_kwargs = {k: dict(v) for k, v in spectrum_kwargs.items()}
        spectrum_kwargs.setdefault("default", {})
        spectrum_kwargs["default"].setdefault("freq_bin_size", freq_bin_size)

        def get_load(name, ambient_temperature=298 * un.K):
            return InputSource.from_caldef(
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
                internal_switch=internal_switch,
            )

        amb = get_load("ambient")
        loads = {"ambient": amb}

        loads |= {
            src: get_load(src, ambient_temperature=amb.temp_ave)
            for src in ("hot_load", "open", "short")
        }

        # Smooth the receiver s11
        receiver = raw_receiver.smoothed(rcv_model_params, freqs=amb.freqs)

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

    def averaged_spectrum(self, load: InputSource, t_load_ns: float, t_load: float):
        """Compute a quick guess at the calibrated spectrum of a given load."""
        return load.spectrum.q.data.squeeze() * t_load_ns + t_load

    @cached_property
    def load_s11_models(self) -> dict[str, np.ndarray]:
        """Dictionary of S11 correction models, one for each source."""
        return {
            name: source.reflection_coefficient.reflection_coefficient
            for name, source in self.loads.items()
        }

    @cached_property
    def source_thermistor_temps(self) -> dict[str, tp.TemperatureType]:
        """Dictionary of input source thermistor temperatures."""
        return {k: source.temp_ave for k, source in self.loads.items()}

    def _load_str_to_load(self, load: InputSource | str):
        if isinstance(load, str):
            try:
                load = self.loads[load]
            except (AttributeError, KeyError) as e:
                raise AttributeError(
                    f"load must be a Load object or a string (one of {self.load_names})"
                ) from e
        else:
            assert isinstance(load, InputSource), (
                f"load must be a Load instance, got the {load} {type(InputSource)}"
            )
        return load

    def get_K(
        self,
    ) -> dict[str, tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]]:
        """Get the source-S11-dependent factors of Monsalve (2017) Eq. 7."""
        lna_s11 = self.receiver.s11
        return {
            name: nw.get_K(gamma_rec=lna_s11, gamma_ant=gamma_ant)
            for name, gamma_ant in self.load_s11_models.items()
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
    def receiver_s11(self) -> sp.ReflectionCoefficient:
        """The S11 of the receiver."""
        return self.receiver.s11

    def inject(
        self,
        receiver: np.ndarray | None = None,
        source_s11s: dict[str, np.ndarray] | None = None,
        averaged_q: dict[str, np.ndarray] | None = None,
        thermistor_temp_ave: dict[str, np.ndarray] | None = None,
    ) -> Self:
        """Make a new :class:`CalibrationObservation` based on this, with injections.

        Parameters
        ----------
        receiver
            A new receiver reflection coefficient.
        source_s11s
            New reflection coefficients for (some of) the input sources.
        averaged_q
            New time-averaged power ratios, Q, for (some of) the input sources.
        thermistor_temp_ave
            New average thermistor temperatures for (some of) the input sources.
            These are the physical temperatures of the sources, before any loss.

        Returns
        -------
        :class:`CalibrationObservation`
            A new observation object with the injected models.
        """
        kw = {}
        if receiver is not None:
            receiver = sp.ReflectionCoefficient(
                reflection_coefficient=receiver, freqs=self.freqs
            )
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
                        newloads[name],
                        reflection_coefficient=sp.ReflectionCoefficient(
                            freqs=self.freqs, reflection_coefficient=s
                        ),
                    )

            averaged_q = averaged_q or {}
            thermistor_temp_ave = thermistor_temp_ave or {}
            for name in {**averaged_q, **thermistor_temp_ave}:
                spec = newloads[name].spectrum
                if name in averaged_q:
                    spec = attrs.evolve(
                        spec, q=spec.q.update(data=averaged_q[name][None, None, None])
                    )
                if name in thermistor_temp_ave:
                    spec = attrs.evolve(spec, temp_ave=thermistor_temp_ave[name])
                newloads[name] = attrs.evolve(newloads[name], spectrum=spec)

            kw["loads"] = newloads
        return self.clone(**kw)
