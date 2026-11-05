"""Definition of a class that contains all the data required for a calibration load."""

from collections.abc import Callable
from typing import Self

import attrs
import numpy as np
from astropy import units as un
from pygsdata.attrs import npfield

from edges import types as tp
from edges.io import CalObsDefEDGES2, CalObsDefEDGES3
from edges.io.loads import canonical_load_name
from edges.serialization import hickleable

from . import sparams as sp
from .sparams import ReflectionCoefficient
from .spectra import LoadSpectrum


@hickleable
@attrs.define(kw_only=True)
class InputSource:
    """Class containing all relevant information for a given calibration source.

    Parameters
    ----------
    spectrum
        The spectrum for this input source.
    reflection_coefficient
        The calibrated reflection coefficient for this input source, defined at the
        frequencies of the spectrum.
    raw_s11
        The un-modeled reflection coefficient, which can be set simply to be able to
        compare to the modeled coefficients.
    ambient_temperature
        The ambient temperature when the spectra were taken. Used only when calculating
        the loss.
    name
        The name of the input source. Optional, but can be useful if set.
    loss
        The loss as a function of frequency (must have the same size as the number of
        frequency channels in the spectrum).
    """

    spectrum: LoadSpectrum = attrs.field(
        validator=attrs.validators.instance_of(LoadSpectrum)
    )
    reflection_coefficient: ReflectionCoefficient = attrs.field(
        validator=attrs.validators.instance_of(ReflectionCoefficient)
    )
    _raw_s11: ReflectionCoefficient | None = attrs.field(
        default=None,
        validator=attrs.validators.optional(
            attrs.validators.instance_of(ReflectionCoefficient)
        ),
    )
    ambient_temperature: tp.TemperatureType = npfield(
        default=298.0 * un.K,
        unit=un.K,
        possible_ndims=(
            0,
            1,
        ),
    )
    name: str = attrs.field(default="", converter=str)
    loss: np.ndarray = npfield(dtype=float, possible_ndims=(1,))

    @loss.default
    def _loss_default(self):
        """Default loss is a flat 1.0."""
        return np.ones(len(self.spectrum.freqs))

    @reflection_coefficient.validator
    def _s11_vld(self, att, val):
        if len(val.freqs) != len(self.spectrum.freqs):
            raise ValueError(
                "reflection_coefficient must have the same number of channels "
                "as the spectra"
            )

    @loss.validator
    def _loss_vld(self, att, val):
        if len(val) != len(self.spectrum.freqs):
            raise ValueError(
                "loss must have the same number of channels as the spectrum"
            )

    @property
    def s11(self) -> ReflectionCoefficient:
        """An alias for the reflection coefficient."""
        return self.reflection_coefficient

    @classmethod
    def from_components(
        cls,
        *,
        spectrum: LoadSpectrum,
        raw_s11: ReflectionCoefficient,
        s11_model_params: sp.S11ModelParams,
        s11_freq_range: tuple[tp.FreqType, tp.FreqType] | None = None,
        loss_model: Callable | None = None,
        loss_model_params: sp.S11ModelParams | None = None,
        ambient_temperature: tp.TemperatureType = 298.0 * un.K,
        name: str = "",
    ) -> Self:
        """Assemble an :class:`InputSource` from its spectrum and raw S11.

        This models the raw (calibrated) reflection coefficient at the frequencies of
        the spectrum and computes the loss of the source. It does not read any data,
        so it can be used with spectra and S11s from any source (see
        :meth:`from_caldef` for reading them from a calibration definition).

        Parameters
        ----------
        spectrum
            The (already averaged) spectrum of the source.
        raw_s11
            The calibrated, but un-modelled, reflection coefficient of the source. It
            is stored as given, as the raw S11 of the source.
        s11_model_params
            The parameters of the model of the reflection coefficient.
        s11_freq_range
            The ``(f_low, f_high)`` range of the measured S11 frequencies to which the
            S11 model is fit. This is independent of the frequencies of the spectrum,
            at which the model is evaluated. If None, the model is fit to all of
            ``raw_s11``.
        loss_model
            A callable model of the loss of the source, taking the modelled reflection
            coefficient. If None, the source has no loss.
        loss_model_params
            Parameters used to smooth/interpolate the S-parameters of ``loss_model``
            (if it is defined by S-parameters, as an attribute ``sparams``) onto the
            frequencies of the spectrum, if they are defined at a different number of
            frequencies. If None, ``loss_model`` is used as given (so its S-parameters,
            if any, must already be defined at the frequencies of the spectrum).
        ambient_temperature
            The ambient temperature when the spectra were taken. Used only when
            calculating the loss.
        name
            The name of the input source.

        Returns
        -------
        InputSource
            The input source, whose reflection coefficient is the model of ``raw_s11``
            evaluated at the frequencies of ``spectrum``.
        """
        s11_to_fit = (
            raw_s11
            if s11_freq_range is None
            else raw_s11.select_frequencies(*s11_freq_range)
        )
        s11 = s11_to_fit.smoothed(s11_model_params, freqs=spectrum.freqs)

        if loss_model is not None:
            if (
                loss_model_params is not None
                and hasattr(loss_model, "sparams")
                and loss_model.sparams.freqs.size != spectrum.freqs.size
            ):
                loss_model = attrs.evolve(
                    loss_model,
                    sparams=loss_model.sparams.smoothed(
                        params=loss_model_params, freqs=spectrum.freqs
                    ),
                )
            loss = loss_model(s11)
        else:
            loss = np.ones(spectrum.freqs.shape)

        return cls(
            spectrum=spectrum,
            raw_s11=raw_s11,
            reflection_coefficient=s11,
            loss=loss,
            ambient_temperature=ambient_temperature,
            name=name,
        )

    @classmethod
    def from_caldef(
        cls,
        caldef: CalObsDefEDGES2 | CalObsDefEDGES3,
        load_name: str,
        internal_switch: sp.SParams | None = None,
        ambient_temperature: tp.TemperatureType | None = None,
        f_low: tp.FreqType = 40 * un.MHz,
        f_high: tp.FreqType = np.inf * un.MHz,
        s11_kwargs: dict | None = None,
        spec_kwargs: dict | None = None,
        loss_model: Callable | None = None,
        loss_model_params: sp.S11ModelParams | None = None,
        restrict_s11_freqs: bool = False,
    ) -> Self:
        """
        Define a full :class:`InputSource` from a path and name.

        Parameters
        ----------
        caldef
            The calibration definition object that points to all the required datafiles.
        load_name
            The name of the load within the calibration definition to use, in any
            spelling accepted by :func:`edges.io.loads.canonical_load_name` (e.g.
            ``"hot_load"`` or ``"HotLoad"``). The returned source has the canonical
            name.
        internal_switch
            The internal switch S-parameters to calibrate the source reflection
            coefficient (optional -- use for EDGES 2). Note tat you can compute
            this with :func:`get_internal_switch_from_caldef`.
        ambient_temperature
            The ambient temperature during the spectrum observations.
        f_low
            The minimum frequency to keep in the spectra.
        f_high
            The maximum frequency to keep in the spectra.
        s11_kwargs
            Keyword arguments affecting how the reflection coefficients are calibrated
            and modelled.
        spec_kwargs
            Keyword arguments affecting how the spectra are defined.
        loss_model
            A callable model of the loss of the source.
        loss_model_params
            Parameters used to smooth/interpolate the S-parameters of the loss model
            onto the frequencies of the spectrum, if the loss model is defined by
            S-parameters at different frequencies. If None (default), use
            :func:`edges.cal.sparams.hot_load_cable_model_params`.
        restrict_s11_freqs
            Whether to fit the S11 model only to the measured S11 frequencies within
            ``[f_low, f_high]``. If False, the model is fit to the full measured
            (VNA) band. Either way, the model is evaluated at the spectrum
            frequencies, and the raw (un-modelled) S11 that is stored covers the full
            measured band.

        Returns
        -------
        load
            The InputSource object, containing all info about spectra and S11's for
            that input source.

        Raises
        ------
        ValueError
            If ``load_name`` is not a known calibration load.
        """
        load_name = canonical_load_name(load_name)
        s11_kwargs = s11_kwargs or {}

        if loss_model_params is None:
            loss_model_params = sp.hot_load_cable_model_params()

        spec, raw_s11 = _read_load_from_caldef(
            caldef,
            load_name,
            internal_switch=internal_switch,
            f_low=f_low,
            f_high=f_high,
            s11_kwargs=s11_kwargs,
            spec_kwargs=spec_kwargs,
        )

        s11_model_params = s11_kwargs.get("model_params")
        if s11_model_params is None:
            s11_model_params = sp.input_source_model_params(name=load_name)

        # Model the S11, optionally fitting only the measurements within
        # [f_low, f_high]. The raw (calibrated, un-modelled) S11 is kept in full.
        return cls.from_components(
            spectrum=spec,
            raw_s11=raw_s11,
            s11_model_params=s11_model_params,
            s11_freq_range=(f_low, f_high) if restrict_s11_freqs else None,
            loss_model=loss_model,
            loss_model_params=loss_model_params,
            ambient_temperature=ambient_temperature,
            name=load_name,
        )

    def get_temp_with_loss(self):
        """Calculate the temperature of the load accounting for loss."""
        gain = self.loss
        return gain * self.spectrum.temp_ave + (1 - gain) * self.ambient_temperature

    @property
    def temp_ave(self) -> np.ndarray:
        """The loss-corrected average temperature of the source (over time).

        This is recomputed on each access (it is cheap), so that it stays consistent
        if ``loss``, ``ambient_temperature`` or ``spectrum`` are changed.
        """
        return self.get_temp_with_loss()

    @property
    def averaged_q(self) -> np.ndarray:
        """The average spectrum power ratio, Q (over time)."""
        return self.spectrum.q.data.squeeze()

    @property
    def freqs(self) -> tp.FreqType:
        """Frequencies of the spectrum."""
        return self.spectrum.q.freqs


def _read_load_from_caldef(
    caldef: CalObsDefEDGES2 | CalObsDefEDGES3,
    load_name: str,
    *,
    internal_switch: sp.SParams | None = None,
    f_low: tp.FreqType = 40 * un.MHz,
    f_high: tp.FreqType = np.inf * un.MHz,
    s11_kwargs: dict | None = None,
    spec_kwargs: dict | None = None,
) -> tuple[LoadSpectrum, ReflectionCoefficient]:
    """Read the spectrum and the calibrated raw S11 of a load from a caldef.

    Parameters
    ----------
    caldef
        The calibration definition object that points to all the required datafiles.
    load_name
        The name of the load within the calibration definition, in any spelling
        accepted by :func:`edges.io.loads.canonical_load_name`.
    internal_switch
        The internal switch S-parameters used to calibrate the reflection coefficient
        (optional -- use for EDGES 2).
    f_low, f_high
        The frequency range of the spectrum to keep. These are also the defaults of
        the range of the spectrum that is read in and averaged (``f_low`` and
        ``f_high`` in ``spec_kwargs``).
    s11_kwargs
        Keyword arguments affecting how the reflection coefficient is calibrated. Only
        ``calkit`` is used here: the model of the internal standards (EDGES-3 only,
        default :data:`edges.cal.sparams.AGILENT_ALAN`).
    spec_kwargs
        Keyword arguments passed to :meth:`LoadSpectrum.from_loaddef`.

    Returns
    -------
    spectrum
        The spectrum of the load.
    raw_s11
        The calibrated, un-modelled reflection coefficient of the load, over the full
        measured band.
    """
    load_name = canonical_load_name(load_name)

    # Copy so that we never modify the caller's dictionaries.
    spec_kwargs = dict(spec_kwargs or {})
    s11_kwargs = s11_kwargs or {}

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

    gamma_src = ReflectionCoefficient.from_s1p(loaddef.s11.external)
    internal_osl = sp.CalkitReadings.from_filespec(loaddef.s11.calkit)

    if isinstance(caldef, CalObsDefEDGES3):
        internal_calkit = s11_kwargs.get("calkit", sp.AGILENT_ALAN)
    else:
        internal_calkit = None

    raw_s11 = sp.calibrate_gamma_src(
        gamma_src,
        internal_osl,
        internal_switch=internal_switch,
        internal_calkit=internal_calkit,
    )
    return spec, raw_s11
