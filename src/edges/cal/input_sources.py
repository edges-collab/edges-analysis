"""Definition of a class that contains all the data required for a calibration load."""

from collections.abc import Callable
from functools import cached_property
from typing import Self

import attrs
import numpy as np
from astropy import units as un
from pygsdata.attrs import npfield

from edges import types as tp
from edges.io import CalObsDefEDGES2, CalObsDefEDGES3, hickleable

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
        loss_model_params: sp.S11ModelParams = sp.hot_load_cable_model_params(),
        restrict_s11_freqs: bool = False,
    ) -> Self:
        """
        Define a full :class:`InputSource` from a path and name.

        Parameters
        ----------
        caldef
            The calibration definition object that points to all the required datafiles.
        load_name
            The name of the load within the calibration definition to use.
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
        restrict_s11_freqs
            Whether to restrict the S11 frequencies to f_low/f_high when calibrating
            and modelling (they will always be restricted to the spectrum frequencies
            after modelling).

        Returns
        -------
        load
            The InputSource object, containing all info about spectra and S11's for
            that input source.
        """
        if not spec_kwargs:
            spec_kwargs = {}
        if not s11_kwargs:
            s11_kwargs = {}

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

        # Fill up kwargs with keywords from this instance
        s11_kwargs["f_low"] = f_low if restrict_s11_freqs else 0 * un.MHz
        s11_kwargs["f_high"] = f_high if restrict_s11_freqs else np.inf * un.MHz

        s11_model_params = s11_kwargs.pop(
            "model_params", sp.input_source_model_params(name=load_name)
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

        # Now, model the S11
        s11 = raw_s11.smoothed(s11_model_params, freqs=spec.freqs)

        if loss_model is not None:
            if (
                hasattr(loss_model, "sparams")
                and loss_model.sparams.freqs.size != spec.freqs.size
            ):
                loss_model = attrs.evolve(
                    loss_model,
                    sparams=loss_model.sparams.smoothed(
                        params=loss_model_params, freqs=spec.freqs
                    ),
                )
            loss = loss_model(s11)
        else:
            loss = np.ones(spec.freqs.shape)

        return cls(
            spectrum=spec,
            raw_s11=raw_s11,
            reflection_coefficient=s11,
            loss=loss,
            ambient_temperature=ambient_temperature,
            name=load_name,
        )

    def get_temp_with_loss(self):
        """Calculate the temperature of the load accounting for loss."""
        gain = self.loss
        return gain * self.spectrum.temp_ave + (1 - gain) * self.ambient_temperature

    @cached_property
    def temp_ave(self) -> np.ndarray:
        """The average temperature of the thermistor (over frequency and time)."""
        return self.get_temp_with_loss()

    @property
    def averaged_q(self) -> np.ndarray:
        """The average spectrum power ratio, Q (over time)."""
        return self.spectrum.q.data.squeeze()

    @property
    def freqs(self) -> tp.FreqType:
        """Frequencies of the spectrum."""
        return self.spectrum.q.freqs
