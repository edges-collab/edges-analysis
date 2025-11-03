"""Definition of a class that contains all the data required for a calibration load."""

from collections.abc import Callable
from functools import cached_property

import attrs
import numpy as np
from astropy import units as un
from pygsdata.attrs import npfield

from edges import types as tp
from edges.cal.s11.s11model import S11ModelParams
from edges.io import CalObsDefEDGES2, CalObsDefEDGES3, hickleable

from . import reflection_coefficient as rc
from .s11 import CalibratedS11
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
    s11: CalibratedS11 = attrs.field()
    _raw_s11: CalibratedS11 | None = attrs.field(default=None)
    ambient_temperature: tp.TemperatureType = npfield(
        default=298.0 * un.K,
        unit=un.K,
        possible_ndims=(
            0,
            1,
        ),
    )
    load_name: str = attrs.field(default="")
    loss: np.ndarray = npfield(dtype=float, possible_ndims=(1,))

    @loss.default
    def _loss_default(self):
        """Default loss is a flat 1.0."""
        return np.ones(len(self.spectrum.freqs))

    @classmethod
    def from_caldef(
        cls,
        caldef: CalObsDefEDGES2 | CalObsDefEDGES3,
        load_name: str,
        ambient_temperature: tp.TemperatureType | None = None,
        f_low: tp.FreqType = 40 * un.MHz,
        f_high: tp.FreqType = np.inf * un.MHz,
        s11_kwargs: dict | None = None,
        spec_kwargs: dict | None = None,
        loss_model: Callable | None = None,
        loss_model_params: S11ModelParams = (
            S11ModelParams.from_hot_load_cable_defaults()
        ),
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
        # TODO: here we only use the calkit defined for the FIRST switching_state,
        # instead of using each calkit for each switching_state. To fix this, we require
        # having meta information inside the S11/ directory.
        s11_kwargs["f_low"] = f_low if restrict_s11_freqs else 0 * un.MHz
        s11_kwargs["f_high"] = f_high if restrict_s11_freqs else np.inf * un.MHz

        s11_model_params = s11_kwargs.pop(
            "model_params",
            S11ModelParams.from_calibration_load_defaults(name=load_name),
        )

        if isinstance(caldef, CalObsDefEDGES2):
            if "internal_switch_kwargs" not in s11_kwargs:
                s11_kwargs["internal_switch_kwargs"] = {}

            if "calkit" not in s11_kwargs["internal_switch_kwargs"]:
                s11_kwargs["internal_switch_kwargs"]["calkit"] = rc.get_calkit(
                    rc.AGILENT_85033E, resistance_of_match=caldef.male_resistance
                )

            raw_s11 = CalibratedS11.from_edges2_loaddef(
                caldef, load=load_name, **s11_kwargs
            )
        elif isinstance(caldef, CalObsDefEDGES3):
            if "calkit" not in s11_kwargs:
                s11_kwargs["calkit"] = rc.AGILENT_ALAN

            raw_s11 = CalibratedS11.from_edges3_loaddef(
                caldef, load=load_name, **s11_kwargs
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
            loss = loss_model(spec.freqs, s11.s11)
        else:
            loss = np.ones(spec.freqs.shape)

        return cls(
            spectrum=spec,
            raw_s11=raw_s11,
            s11=s11,
            loss=loss,
            ambient_temperature=ambient_temperature,
            load_name=load_name,
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
