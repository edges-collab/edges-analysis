"""Definition of a class that contains all the data required for a calibration load."""
from edges.io import hickleable, CalObsDefEDGES2, CalObsDefEDGES3
import attrs
from astropy import units as un
from edges import types as tp
from .spectra import LoadSpectrum
from .s11 import S11Model
import numpy as np
from .loss import HotLoadCorrection
from collections.abc import Callable
from . import reflection_coefficient as rc
from functools import cached_property

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
    s11: S11Model = attrs.field()
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
        return self.s11.load_name

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
            
        loss_kwargs = loss_kwargs or {}
        
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
        s11_kwargs['f_low'] = f_low if restrict_s11_freqs else 0*un.MHz
        s11_kwargs['f_high'] = f_high if restrict_s11_freqs else np.inf*un.MHz
        
        if isinstance(caldef, CalObsDefEDGES2):
            if "internal_switch_kwargs" not in s11_kwargs:
                s11_kwargs["internal_switch_kwargs"] = {}

            if "calkit" not in s11_kwargs["internal_switch_kwargs"]:
                s11_kwargs["internal_switch_kwargs"]["calkit"] = rc.get_calkit(
                    rc.AGILENT_85033E, resistance_of_match=caldef.male_resistance
                )

            s11 = S11Model.from_edges2_loaddef(caldef, load=load_name, **s11_kwargs)
        elif isinstance(caldef, CalObsDefEDGES3):
            if 'calkit' not in s11_kwargs:
                s11_kwargs['calkit'] = rc.AGILENT_ALAN

            s11 = S11Model.from_edges3_loaddef(caldef, load=load_name, **s11_kwargs)

        if s11.model_delay == 0 * un.s:
            s11 = s11.with_model_delay()

        return cls(
            spectrum=spec,
            reflections=s11,
            loss_model=loss_model,
            ambient_temperature=ambient_temperature,
        )
            

    def loss(self, freq: tp.FreqType | None = None):
        """The loss of this load."""
        if freq is None:
            freq = self.freq

        if self.loss_model is None:
            return np.ones(len(freq))

        return self.loss_model(freq, self.s11.s11_model(freq))

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
        return self.s11.s11_model

    @property
    def freq(self) -> tp.FreqType:
        """Frequencies of the spectrum."""
        return self.spectrum.q.freqs
