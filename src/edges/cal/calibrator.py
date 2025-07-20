"""A module defining a Calibrator object that holds noise-wave solutions."""
from edges.io import hickleable
import attrs
from collections.abc import Callable
import numpy as np
from astropy import units as un
from functools import partial
from edges import types as tp
from typing import Self, Literal
import h5py
from ..tools import ComplexSpline, Spline
from .noise_waves import get_linear_coefficients
from edges.modelling import Model, CompositeModel
from .s11 import S11Model
from .load_data import Load

class CalFileReadError(Exception):
    pass

@hickleable
@attrs.define(kw_only=True, frozen=True)
class Calibrator:
    freq: tp.FreqType = attrs.field()

    C1: tp.FloatArray = attrs.field()
    C2: tp.FloatArray = attrs.field()
    Tunc: tp.FloatArray = attrs.field()
    Tcos: tp.FloatArray = attrs.field()
    Tsin: tp.FloatArray = attrs.field()
    
    receiver_s11: tp.ComplexArray = attrs.field()

    t_load: float = attrs.field(default=300)
    t_load_ns: float = attrs.field(default=350)

    def get_modelled(
        self, 
        thing: Literal['C1', 'C2', 'Tunc', 'Tcos', 'Tsin'],
        freq: tp.FreqType,
        model: Callable | Model | None = None,
    ) -> np.ndarray:
        """Evaluate a quantity at particular frequencies."""
        if not hasattr(self, thing):
            raise ValueError(f"thing must be one of C1, C2, Tunc, Tcos, Tsin or receiver_s11, got {thing}")
        
        fqin = self.freq.to_value("MHz")
        fqout = freq.to_value("MHz")
        this = getattr(self, thing)
        
        if model is None:
            model = partial(ComplexSpline, k=3) if np.iscomplex(this) else partial(Spline, k=3)
        
        if isinstance(model, Model):
            if thing == 'receiver_s11':
                raise ValueError("You need a Complex model to model receiver_s11")
            
            return model.at(x=fqin).fit(this).evaluate(fqout)
            
        elif isinstance(model, CompositeModel):
            return model.at(x=fqin).fit(this)(fqout)
        elif callable(model):
            return model(fqin, this)(fqout)
        else:
            raise ValueError("model given is not callable!")
                
    def clone(self, **kwargs):
        """Clone the instance with new parameters."""
        return attrs.evolve(self, **kwargs)

    @classmethod
    def from_calfile(cls, path: tp.PathLike) -> Self:
        """Generate from calfile."""
        return cls.from_file(path)  # added by hickleable
    
    
    def get_linear_coefficients(
        self, 
        ant_s11: S11Model | tp.ComplexArray,
        freq: tp.FreqType | None = None, 
        models: dict[str, Callable | Model | None] | None = None
    ):
        if models is None:
            models = {}
            
        if freq is None:
            freq = self.freq
            
            if isinstance(ant_s11, S11Model):
                ant_s11 = ant_s11.s11_model(freq)
        
            c1 = self.C1
            c2 = self.C2
            tunc = self.Tunc
            tcos = self.Tcos
            tsin = self.Tsin
            rcv  = self.receiver_s11
        else:
            c1 = self.get_modelled('C1', freq, model=models.get("C1"))
            c2 = self.get_modelled('C1', freq, model=models.get("C2"))
            tunc = self.get_modelled('C1', freq, model=models.get("Tunc"))
            tcos = self.get_modelled('C1', freq, model=models.get("Tcos"))
            tsin = self.get_modelled('C1', freq, model=models.get("Tsin"))
            rcv = self.get_modelled('C1', freq, model=models.get("receiver_s11"))
            
        if len(ant_s11) != len(freq):
            raise ValueError("ant_s11 was given as an array, but does not have the same shape as the frequencies!")
       
        return get_linear_coefficients(
            gamma_ant=ant_s11,
            gamma_rec=rcv,
            sca =c1,
            off = c2,
            t_unc=tunc,
            t_cos=tcos,
            t_sin=tsin,
            t_load=self.t_load,
        )

    def calibrate_load(
        self,
        load: Load,
        models: dict[str, Callable | Model | None] | None = None
    ) -> tp.TemperatureType:
        return self.calibrate_q(load.averaged_Q, ant_s11=load.s11_model, models=models)
    
    def calibrate_temp(
        self, 
        temp: tp.TemperatureType, 
        ant_s11: S11Model | tp.ComplexArray,
        freq: tp.FreqType | None = None,
        models: dict[str, Callable | Model | None] | None= None
    ) -> tp.TemperatureType:
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
        a, b = self.get_linear_coefficients(freq=freq, ant_s11=ant_s11, models=models)
        return temp * a + b

    def decalibrate_temp(
        self, temp: tp.TemperatureType, 
        ant_s11: S11Model | tp.ComplexArray, freq: tp.FreqType | None = None, models: dict[str, Callable | Model | None] | None = None) -> tp.TemperatureType:
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
        a, b = self.get_linear_coefficients(freq=freq, ant_s11=ant_s11, models=models)
        return (temp - b) / a

    def calibrate_q(
        self, 
        q: tp.TemperatureType, 
        ant_s11: S11Model | tp.ComplexArray,
        freq: tp.FreqType | None = None,
        models: dict[str, Callable | Model | None]  | None = None
    ) -> tp.TemperatureType:
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

        return self.calibrate_temp(freq, uncal_temp, ant_s11, models=models)

