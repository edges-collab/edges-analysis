"""A module defining a Calibrator object that holds noise-wave solutions."""

from collections.abc import Callable
from functools import partial
from typing import Literal, Self

import attrs
import numpy as np
from astropy import units as un

from edges import types as tp
from edges.io import hickleable
from edges.modeling import CompositeModel, Model

from ..tools import ComplexSpline, Spline
from ..units import vld_unit
from .input_sources import InputSource
from .noise_waves import get_linear_coefficients
from .sparams import ReflectionCoefficient, S11ModelParams

_CALIBRATOR_QUANTITIES = ("Tsca", "Toff", "Tunc", "Tcos", "Tsin", "receiver_s11")


def _freqs_to_mhz(freqs: tp.FreqType) -> tp.FreqType:
    """Express frequency Quantities in MHz, leaving anything else to the validator."""
    if isinstance(freqs, un.Quantity) and freqs.unit.physical_type == "frequency":
        return freqs.to(un.MHz)
    return freqs


@hickleable
@attrs.define(kw_only=True, frozen=True)
class Calibrator:
    """A class holding all information required to perform receiver calibration.

    This object makes sense in the context of the noise-wave formalism.

    Parameters
    ----------
    freqs
        The frequencies at which the calibration quantities are defined, as an astropy
        Quantity with frequency units (stored in MHz).
    Tsca, Toff, Tunc, Tcos, Tsin
        The scale, offset and noise-wave temperatures at each frequency.
    receiver_s11
        The reflection coefficient of the receiver at each frequency.
    unit
        The unit of the temperatures.
    """

    freqs: tp.FreqType = attrs.field(
        converter=_freqs_to_mhz,
        validator=vld_unit("frequency"),
        eq=attrs.cmp_using(eq=np.allclose),
    )

    Tsca: tp.FloatArray = attrs.field(eq=attrs.cmp_using(eq=np.allclose))
    Toff: tp.FloatArray = attrs.field(eq=attrs.cmp_using(eq=np.allclose))
    Tunc: tp.FloatArray = attrs.field(eq=attrs.cmp_using(eq=np.allclose))
    Tcos: tp.FloatArray = attrs.field(eq=attrs.cmp_using(eq=np.allclose))
    Tsin: tp.FloatArray = attrs.field(eq=attrs.cmp_using(eq=np.allclose))

    receiver_s11: tp.ComplexArray = attrs.field(eq=attrs.cmp_using(eq=np.allclose))
    unit: un.Unit = attrs.field(default=un.K)

    def __attrs_post_init__(self):
        """Check that all quantities are defined at every frequency."""
        nfreq = len(self.freqs)
        for name in _CALIBRATOR_QUANTITIES:
            if len(getattr(self, name)) != nfreq:
                raise ValueError(
                    f"{name} must have the same length as freqs ({nfreq}), "
                    f"got {len(getattr(self, name))}"
                )

    def get_modelled(
        self,
        thing: Literal["Tsca", "Toff", "Tunc", "Tcos", "Tsin", "receiver_s11"],
        freq: tp.FreqType,
        model: Callable | Model | None = None,
    ) -> np.ndarray:
        """Evaluate a quantity at particular frequencies."""
        if thing not in _CALIBRATOR_QUANTITIES:
            raise ValueError(
                f"thing must be one of {', '.join(_CALIBRATOR_QUANTITIES)}, got {thing}"
            )

        fqin = self.freqs.to_value("MHz")
        fqout = freq.to_value("MHz")
        this = getattr(self, thing)

        if model is None:
            model = (
                partial(ComplexSpline, k=3)
                if np.iscomplexobj(this)
                else partial(Spline, k=3)
            )

        if isinstance(model, Model):
            if thing == "receiver_s11":
                raise ValueError("You need a Complex model to model receiver_s11")

            return model.at(x=fqin).fit(this).evaluate(fqout)

        if isinstance(model, CompositeModel):
            return model.at(x=fqin).fit(this).evaluate(fqout)
        if callable(model):
            return model(fqin, this)(fqout)
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
        ant_s11: ReflectionCoefficient | tp.ComplexArray,
        freqs: tp.FreqType | None = None,
        models: dict[str, Callable | Model | None] | None = None,
        s11_model_params: S11ModelParams | None = None,
    ):
        """Return the frequency-dependent linear coefficients required to calibrate.

        The returned coefficients a and b are such that

        T_cal = a*Q + b

        Parameters
        ----------
        ant_s11
            The antenna reflection coefficient. Either an array defined at ``freqs``,
            or a :class:`ReflectionCoefficient`. A ``ReflectionCoefficient`` on a
            different frequency grid is modelled with ``s11_model_params`` and
            evaluated at ``freqs``.
        freqs
            The frequencies at which to compute the coefficients. By default, the
            frequencies of the calibrator.
        models
            A dictionary of models to use to interpolate the calibration
            coefficients. If None, interpolate with splines.
        s11_model_params
            How to model ``ant_s11`` if it is a ``ReflectionCoefficient`` that is not
            defined at ``freqs``. By default, ``S11ModelParams()``.
        """
        if models is None:
            models = {}

        if freqs is None or (
            len(freqs) == len(self.freqs) and np.allclose(freqs, self.freqs)
        ):
            freqs = self.freqs

            tsca = self.Tsca
            toff = self.Toff
            tunc = self.Tunc
            tcos = self.Tcos
            tsin = self.Tsin
            rcv = self.receiver_s11
        else:
            tsca = self.get_modelled("Tsca", freqs, model=models.get("Tsca"))
            toff = self.get_modelled("Toff", freqs, model=models.get("Toff"))
            tunc = self.get_modelled("Tunc", freqs, model=models.get("Tunc"))
            tcos = self.get_modelled("Tcos", freqs, model=models.get("Tcos"))
            tsin = self.get_modelled("Tsin", freqs, model=models.get("Tsin"))
            rcv = self.get_modelled(
                "receiver_s11", freqs, model=models.get("receiver_s11")
            )

        if isinstance(ant_s11, ReflectionCoefficient):
            if ant_s11.s11.size != freqs.size or not np.allclose(ant_s11.freqs, freqs):
                ant_s11 = ant_s11.smoothed(
                    params=s11_model_params or S11ModelParams(), freqs=freqs
                )
            ant_s11 = ant_s11.s11

        elif len(ant_s11) != len(freqs):
            raise ValueError(
                "ant_s11 was given as an array, but does not have the same shape as "
                "the frequencies!"
            )

        a, b = get_linear_coefficients(
            gamma_ant=ant_s11,
            gamma_rec=rcv,
            t_sca=tsca,
            t_off=toff,
            t_unc=tunc,
            t_cos=tcos,
            t_sin=tsin,
        )
        a <<= self.unit
        b <<= self.unit
        return a, b

    def calibrate_load(
        self,
        load: InputSource,
        models: dict[str, Callable | Model | None] | None = None,
    ) -> tp.TemperatureType:
        """Calibrate a :class:`Load` object, returning the calibrated temperature."""
        return self.calibrate_q(
            load.averaged_q,
            ant_s11=load.reflection_coefficient.s11,
            freqs=load.freqs,
            models=models,
        )

    def calibrate_q(
        self,
        q: np.ndarray,
        ant_s11: ReflectionCoefficient | tp.ComplexArray,
        freqs: tp.FreqType | None = None,
        models: dict[str, Callable | Model | None] | None = None,
        s11_model_params: S11ModelParams | None = None,
    ) -> tp.TemperatureType:
        """
        Calibrate power-ratio measurements.

        Parameters
        ----------
        q
            The power-ratio measurements.
        ant_s11
            The antenna S11 for the load.
        freqs
            The frequencies at which to calibrate
        models
            A dictionary of models to use to interpolate the calibration
            coefficients. If None, interpolate with splines.
        s11_model_params
            How to model ``ant_s11`` if it is a :class:`ReflectionCoefficient` that is
            not defined at ``freqs``. By default, ``S11ModelParams()``.

        Returns
        -------
        temp : np.ndarray
            The calibrated temperature.
        """
        a, b = self.get_linear_coefficients(
            freqs=freqs,
            ant_s11=ant_s11,
            models=models,
            s11_model_params=s11_model_params,
        )
        return q * a + b

    def decalibrate(
        self,
        temp: tp.TemperatureType,
        ant_s11: ReflectionCoefficient | tp.ComplexArray,
        freqs: tp.FreqType | None = None,
        models: dict[str, Callable | Model | None] | None = None,
        s11_model_params: S11ModelParams | None = None,
    ) -> tp.TemperatureType:
        """
        De-calibrate given calibrated spectrum.

        Parameters
        ----------
        temp
            The spectrum to decalibrate (in K)
        ant_s11
            The antenna S11 for the load.
        freqs
            The frequencies at which to calibrate
        models
            A dictionary of models to use to interpolate the calibration
            coefficients. If None, interpolate with splines.
        s11_model_params
            How to model ``ant_s11`` if it is a :class:`ReflectionCoefficient` that is
            not defined at ``freqs``. By default, ``S11ModelParams()``.

        Returns
        -------
        q
            The uncalibrated power-ratio.

        Notes
        -----
        Using this and then :meth:`calibrate_q` immediately should be an identity
        operation.
        """
        a, b = self.get_linear_coefficients(
            freqs=freqs,
            ant_s11=ant_s11,
            models=models,
            s11_model_params=s11_model_params,
        )
        return (temp - b) / a

    def calibrate_approximate_temperature(
        self,
        temp: tp.FloatArray,
        t_load: float,
        t_load_ns: float,
        ant_s11: ReflectionCoefficient | tp.ComplexArray,
        freqs: tp.FreqType | None = None,
        models: dict[str, Callable | Model | None] | None = None,
    ) -> tp.TemperatureType:
        """
        Calibrate "approximate" temperatures, Tapprox = t_load_ns*Q + t_load.

        Parameters
        ----------
        temp
            The approximate temperature to calibrate.
        t_load
            The "guess" of the load temperature
        t_load_ns
            The guess of the load+noise-source temperature.
        ant_s11
            The antenna S11 for the load.
        freqs
            The frequencies at which to calibrate
        models
            A dictionary of models to use to interpolate the calibration
            coefficients. If None, interpolate with splines.

        Returns
        -------
        temp : np.ndarray
            The calibrated temperature.
        """
        q = (temp - t_load) / t_load_ns
        return self.calibrate_q(freqs=freqs, q=q, ant_s11=ant_s11, models=models)
