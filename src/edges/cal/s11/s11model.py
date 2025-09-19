"""A class for setting parameters to model raw S11 measurements."""

import logging
from collections.abc import Callable
from typing import Self

import attrs
import numpy as np
from astropy import units as un
from scipy.interpolate import InterpolatedUnivariateSpline as Spline

from edges.modeling.models import Polynomial
from edges.modeling.xtransforms import UnitTransform

from ... import types as tp
from ...io.serialization import hickleable
from ...modeling import (
    ComplexMagPhaseModel,
    ComplexRealImagModel,
    Fourier,
    Model,
)
from .. import reflection_coefficient as rc
from .base import CalibratedS11

logger = logging.getLogger(__name__)


@hickleable
@attrs.define(kw_only=True, frozen=True)
class S11ModelParams:
    """A class holding parameters required to model an S11."""

    model: Model = attrs.field(
        default=Fourier(n_terms=55, transform=UnitTransform(range=(0, 1)))
    )
    complex_model_type: type[ComplexMagPhaseModel] | type[ComplexRealImagModel] = (
        attrs.field(default=ComplexMagPhaseModel)
    )
    find_model_delay: bool = attrs.field(default=False)
    model_delay: tp.TimeType = attrs.field(default=0 * un.s)
    set_transform_range: bool = attrs.field(default=True, converter=bool)
    use_spline: bool = attrs.field(default=False)
    fit_method: str = attrs.field(default="lstsq")

    def clone(self, **kwargs):
        """Clone with new parameters."""
        return attrs.evolve(self, **kwargs)

    @classmethod
    def from_calibration_load_defaults(
        cls, name: str, find_model_delay: bool = True, **kwargs
    ) -> Self:
        """Generate a default S11ModelParams from a calibration load name.

        This just sets the default number of terms.
        """
        default_nterms = {
            "ambient": 37,
            "hot_load": 37,
            "open": 105,
            "short": 105,
        }
        n_terms = default_nterms.get(name, 37)

        model = kwargs.pop(
            "model", Fourier(n_terms=n_terms, transform=UnitTransform(range=(0, 1)))
        )

        return cls(model=model, find_model_delay=find_model_delay, **kwargs)

    @classmethod
    def from_receiver_defaults(cls, find_model_delay: bool = True, **kwargs) -> Self:
        """Generate a default S11ModelParams for a receiver."""
        model = kwargs.pop(
            "model", Fourier(n_terms=37, transform=UnitTransform(range=(0, 1)))
        )

        return cls(model=model, find_model_delay=find_model_delay, **kwargs)

    @classmethod
    def from_hot_load_cable_defaults(cls, **kwargs) -> Self:
        """Generate a default S11ModelParams for a hot load cable."""
        model = kwargs.pop(
            "model", Polynomial(n_terms=21, transform=UnitTransform(range=(0, 1)))
        )

        return cls(
            model=model,
            complex_model_type=ComplexRealImagModel,
            set_transform_range=True,
            **kwargs,
        )

    @classmethod
    def from_internal_switch_defaults(cls, **kwargs) -> Self:
        """Generate a default S11ModelParams for an internal switch."""
        model = kwargs.pop(
            "model",
            Polynomial(
                n_terms=7,
                transform=UnitTransform(range=(0, 1)),
            ),
        )

        return cls(
            model=model,
            complex_model_type=kwargs.pop("complex_model_type", ComplexRealImagModel),
            find_model_delay=kwargs.pop("find_model_delay", False),
            set_transform_range=True,
            **kwargs,
        )


def new_s11_modelled(
    raw_s11: np.ndarray | CalibratedS11,
    params: S11ModelParams,
    new_freqs: tp.FreqType | None = None,
    freqs: tp.FreqType | None = None,
) -> CalibratedS11:
    """Create a new CalibratedS11 that has been smoothed/modelled.

    Parameters
    ----------
    raw_s11
        The input CalibratedS11 object.
    params
        The set of parameters defining the model used to smooth/interpolate.
    new_freqs
        Optional new frequencies onto which to interpolate. If not given, retain
        the same set of frequencies.
    freqs
        The frequencies associated with raw_s11. Only required if `raw_s11` is an
        array rather than a CalibratedS11.

    Returns
    -------
    modelled_s11
        A new CalibratedS11 object that has been smoothed.
    """
    if not isinstance(raw_s11, CalibratedS11):
        raw_s11 = CalibratedS11(s11=raw_s11, freqs=freqs)

    if new_freqs is None:
        new_freqs = raw_s11.freqs

    model = get_s11_model(
        params,
        raw_s11=raw_s11,
    )

    if isinstance(model, DelayedS11Model):
        logger.info("Using S11 model with delay=%s", model.delay)

    return CalibratedS11(
        s11=model(new_freqs),
        freqs=new_freqs,
    )


@attrs.define
class DelayedS11Model:
    """An S11 callable model that accounts for a delay in the complex values."""

    cmodel: ComplexMagPhaseModel | ComplexRealImagModel = attrs.field()
    delay: tp.TimeType = attrs.field(default=0 * un.s)

    def __call__(self, freq: tp.FreqType) -> np.ndarray:
        """Evaluate the model at a given frequency."""
        return self.cmodel(freq.to_value("MHz")) * np.exp(
            -1j * 2 * np.pi * (self.delay * freq).to_value("")
        )


def get_s11_model(
    params: S11ModelParams,
    raw_s11: CalibratedS11,
) -> Callable[[tp.FreqType], np.ndarray]:
    """Generate a callable model for the S11.

    This should closely match :meth:`s11_correction`.

    Parameters
    ----------
    raw_s11
        The raw s11 of the

    Returns
    -------
    callable :
        A function of one argument, f, which should be a frequency in the same units
        as `self.freq`.

    Raises
    ------
    ValueError
        If n_terms is not an integer, or not odd.
    """
    if params.use_spline:
        if params.complex_model_type == ComplexRealImagModel:
            splrl = Spline(raw_s11.freqs.to_value("MHz"), np.real(raw_s11.s11))
            splim = Spline(raw_s11.freqs.to_value("MHz"), np.imag(raw_s11.s11))
            return (
                lambda freq: splrl(freq.to_value("MHz"))
                + splim(freq.to_value("MHz")) * 1j
            )
        splmag = Spline(raw_s11.freqs.to_value("MHz"), np.abs(raw_s11.s11))
        splph = Spline(raw_s11.freqs.to_value("MHz"), np.angle(raw_s11.s11))
        return lambda freq: splmag(freq.to_value("MHz")) * np.exp(
            1j * splph(freq.to_value("MHz"))
        )

    transform = params.model.xtransform
    model = params.model
    if params.set_transform_range:
        if hasattr(transform, "range"):
            transform = attrs.evolve(
                transform,
                range=(
                    raw_s11.freqs.min().to_value("MHz"),
                    raw_s11.freqs.max().to_value("MHz"),
                ),
            )
        elif hasattr(transform, "scale"):
            transform = attrs.evolve(
                transform,
                scale=(
                    raw_s11.freqs.min().to_value("MHz")
                    + raw_s11.freqs.max().to_value("MHz")
                )
                / 2,
            )

        model = attrs.evolve(model, xtransform=transform)

    emodel = model.at(x=raw_s11.freqs.to_value("MHz"))

    cmodel = params.complex_model_type(emodel, emodel)

    if params.find_model_delay:
        delay = rc.get_delay(raw_s11.freqs, raw_s11.s11)
    else:
        delay = params.model_delay

    cmodel = cmodel.fit(
        ydata=raw_s11.s11 * np.exp(2 * np.pi * 1j * delay * raw_s11.freqs).to_value(""),
        method=params.fit_method,
    )

    return DelayedS11Model(cmodel=cmodel, delay=delay)
