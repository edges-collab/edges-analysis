"""Methods for modeling and smoothing ReflectionCoefficient (S11) data."""

import logging
from collections.abc import Callable

import attrs
import numpy as np
from astropy import units as un
from scipy.optimize import minimize
from scipy.signal.windows import blackmanharris

from edges import types as tp
from edges.io.serialization import hickleable
from edges.modeling import (
    ComplexMagPhaseModel,
    ComplexRealImagModel,
    Fourier,
    Model,
)
from edges.modeling.xtransforms import UnitTransform

from .datatypes import ReflectionCoefficient, SParams

logger = logging.getLogger(__name__)


def get_rough_delay(gamma: ReflectionCoefficient) -> un.Quantity[un.microsecond]:
    """Calculate the delay of an S11 using FFT."""
    nf = len(gamma.reflection_coefficient)
    power = np.abs(np.fft.fft(gamma.reflection_coefficient * blackmanharris(nf))) ** 2
    kk = np.fft.fftfreq(nf, d=gamma.freqs[1] - gamma.freqs[0])

    return -kk[np.argmax(power)]


def get_delay(
    gamma: ReflectionCoefficient, optimize: bool = False
) -> un.Quantity[un.microsecond]:
    """Find the delay of an S11 using a minimization routine."""
    freq = gamma.freqs.to_value("MHz")  # resulting delay in microsecond

    def _objfun(delay, gamma):
        reph = gamma.rephase(delay * un.microsecond)
        return -np.abs(np.sum(reph.reflection_coefficient))

    if optimize:
        start = -get_rough_delay(gamma)
        dk = 1 / (freq[1] - freq[0])
        res = minimize(
            _objfun, x0=(start,), bounds=((start - dk, start + dk),), args=(gamma,)
        )
        return res.x * un.microsecond

    delays = np.arange(-1e-3, 0.1, 1e-4)
    obj = [_objfun(d, gamma) for d in delays]
    return delays[np.argmin(obj)] * un.microsecond


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
    optimize_model_delay: bool = attrs.field(default=False)
    model_delay: tp.TimeType = attrs.field(default=0 * un.s)
    set_transform_range: bool = attrs.field(default=True, converter=bool)
    fit_method: str = attrs.field(default="lstsq")

    def clone(self, **kwargs):
        """Clone with new parameters."""
        return attrs.evolve(self, **kwargs)


@attrs.define
class DelayedS11Model:
    """An S11 callable model that accounts for a delay in the complex values."""

    cmodel: ComplexMagPhaseModel | ComplexRealImagModel = attrs.field()
    delay: tp.TimeType = attrs.field(default=0 * un.s)

    def __call__(self, freq: tp.FreqType) -> np.ndarray:
        """Evaluate the model at a given frequency."""
        return self.cmodel(freq.to_value("MHz")) * np.exp(
            -2j * np.pi * (self.delay * freq).to_value("")
        )


def get_s11_model(
    params: S11ModelParams,
    gamma: ReflectionCoefficient,
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
    transform = params.model.xtransform
    model = params.model
    if params.set_transform_range:
        if hasattr(transform, "range"):
            transform = attrs.evolve(
                transform,
                range=(
                    gamma.freqs.min().to_value("MHz"),
                    gamma.freqs.max().to_value("MHz"),
                ),
            )
        elif hasattr(transform, "scale"):
            transform = attrs.evolve(
                transform,
                scale=(
                    gamma.freqs.min().to_value("MHz")
                    + gamma.freqs.max().to_value("MHz")
                )
                / 2,
            )

        model = attrs.evolve(model, xtransform=transform)

    emodel = model.at(x=gamma.freqs.to_value("MHz"))

    cmodel = params.complex_model_type(emodel, emodel)

    if params.find_model_delay:
        delay = get_delay(gamma, optimize=params.optimize_model_delay)
    else:
        delay = params.model_delay

    gamma = gamma.rephase(delay)

    cmodel = cmodel.fit(
        ydata=gamma.reflection_coefficient,
        method=params.fit_method,
    )

    return DelayedS11Model(cmodel=cmodel, delay=delay)


def new_s11_modelled(
    gamma: ReflectionCoefficient,
    params: S11ModelParams,
    freqs: tp.FreqType | None = None,
) -> ReflectionCoefficient:
    """Create a new ReflectionCoefficient that has been smoothed/modelled.

    Parameters
    ----------
    raw_s11
        The input ReflectionCoefficient object.
    params
        The set of parameters defining the model used to smooth/interpolate.
    new_freqs
        Optional new frequencies onto which to interpolate. If not given, retain
        the same set of frequencies.

    Returns
    -------
    modelled_s11
        A new ReflectionCoefficient object that has been smoothed.
    """
    if freqs is None:
        freqs = gamma.freqs

    model = get_s11_model(params, gamma=gamma)

    if isinstance(model, DelayedS11Model):
        logger.info(f"Using S11 model with delay={model.delay}")

    return ReflectionCoefficient(freqs=freqs, reflection_coefficient=model(freqs))


def smooth_sparams(
    sparams: SParams,
    params: S11ModelParams,
    freqs: tp.FreqType | None = None,
) -> SParams:
    """Smooth all S-parameters using the S11 modeling procedure.

    Parameters
    ----------
    sparams
        The input SParams object.
    params
        The set of parameters defining the model used to smooth/interpolate.
    new_freqs
        Optional new frequencies onto which to interpolate. If not given, retain
        the same set of frequencies.

    Returns
    -------
    smoothed_sparams
        A new SParams object that has been smoothed.
    """
    s11 = new_s11_modelled(
        ReflectionCoefficient(freqs=sparams.freqs, reflection_coefficient=sparams.s11),
        params,
        freqs=freqs,
    ).reflection_coefficient

    s12 = new_s11_modelled(
        ReflectionCoefficient(freqs=sparams.freqs, reflection_coefficient=sparams.s12),
        params,
        freqs=freqs,
    ).reflection_coefficient
    s21 = new_s11_modelled(
        ReflectionCoefficient(freqs=sparams.freqs, reflection_coefficient=sparams.s21),
        params,
        freqs=freqs,
    ).reflection_coefficient
    s22 = new_s11_modelled(
        ReflectionCoefficient(freqs=sparams.freqs, reflection_coefficient=sparams.s22),
        params,
        freqs=freqs,
    ).reflection_coefficient

    return SParams(freqs=freqs, s11=s11, s12=s12, s21=s21, s22=s22)


# Attach methods to classes for
ReflectionCoefficient.smoothed = new_s11_modelled
SParams.smoothed = smooth_sparams
