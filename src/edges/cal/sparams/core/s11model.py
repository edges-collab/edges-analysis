"""Methods for modeling and smoothing ReflectionCoefficient (S11) data."""

import logging
from collections.abc import Callable

import attrs
import numpy as np
from astropy import units as un

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


def get_delay(gamma: ReflectionCoefficient) -> un.Quantity[un.microsecond]:
    """Find the delay of an S11 with a grid search."""

    def _objfun(delay, gamma):
        reph = gamma.remove_delay(delay * un.microsecond)
        return -np.abs(np.sum(reph.reflection_coefficient))

    delays = np.arange(-1e-3, 0.1, 1e-4)
    obj = [_objfun(d, gamma) for d in delays]
    return delays[np.argmin(obj)] * un.microsecond


@hickleable
@attrs.define(kw_only=True, frozen=True)
class S11ModelParams:
    """A class holding parameters required to model an S11.

    Parameters
    ----------
    model
        The linear model used to fit each component of the data (real/imag or abs/phase)
    complex_model_type
        The type of complex model to use (ComplexMagPhaseModel or ComplexRealImagModel).
    find_model_delay
        Whether to find and remove a delay in the S11 data before fitting the model.
    optimize_model_delay
        Whether to optimize the model delay using a minimization routine (by default,
        do a simple grid search).
    model_delay
        If not finding the model delay, use this fixed delay value.
    set_transform_range
        Whether to set the transform range/scale based on the frequency range of the
        data.
    fit_method
        The fitting method to use when fitting the model to the data, see
        :func:`edges.modeling.fitting.ModelFit`.
    combine_s12s21
        Whether to fit to s12*s21 instead of fitting s12 and s21 separately.
    """

    model: Model = attrs.field(
        default=Fourier(n_terms=55, transform=UnitTransform(range=(0, 1)))
    )
    complex_model_type: type[ComplexMagPhaseModel] | type[ComplexRealImagModel] = (
        attrs.field(default=ComplexMagPhaseModel)
    )
    find_model_delay: bool = attrs.field(default=False)
    model_delay: tp.TimeType = attrs.field(default=0 * un.s)
    set_transform_range: bool = attrs.field(default=True, converter=bool)
    fit_method: str = attrs.field(default="lstsq")
    combine_s12s21: bool = attrs.field(default=True)

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

    delay = get_delay(gamma) if params.find_model_delay else params.model_delay

    gamma = gamma.remove_delay(delay)

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

    logger.info(f"Using S11 model with delay={model.delay}")

    return ReflectionCoefficient(freqs=freqs, reflection_coefficient=model(freqs))


def smooth_sparams(
    sparams: SParams,
    params: S11ModelParams | dict[str, S11ModelParams],
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
    if freqs is None:
        freqs = sparams.freqs

    if isinstance(params, S11ModelParams):
        params = {"s11": params, "s12": params, "s21": params, "s22": params}

    if not params["s11"].combine_s12s21:
        keys = ["s11", "s12", "s21", "s22"]
    else:
        keys = ["s11", "s22"]

    out = {
        param_name: new_s11_modelled(
            ReflectionCoefficient(
                freqs=sparams.freqs, reflection_coefficient=getattr(sparams, param_name)
            ),
            params[param_name],
            freqs=freqs,
        ).reflection_coefficient
        for param_name in keys
    }

    if params["s11"].combine_s12s21:
        s12_s21 = ReflectionCoefficient(
            freqs=sparams.freqs,
            reflection_coefficient=(sparams.s12 * sparams.s21),
        )
        s12_s21_smoothed = new_s11_modelled(
            s12_s21,
            params["s12"],
            freqs=freqs,
        ).reflection_coefficient

        out |= {
            "s12": np.sqrt(s12_s21_smoothed),
            "s21": np.sqrt(s12_s21_smoothed),
        }

    return SParams(freqs=freqs, **out)


# Attach methods to classes for
ReflectionCoefficient.smoothed = new_s11_modelled
SParams.smoothed = smooth_sparams
