"""Likelihoods for linear foreground + EoR profile fitting."""

from functools import cached_property

import attrs
import numpy as np

from ..modeling import Model
from .eor_models import FlattenedGaussian, GaussianAbsorptionProfile
from .partial_linear_model import PartialLinearModel


@attrs.define(frozen=True, kw_only=True)
class LinearFG:
    """Classic traditional EoR fit to a calibrated sky spectrum.

    This class implements a standard foreground + 21cm absorption profile fit to
    a sky spectrum, where the foreground is modeled as a linear model and the 21cm
    absorption profile is modeled as a non-linear function (either a flattened
    Gaussian or a simple Gaussian). The linear foreground model is marginalized
    analytically using a PartialLinearModel.

    Parameters
    ----------
    freqs
        The frequency channels of the observation.
    t_sky
        The calibrated sky spectrum to fit to.
    data_variance
        The variance of the data at each frequency channel.
    fg
        A linear foreground model.
    cosmic_signal
        The 21cm absorption profile model. If not provided, defaults to a
        FlattenedGaussian model.
    """

    freqs: np.ndarray = attrs.field()
    t_sky: np.ndarray = attrs.field()
    data_variance: np.ndarray = attrs.field()
    fg: Model = attrs.field()
    cosmic_signal: FlattenedGaussian | GaussianAbsorptionProfile = attrs.field()

    @cosmic_signal.default
    def _eorcmp(self):
        return (FlattenedGaussian(freqs=self.freqs, params=("amp", "w", "tau", "nu0")),)

    @cached_property
    def partial_linear_model(self):
        """The PartialLinearModel instance for this fit.

        This is an actual likelihood and so can be used within a `yabf` sampler.
        """
        return PartialLinearModel(
            linear_model=self.fg.at(x=self.freqs),
            data={"t_sky": self.t_sky, "data_variance": self.data_variance},
            components=(self.cosmic_signal,),
            data_func=self.transform_data,
            variance_func=None,
        )

    def transform_data(self, ctx: dict, data: dict):
        """Transform the data by subtracting the cosmic signal.

        This method gets passed to the PartialLinearModel to define the likelihood.
        """
        return data["t_sky"] - ctx["eor_spectrum"]
