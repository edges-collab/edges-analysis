"""21cm absorption feature models."""

import attrs
import numpy as np
from yabf import Component, Parameter


def flattened_gaussian(freqs, amp, tau, w, nu0):
    """Compute a flattened-gaussian model for the absorption feature."""
    B = (
        4
        * (freqs - nu0) ** 2
        / w**2
        * np.log(-1 / tau * np.log((1 + np.exp(-tau)) / 2))
    )
    return -amp * (1 - np.exp(-tau * np.exp(B))) / (1 - np.exp(-tau))


def simple_gaussian(freqs, amp, nu0, w):
    """Compute a simple gaussian absorption feature."""
    return -amp * np.exp(-((freqs - nu0) ** 2) / (2 * w**2))


@attrs.define
class FlattenedGaussian(Component):
    """Flattened-Gaussian absorption profile, ala Bowman+2018."""

    provides = ["eor_spectrum"]

    base_parameters = [
        Parameter("amp", 0.5, min=0, latex=r"a_{21}"),
        Parameter("tau", 7, min=0, latex=r"\tau"),
        Parameter("w", 17.0, min=0),
        Parameter("nu0", 75, min=0, latex=r"\nu_0"),
    ]

    freqs: np.ndarray = attrs.field(kw_only=True, eq=attrs.cmp_using(eq=np.array_equal))

    def calculate(self, ctx, **params):
        """Compute the flattened-gaussian absorption model."""
        return flattened_gaussian(self.freqs, **params)

    def spectrum(self, ctx, **params):
        """Return the spectrum from the context."""
        return ctx["eor_spectrum"]


@attrs.define
class GaussianAbsorptionProfile(Component):
    """Standard Gaussian absorption profile."""

    provides = ["eor_spectrum"]

    base_parameters = [
        Parameter("amp", 0.5, min=0, latex=r"a_{21}"),
        Parameter("w", 17.0, min=0),
        Parameter("nu0", 75, min=0, latex=r"\nu_0"),
    ]

    freqs: np.ndarray = attrs.field(kw_only=True, eq=attrs.cmp_using(eq=np.array_equal))

    def calculate(self, ctx, **params):
        """Compute the Gaussian model."""
        return simple_gaussian(self.freqs, **params)

    def spectrum(self, ctx, **params):
        """Return the spectrum from the context."""
        return ctx["eor_spectrum"]
