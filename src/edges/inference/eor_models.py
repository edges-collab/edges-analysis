"""21cm absorption feature models."""

import attrs
import numpy as np
from yabf import Component, Parameter


def flattened_gaussian(freqs, amp, tau, w, nu0):
    r"""Compute a flattened-Gaussian model for the absorption feature.

    This is the profile of Bowman et al. (2018), Eq. 2:

    .. math:: T(\nu) = -A \frac{1 - e^{-\tau e^B}}{1 - e^{-\tau}},
              \quad B = \frac{4 (\nu - \nu_0)^2}{w^2}
              \log\left[-\frac{1}{\tau}\log\left(\frac{1 + e^{-\tau}}{2}\right)\right]

    Parameters
    ----------
    freqs
        The frequencies at which to evaluate the profile.
    amp
        The depth of the absorption feature, :math:`A` (the profile is :math:`-A` at
        :math:`\nu_0`).
    tau
        The flattening parameter, :math:`\tau \geq 0`. The limit :math:`\tau \to 0`
        is a Gaussian with full width at half maximum ``w``, and is evaluated exactly
        at :math:`\tau = 0`.
    w
        The full width at half maximum (FWHM) of the profile, for any ``tau``. Note
        that this differs from :func:`simple_gaussian`, where ``w`` is the standard
        deviation.
    nu0
        The central frequency of the absorption feature.

    Notes
    -----
    The expression is evaluated with ``expm1`` and ``log1p`` so that it is accurate
    for small ``tau``.
    """
    tau = np.asarray(tau, dtype=float)
    is_zero = tau == 0
    safe_tau = np.where(is_zero, 1.0, tau)

    # log[-log((1 + e^-tau)/2) / tau], which tends to -log(2) as tau -> 0.
    log_arg = np.where(
        is_zero,
        np.log(0.5),
        np.log(-np.log1p(np.expm1(-safe_tau) / 2) / safe_tau),
    )
    exp_b = np.exp(4 * (freqs - nu0) ** 2 / w**2 * log_arg)

    # (1 - e^{-tau e^B}) / (1 - e^{-tau}), which tends to e^B as tau -> 0.
    shape = np.where(is_zero, exp_b, np.expm1(-safe_tau * exp_b) / np.expm1(-safe_tau))
    return -amp * shape


def simple_gaussian(freqs, amp, nu0, w):
    """Compute a simple gaussian absorption feature.

    Parameters
    ----------
    freqs
        The frequencies at which to evaluate the profile.
    amp
        The depth of the absorption feature (the profile is ``-amp`` at ``nu0``).
    nu0
        The central frequency of the absorption feature.
    w
        The standard deviation of the Gaussian (*not* the FWHM, unlike
        :func:`flattened_gaussian`).
    """
    return -amp * np.exp(-((freqs - nu0) ** 2) / (2 * w**2))


@attrs.define(frozen=True, kw_only=True, slots=False)
class FlattenedGaussian(Component):
    """Flattened-Gaussian absorption profile, ala Bowman+2018.

    The parameter ``w`` is the full width at half maximum. See
    :func:`flattened_gaussian`.
    """

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


@attrs.define(frozen=True, kw_only=True, slots=False)
class GaussianAbsorptionProfile(Component):
    """Standard Gaussian absorption profile.

    The parameter ``w`` is the standard deviation (not the FWHM). See
    :func:`simple_gaussian`.
    """

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
