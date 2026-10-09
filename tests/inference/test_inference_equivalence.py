"""Equivalence of the cached likelihood code paths with straightforward ones.

Each test compares the library implementation with a simple reference
implementation (the original algorithm) on realistic random inputs.
"""

import numpy as np
import pytest

from edges import modeling as mdl
from edges.inference import FlattenedGaussian, SemiLinearFit


def _slf_setup(n: int, sigma_kind: str, nan: bool = False):
    rng = np.random.default_rng(11)
    freqs = np.linspace(50, 100, n)
    fg = mdl.LinLog(n_terms=5).at(x=freqs)
    eor = FlattenedGaussian(freqs=freqs, params=("amp", "w", "tau", "nu0"))
    data = (
        fg(parameters=[2000, 10, -10, 5, -5])
        + eor()["eor_spectrum"]
        + rng.normal(scale=0.01, size=n)
    )
    if nan:
        data[3::29] = np.nan
    sig = rng.uniform(0.005, 0.02, size=n)
    if sigma_kind == "scalar":
        sigma = 0.01
    elif sigma_kind == "1D":
        sigma = sig
    else:
        sigma = np.diag(sig**2) + 1e-6 * np.exp(
            -(np.subtract.outer(freqs, freqs) ** 2) / 4
        )
    p0 = np.array([a.fiducial for a in eor.child_active_params])
    return fg, eor, data, sigma, [p0, p0 * 1.1, p0 * 0.93]


def _ref_fg_fit(slf: SemiLinearFit, p):
    """The FG fit, done from scratch (the original algorithm)."""
    resid = slf.spectrum - slf.get_eor(p)
    if np.ndim(slf.sigma) == 2:
        return slf.fg.fit(ydata=resid, weights=slf._cov_inverse, method="qr")
    return slf.fg.fit(
        ydata=resid,
        weights=1 / slf.sigma**2 if hasattr(slf.sigma, "__len__") else 1.0,
    )


@pytest.mark.parametrize("sigma_kind", ["scalar", "1D", "cov"])
@pytest.mark.parametrize("nan", [False, True])
def test_semi_linear_fit_cached_fg_fit_is_bit_identical(sigma_kind: str, nan: bool):
    """The cached FG solve gives exactly the parameters/residuals of a full fit."""
    if nan and sigma_kind == "cov":
        pytest.skip("A covariance needs finite data.")
    fg, eor, data, sigma, ps = _slf_setup(150, sigma_kind, nan)
    slf = SemiLinearFit(fg=fg, eor=eor, spectrum=data, sigma=sigma)
    for p in ps:
        ref = _ref_fg_fit(slf, p)
        assert slf.fg_params(p) == ref.model_parameters
        np.testing.assert_array_equal(slf.get_resid(p), ref.residual)
        fit = slf.fg_fit(p)
        assert fit.model_parameters == ref.model_parameters
        np.testing.assert_array_equal(fit.residual, ref.residual)
