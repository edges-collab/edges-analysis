"""Tests of the PartialLinearModel marginal likelihood."""

import numpy as np
import pytest
from scipy import integrate

from edges import modeling as mdl
from edges.inference import PartialLinearModel


def _brute_force_marginal_lnl(data: np.ndarray, var: np.ndarray) -> float:
    """Numerically integrate exp(lnL(theta)) over the single linear parameter theta.

    The model is d_i = theta (a constant), with Gaussian noise of variance var_i. The
    returned value drops the constant -(N - M)/2 log(2 pi), which the code omits.
    """
    n = data.size
    theta = np.linspace(data.min() - 10, data.max() + 10, 200_001)
    lnl = (
        -0.5 * (data[None, :] - theta[:, None]) ** 2 / var[None, :]
        - 0.5 * np.log(2 * np.pi * var)[None, :]
    ).sum(axis=1)
    mx = lnl.max()
    lnz = mx + np.log(integrate.simpson(np.exp(lnl - mx), x=theta))
    return lnz + 0.5 * (n - 1) * np.log(2 * np.pi)


def _one_param_plm(var: np.ndarray, seed: int = 1):
    rng = np.random.default_rng(seed)
    x = np.linspace(50, 100, var.size)
    data = 3.0 + rng.normal(scale=np.sqrt(var))
    plm = PartialLinearModel(
        linear_model=mdl.Polynomial(n_terms=1).at(x=x),
        data={"data": data, "data_variance": var},
        variance_func=lambda ctx, data: data["data_variance"],
    )
    return plm, data


def test_marginal_lnl_brute_force_homoscedastic():
    """Marginal log-likelihood equals brute-force integration (constant variance)."""
    var = np.full(30, 0.1)
    plm, data = _one_param_plm(var)
    np.testing.assert_allclose(
        plm()[0], _brute_force_marginal_lnl(data, var), rtol=0, atol=1e-6
    )


@pytest.mark.xfail(
    strict=True,
    reason=(
        "MOD-2: the marginal likelihood uses the chi^2 of the default 'lstsq' fit, "
        "which squares the inverse-variance weights (MOD-1), so lnL is wrong for "
        "non-uniform variance; fix pending (result-changing)"
    ),
)
def test_marginal_lnl_brute_force_heteroscedastic():
    """Marginal log-likelihood equals brute-force integration (varying variance)."""
    rng = np.random.default_rng(1)
    var = np.exp(rng.normal(size=30)) * 0.1
    plm, data = _one_param_plm(var)
    np.testing.assert_allclose(
        plm()[0], _brute_force_marginal_lnl(data, var), rtol=0, atol=1e-6
    )
