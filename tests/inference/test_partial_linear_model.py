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
        "The marginal likelihood uses the chi^2 of the default 'lstsq' fit, "
        "which squares the inverse-variance weights, so lnL is wrong for "
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


# ---------------------------------------------------------------------------
# Regression tests for Phase-2 bug fixes.
# ---------------------------------------------------------------------------


def _static_plm(var: np.ndarray, n_terms: int = 3, seed: int = 3, **kw):
    rng = np.random.default_rng(seed)
    x = np.linspace(50, 100, var.size)
    lm = mdl.Polynomial(
        n_terms=n_terms, transform=mdl.UnitTransform(range=(50, 100))
    ).at(x=x)
    noise_scale = np.sqrt(np.where(np.isfinite(var) & (var > 0), var, 1.0))
    data = lm(parameters=np.arange(1, n_terms + 1)) + rng.normal(scale=noise_scale)
    return PartialLinearModel(
        linear_model=lm, data={"data": data, "data_variance": var}, **kw
    )


@pytest.mark.parametrize("bad", [0.0, -1.0, np.nan])
def test_bad_variance_raises(bad: float):
    """A single zero, negative or NaN variance gives a clear error."""
    var = np.ones(20)
    var[7] = bad
    plm = _static_plm(var)
    with pytest.raises(ValueError, match="variance"):
        plm()


def test_all_zero_variance_is_unweighted():
    """All-zero variance is (still) treated as an unweighted fit."""
    plm = _static_plm(np.zeros(20))
    assert np.isfinite(plm()[0])


def test_infinite_variance_flags_channel():
    """An infinite variance removes the channel, exactly as if it were dropped."""
    var = np.exp(np.random.default_rng(0).normal(size=20))
    var_inf = var.copy()
    var_inf[7] = np.inf
    keep = np.arange(20) != 7

    var_func = {"variance_func": lambda ctx, data: data["data_variance"]}
    plm_inf = _static_plm(var_inf, **var_func)
    data = plm_inf.data["data"]
    plm_drop = PartialLinearModel(
        linear_model=plm_inf.linear_model.at_x(plm_inf.linear_model.x[keep]),
        data={"data": data[keep], "data_variance": var[keep]},
        **var_func,
    )
    np.testing.assert_allclose(plm_inf()[0], plm_drop()[0], rtol=1e-10)


def test_logdet_cinv_no_overflow():
    """log|Q| is computed with slogdet, so it does not overflow."""
    n_terms = 12
    plm1 = _static_plm(np.full(200, 1.0), n_terms=n_terms)
    plm_tiny = _static_plm(np.full(200, 1e-30), n_terms=n_terms)

    assert np.isfinite(plm_tiny.logdetCinv)
    # Q scales as 1/variance, so log|Q| shifts by -M log(c).
    np.testing.assert_allclose(
        plm_tiny.logdetCinv,
        plm1.logdetCinv - n_terms * np.log(1e-30),
        rtol=1e-10,
    )


def test_logdet_cinv_from_hessian_no_overflow():
    """The non-static (variance_func) branch also uses slogdet."""
    var = np.full(200, 1e-30)
    plm = _static_plm(var, n_terms=12, variance_func=lambda ctx, data: var)
    model = plm.reduce_model()
    val = plm.logdet_cinv(model, None)
    assert np.isfinite(val)
    np.testing.assert_allclose(val, np.linalg.slogdet(model[0].hessian)[1])


def test_sigma_plus_v_inverse_flat_prior_limit():
    """sigma_plus_v_inverse is (Sigma + A^T V A)^-1 for an uninformative prior V.

    Compared to a brute-force inverse with a very broad prior on a well-conditioned
    problem, and checked to annihilate the foreground basis and to reproduce the
    minimum chi^2 of the exact weighted least-squares fit.
    """
    rng = np.random.default_rng(7)
    var = np.exp(rng.normal(size=12)) * 0.5
    plm = _static_plm(var, n_terms=2)
    pinv = plm.sigma_plus_v_inverse

    a = plm.linear_model.basis  # (M, N)
    lam = 1e7
    brute = np.linalg.inv(np.diag(var) + lam * a.T @ a)
    np.testing.assert_allclose(pinv, brute, rtol=0, atol=1e-5)

    np.testing.assert_allclose(pinv @ a.T, 0, atol=1e-10)

    d = plm.data["data"]
    w = 1 / var
    theta = np.linalg.solve((a * w) @ a.T, (a * w) @ d)
    chi2_min = np.sum(w * (d - theta @ a) ** 2)
    np.testing.assert_allclose(d @ pinv @ d, chi2_min, rtol=1e-10)
