"""Physical and statistical tests of the linear-model fitting machinery.

Weights are inverse variances (w = 1/sigma^2) throughout.
"""

import numpy as np
import pytest

from edges import modeling as mdl
from edges.modeling.models import PhysicalIono

ALL_METHODS = ("lstsq", "qr", "alan-qrd", "qrd-c")

MOD1_REASON = (
    "MOD-1: the default 'lstsq' solver applies the weights to both A and y, "
    "minimising sum(w^2 r^2) (treating w as 1/sigma) instead of sum(w r^2) with "
    "w = 1/sigma^2; fix pending (result-changing)"
)
LSTSQ_XFAIL = pytest.param(
    "lstsq", marks=pytest.mark.xfail(strict=True, reason=MOD1_REASON)
)

# One instance of every concrete linear-model class, with "realistic" parameters.
MODEL_CASES = {
    "PhysicalLin": (mdl.PhysicalLin(n_terms=5), [1750, -90, 30, -8, 5]),
    "PhysicalIono": (PhysicalIono(n_terms=5), [1750, -90, -8, 5, 30]),
    "Polynomial": (
        mdl.Polynomial(n_terms=4, transform=mdl.UnitTransform(range=(50, 100))),
        [1, 2, -3, 0.5],
    ),
    "EdgesPoly": (mdl.EdgesPoly(n_terms=3), [1750, -10, 0.3]),
    "LinLog": (mdl.LinLog(n_terms=5), [1750, -90, 30, -8, 5]),
    "LogPoly": (mdl.LogPoly(n_terms=4), [3.2, -2.5, 0.1, 0.05]),
    "Fourier": (mdl.Fourier(n_terms=5, period=50.0), [1, 2, 3, -1, 0.5]),
    "FourierDay": (mdl.FourierDay(n_terms=5), [1, 2, 3, -1, 0.5]),
    "Composite": (
        mdl.CompositeModel(
            models={
                "fg": mdl.PhysicalLin(n_terms=3),
                "poly": mdl.Polynomial(
                    n_terms=2,
                    offset=1,
                    transform=mdl.UnitTransform(range=(50, 100)),
                ),
            }
        ),
        [1750, -90, 30, 0.5, -0.2],
    ),
}


def _heteroscedastic_setup(n: int = 60, seed: int = 42):
    rng = np.random.default_rng(seed)
    x = np.linspace(50, 100, n)
    sigma = np.exp(rng.normal(size=n))
    return rng, x, sigma, 1 / sigma**2


@pytest.mark.parametrize("name", list(MODEL_CASES))
def test_noise_free_recovery_non_uniform_weights(name: str):
    """A noise-free model is recovered exactly whatever the (positive) weights."""
    model, params = MODEL_CASES[name]
    _, x, _, w = _heteroscedastic_setup()
    fm = model.at(x=x)
    y = fm(parameters=params)

    fit = fm.fit(ydata=y, weights=w, method="qr")
    np.testing.assert_allclose(fit.model_parameters, params, rtol=1e-6, atol=1e-10)
    np.testing.assert_allclose(fit.evaluate(), y, rtol=1e-10, atol=0)


@pytest.mark.parametrize("method", ALL_METHODS)
def test_noise_free_recovery_all_solvers(method: str):
    """All solvers agree (exactly) on noise-free data with non-uniform weights."""
    model, params = MODEL_CASES["LinLog"]
    _, x, _, w = _heteroscedastic_setup()
    fm = model.at(x=x)
    fit = fm.fit(ydata=fm(parameters=params), weights=w, method=method)
    np.testing.assert_allclose(fit.model_parameters, params, rtol=1e-7)


@pytest.mark.parametrize("method", [LSTSQ_XFAIL, "qr", "alan-qrd", "qrd-c"])
def test_residuals_w_orthogonal_to_basis(method: str):
    """The WLS normal equations: A^T W r = 0 at the best fit."""
    rng, x, sigma, w = _heteroscedastic_setup()
    fm = mdl.Polynomial(n_terms=4, transform=mdl.UnitTransform(range=(50, 100))).at(x=x)
    y = fm(parameters=[1, 2, -3, 0.5]) + rng.normal(scale=sigma)
    fit = fm.fit(ydata=y, weights=w, method=method)

    normal_eq = fm.basis @ (w * fit.residual)
    # Compare to the size of the individual terms in the sum.
    scale = np.abs(fm.basis) @ np.abs(w * fit.residual)
    np.testing.assert_array_less(np.abs(normal_eq), 1e-8 * scale)


@pytest.mark.parametrize("method", ALL_METHODS)
def test_zero_weight_equals_dropping_point(method: str):
    """Giving a point zero weight is the same as removing it from the data."""
    rng, x, sigma, w = _heteroscedastic_setup()
    model = mdl.Polynomial(n_terms=4, transform=mdl.UnitTransform(range=(50, 100)))
    y = model(x=x, parameters=[1, 2, -3, 0.5]) + rng.normal(scale=sigma)

    drop = np.array([3, 17, 40])
    keep = np.setdiff1d(np.arange(x.size), drop)
    w0 = w.copy()
    w0[drop] = 0.0

    fit_zero = model.at(x=x).fit(ydata=y, weights=w0, method=method)
    fit_drop = model.at(x=x[keep]).fit(ydata=y[keep], weights=w[keep], method=method)

    np.testing.assert_allclose(
        fit_zero.model_parameters, fit_drop.model_parameters, rtol=1e-8
    )
    np.testing.assert_allclose(
        fit_zero.weighted_chi2, fit_drop.weighted_chi2, rtol=1e-8
    )
    np.testing.assert_allclose(fit_zero.hessian, fit_drop.hessian, rtol=1e-10)


@pytest.mark.parametrize("method", ALL_METHODS)
@pytest.mark.parametrize("c", [1e-3, 7.0, 1e4])
def test_weight_scaling_invariance(method: str, c: float):
    """Scaling all weights by c leaves the parameters alone and scales chi^2 by c."""
    rng, x, sigma, w = _heteroscedastic_setup()
    fm = mdl.LinLog(n_terms=4).at(x=x)
    y = fm(parameters=[1750, -90, 30, -8]) + rng.normal(scale=sigma)

    fit1 = fm.fit(ydata=y, weights=w, method=method)
    fitc = fm.fit(ydata=y, weights=c * w, method=method)

    np.testing.assert_allclose(fitc.model_parameters, fit1.model_parameters, rtol=1e-7)
    np.testing.assert_allclose(fitc.weighted_chi2, c * fit1.weighted_chi2, rtol=1e-6)
    np.testing.assert_allclose(
        fitc.parameter_covariance, fit1.parameter_covariance / c, rtol=1e-8
    )


def _mc_parameters(method: str, n_real: int = 1000, n: int = 40, seed: int = 5):
    rng = np.random.default_rng(seed)
    x = np.linspace(50, 100, n)
    fm = mdl.Polynomial(n_terms=3, transform=mdl.UnitTransform(range=(50, 100))).at(x=x)
    sigma = np.exp(rng.normal(size=n))
    w = 1 / sigma**2
    y0 = fm(parameters=[1.0, 2.0, 3.0])
    pars = np.array([
        fm.fit(
            ydata=y0 + rng.normal(scale=sigma), weights=w, method=method
        ).model_parameters
        for _ in range(n_real)
    ])
    cov = fm.fit(ydata=y0, weights=w, method=method).parameter_covariance
    return pars, cov


def test_mc_scatter_matches_covariance_qr():
    """The reported parameter covariance matches the Monte-Carlo scatter (qr)."""
    pars, cov = _mc_parameters("qr")
    # 1000 realisations: the fractional error on a std is ~2%.
    np.testing.assert_allclose(pars.std(axis=0), np.sqrt(np.diag(cov)), rtol=0.08)


@pytest.mark.xfail(strict=True, reason=MOD1_REASON)
def test_mc_scatter_matches_covariance_default_method():
    """The reported covariance matches the MC scatter for the default solver."""
    pars, cov = _mc_parameters("lstsq")
    np.testing.assert_allclose(pars.std(axis=0), np.sqrt(np.diag(cov)), rtol=0.08)


@pytest.mark.xfail(
    strict=True,
    reason=(
        "MOD-5: weighted_rms computes sqrt(sum(w r^2)) / sum(w) instead of "
        "sqrt(sum(w r^2) / sum(w)); fix pending (result-changing)"
    ),
)
def test_weighted_rms_unit_weights_is_rms():
    """With unit weights, the weighted RMS is the plain RMS of the residuals."""
    rng = np.random.default_rng(0)
    x = np.linspace(50, 100, 50)
    fm = mdl.Polynomial(n_terms=2).at(x=x)
    y = fm(parameters=[1.0, 0.01]) + rng.normal(size=x.size)
    fit = fm.fit(ydata=y, weights=np.ones(x.size))
    np.testing.assert_allclose(
        fit.weighted_rms, np.sqrt(np.mean(fit.residual**2)), rtol=1e-12
    )


MOD6_REASON = (
    "MOD-6: degrees_of_freedom is N - M - 1 and counts NaN and zero-weight points; "
    "it should be N - M over finite, positive-weight points; fix pending "
    "(result-changing)"
)


@pytest.mark.xfail(strict=True, reason=MOD6_REASON)
def test_degrees_of_freedom_counts_only_used_points():
    """The dof is the number of used (finite, w>0) points minus the parameters."""
    x = np.linspace(50, 100, 30)
    fm = mdl.Polynomial(n_terms=3, transform=mdl.UnitTransform(range=(50, 100))).at(x=x)
    y = fm(parameters=[1.0, 2.0, 3.0])
    w = np.ones(x.size)
    w[:4] = 0.0
    y[-2:] = np.nan

    fit = fm.fit(ydata=y, weights=w, method="qr")
    assert fit.degrees_of_freedom == 30 - 4 - 2 - 3


@pytest.mark.xfail(strict=True, reason=MOD6_REASON)
def test_mean_reduced_chi2_is_one():
    """For correctly-weighted Gaussian noise, <chi^2 / dof> = 1."""
    rng = np.random.default_rng(11)
    n = 10
    x = np.linspace(50, 100, n)
    fm = mdl.Polynomial(n_terms=3, transform=mdl.UnitTransform(range=(50, 100))).at(x=x)
    sigma = np.exp(rng.normal(size=n))
    w = 1 / sigma**2
    y0 = fm(parameters=[1.0, 2.0, 3.0])
    rchi2 = [
        fm.fit(
            ydata=y0 + rng.normal(scale=sigma), weights=w, method="qr"
        ).reduced_weighted_chi2
        for _ in range(2000)
    ]
    # Standard error of the mean here is ~0.015.
    np.testing.assert_allclose(np.mean(rchi2), 1.0, atol=0.06)


# ---------------------------------------------------------------------------
# Regression tests for Phase-2 bug fixes.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("method", ALL_METHODS)
@pytest.mark.parametrize("weighted", [True, False])
def test_nan_data_is_masked_consistently(method: str, weighted: bool):
    """MOD-7: NaN data is excluded from fit, chi^2, rms, Hessian and covariance."""
    rng, x, sigma, w = _heteroscedastic_setup()
    if not weighted:
        w = 1.0
    model = mdl.Polynomial(n_terms=4, transform=mdl.UnitTransform(range=(50, 100)))
    y = model(x=x, parameters=[1, 2, -3, 0.5]) + rng.normal(scale=sigma)

    bad = np.array([5, 22, 23])
    keep = np.setdiff1d(np.arange(x.size), bad)
    ynan = y.copy()
    ynan[bad] = np.nan

    fit_nan = model.at(x=x).fit(ydata=ynan, weights=w, method=method)
    fit_drop = model.at(x=x[keep]).fit(
        ydata=y[keep], weights=w if np.isscalar(w) else w[keep], method=method
    )

    np.testing.assert_allclose(
        fit_nan.model_parameters, fit_drop.model_parameters, rtol=1e-10
    )
    assert np.all(np.isnan(fit_nan.residual[bad]))
    np.testing.assert_allclose(fit_nan.residual[keep], fit_drop.residual, atol=1e-10)

    for attr in ("weighted_chi2", "weighted_rms", "hessian", "parameter_covariance"):
        val = getattr(fit_nan, attr)
        assert np.all(np.isfinite(val)), attr
        np.testing.assert_allclose(val, getattr(fit_drop, attr), rtol=1e-10)


def test_nan_masking_does_not_change_finite_results():
    """With no NaNs present, the statistics are exactly the unmasked expressions."""
    rng, x, sigma, w = _heteroscedastic_setup()
    fm = mdl.LinLog(n_terms=4).at(x=x)
    y = fm(parameters=[1750, -90, 30, -8]) + rng.normal(scale=sigma)
    fit = fm.fit(ydata=y, weights=w)

    r = fit.residual
    assert fit.weighted_chi2 == np.dot(r.T, w * r)
    assert fit.weighted_rms == np.sqrt(np.dot(r.T, w * r)) / np.sum(w)
    np.testing.assert_array_equal(fit.hessian, (fm.basis * w).dot(fm.basis.T))


@pytest.mark.parametrize("bad", [np.inf, np.nan, -1.0])
@pytest.mark.parametrize("method", ALL_METHODS)
def test_bad_weights_raise(bad: float, method: str):
    """MOD-8: a single infinite, NaN or negative weight gives a clear error."""
    x = np.linspace(50, 100, 20)
    fm = mdl.Polynomial(n_terms=3).at(x=x)
    w = np.ones(x.size)
    w[4] = bad
    with pytest.raises(ValueError, match="Weights must be"):
        fm.fit(ydata=np.ones(x.size), weights=w, method=method)


@pytest.mark.parametrize("bad", [np.inf, np.nan, -1.0])
def test_bad_scalar_weight_raises(bad: float):
    fm = mdl.Polynomial(n_terms=3).at(x=np.linspace(50, 100, 20))
    with pytest.raises(ValueError, match="Weights must be"):
        fm.fit(ydata=np.ones(20), weights=bad)


@pytest.mark.parametrize("weights", [None, 1, 3, np.float32(2.0)])
def test_weights_converter(weights):
    """MOD-19: None, int and numpy scalars are accepted as weights."""
    x = np.linspace(50, 100, 20)
    fm = mdl.Polynomial(n_terms=3, transform=mdl.UnitTransform(range=(50, 100))).at(x=x)
    y = fm(parameters=[1.0, 2.0, 3.0])
    fit = fm.fit(ydata=y, weights=weights)
    assert isinstance(fit.weights, float)
    assert fit.weights == (1.0 if weights is None else float(weights))
    np.testing.assert_allclose(fit.model_parameters, [1.0, 2.0, 3.0], rtol=1e-10)


def test_weights_list_converted_to_array():
    x = np.linspace(50, 100, 5)
    fm = mdl.Polynomial(n_terms=2).at(x=x)
    fit = fm.fit(ydata=x.copy(), weights=[1, 2, 3, 4, 5])
    assert isinstance(fit.weights, np.ndarray)
    assert fit.weights.dtype == float


# ---------------------------------------------------------------------------------
# Generalised least squares: a full (n, n) inverse-covariance weight matrix.
# ---------------------------------------------------------------------------------
GLS_METHODS = ("lstsq", "qr", "alan-qrd")


def _gls_setup(n: int = 50, seed: int = 11):
    """A polynomial model with correlated, heteroscedastic Gaussian noise."""
    x = np.linspace(50, 100, n)
    fm = mdl.Polynomial(n_terms=4, transform=mdl.UnitTransform(range=(50, 100))).at(x=x)
    sigma = np.linspace(0.5, 2.0, n)
    cov = np.outer(sigma, sigma) * np.exp(-np.abs(x[:, None] - x[None, :]) / 3.0)
    truth = np.array([1.0, 2.0, -3.0, 0.5])
    rng = np.random.default_rng(seed)
    y = fm(parameters=truth) + rng.multivariate_normal(np.zeros(n), cov)
    return fm, cov, np.linalg.inv(cov), y, truth


@pytest.mark.parametrize("method", GLS_METHODS)
def test_gls_matches_explicit_formula(method: str):
    fm, _, winv, y, _ = _gls_setup()
    a = fm.basis
    expected = np.linalg.solve(a @ winv @ a.T, a @ winv @ y)
    fit = mdl.ModelFit(fm, ydata=y, weights=winv, method=method)
    np.testing.assert_allclose(fit.model_parameters, expected, rtol=1e-8)
    # The residual is W-orthogonal to the basis.
    np.testing.assert_allclose(a @ winv @ fit.residual, 0, atol=1e-7)


@pytest.mark.parametrize("method", ["qr", "alan-qrd"])
def test_diagonal_weight_matrix_equals_1d_weights(method: str):
    fm, _, _, y, _ = _gls_setup()
    w = np.linspace(0.2, 3.0, y.size)
    fit1d = mdl.ModelFit(fm, ydata=y, weights=w, method=method)
    fit2d = mdl.ModelFit(fm, ydata=y, weights=np.diag(w), method=method)
    np.testing.assert_allclose(
        fit2d.model_parameters, fit1d.model_parameters, rtol=1e-10
    )
    np.testing.assert_allclose(fit2d.weighted_chi2, fit1d.weighted_chi2, rtol=1e-10)
    np.testing.assert_allclose(fit2d.hessian, fit1d.hessian, rtol=1e-10)


def test_gls_chi2_and_hessian():
    fm, _, winv, y, _ = _gls_setup()
    fit = mdl.ModelFit(fm, ydata=y, weights=winv, method="qr")
    r = fit.residual
    np.testing.assert_allclose(fit.weighted_chi2, r @ winv @ r, rtol=1e-12)
    a = fm.basis
    np.testing.assert_allclose(fit.hessian, a @ winv @ a.T, rtol=1e-12)


def test_gls_parameter_covariance_matches_monte_carlo():
    fm, cov, winv, _, truth = _gls_setup()
    rng = np.random.default_rng(3)
    model = fm(parameters=truth)
    params = np.array([
        mdl.ModelFit(
            fm,
            ydata=model + rng.multivariate_normal(np.zeros(model.size), cov),
            weights=winv,
            method="qr",
        ).model_parameters
        for _ in range(2000)
    ])
    predicted = mdl.ModelFit(fm, ydata=model, weights=winv, method="qr")
    np.testing.assert_allclose(
        np.std(params, axis=0),
        np.sqrt(np.diag(predicted.parameter_covariance)),
        rtol=0.08,
    )


@pytest.mark.parametrize(
    ("weights", "match"),
    [
        (np.eye(49), "shape"),
        (np.triu(np.ones((50, 50))) + np.eye(50), "symmetric"),
        (-np.eye(50), "positive definite"),
        (np.full((50, 50), np.nan), "finite"),
        (np.ones((2, 50, 50)), "scalar, a 1D array or a 2D"),
    ],
)
def test_bad_weight_matrix_raises(weights, match):
    fm, _, _, y, _ = _gls_setup()
    with pytest.raises(ValueError, match=match):
        mdl.ModelFit(fm, ydata=y, weights=weights, method="qr")


def test_weight_matrix_not_supported_by_qrd_c():
    fm, _, winv, y, _ = _gls_setup()
    with pytest.raises(ValueError, match="qrd_c"):
        _ = mdl.ModelFit(fm, ydata=y, weights=winv, method="qrd-c").model_parameters


def test_weight_matrix_with_nan_data_raises():
    fm, _, winv, y, _ = _gls_setup()
    y = y.copy()
    y[3] = np.nan
    with pytest.raises(ValueError, match="non-finite data"):
        mdl.ModelFit(fm, ydata=y, weights=winv, method="qr")


def test_weight_matrix_weighted_rms_not_defined():
    fm, _, winv, y, _ = _gls_setup()
    fit = mdl.ModelFit(fm, ydata=y, weights=winv, method="qr")
    with pytest.raises(NotImplementedError, match="weighted_rms"):
        _ = fit.weighted_rms
