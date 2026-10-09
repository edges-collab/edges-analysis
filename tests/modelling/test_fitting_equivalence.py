"""Equivalence of the optimised fitting/basis code paths with straightforward ones.

Each test compares the library implementation with a simple reference
implementation (the original algorithm) on realistic random inputs.
"""

import numpy as np
import pytest
import scipy as sp

from edges import modeling as mdl
from edges.modeling.fitting import _c_qrd, _RepeatedFit


def _ref_qr(basis: np.ndarray, y: np.ndarray, w: np.ndarray | float) -> np.ndarray:
    """The QR solve with a dense diagonal weight matrix (the original algorithm)."""
    w = np.eye(len(y)) if np.isscalar(w) else np.diag(w)
    sqrtw = np.sqrt(w)
    q, r = sp.linalg.qr(np.dot(sqrtw, basis.T), mode="economic")
    return sp.linalg.solve(r, np.dot(q.T, np.dot(sqrtw, y)))


def _ref_alan_qrd(
    basis: np.ndarray, y: np.ndarray, w: np.ndarray | float
) -> np.ndarray:
    """Alan's normal-equation solve with a dense diagonal weight matrix (original)."""
    w = np.eye(len(y)) if np.isscalar(w) else np.diag(w)
    wa = np.dot(basis, w)
    return _c_qrd(np.dot(wa, basis.T), np.dot(wa, y))[1]


def _fit_setup(n: int, seed: int = 0, nan: bool = False):
    rng = np.random.default_rng(seed)
    x = np.linspace(50, 200, n)
    fm = mdl.Polynomial(n_terms=7, transform=mdl.ScaleTransform(scale=75)).at(x=x)
    y = 1000 * (x / 75) ** -2.5 + rng.normal(size=n)
    w = rng.uniform(0.5, 2, size=n)
    w[::17] = 0
    if nan:
        y[5::31] = np.nan
    return fm, y, w


@pytest.mark.parametrize("n", [300, 2000])
@pytest.mark.parametrize("nan", [False, True])
@pytest.mark.parametrize("weighted", [False, True])
def test_qr_row_scaling_matches_dense_diagonal(n: int, nan: bool, weighted: bool):
    """The 'qr' method with row scaling equals the dense diag(w) algorithm."""
    fm, y, w = _fit_setup(n, nan=nan)
    w = w if weighted else 2.5
    fit = fm.fit(ydata=y, weights=w, method="qr")

    mask = np.isfinite(y)
    ref = _ref_qr(fm.basis[:, mask], y[mask], w if np.isscalar(w) else w[mask])
    np.testing.assert_allclose(fit.model_parameters, ref, rtol=1e-12, atol=0)


@pytest.mark.parametrize("n", [300, 2000])
@pytest.mark.parametrize("nan", [False, True])
@pytest.mark.parametrize("weighted", [False, True])
def test_alan_qrd_row_scaling_is_bit_identical(n: int, nan: bool, weighted: bool):
    """The 'alan-qrd' method is bit-for-bit identical to the dense algorithm.

    The normal equations are ill-conditioned and solved in extended precision, so
    any change in the last bits of the normal equations can change the result at
    the 1e-6 level: the comparison must be exact to keep Alan comparisons unchanged.
    """
    fm, y, w = _fit_setup(n, nan=nan)
    w = w if weighted else 2.5
    fit = fm.fit(ydata=y, weights=w, method="alan-qrd")

    mask = np.isfinite(y)
    ref = _ref_alan_qrd(fm.basis[:, mask], y[mask], w if np.isscalar(w) else w[mask])
    np.testing.assert_array_equal(fit.model_parameters, ref)


class _CountingScaler:
    """A basis scaler that counts how often it is evaluated."""

    def __init__(self, power: float):
        self.power = power
        self.calls = 0

    def __call__(self, x: np.ndarray) -> np.ndarray:
        self.calls += 1
        return (x / 75) ** self.power


def _basis_models(scalers: list[_CountingScaler] | None = None) -> dict:
    s1, s2 = scalers or (None, None)
    tr = mdl.ScaleTransform(scale=75.0)
    return {
        "LinLog": mdl.LinLog(n_terms=6),
        "PhysicalLin": mdl.PhysicalLin(n_terms=5),
        "Fourier": mdl.Fourier(n_terms=9, period=40.0),
        "FourierDay": mdl.FourierDay(n_terms=7),
        "LogPoly": mdl.LogPoly(n_terms=4),
        "scaled-poly": mdl.Polynomial(n_terms=5, transform=tr, basis_scaler=s1),
        "composite": mdl.CompositeModel(
            models={
                "a": mdl.Polynomial(n_terms=4, transform=tr, basis_scaler=s1),
                "b": mdl.Polynomial(n_terms=3, transform=tr, basis_scaler=s2),
                "c": mdl.EdgesPoly(n_terms=3),
            }
        ),
    }


def _ref_basis(model, x: np.ndarray) -> np.ndarray:
    """The basis built one term at a time (the original algorithm)."""
    return np.array([
        model.get_basis_term_transformed(indx, x) for indx in range(model.n_terms)
    ])


@pytest.mark.parametrize("name", list(_basis_models()))
def test_basis_matches_per_term_construction(name: str):
    """The cached basis (and get_basis_terms) equal the per-term construction."""
    model = _basis_models([_CountingScaler(0.3), _CountingScaler(-1.2)])[name]
    x = np.linspace(42, 198, 1500)
    ref = _ref_basis(model, x)
    np.testing.assert_array_equal(model.at(x=x).basis, ref)
    np.testing.assert_array_equal(model.get_basis_terms(x), ref)


@pytest.mark.parametrize("name", ["LinLog", "scaled-poly"])
def test_basis_with_partial_init_basis(name: str):
    """Growing the number of terms re-uses the old terms and computes the new ones."""
    model = _basis_models([_CountingScaler(0.3), _CountingScaler(-1.2)])[name]
    x = np.linspace(42, 198, 700)
    small = model.with_nterms(n_terms=3).at(x=x)
    grown = small.with_nterms(n_terms=model.n_terms)
    np.testing.assert_array_equal(grown.basis, _ref_basis(model, x))


def test_basis_scaler_evaluated_once_per_sub_model():
    """Each sub-model's coordinate transform and basis scaler are evaluated once."""
    scalers = [_CountingScaler(0.3), _CountingScaler(-1.2)]
    model = _basis_models(scalers)["composite"]
    _ = model.at(x=np.linspace(50, 100, 50)).basis
    assert [s.calls for s in scalers] == [1, 1]


@pytest.mark.parametrize("n_terms", [1, 5, 10])
@pytest.mark.parametrize("beta", [-2.5, -2.57])
def test_linlog_shared_terms_match_per_term(n_terms: int, beta: float):
    """LinLog's terms computed together equal its per-term construction exactly."""
    model = mdl.LinLog(n_terms=n_terms, beta=beta, f_center=80.0)
    x = np.random.default_rng(1).uniform(40, 200, size=4000)
    ref = _ref_basis(model, x)
    np.testing.assert_array_equal(model.get_basis_terms(x), ref)
    np.testing.assert_array_equal(model.at(x=x).basis, ref)

    xt = model.xtransform(x)
    for indx, term in zip([3, 0], model._get_basis_terms_at([3, 0], xt), strict=False):
        if indx < n_terms:
            np.testing.assert_array_equal(term, model.get_basis_term(indx, xt))


def _ref_ls(van: np.ndarray, y: np.ndarray) -> np.ndarray:
    """The unweighted lstsq solve (the original algorithm)."""
    rcond = y.size * np.finfo(y.dtype).eps
    scl = np.sqrt(np.square(van.T).sum(axis=0))
    return np.linalg.lstsq((van.T / scl), y.T, rcond)[0] / scl


def _ref_wls(van: np.ndarray, y: np.ndarray, w: np.ndarray) -> np.ndarray:
    """The weighted lstsq solve (the original algorithm)."""
    mask = w > 0
    sqrtw = np.sqrt(w[mask])
    lhs = van[:, mask] * sqrtw
    rhs = y[mask] * sqrtw
    rcond = y.size * np.finfo(y.dtype).eps
    scl = np.sqrt(np.square(lhs).sum(1))
    scl[scl == 0] = 1
    c, _, _, _ = np.linalg.lstsq((lhs.T / scl), rhs.T, rcond)
    return (c.T / scl).T


def _ref_gls(van: np.ndarray, y: np.ndarray, w: np.ndarray) -> np.ndarray:
    """The generalised lstsq solve with a weight matrix (the original algorithm)."""
    lt = np.linalg.cholesky(w).T
    return np.linalg.lstsq(lt @ van.T, lt @ y, rcond=None)[0]


def _ref_qr_full(basis: np.ndarray, y: np.ndarray, w) -> np.ndarray:
    """The QR solve, including a full weight matrix (the original algorithm)."""
    if np.ndim(w) == 2:
        sqrtw = np.linalg.cholesky(w).T
        q, r = sp.linalg.qr(np.dot(sqrtw, basis.T), mode="economic")
        return sp.linalg.solve(r, np.dot(q.T, np.dot(sqrtw, y)))
    return _ref_qr(basis, y, w)


def _ref_params(fm, y: np.ndarray, w, method: str) -> np.ndarray:
    """Best-fit parameters from the original ModelFit algorithms."""
    mask = np.isfinite(y)
    basis = fm.basis
    if method == "qr":
        wm = w if np.ndim(w) != 1 else w[mask]
        return _ref_qr_full(basis[:, mask], y[mask], wm)
    if np.isscalar(w):
        return _ref_ls(basis[:, mask], y[mask])
    if np.ndim(w) == 2:
        return _ref_gls(basis, y, w)
    return _ref_wls(basis[:, mask], y[mask], w[mask])


def _weights(kind: str, n: int, rng: np.random.Generator):
    if kind == "scalar":
        return 4.0
    sig = rng.uniform(0.5, 2, size=n)
    if kind == "1D":
        w = 1 / sig**2
        w[::17] = 0
        return w
    x = np.arange(n)
    cov = np.diag(sig**2) + 0.1 * np.exp(-(np.subtract.outer(x, x) ** 2) / 20)
    return np.linalg.inv(cov)


def _fit_data(n: int, nan: bool, rng: np.random.Generator):
    x = np.linspace(50, 100, n)
    fm = mdl.LinLog(n_terms=5).at(x=x)
    y = fm(parameters=[2000, 10, -10, 5, -5]) + rng.normal(size=n)
    if nan:
        y[5::31] = np.nan
    return fm, y


@pytest.mark.parametrize("method", ["lstsq", "qr"])
@pytest.mark.parametrize("wkind", ["scalar", "1D", "matrix"])
@pytest.mark.parametrize("nan", [False, True])
def test_modelfit_solve_is_bit_identical_to_original(method, wkind, nan):
    """ModelFit's lstsq/qr solves are exactly the original algorithms."""
    if nan and wkind == "matrix":
        pytest.skip("A weight matrix needs finite data.")
    rng = np.random.default_rng(3)
    fm, y = _fit_data(300, nan, rng)
    w = _weights(wkind, 300, rng)
    fit = fm.fit(ydata=y, weights=w, method=method)
    np.testing.assert_array_equal(
        np.array(fit.model_parameters), _ref_params(fm, y, w, method)
    )


@pytest.mark.parametrize("method", ["lstsq", "qr", "alan-qrd"])
@pytest.mark.parametrize("wkind", ["scalar", "1D", "matrix"])
@pytest.mark.parametrize("nan", [False, True])
def test_repeated_fit_is_bit_identical_to_modelfit(method, wkind, nan):
    """Cached repeated fits give exactly the parameters/residuals of full fits."""
    if wkind == "matrix" and (nan or method == "alan-qrd"):
        pytest.skip("Unsupported combination for a weight matrix.")
    rng = np.random.default_rng(4)
    fm, y = _fit_data(300, nan, rng)
    w = _weights(wkind, 300, rng)
    rep = _RepeatedFit(fm, weights=w, method=method)
    for _ in range(3):
        yy = y + rng.normal(scale=5, size=y.size)
        ref = fm.fit(ydata=yy, weights=w, method=method)
        ref_pars = np.array(ref.model_parameters)
        np.testing.assert_array_equal(rep.parameters(yy), ref_pars)
        np.testing.assert_array_equal(rep.residual(yy), ref.residual)
        fit = rep.fit(yy)
        np.testing.assert_array_equal(np.array(fit.model_parameters), ref_pars)
        np.testing.assert_array_equal(fit.residual, ref.residual)
        np.testing.assert_array_equal(fit.weighted_chi2, ref.weighted_chi2)


def test_repeated_fit_with_changed_mask():
    """Data with a different mask of finite values fall back to a full fit."""
    rng = np.random.default_rng(5)
    fm, y = _fit_data(200, False, rng)
    w = rng.uniform(0.5, 2, size=y.size)
    rep = _RepeatedFit(fm, weights=w)

    y_nan = y.copy()
    y_nan[::13] = np.nan
    for yy in (y, y_nan, y * 1.01, y_nan * 0.99):
        ref = fm.fit(ydata=yy, weights=w)
        np.testing.assert_array_equal(
            rep.parameters(yy), np.array(ref.model_parameters)
        )
        np.testing.assert_array_equal(rep.residual(yy), ref.residual)


def test_repeated_fit_validates_weights():
    """Invalid weights raise the same error as a full fit."""
    fm, y = _fit_data(50, False, np.random.default_rng(6))
    w = np.ones(50)
    w[3] = -1
    with pytest.raises(ValueError, match="non-negative"):
        _RepeatedFit(fm, weights=w).residual(y)
