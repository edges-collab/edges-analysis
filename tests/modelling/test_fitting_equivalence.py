"""Equivalence of the optimised fitting/basis code paths with straightforward ones.

Each test compares the library implementation with a simple reference
implementation (the original algorithm) on realistic random inputs.
"""

import numpy as np
import pytest
import scipy as sp

from edges import modeling as mdl
from edges.modeling.fitting import _c_qrd


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
