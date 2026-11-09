"""Tests of the tensor-product models."""

import numpy as np
import pytest

from edges import modeling as mdl


@pytest.fixture(scope="module")
def setup():
    freqs = np.linspace(50, 100, 300)
    lsts = np.arange(10) * 0.5
    use = np.ones(lsts.size, dtype=bool)
    fourier = mdl.Fourier(
        n_terms=7, period=1.5, transform=mdl.ZerotooneTransform(range=(50, 100))
    )
    freq_basis = np.asarray(fourier.at(x=freqs).basis)
    lst_basis, _ = mdl.polynomial_basis(lsts, use, 3)
    rng = np.random.default_rng(0)
    coeffs = rng.normal(size=(3, 7))
    data = lst_basis.T @ (coeffs @ freq_basis)
    return freq_basis, lst_basis, data


def test_polynomial_basis_maps_used_range_to_unit_interval():
    coords = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    use = np.array([False, True, True, True, False])
    basis, n = mdl.polynomial_basis(coords, use, 3)
    assert n == 3
    np.testing.assert_allclose(basis[1], [-0.5, 0.0, 0.5, 1.0, 1.5])
    np.testing.assert_allclose(basis[2], basis[1] ** 2)


def test_polynomial_basis_caps_terms_at_used_rows():
    _, n = mdl.polynomial_basis(np.arange(5.0), np.array([1, 1, 0, 0, 0], bool), 4)
    assert n == 2


@pytest.mark.parametrize("method", ["lstsq", "alan-qrd"])
def test_fit_tensor_product_recovers_exact_model(setup, method):
    freq_basis, lst_basis, data = setup
    weights = np.ones_like(data)
    model = mdl.fit_tensor_product(freq_basis, lst_basis, data, weights, method=method)
    np.testing.assert_allclose(model, data, atol=1e-8)


def test_fit_tensor_product_ignores_zero_weight_samples(setup):
    freq_basis, lst_basis, data = setup
    corrupted = data.copy()
    weights = np.ones_like(data)
    corrupted[2, 100:120] = np.nan
    corrupted[5] = 1e6
    weights[2, 100:120] = 0
    weights[5] = 0
    model = mdl.fit_tensor_product(freq_basis, lst_basis, corrupted, weights)
    np.testing.assert_allclose(model, data, atol=1e-8)
