import numpy as np
import pytest
from scipy import stats

from edges.inference import FlattenedGaussian, SemiLinearFit
from edges.modeling import LinLog


def test_semi_linear_fit():
    freqs = np.linspace(50, 100, 100)

    fg = LinLog(parameters=[1750, 1, -1, 2, -2]).at(x=freqs)
    eor = FlattenedGaussian(freqs=freqs, params=["amp", "w"])

    fgx = fg()
    eorx = eor()["eor_spectrum"]
    data = fgx + eorx

    slf = SemiLinearFit(fg=fg, eor=eor, spectrum=data, sigma=0.1)

    best_p = slf()
    print(best_p)
    assert np.allclose(slf.get_eor(best_p.x), eorx)
    assert np.allclose(slf.fg_fit(best_p.x).evaluate(), fgx)
    assert np.allclose(slf.get_resid(best_p.x), 0)


def _setup(nfreq: int = 60, amp: float = 0.3):
    freqs = np.linspace(50, 100, nfreq)
    fg = LinLog(parameters=[1750, 1, -1]).at(x=freqs)
    eor = FlattenedGaussian(freqs=freqs, params=["amp"])
    data = fg() + eor(params=[amp])["eor_spectrum"]
    return freqs, fg, eor, data


def _correlated_cov(freqs: np.ndarray, sigma: np.ndarray, corr_len: float = 3.0):
    """A positive-definite covariance with exponentially decaying correlations."""
    dist = np.abs(freqs[:, None] - freqs[None, :])
    return np.outer(sigma, sigma) * np.exp(-dist / corr_len)


def test_diagonal_covariance_matches_uniform_sigma():
    """sigma**2 * I gives exactly the same fit and likelihood as a scalar sigma."""
    freqs, fg, eor, data = _setup()
    rng = np.random.default_rng(0)
    noisy = data + rng.normal(scale=0.1, size=freqs.size)

    scalar = SemiLinearFit(fg=fg, eor=eor, spectrum=noisy, sigma=0.1)
    cov = SemiLinearFit(fg=fg, eor=eor, spectrum=noisy, sigma=0.01 * np.eye(freqs.size))
    for p in ([0.0], [0.2], [0.5]):
        np.testing.assert_allclose(cov.fg_params(p), scalar.fg_params(p), rtol=1e-8)
        np.testing.assert_allclose(cov.neg_lk(p), scalar.neg_lk(p), rtol=1e-10)


def test_covariance_fg_fit_is_generalised_least_squares():
    """The FG parameters are (A^T C^-1 A)^-1 A^T C^-1 r, for a correlated C."""
    freqs, fg, eor, data = _setup()
    sigma = np.linspace(0.05, 0.2, freqs.size)
    cov = _correlated_cov(freqs, sigma)
    rng = np.random.default_rng(1)
    noisy = data + rng.multivariate_normal(np.zeros(freqs.size), cov)

    slf = SemiLinearFit(fg=fg, eor=eor, spectrum=noisy, sigma=cov)
    p = [0.25]
    r = noisy - slf.get_eor(p)
    a = fg.basis.T
    cinv = np.linalg.inv(cov)
    expected = np.linalg.solve(a.T @ cinv @ a, a.T @ cinv @ r)
    np.testing.assert_allclose(slf.fg_params(p), expected, rtol=1e-7)

    # The residual is C^-1-orthogonal to the FG basis.
    resid = slf.get_resid(p)
    np.testing.assert_allclose(a.T @ cinv @ resid, 0, atol=1e-6)
    np.testing.assert_allclose(slf.fg_fit(p).evaluate(), r - resid, rtol=1e-12)

    # The likelihood is the multivariate normal of that residual.
    expected_lk = stats.multivariate_normal(np.zeros(freqs.size), cov).logpdf(resid)
    np.testing.assert_allclose(slf.neg_lk(p), -expected_lk, rtol=1e-10)


def test_covariance_fit_recovers_signal():
    """Noise-free data with a correlated covariance: the signal is recovered."""
    freqs, fg, eor, data = _setup(amp=0.3)
    cov = _correlated_cov(freqs, np.linspace(0.05, 0.2, freqs.size))
    best = SemiLinearFit(fg=fg, eor=eor, spectrum=data, sigma=cov)()
    np.testing.assert_allclose(best.x, [0.3], atol=1e-4)


def test_diagonal_covariance_matches_array_sigma():
    """Non-uniform per-channel sigma and diag(sigma**2) give the same FG fit."""
    freqs, fg, eor, data = _setup()
    sigma = np.linspace(0.05, 0.5, freqs.size)
    arr = SemiLinearFit(fg=fg, eor=eor, spectrum=data, sigma=sigma)
    cov = SemiLinearFit(fg=fg, eor=eor, spectrum=data, sigma=np.diag(sigma**2))
    np.testing.assert_allclose(arr.fg_params([0.1]), cov.fg_params([0.1]), rtol=1e-8)


@pytest.mark.parametrize(
    ("sigma", "match"),
    [
        (np.eye(5), "shape"),
        (np.ones((60, 59)), "shape"),
        (np.triu(np.ones((60, 60))) + np.eye(60), "symmetric"),
        (-np.eye(60), "positive definite"),
        (np.ones((2, 2, 2)), "1D array, a float, or a 2D covariance"),
    ],
)
def test_bad_covariance_sigma(sigma, match):
    _, fg, eor, data = _setup()
    with pytest.raises(ValueError, match=match):
        SemiLinearFit(fg=fg, eor=eor, spectrum=data, sigma=sigma)


def test_semi_linear_fit_array_sigma():
    """A per-channel sigma array is supported (noise-free recovery)."""
    freqs = np.linspace(50, 100, 100)
    fg = LinLog(parameters=[1750, 1, -1]).at(x=freqs)
    eor = FlattenedGaussian(freqs=freqs, params=["amp"])
    data = fg() + eor(params=[0.3])["eor_spectrum"]
    sigma = np.linspace(0.05, 0.2, freqs.size)

    best = SemiLinearFit(fg=fg, eor=eor, spectrum=data, sigma=sigma)()
    np.testing.assert_allclose(best.x, [0.3], atol=1e-4)


@pytest.mark.parametrize("sigma_kind", ["scalar", "array", "cov"])
@pytest.mark.parametrize("fit_method", ["lstsq", "qr"])
def test_fit_method_is_used_for_the_fg_fit(sigma_kind, fit_method):
    """The FG fit uses the requested solver (exactly as a direct fit with it)."""
    freqs, fg, eor, data = _setup()
    sigmas = np.linspace(0.05, 0.2, freqs.size)
    sigma, weights = {
        "scalar": (0.1, 1.0),
        "array": (sigmas, 1 / sigmas**2),
        "cov": (np.diag(sigmas**2), np.diag(1 / sigmas**2)),
    }[sigma_kind]
    slf = SemiLinearFit(
        fg=fg, eor=eor, spectrum=data, sigma=sigma, fit_method=fit_method
    )
    resid = data - slf.get_eor([0.2])
    if sigma_kind == "cov":
        weights = slf._cov_inverse
    ref = fg.fit(ydata=resid, weights=weights, method=fit_method)
    np.testing.assert_array_equal(
        np.array(slf.fg_params([0.2])), np.array(ref.model_parameters)
    )


@pytest.mark.parametrize(("sigma", "default"), [(0.1, "lstsq"), ("cov", "qr")])
def test_default_fit_method_unchanged(sigma, default):
    freqs, fg, eor, data = _setup()
    if sigma == "cov":
        sigma = 0.01 * np.eye(freqs.size)
    default_fit = SemiLinearFit(fg=fg, eor=eor, spectrum=data, sigma=sigma)
    explicit = SemiLinearFit(
        fg=fg, eor=eor, spectrum=data, sigma=sigma, fit_method=default
    )
    np.testing.assert_array_equal(
        default_fit.fg_params([0.2]), explicit.fg_params([0.2])
    )


def test_unknown_fit_method():
    _, fg, eor, data = _setup()
    with pytest.raises(ValueError, match="Unknown fit_method"):
        SemiLinearFit(fg=fg, eor=eor, spectrum=data, sigma=0.1, fit_method="svd")
