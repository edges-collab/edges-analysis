import numpy as np
import pytest

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


def test_semi_linear_fit_rejects_covariance_sigma():
    """MOD-10: a 2D (covariance) sigma is not supported and says so clearly."""
    freqs = np.linspace(50, 100, 20)
    fg = LinLog(n_terms=3).at(x=freqs)
    eor = FlattenedGaussian(freqs=freqs, params=["amp"])
    with pytest.raises(NotImplementedError, match="covariance"):
        SemiLinearFit(fg=fg, eor=eor, spectrum=np.ones(20), sigma=np.eye(20) * 0.01)


def test_semi_linear_fit_array_sigma():
    """A per-channel sigma array is supported (noise-free recovery)."""
    freqs = np.linspace(50, 100, 100)
    fg = LinLog(parameters=[1750, 1, -1]).at(x=freqs)
    eor = FlattenedGaussian(freqs=freqs, params=["amp"])
    data = fg() + eor(params=[0.3])["eor_spectrum"]
    sigma = np.linspace(0.05, 0.2, freqs.size)

    best = SemiLinearFit(fg=fg, eor=eor, spectrum=data, sigma=sigma)()
    np.testing.assert_allclose(best.x, [0.3], atol=1e-4)
