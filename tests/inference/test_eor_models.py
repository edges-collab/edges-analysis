import warnings

import numpy as np
import pytest

from edges.inference import FlattenedGaussian, GaussianAbsorptionProfile
from edges.inference.eor_models import flattened_gaussian, simple_gaussian


class TestFlattenedGaussian:
    def test_flattened_gaussian(self):
        freqs = np.linspace(50, 100, 25)
        flatgauss = FlattenedGaussian(freqs=freqs)
        eor = flatgauss(params={"w": 1, "nu0": 75.0, "amp": 1.0})["eor_spectrum"]

        assert eor.shape == freqs.shape
        assert np.all(eor <= 0)
        assert np.abs(eor[0]) < 0.001
        assert np.abs(eor[-1]) < 0.001
        assert np.isclose(eor.min(), -1.0)

        neg_eor = flatgauss(params={"w": 1, "nu0": 75.0, "amp": -1.0})["eor_spectrum"]

        assert np.allclose(neg_eor, -eor)


class TestGaussian:
    def test_gaussian(self):
        freqs = np.linspace(50, 100, 25)
        gauss = GaussianAbsorptionProfile(freqs=freqs)
        eor = gauss(params={"w": 1, "nu0": 75.0, "amp": 1.0})["eor_spectrum"]

        assert eor.shape == freqs.shape
        assert np.all(eor <= 0)
        assert np.abs(eor[0]) < 0.001
        assert np.abs(eor[-1]) < 0.001
        assert np.isclose(eor.min(), -1.0)

        neg_eor = gauss(params={"w": 1, "nu0": 75.0, "amp": -1.0})["eor_spectrum"]

        assert np.allclose(neg_eor, -eor)

        assert eor.shape == freqs.shape


def _gaussian_fwhm(freqs, amp, w, nu0):
    return -amp * np.exp(-4 * np.log(2) * (freqs - nu0) ** 2 / w**2)


class TestFlattenedGaussianPhysics:
    """Analytic properties of the Bowman+18 flattened Gaussian."""

    @pytest.mark.parametrize("tau", [1e-3, 0.5, 7.0, 50.0, 500.0])
    def test_depth_at_centre(self, tau):
        assert np.isclose(
            flattened_gaussian(np.array([78.0]), amp=0.5, tau=tau, w=19.0, nu0=78.0),
            -0.5,
            rtol=1e-12,
        )

    @pytest.mark.parametrize("tau", [1e-3, 0.5, 7.0, 50.0, 500.0])
    def test_half_depth_at_half_width(self, tau):
        """``w`` is the full width at half maximum for every ``tau``."""
        f = np.array([78.0 - 19.0 / 2, 78.0 + 19.0 / 2])
        np.testing.assert_allclose(
            flattened_gaussian(f, amp=0.5, tau=tau, w=19.0, nu0=78.0),
            -0.25,
            rtol=1e-10,
        )

    @pytest.mark.parametrize("tau", [0.0, 1e-300, 1e-14, 1e-10, 1e-6])
    def test_gaussian_limit(self, tau):
        """As tau -> 0 the profile becomes a Gaussian of FWHM ``w`` (and is finite)."""
        f = np.linspace(40, 120, 81)
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            fg = flattened_gaussian(f, amp=0.5, tau=tau, w=19.0, nu0=78.0)
        assert np.all(np.isfinite(fg))
        np.testing.assert_allclose(
            fg, _gaussian_fwhm(f, 0.5, 19.0, 78.0), rtol=0, atol=1e-6
        )

    def test_tau_zero_via_component(self):
        """The prior allows tau = 0, so the component must evaluate there."""
        freqs = np.linspace(50, 100, 51)
        eor = FlattenedGaussian(freqs=freqs)(
            params={"tau": 0.0, "w": 19.0, "nu0": 78.0, "amp": 0.5}
        )["eor_spectrum"]
        np.testing.assert_allclose(
            eor, _gaussian_fwhm(freqs, 0.5, 19.0, 78.0), rtol=0, atol=1e-12
        )

    def test_flattening_increases_with_tau(self):
        """Larger tau gives a flatter-bottomed profile (deeper at fixed offset)."""
        f = np.array([78.0 + 19.0 / 4])
        vals = [
            flattened_gaussian(f, amp=1.0, tau=t, w=19.0, nu0=78.0)[0]
            for t in (0.0, 1.0, 7.0, 30.0)
        ]
        assert np.all(np.diff(vals) < 0)


def test_simple_gaussian_width_is_sigma():
    """``simple_gaussian`` uses ``w`` as the standard deviation (not the FWHM)."""
    f = np.array([70.0, 80.0, 90.0])
    np.testing.assert_allclose(
        simple_gaussian(f, amp=0.5, nu0=80.0, w=10.0),
        [-0.5 * np.exp(-0.5), -0.5, -0.5 * np.exp(-0.5)],
        rtol=1e-14,
    )
