import numpy as np

from edges.inference import FlattenedGaussian, GaussianAbsorptionProfile


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
