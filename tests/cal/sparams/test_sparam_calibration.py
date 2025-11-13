import astropy.units as un
import numpy as np

from edges.cal import sparams as sp


class TestGammaEmbed:
    def test_gamma_shift_zero(self):
        rng = np.random.default_rng()
        freqs = np.linspace(50, 100, 100) * un.MHz
        s11 = sp.ReflectionCoefficient(
            reflection_coefficient=rng.normal(size=100), freqs=freqs
        )

        smatrix = sp.SParams(s11=np.zeros(100), s12=np.ones(100), freqs=freqs)
        np.testing.assert_allclose(
            s11.reflection_coefficient,
            sp.gamma_embed(s11, smatrix).reflection_coefficient,
        )

    def test_gamma_impedance_roundtrip(self):
        z0 = 50
        rng = np.random.default_rng()
        z = rng.normal(size=10)

        np.testing.assert_allclose(sp.gamma2impedance(sp.impedance2gamma(z, z0), z0), z)

    def test_gamma_embed_rountrip(self):
        rng = np.random.default_rng()
        s11 = rng.uniform(0, 1, size=10) + rng.uniform(0, 1, size=10) * 1j
        s12 = rng.uniform(0, 1, size=10) + rng.uniform(0, 1, size=10) * 1j
        s22 = rng.uniform(0, 1, size=10) + rng.uniform(0, 1, size=10) * 1j
        freqs = np.linspace(50, 100, 10) * un.MHz

        gamma = sp.ReflectionCoefficient(
            freqs=freqs,
            reflection_coefficient=rng.uniform(0, 1, size=10)
            + rng.uniform(0, 1, size=10) * 1j,
        )
        smat = sp.SParams(s11=s11, s12=s12, s22=s22, freqs=freqs)

        np.testing.assert_allclose(
            sp.gamma_de_embed(sp.gamma_embed(gamma, smat), smat).reflection_coefficient,
            gamma.reflection_coefficient,
        )


class TestAverageSparams:
    def test_average_sparams(self):
        rng = np.random.default_rng()
        s11_1 = rng.uniform(0, 1, size=10) + rng.uniform(0, 1, size=10) * 1j
        s12_1 = rng.uniform(0, 1, size=10) + rng.uniform(0, 1, size=10) * 1j
        s21_1 = rng.uniform(0, 1, size=10) + rng.uniform(0, 1, size=10) * 1j
        s22_1 = rng.uniform(0, 1, size=10) + rng.uniform(0, 1, size=10) * 1j

        s11_2 = rng.uniform(0, 1, size=10) + rng.uniform(0, 1, size=10) * 1j
        s12_2 = rng.uniform(0, 1, size=10) + rng.uniform(0, 1, size=10) * 1j
        s21_2 = rng.uniform(0, 1, size=10) + rng.uniform(0, 1, size=10) * 1j
        s22_2 = rng.uniform(0, 1, size=10) + rng.uniform(0, 1, size=10) * 1j

        freqs = np.linspace(50, 100, 10) * un.MHz

        sp1 = sp.SParams(s11=s11_1, s12=s12_1, s21=s21_1, s22=s22_1, freqs=freqs)
        sp2 = sp.SParams(s11=s11_2, s12=s12_2, s21=s21_2, s22=s22_2, freqs=freqs)

        sp_avg = sp.average_sparams([sp1, sp2])

        np.testing.assert_allclose(
            sp_avg.s11,
            0.5 * (s11_1 + s11_2),
        )
        np.testing.assert_allclose(
            sp_avg.s12,
            0.5 * (s12_1 + s12_2),
        )
        np.testing.assert_allclose(
            sp_avg.s21,
            0.5 * (s21_1 + s21_2),
        )
        np.testing.assert_allclose(
            sp_avg.s22,
            0.5 * (s22_1 + s22_2),
        )
