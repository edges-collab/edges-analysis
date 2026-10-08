import astropy.units as un
import numpy as np
import pytest

from edges.cal import sparams as sp
from edges.cal.sparams.core.sparam_calibration import (
    sparams_from_calkit_measurements,
)
from edges.cal.sparams.devices.internal_switch import get_internal_switch_sparams


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


def _error_network(freqs, rng, delay_ns: float = 2.0, reciprocal: bool = True):
    """A smooth, random, passive error network (e.g. a VNA's port error terms)."""
    n = len(freqs)
    x = freqs.to_value("MHz") / 100
    e00 = (0.05 + 0.02j) * rng.normal() + 0.01 * x + 0.03j * x**2
    e11 = (0.1 - 0.05j) * rng.normal() - 0.02j * x
    e10e01 = rng.uniform(0.7, 0.95) * np.exp(
        -2j * np.pi * (freqs * delay_ns * un.ns).to_value("")
    )
    s12 = np.sqrt(e10e01)
    s21 = e10e01 / s12
    if not reciprocal:
        s12 = s12 * 1.3
        s21 = s21 / 1.3
    return sp.SParams(
        freqs=freqs,
        s11=e00 * np.ones(n),
        s12=s12,
        s21=s21,
        s22=e11 * np.ones(n),
    )


def _measure(kit: sp.CalkitReadings, *networks: sp.SParams) -> sp.CalkitReadings:
    """Embed each standard of the kit through the networks (last is nearest VNA)."""
    out = {}
    for kind in ("open", "short", "match"):
        g = getattr(kit, kind)
        for net in networks:
            g = g.embed(net)
        out[kind] = g
    return sp.CalkitReadings(**out)


class TestOSLCalibration:
    freqs = np.linspace(40, 200, 161) * un.MHz

    @pytest.mark.parametrize("seed", [0, 1, 2])
    @pytest.mark.parametrize("reciprocal", [True, False])
    def test_round_trip_nonideal_kit(self, seed, reciprocal):
        """Synthesised raw OSL readings give back the error terms and the DUT."""
        rng = np.random.default_rng(seed)
        err = _error_network(self.freqs, rng, reciprocal=reciprocal)
        kit = sp.AGILENT_85033E
        meas = _measure(kit.at_freqs(self.freqs), err)

        for model in (kit, kit.at_freqs(self.freqs)):
            est = sparams_from_calkit_measurements(meas, model)
            np.testing.assert_allclose(est.s11, err.s11, atol=1e-12)
            np.testing.assert_allclose(est.s22, err.s22, atol=1e-12)
            np.testing.assert_allclose(est.s12 * est.s21, err.s12 * err.s21, atol=1e-12)

            dut = sp.ReflectionCoefficient(
                freqs=self.freqs,
                reflection_coefficient=0.4 * np.exp(1j * np.linspace(0, 7, 161)),
            )
            rec = dut.embed(err).de_embed(est)
            np.testing.assert_allclose(
                rec.reflection_coefficient, dut.reflection_coefficient, atol=1e-11
            )

    def test_ideal_standards_and_readings_give_identity(self):
        ideal = sp.CalkitReadings.ideal(self.freqs)
        est = sparams_from_calkit_measurements(ideal)
        np.testing.assert_allclose(est.s11, 0, atol=1e-14)
        np.testing.assert_allclose(est.s22, 0, atol=1e-14)
        np.testing.assert_allclose(est.s12 * est.s21, 1, rtol=1e-14)

        # Same with the ideal model given explicitly.
        est = sparams_from_calkit_measurements(ideal, model=ideal)
        np.testing.assert_allclose(est.s12 * est.s21, 1, rtol=1e-14)

    def test_calkit_readings_de_embed(self):
        rng = np.random.default_rng(10)
        err = _error_network(self.freqs, rng)
        kit = sp.AGILENT_85033E.at_freqs(self.freqs)
        meas = _measure(kit, err)
        de = meas.de_embed(err)
        for kind in ("open", "short", "match"):
            np.testing.assert_allclose(
                getattr(de, kind).reflection_coefficient,
                getattr(kit, kind).reflection_coefficient,
                atol=1e-12,
            )


class TestInternalSwitchAveraging:
    freqs = np.arange(40, 200.01, 0.25) * un.MHz

    @pytest.mark.xfail(
        strict=True,
        reason=(
            "S12 and S21 are taken as the principal-branch sqrt of S12*S21, which "
            "flips sign where the phase of S12*S21 crosses pi; averaging repeats whose "
            "flips fall at different frequencies cancels S12/S21 there; "
            "fix pending (result-changing)"
        ),
    )
    def test_repeat_averaging_across_branch_cut(self):
        """Averaged repeats of a long (5 ns) SP2T keep S12*S21 near the truth."""
        rng = np.random.default_rng(3)
        sp4t = _error_network(self.freqs, rng, delay_ns=1.0)
        kit = sp.AGILENT_85033E.at_freqs(self.freqs)
        internal = _measure(sp.CalkitReadings.ideal(self.freqs), sp4t)

        products = []
        externals = []
        for delay in (5.0, 5.05):
            sp2t = _error_network(self.freqs, np.random.default_rng(4), delay_ns=delay)
            products.append(sp2t.s12 * sp2t.s21)
            externals.append(_measure(kit, sp2t, sp4t))

        out = get_internal_switch_sparams(
            internal_osl=[internal, internal],
            external_osl=externals,
            external_calkit=sp.AGILENT_85033E,
        )
        np.testing.assert_allclose(
            out.s12 * out.s21, np.mean(products, axis=0), rtol=1e-3
        )
