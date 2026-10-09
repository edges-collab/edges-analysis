from pathlib import Path

import numpy as np
import pytest
from astropy import units as un

from edges import modeling as mdl
from edges.cal import CalibrationObservation, Calibrator, ReflectionCoefficient
from edges.cal import noise_waves as nw
from edges.cal.sparams import S11ModelParams

NFREQ = 101
FREQS = np.linspace(50, 100, NFREQ) * un.MHz


def _synthetic_calibrator() -> Calibrator:
    """A calibrator with smooth, non-trivial noise-wave parameters."""
    f = FREQS.to_value("MHz") / 75.0
    return Calibrator(
        freqs=FREQS,
        Tsca=1500 + 100 * f,
        Toff=300 - 10 * f**2,
        Tunc=20 + 5 * f,
        Tcos=10 - 3 * f,
        Tsin=-15 + 4 * f**2,
        receiver_s11=0.05 * np.exp(1j * 2 * np.pi * FREQS.to_value("MHz") / 40),
    )


def _ant_s11() -> np.ndarray:
    """A non-zero, frequency-dependent antenna reflection coefficient."""
    fq = FREQS.to_value("MHz")
    return 0.3 * (fq / 75) ** -0.5 * np.exp(-1j * fq / 7)


def test_write_read_roundtrip(identity_calibrator: Calibrator, tmpdir: Path):
    identity_calibrator.write(tmpdir / "tmp.h5")
    new = Calibrator.from_calfile(tmpdir / "tmp.h5")
    assert new == identity_calibrator


def test_cal_uncal_round_trip(calobs: CalibrationObservation, calibrator: Calibrator):
    ant_s11 = calobs.open.reflection_coefficient.s11
    a, b = calibrator.get_linear_coefficients(freqs=calobs.freqs, ant_s11=ant_s11)
    q = calobs.open.averaged_q

    # The linear coefficients must be those of Monsalve+17 Eq. 7.
    a_expected, b_expected = nw.get_linear_coefficients(
        gamma_ant=ant_s11,
        gamma_rec=calibrator.receiver_s11,
        t_sca=calibrator.Tsca,
        t_off=calibrator.Toff,
        t_unc=calibrator.Tunc,
        t_cos=calibrator.Tcos,
        t_sin=calibrator.Tsin,
    )
    np.testing.assert_allclose(a.to_value("K"), a_expected, rtol=1e-12)
    np.testing.assert_allclose(b.to_value("K"), b_expected, rtol=1e-12)

    tcal = calibrator.calibrate_load(calobs.open)
    mask = np.isfinite(q)
    np.testing.assert_allclose(
        tcal[mask].to_value("K"), (q * a + b)[mask].to_value("K"), rtol=1e-12
    )

    decal = calibrator.decalibrate(
        tcal,
        ant_s11=calobs.open.reflection_coefficient.s11,
    )
    np.testing.assert_allclose(decal, calobs.open.averaged_q, atol=3e-5)

    new_tcal = calibrator.calibrate_q(
        decal,
        ant_s11=calobs.open.reflection_coefficient.s11,
    )

    np.testing.assert_allclose(new_tcal, tcal, atol=1e-6)


class TestCalibratorPhysics:
    """Physical limits and invariants of the noise-wave calibration."""

    def test_calibrate_decalibrate_identity(self):
        cal = _synthetic_calibrator()
        ant_s11 = _ant_s11()
        temp = np.linspace(200, 5000, NFREQ) * un.K

        q = cal.decalibrate(temp, ant_s11=ant_s11)
        assert not np.allclose(q, q[0])
        back = cal.calibrate_q(q, ant_s11=ant_s11)

        np.testing.assert_allclose(back.to_value("K"), temp.to_value("K"), atol=1e-9)

    def test_q_zero_gives_offset(self):
        cal = _synthetic_calibrator()
        ant_s11 = _ant_s11()
        _, b = cal.get_linear_coefficients(ant_s11=ant_s11)
        t0 = cal.calibrate_q(np.zeros(NFREQ), ant_s11=ant_s11)
        np.testing.assert_allclose(t0.to_value("K"), b.to_value("K"), rtol=0, atol=0)

    def test_linearity(self):
        """Doubling the temperature above the offset doubles Q (T - b = a*Q)."""
        cal = _synthetic_calibrator()
        ant_s11 = _ant_s11()
        _, b = cal.get_linear_coefficients(ant_s11=ant_s11)

        temp = np.linspace(500, 3000, NFREQ) * un.K
        q1 = cal.decalibrate(temp, ant_s11=ant_s11)
        q2 = cal.decalibrate(b + 2 * (temp - b), ant_s11=ant_s11)
        np.testing.assert_allclose(q2, 2 * q1, rtol=1e-12)

        t1 = cal.calibrate_q(q1, ant_s11=ant_s11)
        t2 = cal.calibrate_q(2 * q1, ant_s11=ant_s11)
        np.testing.assert_allclose(
            (t2 - b).to_value("K"), 2 * (t1 - b).to_value("K"), rtol=1e-12
        )

    def test_matched_antenna_and_receiver(self):
        """With Γ_ant = Γ_rec = 0, T = Tsca*Q + Toff - Tunc*0 (no noise waves)."""
        cal = _synthetic_calibrator().clone(receiver_s11=np.zeros(NFREQ, dtype=complex))
        q = np.linspace(0.1, 2, NFREQ)
        t = cal.calibrate_q(q, ant_s11=np.zeros(NFREQ, dtype=complex))
        np.testing.assert_allclose(t.to_value("K"), cal.Tsca * q + cal.Toff, rtol=1e-12)


class TestCalibratorInputs:
    def test_ndarray_freqs_coerced_to_mhz(self):
        cal = _synthetic_calibrator()
        cal2 = Calibrator(
            freqs=FREQS.to_value("MHz"),
            Tsca=cal.Tsca,
            Toff=cal.Toff,
            Tunc=cal.Tunc,
            Tcos=cal.Tcos,
            Tsin=cal.Tsin,
            receiver_s11=cal.receiver_s11,
        )
        assert isinstance(cal2.freqs, un.Quantity)
        assert cal2.freqs.unit == un.MHz
        assert cal2 == cal

    def test_quantity_freqs_converted_to_mhz(self):
        cal = _synthetic_calibrator().clone(freqs=FREQS.to("GHz"))
        assert cal.freqs.unit == un.MHz
        np.testing.assert_allclose(cal.freqs.value, FREQS.value, rtol=1e-12)

    @pytest.mark.parametrize(
        "name", ["Tsca", "Toff", "Tunc", "Tcos", "Tsin", "receiver_s11"]
    )
    def test_mismatched_lengths(self, name):
        cal = _synthetic_calibrator()
        with pytest.raises(ValueError, match=f"{name} must have the same length"):
            cal.clone(**{name: getattr(cal, name)[:-1]})

    def test_get_modelled_bad_thing(self):
        cal = _synthetic_calibrator()
        with pytest.raises(ValueError, match="thing must be one of"):
            cal.get_modelled("unit", FREQS)

    def test_get_modelled_receiver_s11(self):
        cal = _synthetic_calibrator()
        out = cal.get_modelled("receiver_s11", FREQS[::2])
        np.testing.assert_allclose(out, cal.receiver_s11[::2], atol=1e-8)

    def test_get_modelled_with_model(self):
        cal = _synthetic_calibrator()
        model = mdl.Polynomial(n_terms=3, transform=mdl.ScaleTransform(scale=75.0))
        out = cal.get_modelled("Toff", FREQS[::3], model=model)
        np.testing.assert_allclose(out, cal.Toff[::3], rtol=1e-10)

    def test_get_modelled_with_composite_model(self):
        """CompositeModels are fit and then evaluated at the new frequencies."""
        cal = _synthetic_calibrator()
        tr = mdl.ScaleTransform(scale=75.0)
        model = mdl.CompositeModel(
            models={
                "a": mdl.Polynomial(n_terms=2, transform=tr),
                "b": mdl.Polynomial(n_terms=3, offset=2, transform=tr),
            }
        )
        out = cal.get_modelled("Tsin", FREQS[::3], model=model)
        np.testing.assert_allclose(out, cal.Tsin[::3], rtol=1e-10)

    def test_s11_model_params_used_for_off_grid_s11(self):
        """s11_model_params controls how an off-grid antenna S11 is modelled."""
        cal = _synthetic_calibrator()
        fine = np.linspace(48, 102, 271) * un.MHz
        fq = fine.to_value("MHz")
        rc = ReflectionCoefficient(
            freqs=fine,
            reflection_coefficient=0.3 * (fq / 75) ** -0.5 * np.exp(-1j * fq / 7),
        )
        params = S11ModelParams(
            model=mdl.Fourier(n_terms=7, transform=mdl.UnitTransform(range=(0, 1)))
        )
        q = np.linspace(0.1, 0.9, NFREQ)

        on_grid = rc.smoothed(params=params, freqs=FREQS).s11
        expected = cal.calibrate_q(q, ant_s11=on_grid, freqs=FREQS)
        got = cal.calibrate_q(q, ant_s11=rc, freqs=FREQS, s11_model_params=params)
        np.testing.assert_allclose(
            got.to_value("K"), expected.to_value("K"), rtol=1e-12, atol=0
        )

        # The parameters actually matter: the default model gives a different answer.
        default = cal.calibrate_q(q, ant_s11=rc, freqs=FREQS)
        assert not np.allclose(default.to_value("K"), got.to_value("K"), rtol=1e-6)

        # decalibrate uses the same S11 model, so it inverts calibrate_q.
        back = cal.decalibrate(got, ant_s11=rc, freqs=FREQS, s11_model_params=params)
        np.testing.assert_allclose(back, q, rtol=1e-12, atol=0)

    def test_reflection_coefficient_on_same_grid(self):
        """An ant_s11 ReflectionCoefficient on the calibrator's grid is used as-is."""
        cal = _synthetic_calibrator()
        ant_s11 = _ant_s11()
        rc = ReflectionCoefficient(freqs=FREQS, reflection_coefficient=ant_s11)

        a, b = cal.get_linear_coefficients(ant_s11=rc)
        a0, b0 = cal.get_linear_coefficients(ant_s11=ant_s11)
        np.testing.assert_allclose(a.to_value("K"), a0.to_value("K"), rtol=0, atol=0)
        np.testing.assert_allclose(b.to_value("K"), b0.to_value("K"), rtol=0, atol=0)
