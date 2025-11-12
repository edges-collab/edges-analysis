from types import SimpleNamespace

import numpy as np
import pytest
from astropy import units as un

from edges.cal.sparams.core import datatypes as dt


def freqs_mhz(n=3):
    return np.arange(1, n + 1) * un.MHz


class TestSParams:
    def test_basic_and_matrix_roundtrip(self):
        freqs = freqs_mhz(4)
        s11 = np.array([0.1, 0.2, 0.0, 0.5], dtype=complex)
        s12 = np.array([0.0, 0.0, 0.0, 0.0], dtype=complex)
        s21 = np.array([1.0, 0.5, 0.2, 1.5], dtype=complex)
        s22 = np.array([0.0, 0.0, 0.0, 0.0], dtype=complex)

        sp = dt.SParams(freqs=freqs, s11=s11, s12=s12, s21=s21, s22=s22)

        assert sp.nfreqs == len(freqs)
        assert sp.s.shape == (2, 2, sp.nfreqs)
        np.testing.assert_allclose(sp.determinant, s11 * s22 - s12 * s21)

        t = sp.as_transfer_matrix()
        sp2 = dt.SParams.from_transfer_matrix(freqs=freqs, t=t)
        np.testing.assert_allclose(sp.s, sp2.s)

        np.testing.assert_allclose(sp.complex_linear_gain, s21)
        np.testing.assert_allclose(sp.scalar_linear_gain, np.abs(s21))
        expected_db = 20.0 * np.log10(np.abs(s21))
        np.testing.assert_allclose(sp.scalar_logarithmic_gain, expected_db)

    def test_cascade_and_reciprocal_lossless(self):
        freqs = freqs_mhz(2)
        s11 = np.ones(2, dtype=complex)
        s12 = np.zeros(2, dtype=complex)
        s21 = np.zeros(2, dtype=complex)
        s22 = np.ones(2, dtype=complex)

        sp = dt.SParams(freqs=freqs, s11=s11, s12=s12, s21=s21, s22=s22)
        assert sp.is_reciprocal()
        assert sp.is_lossless()

        # cascading this degenerate matrix triggers divide-by-zero / invalid-value
        # runtime warnings from numpy; assert they are raised
        with pytest.warns(RuntimeWarning):
            casc = sp.cascade_with(sp)
        assert isinstance(casc, dt.SParams)

    def test_return_losses_and_vswr(self):
        freqs = freqs_mhz(1)
        s11 = np.array([0.5 + 0j])
        s12 = np.array([0.0 + 0j])
        s21 = np.array([1.0 + 0j])
        s22 = np.array([0.0 + 0j])

        sp = dt.SParams(freqs=freqs, s11=s11, s12=s12, s21=s21, s22=s22)

        expected_input_rl = -20.0 * np.log10(np.abs(s11))
        np.testing.assert_allclose(sp.input_return_loss, expected_input_rl)

        assert pytest.approx(sp.voltage_standing_wave_ratio_in()[0]) == 3.0

    def test_gain_and_loss_properties(self):
        freqs = freqs_mhz(1)
        s21 = np.array([0.5 + 0.5j])
        s = dt.SParams(
            freqs=freqs, s11=np.zeros(1), s12=np.zeros(1), s21=s21, s22=np.zeros(1)
        )
        np.testing.assert_allclose(s.complex_linear_gain, s21)
        np.testing.assert_allclose(s.scalar_linear_gain, np.abs(s21))
        np.testing.assert_allclose(
            s.scalar_logarithmic_gain, 20.0 * np.log10(np.abs(s21))
        )

    @pytest.mark.parametrize(
        ("s21_vals", "expected_db"),
        [
            (np.array([1.0]), np.array([0.0])),
            (np.array([0.5]), 20.0 * np.log10(np.array([0.5]))),
            (np.array([2.0]), 20.0 * np.log10(np.array([2.0]))),
        ],
    )
    def test_scalar_logarithmic_gain_param(self, s21_vals, expected_db):
        freqs = freqs_mhz(1)
        s21 = s21_vals.astype(complex)
        s = dt.SParams(
            freqs=freqs, s11=np.zeros(1), s12=np.zeros(1), s21=s21, s22=np.zeros(1)
        )
        np.testing.assert_allclose(s.scalar_logarithmic_gain, expected_db)


class TestReflectionCoefficient:
    def test_select_rephase_and_validators(self, tmp_path):
        freqs = np.array([1.0, 2.0, 3.0]) * un.MHz
        gamma = np.array([1 + 0j, 0 + 1j, -1 + 0j])
        rc = dt.ReflectionCoefficient(freqs=freqs, reflection_coefficient=gamma)

        rc_sel = rc.select_frequencies(f_low=2 * un.MHz)
        np.testing.assert_allclose(rc_sel.freqs, freqs[1:])
        np.testing.assert_allclose(rc_sel.reflection_coefficient, gamma[1:])

        rc_rp = rc.rephase(0 * un.s)
        np.testing.assert_allclose(rc_rp.reflection_coefficient, gamma)

        with pytest.raises(ValueError):
            dt.ReflectionCoefficient(freqs=freqs, reflection_coefficient=gamma[:2])

        p = tmp_path / "test.csv"
        p.write_text("freq,real,imag\n1,1.0,0.0\n2,0.0,1.0\n3,-1.0,0.0\n")
        rc2 = dt.ReflectionCoefficient.from_csv(str(p), freq_unit=un.Hz)
        np.testing.assert_allclose(rc2.freqs, np.array([1.0, 2.0, 3.0]) * un.Hz)
        np.testing.assert_allclose(
            rc2.reflection_coefficient, np.array([1 + 0j, 0 + 1j, -1 + 0j])
        )

    @pytest.mark.parametrize(
        "delay",
        [0 * un.s, 1e-9 * un.s, -5e-9 * un.s],
    )
    def test_rephase_parametrized(self, delay):
        freqs = np.array([10.0, 20.0]) * un.MHz
        gamma = np.array([0.1 + 0j, -0.2 + 0j])
        rc = dt.ReflectionCoefficient(freqs=freqs, reflection_coefficient=gamma)
        rc_ph = rc.rephase(delay)
        expected = gamma * np.exp(2j * np.pi * (freqs * delay).to_value(""))
        np.testing.assert_allclose(rc_ph.reflection_coefficient, expected)

    @pytest.mark.parametrize(
        ("f_low", "f_high", "expected_count"),
        [
            (0 * un.MHz, np.inf * un.MHz, 3),
            (15 * un.MHz, np.inf * un.MHz, 2),
            (25 * un.MHz, 35 * un.MHz, 1),
            (40 * un.MHz, 50 * un.MHz, 0),
        ],
    )
    def test_select_frequencies_param(self, f_low, f_high, expected_count):
        freqs = np.array([10.0, 20.0, 30.0]) * un.MHz
        gamma = np.array([0.0 + 0j, 0.0 + 0j, 0.0 + 0j])
        rc = dt.ReflectionCoefficient(freqs=freqs, reflection_coefficient=gamma)
        rc_sel = rc.select_frequencies(f_low=f_low, f_high=f_high)
        assert rc_sel.freqs.size == expected_count

    @pytest.mark.parametrize(("ext", "delimiter"), [(".csv", ","), (".txt", " ")])
    def test_from_csv_variants(self, tmp_path, ext, delimiter):
        p = tmp_path / f"gamma{ext}"
        # write header and three rows using the specified delimiter
        rows = ["freq real imag", "1 1.0 0.0", "2 0.0 1.0", "3 -1.0 0.0"]
        if delimiter == ",":
            content = "freq,real,imag\n1,1.0,0.0\n2,0.0,1.0\n3,-1.0,0.0\n"
        else:
            content = "\n".join(rows) + "\n"
        p.write_text(content)
        rc = dt.ReflectionCoefficient.from_csv(str(p), freq_unit=un.Hz)
        np.testing.assert_allclose(
            rc.reflection_coefficient, np.array([1 + 0j, 0 + 1j, -1 + 0j])
        )

    def test_from_s1p_and_from_filespec(self, monkeypatch):
        freqs = np.array([10, 20]) * un.MHz
        s11 = np.array([0.1 + 0j, -0.2 + 0j])

        def fake_read_s1p(path):
            return {"frequency": freqs, "s11": s11}

        monkeypatch.setattr(dt, "read_s1p", fake_read_s1p)

        rc = dt.ReflectionCoefficient.from_s1p("/fake/path.s1p")
        np.testing.assert_allclose(rc.freqs, freqs)
        np.testing.assert_allclose(rc.reflection_coefficient, s11)


class TestCalkitReadings:
    def test_ideal_and_validation(self):
        freqs = np.array([1.0, 2.0, 3.0]) * un.MHz
        ideal = dt.CalkitReadings.ideal(freqs=freqs)
        for attr, expected in (
            (
                ideal.open.reflection_coefficient,
                np.ones_like(freqs.value, dtype=complex),
            ),
            (
                ideal.short.reflection_coefficient,
                -1 * np.ones_like(freqs.value, dtype=complex),
            ),
            (
                ideal.match.reflection_coefficient,
                np.zeros_like(freqs.value, dtype=complex),
            ),
        ):
            if hasattr(attr, "unit"):
                np.testing.assert_allclose(attr.value, expected)
                assert attr.unit == freqs.unit
            else:
                np.testing.assert_allclose(attr, expected)

        wrong_short = dt.ReflectionCoefficient(
            freqs=np.array([1.0]) * un.MHz, reflection_coefficient=np.array([1.0 + 0j])
        )
        good_open = dt.ReflectionCoefficient(
            freqs=freqs, reflection_coefficient=np.ones_like(freqs.value, dtype=complex)
        )
        good_match = dt.ReflectionCoefficient(
            freqs=freqs,
            reflection_coefficient=np.zeros_like(freqs.value, dtype=complex),
        )
        with pytest.raises(ValueError):
            dt.CalkitReadings(open=good_open, short=wrong_short, match=good_match)

    def test_from_filespec_uses_from_s1p(self, monkeypatch):
        freqs = np.array([10, 20]) * un.MHz
        s11 = np.array([0.1 + 0j, -0.2 + 0j])

        def fake_from_s1p(path, **kwargs):
            return dt.ReflectionCoefficient(freqs=freqs, reflection_coefficient=s11)

        monkeypatch.setattr(
            dt.ReflectionCoefficient,
            "from_s1p",
            classmethod(lambda cls, path, **kw: fake_from_s1p(path, **kw)),
        )

        filespec = SimpleNamespace(open="a.s1p", short="b.s1p", match="c.s1p")
        cr = dt.CalkitReadings.from_filespec(filespec)
        assert isinstance(cr.open, dt.ReflectionCoefficient)
        assert isinstance(cr.short, dt.ReflectionCoefficient)
        assert isinstance(cr.match, dt.ReflectionCoefficient)
