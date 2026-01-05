from types import SimpleNamespace

import numpy as np
import pytest
from astropy import units as un

from edges.cal import sparams as sp
from edges.cal.sparams.core import datatypes as dt


def freqs_mhz(n=3):
    return np.arange(1, n + 1) * un.MHz


class TestSParams:
    def setup_class(self):
        rng = np.random.default_rng()
        s11 = rng.uniform(0, 1, size=10) + rng.uniform(0, 1, size=10) * 1j
        s12 = rng.uniform(0, 1, size=10) + rng.uniform(0, 1, size=10) * 1j
        freqs = np.linspace(50, 100, 10) * un.MHz

        self.smatrix = dt.SParams(s11=s11, s12=s12, freqs=freqs)

    def test_from_sparams(self):
        assert np.allclose(self.smatrix.s21, self.smatrix.s12)
        assert np.allclose(self.smatrix.s22, self.smatrix.s11)

    def test_roundtrip_transfer_matrix(self):
        transfer_matrix = self.smatrix.as_transfer_matrix()

        assert transfer_matrix.shape == (2, 2, len(self.smatrix.s11))

        new_smatrix = dt.SParams.from_transfer_matrix(
            self.smatrix.freqs, transfer_matrix
        )
        assert np.allclose(new_smatrix.s11, self.smatrix.s11)
        assert np.allclose(new_smatrix.s12, self.smatrix.s12)
        assert np.allclose(new_smatrix.s21, self.smatrix.s21)
        assert np.allclose(new_smatrix.s22, self.smatrix.s22)

    def test_cascade(self):
        new = self.smatrix.cascade_with(self.smatrix)
        assert isinstance(new, dt.SParams)
        assert new.s11.shape == self.smatrix.s11.shape

    def test_reciprocal(self):
        assert self.smatrix.is_reciprocal()

    def test_lossless(self):
        freqs = np.linspace(50, 100, 10) * un.MHz
        lossless_smatrix = dt.SParams(
            s11=np.ones(10),
            s12=np.zeros(10),
            freqs=freqs,
        )
        assert lossless_smatrix.is_lossless()

    def test_properties(self):
        assert np.allclose(self.smatrix.complex_linear_gain, self.smatrix.s12)
        assert np.allclose(self.smatrix.scalar_linear_gain, np.abs(self.smatrix.s21))
        assert np.allclose(
            self.smatrix.scalar_logarithmic_gain,
            20 * np.log10(np.abs(self.smatrix.s21)),
        )
        assert np.allclose(
            self.smatrix.insertion_loss, -self.smatrix.scalar_logarithmic_gain
        )
        assert np.allclose(
            self.smatrix.input_return_loss, -20 * np.log10(np.abs(self.smatrix.s11))
        )
        assert np.allclose(
            self.smatrix.output_return_loss, -20 * np.log10(np.abs(self.smatrix.s22))
        )
        assert np.allclose(
            self.smatrix.reverse_gain, 20 * np.log10(np.abs(self.smatrix.s12))
        )

    def test_voltage_standing_wave_ratio(self):
        vswr = self.smatrix.voltage_standing_wave_ratio_in()
        assert np.all(np.isfinite(vswr))

        vswr = self.smatrix.voltage_standing_wave_ratio_out()
        assert np.all(np.isfinite(vswr))

    def test_from_transmission_line(self):
        line = sp.KNOWN_CABLES["balun-tube"].as_transmission_line(
            freqs=np.array([50]) * un.MHz
        )
        smatrix = line.scattering_parameters(
            line_length=1 * un.m
        )  # dt.SParams.from_transmission_line(line, length=1 * un.m)

        assert smatrix.is_reciprocal()

    def test_from_calkit_and_vna(self):
        calkit = sp.AGILENT_85033E
        freq = np.linspace(50, 100, 10) * un.MHz
        vna = dt.CalkitReadings(
            open=dt.ReflectionCoefficient(
                reflection_coefficient=np.array([1 + 0j] * 10), freqs=freq
            ),
            short=dt.ReflectionCoefficient(
                reflection_coefficient=np.array([-1 + 0j] * 10), freqs=freq
            ),
            match=dt.ReflectionCoefficient(
                reflection_coefficient=np.array([0 + 0j] * 10), freqs=freq
            ),
        )

        smatrix = dt.SParams.from_calkit_measurements(model=calkit, measurements=vna)
        assert smatrix.is_reciprocal()

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

    def test_smatrix_alias(self):
        assert np.allclose(self.smatrix.s, self.smatrix.smatrix)

    def test_bad_cascading(self):
        diff_smatrix = dt.SParams(
            freqs=self.smatrix.freqs[:-1],
            s11=self.smatrix.s11[:-1],
            s12=self.smatrix.s12[:-1],
            s21=self.smatrix.s21[:-1],
            s22=self.smatrix.s22[:-1],
        )
        with pytest.raises(ValueError, match="Both SMatrices must have the same shape"):
            self.smatrix.cascade_with(diff_smatrix)

    def test_reverse_isolation(self):
        assert np.allclose(
            self.smatrix.reverse_isolation, -np.abs(self.smatrix.reverse_gain)
        )


class TestReflectionCoefficient:
    def test_select_rephase_and_validators(self, tmp_path):
        freqs = np.array([1.0, 2.0, 3.0]) * un.MHz
        gamma = np.array([1 + 0j, 0 + 1j, -1 + 0j])
        rc = dt.ReflectionCoefficient(freqs=freqs, reflection_coefficient=gamma)

        rc_sel = rc.select_frequencies(f_low=2 * un.MHz)
        np.testing.assert_allclose(rc_sel.freqs, freqs[1:])
        np.testing.assert_allclose(rc_sel.reflection_coefficient, gamma[1:])

        rc_rp = rc.remove_delay(0 * un.s)
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
        rc_ph = rc.remove_delay(delay)
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

    def test_bad_sparam_shape(self):
        freqs = np.array([10, 20]) * un.MHz
        s11 = np.array([0.1 + 0j, -0.2 + 0j])
        s12 = np.array([0.0 + 0j])  # wrong shape

        with pytest.raises(
            ValueError, match="Number of frequencies in S-matrix must match"
        ):
            dt.SParams(freqs=freqs, s11=s11, s12=s12)


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

    def test_different_freqs_in_standards(self):
        freqs = np.linspace(50, 100, 100) * un.MHz
        s = np.linspace(0, 1, 100) + 0j

        vna1 = dt.ReflectionCoefficient(freqs=freqs, reflection_coefficient=s)
        vna2 = dt.ReflectionCoefficient(freqs=freqs[:80], reflection_coefficient=s[:80])

        with pytest.raises(
            ValueError, match="short standard does not have same frequencies"
        ):
            dt.CalkitReadings(open=vna1, short=vna2, match=vna1)

        with pytest.raises(
            ValueError, match="match standard does not have same frequencies"
        ):
            dt.CalkitReadings(open=vna1, short=vna1, match=vna2)

        sr = dt.CalkitReadings(open=vna1, short=vna1, match=vna1)
        assert np.all(sr.freqs == vna1.freqs)

        # Also catch when the frequencies are same shape but different values
        vna3 = dt.ReflectionCoefficient(
            freqs=freqs + 1 * un.MHz, reflection_coefficient=s
        )
        with pytest.raises(
            ValueError, match="match standard does not have same frequencies"
        ):
            dt.CalkitReadings(open=vna1, short=vna1, match=vna3)
