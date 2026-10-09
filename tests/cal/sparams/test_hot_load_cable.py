from pathlib import Path

import numpy as np
import pytest
from astropy import units as un

from edges.cal import sparams as sp
from edges.cal.loss import LossFunctionGivenSparams
from edges.data import DATA_PATH


class TestSparamsFromSemiRigid:
    def test_2017_semi_rigid(self):
        hlc = sp.read_semi_rigid_cable_sparams_file(
            path=":semi_rigid_s_parameters_2017.txt"
        )
        assert hlc.s12.dtype == complex

    def test_seven_columns(self, tmp_path):
        p = tmp_path / "sr.txt"
        p.write_text(
            "# f re(s11) im(s11) re(s12s21) im(s12s21) re(s22) im(s22)\n"
            "50 0.1 0.2 0.9 -0.1 0.3 0.4\n"
            "60 0.2 0.3 0.8 -0.2 0.4 0.5\n"
        )
        hlc = sp.read_semi_rigid_cable_sparams_file(p)
        np.testing.assert_allclose(hlc.freqs.to_value("MHz"), [50, 60])
        np.testing.assert_allclose(hlc.s11, [0.1 + 0.2j, 0.2 + 0.3j])
        # The file stores the product S12*S21; each of s12 and s21 is its square root.
        np.testing.assert_allclose(
            hlc.s12 * hlc.s21, [0.9 - 0.1j, 0.8 - 0.2j], rtol=1e-14
        )
        np.testing.assert_allclose(hlc.s12, hlc.s21, rtol=0, atol=0)
        np.testing.assert_allclose(
            hlc.s12, np.sqrt([0.9 - 0.1j, 0.8 - 0.2j]), rtol=1e-14
        )
        np.testing.assert_allclose(hlc.s22, [0.3 + 0.4j, 0.4 + 0.5j])

    def test_six_columns(self, tmp_path):
        p = tmp_path / "sr.txt"
        p.write_text(
            "# f re(s11) im(s11) abs(s21) re(s22) im(s22)\n"
            "50 0.1 0.2 0.9 0.3 0.4\n"
            "60 0.2 0.3 0.8 0.4 0.5\n"
        )
        hlc = sp.read_semi_rigid_cable_sparams_file(p, f_low=55 * un.MHz)
        np.testing.assert_allclose(hlc.freqs.to_value("MHz"), [60])
        np.testing.assert_allclose(hlc.s11, [0.2 + 0.3j])
        np.testing.assert_allclose(hlc.s12, [0.8])
        np.testing.assert_allclose(hlc.s21, [0.8])
        np.testing.assert_allclose(hlc.s22, [0.4 + 0.5j])

    @pytest.mark.parametrize("ncols", [3, 5, 8, 9])
    def test_bad_number_of_columns(self, tmp_path, ncols):
        """Regression test: unexpected column counts raise instead of misparsing."""
        p = tmp_path / "sr.txt"
        p.write_text("\n".join(" ".join(["1.0"] * ncols) for _ in range(3)) + "\n")
        with pytest.raises(ValueError, match=f"{ncols} columns"):
            sp.read_semi_rigid_cable_sparams_file(p)


def _available_gain(
    gamma: np.ndarray,
    s11: np.ndarray,
    s12s21: np.ndarray,
    s22: np.ndarray,
) -> np.ndarray:
    """Available gain of a reciprocal 2-port cable, from its S12*S21 product.

    This is the loss used by Alan's C code (and alanmode): the load reflection
    coefficient T at port 2 is de-embedded from the reflection coefficient ``gamma``
    measured at port 1, and |S21|^2 = |S12*S21| for a reciprocal cable.
    """
    t = (gamma - s11) / (s12s21 + s22 * (gamma - s11))
    return (
        np.abs(s12s21)
        * (1 - np.abs(t) ** 2)
        / ((1 - np.abs(gamma) ** 2) * np.abs(1 - s22 * t) ** 2)
    )


class TestSemiRigidCableLoss:
    """The loss computed from the built-in cable files is the cable's available gain."""

    @pytest.fixture(scope="class")
    def gamma(self) -> sp.ReflectionCoefficient:
        freqs = np.linspace(50, 100, 51) * un.MHz
        # A hot-load-like reflection coefficient (|gamma| ~ 0.1), varying with freq.
        g = 0.1 * np.exp(2j * np.pi * freqs.to_value("MHz") / 37.0)
        return sp.ReflectionCoefficient(freqs=freqs, reflection_coefficient=g)

    def test_2015_file_loss_is_available_gain(self, gamma):
        raw = np.genfromtxt(DATA_PATH / "semi_rigid_s_parameters_WITH_HEADER.txt")
        raw = raw[np.isin(raw[:, 0], gamma.freqs.to_value("MHz"))]
        assert len(raw) == gamma.freqs.size

        s11 = raw[:, 1] + 1j * raw[:, 2]
        s12s21 = raw[:, 3] + 1j * raw[:, 4]
        s22 = raw[:, 5] + 1j * raw[:, 6]

        hlc = sp.read_semi_rigid_cable_sparams_file(
            path=":semi_rigid_s_parameters_WITH_HEADER.txt",
            f_low=gamma.freqs.min(),
            f_high=gamma.freqs.max(),
        )
        np.testing.assert_allclose(
            hlc.freqs.to_value("MHz"), gamma.freqs.to_value("MHz"), rtol=0, atol=0
        )
        np.testing.assert_allclose(hlc.s12 * hlc.s21, s12s21, rtol=1e-14, atol=0)

        loss = LossFunctionGivenSparams(hlc)(gamma)
        expected = _available_gain(gamma.reflection_coefficient, s11, s12s21, s22)
        np.testing.assert_allclose(loss, expected, rtol=1e-12, atol=0)

        # Alan's convention (alanmode): the S12*S21 product is carried as s12 and
        # square-rooted before computing the loss. It must agree.
        alan = LossFunctionGivenSparams(
            sp.SParams(
                freqs=hlc.freqs,
                s11=s11,
                s12=np.sqrt(s12s21),
                s21=np.sqrt(s12s21),
                s22=s22,
            )
        )(gamma)
        np.testing.assert_allclose(loss, alan, rtol=1e-12, atol=0)

        # |S12*S21| is 0.9965 at 50 MHz and 0.9942 at 100 MHz. The loss must not be
        # squared (which would give |S12*S21|^2 ~ 0.9884 at 100 MHz).
        np.testing.assert_allclose(
            np.abs(hlc.s12 * hlc.s21)[[0, -1]], [0.996479, 0.994151], atol=1e-6, rtol=0
        )
        assert np.all(loss > 0.99)
        assert np.all(loss < 1)

    def test_2015_file_matched_load(self):
        """With a matched load at port 2, the gain is |S21|^2 / (1 - |S11|^2)."""
        hlc = sp.read_semi_rigid_cable_sparams_file(
            path=":semi_rigid_s_parameters_WITH_HEADER.txt",
            f_low=50 * un.MHz,
            f_high=100 * un.MHz,
        )
        # A matched load at port 2 is seen as S11 at port 1.
        gamma = sp.ReflectionCoefficient(
            freqs=hlc.freqs, reflection_coefficient=hlc.s11
        )
        loss = LossFunctionGivenSparams(hlc)(gamma)
        expected = np.abs(hlc.s12 * hlc.s21) / (1 - np.abs(hlc.s11) ** 2)
        np.testing.assert_allclose(loss, expected, rtol=1e-12, atol=0)

    def test_2017_file_loss_is_available_gain(self):
        raw = np.genfromtxt(DATA_PATH / "semi_rigid_s_parameters_2017.txt")
        raw = raw[(raw[:, 0] >= 60) & (raw[:, 0] <= 61)]

        hlc = sp.read_semi_rigid_cable_sparams_file(
            path=":semi_rigid_s_parameters_2017.txt",
            f_low=60 * un.MHz,
            f_high=61 * un.MHz,
        )
        s11 = raw[:, 1] + 1j * raw[:, 2]
        s21_abs = raw[:, 3]
        s22 = raw[:, 4] + 1j * raw[:, 5]

        # The 2017 file tabulates |S21|, so |s12 * s21| = |S21|^2.
        np.testing.assert_allclose(
            np.abs(hlc.s12 * hlc.s21), s21_abs**2, rtol=1e-14, atol=0
        )

        g = 0.1 * np.exp(2j * np.pi * raw[:, 0] / 0.37)
        gamma = sp.ReflectionCoefficient(freqs=hlc.freqs, reflection_coefficient=g)
        loss = LossFunctionGivenSparams(hlc)(gamma)
        expected = _available_gain(g, s11, s21_abs**2 + 0j, s22)
        np.testing.assert_allclose(loss, expected, rtol=1e-12, atol=0)


def test_2015_file_loss_close_to_alan_b18(testdata_path: Path):
    """The default cable file gives a hot-load loss close to Alan's B18 loss.

    Alan's B18 calibration uses a different measurement of the semi-rigid cable
    (from his 2015-09-16 S11 file), so the two do not agree exactly: they differ
    by up to 2.8e-4. A squared loss would differ by up to 5.6e-3, and no loss at all
    by 6e-3.
    """
    data = testdata_path / "alanmode" / "b18_cal_test"
    alan_loss = np.genfromtxt(data / "hot_load_loss.txt")
    s11m = np.genfromtxt(data / "s11_modelled_alan.txt", names=True)
    mask = (s11m["freq"] >= 50) & (s11m["freq"] <= 100)
    gamma = sp.ReflectionCoefficient(
        freqs=s11m["freq"][mask] * un.MHz,
        reflection_coefficient=(s11m["hot_real"] + 1j * s11m["hot_imag"])[mask],
    )

    hlc = sp.read_semi_rigid_cable_sparams_file(
        f_low=50 * un.MHz, f_high=100 * un.MHz
    ).smoothed(params=sp.hot_load_cable_model_params(), freqs=gamma.freqs)
    loss = LossFunctionGivenSparams(hlc)(gamma)
    np.testing.assert_allclose(loss, alan_loss[mask, 1], atol=3e-4, rtol=0)
