import numpy as np
import pytest
from astropy import units as un

from edges.cal import sparams as sp


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
        np.testing.assert_allclose(hlc.s12, [0.9 - 0.1j, 0.8 - 0.2j])
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
        np.testing.assert_allclose(hlc.s22, [0.4 + 0.5j])

    @pytest.mark.parametrize("ncols", [3, 5, 8, 9])
    def test_bad_number_of_columns(self, tmp_path, ncols):
        """Regression test: unexpected column counts raise instead of misparsing."""
        p = tmp_path / "sr.txt"
        p.write_text("\n".join(" ".join(["1.0"] * ncols) for _ in range(3)) + "\n")
        with pytest.raises(ValueError, match=f"{ncols} columns"):
            sp.read_semi_rigid_cable_sparams_file(p)
