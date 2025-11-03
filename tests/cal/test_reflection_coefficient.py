import numpy as np
import pytest
from astropy import units as u

from edges import modeling as mdl
from edges.cal import reflection_coefficient as rc
from edges.cal.ee import KNOWN_CABLES
from edges.cal.s11.calkit_standards import StandardsReadings
from edges.io.vna import SParams


def test_gamma_shift_zero():
    rng = np.random.default_rng()
    s11 = rng.normal(size=100)
    smatrix = rc.SMatrix([[0, 1], [1, 0]])
    np.testing.assert_allclose(s11, rc.gamma_embed(smatrix, s11))


def test_gamma_impedance_roundtrip():
    z0 = 50
    rng = np.random.default_rng()
    z = rng.normal(size=10)

    np.testing.assert_allclose(rc.gamma2impedance(rc.impedance2gamma(z, z0), z0), z)


def test_gamma_embed_rountrip():
    rng = np.random.default_rng()
    s11 = rng.uniform(0, 1, size=10) + rng.uniform(0, 1, size=10) * 1j
    s12 = rng.uniform(0, 1, size=10) + rng.uniform(0, 1, size=10) * 1j
    s22 = rng.uniform(0, 1, size=10) + rng.uniform(0, 1, size=10) * 1j

    gamma = rng.uniform(0, 1, size=10) + rng.uniform(0, 1, size=10) * 1j
    smat = rc.SMatrix([[s11, s12], [s12, s22]])

    np.testing.assert_allclose(
        rc.gamma_de_embed(rc.gamma_embed(smat, gamma), smat),
        gamma,
    )


class TestCalkit:
    def test_load_calkit_with_resistance(self):
        new = rc.get_calkit(rc.AGILENT_85033E, resistance_of_match=49.0 * u.Ohm)
        default = rc.get_calkit(rc.AGILENT_85033E)
        assert new.match.resistance == 49.0 * u.Ohm
        assert default.match.resistance == 50.0 * u.Ohm

    def test_calkit_standard_name(self):
        assert rc.CalkitStandard(resistance=50).name == "match"

        assert (
            rc.CalkitStandard(
                resistance=np.inf,
                capacitance_model=rc.AGILENT_85033E.open.capacitance_model,
            ).name
            == "open"
        )

        assert (
            rc.CalkitStandard(
                resistance=0, inductance_model=rc.AGILENT_85033E.short.inductance_model
            ).name
            == "short"
        )

    def test_calkit_termination_impedance(self):
        with pytest.raises(TypeError, match="freq must be a frequency quantity!"):
            # requires frequency to be in units
            rc.AGILENT_85033E.open.termination_impedance(np.linspace(50, 100, 100))

        assert (
            rc.AGILENT_85033E.match.termination_impedance(50 * u.MHz)
            == rc.AGILENT_85033E.match.resistance
        )

    def test_calkit_units(self):
        freq = np.linspace(50, 100, 100) * u.MHz

        ag = rc.AGILENT_85033E.open

        assert ag.termination_impedance(freq).unit == u.ohm
        assert ag.termination_gamma(freq).unit == u.dimensionless_unscaled
        assert ag.lossy_characteristic_impedance(freq).unit == u.ohm
        assert u.get_physical_type(ag.gl(freq)) == "dimensionless"
        assert ag.offset_gamma(freq).unit == u.dimensionless_unscaled
        assert ag.reflection_coefficient(freq).unit == u.dimensionless_unscaled

    def test_calkit_quantities_match_trivial(self):
        """A test that for a simple calkit definition, the outputs are correct."""
        std = rc.CalkitStandard(
            resistance=50.0 * u.Ohm,
            offset_impedance=50 * u.Ohm,
            offset_delay=0 * u.ps,
            offset_loss=0.0,
        )

        assert std.intrinsic_gamma == 0.0
        assert std.capacitance_model is None
        assert std.inductance_model is None
        assert std.termination_gamma(freq=150 * u.MHz) == 0.0
        assert std.termination_impedance(freq=150 * u.MHz) == 50 * u.Ohm
        assert std.lossy_characteristic_impedance(freq=150 * u.MHz) == 50 * u.Ohm
        assert std.gl(freq=150 * u.MHz) == 0.0
        assert std.reflection_coefficient(150 * u.MHz) == 0.0

    def test_calkit_quantities_match_with_delay(self):
        """A test that for a simple calkit definition, the outputs are correct."""
        std = rc.CalkitStandard(
            resistance=50.0 * u.Ohm,
            offset_impedance=50 * u.Ohm,
            offset_delay=50 * u.ps,
            offset_loss=0.0,
        )

        assert std.intrinsic_gamma == 0.0
        assert std.capacitance_model is None
        assert std.inductance_model is None
        assert std.termination_gamma(freq=150 * u.MHz) == 0.0
        assert std.termination_impedance(freq=150 * u.MHz) == 50 * u.Ohm
        assert std.lossy_characteristic_impedance(freq=150 * u.MHz) == 50 * u.Ohm
        assert std.gl(freq=200 * u.MHz) == 1e-2 * 2j * np.pi
        assert std.reflection_coefficient(150 * u.MHz) == 0.0

    def test_calkit_quantities_match_with_loss(self):
        """A test that for a simple calkit definition, the outputs are correct."""
        std = rc.CalkitStandard(
            resistance=50.0 * u.Ohm,
            offset_impedance=50 * u.Ohm,
            offset_delay=(25 / np.pi) * u.ps,
            offset_loss=4 * np.pi * u.Gohm / u.s,
        )

        assert std.intrinsic_gamma == 0.0
        assert std.capacitance_model is None
        assert std.inductance_model is None
        assert std.termination_gamma(freq=150 * u.MHz) == 0.0
        assert std.termination_impedance(freq=150 * u.MHz) == 50 * u.Ohm
        assert std.lossy_characteristic_impedance(freq=1 * u.GHz) == (51 - 1j) * u.Ohm
        assert std.gl(freq=1 * u.GHz) == 5e-2j + (1 + 1j) * 1e-3
        assert std.reflection_coefficient(150 * u.MHz) != 0.0

    def test_calkit_quantities_open_trivial(self):
        """A test that for a simple calkit definition, the outputs are correct."""
        std = rc.CalkitStandard(
            resistance=np.inf * u.Ohm,
            offset_impedance=50 * u.Ohm,
            offset_delay=0 * u.ps,
            offset_loss=0 * u.Gohm / u.s,
            capacitance_model=mdl.Polynomial(
                parameters=[1e-9 / (100 * np.pi), 0, 0, 0]
            ),
        )

        assert std.intrinsic_gamma == 1.0
        assert std.capacitance_model(1e9) == 1e-9 / (100 * np.pi)
        assert std.inductance_model is None
        assert std.termination_impedance(freq=1 * u.GHz) == -50j * u.Ohm
        assert std.termination_gamma(freq=1 * u.GHz) == -1j
        assert std.lossy_characteristic_impedance(freq=1 * u.GHz) == 50 * u.Ohm
        assert std.gl(freq=1 * u.GHz) == 0.0
        assert std.reflection_coefficient(1 * u.GHz) == -1j


class TestTwoPortNetwork:
    def test_invalid_shape(self):
        with pytest.raises(ValueError, match="Matrix must have shape"):
            rc.TwoPortNetwork(np.ones((3, 3, 10)))

        with pytest.raises(ValueError, match="x must have ndim in"):
            rc.TwoPortNetwork(np.ones((3, 3, 10, 11)))

    def test_roundtrip_zmatrix(self):
        rng = np.random.default_rng()
        z = rng.normal(size=(2, 2, 1))
        network = rc.TwoPortNetwork.from_zmatrix(z)
        np.testing.assert_allclose(network.zmatrix, z)

        # Check that we can convert back to a TwoPortNetwork
        network2 = rc.TwoPortNetwork.from_zmatrix(network.zmatrix)
        np.testing.assert_allclose(network2.impedance_matrix, z)

    def test_roundtrip_ymatrix(self):
        rng = np.random.default_rng()
        y = rng.normal(size=(2, 2, 1))
        network = rc.TwoPortNetwork.from_ymatrix(y)
        np.testing.assert_allclose(network.ymatrix, y)

        # Check that we can convert back to a TwoPortNetwork
        network2 = rc.TwoPortNetwork.from_ymatrix(network.ymatrix)
        np.testing.assert_allclose(network2.admittance_matrix, y)

    def test_roundtrip_hmatrix(self):
        rng = np.random.default_rng()
        h = rng.normal(size=(2, 2, 1))
        network = rc.TwoPortNetwork.from_hmatrix(h)
        np.testing.assert_allclose(network.hmatrix, h)

        # Check that we can convert back to a TwoPortNetwork
        network2 = rc.TwoPortNetwork.from_hmatrix(network.hmatrix)
        np.testing.assert_allclose(network2.hybrid_matrix, h)

    def test_roundtrip_abcd(self):
        rng = np.random.default_rng()
        abcd = rng.normal(size=(2, 2, 1))
        network = rc.TwoPortNetwork.from_abcd(abcd)
        np.testing.assert_allclose(network.x, abcd)

    def test_aliases(self):
        rng = np.random.default_rng()
        z = rng.normal(size=(2, 2, 1))
        network = rc.TwoPortNetwork.from_zmatrix(z)
        assert np.allclose(network.A, network.x[0, 0])
        assert np.allclose(network.B, network.x[0, 1])
        assert np.allclose(network.C, network.x[1, 0])
        assert np.allclose(network.D, network.x[1, 1])

    def test_reciprocity(self):
        network = rc.TwoPortNetwork([[1, 0], [0, 1]])
        assert network.is_reciprocal()
        assert network.is_symmetric()

        non_reciprocal = rc.TwoPortNetwork([[1, 1], [1, 1]])
        assert not non_reciprocal.is_reciprocal()
        assert network.is_symmetric()

    def test_lossless(self):
        network = rc.TwoPortNetwork([[1, 0], [0, 1]])
        assert network.is_lossless()

        lossy = rc.TwoPortNetwork([[1, 0], [0, 1 + 0.1j]])
        assert not lossy.is_lossless()

    @pytest.mark.parametrize(
        "addfunc",
        ["add_in_series", "add_in_parallel", "add_in_series_parallel", "cascade_with"],
    )
    def add_bad(self, addfunc):
        network1 = rc.TwoPortNetwork([[1, 0], [0, 1]])

        fnc = getattr(network1, addfunc)
        with pytest.raises(ValueError, match="Two matrices must be of the same type"):
            fnc([[0, 1], [1, 0]])

        with pytest.raises(
            ValueError, match="Two matrices must have the same dimensions"
        ):
            fnc(np.array([[[1, 0], [0, 1]], [[1, 0], [0, 1 + 0.1j]]]))

    def test_add_in_series(self):
        network1 = rc.TwoPortNetwork.from_zmatrix([[1, 1], [1, 1]])
        network2 = rc.TwoPortNetwork.from_zmatrix([[1, 1], [1, 1 + 0.1j]])

        combined = network1.add_in_series(network2)
        expected = rc.TwoPortNetwork.from_zmatrix([[2, 2], [2, 2 + 0.1j]])

        assert combined == expected

    def test_add_in_parallel(self):
        network1 = rc.TwoPortNetwork.from_ymatrix([[1, 1], [1, 1]])
        network2 = rc.TwoPortNetwork.from_ymatrix([[1, 1], [1, 1 + 0.1j]])

        combined = network1.add_in_parallel(network2)
        expected = rc.TwoPortNetwork.from_ymatrix([[2, 2], [2, 2 + 0.1j]])

        assert combined == expected

    def test_add_in_series_parallel(self):
        network1 = rc.TwoPortNetwork.from_hmatrix([[1, 1], [1, 1]])
        network2 = rc.TwoPortNetwork.from_hmatrix([[1, 1], [1, 1 + 0.1j]])

        combined = network1.add_in_series_parallel(network2)
        expected = rc.TwoPortNetwork.from_hmatrix([[2, 2], [2, 2 + 0.1j]])

        assert combined == expected

    def test_cascade_with(self):
        network1 = rc.TwoPortNetwork([[1, 0], [0, 1]])
        network2 = rc.TwoPortNetwork([[1, 0], [0, 1]])

        combined = network1.cascade_with(network2)
        expected = rc.TwoPortNetwork([[1, 0], [0, 1]])

        assert combined == expected

    def test_roundtrip_sparams(self):
        rng = np.random.default_rng()
        s11 = rng.uniform(0, 1, size=10) + rng.uniform(0, 1, size=10) * 1j
        s12 = rng.uniform(0, 1, size=10) + rng.uniform(0, 1, size=10) * 1j

        sparams = rc.SMatrix.from_sparams(s11=s11, s12=s12)
        network = rc.TwoPortNetwork.from_smatrix(sparams, z0=50 * u.Ohm)
        out_sp = network.as_smatrix(source_impedance=50)
        assert out_sp == sparams

    def test_from_transmission_line(self):
        line = KNOWN_CABLES["balun-tube"].as_transmission_line(freq=50 * u.MHz)
        network = rc.TwoPortNetwork.from_transmission_line(line, length=1 * u.m)

        assert network.is_reciprocal()
        assert network.is_symmetric()


class TestSMatrix:
    def setup_class(self):
        rng = np.random.default_rng()
        s11 = rng.uniform(0, 1, size=10) + rng.uniform(0, 1, size=10) * 1j
        s12 = rng.uniform(0, 1, size=10) + rng.uniform(0, 1, size=10) * 1j

        self.smatrix = rc.SMatrix.from_sparams(s11=s11, s12=s12)

    def test_from_sparams(self):
        assert np.allclose(self.smatrix.s21, self.smatrix.s12)
        assert np.allclose(self.smatrix.s22, self.smatrix.s11)

    def test_roundtrip_transfer_matrix(self):
        transfer_matrix = self.smatrix.as_transfer_matrix()

        assert transfer_matrix.shape == (2, 2, len(self.smatrix.s11))

        new_smatrix = rc.SMatrix.from_transfer_matrix(transfer_matrix)
        assert np.allclose(new_smatrix.s11, self.smatrix.s11)
        assert np.allclose(new_smatrix.s12, self.smatrix.s12)
        assert np.allclose(new_smatrix.s21, self.smatrix.s21)
        assert np.allclose(new_smatrix.s22, self.smatrix.s22)

    def test_cascade(self):
        new = self.smatrix.cascade_with(self.smatrix)
        assert isinstance(new, rc.SMatrix)
        assert new.s11.shape == self.smatrix.s11.shape

    def test_reciprocal(self):
        assert self.smatrix.is_reciprocal()

    def test_lossless(self):
        lossless_smatrix = rc.SMatrix.from_sparams(
            s11=np.ones(10),
            s12=np.zeros(10),
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
        line = KNOWN_CABLES["balun-tube"].as_transmission_line(freq=50 * u.MHz)
        smatrix = rc.SMatrix.from_transmission_line(line, length=1 * u.m)

        assert smatrix.is_reciprocal()

    def test_from_calkit_and_vna(self):
        calkit = rc.AGILENT_85033E
        freq = np.linspace(50, 100, 10) * u.MHz
        vna = StandardsReadings(
            open=SParams(s11=np.array([1 + 0j] * 10), freq=freq),
            short=SParams(s11=np.array([-1 + 0j] * 10), freq=freq),
            match=SParams(s11=np.array([0 + 0j] * 10), freq=freq),
        )

        smatrix = rc.SMatrix.from_calkit_and_vna(calkit, vna)
        assert smatrix.is_reciprocal()
