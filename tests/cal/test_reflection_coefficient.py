import numpy as np
import pytest
from astropy import units as un

from edges import modeling as mdl
from edges.cal import sparams as sp
from edges.cal.sparams import KNOWN_CABLES, CalkitReadings


def test_gamma_shift_zero():
    rng = np.random.default_rng()
    freqs = np.linspace(50, 100, 100) * un.MHz
    s11 = sp.ReflectionCoefficient(
        reflection_coefficient=rng.normal(size=100), freqs=freqs
    )

    smatrix = sp.SParams(s11=np.zeros(100), s12=np.ones(100), freqs=freqs)
    np.testing.assert_allclose(
        s11.reflection_coefficient, sp.gamma_embed(s11, smatrix).reflection_coefficient
    )


def test_gamma_impedance_roundtrip():
    z0 = 50
    rng = np.random.default_rng()
    z = rng.normal(size=10)

    np.testing.assert_allclose(sp.gamma2impedance(sp.impedance2gamma(z, z0), z0), z)


def test_gamma_embed_rountrip():
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


class TestCalkit:
    def test_load_calkit_with_resistance(self):
        new = sp.get_calkit(sp.AGILENT_85033E, resistance_of_match=49.0 * un.Ohm)
        default = sp.get_calkit(sp.AGILENT_85033E)
        assert new.match.resistance == 49.0 * un.Ohm
        assert default.match.resistance == 50.0 * un.Ohm

    def test_calkit_standard_name(self):
        assert sp.CalkitStandard(resistance=50).name == "match"

        assert (
            sp.CalkitStandard(
                resistance=np.inf,
                capacitance_model=sp.AGILENT_85033E.open.capacitance_model,
            ).name
            == "open"
        )

        assert (
            sp.CalkitStandard(
                resistance=0, inductance_model=sp.AGILENT_85033E.short.inductance_model
            ).name
            == "short"
        )

    def test_calkit_termination_impedance(self):
        with pytest.raises(TypeError, match="freq must be a frequency quantity!"):
            # requires frequency to be in units
            sp.AGILENT_85033E.open.termination_impedance(np.linspace(50, 100, 100))

        assert (
            sp.AGILENT_85033E.match.termination_impedance(50 * un.MHz)
            == sp.AGILENT_85033E.match.resistance
        )

    def test_calkit_units(self):
        freq = np.linspace(50, 100, 100) * un.MHz

        ag = sp.AGILENT_85033E.open

        assert ag.termination_impedance(freq).unit == un.ohm
        assert ag.termination_gamma(freq).unit == un.dimensionless_unscaled
        assert ag.lossy_characteristic_impedance(freq).unit == un.ohm
        assert un.get_physical_type(ag.gl(freq)) == "dimensionless"
        assert ag.offset_gamma(freq).unit == un.dimensionless_unscaled
        assert isinstance(ag.reflection_coefficient(freq), sp.ReflectionCoefficient)

    def test_calkit_quantities_match_trivial(self):
        """A test that for a simple calkit definition, the outputs are correct."""
        std = sp.CalkitStandard(
            resistance=50.0 * un.Ohm,
            offset_impedance=50 * un.Ohm,
            offset_delay=0 * un.ps,
            offset_loss=0.0,
        )

        assert std.intrinsic_gamma == 0.0
        assert std.capacitance_model is None
        assert std.inductance_model is None
        assert std.termination_gamma(freq=150 * un.MHz) == 0.0
        assert std.termination_impedance(freq=150 * un.MHz) == 50 * un.Ohm
        assert std.lossy_characteristic_impedance(freq=150 * un.MHz) == 50 * un.Ohm
        assert std.gl(freq=150 * un.MHz) == 0.0
        assert (
            std.reflection_coefficient(np.array([150]) * un.MHz).reflection_coefficient[
                0
            ]
            == 0.0
        )

    def test_calkit_quantities_match_with_delay(self):
        """A test that for a simple calkit definition, the outputs are correct."""
        std = sp.CalkitStandard(
            resistance=50.0 * un.Ohm,
            offset_impedance=50 * un.Ohm,
            offset_delay=50 * un.ps,
            offset_loss=0.0,
        )

        assert std.intrinsic_gamma == 0.0
        assert std.capacitance_model is None
        assert std.inductance_model is None
        assert std.termination_gamma(freq=150 * un.MHz) == 0.0
        assert std.termination_impedance(freq=150 * un.MHz) == 50 * un.Ohm
        assert std.lossy_characteristic_impedance(freq=150 * un.MHz) == 50 * un.Ohm
        assert std.gl(freq=200 * un.MHz) == 1e-2 * 2j * np.pi
        assert (
            std.reflection_coefficient(np.array([150]) * un.MHz).reflection_coefficient[
                0
            ]
            == 0.0
        )

    def test_calkit_quantities_match_with_loss(self):
        """A test that for a simple calkit definition, the outputs are correct."""
        std = sp.CalkitStandard(
            resistance=50.0 * un.Ohm,
            offset_impedance=50 * un.Ohm,
            offset_delay=(25 / np.pi) * un.ps,
            offset_loss=4 * np.pi * un.Gohm / un.s,
        )

        assert std.intrinsic_gamma == 0.0
        assert std.capacitance_model is None
        assert std.inductance_model is None
        assert std.termination_gamma(freq=150 * un.MHz) == 0.0
        assert std.termination_impedance(freq=150 * un.MHz) == 50 * un.Ohm
        assert std.lossy_characteristic_impedance(freq=1 * un.GHz) == (51 - 1j) * un.Ohm
        assert std.gl(freq=1 * un.GHz) == 5e-2j + (1 + 1j) * 1e-3
        assert (
            std.reflection_coefficient(np.array([150]) * un.MHz).reflection_coefficient[
                0
            ]
            != 0.0
        )

    def test_calkit_quantities_open_trivial(self):
        """A test that for a simple calkit definition, the outputs are correct."""
        std = sp.CalkitStandard(
            resistance=np.inf * un.Ohm,
            offset_impedance=50 * un.Ohm,
            offset_delay=0 * un.ps,
            offset_loss=0 * un.Gohm / un.s,
            capacitance_model=mdl.Polynomial(
                parameters=[1e-9 / (100 * np.pi), 0, 0, 0]
            ),
        )

        assert std.intrinsic_gamma == 1.0
        assert std.capacitance_model(1e9) == 1e-9 / (100 * np.pi)
        assert std.inductance_model is None
        assert std.termination_impedance(freq=1 * un.GHz) == -50j * un.Ohm
        assert std.termination_gamma(freq=1 * un.GHz) == -1j
        assert std.lossy_characteristic_impedance(freq=1 * un.GHz) == 50 * un.Ohm
        assert std.gl(freq=1 * un.GHz) == 0.0
        assert (
            std.reflection_coefficient(np.array([1]) * un.GHz).reflection_coefficient[0]
            == -1j
        )


class TestTwoPortNetwork:
    def test_invalid_shape(self):
        with pytest.raises(ValueError, match="Matrix must have shape"):
            sp.TwoPortNetwork(np.ones((3, 3, 10)))

        with pytest.raises(ValueError, match="x must have ndim in"):
            sp.TwoPortNetwork(np.ones((3, 3, 10, 11)))

    def test_roundtrip_zmatrix(self):
        rng = np.random.default_rng()
        z = rng.normal(size=(2, 2, 1))
        network = sp.TwoPortNetwork.from_zmatrix(z)
        np.testing.assert_allclose(network.zmatrix, z)

        # Check that we can convert back to a TwoPortNetwork
        network2 = sp.TwoPortNetwork.from_zmatrix(network.zmatrix)
        np.testing.assert_allclose(network2.impedance_matrix, z)

    def test_roundtrip_ymatrix(self):
        rng = np.random.default_rng()
        y = rng.normal(size=(2, 2, 1))
        network = sp.TwoPortNetwork.from_ymatrix(y)
        np.testing.assert_allclose(network.ymatrix, y)

        # Check that we can convert back to a TwoPortNetwork
        network2 = sp.TwoPortNetwork.from_ymatrix(network.ymatrix)
        np.testing.assert_allclose(network2.admittance_matrix, y)

    def test_roundtrip_hmatrix(self):
        rng = np.random.default_rng()
        h = rng.normal(size=(2, 2, 1))
        network = sp.TwoPortNetwork.from_hmatrix(h)
        np.testing.assert_allclose(network.hmatrix, h)

        # Check that we can convert back to a TwoPortNetwork
        network2 = sp.TwoPortNetwork.from_hmatrix(network.hmatrix)
        np.testing.assert_allclose(network2.hybrid_matrix, h)

    def test_roundtrip_abcd(self):
        rng = np.random.default_rng()
        abcd = rng.normal(size=(2, 2, 1))
        network = sp.TwoPortNetwork.from_abcd(abcd)
        np.testing.assert_allclose(network.x, abcd)

    def test_aliases(self):
        rng = np.random.default_rng()
        z = rng.normal(size=(2, 2, 1))
        network = sp.TwoPortNetwork.from_zmatrix(z)
        assert np.allclose(network.A, network.x[0, 0])
        assert np.allclose(network.B, network.x[0, 1])
        assert np.allclose(network.C, network.x[1, 0])
        assert np.allclose(network.D, network.x[1, 1])

    def test_reciprocity(self):
        network = sp.TwoPortNetwork([[1, 0], [0, 1]])
        assert network.is_reciprocal()
        assert network.is_symmetric()

        non_reciprocal = sp.TwoPortNetwork([[1, 1], [1, 1]])
        assert not non_reciprocal.is_reciprocal()
        assert network.is_symmetric()

    def test_lossless(self):
        network = sp.TwoPortNetwork([[1, 0], [0, 1]])
        assert network.is_lossless()

        lossy = sp.TwoPortNetwork([[1, 0], [0, 1 + 0.1j]])
        assert not lossy.is_lossless()

    @pytest.mark.parametrize(
        "addfunc",
        ["add_in_series", "add_in_parallel", "add_in_series_parallel", "cascade_with"],
    )
    def add_bad(self, addfunc):
        network1 = sp.TwoPortNetwork([[1, 0], [0, 1]])

        fnc = getattr(network1, addfunc)
        with pytest.raises(ValueError, match="Two matrices must be of the same type"):
            fnc([[0, 1], [1, 0]])

        with pytest.raises(
            ValueError, match="Two matrices must have the same dimensions"
        ):
            fnc(np.array([[[1, 0], [0, 1]], [[1, 0], [0, 1 + 0.1j]]]))

    def test_add_in_series(self):
        network1 = sp.TwoPortNetwork.from_zmatrix([[1, 1], [1, 1]])
        network2 = sp.TwoPortNetwork.from_zmatrix([[1, 1], [1, 1 + 0.1j]])

        combined = network1.add_in_series(network2)
        expected = sp.TwoPortNetwork.from_zmatrix([[2, 2], [2, 2 + 0.1j]])

        assert combined == expected

    def test_add_in_parallel(self):
        network1 = sp.TwoPortNetwork.from_ymatrix([[1, 1], [1, 1]])
        network2 = sp.TwoPortNetwork.from_ymatrix([[1, 1], [1, 1 + 0.1j]])

        combined = network1.add_in_parallel(network2)
        expected = sp.TwoPortNetwork.from_ymatrix([[2, 2], [2, 2 + 0.1j]])

        assert combined == expected

    def test_add_in_series_parallel(self):
        network1 = sp.TwoPortNetwork.from_hmatrix([[1, 1], [1, 1]])
        network2 = sp.TwoPortNetwork.from_hmatrix([[1, 1], [1, 1 + 0.1j]])

        combined = network1.add_in_series_parallel(network2)
        expected = sp.TwoPortNetwork.from_hmatrix([[2, 2], [2, 2 + 0.1j]])

        assert combined == expected

    def test_cascade_with(self):
        network1 = sp.TwoPortNetwork([[1, 0], [0, 1]])
        network2 = sp.TwoPortNetwork([[1, 0], [0, 1]])

        combined = network1.cascade_with(network2)
        expected = sp.TwoPortNetwork([[1, 0], [0, 1]])

        assert combined == expected

    def test_roundtrip_sparams(self):
        rng = np.random.default_rng()
        freqs = np.linspace(50, 100, 10) * un.MHz
        s11 = rng.uniform(0, 1, size=10) + rng.uniform(0, 1, size=10) * 1j
        s12 = rng.uniform(0, 1, size=10) + rng.uniform(0, 1, size=10) * 1j

        sparams = sp.SParams(s11=s11, s12=s12, freqs=freqs)
        network = sp.TwoPortNetwork.from_smatrix(sparams, z0=50 * un.Ohm)
        out_sp = network.as_sparams(freqs=freqs, source_impedance=50)
        assert out_sp == sparams

    def test_from_transmission_line(self):
        line = KNOWN_CABLES["balun-tube"].as_transmission_line(freqs=50 * un.MHz)
        network = sp.TwoPortNetwork.from_transmission_line(line, length=1 * un.m)

        assert network.is_reciprocal()
        assert network.is_symmetric()


class TestSMatrix:
    def setup_class(self):
        rng = np.random.default_rng()
        s11 = rng.uniform(0, 1, size=10) + rng.uniform(0, 1, size=10) * 1j
        s12 = rng.uniform(0, 1, size=10) + rng.uniform(0, 1, size=10) * 1j
        freqs = np.linspace(50, 100, 10) * un.MHz

        self.smatrix = sp.SParams(s11=s11, s12=s12, freqs=freqs)

    def test_from_sparams(self):
        assert np.allclose(self.smatrix.s21, self.smatrix.s12)
        assert np.allclose(self.smatrix.s22, self.smatrix.s11)

    def test_roundtrip_transfer_matrix(self):
        transfer_matrix = self.smatrix.as_transfer_matrix()

        assert transfer_matrix.shape == (2, 2, len(self.smatrix.s11))

        new_smatrix = sp.SParams.from_transfer_matrix(
            self.smatrix.freqs, transfer_matrix
        )
        assert np.allclose(new_smatrix.s11, self.smatrix.s11)
        assert np.allclose(new_smatrix.s12, self.smatrix.s12)
        assert np.allclose(new_smatrix.s21, self.smatrix.s21)
        assert np.allclose(new_smatrix.s22, self.smatrix.s22)

    def test_cascade(self):
        new = self.smatrix.cascade_with(self.smatrix)
        assert isinstance(new, sp.SParams)
        assert new.s11.shape == self.smatrix.s11.shape

    def test_reciprocal(self):
        assert self.smatrix.is_reciprocal()

    def test_lossless(self):
        freqs = np.linspace(50, 100, 10) * un.MHz
        lossless_smatrix = sp.SParams(
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
        line = KNOWN_CABLES["balun-tube"].as_transmission_line(
            freqs=np.array([50]) * un.MHz
        )
        smatrix = line.scattering_parameters(
            line_length=1 * un.m
        )  # sp.SParams.from_transmission_line(line, length=1 * un.m)

        assert smatrix.is_reciprocal()

    def test_from_calkit_and_vna(self):
        calkit = sp.AGILENT_85033E
        freq = np.linspace(50, 100, 10) * un.MHz
        vna = CalkitReadings(
            open=sp.ReflectionCoefficient(
                reflection_coefficient=np.array([1 + 0j] * 10), freqs=freq
            ),
            short=sp.ReflectionCoefficient(
                reflection_coefficient=np.array([-1 + 0j] * 10), freqs=freq
            ),
            match=sp.ReflectionCoefficient(
                reflection_coefficient=np.array([0 + 0j] * 10), freqs=freq
            ),
        )

        smatrix = sp.SParams.from_calkit_measurements(model=calkit, measurements=vna)
        assert smatrix.is_reciprocal()
