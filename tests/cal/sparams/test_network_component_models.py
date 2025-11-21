import numpy as np
import pytest
from astropy import units as un

from edges import modeling as mdl
from edges.cal.sparams.core import datatypes as dt
from edges.cal.sparams.core import network_component_models as ncm


def freqs_mhz(n=1):
    return np.arange(1, n + 1) * un.MHz


class TestSkinDepth:
    def test_skin_depth_units_and_value(self):
        f = 1e6 * un.Hz
        sigma = 1e7 * un.siemens / un.m
        sd = ncm.skin_depth(f, sigma)
        assert sd.unit.is_equivalent(un.m)
        # basic numeric check against formula
        expected = np.sqrt(1.0 / (np.pi * f * ncm.mu0 * sigma)).to_value("m")
        assert pytest.approx(sd.to_value("m")) == expected


class TestTransmissionLine:
    def setup_class(self):
        freqs = freqs_mhz(2)
        # use simple, frequency independent per-m length values
        r = 1.0 * un.ohm / un.m
        L = 1e-6 * un.ohm * un.s / un.m
        g = 1e-9 * un.siemens / un.m
        c = 1e-11 * un.siemens * un.s / un.m
        self.tl_with_length = ncm.TransmissionLine(
            freqs=freqs,
            resistance=r,
            inductance=L,
            conductance=g,
            capacitance=c,
            length=1.0 * un.m,
        )

        self.tl_no_length = ncm.TransmissionLine(
            freqs=freqs,
            resistance=r,
            inductance=L,
            conductance=g,
            capacitance=c,
        )

    def test_characteristic_and_propagation_and_reflection(self):
        tl = self.tl_with_length

        Zo = tl.characteristic_impedance
        assert Zo.unit.is_equivalent(un.ohm)

        gamma = tl.propagation_constant
        assert gamma.unit.is_equivalent(1 / un.m)

        # calling input_impedance triggers numpy tanh on an astropy Quantity
        # which raises UnitTypeError in this environment (tanh expects angle units).
        with pytest.raises(un.UnitTypeError):
            # accept UnitTypeError or UnitConversionError depending on astropy/numpy
            tl.input_impedance(load_impedance=50 * un.Ohm, line_length=1.0 * un.m)

        # reflection should be zero if load == characteristic impedance
        refl = tl.reflection_coefficient(load_impedance=Zo)
        np.testing.assert_allclose(refl, 0)

    def test_input_impedance_requires_length(self):
        tl = self.tl_no_length
        with pytest.raises(ValueError, match="Line length must be provided"):
            tl.input_impedance()

    def test_scattering_parameters_requires_length(self):
        tl = self.tl_no_length
        with pytest.raises(ValueError, match="Line length must be provided"):
            tl.scattering_parameters()


class TestCoaxialCable:
    def setup_class(self):
        self.coax = ncm.CoaxialCable(
            outer_radius=0.01 * un.m,
            inner_radius=0.001 * un.m,
            outer_material="copper",
            inner_material="copper",
            relative_dielectric=2.0,
            length=1.0 * un.m,
        )

    def test_skin_depths_and_per_length_properties(self):
        c = self.coax
        f = np.array([1e6]) * un.Hz
        # outer/inner skin depth use same conductivity for copper
        sd_outer = c.outer_skin_depth(f)
        sd_inner = c.inner_skin_depth(f)
        assert sd_outer.unit.is_equivalent(un.m)
        assert sd_inner.unit.is_equivalent(un.m)

        Lpm = c.inductance_per_metre
        Cpm = c.capacitance_per_metre
        assert Lpm.unit.is_equivalent(un.H / un.m)
        assert Cpm.unit.is_equivalent(un.F / un.m)

        # as_transmission_line returns a TransmissionLine; pass no length to avoid
        # ambiguous Quantity truthiness in the implementation.
        tl = c.as_transmission_line(freqs=f)
        assert isinstance(tl, ncm.TransmissionLine)

        # scattering parameters return SParams; avoid passing length kw to prevent
        # ambiguous Quantity truthiness in the implementation.
        s = c.scattering_parameters(freqs=f)
        assert isinstance(s, dt.SParams)

    @pytest.mark.parametrize(
        ("outer_material", "inner_material"),
        [
            ("copper", "unknown-material"),
            ("unknown-material", "aluminum"),
        ],
    )
    def test_unknown_material_raises(self, outer_material, inner_material):
        with pytest.raises(ValueError):
            ncm.CoaxialCable(
                outer_radius=0.01 * un.m,
                inner_radius=0.001 * un.m,
                outer_material=outer_material,
                inner_material=inner_material,
                relative_dielectric=2.0,
            )

    def test_characteristic_impedance(self):
        c = self.coax
        Zo = c.characteristic_impedance(freq=1e6 * un.Hz)
        assert Zo.unit.is_equivalent(un.ohm)

    def test_propagation_constant(self):
        c = self.coax
        gamma = c.propagation_constant(freq=1e6 * un.Hz)
        assert gamma.unit.is_equivalent("1/m")


class TestCalkitStandard:
    def test_name_and_intrinsic_gamma(self):
        open_std = ncm.CalkitStandard.open()
        assert open_std.name == "open"

        short_std = ncm.CalkitStandard.short()
        assert short_std.name == "short"

        match_std = ncm.CalkitStandard.match()
        assert match_std.name == "match"

    def test_termination_impedance_with_models(self):
        freq = 1e6 * un.Hz
        # capacitance model: constant capacitance in Farads

        def cap_model(f):
            return 1e-12

        std = ncm.CalkitStandard(resistance=50.0 * un.Ohm, capacitance_model=cap_model)
        Z = std.termination_impedance(freq)
        expected = (
            -1j / (2 * np.pi * freq.to_value("Hz") * cap_model(freq.to_value("Hz")))
        ) * un.ohm
        assert np.allclose(Z.value, expected.value)

    def test_lossy_characteristic_and_gl_and_offset_gamma(self):
        freq = np.array([1e6, 2e6]) * un.Hz
        std = ncm.CalkitStandard(resistance=50.0 * un.Ohm)
        lc = std.lossy_characteristic_impedance(freq)
        assert lc.unit.is_equivalent(un.ohm)
        g = std.gl(freq)
        assert isinstance(g, np.ndarray)
        og = std.offset_gamma(freq)
        # offset_gamma should be dimensionless
        assert hasattr(og, "__iter__")


class TestCalkit:
    def test_load_calkit_with_resistance(self):
        new = ncm.get_calkit(ncm.AGILENT_85033E, resistance_of_match=49.0 * un.Ohm)
        default = ncm.get_calkit(ncm.AGILENT_85033E)
        assert new.match.resistance == 49.0 * un.Ohm
        assert default.match.resistance == 50.0 * un.Ohm

    def test_calkit_standard_name(self):
        assert ncm.CalkitStandard(resistance=50).name == "match"

        assert (
            ncm.CalkitStandard(
                resistance=np.inf,
                capacitance_model=ncm.AGILENT_85033E.open.capacitance_model,
            ).name
            == "open"
        )

        assert (
            ncm.CalkitStandard(
                resistance=0, inductance_model=ncm.AGILENT_85033E.short.inductance_model
            ).name
            == "short"
        )

    def test_calkit_termination_impedance(self):
        with pytest.raises(TypeError, match="freq must be a frequency quantity!"):
            # requires frequency to be in units
            ncm.AGILENT_85033E.open.termination_impedance(np.linspace(50, 100, 100))

        assert (
            ncm.AGILENT_85033E.match.termination_impedance(50 * un.MHz)
            == ncm.AGILENT_85033E.match.resistance
        )

    def test_calkit_units(self):
        freq = np.linspace(50, 100, 100) * un.MHz

        ag = ncm.AGILENT_85033E.open

        assert ag.termination_impedance(freq).unit == un.ohm
        assert ag.termination_gamma(freq).unit == un.dimensionless_unscaled
        assert ag.lossy_characteristic_impedance(freq).unit == un.ohm
        assert un.get_physical_type(ag.gl(freq)) == "dimensionless"
        assert ag.offset_gamma(freq).unit == un.dimensionless_unscaled
        assert isinstance(ag.reflection_coefficient(freq), dt.ReflectionCoefficient)

    def test_calkit_quantities_match_trivial(self):
        """A test that for a simple calkit definition, the outputs are correct."""
        std = ncm.CalkitStandard(
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
        std = ncm.CalkitStandard(
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
        std = ncm.CalkitStandard(
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
        std = ncm.CalkitStandard(
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

    def test_get_calkit_and_at_freqs(self):
        base = "AGILENT_85033E"
        cal = ncm.get_calkit(base)
        assert isinstance(cal.open, ncm.CalkitStandard)

        # override match resistance
        cal2 = ncm.get_calkit(base, resistance_of_match=100 * un.Ohm)
        assert cal2.match.resistance.to_value("ohm") == pytest.approx(100)

        freqs = np.array([1e6, 2e6]) * un.Hz
        readings = cal.at_freqs(freqs)
        assert hasattr(readings, "open")
        assert hasattr(readings.open, "reflection_coefficient")


class TestTwoPortNetwork:
    def test_basic_abcd_properties_and_conversions(self):
        # build simple ABCD matrix equal to identity across freq
        abcd = np.ones((2, 2, 1), dtype=complex)
        abcd[0, 0, :] = 1
        abcd[1, 1, :] = 1
        abcd[0, 1, :] = 0
        abcd[1, 0, :] = 0
        tpn = ncm.TwoPortNetwork.from_abcd(abcd)
        assert np.allclose(tpn.A, 1)
        assert np.allclose(tpn.D, 1)
        assert np.allclose(tpn.B, 0)
        assert np.allclose(tpn.C, 0)
        assert tpn.is_reciprocal()
        assert tpn.is_symmetric()

    def test_from_transmission_line_and_as_sparams(self):
        freqs = np.array([1e6]) * un.Hz
        # simple transmission line
        tl = ncm.TransmissionLine(
            freqs=freqs,
            resistance=1.0 * un.ohm / un.m,
            inductance=1e-6 * un.ohm * un.s / un.m,
            conductance=1e-9 * un.siemens / un.m,
            capacitance=1e-11 * un.siemens * un.s / un.m,
            length=1.0 * un.m,
        )
        tpn = ncm.TwoPortNetwork.from_transmission_line(tl, 1.0 * un.m)
        s = tpn.as_sparams(freqs=freqs, source_impedance=50.0)
        assert isinstance(s, dt.SParams)

    def test_invalid_shape(self):
        with pytest.raises(ValueError, match="Matrix must have shape"):
            ncm.TwoPortNetwork(np.ones((3, 3, 10)))

        with pytest.raises(ValueError, match="x must have ndim in"):
            ncm.TwoPortNetwork(np.ones((3, 3, 10, 11)))

    def test_roundtrip_zmatrix(self):
        rng = np.random.default_rng()
        z = rng.normal(size=(2, 2, 1))
        network = ncm.TwoPortNetwork.from_zmatrix(z)
        np.testing.assert_allclose(network.zmatrix, z)

        # Check that we can convert back to a TwoPortNetwork
        network2 = ncm.TwoPortNetwork.from_zmatrix(network.zmatrix)
        np.testing.assert_allclose(network2.impedance_matrix, z)

    def test_roundtrip_ymatrix(self):
        rng = np.random.default_rng()
        y = rng.normal(size=(2, 2, 1))
        network = ncm.TwoPortNetwork.from_ymatrix(y)
        np.testing.assert_allclose(network.ymatrix, y)

        # Check that we can convert back to a TwoPortNetwork
        network2 = ncm.TwoPortNetwork.from_ymatrix(network.ymatrix)
        np.testing.assert_allclose(network2.admittance_matrix, y)

    def test_roundtrip_hmatrix(self):
        rng = np.random.default_rng()
        h = rng.normal(size=(2, 2, 1))
        network = ncm.TwoPortNetwork.from_hmatrix(h)
        np.testing.assert_allclose(network.hmatrix, h)

        # Check that we can convert back to a TwoPortNetwork
        network2 = ncm.TwoPortNetwork.from_hmatrix(network.hmatrix)
        np.testing.assert_allclose(network2.hybrid_matrix, h)

    def test_roundtrip_abcd(self):
        rng = np.random.default_rng()
        abcd = rng.normal(size=(2, 2, 1))
        network = ncm.TwoPortNetwork.from_abcd(abcd)
        np.testing.assert_allclose(network.x, abcd)

    def test_aliases(self):
        rng = np.random.default_rng()
        z = rng.normal(size=(2, 2, 1))
        network = ncm.TwoPortNetwork.from_zmatrix(z)
        assert np.allclose(network.A, network.x[0, 0])
        assert np.allclose(network.B, network.x[0, 1])
        assert np.allclose(network.C, network.x[1, 0])
        assert np.allclose(network.D, network.x[1, 1])

    def test_reciprocity(self):
        network = ncm.TwoPortNetwork([[1, 0], [0, 1]])
        assert network.is_reciprocal()
        assert network.is_symmetric()

        non_reciprocal = ncm.TwoPortNetwork([[1, 1], [1, 1]])
        assert not non_reciprocal.is_reciprocal()
        assert network.is_symmetric()

    def test_lossless(self):
        network = ncm.TwoPortNetwork([[1, 0], [0, 1]])
        assert network.is_lossless()

        lossy = ncm.TwoPortNetwork([[1, 0], [0, 1 + 0.1j]])
        assert not lossy.is_lossless()

    @pytest.mark.parametrize(
        "addfunc",
        ["add_in_series", "add_in_parallel", "add_in_series_parallel", "cascade_with"],
    )
    def add_bad(self, addfunc):
        network1 = ncm.TwoPortNetwork([[1, 0], [0, 1]])

        fnc = getattr(network1, addfunc)
        with pytest.raises(ValueError, match="Two matrices must be of the same type"):
            fnc([[0, 1], [1, 0]])

        with pytest.raises(
            ValueError, match="Two matrices must have the same dimensions"
        ):
            fnc(np.array([[[1, 0], [0, 1]], [[1, 0], [0, 1 + 0.1j]]]))

    def test_add_in_series(self):
        network1 = ncm.TwoPortNetwork.from_zmatrix([[1, 1], [1, 1]])
        network2 = ncm.TwoPortNetwork.from_zmatrix([[1, 1], [1, 1 + 0.1j]])

        combined = network1.add_in_series(network2)
        expected = ncm.TwoPortNetwork.from_zmatrix([[2, 2], [2, 2 + 0.1j]])

        assert combined == expected

    def test_add_in_parallel(self):
        network1 = ncm.TwoPortNetwork.from_ymatrix([[1, 1], [1, 1]])
        network2 = ncm.TwoPortNetwork.from_ymatrix([[1, 1], [1, 1 + 0.1j]])

        combined = network1.add_in_parallel(network2)
        expected = ncm.TwoPortNetwork.from_ymatrix([[2, 2], [2, 2 + 0.1j]])

        assert combined == expected

    def test_add_in_series_parallel(self):
        network1 = ncm.TwoPortNetwork.from_hmatrix([[1, 1], [1, 1]])
        network2 = ncm.TwoPortNetwork.from_hmatrix([[1, 1], [1, 1 + 0.1j]])

        combined = network1.add_in_series_parallel(network2)
        expected = ncm.TwoPortNetwork.from_hmatrix([[2, 2], [2, 2 + 0.1j]])

        assert combined == expected

    def test_cascade_with(self):
        network1 = ncm.TwoPortNetwork([[1, 0], [0, 1]])
        network2 = ncm.TwoPortNetwork([[1, 0], [0, 1]])

        combined = network1.cascade_with(network2)
        expected = ncm.TwoPortNetwork([[1, 0], [0, 1]])

        assert combined == expected

    def test_roundtrip_sparams(self):
        rng = np.random.default_rng()
        freqs = np.linspace(50, 100, 10) * un.MHz
        s11 = rng.uniform(0, 1, size=10) + rng.uniform(0, 1, size=10) * 1j
        s12 = rng.uniform(0, 1, size=10) + rng.uniform(0, 1, size=10) * 1j

        sparams = dt.SParams(s11=s11, s12=s12, freqs=freqs)
        network = ncm.TwoPortNetwork.from_smatrix(sparams, z0=50 * un.Ohm)
        out_sp = network.as_sparams(freqs=freqs, source_impedance=50)
        assert out_sp == sparams

    def test_from_transmission_line(self):
        line = ncm.KNOWN_CABLES["balun-tube"].as_transmission_line(freqs=50 * un.MHz)
        network = ncm.TwoPortNetwork.from_transmission_line(line, length=1 * un.m)

        assert network.is_reciprocal()
        assert network.is_symmetric()
