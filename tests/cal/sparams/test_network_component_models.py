import attrs
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

        zin = tl.input_impedance(load_impedance=50 * un.Ohm, line_length=1.0 * un.m)
        assert zin.unit.is_equivalent(un.ohm)
        assert zin.shape == tl.freqs.shape

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
    def test_add_bad(self, addfunc):
        network1 = ncm.TwoPortNetwork([[1, 0], [0, 1]])

        fnc = getattr(network1, addfunc)
        with pytest.raises(TypeError, match="Two matrices must be of the same type"):
            fnc([[0, 1], [1, 0]])

        # A network with a different number of frequencies.
        network2 = ncm.TwoPortNetwork(np.ones((2, 2, 3)))
        with pytest.raises(
            ValueError, match="Two matrices must have the same dimensions"
        ):
            fnc(network2)

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


def _lossless_line(
    z0: float = 50.0, freqs: un.Quantity | None = None, length=1.0 * un.m
) -> ncm.TransmissionLine:
    """A lossless line with phase velocity 2e8 m/s and the given impedance."""
    if freqs is None:
        freqs = np.linspace(40, 200, 33) * un.MHz
    v = 2e8  # m/s
    return ncm.TransmissionLine(
        freqs=freqs.to("Hz"),
        resistance=0 * un.ohm / un.m,
        inductance=z0 / v * un.ohm * un.s / un.m,
        conductance=0 * un.siemens / un.m,
        capacitance=1 / (z0 * v) * un.siemens * un.s / un.m,
        length=length,
    )


class TestTransmissionLinePhysics:
    """Physical limits and invariants of the TransmissionLine model."""

    def test_lossless_characteristic_impedance(self):
        tl = _lossless_line(z0=61.0)
        expected = np.sqrt(tl.inductance / tl.capacitance).to_value("ohm")
        np.testing.assert_allclose(
            tl.characteristic_impedance.to_value("ohm"), expected, rtol=1e-12
        )
        np.testing.assert_allclose(expected, 61.0, rtol=1e-12)

    def test_lossless_propagation_is_pure_phase(self):
        tl = _lossless_line()
        gamma = tl.propagation_constant.to_value("1/m")
        np.testing.assert_allclose(gamma.real, 0, atol=1e-15)
        # The phase constant is omega / v.
        np.testing.assert_allclose(
            gamma.imag, 2 * np.pi * tl.freqs.to_value("Hz") / 2e8, rtol=1e-12
        )

    @pytest.mark.parametrize("z0", [50.0, 61.0, 75.0])
    def test_lossless_line_unitary(self, z0):
        tl = _lossless_line(z0=z0, length=0.37 * un.m)
        s = tl.scattering_parameters(load_impedance=50 * un.ohm)
        np.testing.assert_allclose(np.abs(s.s12), np.abs(s.s21), rtol=1e-12)
        np.testing.assert_allclose(
            np.abs(s.s11) ** 2 + np.abs(s.s21) ** 2, 1.0, rtol=1e-12
        )
        assert s.is_lossless()
        assert s.is_reciprocal()
        if z0 == 50.0:
            np.testing.assert_allclose(np.abs(s.s21), 1.0, rtol=1e-12)
            np.testing.assert_allclose(s.s11, 0, atol=1e-14)

    def test_zero_length_is_identity(self):
        cable = ncm.KNOWN_CABLES["UT-141C-SP"]
        tl = cable.as_transmission_line(np.linspace(40, 200, 17) * un.MHz)
        s = tl.scattering_parameters(line_length=0 * un.m)
        np.testing.assert_allclose(s.s11, 0, atol=1e-15)
        np.testing.assert_allclose(s.s22, 0, atol=1e-15)
        np.testing.assert_allclose(s.s12, 1, rtol=1e-14)
        np.testing.assert_allclose(s.s21, 1, rtol=1e-14)

    def test_quarter_wave_transformer(self):
        """A lossless lambda/4 line has Zin = Z0^2 / ZL."""
        freq = np.array([100.0]) * un.MHz
        # lambda = v / f = 2 m, so lambda / 4 = 0.5 m.
        tl = _lossless_line(z0=50.0, freqs=freq, length=0.5 * un.m)
        for zl in [25.0, 100.0, 10.0 + 30j]:
            zin = tl.input_impedance(load_impedance=zl * un.ohm)
            np.testing.assert_allclose(zin.to_value("ohm"), 50.0**2 / zl, rtol=1e-9)

    def test_matched_load_input_impedance(self):
        """A line terminated in its own characteristic impedance has Zin = Z0."""
        cable = ncm.KNOWN_CABLES["UT-141C-SP"]
        tl = cable.as_transmission_line(np.linspace(40, 200, 17) * un.MHz)
        z0 = tl.characteristic_impedance
        zin = tl.input_impedance(load_impedance=z0, line_length=0.7 * un.m)
        np.testing.assert_allclose(zin.to_value("ohm"), z0.to_value("ohm"), rtol=1e-12)

    def test_input_impedance_matches_sparams(self):
        """Zin from the line formula agrees with embedding Gamma_L through S."""
        cable = ncm.KNOWN_CABLES["UT-141C-SP"]
        freqs = np.linspace(40, 200, 17) * un.MHz
        tl = cable.as_transmission_line(freqs)
        zin = tl.input_impedance(load_impedance=20.0 * un.ohm, line_length=0.3 * un.m)
        gamma_in = ncm.ee.impedance2gamma(zin.to_value("ohm"), 50.0)

        s = tl.scattering_parameters(line_length=0.3 * un.m)
        gamma_l = dt.ReflectionCoefficient(
            freqs=freqs,
            reflection_coefficient=np.full(
                len(freqs), ncm.ee.impedance2gamma(20.0, 50.0), dtype=complex
            ),
        )
        np.testing.assert_allclose(
            gamma_l.embed(s).reflection_coefficient, gamma_in, rtol=1e-10
        )


class TestCoaxialCablePhysics:
    def setup_class(self):
        self.coax = ncm.CoaxialCable(
            outer_radius=0.01 * un.m,
            inner_radius=0.001 * un.m,
            outer_material="copper",
            inner_material="copper",
            relative_dielectric=2.0,
            relative_conductance_interior=0,  # zero dielectric loss tangent
        )

    def test_attenuation_scales_as_sqrt_freq(self):
        """With G = 0 the loss is due to the skin effect only, so alpha ~ sqrt(f)."""
        f = np.array([1e7, 4e7, 1.6e8, 6.4e8]) * un.Hz
        alpha = self.coax.propagation_constant(f).real.to_value("1/m")
        np.testing.assert_allclose(alpha[1:] / alpha[:-1], 2.0, rtol=5e-3)

    def test_characteristic_impedance_matches_coax_formula(self):
        """Z0 ~ eta0 / (2 pi sqrt(eps_r)) ln(b/a), plus a small internal-L term."""
        f = np.array([1e7, 1e8, 1e9]) * un.Hz
        z0 = self.coax.characteristic_impedance(f).to_value("ohm")
        eta0 = np.sqrt(ncm.mu0 / ncm.eps0).to_value("ohm")
        expected = eta0 / (2 * np.pi * np.sqrt(2.0)) * np.log(10.0)
        np.testing.assert_allclose(z0.real, expected, rtol=1e-2)
        # The internal inductance only ever increases Z0, and its effect falls
        # with frequency as the skin depth shrinks.
        assert np.all(z0.real > expected)
        assert np.all(np.diff(z0.real - expected) < 0)

    def test_methods_accept_length(self):
        """Regression test: passing ``length=`` must not take a Quantity's truth."""
        f = np.linspace(50, 100, 5) * un.MHz
        length = 0.5 * un.m

        tl = self.coax.as_transmission_line(f, length=length)
        assert tl.length == length

        np.testing.assert_allclose(
            self.coax.characteristic_impedance(f, length=length),
            self.coax.characteristic_impedance(f),
        )
        np.testing.assert_allclose(
            self.coax.propagation_constant(f, length=length),
            self.coax.propagation_constant(f),
        )
        s = self.coax.scattering_parameters(f, length=length)
        expected = self.coax.as_transmission_line(f).scattering_parameters(
            line_length=length
        )
        np.testing.assert_allclose(s.s, expected.s)

    def test_length_defaults_to_instance_length(self):
        coax = attrs.evolve(self.coax, length=0.25 * un.m)
        f = np.linspace(50, 100, 5) * un.MHz
        assert coax.as_transmission_line(f).length == 0.25 * un.m
        assert coax.as_transmission_line(f, length=1 * un.m).length == 1 * un.m


class TestCalkitStandardPhysics:
    freqs = np.linspace(40, 200, 161) * un.MHz

    def test_ideal_short_is_minus_one(self):
        std = ncm.CalkitStandard.short(offset_delay=0 * un.ps, offset_loss=0)
        np.testing.assert_allclose(
            std.reflection_coefficient(self.freqs).reflection_coefficient,
            -1,
            atol=1e-15,
        )

    def test_ideal_open_is_plus_one(self):
        """Regression test: an open with R=inf and no capacitance gives +1."""
        std = ncm.CalkitStandard.open(offset_delay=0 * un.ps, offset_loss=0)
        assert std.name == "open"
        gamma = std.reflection_coefficient(self.freqs).reflection_coefficient
        np.testing.assert_allclose(gamma, 1, atol=1e-15)

    @pytest.mark.parametrize("delay", [0, 10, 31.785, 100, 1000])
    @pytest.mark.parametrize("kind", ["open", "short"])
    def test_lossless_standard_has_unit_magnitude(self, delay, kind):
        std = getattr(ncm.CalkitStandard, kind)(
            offset_delay=delay * un.ps, offset_loss=0
        )
        gamma = std.reflection_coefficient(self.freqs).reflection_coefficient
        np.testing.assert_allclose(np.abs(gamma), 1, rtol=1e-12)

        # A pure offset delay just rotates the phase: Gamma = Gamma_T exp(-2j w tau)
        expected = (1 if kind == "open" else -1) * np.exp(
            -2j * np.pi * 2 * (self.freqs * delay * un.ps).to_value("")
        )
        np.testing.assert_allclose(gamma, expected, rtol=1e-12)

    @pytest.mark.parametrize("kind", ["open", "short"])
    def test_lossless_reactive_standard_has_unit_magnitude(self, kind):
        """The 85033E open/short are pure reactances, so lossless means |G| = 1."""
        std = attrs.evolve(getattr(ncm.AGILENT_85033E, kind), offset_loss=0)
        gamma = std.reflection_coefficient(self.freqs).reflection_coefficient
        np.testing.assert_allclose(np.abs(gamma), 1, rtol=1e-12)

    @pytest.mark.parametrize("kind", ["open", "short", "match"])
    def test_85033e_passive(self, kind):
        std = getattr(ncm.AGILENT_85033E, kind)
        gamma = std.reflection_coefficient(self.freqs).reflection_coefficient
        assert np.all(np.isfinite(gamma))
        assert np.all(np.abs(gamma) <= 1)


class TestCalkitValidation:
    def test_wrong_standard_raises_value_error(self):
        kit = ncm.AGILENT_85033E
        with pytest.raises(ValueError, match="open"):
            ncm.Calkit(open=kit.short, short=kit.short, match=kit.match)
        with pytest.raises(ValueError, match="short"):
            ncm.Calkit(open=kit.open, short=kit.match, match=kit.match)
        with pytest.raises(ValueError, match="match"):
            ncm.Calkit(open=kit.open, short=kit.short, match=kit.open)

    def test_get_calkit_does_not_mutate_input(self):
        match = {"offset_delay": 30 * un.ps}
        kit = ncm.get_calkit(
            "AGILENT_85033E", resistance_of_match=49 * un.ohm, match=match
        )
        assert match == {"offset_delay": 30 * un.ps}
        assert kit.match.resistance == 49 * un.ohm
        assert kit.match.offset_delay == 30 * un.ps


class TestTwoPortNetworkPhysics:
    freqs = np.linspace(40, 200, 9) * un.MHz

    def _lossless_network(self, z0=60.0, theta=1.0):
        n = len(self.freqs)
        c, s = np.cos(theta) * np.ones(n), np.sin(theta) * np.ones(n)
        return ncm.TwoPortNetwork(np.array([[c, 1j * z0 * s], [1j * s / z0, c]]))

    def test_transmission_line_matches_sparams(self):
        """ABCD of a line converted to S agrees with the direct S formula."""
        cable = ncm.KNOWN_CABLES["UT-141C-SP"]
        tl = cable.as_transmission_line(self.freqs)
        tpn = ncm.TwoPortNetwork.from_transmission_line(tl, 0.3 * un.m)
        s1 = tpn.as_sparams(self.freqs, 50.0)
        s2 = tl.scattering_parameters(line_length=0.3 * un.m)
        np.testing.assert_allclose(s1.s, s2.s, rtol=1e-10)

    def test_lossless_equal_impedances_unitary(self):
        s = self._lossless_network().as_sparams(self.freqs, 50.0)
        assert s.is_reciprocal()
        assert s.is_lossless()

    def test_lossless_unequal_impedances_unitary(self):
        """With zs != zl a reciprocal lossless network still has a unitary S."""
        s = self._lossless_network().as_sparams(
            self.freqs, source_impedance=50.0, load_impedance=75.0
        )
        np.testing.assert_allclose(s.s12, s.s21, rtol=1e-12)
        np.testing.assert_allclose(
            np.abs(s.s11) ** 2 + np.abs(s.s21) ** 2, 1, rtol=1e-12
        )
        assert s.is_lossless()

    def test_unequal_impedances_roundtrip_through_reference_change(self):
        """S with (zs, zl) equals the 50-ohm S renormalised to the new references."""
        rng = np.random.default_rng(7)
        n = len(self.freqs)
        abcd = rng.normal(size=(2, 2, n)) + 1j * rng.normal(size=(2, 2, n))
        abcd[0, 1] *= 50
        abcd[1, 0] /= 50
        net = ncm.TwoPortNetwork(abcd)
        zs, zl = 50.0, 75.0
        s = net.as_sparams(self.freqs, source_impedance=zs, load_impedance=zl)

        # Independent route: Z-matrix -> S = F (Z - R)(Z + R)^-1 F^-1 with
        # F = diag(1 / (2 sqrt(R))), valid for real reference impedances.
        z = np.moveaxis(net.zmatrix, -1, 0)
        r = np.diag([zs, zl])
        f = np.diag(1 / (2 * np.sqrt([zs, zl])))
        expected = f @ (z - r) @ np.linalg.inv(z + r) @ np.linalg.inv(f)
        np.testing.assert_allclose(s.s11, expected[:, 0, 0], rtol=1e-10)
        np.testing.assert_allclose(s.s12, expected[:, 0, 1], rtol=1e-10)
        np.testing.assert_allclose(s.s21, expected[:, 1, 0], rtol=1e-10)
        np.testing.assert_allclose(s.s22, expected[:, 1, 1], rtol=1e-10)

    def test_quantity_impedances(self):
        net = self._lossless_network()
        s1 = net.as_sparams(self.freqs, 50.0, 75.0)
        s2 = net.as_sparams(self.freqs, 50 * un.ohm, 0.075 * un.kohm)
        np.testing.assert_allclose(s1.s, s2.s, rtol=1e-14)

    @pytest.mark.parametrize("bad", [50 + 5j, -50.0, 0.0])
    def test_bad_reference_impedance(self, bad):
        with pytest.raises(ValueError, match="source_impedance must be"):
            self._lossless_network().as_sparams(self.freqs, source_impedance=bad)
        with pytest.raises(ValueError, match="load_impedance must be"):
            self._lossless_network().as_sparams(
                self.freqs, source_impedance=50.0, load_impedance=bad
            )

    @pytest.mark.parametrize("z0", [50.0, 50 * un.ohm, 0.05 * un.kohm])
    def test_from_smatrix_z0_units(self, z0):
        """Regression test: a z0 Quantity is converted to ohms, not stripped."""
        rng = np.random.default_rng(42)
        n = len(self.freqs)
        s = dt.SParams(
            freqs=self.freqs,
            s11=0.1 * rng.normal(size=n) + 0.1j * rng.normal(size=n),
            s12=0.8 + 0.1j * rng.normal(size=n),
            s21=0.7 + 0.1j * rng.normal(size=n),
            s22=0.2 * rng.normal(size=n) + 0j,
        )
        net = ncm.TwoPortNetwork.from_smatrix(s, z0=z0)
        ref = ncm.TwoPortNetwork.from_smatrix(s, z0=50.0)
        np.testing.assert_allclose(net.x, ref.x, rtol=1e-12)
        out = net.as_sparams(self.freqs, source_impedance=50.0)
        np.testing.assert_allclose(out.s, s.s, rtol=1e-10, atol=1e-14)
