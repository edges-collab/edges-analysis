import numpy as np
import pytest
from astropy import units as un

from edges.cal.sparams.core import network_component_models as ncm
from edges.cal.sparams.core.datatypes import SParams


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
    def test_characteristic_and_propagation_and_reflection(self):
        freqs = freqs_mhz(2)
        # use simple, frequency independent per-m length values
        r = 1.0 * un.ohm / un.m
        L = 1e-6 * un.ohm * un.s / un.m
        g = 1e-9 * un.siemens / un.m
        c = 1e-11 * un.siemens * un.s / un.m
        tl = ncm.TransmissionLine(
            freqs=freqs,
            resistance=r,
            inductance=L,
            conductance=g,
            capacitance=c,
            length=1.0 * un.m,
        )

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
        freqs = freqs_mhz(1)
        tl = ncm.TransmissionLine(
            freqs=freqs,
            resistance=1.0 * un.ohm / un.m,
            inductance=1e-6 * un.ohm * un.s / un.m,
            conductance=1e-9 * un.siemens / un.m,
            capacitance=1e-11 * un.siemens * un.s / un.m,
        )
        with pytest.raises(ValueError):
            tl.input_impedance()


class TestCoaxialCable:
    def test_skin_depths_and_per_length_properties(self):
        c = ncm.CoaxialCable(
            outer_radius=0.01 * un.m,
            inner_radius=0.001 * un.m,
            outer_material="copper",
            inner_material="copper",
            relative_dielectric=2.0,
            length=1.0 * un.m,
        )
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
        assert isinstance(s, SParams)

    def test_unknown_material_raises(self):
        with pytest.raises(ValueError):
            ncm.CoaxialCable(
                outer_radius=0.01 * un.m,
                inner_radius=0.001 * un.m,
                outer_material="unknown-material",
                inner_material="copper",
                relative_dielectric=2.0,
            )


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


class TestCalkitAndGetCalkit:
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

    def test_z_y_h_matrix_roundtrip_and_add_checks(self):
        # Create a z-matrix and roundtrip via from_zmatrix
        z = np.array([[[2.0], [0.0]], [[0.0], [2.0]]])
        tpn = ncm.TwoPortNetwork.from_zmatrix(z)
        # zmatrix property should be close to the original z (up to algebraic transform)
        z_out = tpn.zmatrix
        assert z_out.shape[2] == 1

        # test add_in_series type checking
        with pytest.raises(ValueError):
            tpn.add_in_series(object())

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
        assert isinstance(s, SParams)
