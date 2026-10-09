import attrs
import pytest
from astropy import units as un

from edges import units


class TestIsUnit:
    def test_strtype(self):
        assert units.is_unit("MHz")
        assert not units.is_unit("fancy_string")

    def test_unittype(self):
        assert units.is_unit(un.MHz)
        assert not units.is_unit(42)

    @pytest.mark.parametrize(
        "unit", [un.m, un.K, un.m / un.s, un.MHz**2, un.dimensionless_unscaled]
    )
    def test_all_unit_types(self, unit):
        assert units.is_unit(unit)


class TestVldUnit:
    @pytest.mark.parametrize(
        ("unit", "good", "bad"),
        [
            (un.m, 1 * un.km, 1 * un.s),
            (un.m / un.s, 3 * un.km / un.hour, 1 * un.m),
            ("MHz", 1 * un.Hz, 1 * un.m),
            ("frequency", 1 * un.GHz, 1 * un.m),
        ],
    )
    def test_validation(self, unit, good, bad):
        @attrs.define
        class A:
            x = attrs.field(validator=units.vld_unit(unit))

        assert A(good).x == good
        with pytest.raises(un.UnitConversionError):
            A(bad)
        with pytest.raises(TypeError, match="must be an astropy Quantity"):
            A(1.0)
