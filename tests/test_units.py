from astropy import units as un

from edges import units


class TestIsUnit:
    def test_strtype(self):
        assert units.is_unit("MHz")
        assert not units.is_unit("fancy_string")

    def test_unittype(self):
        assert units.is_unit(un.MHz)
        assert not units.is_unit(42)
