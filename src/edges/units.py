"""Functions for dealing with units and quantities."""

import warnings
from collections.abc import Callable
from typing import Any

import attrs
from astropy import units
from astropy import units as u


def is_unit(unit: str) -> bool:
    """Whether the given input is a recognized unit."""
    if isinstance(unit, u.Unit):
        return True

    try:
        u.Unit(unit)
        return True
    except ValueError:
        return False


def vld_unit(
    unit: str | u.Unit, equivalencies=()
) -> Callable[[Any, attrs.Attribute, Any], None]:
    """Attr validator to check physical type."""
    utype = is_unit(unit)
    if not utype:
        # must be a physical type. This errors with ValueError if unit is not
        # really a physical type.
        u.get_physical_type(unit)

    def _check_type(self: Any, att: attrs.Attribute, val: Any):
        if not isinstance(val, u.Quantity):
            raise TypeError(f"{att.name} must be an astropy Quantity!")

        if utype and not val.unit.is_equivalent(unit, equivalencies):
            raise u.UnitConversionError(
                f"{att.name} not convertible to {unit}. Got {val.unit}"
            )

        if not utype and val.unit.physical_type != unit:
            raise u.UnitConversionError(
                f"{att.name} must have physical type of '{unit}'. "
                f"Got '{val.unit.physical_type}'"
            )

    return _check_type


def unit_convert_or_apply(
    x: float | units.Quantity,
    unit: str | units.Unit,
    in_place: bool = False,
    warn: bool = False,
) -> units.Quantity:
    """Safely convert a given value to a quantity."""
    if warn and not isinstance(x, units.Quantity):
        warnings.warn(
            f"Value passed without units, assuming '{unit}'. "
            "Consider specifying units for future compatibility.",
            stacklevel=2,
        )

    return units.Quantity(x, unit, copy=not in_place)


def unit_converter(
    unit: str | units.Unit,
) -> Callable[[float | units.Quantity], units.Quantity]:
    """Return a function that will convert values to a given quantity."""
    return lambda x: unit_convert_or_apply(x, unit)
