"""A module defining how objects in the edges library should be serialized.

This includes ways to read/write them to HDF5 files.
"""

from datetime import datetime
from typing import Any, TypeVar, get_origin

import attrs
import cattrs
import cattrs.strategies
import h5py
import hickle
import numpy as np
from astropy.coordinates import EarthLocation
from astropy.table import QTable
from astropy.time import Time
from astropy.units import Quantity, Unit
from astropy.units.core import UnitBase

from .. import types as tp

T = TypeVar("T")

converter = cattrs.Converter()


@converter.register_structure_hook
def ndarray_hook(val: Any, _) -> np.ndarray:
    """Structure a numpy array."""
    return np.asarray(val)


@converter.register_unstructure_hook
def ndarray_unstructure_hook(val: np.ndarray) -> np.ndarray:
    """Unstructure a numpy array."""
    return val


def _is_numpy_ndarray_generic_type(typ: Any) -> bool:
    """Match ``NDArray[float]``-style annotations (not bare ``ndarray``)."""
    return get_origin(typ) is np.ndarray


def _structure_numpy_ndarray_generic(val: Any, _: Any) -> np.ndarray:
    return np.asarray(val)


converter.register_structure_hook_func(
    _is_numpy_ndarray_generic_type, _structure_numpy_ndarray_generic
)


@converter.register_structure_hook
def _astropy_unit_hook(val: Any, _) -> Unit:
    """Structure an astropy Unit (hickle may restore ``IrreducibleUnit`` etc.)."""
    if isinstance(val, UnitBase):
        return val
    if isinstance(val, str):
        return Unit(val)
    raise TypeError(f"Cannot structure {type(val)!r} as astropy Unit")


@converter.register_unstructure_hook
def _astropy_unit_unstructure_hook(val: UnitBase) -> str:
    """Serialize as plain string."""
    return str(val)


@converter.register_structure_hook
def _astropy_quantity_hook(val: dict[str, Any], _) -> Quantity:
    """Convert an astropy quantity to a numpy array."""
    return Quantity(
        val["value"],
        unit=val["unit"],
        dtype=val.get("dtype"),
    )


@converter.register_unstructure_hook
def _astropy_quantity_unstructure_hook(val: Quantity) -> dict[str, Any]:
    """Convert an astropy quantity to a numpy array."""
    return {
        "value": val.value,
        "unit": str(val.unit),
        "dtype": val.dtype.str if val.dtype else None,
    }


@converter.register_structure_hook
def _astropy_time_hook(val: dict[str, Any] | np.ndarray | float, _) -> Time:
    """Structure an astropy Time.

    Accepts the dict written by :func:`_astropy_time_unstructure_hook`, and also
    the bare Julian dates written by older versions (which are interpreted as UTC,
    with no location, as they always were).
    """
    if not isinstance(val, dict):
        return Time(val, format="jd")

    location = val.get("location")
    if location is not None:
        location = EarthLocation.from_geocentric(*np.asarray(location), unit="m")

    out = Time(
        val["jd1"],
        val["jd2"],
        format="jd",
        scale=str(val["scale"]),
        location=location,
    )
    if (fmt := val.get("format")) is not None and str(fmt) in Time.FORMATS:
        out.format = str(fmt)
    return out


@converter.register_unstructure_hook
def _astropy_time_unstructure_hook(val: Time) -> dict[str, Any]:
    """Unstructure an astropy Time without loss.

    The two-part Julian date is stored (so there is no loss of precision), along
    with the time scale, the format and the geocentric location in metres (if any).
    """
    out = {
        "jd1": val.jd1,
        "jd2": val.jd2,
        "scale": val.scale,
        "format": val.format,
    }
    if val.location is not None:
        out["location"] = np.array([
            c.to_value("m") for c in val.location.to_geocentric()
        ])
    return out


@converter.register_structure_hook
def _datetime_hook(val: str, _) -> datetime:
    """Convert a datetime to string."""
    return datetime.fromisoformat(val)


@converter.register_unstructure_hook
def _datetime_unstructure_hook(val: datetime) -> str:
    """Convert a str to datetime."""
    return val.isoformat()


@converter.register_structure_hook
def _location_hook(val: np.ndarray, _) -> EarthLocation:
    """Convert a datetime to string."""
    return EarthLocation(lat=val[0], lon=val[1], height=val[2])


@converter.register_unstructure_hook
def _location_unstructure_hook(val: EarthLocation) -> list[Quantity]:
    """Convert a str to datetime."""
    return [val.lat, val.lon, val.height]


@converter.register_structure_hook
def _qtable_hook(val: dict[str, np.ndarray | Quantity], _) -> QTable:
    """Convert a datetime to string."""
    return QTable(data=val)


@converter.register_unstructure_hook
def _qtable_unstructure_hook(val: QTable) -> dict[str, np.ndarray | Quantity]:
    """Convert a str to datetime."""
    return {col: val[col] for col in val.columns}


def write_object_to_hdf5(obj: Any, path: tp.PathLike | h5py.Group):
    """Write an attrs class to HDF5.

    Parameters
    ----------
    obj
        The object to write.
    path
        The file to write to (it is overwritten, and closed afterwards), or an
        open h5py group to write into (which is left open).
    """
    dct = converter.unstructure(obj)

    if isinstance(path, h5py.Group):
        hickle.dump(dct, path)
    else:
        with h5py.File(path, "w") as fl:
            hickle.dump(dct, fl)


def load_hdf5(struc, path: tp.PathLike | h5py.Group):
    """Load an HDF5 file as a given type."""
    data = hickle.load(path)
    return converter.structure(data, struc)


def _structure_to_hickleable(cls, data: dict, conv):
    """Convert a structure to a hickleable class."""
    if not attrs.has(cls):
        raise TypeError(f"Class {cls.__name__} has not been defined with attrs!")

    clsname = data["__classname__"]

    # Define a recursive function to get ALL subclasses of the class
    # to which we're trying to structure the data, because
    # maybe we want to structure to a subclass rather than the parent class.
    def all_subclasses(cls):
        return set(cls.__subclasses__()).union([
            s for c in cls.__subclasses__() for s in all_subclasses(c)
        ])

    # If the class name saved with the unstructured data does not match this class,
    # move on to a subclass that _does_ match.
    if clsname != cls.__name__:
        allsubs = all_subclasses(cls)
        return conv.structure(
            data, next(kls for kls in allsubs if kls.__name__ == clsname)
        )

    # Otherwise, if we match this class, do a direct structuring of each field.
    fields_dict = attrs.fields_dict(cls)

    dd = {
        fields_dict[k].alias: conv.structure(data[k], fields_dict[k].type)
        for k in fields_dict
    }
    return cls(**dd)


def _unstructure_hickleable(obj, conv) -> dict:
    dct = conv.unstructure(attrs.asdict(obj, recurse=False))
    dct["__classname__"] = obj.__class__.__name__
    return dct


def hickleable(cls: T) -> T:
    """Render an attrs-defined class recursively hickleable."""
    # First, check whether all of the attributes have types
    if not attrs.has(cls):
        raise TypeError(
            f"Class {cls.__name__} has been defined incorrectly, it needs to be attrs!"
        )

    if untyped := [fld.name for fld in attrs.fields(cls) if not fld.type]:
        raise TypeError(f"Class {cls} has untyped fields: {untyped}")

    # Give our class a reader and writer
    if not hasattr(cls, "write"):
        cls.write = write_object_to_hdf5
    if not hasattr(cls, "from_file"):
        cls.from_file = classmethod(load_hdf5)

    # Give our attrs/hickleable class a _structure and _unstructure method.
    # These will be used to hickle the class.
    cls._structure = classmethod(_structure_to_hickleable)
    cls._unstructure = _unstructure_hickleable

    return cls


cattrs.strategies.use_class_methods(converter, "_structure", "_unstructure")
