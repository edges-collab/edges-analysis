"""A module defining how objects in the edges library should be serialized.

This includes ways to read/write them to HDF5 files.
"""

from datetime import datetime
from typing import Any, TypeVar

import attrs
import cattrs
import h5py
import hickle
import numpy as np
from astropy.coordinates import EarthLocation
from astropy.table import QTable
from astropy.time import Time
from astropy.units import Quantity

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
def _astropy_time_hook(val: np.ndarray, _) -> Time:
    """Convert an astropy quantity to a numpy array."""
    return Time(
        val,
        format="jd",
    )


@converter.register_unstructure_hook
def _astropy_time_unstructure_hook(val: Time) -> np.ndarray:
    """Convert an astropy quantity to a numpy array."""
    return val.jd


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
def _qtable_hook(val: np.ndarray, _) -> QTable:
    """Convert a datetime to string."""
    return QTable(data=val)


@converter.register_unstructure_hook
def _qtable_unstructure_hook(val: QTable) -> dict[str, np.ndarray]:
    """Convert a str to datetime."""
    return {col: val[col] for col in val.columns}


def write_object_to_hdf5(obj: Any, path: tp.PathLike | h5py.Group):
    """Write an attrs class to HDF5."""
    if not isinstance(path, h5py.Group):
        path = h5py.File(path, "w")

    dct = converter.unstructure(obj)
    hickle.dump(dct, path)


def load_hdf5(struc, path: tp.PathLike | h5py.Group):
    """Load an HDF5 file as a given type."""
    data = hickle.load(path)
    return converter.structure(data, struc)


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

    return cls
