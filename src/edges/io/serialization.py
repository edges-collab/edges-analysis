"""A module defining how objects in the edges library should be serialized.

This includes ways to read/write them to HDF5 files.
"""

from typing import Any

import attrs
import cattrs
import h5py
import hickle

from .. import types as tp


def write_object_to_hdf5(obj: Any, path: tp.PathLike | h5py.Group):
    if not isinstance(path, h5py.Group):
        path = h5py.File(path, "w")

    dct = cattrs.unstructure(obj)
    hickle.dump(dct, path)


def load_hdf5(struc, path: tp.PathLike | h5py.Group):
    data = hickle.load(path)
    cattrs.structure(data, struc)


def hickleable(cls):
    """Render an attrs-defined class recursively hickleable."""
    # First, check whether all of the attributes have types
    if not attrs.has(cls):
        raise TypeError(
            f"Class {cls.__name__} has been defined incorrectly, it needs to be attrs!"
        )

    if untyped := [fld.name for fld in attrs.fields(cls) if not fld.type]:
        raise TypeError(f"Class {cls} has untyped fields: {untyped}")

    # Give our class a reader and writer
    cls.write = write_object_to_hdf5
    cls.from_file = load_hdf5

    return cls
