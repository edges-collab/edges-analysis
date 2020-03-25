import numpy as np
import h5py
import attr
from pathlib import Path
import contextlib


@attr.s
class _HDF5Group:
    filename = attr.ib(converter=Path, validator=lambda x, att, val: val.exists())
    structure = attr.ib(converter=dict)
    group_path = attr.ib(converter=str)
    always_lazy = attr.ib(default=False, converter=bool)
    lazy = attr.ib(default=True, converter=bool)
    open = False

    @lazy.validator
    def _lazy_vld(self, att, val):
        if self.always_lazy and not val:
            raise ValueError("Can't be always lazy but not lazy.")

    def __attrs_post_init__(self):
        self.__memcache__ = {}

        if not self.lazy:
            self.load_all()

    def get_group(self):
        fl = h5py.File(self.filename, "r")
        grp = fl
        for path in self.group_path.split("."):
            grp = grp[path]

        self.open = True
        return grp

    def load_all(self):
        for k in self.structure:
            self[k]

    @contextlib.contextmanager
    def _open(self):
        """Context manager for using certain configuration options for a set time."""
        fl = h5py.File(self.filename, "r")
        grp = fl

        for bit in self.group_path.split("."):
            grp = grp[bit]

        yield grp

        fl.close()

    def __getitem__(self, item):
        if item in self.__memcache__:
            return self.__memcache__[item]

        with self._open() as fl:
            if item in ("attrs", "meta"):
                out = dict(fl.attrs)
            elif isinstance(fl[item], h5py.Group):
                out = _HDF5Group(self.filename, item)
            elif isinstance(fl[item], h5py.Dataset):
                out = fl[item][...]
            else:
                raise NotImplementedError("that item is not supported yet.")

        if not self.always_lazy:
            self.__memcache__[item] = out

        return out


@attr.s
class HDF5Object:
    _structure = None
    _require_all = True
    _require_no_extra = True
    default_root = Path(".")

    filename = attr.ib(default=None, converter=lambda x: x if x is None else Path(x))
    require_all = attr.ib(default=_require_all, converter=bool)
    require_no_extra = attr.ib(default=_require_no_extra, converter=bool)
    always_lazy = attr.ib(default=False, converter=bool)
    lazy = attr.ib(default=True, converter=bool)

    def __attrs_post_init__(self):
        self.__memcache__ = {}

        if self.filename and self.filename.exists():
            self.check(self.filename, self.require_no_extra, self.require_all)

        if not self.lazy:
            self.load_all(self.filename)

    @classmethod
    def _checkgrp(cls, grp, strc, false_if_extra=False, false_if_absent=True):
        for k, v in strc.items():
            if k == "meta" and k not in grp:
                k = "attrs"

            if k not in grp and false_if_absent:
                raise TypeError()
            elif isinstance(v, dict):
                cls._checkgrp(grp[k], v)
            elif v:
                assert v(grp[k])

        # Ensure there's no extra keys in the group
        if false_if_extra and len(strc) < len(grp.keys()):
            raise ValueError()

    @classmethod
    def from_data(cls, data, **kwargs):
        inst = cls(**kwargs)

        false_if_extra = kwargs.get("require_no_extra", cls._require_no_extra)
        false_if_absent = kwargs.get("require_all", cls._require_all)

        cls._checkgrp(data, cls._structure, false_if_extra, false_if_absent)

        inst.__memcache__ = data

        return inst

    def load_all(self, filename=None):
        if filename and not self.filename:
            self.filename = filename

        filename = filename or self.filename

        if not filename:
            raise ValueError("You need to provide a filename to load")

        for k, v in self._structure.items():
            self[k]

    def write(self, filename=None, clobber=False):
        if filename is None and self.filename is None:
            raise ValueError(
                "You need to pass a filename since there is no instance filename."
            )

        filename = Path(filename or self.filename)

        if not filename.is_absolute():
            filename = self.default_root / filename

        if self.filename is None:
            self.filename = filename

        if filename.exists() and not clobber:
            raise FileExistsError(f"file {filename} already exists!")

        def _write(grp, struct, cache):
            for k, v in cache.items():
                if isinstance(v, dict):
                    g = grp.create_group(k)
                    _write(g, struct[k], v)
                elif np.isscalar(cache[k]):
                    try:
                        grp.attrs[k] = v
                    except TypeError:
                        raise TypeError(
                            f"For key '{k}' in class '{self.__class__.__name__}', type '{type(cache[k])}' is not allowed in HDF5."
                        )
                else:
                    try:
                        grp[k] = v
                    except TypeError:
                        raise TypeError(
                            f"For key '{k}' in class '{self.__class__.__name__}', type '{type(cache[k])}' is not "
                            f"allowed in HDF5."
                        )

        with h5py.File(filename, "w") as fl:
            _write(fl, self._structure, self.__memcache__)

    @classmethod
    def check(cls, filename, false_if_extra=None, false_if_absent=None):
        false_if_extra = false_if_extra or cls._require_no_extra
        false_if_absent = false_if_absent or cls._require_all

        if not cls._structure:
            return True

        with h5py.File(filename, "r") as fl:
            cls._checkgrp(fl, cls._structure, false_if_extra, false_if_absent)

    def __getitem__(self, item):
        if item in self.__memcache__:
            return self.__memcache__[item]

        if item not in self._structure:
            raise KeyError(
                f"'{item}' is not a valid part of {self.__class__.__name__}. Valid keys: {self._structure.keys()}"
            )

        with h5py.File(self.filename, "r") as fl:
            if item in ("attrs", "meta"):
                out = dict(fl.attrs)
            elif isinstance(fl[item], h5py.Group):
                out = _HDF5Group(self.filename, self._structure[item], item)
            elif isinstance(fl[item], h5py.Dataset):
                out = fl[item][...]
            else:
                raise NotImplementedError("that item is not supported yet.")

        if not self.always_lazy:
            self.__memcache__[item] = out

        return out
