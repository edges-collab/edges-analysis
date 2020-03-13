import numpy as np
import h5py
import attr
from pathlib import Path
import contextlib


@attr.s
class _HDF5Group:
    filename = attr.ib(converter=Path, validator=lambda x: x.exists())
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


@attr.s
class HDF5Object:
    _structure = None
    _require_all = True
    _require_no_extra = True

    filename = attr.ib(default=None, converter=lambda x: x if x is None else Path(x))
    require_all = attr.ib(default=_require_all, converter=bool)
    require_no_extra = attr.ib(default=_require_no_extra, converter=bool)
    always_lazy = attr.ib(default=False, converter=bool)
    lazy = attr.ib(default=True, converter=bool)

    @filename.validator
    def _fn_validator(self, att, val):
        if val is not None:
            assert val.exists()

    def __attrs_post_init__(self):
        self.check(self.filename, self.require_no_extra, self.require_all)

        self.__memcache__ = {}

        if not self.lazy:
            self.load_all(self.filename)

    @classmethod
    def _checkgrp(cls, grp, strc, false_if_extra=False, false_if_absent=True):
        for k, v in strc:
            if k == "meta":
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

    def load_all(self, filename=None):
        if filename and not self.filename:
            self.filename = filename

        filename = filename or self.filename

        if not filename:
            raise ValueError("You need to provide a filename to load")

        for k, v in self._structure.items():
            self[k]

    def write(self, filename=None, clobber=False):
        filename = filename or self.filename

        if Path(filename).exists() and not clobber:
            raise FileExistsError(f"file {filename} already exists!")

        def _write(grp, struct, cache):
            for k, v in struct.items():
                if isinstance(v, dict):
                    _write(grp[k], struct[k], cache[k])
                elif np.isscalar(cache[k]):
                    grp.attrs[k] = cache[k]
                else:
                    grp[k] = cache[k]

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

        with open(self.filename, "r") as fl:
            if item in ("attrs", "meta"):
                out = dict(fl.attrs)
            if isinstance(fl[item], h5py.Group):
                out = _HDF5Group(self.filename, self._structure[item], item)
            elif isinstance(fl[item], h5py.Dataset):
                out = fl[item][...]
            else:
                raise NotImplementedError("that item is not supported yet.")

        if not self.always_lazy:
            self.__memcache__[item] = out

        return out


def auxiliary_data(weather_file, thermlog_file, band, year, day):
    # TODO: move to edges-io
    array1 = read_weather_file(day, weather_file, year)
    array2 = read_thermlog_file(band, day, thermlog_file, year)

    return array1, array2


def read_thermlog_file(band, day, filename, year):
    # TODO: move to edges-io
    # gather data from 'thermlog.txt' file
    with open(filename, "r") as fl:
        lines_all = fl.readlines()

    if (band == "high_band") and (year == 2015):
        i2 = 24000  # ~ day 108
    elif (band == "high_band") and (year == 2016):
        i2 = 58702  # beginning of year 2016
    elif (band == "low_band") and (year == 2015):
        i2 = 0
    elif (band == "low_band") and (year == 2016):
        i2 = 14920  # beginning of year 2016
    elif (band == "low_band") and (year == 2017):
        i2 = 59352  # beginning of year 2017
    elif (band == "low_band2") and (year == 2017) and (day < 332):
        return np.array([0])
    elif band == "low_band2" and year == 2017:
        i2 = 0
    elif (band == "low_band2") and (year == 2018):
        i2 = 4768
    elif (band == "low_band3") and (year == 2018):
        i2 = 0
    elif (band == "mid_band") and (year == 2018) and (day <= 171):
        i2 = 5624  # beginning of year 2018, file "thermlog_mid.txt"
    elif (band == "mid_band") and (year == 2018) and (day >= 172):
        i2 = 16154

    line = lines_all[i2]
    year_iter = int(line[0:4])
    day_of_year = int(line[5:8])

    out = np.zeros((0, 2))
    while day_of_year <= day and year_iter <= year:
        if day_of_year == day:

            date_time = line[0:17]
            ttt = date_time.split(":")
            seconds = 3600 * int(ttt[2]) + 60 * int(ttt[3]) + int(ttt[4])

            try:
                rec_temp = float(line[48:53])
            except ValueError:
                rec_temp = 0

            tmp = np.array([seconds, rec_temp]).reshape((1, -1))
            out = np.append(out, tmp, axis=0)

        i2 += 1
        if i2 != 26348:
            line = lines_all[i2]
            year_iter = int(line[0:4])
            day_of_year = int(line[5:8])

    return out


def read_weather_file(day, weather_file, year):
    # TODO: move to edges-io
    # Gather data from 'weather.txt' file
    with open(weather_file, "r") as f1:
        lines_all_1 = f1.readlines()
    array1 = np.zeros((0, 4))
    # TODO: this is really arbitrary and hard to understand
    if year == 2015:
        i1 = 92000  # ~ day 100
    elif year == 2016:
        i1 = 165097  # start of year 2016
    elif (year == 2017) and (day < 330):
        i1 = 261356  # start of year 2017
    elif (year == 2017) and (day > 331):
        i1 = 0  # start of year 2017 in file weather2.txt
    elif year == 2018:
        i1 = 9806  # start of year in file weather2.txt
    else:
        raise ValueError("year must be between 2015-2018 inclusive")
    line1 = lines_all_1[i1]
    year_iter_1 = int(line1[0:4])
    day_of_year_1 = int(line1[5:8])
    while day_of_year_1 <= day and year_iter_1 <= year:
        if day_of_year_1 == day:
            date_time = line1[0:17]
            ttt = date_time.split(":")
            seconds = 3600 * int(ttt[2]) + 60 * int(ttt[3]) + int(ttt[4])

            try:
                amb_temp = float(line1[59:65])
            except ValueError:
                amb_temp = 0

            try:
                amb_hum = float(line1[87:93])
            except ValueError:
                amb_hum = 0

            try:
                rec_temp = float(line1[113:119])
            except ValueError:
                rec_temp = 0

            array1_temp1 = np.array([seconds, amb_temp, amb_hum, rec_temp])
            array1_temp2 = array1_temp1.reshape((1, -1))
            array1 = np.append(array1, array1_temp2, axis=0)

        i1 += 1
        if i1 not in [28394, 1768]:
            line1 = lines_all_1[i1]
            year_iter_1 = int(line1[0:4])
            day_of_year_1 = int(line1[5:8])
    return array1
