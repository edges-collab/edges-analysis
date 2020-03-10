import numpy as np
import h5py
import attr
from attr import validators
from pathlib import Path
import contextlib
from edges_cal import modelling as mdl
from . import tools
from ..config import config


@attr.s
class _HDF5Group:
    filename = attr.ib(converter=Path, validator=lambda x: x.exists())
    group_path = attr.ib(converter=str)
    always_lazy = attr.ib(default=False, converter=bool)
    lazy = attr.ib(default=True, converter=bool)
    open = False

    def __attrs_post_init__(self):
        self.__memcache__ = {}

        if not self.lazy:
            self.load_all(self.filename)

    def get_group(self):
        fl = h5py.File(self.filename, "r")
        grp = fl
        for path in self.group_path.split("."):
            grp = grp[path]

        self.open = True
        return grp

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
            if isinstance(fl[item], h5py.Group):
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
    def from_data(cls, data, **kwargs):
        inst = cls(**kwargs)

        false_if_extra = kwargs.get("require_no_extra", cls._require_no_extra)
        false_if_absent = kwargs.get("require_all", cls._require_all)

        def _check(grp, strc):
            for k, v in strc:
                if k not in grp and false_if_absent:
                    raise TypeError()
                elif isinstance(v, dict):
                    _check(grp[k], v)
                elif v:
                    assert v(grp[k])

            # Ensure there's no extra keys in the group
            if false_if_extra and len(strc) < len(grp.keys()):
                raise ValueError()

        _check(data, cls._structure)

        inst.__memcache__ = data

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

        def _check(grp, strc):
            for k, v in strc.items():
                if k not in grp and false_if_absent:
                    raise TypeError()
                elif isinstance(v, dict):
                    _check(grp[k], v)
                elif v:
                    assert v(grp[k])

            # Ensure there's no extra keys in the group
            if false_if_extra and len(strc) < len(grp.keys()):
                raise ValueError()

        with h5py.File(filename, "r") as fl:
            _check(fl, cls._structure)

    def __getitem__(self, item):
        if item in self.__memcache__:
            return self.__memcache__[item]

        with open(self.filename, "r") as fl:
            if item in ("attrs", "meta"):
                out = dict(fl.attrs)
            if isinstance(fl[item], h5py.Group):
                out = _HDF5Group(self.filename, item)
            elif isinstance(fl[item], h5py.Dataset):
                out = fl[item][...]
            else:
                raise NotImplementedError("that item is not supported yet.")

        if not self.always_lazy:
            self.__memcache__[item] = out

        return out


def auxiliary_data(weather_file, thermlog_file, band, year, day):
    # scp -P 64122 loco@150.101.175.77:/media/DATA/EDGES_data/weather.txt /home/raul/Desktop/
    # scp -P 64122 loco@150.101.175.77:/media/DATA/EDGES_data/thermlog.txt /home/raul/Desktop/

    # OR

    # scp raul@enterprise.sese.asu.edu:/data1/edges/data/2014_February_Boolardy/weather.txt Desktop/
    # scp raul@enterprise.sese.asu.edu:/data1/edges/data/2014_February_Boolardy/thermlog_low.txt
    # Desktop/
    # scp raul@enterprise.sese.asu.edu:/data1/edges/data/2014_February_Boolardy/thermlog.txt
    # Desktop/

    array1 = read_weather_file(day, weather_file, year)
    array2 = read_thermlog_file(band, day, thermlog_file, year)

    return array1, array2


def read_thermlog_file(band, day, filename, year):
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


def read_weather_file(day, weather_file, year):
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


def _levelxread(path_file, out_keys):
    with h5py.File(path_file, "r") as hf:
        out = tuple(np.array(hf.get(k)) for k in out_keys)

    return out


def level2read(path_file):
    out_keys = ["frequency", "antenna_temperature", "metadata", "weights"]
    return _levelxread(path_file, out_keys)


def level3read(path_file):
    out_keys = [
        "frequency",
        "antenna_temperature",
        "parameters",
        "residuals",
        "weights",
        "rms",
        "total_power",
        "metadata",
    ]
    return _levelxread(path_file, out_keys)


def level3_single_file_test(
    path_file, gha_1, gha_2, f_low, f_high, save, save_spectrum_name
):
    f, t, p, r, w, rms, tp, m = level3read(path_file)

    gha = m[:, 4]
    gha[gha < 0] += 24

    if gha_2 > gha_1:
        mask = (gha >= gha_1) & (gha <= gha_2)
    else:
        mask = (gha >= gha_1) | (gha <= gha_2)

    avr, avw = tools.spectral_averaging(r[mask, :], w[mask, :])
    avp = np.mean(p[mask, :], axis=0)
    fb, rb, wb, sb = tools.average_in_frequency(avr, freq=f, weights=avw, n_samples=128)

    mb = mdl.model_evaluate("LINLOG", avp, fb / 200)

    tb = mb + rb

    freq_mask = (fb >= f_low) & (fb <= f_high)
    ff = fb[freq_mask]
    tt = tb[freq_mask]
    ww = wb[freq_mask]
    ss = sb[freq_mask]

    if save:
        outT = np.array([ff, tt, ww, ss])
        out = outT.T

        save_path = config["edges_folder"] + "mid_band/spectra/level5/one_day_tests/"
        np.savetxt(save_path + save_spectrum_name, out)

    return ff, tt, ww, ss


def data_selection(
    m,
    use_gha=True,
    time_1=0,
    time_2=24,
    sun_el_max=90,
    moon_el_max=90,
    amb_hum_max=200,
    min_receiver_temp=0,
    max_receiver_temp=100,
):
    # Master index
    index = np.arange(len(m[:, 0]))

    if use_gha:
        gha = m[:, 4]
        gha[gha < 0] += 24

        index_time_1 = index[gha >= time_1]
        index_time_2 = index[gha < time_2]
    else:
        index_time_1 = index[m[:, 3] >= time_1]
        index_time_2 = index[m[:, 3] < time_2]

    # Sun elevation, Moon elevation, ambient humidity, and receiver temperature
    index_sun = index[m[:, 6] <= sun_el_max]
    index_moon = index[m[:, 8] <= moon_el_max]
    index_hum = index[m[:, 10] <= amb_hum_max]
    index_Trec = index[
        (m[:, 11] >= min_receiver_temp) & (m[:, 11] <= max_receiver_temp)
    ]
    # Combined index
    if time_1 < time_2:
        index1 = np.intersect1d(index_time_1, index_time_2)
    else:
        index1 = np.union1d(index_time_1, index_time_2)

    return index1 & index_sun & index_moon & index_hum & index_Trec
    # index2 = np.intersect1d(index_sun, index_moon)
    # index3 = np.intersect1d(index2, index_hum)
    # index4 = np.intersect1d(index3, index_Trec)
    # return np.intersect1d(index1, index4)


def level4read(path_file):
    out_keys = [
        "frequency",
        "parameters",
        "residuals",
        "weights",
        "index",
        "gha_edges",
        "year_day",
    ]
    return _levelxread(path_file, out_keys)


def level4_binned_read(path_file):
    out_keys = ["frequency", "residuals", "weights", "stddev", "gha_edges", "year_day"]
    return _levelxread(path_file, out_keys)


def level4_save_averaged_spectra(case, gha_case, first_day, last_day):
    header_text = (
        "f [MHz], t_ant (GHA=0-23) [K], std (GHA=0-23) [K], Nsamples (GHA=0-23)"
    )
    file_name = "GHA_every_1hr.txt"
    file_path = config["edges_folder"] + "mid_band/spectra/level4/{}/binned_averages/"

    paths = {
        2: "calibration_2019_10_no_ground_loss_no_beam_corrections",
        3: "case_nominal_50-150MHz_no_ground_loss_no_beam_corrections",
        406: "case_nominal_50-150MHz_LNA1_a2_h2_o2_s1_sim2",
        5: "case_nominal_14_14_terms_55-150MHz_no_ground_loss_no_beam_corrections",
        501: "case_nominal_50-150MHz_LNA2_a2_h2_o2_s1_sim2_all_lc_yes_bc",
    }

    if case not in paths:
        raise ValueError("case must be one of {}".format(paths.keys()))

    pth = paths[case]
    file_path = file_path.format(pth)

    if gha_case != 24:
        raise ValueError("gha_case must be 24")

    for i in range(24):
        fb, tb, rb, wb, sb = tools.level4_integration(
            case, [i], first_day, last_day, 55, 150, 5
        )
        if not i:
            out = np.zeros((1 + 3 * 24, len(fb)))
            out[0, :] = fb

        out[i + 1, :] = tb
        out[i + 1 + 24, :] = sb
        out[i + 1 + 48, :] = wb

    np.savetxt(file_path + file_name, out.T, header=header_text)

    return out


def level4_foreground_fits_read(path_file):
    out_keys = ["fref", "fit2", "fit3", "fit4", "fit5"]
    return _levelxread(path_file, out_keys)


def calibration_rms_read(path_file):
    out_keys = ["RMS", "index_cterms", "index_wterms"]
    return _levelxread(path_file, out_keys)
