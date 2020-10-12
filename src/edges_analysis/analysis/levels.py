from os.path import dirname, basename
import os
from typing import Tuple, Optional, Sequence, List, Union, Dict
from methodtools import lru_cache
import glob
import yaml
import matplotlib.pyplot as plt
import warnings
import tqdm
import json
import sys
import re
from multiprocess.pool import Pool
from multiprocessing import cpu_count
import h5py
from functools import partial

import numpy as np
from edges_cal import (
    FrequencyRange,
    modelling as mdl,
    xrfi as rfi,
    Calibration,
)
import p_tqdm
from edges_io.auxiliary import auxiliary_data
from edges_io.logging import logger
from edges_io.h5 import HDF5Object
import time
import attr
from pathlib import Path
from cached_property import cached_property
from read_acq import decode_file

# import src.edges_analysis
from . import s11 as s11m, loss, beams, tools, filters, coordinates
from .coordinates import get_jd, dt_from_jd
from ..config import config
from .. import const
from datetime import datetime


@attr.s
class _Level(HDF5Object):
    """Base object for formal data reduction levels in edges-analysis.

    The structure is such that three groups will always be available:
    * spectra : containing frequency-based data. Arrays here include ``frequency``,
      ``antenna_temp`` and possibly ``weights``. Each of these will always have
      the frequency as their _last_ axis.
    * ancillary : containing non-defining data that is not frequency based (usually
      time based). May contain arrays such as ``time``, ``lst``, ``ambient_temp`` etc.
    * meta : parameters defining the data (eg. input parameters) or other scalars
      that describe the data.
    """

    default_root = config["paths"]["field_products"]

    _structure = {
        "frequency": None,
        "spectra": {
            "weights": None,
            "spectrum": None,
        },  # All spectra components assumed to be the same shape, with last axis being frequency.
        "ancillary": None,  # A structured array with last axis being time / GHA
        "meta": None,
    }

    def clear(self):
        """Clears all in-memory data that exists in the base file."""
        self.__memcache__ = {}

    @classmethod
    def _get_previous_level(cls):
        _prev_level = int(cls.__name__[-1]) - 1
        if _prev_level:
            _prev_level = getattr(sys.modules[__name__], f"Level{_prev_level}")
        else:
            raise AttributeError(f"from_previous_level is not defined for {cls.__name__}")
        return _prev_level

    @classmethod
    def from_previous_level(cls, prev_level, filename=None, clobber=False, **kwargs):
        _prev_level = cls._get_previous_level()

        if not isinstance(prev_level, _prev_level):
            if hasattr(prev_level, "__len__"):
                prev_level = [
                    p if isinstance(p, _prev_level) else _prev_level(p) for p in prev_level
                ]
            else:
                prev_level = _prev_level(prev_level)

        # Sort the files by their filenames. *Usually* this will correspond to date.
        if isinstance(prev_level, list):
            prev_level = sorted(
                prev_level, key=lambda x: (x.meta["year"], x.meta["day"], x.meta["hour"])
            )

        freq, data, ancillary, meta = cls._from_prev_level(prev_level, **kwargs)

        meta["prev_level_files"] = (
            ":".join([str(p.filename) for p in prev_level])
            if isinstance(prev_level, list)
            else str(prev_level.filename)
        )

        if clobber and Path(filename).exists():
            os.remove(filename)

        out = cls.from_data(
            {"frequency": freq, "spectra": data, "ancillary": ancillary, "meta": meta},
        )

        if filename:
            out.write(filename)

        return out

    @cached_property
    def previous_level(self):
        try:
            fnames = self.meta["prev_level_files"].split(":")
        except KeyError:
            raise TypeError(f"No previous levels for {self.__class__.__name__}")

        if len(fnames) == 1:
            return self._get_previous_level()(fnames[0])
        else:
            return [self._get_previous_level()(fname) for fname in fnames]

    @classmethod
    def _from_prev_level(cls, prev_level, **kwargs):
        pass

    @property
    def meta(self):
        return self.load("meta")

    @property
    def spectrum(self):
        try:
            return self.spectra.load("spectrum")
        except AttributeError:
            return self.spectra["spectrum"]

    @property
    def raw_frequencies(self):
        return self.load("frequency")

    @property
    def freq(self):
        return FrequencyRange(self.raw_frequencies)

    @property
    def ancillary(self):
        return self.load("ancillary")

    @property
    def spectra(self):
        return self["spectra"]

    @property
    def weights(self):
        try:
            return self.spectra.load("weights")
        except AttributeError:
            return self.spectra["weights"]


@attr.s
class Level1(_Level):
    """Object representing the level-1 stage of processing.

    This object essentially represents a Calibrated spectrum, replete with ancillary
    metadata.

    Attributes
    ----------
    spectra : dict
        Containing
        * ``frequency``: the raw frequencies of the spectrum.
        * ``antenna_temp``: the nominal un-calibrated temperature measured by the antenna.
        * ``switch_powers``: (3, NFREQ) array containing the 3-switch position power
          measurements from the antenna.
        * ``Q``: (Q = (p0 - p1)/(p0 - p2))
    ancillary : dict
        Containing:
        * ``times``: string representations of the time of each switch-0 reading.
        * ``adcmin``: Minimum analog-to-digital converter value (size NTIMES)
        * ``adcmax``: Maximum analog-to-digital converter value (size NTIMES)
    meta : dict
        Containing:
        * ``year``: integer year that the data was taken (first measurement in file)
        * ``day``: integer day that the data was taken (first measurement in file)
        * ``hour``: integer hour that the data was taken (first measurement in file)
        * ``temperature``: not sure what temperature this is ?? # TODO
        * ``nblk``: also not sure what this is # TODO

    """

    _structure = {
        "frequency": None,
        # All spectra components assumed to be the same shape, with last axis being frequency.
        "spectra": {
            "weights": None,
            "spectrum": None,
            "Q": None,
            "switch_powers": None,
        },
        "ancillary": None,  # A structured array with last axis being time
        "meta": {},
    }

    @classmethod
    def from_acq(
        cls,
        filename,
        band,
        calfile,
        s11_path,
        weather_file=None,
        thermlog_file=None,
        out_file=None,
        progress=True,
        leave_progress=True,
        xrfi_pipe: [None, dict] = None,
        s11_file_pattern: str = r"{y}_{jd}_{h}_*_input{input}.s1p",
        ignore_s11_files: [None, List[str]] = None,
        **cal_kwargs,
    ):
        t = time.time()
        Q, p, ancillary = decode_file(
            filename,
            write_formats=[],
            meta=True,
            progress=progress,
            leave_progress=leave_progress,
        )
        logger.info(f"Time for reading: {time.time() - t:.2f} sec.")

        logger.info("Converting time strings to datetimes...")
        t = time.time()
        times = cls.get_datetimes(ancillary.data["time"])
        logger.info(f"...  finished in {time.time() - t:.2f} sec.")

        meta = {
            "year": times[0].year,
            "day": get_jd(times[0]),
            "hour": times[0].hour,
            "band": band,
            "xrfi_pipe": xrfi_pipe,
            **ancillary.meta,
        }

        time_based_anc = ancillary.data

        logger.info("Getting ancillary weather data...")
        t = time.time()
        new_anc, new_meta = cls._get_weather_thermlog(band, times, weather_file, thermlog_file)
        meta = {**meta, **new_meta}

        time_based_anc = tools.join_struct_arrays((time_based_anc, new_anc))
        logger.info(f"... finished in {time.time() - t:.2f} sec.")

        s11_files = cls.get_s11_paths(
            s11_path, band, times[0], s11_file_pattern, ignore_files=ignore_s11_files
        )

        logger.info("Calibrating data ...")
        t = time.time()
        calspec, freq, weights, new_meta = cls._calibrate(
            spectrum=Q,
            frequencies=ancillary.frequencies,
            band=band,
            calfile=Path(calfile).expanduser(),
            ambient_temp=time_based_anc["ambient_temp"],
            lst=time_based_anc["lst"],
            s11_files=s11_files,
            configuration="",
            **cal_kwargs,
        )
        logger.info(f"... finished in {time.time() - t:.2f} sec.")

        # RFI cleaning.
        # We need to do any rfi cleaning desired on the raw powers right here, as in
        # future levels they are not stored.
        if xrfi_pipe:
            logger.info("Running xRFI...")
            t = time.time()
            for pspec in p:
                tools.run_xrfi_pipe(pspec, weights, xrfi_pipe)
            logger.info(f"... finished in {time.time() - t:.2f} sec.")

        meta = {**meta, **new_meta}

        data = {
            "frequency": freq.freq,
            "spectrum": calspec,
            "switch_powers": [pp[:, freq.mask] for pp in p],
            "weights": weights,
            "Q": Q[:, freq.mask],
        }

        if out_file is None:
            out_file = (
                Path(config["paths"]["field_products"])
                / "level1"
                / basename(filename).replace(".acq", ".h5")
            )

        return cls.from_data(
            {
                "frequency": freq.freq,
                "spectra": data,
                "ancillary": time_based_anc,
                "meta": meta,
            },
            filename=str(out_file),
        )

    def get_subset(self, integrations=100):
        """Write a subset of the data to a new mock Level1 file."""
        freq = self.raw_frequencies
        spectra = {k: self.spectra.load(k) for k in self.spectra.keys()}
        ancillary = self.load("ancillary")
        meta = self.meta

        spectra = {k: s[:integrations] for k, s in spectra.items()}
        ancillary = ancillary[:integrations]

        return self.from_data(
            {"frequency": freq, "spectra": spectra, "ancillary": ancillary, "meta": meta}
        )

    @classmethod
    def default_s11_directory(cls, band):
        return Path(config["paths"]["raw_field_data"]) / "mro" / band / "s11"

    @classmethod
    def _get_closest_s11_time(
        cls,
        s11_dir: Path,
        time: datetime,
        s11_file_pattern: str = "{y}_{jd}_{h}_*_input{input}.s1p",
        ignore_files=None,
    ):
        """From a given filename pattern, within a directory, find file closest to time.

        Parameters
        ----------
        s11_dir : Path
            The directory in which to search for S11 files.
        time : datetime
            The time to find the closest match to.
        s11_file_pattern : str
            A pattern that matches files in the directory. A few tags are available:
            {input}: tags the input number (should be 1-4)
            {y}: year (four digit number)
            {m}: month (two-digit number)
            {d}: day of month (two-digit number)
            {jd}: day of year (three-digit number)
            {h}: hour of day (observation start) (two digit number)
        ignore_files : list, optional
            A list of file patterns to ignore. They need only partially match
            the actual filenames. So for example, you could specify ``ignore_files=['2020_076']``
            and it will ignore the file ``/home/user/data/2020_076_01_02_input1.s1p``.
            Full regex can be used.
        """
        # Replace the suffix dot with a literal dot for regex
        s11_file_pattern = s11_file_pattern.replace(".", r"\.")

        # Replace any glob-style asterisks with non-greedy regex version
        s11_file_pattern = s11_file_pattern.replace("*", r".*?")

        # First, we need to build a regex pattern out of the s11_file_pattern
        dct = {
            "input": r"(?P<input>\d)",
            "y": r"(?P<year>\d\d\d\d)",
            "m": r"(?P<month>\d\d)",
            "d": r"(?P<day>\d\d)",
            "jd": r"(?P<jd>\d\d\d)",
            "h": r"(?P<hour>\d\d)",
        }
        dct = {d: v for d, v in dct.items() if "{%s}" % d in s11_file_pattern}

        if not ("d" in dct or "jd" in dct):
            raise ValueError("s11_file_pattern must contain a tag {d} or {jd}.")
        if "d" in dct and "jd" in dct:
            raise ValueError("s11_file_pattern must not contain both {d} and {jd}.")

        p = re.compile(s11_file_pattern.format(**dct))

        ignore = [re.compile(ign) for ign in (ignore_files or [])]

        files = list(s11_dir.glob("*"))

        s11_times = []
        indx = []
        for i, fl in enumerate(files):
            match = p.match(str(fl.name))

            # Ignore files that don't match the pattern
            if not match:
                continue
            if any(ign.match(str(fl.name)) for ign in ignore):
                continue

            d = match.groupdict()

            indx.append(i)

            # Different time constructor for Day of year vs Day of month
            if "jd" in d:
                dt = tools.dt_from_year_day(
                    int(d.get("year", time.year)),
                    int(d.get("jd")),
                    int(d.get("hour", 0)),
                )
            else:
                dt = datetime(
                    int(d.get("year", time.year)),
                    int(d.get("month", time.month)),
                    int(d.get("day")),
                    int(d.get("hour", 0)),
                )
            s11_times.append(dt)

        if not len(s11_times):
            raise FileNotFoundError(
                f"No files found matching the input pattern. Available files: {[fl.name for fl in files]}. Regex pattern: {p.pattern}. "
            )

        files = [fl for i, fl in enumerate(files) if i in indx]
        time_diffs = np.array([abs((time - t).total_seconds()) for t in s11_times])
        indx = np.where(time_diffs == time_diffs.min())[0]

        # Gets a representative closest time file
        closest = [fl for i, fl in enumerate(files) if i in indx]

        assert (
            len(closest) == 4
        ), f"There need to be four input S1P files of the same time, got {closest}."
        return sorted(closest)

    @classmethod
    def get_s11_paths(
        cls,
        s11_path: [str, Path, Tuple, List],
        band: str,
        begin_time: datetime,
        s11_file_pattern: str,
        ignore_files: [None, List[str]] = None,
    ):
        """Given an s11_path, return list of paths for each of the inputs"""

        # If we get four files, make sure they exist and pass them back
        if isinstance(s11_path, (tuple, list)):
            if len(s11_path) != 4:
                raise ValueError("If passing explicit paths to S11 inputs, length must be 4.")

            fls = []
            for pth in s11_path:
                p = Path(pth).expanduser().absolute()
                assert p.exists()
                fls.append(p)

            return fls

        # Otherwise it must be a path.
        s11_path = Path(s11_path).expanduser()

        if str(s11_path).startswith(":"):
            s11_path = cls.default_s11_directory(band) / str(s11_path)[1:]

        if s11_path.is_dir():
            # Get closest measurement
            return cls._get_closest_s11_time(
                s11_path, begin_time, s11_file_pattern, ignore_files=ignore_files
            )
        else:
            # The path *must* have an {input} tag in it which we can search on
            fls = glob.glob(str(s11_path).format(input="?"))
            assert (
                len(fls) == 4
            ), f"There are not exactly four files matching {s11_path}. Found: {fls}."
            return sorted([Path(fl) for fl in fls])

    @property
    def raw_time_data(self):
        """Raw string times at which the spectra were taken."""
        return self.ancillary["time"]

    @cached_property
    def datetimes(self):
        """List of python datetimes at which the spectra were taken."""
        return self.get_datetimes(self.raw_time_data)

    @classmethod
    def get_datetimes(cls, times):
        return [datetime.strptime(d, "%Y:%j:%H:%M:%S") for d in times.astype(str)]

    @classmethod
    def _get_weather_thermlog(
        cls,
        band: str,
        times: List[datetime],
        weather_file: [None, Path, str] = None,
        thermlog_file: [None, Path, str] = None,
    ):
        """
        Read the appropriate weather and thermlog file, returning their contents.

        Parameters
        ----------
        band : str
            The band/telescope of the data (mid, low2, low3, high).
        times : list of datetimes
            List of datetime objects giving the date-times of the (beginning of) observations.
        weather_file : path, optional
            Path to a weather file from which to read the weather data. Must be
            formatted appropriately. By default, will choose an appropriate file from
            the configured `raw_field_data` directory. If provided, will search in
            the current directory and the `raw_field_data` directory for the given
            file (if not an absolute path).
        thermlog_file : path, optional
            Path to a thermlog file from which to read the thermlog data. Must be
            formatted appropriately. By default, will choose an appropriate file from
            the configured `raw_field_data` directory. If provided, will search in
            the current directory and the `raw_field_data` directory for the given
            file (if not an absolute path).

        Returns
        -------
        auxiliary : numpy structured array
            Containing
            * "ambient_temp": Ambient temperature as a function of time
            * "ambient_humidity": Ambient humidity as a function of time
            * "receiver1_temp": Receiver1 temperature as a function of time
            * "receiver2_temp": Receiver2 temperature as a function of time
            * "lst": LST for each observation in the spectrum.
            * "gha": GHA for each observation in the spectrum.
            * "sun_moon_azel": Coordinates of the sun and moon as function of time.
        meta : dict
            Containing
            * "thermlog_file": absolute path to the thermlog information used (filled in with
              the default if necessary).
            * "weather_file": absolute path to the weather information used (filled in with
              the default if necessary).
        """

        start = min(times)
        end = max(times)

        pth = Path(config["paths"]["raw_field_data"])
        if weather_file is not None:
            weather_file = Path(weather_file)
            if not (weather_file.exists() or weather_file.is_absolute()):
                weather_file = pth / weather_file
        else:
            if (start.year, start.day) <= (2017, 329):
                weather_file = pth / "weather_upto_20171125.txt"
            else:
                weather_file = pth / "weather2.txt"

        if thermlog_file is not None:
            thermlog_file = Path(thermlog_file)
            if not (thermlog_file.exists() or thermlog_file.is_absolute()):
                thermlog_file = pth / thermlog_file
        else:
            thermlog_file = pth / f"thermlog_{band}.txt"

        # Get all aux data covering our times, up to the next minute (so we have some
        # overlap).
        weather, thermlog = auxiliary_data(
            weather_file,
            thermlog_file,
            year=start.year,
            day=get_jd(start),
            hour=start.hour,
            end_time=(end.year, get_jd(end), end.hour, end.minute + 1),
        )
        print("Setting up arrays...")

        t = time.time()
        # Get the seconds since obs start for the data (not the auxiliary).
        seconds = np.array([(t - times[0]).total_seconds() for t in times])

        time_based_anc = np.zeros(
            len(seconds),
            dtype=[("seconds", int)]
            + [
                (name, float)
                for name, (kind, off) in weather.dtype.fields.items()
                if kind.kind == "f"
            ]
            + [
                (name, float)
                for name, (kind, off) in thermlog.dtype.fields.items()
                if kind.kind == "f"
            ]
            + [
                ("lst", float),
                ("gha", float),
                ("sun_az", float),
                ("sun_el", float),
                ("moon_az", float),
                ("moon_el", float),
            ],
        )
        time_based_anc["seconds"] = seconds
        print(f".... took {time.time() - t} sec.")

        t = time.time()
        # Interpolate weather
        for name, (kind, _) in weather.dtype.fields.items():
            if kind.kind == "i":
                continue

            wth_seconds = [
                (
                    dt_from_jd(x["year"], int(x["day"]), x["hour"], x["minute"], x["second"])
                    - times[0]
                ).total_seconds()
                for x in weather
            ]
            time_based_anc[name] = np.interp(seconds, wth_seconds, weather[name])

            # Convert to celsius
            if name.endswith("_temp"):
                time_based_anc[name] -= 273.15

        for name, (kind, _) in thermlog.dtype.fields.items():
            if kind.kind == "i":
                continue

            wth_seconds = [
                (
                    dt_from_jd(x["year"], int(x["day"]), x["hour"], x["minute"], x["second"])
                    - times[0]
                ).total_seconds()
                for x in thermlog
            ]

            time_based_anc[name] = np.interp(seconds, wth_seconds, thermlog[name])
        print(f"Took {time.time() - t} sec to interpolate auxiliary data.")

        # LST
        t = time.time()
        time_based_anc["lst"] = coordinates.utc2lst(times, const.edges_lon_deg)
        time_based_anc["gha"] = coordinates.lst2gha(time_based_anc["lst"])
        print(f"Took {time.time() - t} sec to get lst/gha")

        # Sun/Moon coordinates
        t = time.time()
        sun, moon = coordinates.sun_moon_azel(const.edges_lat_deg, const.edges_lon_deg, times)
        print(f"Took {time.time() - t} sec to get sun/moon coords.")

        time_based_anc["sun_az"] = sun[:, 0]
        time_based_anc["sun_el"] = sun[:, 1]
        time_based_anc["moon_az"] = moon[:, 0]
        time_based_anc["moon_el"] = moon[:, 1]

        meta = {
            "thermlog_file": str(thermlog_file.absolute()),
            "weather_file": str(weather_file.absolute()),
        }
        return time_based_anc, meta

    @classmethod
    def _get_antenna_s11(cls, s11_files, freq, switch_state_dir, n_terms, switch_state_run_num):
        # Get files
        return s11m.antenna_s11_remove_delay(
            s11_files,
            freq,
            switch_state_dir=switch_state_dir,
            delay_0=0.17,
            n_fit=n_terms,
            switch_state_run_num=switch_state_run_num,
        )

    @cached_property
    def _antenna_s11(self):
        s11_files = self.meta["s11_files"].split(":")
        freq = self.raw_frequencies
        switch_state_dir = self.meta["switch_state_dir"]
        switch_state_run_num = self.meta["switch_state_run_num"]
        n_terms = self.meta["antenna_s11_n_terms"]

        return self._get_antenna_s11(
            s11_files, freq, switch_state_dir, n_terms, switch_state_run_num
        )

    @property
    def antenna_s11_model(self):
        return self._antenna_s11[0]

    @property
    def antenna_s11(self):
        return self.antenna_s11_model(self.raw_frequencies)

    @property
    def raw_antenna_s11(self):
        return self._antenna_s11[1]

    @property
    def raw_antenna_s11_freq(self):
        return self._antenna_s11[2]

    @cached_property
    def calibration(self):
        """The Calibration object used to calibrate this observation."""
        return Calibration(self.meta["calfile"])

    @classmethod
    def _calibrate(
        cls,
        spectrum,
        frequencies,
        band,
        calfile: [str, Calibration],
        ambient_temp,
        lst,
        s11_files,
        configuration="",
        switch_state_dir=None,
        weights=None,
        antenna_s11_n_terms=15,
        antenna_correction=True,
        balun_correction=True,
        ground_correction=True,
        beam_file=None,
        f_low: float = 50,
        f_high: float = 150,
        n_fg=7,
        switch_state_run_num=None,
    ) -> Tuple[np.ndarray, FrequencyRange, np.ndarray, dict]:
        """
        Calibrate data.

        This method performs the following operations on the data:

        * Restricts frequency range
        * Applies a lab-based calibration solution
        * Corrects for ground/balun/antenna losses
        * Corrects for beam factor
        * Flags RFI using an explicit list of channels
        * Flags RFI using a moving window polynomial filter

        Parameters
        ----------
        calobs : :class:`CalibrationObservation` instance or path
            The lab-based calibration observation to use to calibrate the data.
        s11_path : path, optional
            Path to the S11 measurements of the antenna. It should be an absolute
            path, to which will be appended "_inputX.s1p" to obtain the four
            necessary inputs.
        antenna_s11_n_terms : int, optional
            Number of terms used in fitting the S11 model.
        antenna_correction : bool, optional
            Whether to perform the antenna correction
        balun_correction : bool, optional
            Whether to perform the balun correction
        ground_correction : bool, optional
            Whether to perform the ground correction
        beam_file : path, optional
            Filename (not absolute) of a beam model to use for correcting for the beam
            factor. Not used if not provided.
        configuration : str, optional
            Specification of the antenna -- orientation etc. Should be a predefined
            format, eg '45deg'.
        f_low : float
            Minimum frequency to use.
        f_high : float
            Maximum frequency to use.
        n_fg : int, optional
            Number of foreground terms to use in obtaining the model and residual.

        Returns
        -------
        data : dict
            Same keys as for :func:`level1_to_level2` but adding `weights`.
        ancillary : dict
            The same ancillary data as contained in `level2`.
        meta : dict
            Contains all input parameters, as well as level2 meta.
        """
        if not isinstance(calfile, Calibration):
            calfile = Calibration(calfile)

        if switch_state_dir is not None:
            warnings.warn(
                "You should use the switch state that is inherently in the calibration object."
            )
            switch_state_dir = str(Path(switch_state_dir).absolute())
        else:
            switch_state_dir = calfile.internal_switch.path

        if switch_state_run_num is not None:
            warnings.warn(
                "You should use the switch state run_num that is inherently in the calibration object."
            )
            switch_state_run_num = switch_state_run_num
        else:
            switch_state_run_num = calfile.internal_switch.run_num

        meta = {
            "s11_files": ":".join([str(f) for f in s11_files]),
            "antenna_s11_n_terms": antenna_s11_n_terms,
            "antenna_correction": antenna_correction,
            "balun_correction": balun_correction,
            "ground_correction": ground_correction,
            "beam_file": str(Path(beam_file).absolute()) if beam_file is not None else "",
            "f_low": f_low,
            "f_high": f_high,
            "n_poly_xrfi": n_fg,
            "wterms": calfile.wterms,
            "cterms": calfile.cterms,
            "calfile": str(calfile.calfile),
            "calobs_path": str(calfile.calobs_path),
            "switch_state_dir": switch_state_dir,
            "switch_state_run_num": switch_state_run_num,
        }

        if np.all(spectrum == 0):
            raise Exception("The level2 file given has no non-zero spectra!")

        if weights is None:
            weights = np.ones_like(spectrum)

        # Cut the frequency range
        freq = FrequencyRange(frequencies, f_low=f_low, f_high=f_high)
        Q = spectrum[:, freq.mask]
        weights = weights[:, freq.mask]

        s11_ant = cls._get_antenna_s11(
            s11_files,
            freq.freq,
            switch_state_dir,
            antenna_s11_n_terms,
            switch_state_run_num,
        )
        # Calibrated antenna temperature with losses and beam chromaticity
        calibrated_temp = calfile.calibrate_Q(freq.freq, Q, s11_ant)

        # Antenna Loss (interface between panels and balun)
        G = np.ones_like(freq.freq)
        if antenna_correction:
            G *= loss.antenna_loss(
                antenna_correction, freq.freq, band=band, configuration=configuration
            )

        # Balun+Connector Loss
        if balun_correction:
            Gb, Gc = loss.balun_and_connector_loss(band, freq.freq, s11_ant)
            G *= Gb * Gc

        # Ground Loss
        if isinstance(ground_correction, (str, Path)):
            G *= loss.ground_loss(
                ground_correction, freq.freq, band=band, configuration=configuration
            )
        elif isinstance(ground_correction, float):
            G *= ground_correction

        a = ambient_temp + 273.15 if ambient_temp[0] < 200 else ambient_temp
        calibrated_temp = (calibrated_temp - np.outer(a, (1 - G))) / G

        # Beam factor
        if beam_file:
            beam_fac = beams.InterpolatedBeamFactor.from_beam_factor(
                beam_file, band=band, f_new=freq.freq
            )
            bf = beam_fac.evaluate(lst)

            # Remove beam chromaticity
            calibrated_temp /= bf

        return calibrated_temp, freq, weights, meta

    # @lru_cache()
    def bin_in_frequency(self, indx=None, resolution=0.0488) -> Tuple[np.ndarray, np.ndarray]:
        """
        Perform a frequency-average over the spectrum.

        Parameters
        ----------
        indx : int, optional
            The (time) index at which to compute the frequency-averaged spectrum.
            If not given, returns a 2D array, with time on the first axis.
        resolution : float, optional
            The frequency resolution of the output.

        Returns
        -------
        f
            The new frequency bin-centres.
        t
            The weighted-average of the spectrum in each bin
        w
            The total weight in each bin
        """
        if indx is not None:
            s, w = self.spectrum[indx], self.weights[indx]
        else:
            s, w = self.spectrum, self.weights

        bins = np.arange(self.freq.min, self.freq.max, resolution)

        centres = (bins[:-1] + bins[1:]) / 2
        new_spec = tools.non_stationary_bin_avg(
            data=s,
            x=self.raw_frequencies,
            weights=w,
            bins=bins,
        )
        new_weights = tools.get_binned_weights(x=self.raw_frequencies, weights=w, bins=bins)

        return centres, new_spec, new_weights

    # @lru_cache()
    def model(self, indx, model="polynomial", n_terms=5, resolution=0.0488, **model_kwargs):
        """
        Determine a callable model of the spectrum at a given time, optionally
        computed over averaged original data.

        Parameters
        ----------
        indx : int
            The (time) index to compute the model for.
        model : str, optional
            The kind of model to fit.
        n_terms : int, optional
            The number of terms to use in the fit.

        Returns
        -------
        callable :
            Function of frequency (in units of self.raw_frequency) that will return
            the model.
        """
        if resolution:
            f, s, w = self.bin_in_frequency(indx, resolution)
        else:
            f = self.raw_frequencies
            s = self.spectrum[indx]
            w = self.weights[indx]

        freq = FrequencyRange(f)
        model = mdl.ModelFit(model, freq.freq, s, weights=w, n_terms=n_terms, **model_kwargs)

        return model.evaluate

    # @lru_cache()
    def get_model_rms(
        self,
        model: [str, mdl.Model] = "polynomial",
        n_terms: int = 5,
        resolution: float = 0.0488,
        freq_range: Tuple[float, float] = (-np.inf, np.inf),
        indices=None,
        **model_kwargs,
    ):
        """Obtain the RMS of the residual of a model-fit to a particular integration.

        This method is cached, so that calling it again for the same arguments is
        fast.

        Parameters
        ----------
        indx
            The index of the integration for which to return the RMS. By default,
            returns an array with the RMS for each integration.
        model
            The model to fit (in edges-cal modelling).
        n_terms
            The number of model terms to use in the fit.
        resolution
            The spectrum itself is able to be averaged in frequency bins before the model
            is applied. This gives the resolution of those bins.
        freq_range
            The frequency range within which to fit the model (default the whole
            frequency range).

        Other Parameters
        ----------------
        All other parameters are passed to :class:`edges_cal.modelling.ModelFit`, which
        may be used to construct the model itself. For instance, to construct a LinLog
        model, use the "polynomial" model with an extra parameter of ``log_x=True``.

        Notes
        -----
        The averaging into frequency bins is *only* done for the fit itself. The final
        residuals are computed on the un-averaged spectrum. No flags/weights may be
        given to the function (primarily because this breaks caching), but the *intrinsic*
        weights of the object are used in the fit (and zero-weights are accounted for
        in the "mean" part of the RMS).
        """
        # Get the whole thing averaged (pretty quick)
        f, s, w = self.bin_in_frequency(resolution=resolution)[:3]

        if isinstance(model, str):
            model = mdl.Model._models[model.lower()](default_x=f, n_terms=n_terms, **model_kwargs)

        freq = FrequencyRange(self.raw_frequencies, f_low=freq_range[0], f_high=freq_range[1])

        def _get_rms(indx):
            m = mdl.ModelFit(model, ydata=s[indx], weights=w[indx]).evaluate(freq.freq)
            resid = self.spectrum[indx, freq.mask] - m
            mask = self.weights[indx, freq.mask] > 0
            return np.sqrt(np.mean(resid[mask] ** 2))

        if indices is None:
            indices = range(len(self.spectrum))

        res = [_get_rms(i) for i in indices]

        return np.array(res)

    def plot_waterfall(self, quantity: str = "spectrum", ax: [None, plt.Axes] = None, cbar=True):
        if quantity in ["p0", "p1", "p2"]:
            q = self.spectra["switch_powers"][int(quantity[-1])]
        else:
            q = self.spectra[quantity]

        if ax is not None:
            fig = ax.figure
        else:
            fig, ax = plt.subplots(1, 1)

        img = ax.imshow(
            q,
            extent=(
                self.raw_frequencies.min(),
                self.raw_frequencies.max(),
                self.ancillary["seconds"].min(),
                self.ancillary["seconds"].max(),
            ),
            aspect="auto",
        )
        if cbar:
            cb = plt.colorbar(img, ax=ax)
            cb.set_label(quantity)

        return ax

    def plot_waterfalls(self, quanties="all"):
        if quanties == "all":
            quanties = ["spectrum", "Q", "weights", "p0", "p1", "p2"]

        fig, ax = plt.subplots(
            len(quanties),
            1,
            sharex=True,
            sharey=True,
            figsize=(10, 10),
            gridspec_kw={"hspace": 0.05, "wspace": 0.05},
        )

        for i, (q, axx) in enumerate(zip(quanties, ax)):
            self.plot_waterfall(q, ax=axx)

        return fig, ax

    def plot_time_averaged_spectrum(
        self,
        quantity="spectrum",
        integrator="mean",
        ax: [None, plt.Axes] = None,
        logy=True,
    ):
        if ax is not None:
            fig = ax.figure
        else:
            fig, ax = plt.subplots(1, 1)

        q, w = self.integrate_over_time(quantity=quantity, integrator=integrator)

        unit = "[K]"
        if quantity == "Q":
            unit = ""

        ax.plot(self.raw_frequencies, q)
        ax.set_xlabel("Frequency [MHz]")
        ax.set_ylabel(f"{quantity} {unit}")

        if logy:
            ax.set_yscale("log")

        return ax

    def plot_s11(self, ax=None):
        if ax is None:
            fig, ax = plt.subplots(2, 2, figsize=(8, 8), sharex=True)
        ax[0, 0].plot(self.raw_frequencies, 20 * np.log10(np.abs(self.antenna_s11)))
        ax[0, 0].set_title("Magnitude of Antenna S11")
        ax[0, 0].set_xlabel("Frequency [MHz]")
        ax[0, 0].set_ylabel("$|S_{11}|$ [dB]")

        ax[0, 1].plot(self.raw_frequencies, (180 / np.pi) * np.unwrap(np.angle(self.antenna_s11)))
        ax[0, 1].set_title("Phase of Antenna S11")
        ax[0, 1].set_xlabel("Frequency [MHz]")
        ax[0, 1].set_ylabel(r"$\angle S_{11}$ [${}^\circ$]")

        ax[1, 0].plot(
            self.raw_antenna_s11_freq,
            np.real(self.raw_antenna_s11)
            - np.real(self.antenna_s11_model(self.raw_antenna_s11_freq)),
        )
        ax[1, 0].set_title("Residual (Real Part)")
        ax[1, 0].set_xlabel("Frequency [MHz]")
        ax[1, 0].set_ylabel(r"Data - Model")

        ax[1, 1].plot(
            self.raw_antenna_s11_freq,
            np.imag(self.raw_antenna_s11)
            - np.imag(self.antenna_s11_model(self.raw_antenna_s11_freq)),
        )
        ax[1, 1].set_title("Residual (Imag Part)")
        ax[1, 1].set_xlabel("Frequency [MHz]")
        ax[1, 1].set_ylabel(r"Data - Model")

        return ax

    def _integrate_spectra(self, quantity="spectrum", integrator="mean", axis=0):
        """Integrate spectra over given axis."""
        if quantity in ["p0", "p1", "p2"]:
            q = self.spectra["switch_powers"][int(quantity[-1])]
        else:
            q = self.spectra[quantity]

        if integrator in ("mean", "standard_deviation"):
            q, w = getattr(tools, "weighted_" + integrator)(q, self.weights, axis=axis)
        else:
            q = tools.weighted_sorted_metric(q, self.weights, metric=integrator, axis=axis)
            w = np.where(np.all(self.weights == np.nan, axis=axis), 0, 1)

        return q, w

    def integrate_over_time(self, quantity="spectrum", integrator="mean"):
        """Integrate the spectrum over time"""
        return self._integrate_spectra(quantity=quantity, integrator=integrator, axis=0)

    def integrate_over_frequency(self, quantity="spectrum", integrator="mean"):
        """Integrate the spectrum over time"""
        return self._integrate_spectra(quantity=quantity, integrator=integrator, axis=1)

    def aux_filter(
        self,
        sun_el_max: float = 90,
        moon_el_max: float = 90,
        ambient_humidity_max: float = 40,
        min_receiver_temp: float = 0,
        max_receiver_temp: float = 100,
        flags: [None, np.ndarray] = None,
    ) -> np.ndarray:
        """
        Perform an auxiliary filter on the object.

        Parameters
        ----------
        sun_el_max
            Maxmimum elevation of the sun to keep.
        moon_el_max
            Maxmimum elevation of the moon to keep.
        ambient_humidity_max
            Maximum ambient humidity to keep.
        min_receiver_temp
            Minimum receiver temperature to keep.
        max_receiver_temp
            Maximum receiver temp to keep.
        flags
            If given, do filtering in-place.

        Returns
        -------
        flags
            Boolean array giving which entries are bad.
        """

        return filters.time_filter_auxiliary(
            gha=self.ancillary["gha"],
            sun_el=self.ancillary["sun_el"],
            moon_el=self.ancillary["moon_el"],
            humidity=self.ancillary["ambient_hum"],
            receiver_temp=self.ancillary["receiver_temp"],
            sun_el_max=sun_el_max,
            moon_el_max=moon_el_max,
            amb_hum_max=ambient_humidity_max,
            min_receiver_temp=min_receiver_temp,
            max_receiver_temp=max_receiver_temp,
            flags=flags,
        )

    aux_filter.axis = "time"

    def rfi_filter(
        self, xrfi_pipe: dict, flags: [None, np.ndarray] = None, n_threads: int = cpu_count()
    ) -> np.ndarray:
        """
        Perform filtering on auxiliary data and RFI for a level 1 file.

        Parameters
        ----------
        xrfi_pipe
            A dictionary with keys specifying RFI function names, and values being
            dictionaries of parameters to pass to the function.

        Returns
        -------
        flags
            The boolean flag array, specifying which freqs/times are bad.
        """
        if flags is None:
            flags = np.zeros(self.weights.shape, dtype=bool)

        if "explicit" in xrfi_pipe:
            kwargs = xrfi_pipe.pop("explicit")

            if kwargs["file"] is None:
                known_rfi_file = Path(dirname(__file__)) / "data" / "known_rfi_channels.yaml"
            else:
                known_rfi_file = kwargs["file"]

            flags |= rfi.xrfi_explicit(
                self.raw_frequencies,
                rfi_file=known_rfi_file,
            )

            if np.all(flags):
                return flags

        return tools.run_xrfi_pipe(self.spectrum, flags, xrfi_pipe, n_threads=n_threads)

    rfi_filter.axis = "both"

    def rms_filter(
        self,
        rms_info: [filters.RMSInfo, str, Path],
        n_sigma_rms: int = 3,
        flags=None,
    ):
        if flags is None:
            flags = np.zeros(self.weights.shape, dtype=bool)

        if not isinstance(rms_info, filters.RMSInfo):
            rms_info = filters.RMSInfo.from_file(rms_info)

        rms = [self.get_model_rms(freq_range=band, **rms_info.model) for band in rms_info.bands]

        flags |= filters.rms_filter(rms_info, self.ancillary["gha"], rms, n_sigma_rms)
        return flags

    rms_filter.axis = "time"

    def total_power_filter(
        self,
        flags=None,
        n_poly: int = 3,
        n_sigma: float = 3.0,
        bands: [None, List[Tuple[float, float]]] = None,
        std_thresholds=None,
    ):
        if flags is None:
            flags = np.zeros(self.weights.T.shape, dtype=bool)

        flags |= filters.total_power_filter(
            self.ancillary["gha"],
            self.spectrum,
            self.freq.freq,
            flags=flags.T,
            n_poly=n_poly,
            n_sigma=n_sigma,
            bands=bands,
            std_thresholds=std_thresholds,
        )
        return flags

    total_power_filter.axis = "time"


class Level2(_Level):
    """
    Object representing a Level-2 Calibrated Data Set.

    Given a sequence of :class:`Level1` objects, this class combines them,
    filters out some times (see below), and integrates over time into given bins in
    LST/GHA.

    Times are filtered based on sun/moon position, humidity, receiver temperature,
    the total RMS of two halves of the spectrum after subtraction of a 5-term polynomial
    (fitted to frequency-binned data), the total RMS of the full spectrum after
    subtraction of a 3-term polynomial (fitted to frequency-binned data), and the total
    summed temperature over the spectrum (for each half, and the whole).

    Times are then averaged within provided bins of GHA.

    Further frequency filtering is performed based on a moving window polynomial filter
    of the mean data within each GHA bin.

    Parameters
    ----------
    level1 : list of :class:`Level3` instances or paths
        A bunch of level3 objects to combine and integrate into the level4 data.
    sun_el_max : float
        The maximum elevation of the sun before the time is filtered.
    moon_el_max : float
        The maximum elevation of the moon before the time is filtered.
    ambient_humidity_max : float, optional
        The maximum humidity allowed before the time is filtered.
    min_receiver_temp : float, optional
        The minimum temperature of the receiver before the observation is filtered.
    max_receiver_temp : float, optional
        The maximum temperature of the receiver before the observation is filtered.
    n_sigma_rms : int, optional
        The number of sigma at which to filter a time due to its RMS after subtracting
        a smooth model.
    rfi_window_size : float, optional
        The size of the moving polynomial window used to locate frequency-based RFI.
        In MHz.
    n_poly_rfi : int, optional
        The order of the polynomial used to fit the sliding window for RFI.
    n_bootstrap_rfi : int, optional
        The number of bootstrap samples to take to initialize the RMS for the first
        sliding window.
    n_sigma_rfi : float, optional
        Number of sigma to use for clipping RFI.

    Returns
    -------
    data : dict
        The same keys as in :func:`level2_to_level3`. However, the axes for the weights
        and spectra in this case are ``(N_FILES, N_GHA, N_FREQ)``, where ``N_FILES``
        is the original number of Level3 objects input.
    ancillary : dict
        Containing:
        * "n_total_times_per_file": The number of observations per input file.
        * "used_times": A 2D boolean array specifying which times (second axis)
          are used for which files (first axis). The array is padded out to the maximum
          number of times in any of the files, and the `n_total_times_per_file` should
          be used to index the relevant arrays.
        * 'years': The year for each file.
        * 'days': The day for each file
        * 'hours': The starting hour for each file.
    meta : dict
        Containing all input parameters, and the number of files used.
    """

    _structure = {
        "frequency": None,
        "spectra": {
            "weights": None,
            "spectrum": None,
        },
        "ancillary": {
            "n_total_times_per_file": None,
            "years": None,
            "days": None,
            "hours": None,
        },
        "meta": {},
    }

    @classmethod
    def run_filter(cls, fnc, level1, flags=None, nthreads=None, **kwargs):

        if nthreads is None:
            nthreads = min(len(level1), cpu_count())

        logger.info(f"Running {fnc} filter with {nthreads} threads...")

        axis = getattr(Level1, f"{fnc}_filter").axis

        if flags is None:
            flags = [None] * len(level1)

        def filter(flg, l1):
            if flg is None:
                flg = ~l1.weights.astype("bool")
            elif np.all(flg):
                return flg

            this_flag = getattr(l1, f"{fnc}_filter")(
                flags=flg.T if axis == "time" else flg, **kwargs
            )

            if axis in ("both", "freq"):
                flg |= this_flag
            else:
                flg |= this_flag.T

            return flg

        iterator = p_tqdm.p_imap(filter, flags, level1, num_cpus=nthreads)

        for i, flg in enumerate(iterator):
            flags[i] = flg

        # Warn the user if files are fully flagged.
        all_flagged = [np.all(flg) for flg in flags]
        if all(all_flagged):
            logger.error(f"All files were fully flagged during {fnc} filter.")
            sys.exit()

        if any(all_flagged):
            logger.warning(
                f"The following {len(all_flagged)} files were fully flagged during {fnc} filter:"
            )
            msg = ""
            for i, (flagged, l1) in enumerate(zip(all_flagged, level1)):
                if flagged:
                    msg += f"{l1.filename.name} | "
            logger.warning(msg[:-3])

        return (
            [flg for i, flg in enumerate(flags) if not all_flagged[i]],
            [l1 for i, l1 in enumerate(level1) if not all_flagged[i]],
        )

    @classmethod
    def _from_prev_level(
        cls,
        level1: List[Level1],
        gha_min: float = 0.0,
        gha_max: float = 24.0,
        gha_bin_size: float = 0.1,
        sun_el_max: float = 90,
        moon_el_max: float = 90,
        ambient_humidity_max: float = 40,
        min_receiver_temp: float = 0,
        max_receiver_temp: float = 100,
        rms_filter_file: [None, Path, str] = None,
        do_total_power_filter: bool = True,
        xrfi_pipe: [None, dict] = None,
        n_poly_tp_filter: int = 3,
        n_sigma_tp_filter: float = 3.0,
        bands_tp_filter: [None, List[Tuple[float, float]]] = None,
        std_thresholds_tp_filter: [None, List[float]] = None,
        do_rms_filter: bool = True,
        rms_bands: Sequence[Union[Tuple, str]] = ("full", "low", "high"),
        n_poly_rms: int = 3,
        n_sigma_rms: float = 3,
        n_terms_rms: int = 16,
        n_std_rms: int = 6,
        n_files_rms: [None, int] = None,
        n_threads: int = cpu_count(),
    ):
        xrfi_pipe = xrfi_pipe or {}

        if gha_min < 0 or gha_min > 24 or gha_min >= gha_max:
            raise ValueError("gha_min must be between 0 and 24")

        if gha_max < 0 or gha_max > 24:
            raise ValueError("gha_max must be between 0 and 24")

        meta = {
            "n_files": len(level1),
            "gha_min": gha_min,
            "gha_max": gha_max,
            "gha_bin_size": gha_bin_size,
            "sun_el_max": sun_el_max,
            "moon_el_max": moon_el_max,
            "ambient_humidity_max": ambient_humidity_max,
            "min_receiver_temp": min_receiver_temp,
            "max_receiver_temp": max_receiver_temp,
            "xrfi_pipe": json.dumps(xrfi_pipe),  # TODO: this is not wonderful
        }

        if do_total_power_filter:
            meta["n_poly_tp_filter"] = n_poly_tp_filter
            meta["n_sigma_tp_filter"] = n_sigma_tp_filter
            meta["bands_tp_filter"] = str(bands_tp_filter)
            meta["std_thresholds_tp_filter"] = str(std_thresholds_tp_filter)

        if do_rms_filter:
            meta["bands"] = str(rms_bands)
            meta["n_poly_rms"] = n_poly_rms
            meta["n_sigma_rms"] = n_sigma_rms
            meta["n_std_rms"] = n_std_rms
            meta["n_terms_rms"] = n_terms_rms
            meta["n_files_rms"] = n_files_rms

        # Sort the inputs in ascending date.
        level1 = sorted(level1, key=lambda x: (x.meta["year"], x.meta["day"], x.meta["hour"]))

        orig_dates = [(x.meta["year"], x.meta["day"], x.meta["hour"]) for x in level1]

        years = [x.meta["year"] for x in level1]
        days = [x.meta["day"] for x in level1]
        hours = [x.meta["hour"] for x in level1]

        # Create a master array of indices of good-quality spectra (over the time axis)
        # used in the final averages
        n_times = np.array([len(l1.ancillary) for l1 in level1])

        flags, level1 = cls.run_filter(
            "aux",
            level1,
            sun_el_max=sun_el_max,
            moon_el_max=moon_el_max,
            ambient_humidity_max=ambient_humidity_max,
            min_receiver_temp=min_receiver_temp,
            max_receiver_temp=max_receiver_temp,
        )
        if xrfi_pipe:
            flags, level1 = cls.run_filter("rfi", level1, flags=flags, xrfi_pipe=xrfi_pipe)

        if do_total_power_filter:
            flags, level1 = cls.run_filter(
                "total_power",
                level1,
                flags=flags,
                n_poly=n_poly_tp_filter,
                n_sigma=n_sigma_tp_filter,
                std_thresholds=std_thresholds_tp_filter,
                bands=bands_tp_filter,
            )

        if do_rms_filter:
            flags, level1 = cls._run_rms_filter(
                rms_filter_file=rms_filter_file,
                flags=flags,
                level1=level1,
                bands=rms_bands,
                n_poly=n_poly_rms,
                n_sigma=n_sigma_rms,
                n_terms=n_terms_rms,
                n_std=n_std_rms,
                n_files=n_files_rms,
            )

        final_dates = [(x.meta["year"], x.meta["day"], x.meta["hour"]) for x in level1]
        files_flagged = np.array([date not in final_dates for date in orig_dates])

        meta["n_files_flagged"] = sum(files_flagged)

        n_files = len(level1) - meta["n_files_flagged"]

        if not n_files:
            raise Exception("All input files have been filtered completely.")

        spectra, weights, gha_edges = cls.bin_gha(
            level1, gha_min, gha_max, gha_bin_size, flags=flags
        )

        data = {
            "spectrum": spectra,
            "weights": weights,
        }

        ancillary = {
            "n_total_times_per_file": n_times,
            "files_flagged": files_flagged,
            "years": years,
            "days": days,
            "hours": hours,
            "gha_edges": gha_edges,
        }

        return level1[0].raw_frequencies, data, ancillary, meta

    @cached_property
    def calibration(self):
        """The calibration object used to calibrate these spectra."""
        return self.previous_level.calibration

    @property
    def unflagged_level1(self) -> List[Level1]:
        """List of Level1 objects kept in the Level2 spectra (in order)."""
        return [
            l1 for i, l1 in enumerate(self.previous_level) if not self.ancillary["files_flagged"][i]
        ]

    @classmethod
    def _run_rms_filter(
        cls,
        rms_filter_file: [None, str, Path, filters.RMSInfo],
        flags: Sequence[np.ndarray],
        level1: Sequence[Level1],
        bands: Sequence[Union[Tuple, str]] = ("full", "low", "high"),
        n_poly: int = 3,
        n_sigma: float = 3,
        n_terms: int = 16,
        n_std: int = 6,
        n_files: [None, int] = None,
    ) -> Tuple[list, list]:
        if rms_filter_file and not isinstance(rms_filter_file, dict):
            rms_filter_file = Path(rms_filter_file)

        n_files = n_files or len(level1)

        if not isinstance(rms_filter_file, dict) and (
            not rms_filter_file or not rms_filter_file.exists()
        ):
            rms_info = filters.get_rms_info(
                level1=level1[:n_files],
                bands=bands,
                n_poly=n_poly,
                n_sigma=n_sigma,
                n_terms=n_terms,
                n_std=n_std,
            )
        else:
            rms_info = filters.RMSInfo.from_file(rms_filter_file)

        # Write out a file with the rms_info in it.
        if rms_filter_file and not rms_filter_file.exists():
            rms_info.write(rms_filter_file)

        return cls.run_filter(
            "rms",
            level1=level1,
            flags=flags,
            rms_info=rms_info,
            n_sigma_rms=n_sigma,
        )

    @classmethod
    def bin_gha(cls, level1, gha_min, gha_max, gha_bin_size, per_bin_model=False, flags=None):

        gha_edges = np.arange(gha_min, gha_max, gha_bin_size)
        if np.isclose(gha_max, gha_edges.max() + gha_bin_size):
            gha_edges = np.concatenate((gha_edges, [gha_edges.max() + gha_bin_size]))

        # Averaging data within GHA bins
        weights = np.zeros((len(level1), len(gha_edges) - 1, level1[0].freq.n))
        spectra = np.zeros((len(level1), len(gha_edges) - 1, level1[0].freq.n))

        pbar = tqdm.tqdm(enumerate(level1), unit="files", total=len(level1))
        for i, l1 in pbar:
            pbar.set_description(f"GHA Binning for {l1.filename.name}")

            gha = l1.ancillary["gha"]
            w = np.where(flags[i], 0, l1.weights) if flags is not None else None

            s = tools.non_stationary_bin_avg(data=l1.spectrum.T, x=gha, weights=w.T, bins=gha_edges)
            w = tools.get_binned_weights(x=gha, bins=gha_edges, weights=weights.T)

            weights[i] = w.T
            spectra[i] = s.T

        return spectra, weights, gha_edges

    def plot_daily_residuals(
        self,
        n_terms: int = 5,
        gha_model_type: str = "polynomial",
        freq_model_type: str = "polynomial",
        separation: float = 1,
        ax: [None, plt.Axes] = None,
        gha_min: float = 0,
        gha_max: float = 24,
        freq_kwargs: [Dict, None] = None,
        gha_kwargs: [Dict, None] = None,
    ) -> plt.Axes:
        """
        Make a single plot of residuals for each day in the dataset.

        Parameters
        ----------
        n_terms
            Number of terms in each fit.
        gha_model_type
            The kind of model to fit to the GHA structure (in order to average).
        freq_model_type
            The kind of model to fit to the frequency spectra (to get residuals).
        separation
            The separation between residuals in K (on the plot).
        ax
            An optional axis on which to plot.
        gha_min
            A minimum GHA to include in the averaged residuals.
        gha_max
            A maximum GHA to include in the averaged residuals.
        freq_kwargs
            Arguments to the model fit to frequency
        gha_kwargs
            Arguments to the model fit to GHA.

        Returns
        -------
        ax
            The matplotlib Axes on which the plot is made.
        """
        freq_kwargs = freq_kwargs or {}
        gha_kwargs = gha_kwargs or {}

        gha = (self.ancillary["gha_edges"][1:] + self.ancillary["gha_edges"][:-1]) / 2
        model_freq = mdl.Model._models[freq_model_type.lower()](
            n_terms=n_terms, default_x=self.freq.freq, **freq_kwargs
        )

        if ax is None:
            fig, ax = plt.subplots(1, 1, figsize=(7, 12))

        mask = (gha > gha_min) & (gha < gha_max)
        model_gha = mdl.Model._models[gha_model_type.lower()](
            n_terms=n_terms, default_x=gha[mask], **gha_kwargs
        )
        for ix, (spec, weight) in enumerate(zip(self.spectrum, self.weights)):
            mean_spec = tools.non_stationary_weighted_average(
                data=spec[mask].T, x=gha[mask], weights=weight[mask].T, model=model_gha
            )
            w = np.sum(weight, axis=0)
            fit = mdl.ModelFit(model_freq, ydata=mean_spec, weights=w)
            ax.plot(self.freq.freq, fit.residual - ix * separation)
            ax.text(
                self.freq.max + 10,
                ix * separation,
                f'{self.previous_level[ix].meta["day"]} RMS={fit.weighted_rms()}',
            )

        return ax

    def plot_waterfall(self, day=None, indx=None, quantity="flagged"):
        days = [l1.meta["day"] for l1 in self.previous_level]
        if day is not None:
            indx = days.index(day)

        if indx is None:
            raise ValueError("Must either supply 'day' or 'indx'")

        extent = (
            self.freq.min,
            self.freq.max,
            self.ancillary["gha_edges"].min(),
            self.ancillary["gha_edges"].max(),
        )
        if quantity == "flagged":
            plt.imshow(
                np.where(self.weights[indx] > 0, self.spectrum[indx], np.nan),
                aspect="auto",
                extent=extent,
            )
        else:
            plt.imshow(self.spectrum, aspect="auto", extent=extent)


class Level3(_Level):
    _structure = {
        "frequency": None,
        "spectra": {
            "weights": None,
            "spectrum": None,
        },  # All spectra components assumed to be the same shape, with last axis being frequency.
        "ancillary": {"std_dev": None, "years": None, "gha_edges": None},
        "meta": {},
    }

    @cached_property
    def calibration(self):
        """The calibration object used to calibrate these spectra."""
        return self.previous_level.calibration

    @classmethod
    def _from_prev_level(
        cls,
        level2: [Level2],
        day_range: Optional[Tuple[int, int]] = None,
        ignore_days: Optional[Sequence[int]] = None,
        f_low: Optional[float] = None,
        f_high: Optional[float] = None,
        freq_resolution: Optional[float] = None,
        gha_filter_file: [None, str, Path] = None,
        xrfi_pipe: [None, dict] = None,
    ):
        """
        Convert a level2 to a level3.

        This step integrates over days to form a spectrum as a function of GHA and
        frequency. It also applies an optional further frequency averaging.

        Parameters
        ----------
        level4 : :class:`Level4` instance
            The level4 object to convert.
        day_range : 2-tuple
            Min and max days to include (from a given year).
        ignore_days : sequence
            A sequence of days to ignore in the integration.
        f_low : float, optional
            A minimum frequency to use.
        f_high : float, optional
            A maximum frequency to use.
        freq_resolution : float, optional
            A frequency resolution to average down to.

        Returns
        -------
        data : dict
            Consisting of ``spectrum``, ``weights`` and ``frequency``. Both spectrum and
            weights are 2D, with the first axis the length of the GHA bins in the Level4
            object.
        ancillary : dict
            Consisting of
            * ``std_dev``: The standard deviation in each bin of (GHA, frequency)
            * ``years``: An array of years in which all observations in the dataset were
              taken.
        meta : dict
            Consisting of
            * ``day_range``: The user-specified allowed range of days going into the
              integration.
            * ``ignore_days``: The user-specified list of explicit days ignored in the
              integration.
        """
        xrfi_pipe = xrfi_pipe or {}

        # Compute the residuals
        days = level2.ancillary["days"]
        freq = FrequencyRange(level2.raw_frequencies, f_low=f_low, f_high=f_high)

        if day_range is None:
            day_range = (days.min(), days.max())

        if ignore_days is None:
            ignore_days = []

        if gha_filter_file:
            with open(gha_filter_file, "r") as fl:
                # gha filter is a dict of {gha: days} specifying a list of days to
                # *keep* for each GHA.
                gha_filter = yaml.load(fl, Loader=yaml.FullLoader)

            spec, wght = [], []
            gha_edges = level2.ancillary["gha_edges"]
            for i, (low, high) in enumerate(zip(gha_edges[:-1], gha_edges[1:])):
                gha_range = range(int(np.floor(low)), int(np.floor(high)))

                # a list of lists of good days for all integer GHA in this range.
                good_days = [gha_filter[gha] for gha in gha_range]

                for j, day in enumerate(days):
                    if all(day in g for g in good_days) and day not in ignore_days:
                        spec.append(level2.spectrum[j, i, freq.mask])
                        wght.append(level2.weights[j, i, freq.mask])

            spec = np.array(spec)
            wght = np.array(wght)

        else:
            spec = level2.spectrum[:, :, freq.mask]
            wght = level2.weights[:, :, freq.mask]

        # Perform xRFI on GHA-averaged spectra.
        if xrfi_pipe:
            for s, w in zip(spec, wght):
                tools.run_xrfi_pipe(s, w <= 0, xrfi_pipe)

        # Take mean over nights.
        spec, wght = tools.weighted_mean(spec, wght, axis=0)

        if freq_resolution:
            bins = np.arange(freq.min, freq.max, freq_resolution)
            spec = tools.non_stationary_bin_avg(data=spec, x=freq.freq, weights=wght, bins=bins)
            wght = tools.get_binned_weights(x=freq.freq, bins=bins, weights=wght)
            f = (bins[1:] + bins[:-1]) / 2

        data = {
            "spectrum": spec,
            "weights": wght,
        }

        ancillary = {
            "years": np.unique(level2.ancillary["years"]),
            "gha_edges": level2.ancillary["gha_edges"],
        }
        meta = {
            "day_range": day_range,
            "ignore_days": ignore_days,
            "gha_filter_file": gha_filter_file or "",
            "xrfi_pipe": xrfi_pipe,
            **level2.meta,
        }

        return f, data, ancillary, meta

    @property
    def gha_edges(self):
        """The edges of the GHA bins."""
        return self.ancillary["gha_edges"]

    def plot_waterfall(self, quantity="flagged"):
        extent = (self.freq.min, self.freq.max, self.gha_edges.min, self.gha_edges.max)
        if quantity == "flagged":
            plt.imshow(np.where(self.weights > 0, self.spectrum, 0), extent=extent, aspect="auto")
        else:
            plt.imshow(getattr(self, quantity), extent=extent, aspect="auto")

        plt.xlabel("Frequency")
        plt.ylabel("GHA")

    def bin_gha(self, gha_bins=None) -> Tuple[np.ndarray, np.ndarray]:
        gha_centres = (self.gha_edges[:-1] + self.gha_edges[1:]) / 2
        if gha_bins is None:
            gha_bins = [self.gha_edges.min(), self.gha_edges.max()]

        s = tools.non_stationary_bin_avg(
            data=self.spectrum.T,
            x=gha_centres,
            bins=gha_bins,
            weights=self.weights.T,
        )
        w = tools.get_binned_weights(x=gha_centres, bins=gha_bins, weights=self.weights.T)

        return s, w


class Level4(_Level):
    """
    A Level-4 Calibrated Spectrum.

    This step performs a final average over GHA to yield a simple spectrum of
    frequency. It also allows a final averaging in frequency after the GHA average.

    Parameters
    ----------
    level3 : :class:`Level5` instance
        The Level3 object to convert.
    f_low : float, optional
        The min frequency to keep in the final average.
    f_high : float, optional
        The max frequency to keep in the final average.
    ignore_freq_ranges : list of tuple, optional
        An optional list of 2-tuples specifying frequency ranges to omit in the final
        average (they are weighted to zero).
    freq_resolution : float, optional
        An optional frequency resolution down to which to average.

    Returns
    -------
    data : dict
        Consisting of ``spectrum``, ``weights`` and ``frequency``. In this step,
        all of these are 1D and the same shape.
    ancillary : dict
        Consisting of
        * ``std_dev``: The standard deviation of the spectrum in each bin.
    meta : dict
        Consisting of all level 5 meta info, plus
        * ``ignore_freq_ranges``: the list of ignored frequency ranges in the object.
    """

    _structure = {
        "frequency": None,
        "spectra": {
            "weights": None,
            "spectrum": None,
        },  # All spectra components assumed to be the same shape, with last axis being frequency.
        "ancillary": {
            "std_dev": None,
        },
        "meta": {},
    }

    @cached_property
    def calibration(self):
        """The calibration object used to calibrate these spectra."""
        return self.previous_level.calibration

    @classmethod
    def _from_prev_level(
        cls,
        level3: [Level3],
        f_low: Optional[float] = None,
        f_high: Optional[float] = None,
        ignore_freq_ranges: Optional[Sequence[Tuple[float, float]]] = None,
        freq_resolution: Optional[float] = None,
        xrfi_pipe: [None, dict] = None,
        gha_min=0,
        gha_max=24,
    ):
        xrfi_pipe = xrfi_pipe or {}
        meta = {
            "ignore_freq_ranges": ignore_freq_ranges,
            "xrfi_pipe": xrfi_pipe,
            "gha_min": gha_min,
            "gha_max": gha_max,
            **level3.meta,
        }

        freq = FrequencyRange(level3.raw_frequencies, f_low=f_low, f_high=f_high)

        spec = level3.spectrum[:, freq.mask]
        wght = level3.weights[:, freq.mask]

        # Another round of XRFI
        tools.run_xrfi_pipe(spec, wght, xrfi_pipe)

        if ignore_freq_ranges:
            for (low, high) in ignore_freq_ranges:
                wght[:, (freq.freq >= low) & (freq.freq <= high)] = 0

        if freq_resolution:
            bins = np.arange(freq.min, freq.max, freq_resolution)
            spec = tools.non_stationary_bin_avg(data=spec, x=freq.freq, weights=wght, bins=bins)
            wght = tools.get_binned_weights(x=freq.freq, bins=bins, weights=wght)
            f = (bins[1:] + bins[:-1]) / 2
        else:
            f = freq.freq

        mask = (level3.gha_edges[:-1] >= gha_min) & (level3.gha_edges[1:] <= gha_max)
        spec, wght = tools.weighted_mean(spec[mask], wght[mask], axis=0)

        data = {"spectrum": spec, "weights": wght}

        ancillary = {"gha_edges": level3.gha_edges[:-1][mask]}

        return f, data, ancillary, meta
