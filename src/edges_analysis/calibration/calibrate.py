"""Module defining calibration routines for field data in EDGES."""
from __future__ import annotations

import glob
import hickle
import numpy as np
import os
import re
from astropy.time import Time
from datetime import datetime
from edges_cal import types as tp
from edges_cal.cal_coefficients import CalibrationObservation, Calibrator
from pathlib import Path

from .. import beams, const
from .. import coordinates as coords
from ..config import config
from ..gsdata import GSData, gsregister
from . import loss
from .labcal import LabCalibration


@gsregister("calibrate")
def dicke_calibration(data: GSData) -> GSData:
    """Calibrate field data using the Dicke switch data."""
    iant = data.loads.index("ant")
    iload = data.loads.index("internal_load")
    ilns = data.loads.index("internal_load_plus_noise_source")

    q = (data.data[iant] - data.data[iload]) / (data.data[ilns] - data.data[iload])

    return data.update(
        data=q[np.newaxis],
        data_unit="uncalibrated",
        time_array=data.time_array[:, [iant]],
        loads=("ant",),
        nsamples=data.nsamples[[iant]],
        flags={
            name: np.any(flag, axis=0)[np.newaxis] for name, flag in data.flags.items()
        },
        data_model=None,
    )


@gsregister("calibrate")
def approximate_temperature(data: GSData, *, tload: float, tns: float):
    """Convert an uncalibrated object to an uncalibrated_temp object.

    This uses a guess for T0 and T1 that provides an approximate temperature spectrum.
    One does not need this step to perform actual calibration, and if actual calibration
    is done following applying this function, you will need to provide the same tload
    and tns as used here.
    """
    if data.data_unit != "uncalibrated":
        raise ValueError(
            "data_unit must be 'uncalibrated' to calculate approximate temperature"
        )
    return data.update(
        data=data.data * tns + tload, data_unit="uncalibrated_temp", data_model=None
    )


def get_default_s11_directory(band: str) -> Path:
    """Get the default S11 directory for this observation."""
    return Path(config["paths"]["raw_field_data"]) / "mro" / band / "s11"


def _get_closest_s11_time(
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
        the actual filenames. So for example, you could specify
        ``ignore_files=['2020_076']`` and it will ignore the file
        ``/home/user/data/2020_076_01_02_input1.s1p``. Full regex can be used.
    """
    if isinstance(time, Time):
        time = time.to_datetime()
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

    if "d" not in dct and "jd" not in dct:
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
            dt = coords.dt_from_jd(
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
            f"No files found matching the input pattern. Available files: "
            f"{[fl.name for fl in files]}. Regex pattern: {p.pattern}. "
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


def get_s11_paths(
    s11_path: str | Path | tuple | list,
    band: str,
    begin_time: datetime,
    s11_file_pattern: str,
    ignore_files: list[str] | None = None,
):
    """Given an s11_path, return list of paths for each of the inputs."""
    # If we get four files, make sure they exist and pass them back
    if isinstance(s11_path, (tuple, list)):
        if len(s11_path) != 4:
            raise ValueError(
                "If passing explicit paths to S11 inputs, length must be 4."
            )

        fls = []
        for pth in s11_path:
            p = Path(pth).expanduser().absolute()
            assert p.exists()
            fls.append(p)

        return fls
    if (os.path.isfile(s11_path)) and (not s11_path.endswith("s1p")):
        return [s11_path]
    # Otherwise it must be a path.
    s11_path = Path(s11_path).expanduser()

    if str(s11_path).startswith(":"):
        s11_path = get_default_s11_directory(band) / str(s11_path)[1:]

    if s11_path.is_dir():
        # Get closest measurement
        return _get_closest_s11_time(
            s11_path, begin_time, s11_file_pattern, ignore_files=ignore_files
        )
    # The path *must* have an {input} tag in it which we can search on
    fls = glob.glob(str(s11_path).format(input="?"))
    assert (
        len(fls) == 4
    ), f"There are not exactly four files matching {s11_path}. Found: {fls}."

    return sorted(Path(fl) for fl in fls)


def get_labcal(
    calobs: Calibrator,
    s11_path: str | Path | tuple | list,
    band: str,
    begin_time: datetime,
    s11_file_pattern: str,
    ignore_s11_files: list[str] | None = None,
    antenna_s11_n_terms: int = 15,
):
    """Given an s11_path, return list of paths for each of the inputs."""
    # If we get four files, make sure they exist and pass them back

    s11_files = get_s11_paths(
        s11_path,
        band,
        begin_time,
        s11_file_pattern,
        ignore_files=ignore_s11_files,
    )

    if not isinstance(calobs, Calibrator):
        try:
            calobs = hickle.load(calobs)
            if isinstance(calobs, CalibrationObservation):
                calobs = calobs.to_calibrator()
        except Exception:
            pass
    if not isinstance(calobs, Calibrator):
        try:
            calobs = Calibrator.from_calfile(calobs)
        except Exception:
            calobs = Calibrator.from_old_calfile(calobs)

    return LabCalibration.from_s11_files(
        calobs=calobs,
        s11_files=s11_files,
        n_terms=antenna_s11_n_terms,
    )


@gsregister("calibrate")
def apply_noise_wave_calibration(
    data: GSData,
    calobs: Calibrator | Path,
    band: str,
    s11_path: str | Path,
    s11_file_pattern: str = r"{y}_{jd}_{h}_*_input{input}.s1p",
    ignore_s11_files: list[str] | None = None,
    antenna_s11_n_terms: int = 15,
    tload: float | None = None,
    tns: float | None = None,
) -> GSData:
    """Apply noise-wave calibration to data.

    This function requires a :class:`edges_cal.cal_coefficients.Calibrator` object
    (or a path to a file containing such an object) which must be created beforehand.
    The antenna S11 used is found automatically by searching for the file that has
    the closest match to the time of the data. This can be constrained by passing
    options that match the regex pattern for the S11 files.

    Parameters
    ----------
    data
        Data to be calibrated.
    calobs
        Calibrator object or path to file containing calibrator object.
    band
        The band that the data is in (eg. low, mid, high). Used for auto-finding
        S11 files.
    s11_path
        Path to directory containing Antenna S11 files.
    s11_file_pattern
        The format-pattern used to search for the S11 files in ``s11_path``.
        This can be used to limit the search to a specific time.
    ignore_s11_files
        A list of S11 files to ignore in the search.
    antenna_s11_n_terms
        The number of terms to use in the antenna S11 model.
    """
    if data.data_unit not in ("uncalibrated", "uncalibrated_temp"):
        raise ValueError("Data must be uncalibrated to apply calibration!")

    if data.data_unit == "uncalibrated_temp" and (tload is None or tns is None):
        raise ValueError(
            "You need to supply tload and tns if data_unit is uncalibrated_temp"
        )

    if data.nloads != 1:
        raise ValueError("Can only apply noise-wave calibration to single load data!")

    labcal = get_labcal(
        calobs=calobs,
        s11_path=s11_path,
        band=band,
        begin_time=data.time_array.min(),
        s11_file_pattern=s11_file_pattern,
        ignore_s11_files=ignore_s11_files,
        antenna_s11_n_terms=antenna_s11_n_terms,
    )

    if data.data_unit == "uncalibrated_temp":
        q = (data.data - tload) / tns
    else:
        q = data.data
    new_data = labcal.calibrate_q(q, freq=data.freq_array)
    return data.update(data=new_data, data_unit="temperature", data_model=None)


@gsregister("calibrate")
def apply_loss_correction(
    data: GSData,
    band: str,
    ambient_temp: np.ndarray | float | str = None,
    antenna_correction: tp.PathLike | None = ":",
    configuration="",
    balun_correction: tp.PathLike | None = ":",
    ground_correction: tp.PathLike | None | float = ":",
    s11_path: str | Path = None,
    calobs: Calibrator | Path = None,
    s11_file_pattern: str = r"{y}_{jd}_{h}_*_input{input}.s1p",
    ignore_s11_files: list[str] | None = None,
    antenna_s11_n_terms: int = 15,
) -> GSData:
    """Apply antenna, balun and ground loss corrections.

    Parameters
    ----------
    data
        Data to be calibrated.
    band
        The band that the data is in (eg. low, mid, high). Used for auto-finding loss
        model files.
    ambient_temp
        The ambient temperature of the data. If not provided, the temperature is looked
        for in the data's ``auxiliary_measurements`` attribute, under the
        ``ambient_temp`` key. You can specify this as a different key, or simply provide
        a float or array (with the same length as the time dimension of the data).
    antenna_correction
        Path to file containing antenna loss correction coefficients. A file will be
        located automatically if set to ``":"``.
    configuration
        The configuration of the antenna (eg. "45" for low-45).
    balun_correction
        Path to file containing balun loss correction coefficients. A file will be
        located automatically if set to ``":"``.
    ground_correction
        Path to file containing ground loss correction coefficients. A file will be
        located automatically if set to ``":"``.
    s11_path
        Path to directory containing Antenna S11 files.
    s11_file_pattern
        The format-pattern used to search for the S11 files in ``s11_path``.
        This can be used to limit the search to a specific time.
    ignore_s11_files
        A list of S11 files to ignore in the search.
    antenna_s11_n_terms
        The number of terms to use in the antenna S11 model.
    calobs
        Calibrator object or path to file containing calibrator object.
    """
    if data.data_unit != "temperature":
        raise ValueError("Data must be temperature to apply antenna loss correction!")

    if ambient_temp is None:
        ambient_temp = "ambient_temp"

    if isinstance(ambient_temp, str):
        ambient_temp = data.auxiliary_measurements.get(ambient_temp)

    if ambient_temp is None:
        raise ValueError("Ambient temperature must be provided or stored in data!")

    if not hasattr(ambient_temp, "__len__"):
        ambient_temp = ambient_temp * np.ones(len(data.time_array))

    f = data.freq_array.to_value("MHz")
    gain = np.ones_like(f)

    if antenna_correction:
        gain *= loss.antenna_loss(
            antenna_correction, f, band=band, configuration=configuration
        )

    # Balun+Connector Loss
    if balun_correction:
        labcal = get_labcal(
            calobs=calobs,
            s11_path=s11_path,
            band=band,
            begin_time=data.time_array.min(),
            s11_file_pattern=s11_file_pattern,
            ignore_s11_files=ignore_s11_files,
            antenna_s11_n_terms=antenna_s11_n_terms,
        )
        balun_gain, connector_gain = loss.balun_and_connector_loss(
            band, f, labcal.antenna_s11_model(data.freq_array)
        )
        gain *= balun_gain * connector_gain

    # Ground Loss
    if isinstance(ground_correction, (str, Path)):
        gain *= loss.ground_loss(
            ground_correction, f, band=band, configuration=configuration
        )
    elif isinstance(ground_correction, float):
        gain *= ground_correction

    a = ambient_temp + const.absolute_zero if ambient_temp[0] < 200 else ambient_temp
    spec = (data.data - np.outer(a, (1 - gain))) / gain

    return data.update(data=spec, data_unit="temperature", data_model=None)


@gsregister("calibrate")
def apply_beam_correction(
    data: GSData,
    band: str | None = None,
    beam_file: tp.PathLike | None = ":",
    gha_min: float | None = None,
    gha_max: float | None = None,
    time_resolution: int | None = None,
    average_before_correction: bool = True,
    beam_factor_file: tp.PathLike | None = ":",
) -> GSData:
    """Apply beam correction to the data.

    Parameters
    ----------
    data
        Data to be calibrated.
    band
        The band that the data is in (eg. low, mid, high). Used for auto-finding beam
        file (if one exists in the standard location).
    beam_file
        Path to file containing beam correction coefficients. If there is an existing
        beam file in the standard location, this can be set to ``":"``.
    gha_min
        The minimum GHA to use for the beam correction. If not provided, the GHA is
        calculated from the LST bins of the data. If the data has multiple LST bins,
        and ``average_before_correction`` is ``True``, then this cannot be provided.
    gha_max
        The maximum GHA to use for the beam correction. If not provided, the GHA is
        calculated from the LST bins of the data. If the data has multiple LST bins,
        and ``average_before_correction`` is ``True``, then this cannot be provided.
    time_resolution
        The time resolution to use for the beam correction. If not provided, the
        resolution is calculated from the LST bins of the data. If the data has multiple
        LST bins, and ``average_before_correction`` is ``True``, then this cannot be
        provided.
    average_before_correction
        Whether to average the beam correction across the time dimension before
        applying it to the data.
    """
    beam_fac = beams.InterpolatedBeamFactor.from_beam_factor(
        beam_file, band=band, f_new=data.freq_array
    )

    if not (
        (gha_min is None and gha_max is None and time_resolution is None)
        or (gha_min is not None and gha_max is not None and time_resolution is not None)
    ):
        raise ValueError(
            "All of gha_min, gha_max and time_resolution must be provided, if any!"
        )
    if beam_factor_file is not None:
        alan_beam_factor = np.genfromtxt(beam_factor_file)
        bf = alan_beam_factor[:, 3]
        return data.update(data=data.data / bf, data_model=None)
    if not average_before_correction:
        if gha_min is not None:
            raise ValueError(
                "gha_min, gha_max and time_resolution cannot be provided if "
                "average_before_correction is False!"
            )

        bf = beam_fac.evaluate(data.lst_array.hour)
        return data.update(data=data.data / bf, data_model=None)
    else:
        if gha_min is not None:
            gha_min %= 24
            gha_max %= 24
            while gha_max < gha_min:
                gha_max += 24

            gha_list = np.arange(gha_min, gha_max, time_resolution)
            lst_list = coords.gha2lst(gha_list)
            bf = beam_fac.evaluate(lst_list)
        else:
            bf = beam_fac.evaluate(data.lst_array.hour)
        return data.update(data=data.data / np.average(bf, axis=0), data_model=None)
