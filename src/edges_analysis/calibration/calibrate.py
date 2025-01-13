"""Module defining calibration routines for field data in EDGES."""

from __future__ import annotations

import glob
import re
from datetime import datetime
from pathlib import Path

import hickle
import numpy as np
from astropy import units as un
from astropy.time import Time
from edges_cal import modelling as mdl
from edges_cal import types as tp
from edges_cal.calobs import (
    CalibrationObservation,
    Calibrator,
)
from pygsdata import GSData, gsregister

from .. import beams
from .. import coordinates as coords
from ..config import config
from .labcal import LabCalibration
from .s11 import AntennaS11


@gsregister("calibrate")
def dicke_calibration(data: GSData) -> GSData:
    """Calibrate field data using the Dicke switch data.

    Assumes that the data has the three loads "ant", "internal_load" and
    "internal_load_plus_noise_source". The data is calibrated using the
    Dicke switch model (i.e.
    ``(ant - internal_load)/(internal_load_plus_noise_source - internal_load)``).
    """
    iant = data.loads.index("ant")
    iload = data.loads.index("internal_load")
    ilns = data.loads.index("internal_load_plus_noise_source")

    with np.errstate(divide="ignore", invalid="ignore"):
        q = (data.data[iant] - data.data[iload]) / (data.data[ilns] - data.data[iload])

    return data.update(
        data=q[np.newaxis],
        data_unit="uncalibrated",
        times=data.times[:, [iant]],
        time_ranges=data.time_ranges[:, [iant]],
        effective_integration_time=data.effective_integration_time[[iant]],
        lsts=data.lsts[:, [iant]],
        lst_ranges=data.lst_ranges[:, [iant]],
        loads=("ant",),
        nsamples=data.nsamples[[iant]],
        flags={name: flag.any(axis="load") for name, flag in data.flags.items()},
        residuals=None,
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
        data=data.data * tns + tload,
        data_unit="uncalibrated_temp",
        residuals=data.residuals * tns if data.residuals is not None else None,
    )


def get_default_s11_directory(band: str) -> Path:
    """Get the default S11 directory for this observation."""
    return Path(config["paths"]["raw_field_data"]) / "mro" / band / "s11"


def _get_closest_s11_time(
    s11_dir: Path,
    time: datetime,
    s11_file_pattern: str = "{y}_{jd}_{h}_*_input{input}.s1p",
    ignore_files=None,
) -> list[Path]:
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

    if isinstance(ignore_files, str):
        ignore_files = [ignore_files]

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
    dct = {d: v for d, v in dct.items() if f"{{{d}}}" in s11_file_pattern}

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

    if len(closest) != 4:
        raise FileNotFoundError(
            f"There need to be four input S1P files of the same time, got {closest}."
        )
    return sorted(closest)


def get_s11_paths(
    s11_path: str | Path | tuple | list,
    band: str | None = None,
    begin_time: datetime | None = None,
    s11_file_pattern: str = "{y}_{jd}_{h}_*_input{input}.s1p",
    ignore_files: list[str] | None = None,
):
    """Given an s11_path, return list of paths for each of the inputs."""
    # If we get four files, make sure they exist and pass them back
    if isinstance(s11_path, tuple | list):
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

    s11_path = Path(s11_path).expanduser()

    if s11_path.is_file() and s11_path.suffix != ".s1p":
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
    # The path *must* have an {load} tag in it which we can search on
    fls = glob.glob(str(s11_path).format(load="?"))
    if len(fls) != 4:
        raise FileNotFoundError(
            f"There are not exactly four files matching {s11_path}. Found: {fls}."
        )

    return sorted(Path(fl) for fl in fls)


def get_labcal(
    calobs: Calibrator,
    s11_path: str | Path | tuple | list | None = None,
    ant_s11_object: str | Path | AntennaS11 | None = None,
    band: str | None = None,
    begin_time: datetime | None = None,
    s11_file_pattern: str | None = None,
    ignore_s11_files: list[str] | None = None,
    antenna_s11_n_terms: int = 15,
    **kwargs,
) -> LabCalibration:
    """Get a LabCalibration object from the given inputs.

    Parameters
    ----------
    calobs : Calibrator or str or Path or hickle file
        The calibrator object to use for calibration.
    s11_path : str or Path or tuple or list, optional
        The path to the S11 files. If a tuple or list, it must contain four paths to
        the four S11 files. If None, the S11 files will be searched for in the default
        directory.
    band : str, optional
        The band that the data is in (eg. low, mid, high). Used for auto-finding S11
        files.
    begin_time : datetime, optional
        The time at which the data begins. Used for auto-finding S11 files.
    s11_file_pattern : str, optional
        The format-pattern used to search for the S11 files in ``s11_path``. This can be
        used to limit the search to a specific time.
    ignore_s11_files : list, optional
        A list of S11 files to ignore in the search.
    antenna_s11_n_terms : int, optional
        The number of terms to use in the antenna S11 model
    """
    # If we get four files, make sure they exist and pass them back
    if s11_path is None and ant_s11_object is None:
        raise ValueError("Must provide either s11_path or ant_s11_object.")

    if not isinstance(calobs, Calibrator):
        try:
            calobs = hickle.load(calobs)
            if isinstance(calobs, CalibrationObservation):
                calobs = calobs.to_calibrator()
        except Exception:
            pass
    if not isinstance(calobs, Calibrator):
        calobs = Calibrator.from_calfile(calobs)

    if ant_s11_object is not None:
        if not isinstance(ant_s11_object, AntennaS11):
            ants11 = hickle.load(ant_s11_object)
        else:
            ants11 = ant_s11_object
        return LabCalibration(
            calobs=calobs,
            antenna_s11_model=ants11,
        )
    else:
        s11_files = get_s11_paths(
            s11_path,
            band,
            begin_time,
            s11_file_pattern,
            ignore_files=ignore_s11_files,
        )

        return LabCalibration.from_s11_files(
            calobs=calobs, s11_files=s11_files, n_terms=antenna_s11_n_terms, **kwargs
        )


@gsregister("calibrate")
def apply_noise_wave_calibration(
    data: GSData,
    calobs: Calibrator | Path,
    band: str | None = None,
    s11_path: str | Path | None = None,
    ant_s11_object: str | Path | None = None,
    s11_file_pattern: str = r"{y}_{jd}_{h}_*_input{input}.s1p",
    ignore_s11_files: list[str] | None = None,
    antenna_s11_n_terms: int = 15,
    tload: float | None = None,
    tns: float | None = None,
    **kwargs,
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

    if (
        data.data_unit == "uncalibrated_temp"
        and not isinstance(calobs, Calibrator)
        and (tload is None or tns is None)
    ):
        raise ValueError(
            "You need to supply tload and tns if data_unit is uncalibrated_temp"
        )

    if data.nloads != 1:
        raise ValueError("Can only apply noise-wave calibration to single load data!")

    labcal = get_labcal(
        calobs=calobs,
        s11_path=s11_path,
        band=band,
        begin_time=data.times.min(),
        s11_file_pattern=s11_file_pattern,
        ignore_s11_files=ignore_s11_files,
        antenna_s11_n_terms=antenna_s11_n_terms,
        ant_s11_object=ant_s11_object,
        **kwargs,
    )

    if data.data_unit == "uncalibrated_temp":
        q = (data.data - labcal.calobs.t_load) / labcal.calobs.t_load_ns
    else:
        q = data.data
    new_data = labcal.calibrate_q(q, freq=data.freqs)

    if data.model is not None:
        qmodel = (
            (data.model - tload) / tns
            if data.data_unit == "uncalibrated_temp"
            else data.model
        )
        resids = new_data - labcal.calibrate_q(qmodel, freq=data.freqs)
    else:
        resids = None

    return data.update(data=new_data, data_unit="temperature", residuals=resids)


@gsregister("calibrate")
def apply_loss_correction(
    data: GSData,
    ambient_temp: tp.TemperatureType,
    loss: np.ndarray | None = None,
    loss_function: callable | None = None,
    **kwargs,
) -> GSData:
    """Apply a loss-correction to data.

    Parameters
    ----------
    data
        The GSData object on which to apply the loss-correction.
    ambient_temp
        The ambient temperature at which to apply the loss-correction.
    loss
        An array of losses, where the size of the array must be equal to the number
        of frequencies in the data. If None, a loss function is used to compute
        the losses.
    loss_function
        A function to compute the loss. The function must accept an array of frequencies
        as its first argument, and may accept arbitrary other keyword arguments, which
        will can be passed as kwargs to this function. Either this or loss must be
        specified.

    Notes
    -----
    Loss functions can be stacked, either by multiplying the losses before passing
    them to this function, or by calling this function multiple times, once for each
    loss.
    """
    if loss is None and loss_function is None:
        raise ValueError("Either loss or loss_function must be provided!")

    if loss is None:
        loss = loss_function(data.freqs, **kwargs)

    if data.data_unit != "temperature":
        raise ValueError("Data must be temperature to apply antenna loss correction!")

    a = ambient_temp.to_value(un.K)
    spec = (data.data - np.outer(a, (1 - loss))) / loss

    return data.update(
        data=spec,
        data_unit="temperature",
        residuals=None,
    )


@gsregister("calibrate")
def apply_beam_factor_directly(data: GSData, beam_file: str | Path) -> GSData:
    """Apply a beam correction factor from a file directly to the data.

    This function multiplies the data by the beam correction factor
    from the provided beamfile. It handles data with a single load and updates the data
    unit to "temperature".

    Parameters
    ----------
    data
        The GSData object containing the data to correct.
    beamfile
        The path to the beamfile containing the correction factors. The correction
        factors should be in the fourth column of the csv file, and should have a size
        equal to the number of frequencies in the data.

    Returns
    -------
    data
        A new GSData object with the corrected data and residuals.

    Raises
    ------
    NotImplementedError
        If the data contains more than one load.
    """
    if len(data.loads) > 1:
        raise NotImplementedError(
            "Can only apply beam correction to data with a single load"
        )

    new_data = data.data.copy()
    resids = data.residuals.copy() if data.residuals is not None else None
    bf = np.loadtxt(beam_file)
    new_data *= bf[:, 3]
    if resids is not None:
        resids *= bf[:, 3]
    return data.update(data=new_data, residuals=resids, data_unit="temperature")


@gsregister("calibrate")
def apply_beam_correction(
    data: GSData,
    beam: str | Path | beams.BeamFactor,
    freq_model: mdl.Model,
    integrate_before_ratio: bool = True,
    oversample_factor: int = 5,
    resample_beam_lsts: bool = True,
) -> GSData:
    """Apply beam correction to the data.

    This always applies the beam correction to each time sample in the data. If you want
    to average the data *before* applying the beam correction, you must average the data
    before applying this function to it. The input beam factor object should cover the
    full range of LSTs included in the data itself.

    The beam factor object is defined at a set of LSTs, and by default, the correction
    applied to the data is the *average* beam factor in each LST-bin of the data.
    To use the beam factor *interpolated* to the LSTs of the data instead, set
    ``interpolate_to_lsts`` to True.

    There are two ways to define the average beam factor within an LST-bin: either
    by taking the mean of ratios (of beam-weighted foreground model to *reference*
    beam-weighted foreground model) or the ratio of means. Switch between these
    by using the ``integrate_before_ratio`` parameter.

    Parameters
    ----------
    data
        Data to be calibrated.
    beam
        Either a path to a file containing beam correction coefficients, or the
        BeamFactor object itself.
    freq_model
        The (linear) model to use when evaluating the beam factor at the data freqs.
    integrate_before_ratio
        Whether to integrate (over time) the beam-weighted sky temperature and the
        reference sky temperature individually before taking their ratio to get
        the beam factor.
    oversample_factor
        The number of LST samples to use when interpolating the beam factor to the
        LSTs of the data. For every data LST, ``oversample_factor`` LSTs will be
        interpolated to (regularly spaced between each data LST, regardless of whether
        the data LSTs are regular). This is only used if ``resample_beam_lsts`` is True.
    resample_beam_lsts
        Whether to resample LSTs before averaging (by ``oversample_factor``).
    """
    if isinstance(beam, str | Path):
        beam = hickle.load(beam)

    if len(data.loads) > 1:
        raise NotImplementedError(
            "Can only apply beam correction to data with a single load"
        )

    if len(beam.lsts) < 4 and resample_beam_lsts:
        raise ValueError(
            "Your beam has a single LST so you cannot interpolate over LSTs."
        )

    if resample_beam_lsts:
        new_beam_lsts = []
        for lst0, lst1 in data.lst_ranges[:, 0, :]:
            lst1 = lst1.hour
            if lst1 < lst0.hour:
                lst1 = lst1 + 24

            new_beam_lsts.append(
                np.linspace(lst0.hour, lst1, oversample_factor + 1)[:-1]
            )
        new_beam_lsts = np.concatenate(new_beam_lsts)
        beam = beam.at_lsts(new_beam_lsts)

    new_data = data.data.copy()

    resids = data.residuals.copy() if data.residuals is not None else None

    for i, (lst0, lst1) in enumerate(data.lst_ranges[:, 0, :]):
        new = beam.between_lsts(lst0.hour, lst1.hour)
        if integrate_before_ratio:
            bf = new.get_integrated_beam_factor(
                model=freq_model, freqs=data.freqs.to_value("MHz")
            )
        else:
            bf = new.get_mean_beam_factor(
                model=freq_model, freqs=data.freqs.to_value("MHz")
            )

        new_data[:, :, i] /= bf
        if resids is not None:
            resids[:, :, i] /= bf

    return data.update(data=new_data, residuals=resids, data_unit="temperature")
