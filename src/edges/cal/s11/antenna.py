"""Functions for creating CalibratedS11 objects from antenna S11 measurement files."""

import contextlib
import glob
import re
import warnings
from datetime import datetime
from pathlib import Path

import numpy as np
from astropy.time import Time

from edges import types as tp
from edges.io import CalkitFileSpec, LoadS11, SwitchingState

from .base import CalibratedS11
from .cal_loads import get_loads11_from_load_and_switch


def _get_closest_s11_time(
    s11_dir: Path,
    time: Time,
    fileglob: str = "*_input*.s1p",
    dateformat: str = "%Y_%j_%H",
    date_slice: slice = slice(None),
    ignore_files=None,
) -> list[Path]:
    """From a given filename pattern, within a directory, find file closest to time.

    Parameters
    ----------
    s11_dir
        The directory in which to search for S11 files.
    time
        The time to find the closest match to.
    fileglob
        A string specifying a glob pattern which will yield the set of files
        to consider in determining which is closest to the time.
    dateformat
        A format string that is parseable by datetime.strptime that will
        be used to parse the date/time of each file. The part of the filename
        that is parsed is given by `date_slice`.
    date_slice
        A slice specifying the range of characters in the filename to use for
        parsing the date.
    ignore_files : list, optional
        A list of file patterns to ignore. They need only partially match
        the actual filenames. So for example, you could specify
        ``ignore_files=['2020_076']`` and it will ignore the file
        ``/home/user/data/2020_076_01_02_input1.s1p``. Full regex can be used.
    """
    if isinstance(time, datetime):
        time = Time(time)

    if isinstance(ignore_files, str):
        ignore_files = [ignore_files]

    ignore = [re.compile(ign) for ign in (ignore_files or [])]

    files = sorted(
        fl
        for fl in s11_dir.glob(fileglob)
        if all(ign.search(str(fl)) is None for ign in ignore)
    )

    if not files:
        raise FileNotFoundError(
            f"No files found matching the input pattern. Available files: "
            f"{[fl.name for fl in files]}. Regex pattern: {fileglob}. "
        )

    fnames = [fl.name[date_slice] for fl in files]
    times = []
    good_idx = []
    for i, fname in enumerate(fnames):
        with contextlib.suppress(ValueError):
            times.append(Time.strptime(fname, dateformat))
            good_idx.append(i)
    times = Time(times)

    if len(good_idx) == 0:
        raise ValueError("No files were parseable as datetimes.")

    if len(good_idx) != len(files):
        warnings.warn(
            f"Only {len(good_idx)} of {len(files)} were parseable.", stacklevel=2
        )

    files = [fl for i, fl in enumerate(files) if i in good_idx]

    time_diffs = np.abs((times - time).sec)
    indices = np.where(time_diffs == time_diffs.min())[0]

    # Gets a representative closest time file
    closest = [files[i] for i in indices]

    if len(indices) != 4:
        raise FileNotFoundError(
            "There need to be four input S1P files of the same time, got "
            f"{len(indices)}: {closest}"
        )

    return sorted(closest)


def get_antenna_s11_paths(
    s11_path: str | Path | tuple | list,
    time: Time | datetime | None = None,
    fileglob: str = "*_input*.s1p",
    dateformat: str = "%Y_%j_%H",
    date_slice: slice = slice(None),
    ignore_files=None,
):
    """Given an s11_path, return list of paths for each of the four standards.

    In general, the antenna S11 is defined -- like all calibration loads -- by
    four calkit readings (short open load external). This function finds the four
    files associated with these readings in a unified interface. One of several modes
    can be used:

    1. `s11_path` is a list of four files. In this case, the four files are simply
       returned.
    2. `s11_path` is a glob pattern with a format-specified "{load}" in it. In this
       case, the pattern will be searched, with "{load}" replaced by `?`. This must
       result in 4 files.
    3. `s11_path` is a directory, in which case the directory will be searched
       for all files matching the `fileglob`, and each will be parsed for a datetime
       using the `dateformat` specifier. The four files with matching times closest
       to `time` will be returned.
    4. `s11_path` is a single file whose suffix is not .s1p. In this case, a list of
       just that file will be returned, under the assumption that the file represents
       a pre-calibrated S11.
    """
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

    # Check if s11_file is a glob pattern with {load} in it.
    if isinstance(s11_path, str) and "{load}" in s11_path:
        fls = glob.glob(str(s11_path).format(load="?"))
        if len(fls) != 4:
            raise FileNotFoundError(
                f"There are not exactly four files matching {s11_path}. Found: {fls}."
            )
        return sorted(Path(fl) for fl in fls)

    # Otherwise it must be a path.
    s11_path = Path(s11_path).expanduser()

    if s11_path.is_file() and s11_path.suffix != ".s1p":
        return [s11_path]

    if s11_path.is_dir():
        # Get closest measurement
        return _get_closest_s11_time(
            s11_path,
            time=time,
            ignore_files=ignore_files,
            dateformat=dateformat,
            date_slice=date_slice,
            fileglob=fileglob,
        )

    raise ValueError(f"s11_path {s11_path} not in a recognized format")


def get_ants11_from_edges2_files(
    files: tuple[tp.PathLike, tp.PathLike, tp.PathLike, tp.PathLike],
    switchdef: SwitchingState,
    **kwargs,
) -> CalibratedS11:
    """Construct a CalibratedS11 for the antenna measurements."""
    return get_loads11_from_load_and_switch(
        loaddef=LoadS11(
            calkit=CalkitFileSpec(open=files[0], short=files[1], match=files[2]),
            external=files[3],
        ),
        switchdef=switchdef,
        **kwargs,
    )
