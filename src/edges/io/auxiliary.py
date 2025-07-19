"""Module defining EDGES-specific reading functions for weather and auxiliary data."""

from __future__ import annotations

import re
import warnings
from datetime import UTC, datetime, timedelta
from pathlib import Path

import numpy as np

_NEW_WEATHER_PATTERN = re.compile(
    r"(?P<year>\d{4}):(?P<day>\d{3}):(?P<hour>\d{2}):(?P<minute>\d{2}):(?P<second>\d{2})  "
    r"rack_temp  (?P<rack_temp>\d{3}.\d{2}) Kelvin, "
    r"ambient_temp  (?P<ambient_temp>\d{3}.\d{2}) Kelvin, "
    r"ambient_hum  (?P<ambient_hum>[\d\- ]{3}.\d{2}) percent, "
    r"frontend  (?P<frontend_temp>\d{3}.\d{2}) Kelvin, "
    r"rcv3_lna  (?P<lna_temp>\d{3}.\d{2}) Kelvin"
)

_OLD_WEATHER_PATTERN = re.compile(
    r"(?P<year>\d{4}):(?P<day>\d{3}):(?P<hour>\d{2}):(?P<minute>\d{2}):(?P<second>\d{2})  "
    r"rack_temp  (?P<rack_temp>\d{3}.\d{2}) Kelvin, "
    r"ambient_temp  (?P<ambient_temp>\d{3}.\d{2}) Kelvin, "
    r"ambient_hum  (?P<ambient_hum>[\d\- ]{3}.\d{2}) percent, "
)

_THERMLOG_PATTERN = re.compile(
    r"(?P<year>\d{4}):(?P<day>\d{3}):(?P<hour>\d{2}):(?P<minute>\d{2}):(?P<second>\d{2})  "
    r"temp_set (?P<temp_set>[\d\- ]+.\d{2}) deg_C "
    r"tmp (?P<receiver_temp>[\d\- ]+.\d{2}) deg_C "
    r"pwr (?P<power_percent>[\d\- ]+.\d{2}) percent"
)


def _parse_lines(text, pattern):
    for match in pattern.finditer(text):
        dct = {}
        for k, v in match.groupdict().items():
            try:
                dct[k] = int(v)
            except ValueError:
                dct[k] = float(v)
        yield dct


def _get_chunk_pos_and_size(
    fname: str | Path,
    start_time: tuple[int, int, int, int],
    end_time: tuple[int, int, int, int] | None = None,
    n_hours: int | None = None,
):
    """Get the chunk and position size for a given time range in a file.

    Parameters
    ----------
    fname : path
        File to read.
    start_time : tuple
        Tuple of (year, day, hour, minute) at which to start reading data.
    end_time : tuple
        Tuple of (year, day, hour, minute) at which to end reading data. This is exclusive,
        so that if `start_time` is (2020, 1, 0, 0) and `end_time` is (2020, 2, 0, 0),
        you get a whole day. The default is to get the *rest of* the day.

    Returns
    -------
    int :
        Starting position in file.
    nlines :
        Number of lines required to read for this chunk.

    """
    if end_time is None:
        if n_hours is None:
            end_time = f"{start_time[0]:04}:{start_time[1] + 1:03}:00:00"
        else:
            first_day = datetime(
                start_time[0],
                1,
                1,
                hour=start_time[2],
                minute=start_time[3],
                tzinfo=UTC,
            )
            dt = first_day + timedelta(days=start_time[1])
            end = dt + timedelta(hours=n_hours)
            jd = (end - first_day).days
            end_time = f"{end.year:04}:{jd:03}:{end.hour:02}:{end.minute:02}"
    else:
        end_time = (
            f"{end_time[0]:04}:{end_time[1]:03}:{end_time[2]:02}:{end_time[3]:02}"
        )

    start_time = (
        f"{start_time[0]:04}:{start_time[1]:03}:{start_time[2]:02}:{start_time[3]:02}"
    )

    fname = Path(fname)
    line = "0000:000:00:00"
    with fname.open("r") as fl:
        # Get our starting position in the file.
        while line and line[:14] < start_time:
            line = fl.readline()

        # Got to the end of the file without finding our year/day
        if not line:
            raise ValueError(
                f"The file provided [{fname}]does not contain the year/day desired "
                f"[{start_time[0]}/{start_time[1]}]."
            )

        # First line is current position, minus one line (which is the line length
        # plus a newline character).
        start_pos = fl.tell() - len(line)

        # Get the number of lines in this day.
        n_lines = 1
        while line and line[:14] < end_time:
            line = fl.readline()
            n_lines += 1

        end_pos = fl.tell() - len(line)

    return start_pos, n_lines - 1, end_pos - start_pos


def read_weather_file(
    weather_file: str | Path,
    year: int,
    day: int,
    hour: int = 0,
    minute: int = 0,
    n_hours: int | None = None,
    end_time: tuple[int, int, int, int] | None = None,
):
    """Read (a chunk of) the weather file maintained by the on-site (MRO) monitoring.

    The primary location of this file is on the enterprise cluster at
    ``/data5/edges/data/2014_February_Boolardy/weather2.txt``, but the function
    requires you to pass in the filename manually, as you may have copied the file
    to your own system or elsewhere.

    Parameters
    ----------
    weather_file : path or str
        The path to the file on the system.
    year : int
        The year defining the start of the chunk of times to return.
    day : int
        The day defining the start of the chunk of times to return.
    hour : int
        The hour defining the start of the chunk of times to return.
    minute : int
        The minute defining the start of the chunk of times to return.
    n_hours : int
        Number of hours of data to return. Default is to return the rest of the day.
    end_time : tuple of int
        The (year, day, hour, minute) defining the end of the returned data (exclusive).
        Default is to return the rest of the starting day.

    Returns
    -------
    structured array :
        A numpy structured array with the field names:
        * ``seconds``: seconds since the start of the chosen day.
        * ``rack_temp``: temperature of the rack (K)
        * ``ambient_temp``: ambient temperature on site (K)
        * ``ambient_hum``: ambient humidity on site (%)
        * ``frontend_temp``: temperature of the frontend (K)
        * ``lna_temp``: temperature of the LNA (K).

    """
    weather_file = Path(weather_file)
    with weather_file.open("r") as fl:
        if _NEW_WEATHER_PATTERN.match(fl.readline()) is not None:
            pattern = _NEW_WEATHER_PATTERN
        else:
            pattern = _OLD_WEATHER_PATTERN

    start_line, n_lines, nchar = _get_chunk_pos_and_size(
        weather_file, (year, day, hour, minute), end_time=end_time, n_hours=n_hours
    )
    dtype = [
        ("year", int),
        ("day", int),
        ("hour", int),
        ("minute", int),
        ("second", int),
        ("rack_temp", float),
        ("ambient_temp", float),
        ("ambient_hum", float),
        ("frontend_temp", float),
        ("lna_temp", float),
    ]

    weather = np.zeros(n_lines, dtype)

    with weather_file.open("r") as fl:
        # Go back to the starting position of the day, and read in each line of the day.
        fl.seek(start_line)

        matches = _parse_lines(fl.read(nchar), pattern)

        i = -1
        for i, match in enumerate(matches):
            w = (
                match["year"],
                match["day"],
                match["hour"],
                match["minute"],
                match["second"],
                match["rack_temp"],
                match["ambient_temp"],
                match["ambient_hum"],
            )

            if pattern == _NEW_WEATHER_PATTERN:
                w = (*w, match["frontend_temp"], match["lna_temp"])
            else:
                w = (*w, np.nan, np.nan)

            weather[i] = w

        if i < len(weather) - 1:
            warnings.warn(
                f"Only {i + 1}/{n_lines} lines of {weather_file} were able to be parsed.",
                stacklevel=2,
            )
            weather = weather[: i + 1]

    return weather


def read_thermlog_file(
    filename: str | Path,
    year: int,
    day: int,
    hour: int = 0,
    minute: int = 0,
    n_hours: int | None = None,
    end_time: tuple[int, int, int, int] | None = None,
):
    """Read (a chunk of) the thermlog file maintained by the on-site (MRO) monitoring.

    The primary location of this file is on the enterprise cluster at
    ``/data5/edges/data/2014_February_Boolardy/thermlog_{band}.txt``, but the function
    requires you to pass in the filename manually, as you may have copied the file
    to your own system or elsewhere.

    Parameters
    ----------
    filename : path or str
        The path to the file on the system.
    year : int
        The year defining the chunk of times to return.
    day : int
        The day defining the chunk of times to return.
    hour : int
        The hour defining the start of the chunk of times to return.
    minute : int
        The minute defining the start of the chunk of times to return.
    n_hours : int
        Number of hours of data to return. Default is to return the rest of the day.
    end_time : tuple of int
        The (year, day, hour, minute) defining the end of the returned data (exclusive).
        Default is to return the rest of the starting day.

    Returns
    -------
    structured array :
        A numpy structured array with the field names:
        * ``seconds``: seconds since the start of the chosen day.
        * ``temp_set``: temperature that it was set to (?) (C)
        * ``receiver_temp``: temperature of the receiver (C)
        * ``power_percent``: power of something (%)

    """
    start_line, n_lines, nchar = _get_chunk_pos_and_size(
        filename, (year, day, hour, minute), end_time=end_time, n_hours=n_hours
    )

    therm = np.zeros(
        n_lines,
        dtype=[
            ("year", int),
            ("day", int),
            ("hour", int),
            ("minute", int),
            ("second", int),
            ("temp_set", float),
            ("receiver_temp", float),
            ("power_percent", float),
        ],
    )

    with Path(filename).open("r") as fl:
        fl.seek(start_line)

        matches = _parse_lines(fl.read(nchar), _THERMLOG_PATTERN)

        i = -1
        for i, match in enumerate(matches):
            therm[i] = (
                match["year"],
                match["day"],
                match["hour"],
                match["minute"],
                match["second"],
                match["temp_set"],
                match["receiver_temp"],
                match["power_percent"],
            )
        if i < len(therm) - 1:
            warnings.warn(
                f"Only {i + 1}/{n_lines} lines of {filename} were able to be parsed.",
                stacklevel=2,
            )
            therm = therm[: i + 1]

    return therm


def read_auxiliary_data(
    weather_file: str | Path,
    thermlog_file: str | Path,
    year: int,
    day: int,
    hour: int = 0,
    minute: int = 0,
    n_hours: int | None = None,
    end_time: tuple[int, int, int, int] | None = None,
):
    """Read both weather and thermlog files for a given time range.

    Parameters
    ----------
    weather_file : path or str
        The file containing the weather information.
    thermlog_file : path or str
        The file containing the thermlog information.
    year : int
        The year defining the chunk of times to return.
    day : int
        The day defining the chunk of times to return.
    hour : int
        The hour defining the start of the chunk of times to return.
    minute : int
        The minute defining the start of the chunk of times to return.
    n_hours : int
        Number of hours of data to return. Default is to return the rest of the day.
    end_time : tuple of int
        The (year, day, hour, minute) defining the end of the returned data (exclusive).
        Default is to return the rest of the starting day.

    Returns
    -------
    structured array :
        The weather data (see :func:`read_weather_file`).
    structured array :
        The thermlog data (see :func:`read_thermlog_file`)

    """
    weather = read_weather_file(
        weather_file,
        year,
        day,
        hour=hour,
        minute=minute,
        n_hours=n_hours,
        end_time=end_time,
    )
    thermlog = read_thermlog_file(
        thermlog_file,
        year,
        day,
        hour=hour,
        minute=minute,
        n_hours=n_hours,
        end_time=end_time,
    )

    return weather, thermlog
