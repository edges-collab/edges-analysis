"""Module defining EDGES-specific reading functions for weather and auxiliary data."""

import re
from datetime import datetime, time, timedelta
from pathlib import Path

from astropy import units as un
from astropy.table import QTable
from astropy.time import Time

_time_format = (
    r"(?P<year>\d{4}):(?P<day>\d{3}):(?P<hour>\d{2}):"
    r"(?P<minute>\d{2}):(?P<second>\d{2})"
)

_NEW_WEATHER_PATTERN = re.compile(
    r"rack_temp  (?P<rack_temp>\d{3}.\d{2}) Kelvin, "
    r"ambient_temp  (?P<ambient_temp>\d{3}.\d{2}) Kelvin, "
    r"ambient_hum  (?P<ambient_hum>[\d\- ]{3}.\d{2}) percent, "
    r"frontend  (?P<frontend_temp>\d{3}.\d{2}) Kelvin, "
    r"rcv3_lna  (?P<lna_temp>\d{3}.\d{2}) Kelvin"
)

_OLD_WEATHER_PATTERN = re.compile(
    r"rack_temp  (?P<rack_temp>\d{3}.\d{2}) Kelvin, "
    r"ambient_temp  (?P<ambient_temp>\d{3}.\d{2}) Kelvin, "
    r"ambient_hum  (?P<ambient_hum>[\d\- ]{3}.\d{2}) percent, "
)

_THERMLOG_PATTERN = re.compile(
    r"temp_set (?P<temp_set>[\d\- ]+.\d{2}) deg_C "
    r"tmp (?P<receiver_temp>[\d\- ]+.\d{2}) deg_C "
    r"pwr (?P<power_percent>[\d\- ]+.\d{2}) percent"
)

_UNITS = {
    "rack_temp": un.K,
    "ambient_temp": un.K,
    "ambient_hum": un.percent,
    "frontend_temp": un.K,
    "lna_temp": un.K,
    "temp_set": un.deg_C,
    "receiver_temp": un.deg_C,
    "power_percent": un.percent,
}


def _parse_lines(text, pattern):
    for match in pattern.finditer(text):
        dct = {}
        for k, v in match.groupdict().items():
            try:
                dct[k] = int(v)
            except ValueError:
                dct[k] = float(v)
        yield dct


def _get_end_time(
    start_time: Time,
    n_hours: int | None = None,
) -> datetime:
    if n_hours is None:
        # End time is the first second of the next day.
        return datetime.combine(start_time.datetime + timedelta(days=1), time.min)
    return start_time.datetime + timedelta(hours=n_hours)


def _read_aux_file(
    aux_file: str | Path,
    start: Time,
    n_hours: int | None = None,
    end: Time | None = None,
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
    aux_file = Path(aux_file)
    with aux_file.open("r") as fl:
        for pattern in (_NEW_WEATHER_PATTERN, _OLD_WEATHER_PATTERN, _THERMLOG_PATTERN):
            if pattern.search(fl.readline()) is not None:
                break
        else:
            raise ValueError(
                f"No patterns matched the first line of the aux file {aux_file}"
            )

    if end is None:
        end = Time(_get_end_time(start_time=start, n_hours=n_hours))

    auxdata = []
    with aux_file.open("r") as fl:
        # Go back to the starting position of the day, and read in each line of the day.
        for line in fl:
            t = Time.strptime(line[:17], "%Y:%j:%H:%M:%S")
            if t < start:
                continue
            if t > end:
                break

            matches = re.search(pattern, line)
            dct = matches.groupdict()
            auxdata.append(dct | {"time": t})

    auxdata = QTable(rows=auxdata)
    for col in auxdata.columns:
        if col == "time":
            continue
        auxdata[col] = auxdata[col].astype(float) * _UNITS[col]

    return auxdata


def read_weather_file(
    weather_file: str | Path,
    start: Time,
    n_hours: int | None = None,
    end: Time | None = None,
):
    """Read a standard weather file."""
    return _read_aux_file(
        aux_file=weather_file,
        start=start,
        n_hours=n_hours,
        end=end,
    )


def read_thermlog_file(
    filename: str | Path,
    start: Time,
    n_hours: int | None = None,
    end: Time | None = None,
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
    return _read_aux_file(
        aux_file=filename,
        start=start,
        n_hours=n_hours,
        end=end,
    )


def read_auxiliary_data(
    weather_file: str | Path,
    thermlog_file: str | Path,
    start: Time,
    n_hours: int | None = None,
    end: Time | None = None,
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
        start,
        n_hours=n_hours,
        end=end,
    )
    thermlog = read_thermlog_file(
        thermlog_file,
        start,
        n_hours=n_hours,
        end=end,
    )

    return weather, thermlog
