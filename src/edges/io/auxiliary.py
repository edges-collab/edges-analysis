"""Module defining EDGES-specific reading functions for weather and auxiliary data."""

import re
import warnings
from datetime import datetime, time, timedelta
from pathlib import Path

import numpy as np
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
    r"ambient_hum  (?P<ambient_hum>[\d\- ]{3}.\d{2}) percent"
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


_PATTERNS = (_NEW_WEATHER_PATTERN, _OLD_WEATHER_PATTERN, _THERMLOG_PATTERN)


def _get_end_time(
    start_time: Time,
    n_hours: int | None = None,
) -> datetime:
    if n_hours is None:
        # End time is the first second of the next day.
        return datetime.combine(start_time.datetime + timedelta(days=1), time.min)
    return start_time.datetime + timedelta(hours=n_hours)


def _parse_aux_line(line: str) -> dict[str, float] | None:
    """Parse the values of a line of an aux file, or return None if it is malformed.

    The format is detected for each line separately (the first pattern that
    matches wins), since files can change format part-way through.
    """
    for pattern in _PATTERNS:
        if (match := pattern.search(line)) is not None:
            try:
                return {k: float(v) for k, v in match.groupdict().items()}
            except ValueError:
                return None
    return None


def _read_aux_file(
    aux_file: str | Path,
    start: Time,
    n_hours: int | None = None,
    end: Time | None = None,
) -> QTable:
    """Read (a chunk of) an auxiliary (weather or thermlog) file.

    Each line of the file is a ``YYYY:DDD:HH:MM:SS`` time stamp followed by values
    in one of the weather or thermlog formats. The format is detected per line, so
    a file can mix formats (e.g. old weather lines without the frontend and LNA
    temperatures, followed by new ones); values that a line does not have are NaN.
    Blank or malformed lines are skipped, with a single warning giving their count.
    The lines are assumed to be in time order.

    Parameters
    ----------
    aux_file
        The path to the file.
    start
        The start of the chunk of times to return (inclusive).
    n_hours
        Number of hours of data to return, if ``end`` is not given. Default is to
        return the rest of the day.
    end
        The end of the chunk of times to return. Default is ``start`` plus
        ``n_hours``.

    Returns
    -------
    QTable
        A table with a ``time`` column and one column per value in the file (see
        :func:`read_weather_file` and :func:`read_thermlog_file`).

    Raises
    ------
    ValueError
        If no line in the time range is in a known format, but some are malformed.
    """
    aux_file = Path(aux_file)

    if end is None:
        end = Time(_get_end_time(start_time=start, n_hours=n_hours))

    auxdata = []
    n_bad = 0
    with aux_file.open("r") as fl:
        for line in fl:
            if not line.strip():
                n_bad += 1
                continue

            try:
                t = Time.strptime(line[:17], "%Y:%j:%H:%M:%S")
            except ValueError:
                n_bad += 1
                continue

            if t < start:
                continue
            if t > end:
                break

            values = _parse_aux_line(line[17:])
            if values is None:
                n_bad += 1
                continue
            auxdata.append(values | {"time": t})

    if n_bad:
        if not auxdata:
            raise ValueError(
                f"No lines of the aux file {aux_file} in the requested time range "
                f"matched a known format ({n_bad} blank or malformed lines)."
            )
        warnings.warn(
            f"Skipped {n_bad} blank or malformed lines in the aux file {aux_file}.",
            stacklevel=3,
        )

    if not auxdata:
        return QTable()

    # Keep the column order of the patterns, with NaN for values a line lacks.
    names = [
        name
        for pattern in _PATTERNS
        for name in pattern.groupindex
        if any(name in row for row in auxdata)
    ]
    names = list(dict.fromkeys(names))  # e.g. rack_temp is in several patterns
    out = QTable()
    for name in names:
        out[name] = [row.get(name, np.nan) for row in auxdata] * _UNITS[name]
    out["time"] = Time([row["time"] for row in auxdata])
    return out


def read_weather_file(
    weather_file: str | Path,
    start: Time,
    n_hours: int | None = None,
    end: Time | None = None,
) -> QTable:
    """Read (a chunk of) the weather file maintained by the on-site (MRO) monitoring.

    The primary location of this file is on the enterprise cluster at
    ``/data5/edges/data/2014_February_Boolardy/weather2.txt``, but the function
    requires you to pass in the filename manually, as you may have copied the file
    to your own system or elsewhere.

    The format is detected per line, so files mixing the old format (without the
    frontend and LNA temperatures) and the new one can be read; values a line does
    not have are NaN. Blank or malformed lines are skipped with a warning.

    Parameters
    ----------
    weather_file
        The path to the file on the system.
    start
        The start of the chunk of times to return (inclusive).
    n_hours
        Number of hours of data to return, if ``end`` is not given. Default is to
        return the rest of the day of ``start``.
    end
        The end of the chunk of times to return. Default is ``start`` plus
        ``n_hours``.

    Returns
    -------
    QTable
        A table with the columns:

        * ``rack_temp``: temperature of the rack (K)
        * ``ambient_temp``: ambient temperature on site (K)
        * ``ambient_hum``: ambient humidity on site (%)
        * ``frontend_temp``: temperature of the frontend (K), if in the file
        * ``lna_temp``: temperature of the LNA (K), if in the file
        * ``time``: the time of each reading.
    """
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
) -> QTable:
    """Read (a chunk of) the thermlog file maintained by the on-site (MRO) monitoring.

    The primary location of this file is on the enterprise cluster at
    ``/data5/edges/data/2014_February_Boolardy/thermlog_{band}.txt``, but the
    function requires you to pass in the filename manually, as you may have copied
    the file to your own system or elsewhere. Blank or malformed lines are skipped
    with a warning.

    Parameters
    ----------
    filename
        The path to the file on the system.
    start
        The start of the chunk of times to return (inclusive).
    n_hours
        Number of hours of data to return, if ``end`` is not given. Default is to
        return the rest of the day of ``start``.
    end
        The end of the chunk of times to return. Default is ``start`` plus
        ``n_hours``.

    Returns
    -------
    QTable
        A table with the columns:

        * ``temp_set``: the receiver temperature set-point (degC)
        * ``receiver_temp``: temperature of the receiver (degC)
        * ``power_percent``: power of the thermal controller (%)
        * ``time``: the time of each reading.
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
) -> tuple[QTable, QTable]:
    """Read both weather and thermlog files for a given time range.

    Parameters
    ----------
    weather_file
        The file containing the weather information.
    thermlog_file
        The file containing the thermlog information.
    start
        The start of the chunk of times to return (inclusive).
    n_hours
        Number of hours of data to return, if ``end`` is not given. Default is to
        return the rest of the day of ``start``.
    end
        The end of the chunk of times to return. Default is ``start`` plus
        ``n_hours``.

    Returns
    -------
    weather : QTable
        The weather data (see :func:`read_weather_file`).
    thermlog : QTable
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
