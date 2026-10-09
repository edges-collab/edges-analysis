"""Module for dealing with auxiliary data for EDGES observations."""

import logging
import time
from pathlib import Path

import numpy as np
from astropy.table import QTable
from astropy.time import Time
from pygsdata import GSData, gsregister

from .. import types as tp
from ..config import config
from ..io.auxiliary import read_thermlog_file, read_weather_file

logger = logging.getLogger(__name__)


class WeatherError(ValueError):
    """Error for weather data issues."""


def _interpolate_times(thing: QTable, times: Time) -> QTable:
    t0 = times[0]

    seconds = (times - t0).to_value("s")
    interpolated = {}

    thing_seconds = (thing["time"] - t0).to_value("s")

    for col in thing.colnames:
        if col == "time":
            continue

        interpolated[col] = np.interp(seconds, thing_seconds, thing[col])

    return QTable(interpolated)


def _resolve_aux_file(aux_file: tp.PathLike | None, default: str) -> Path:
    """Find an auxiliary data file, possibly within the ``raw_field_data`` directory.

    Parameters
    ----------
    aux_file
        The given path to the file. If None, use ``default``. If relative and not
        found from the current directory, it is looked for in the configured
        ``raw_field_data`` directory.
    default
        The name of the file to use (in the configured ``raw_field_data`` directory)
        if ``aux_file`` is None.

    Returns
    -------
    Path
        The path to the existing auxiliary file.
    """
    root = config.raw_field_data

    if aux_file is None:
        if root is None:
            raise ValueError(
                "No auxiliary file given, and the raw_field_data configuration option "
                "(which says where the raw data is) is not set."
            )
        aux_file = root / default
    else:
        aux_file = Path(aux_file)
        if not aux_file.exists() and not aux_file.is_absolute() and root is not None:
            aux_file = root / aux_file

    if not aux_file.exists():
        raise FileNotFoundError(f"Auxiliary file '{aux_file}' does not exist.")

    return aux_file


@gsregister("supplement")
def add_weather_data(data: GSData, weather_file: tp.PathLike | None = None) -> GSData:
    """Add weather data to a :class`GSData` object.

    This adds data specifically from a file maintained by EDGES.

    Parameters
    ----------
    data
        Object into which to add the weather data.
    weather_file
        Path to a weather file from which to read the weather data. Must be
        formatted appropriately. By default, will choose an appropriate file from
        the configured `raw_field_data` directory (``weather_upto_20171125.txt``
        for data starting on or before 2017-11-25, otherwise ``weather2.txt``).
        If provided, will search in the current directory and the
        `raw_field_data` directory for the given file (if not an absolute path).
    """
    times = data.times[..., data.loads.index("ant")]
    start = min(times)
    end = max(times)

    start_dt = start.datetime
    if (start_dt.year, start_dt.timetuple().tm_yday) <= (2017, 329):
        default = "weather_upto_20171125.txt"
    else:
        default = "weather2.txt"
    weather_file = _resolve_aux_file(weather_file, default)

    # Get all aux data covering our times, up to the next minute (so we have some
    # overlap).
    weather = read_weather_file(
        weather_file,
        start=start,
        end=end,
    )

    if len(weather) == 0:
        raise WeatherError(
            f"Weather file '{weather_file}' has no dates between "
            f"{start.strftime('%Y/%m/%d')} "
            f"and {end.strftime('%Y/%m/%d')}."
        )

    logger.info("Setting up arrays...")

    t = time.time()
    # Interpolate weather
    interpolated = _interpolate_times(weather, times)

    logger.info(f"Took {time.time() - t} sec to interpolate weather data.")
    return data.update(
        auxiliary_measurements=data.auxiliary_measurements | interpolated
    )


@gsregister("supplement")
def add_thermlog_data(
    data: GSData, band: str | None = None, thermlog_file: tp.PathLike | None = None
) -> GSData:
    """Add thermlog data to a :class`GSData` object.

    This adds data specifically from a file maintained by EDGES.

    Parameters
    ----------
    data
        Object into which to add the thermlog data.
    band
        The instrument taking the data. Only required if ``thermlog_file`` is not
        given, in which case the file ``thermlog_{band}.txt`` in the configured
        `raw_field_data` directory is used.
    thermlog_file
        Path to a thermlog file from which to read the thermlog data. Must be
        formatted appropriately. If provided, will search in the current directory
        and the `raw_field_data` directory for the given file (if not an absolute
        path).
    """
    times = data.times[..., data.loads.index("ant")]
    start = min(times)
    end = max(times)

    if thermlog_file is None and band is None:
        raise ValueError("Either thermlog_file or band must be given.")
    thermlog_file = _resolve_aux_file(thermlog_file, f"thermlog_{band}.txt")

    # Get all aux data covering our times, up to the next minute (so we have some
    # overlap).
    thermlog = read_thermlog_file(
        thermlog_file,
        start=start,
        end=end,
    )

    if len(thermlog) == 0:
        raise WeatherError(
            f"Thermlog file '{thermlog_file}' has no dates between "
            f"{start.strftime('%Y/%m/%d')} "
            f"and {end.strftime('%Y/%m/%d')}."
        )

    logger.info("Setting up arrays...")

    t = time.time()
    interpolated = _interpolate_times(thermlog, times)

    logger.info(f"Took {time.time() - t} sec to interpolate thermlog data.")

    return data.update(
        auxiliary_measurements=data.auxiliary_measurements | interpolated
    )
