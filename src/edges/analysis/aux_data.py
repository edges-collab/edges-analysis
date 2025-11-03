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
        the configured `raw_field_data` directory. If provided, will search in
        the current directory and the `raw_field_data` directory for the given
        file (if not an absolute path).
    """
    times = data.times[..., data.loads.index("ant")]
    start = min(times)
    end = max(times)

    pth = config.raw_field_data
    if (pth is None and weather_file is None) or not Path(weather_file).exists():
        raise ValueError(
            "weather file not given, but not configuration set to specify where "
            "raw data is"
        )

    if weather_file is not None:
        if not weather_file.exists() and not weather_file.is_absolute():
            weather_file = pth / weather_file
    elif (start.year, start.day) <= (2017, 329):
        weather_file = pth / "weather_upto_20171125.txt"
    else:
        weather_file = pth / "weather2.txt"

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
        Object into which to add the weather data.
    band
        The instrument taking the data. Only provide to automatically find the
        correct data.
    thermlog_file
        Path to a weather file from which to read the weather data. Must be
        formatted appropriately. By default, will choose an appropriate file from
        the configured `raw_field_data` directory. If provided, will search in
        the current directory and the `raw_field_data` directory for the given
        file (if not an absolute path).
    """
    times = data.times[..., data.loads.index("ant")]
    start = min(times)
    end = max(times)

    pth = config.raw_field_data
    if (pth is None and thermlog_file is None) or not Path(thermlog_file).exists():
        raise ValueError(
            "thermlog file not given, but not configuration set to specify where "
            "raw data is"
        )

    if thermlog_file is None:
        thermlog_file = pth / f"thermlog_{band}.txt"
    elif not thermlog_file.exists() and not thermlog_file.is_absolute():
        thermlog_file = pth / thermlog_file

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
