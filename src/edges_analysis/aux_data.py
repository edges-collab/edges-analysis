"""Module for dealing with auxiliary data for EDGES observations."""
from .gsdata import GSData, register_gsprocess
import edges_cal.types as tp
from pathlib import Path
from .config import config
from edges_io.auxiliary import read_thermlog_file, read_weather_file
from .coordinates import get_jd, dt_from_jd
import logging
import time
import numpy as np

logger = logging.getLogger(__name__)

class WeatherError(ValueError):
    pass


def _interpolate_times(thing, times):
    seconds = np.array([(t - times[0]).total_seconds() for t in times])
    interpolated = {}

    thing_seconds = [
        (
            dt_from_jd(
                x["year"], int(x["day"]), x["hour"], x["minute"], x["second"]
            )
            - times[0]
        ).total_seconds()
        for x in thing
    ]

    for name, (kind, _) in thing.dtype.fields.items():
        if kind.kind == "i":
            continue

        interpolated[name] = np.interp(seconds, thing_seconds, thing[name])

        # Convert to celsius
        if name.endswith("_temp") and np.any(interpolated[name] > 273.15):
            interpolated[name] -= 273.15

    return interpolated

@register_gsprocess
def add_weather_data(data: GSData, weather_file: tp.PathLike) -> GSData:
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
    times = data.time_array

    start = min(times)
    end = max(times)

    pth = Path(config["paths"]["raw_field_data"])
    if weather_file is not None:
        weather_file = Path(weather_file)
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
        year=start.year,
        day=get_jd(start),
        hour=start.hour,
        end_time=(end.year, get_jd(end), end.hour, end.minute + 1),
    )

    if len(weather) == 0:
        raise WeatherError(
            f"Weather file '{weather_file}' has no dates between "
            f"{start.strftime('%Y/%m/%d')} "
            f"and {end.strftime('%Y/%m/%d')}."
        )


    logger.info("Setting up arrays...")

    t = time.time()
    # Get the seconds since obs start for the data (not the auxiliary).

    logger.info(f".... took {time.time() - t} sec.")

    t = time.time()
    # Interpolate weather

    interpolated = _interpolate_times(weather, times)

    logger.info(f"Took {time.time() - t} sec to interpolate auxiliary data.")

    return data.update(auxiliary_measurements={**data.auxiliary_measurements, **interpolated})


@register_gsprocess
def add_thermlog_data(data: GSData, band: str, thermlog_file: tp.PathLike) -> GSData:
    """Add thermlog data to a :class`GSData` object.

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
    times = data.time_array

    start = min(times)
    end = max(times)

    pth = Path(config["paths"]["raw_field_data"])
    if thermlog_file is not None:
        thermlog_file = Path(thermlog_file)
        if not thermlog_file.exists() and not thermlog_file.is_absolute():
            thermlog_file = pth / thermlog_file
    else:
        thermlog_file = pth / f"thermlog_{band}.txt"

    # Get all aux data covering our times, up to the next minute (so we have some
    # overlap).
    weather, thermlog = read_thermlog_file(
        thermlog_file,
        year=start.year,
        day=get_jd(start),
        hour=start.hour,
        end_time=(end.year, get_jd(end), end.hour, end.minute + 1),
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

    logger.info(f"Took {time.time() - t} sec to interpolate auxiliary data.")

    return data.update(auxiliary_measurements={**data.auxiliary_measurements, **interpolated})
