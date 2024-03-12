"""A module providing time/coordinate calculations that match Alan's C-code.

These are not meant to be used in general, but only when trying to specifically match
the C-code for a particular reason.

They have essentially just been transcripted from the C-code, and keep the same names
as functions there.
"""

from __future__ import annotations

from datetime import datetime
from typing import Union

import numpy as np

Number = Union[float, np.ndarray]


def isleapyear(year: int) -> bool:
    """Check if a year is a leap year."""
    return (year % 4 == 0 and year % 100 != 0) or year % 400 == 0


def gst(time: float) -> float:
    """Convert time to GST."""
    secs = tosecs(2011, 1, 17, 15, 58.0778)
    return (((time - secs) / 86164.09053) % 1) * 2 * np.pi


def datetime_tosecs(dt: datetime):
    """Convert a datetime object to seconds since New Year 1970."""
    return tosecs(
        dt.year,
        dt.timetuple().tm_yday,
        dt.hour,
        dt.minute,
        round(dt.second + dt.microsecond / 1e6),
    )


def tosecs(yr: int, day: int, hour: int, minutes: int, sec: int | float) -> int | float:
    """Convert a date-time to seconds since New Year 1970."""
    secs = (yr - 1970) * 31536000 + (day - 1) * 86400 + hour * 3600 + minutes * 60 + sec
    for i in range(1970, yr):
        if isleapyear(i):
            secs += 86400
    return max(0, secs)


def galactic_to_radec(glat: Number, glon: Number) -> tuple[Number, Number]:
    """Galactic to radec  2000 epoch pole at 12h51.4 27.1.

    Parameters
    ----------
    glat
        Galactic latitude in degrees.
    glon
        Galactic longitude in degrees.
    """
    deg2rad = np.pi / 180
    glat = glat * deg2rad
    glon = glon * deg2rad  # no in-place so that input is not modified

    d0 = -(28.0 + 56.0 / 60.0) * deg2rad
    r0 = (17.0 + 45.5 / 60.0) * np.pi / 12.0
    dp = 27.1 * np.pi / 180.0
    rp = (12.0 + 51.4 / 60.0) * np.pi / 12.0
    zr = np.sin(d0)
    xr = np.cos(r0 - rp) * np.cos(d0)
    yr = np.sin(r0 - rp) * np.cos(d0)
    xg = xr * np.sin(dp) - zr * np.cos(dp)
    yg = yr
    a = np.arctan2(yg, xg)
    xg = np.cos((glon) + a) * np.cos(glat)
    yg = np.sin((glon) + a) * np.cos(glat)
    zg = np.sin(glat)
    xr = xg * np.sin(dp) + zg * np.cos(dp)
    yr = yg
    zr = zg * np.sin(dp) - xg * np.cos(dp)

    dec = np.arctan2(zr, np.sqrt(xr * xr + yr * yr))
    ra = np.arctan2(yr, xr) + rp
    return ra, dec


def toyrday(secs: float) -> tuple[int, int, int, int, int]:
    """Convert Seconds to Yr/Day/Hr/Min/Sec."""
    day = np.floor(secs / 86400.0)
    sec = secs - day * 86400.0
    year = 1970
    while day > 365:
        days = 366 if isleapyear(year) else 365
        day -= days
        year += 1

    hour = sec / 3600.0
    sec -= hour * 3600.0
    minute = sec / 60.0
    sec -= minute * 60
    day += 1

    if day == 366 and not isleapyear(year):
        # fix for alan
        day -= 365
        year += 1

    return year, int(day), int(hour), int(minute), int(sec)


def radec_azel2(
    ha: Number, sindec: Number, cosdec: Number, sinlat: float, coslat: float
) -> tuple[Number, Number]:
    """Convert from RA/DEC to Az/El.

    Parameters
    ----------
    ha
        Hour angle of the observed sources (including time and location information of
        the antenna, as well) in radians.
    sindec
        Sine of the declination of the observation in radians.
    cosdec
        Cosine of the declination of the observation in radians.
    sinlat
        Sine of the latitude of the antenna in radians.
    coslat
        Cosine of the latitude of the antenna in radians.
    """
    w = np.sin(ha) * cosdec
    r = np.cos(ha) * cosdec
    zen = r * coslat + sindec * sinlat
    north = -r * sinlat + sindec * coslat
    el = np.arctan2(zen, np.sqrt(north * north + w * w))
    az = np.arctan2(-w, north) % (2 * np.pi)
    return az, el


def radec_azel(
    ha: Number,
    dec: Number,
    lat: float,
) -> tuple[Number, Number]:
    """Convert from sky to antenna coords az/el.

    Parameters
    ----------
    ha
        Hour angle of the observed sources (including time and location information of
        the antenna, as well) in radians.
    dec
        Declination of the observation in radians.
    lat
        Latitude of the antenna in radians.
    """
    return radec_azel2(ha, np.sin(dec), np.cos(dec), np.sin(lat), np.cos(lat))


def radec_azel_from_lst(
    lst: float, ra: Number, dec: Number, lat: float
) -> tuple[Number, Number]:
    """Convert from sky to antenna coords az/el.

    Parameters
    ----------
    lst
        Local sidereal time in radians.
    ra
        Right ascension of the sources in radians.
    dec
        Declination of the sources in radians.
    lat
        Latitude of the antenna in radians.
    """
    ha = lst - ra
    return radec_azel(ha, dec, lat)
