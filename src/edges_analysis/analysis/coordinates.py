import datetime as dt
import numpy as np
from astropy import time as apt, coordinates as apc
from .. import const


def utc2lst(utc_time_array, longitude):
    """
    Convert an array representing UTC date/time to a 1D array of LST date/time.

    Parameters
    ----------
    utc_time_array : array-like
        Nx6 array of floats or integers, where each row is of the form
        [yyyy, mm, dd,HH, MM, SS]. It can also be a 6-element 1D array.
    longitude : float
        Terrestrial longitude of observatory (float) in degrees.

    Returns
    -------
    LST : 1D array of LST dates/times

    Examples
    --------
    >>> LST = utc2lst(utc_time_array, -27.5)
    """
    # convert input array to "int"
    if not isinstance(utc_time_array[0], dt.datetime):
        utc_time_array = [dt.datetime(*utc) for utc in np.atleast_2d(utc_time_array).astype(int)]

    # python "datetime" to astropy "Time" format
    t = apt.Time(utc_time_array, format="datetime", scale="utc")

    # necessary approximation to compute sidereal time
    t.delta_ut1_utc = 0

    return t.sidereal_time("apparent", str(longitude) + "d", model="IAU2006A").value


def sun_moon_azel(lat, lon, utc_array):
    """Get local coordinates of the Sun using Astropy."""

    obs_location = apc.EarthLocation(lat=lat, lon=lon)

    # Compute local coordinates of Sun and Moon
    utc_array = utc_array
    sun = np.zeros((len(utc_array), 2))
    moon = np.zeros((len(utc_array), 2))

    utc_array = [utc if isinstance(utc, dt.datetime) else dt.datetime(*utc) for utc in utc_array]
    times = apt.Time(utc_array, format="datetime")

    Sun = apc.get_sun(times).transform_to(apc.AltAz(location=obs_location))
    Moon = apc.get_moon(times).transform_to(apc.AltAz(location=obs_location))

    sun[:, 0] = Sun.az.deg
    sun[:, 1] = Sun.alt.deg
    moon[:, 0] = Moon.az.deg
    moon[:, 1] = Moon.alt.deg

    return sun, moon


def f2z(fe):
    # Constants and definitions
    c = 299792458  # wikipedia, m/s
    f21 = 1420.40575177e6  # wikipedia,
    lambda21 = c / f21  # frequency to wavelength, as emitted
    # frequency to wavelength, observed. fe comes in MHz but it
    # has to be converted to Hertz
    lmbda = c / (fe * 1e6)
    return (lmbda - lambda21) / lambda21


def z2f(z):
    # Constants and definitions
    c = 299792458  # wikipedia, m/s
    f21 = 1420.40575177e6  # wikipedia,
    l21 = c / f21  # frequency to wavelength, as emitted
    lmbda = l21 * (1 + z)
    return c / (lmbda * 1e6)


def lst2gha(lst):
    gha = lst - const.galactic_centre_lst
    gha[gha < 0] += 24
    return gha


def get_jd(d):
    """Get the day of the year from a datetime object"""
    dt0 = dt.datetime(d.year, 1, 1)
    return (d - dt0).days + 1


def dt_from_jd(y, d, *args):
    begin = dt.datetime(y, 1, 1, *args)
    return begin + dt.timedelta(days=d)
