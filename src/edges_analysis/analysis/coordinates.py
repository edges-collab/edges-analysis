import datetime as dt
import numpy as np
from astropy import time as apt, coordinates as apc


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
    uta = np.atleast_2d(utc_time_array.astype(int))

    # converting UTC to LST
    lst = np.zeros(len(uta))

    for i, ut in enumerate(uta):
        # time stamp in python "datetime" format
        if not isinstance(ut, dt.datetime):
            ut = dt.datetime(*ut)

        # python "datetime" to astropy "Time" format
        t = apt.Time(ut, format="datetime", scale="utc")

        # necessary approximation to compute sidereal time
        t.delta_ut1_utc = 0

        # LST at longitude LONG_float, in degrees
        lst[i] = t.sidereal_time(
            "apparent", str(longitude) + "d", model="IAU2006A"
        ).value

    return lst


def sun_moon_azel(lat, lon, utc_array):
    """Get local coordinates of the Sun using Astropy."""

    obs_location = apc.EarthLocation(lat=lat, lon=lon)

    # Compute local coordinates of Sun and Moon
    utc_array = np.atleast_2d(utc_array)
    sun = np.zeros((len(utc_array), 2))
    moon = np.zeros((len(utc_array), 2))

    for i, utc in enumerate(utc_array):
        time = apt.Time(dt.datetime(*utc))

        Sun = apc.get_sun(time)
        Moon = apc.get_moon(time)

        Sun = sun.transform_to(apc.AltAz(location=obs_location))
        Moon = moon.transform_to(apc.AltAz(location=obs_location))

        sun[i, 0] = Sun.alt.deg
        sun[i, 1] = Sun.az.deg
        moon[i, 0] = Moon.alt.deg
        moon[i, 1] = Moon.az.deg

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
