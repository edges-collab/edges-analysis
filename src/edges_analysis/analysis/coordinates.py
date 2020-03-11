import datetime as dt

import ephem as eph
import numpy as np
from astropy import time as apt


def utc2lst(utc_time_array, longitude):
    """
    Last modification: May 29, 2015.

    This function converts an Nx6 array of floats or integers representing UTC date/time,
    to a 1D array of LST date/time.

    Definition:
    LST = utc2lst(utc_time_array, LONG_float)

    Input parameters:
    utc_time_array: Nx6 array of floats or integers, where each row is of the form [yyyy, mm, dd,
    HH, MM, SS]. It can also be a 6-element 1D array.
    LONG_float: terrestrial longitude of observatory (float) in degrees.

    Output parameters:
    LST: 1D array of LST dates/times

    Usage:
    LST = utc2lst(utc_time_array, -27.5)
    """

    # converting input array to "int"
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
    """
    Local coordinates of the Sun using Astropy
    """

    obs_location = eph.Observer()
    obs_location.lat = str(lat)
    obs_location.lon = str(lon)

    # Compute local coordinates of Sun and Moon
    utc_array = np.atleast_2d(utc_array)
    sun = np.zeros((len(utc_array), 2))
    moon = np.zeros((len(utc_array), 2))

    for i, utc in enumerate(utc_array):
        if not isinstance(utc, dt.datetime):
            obs_location.date = dt.datetime(*utc)
        else:
            obs_location.date = utc

        sun = eph.Sun(obs_location)
        moon = eph.Moon(obs_location)

        sun[i, 0] = (180 / np.pi) * eph.degrees(sun.az)
        sun[i, 1] = (180 / np.pi) * eph.degrees(sun.alt)
        moon[i, 0] = (180 / np.pi) * eph.degrees(moon.az)
        moon[i, 1] = (180 / np.pi) * eph.degrees(moon.alt)

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
