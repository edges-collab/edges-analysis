import datetime as dt

import ephem as eph
import numpy as np
from astropy import time as apt


def utc2lst(utc_time_array, LONG_float):
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
    uta = utc_time_array.astype(int)

    if uta.ndim == 1:
        len_array = 1
    elif uta.ndim == 2:
        len_array = len(uta)
    # converting UTC to LST
    LST = np.zeros(len_array)

    for i in range(len_array):
        if uta.ndim == 1:
            yyyy = uta[0]
            mm = uta[1]
            dd = uta[2]
            HH = uta[3]
            MM = uta[4]
            SS = uta[5]

        elif uta.ndim == 2:
            yyyy = uta[i, 0]
            mm = uta[i, 1]
            dd = uta[i, 2]
            HH = uta[i, 3]
            MM = uta[i, 4]
            SS = uta[i, 5]

        # time stamp in python "datetime" format
        udt = dt.datetime(yyyy, mm, dd, HH, MM, SS)

        # python "datetime" to astropy "Time" format
        t = apt.Time(udt, format="datetime", scale="utc")

        # necessary approximation to compute sidereal time
        t.delta_ut1_utc = 0

        # LST at longitude LONG_float, in degrees
        LST_object = t.sidereal_time(
            "apparent", str(LONG_float) + "d", model="IAU2006A"
        )
        LST[i] = LST_object.value

    return LST


def SUN_MOON_azel(LAT, LON, UTC_array):
    #
    # Local coordinates of the Sun using Astropy
    #
    # EDGES_lat_deg = -26.7
    # EDGES_lon_deg = 116.6
    #

    # Observation coordinates
    OBS_lat_deg = str(LAT)
    OBS_lon_deg = str(LON)
    # print(' ')
    # print('Observation Coordinates: ' + 'LAT: ' + OBS_lat_deg + ' LON: ' + OBS_lon_deg)
    # print('------------------------')

    OBS_location = eph.Observer()
    OBS_location.lat = OBS_lat_deg
    OBS_location.lon = OBS_lon_deg

    # Compute local coordinates of Sun and Moon
    SH = UTC_array.shape

    if len(SH) == 1:
        coord = np.zeros(4)

        OBS_location.date = dt.datetime(
            UTC_array[0],
            UTC_array[1],
            UTC_array[2],
            UTC_array[3],
            UTC_array[4],
            UTC_array[5],
        )
        Sun = eph.Sun(OBS_location)
        Moon = eph.Moon(OBS_location)

        coord[0] = (180 / np.pi) * eph.degrees(Sun.az)
        coord[1] = (180 / np.pi) * eph.degrees(Sun.alt)
        coord[2] = (180 / np.pi) * eph.degrees(Moon.az)
        coord[3] = (180 / np.pi) * eph.degrees(Moon.alt)
    elif len(SH) == 2:
        coord = np.zeros((SH[0], 4))
        for i in range(SH[0]):
            OBS_location.date = dt.datetime(
                UTC_array[i, 0],
                UTC_array[i, 1],
                UTC_array[i, 2],
                UTC_array[i, 3],
                UTC_array[i, 4],
                UTC_array[i, 5],
            )
            Sun = eph.Sun(OBS_location)
            Moon = eph.Moon(OBS_location)

            coord[i, 0] = (180 / np.pi) * eph.degrees(Sun.az)
            coord[i, 1] = (180 / np.pi) * eph.degrees(Sun.alt)
            coord[i, 2] = (180 / np.pi) * eph.degrees(Moon.az)
            coord[i, 3] = (180 / np.pi) * eph.degrees(Moon.alt)
    else:
        raise ValueError("SH must be length 1 or 2.")

    return coord


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
