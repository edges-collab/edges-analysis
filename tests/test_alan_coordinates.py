"""Perform tests of the _coordinates_alan module."""

import numpy as np
from astropy import coordinates as apc
from astropy import units as u
from astropy.time import Time
from edges_analysis import _coordinates_alan as crda
from edges_analysis import const, sky_models


def test_alan_coordinates_azel():
    sky_model = sky_models.Haslam408AllNoh()
    t = Time("2016-09-16T16:26:57", format="isot", scale="utc")
    loc = const.KNOWN_LOCATIONS["alan-edges"]

    antenna_frame = apc.AltAz(location=loc, obstime=t)

    c = sky_model.coords.reshape((512, 1024))
    altaz = c.transform_to(antenna_frame)

    seconds_since_ny1970 = 1474043217.33333333
    gstt = crda.gst(seconds_since_ny1970)
    mya_raa, mya_dec = crda.galactic_to_radec(
        sky_model.coords.b.deg, sky_model.coords.l.deg
    )
    mya_azz, mya_el = crda.radec_azel(
        gstt - mya_raa + loc.lon.rad, mya_dec, loc.lat.rad
    )

    my_azel = apc.SkyCoord(alt=mya_el * u.rad, az=mya_azz * u.rad, frame=antenna_frame)

    diff = my_azel.separation(altaz.reshape(my_azel.shape))

    # Pretty weak measure of "working": 1/4 of a degree different from astropy.
    assert np.max(diff.deg) < 0.25


def test_alan_coordinates_toyrday():
    seconds_since_ny1970 = 1474043217.33333333
    year, day, hour, minute, sec = crda.toyrday(seconds_since_ny1970)
    assert year == 2016
    assert day == 260
    assert hour == 16


def test_ra_dec_at_lst():
    az, el = crda.radec_azel_from_lst(lst=0, ra=0, dec=0, lat=0)
    assert az == 0.0
    assert el == np.pi / 2
