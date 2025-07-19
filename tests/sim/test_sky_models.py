"""Test sky models."""

import numpy as np
from astropy import coordinates as apc
from astropy import units as u
from astropy.time import Time
from read_acq import _coordinates as crda

from edges import const
from edges.sim import sky_models as sm


def test_haslam_accuracy():
    """Simply test that the two HASLAM variants are close together."""
    h1 = sm.Haslam408()
    h2 = sm.Haslam408AllNoh()

    # Interpolate the healpix one to the same grid as the theta/phi one.
    h1_interp = h1.healpix.interpolate_bilinear_skycoord(h2.coords, h1.temperature)

    # Less than 2% of the pixels should be more than 10% different.
    assert (
        np.sum(np.abs(h1_interp - h2.temperature) / h2.temperature > 0.1)
        / len(h2.temperature)
        < 0.02
    )


def test_alan_coordinates_azel():
    sky_model = sm.Haslam408AllNoh()
    t = Time("2016-09-16T16:26:57", format="isot", scale="utc")
    loc = const.KNOWN_TELESCOPES["edges-low-alan"].location

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
