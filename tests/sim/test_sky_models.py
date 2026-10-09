"""Test sky models."""

import astropy_healpix as ahp
import attrs
import numpy as np
import pytest
from astropy import coordinates as apc
from astropy import units as u
from astropy.coordinates import Galactic
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


def _synthetic_guzman_like() -> sm.SkyModel:
    hp = ahp.HEALPix(nside=8, order="ring", frame=Galactic())
    coords = hp.healpix_to_skycoord(np.arange(hp.npix))
    dec = coords.icrs.dec.deg
    # Big-endian float32, like a FITS column.
    temp = (5000.0 + 100 * np.cos(coords.l.rad)).astype(">f4")
    temp[dec > 68.0] = -32768
    temp[dec < -78.0] = -32768
    temp[100] = -32768  # a stray blank pixel
    return sm.SkyModel(frequency=45.0, temperature=temp, coords=coords, healpix=hp)


def test_guzman45_masks_blank_pixels(monkeypatch):
    raw = _synthetic_guzman_like()
    monkeypatch.setattr(
        sm.SkyModel, "from_lambda", classmethod(lambda cls, *args, **kwargs: raw)
    )
    out = sm.Guzman45()

    raw_temp = raw.temperature.astype(float)
    dec = raw.coords.icrs.dec.deg
    blank = raw_temp <= -1e4
    ncp = dec > 68.0

    # Good pixels are kept as they are.
    np.testing.assert_allclose(out.temperature[~blank], raw_temp[~blank], rtol=0)
    # Blank pixels outside the NCP cap (e.g. around the SCP) are NaN.
    assert np.all(np.isnan(out.temperature[blank & ~ncp]))
    assert np.sum(blank & ~ncp & (dec < -78)) > 0
    # The NCP cap is filled with the mean of the band just below it.
    band = (dec > 60) & (dec < 68)
    np.testing.assert_allclose(
        out.temperature[ncp], np.mean(raw_temp[band & ~blank]), rtol=1e-12
    )
    assert not np.any(out.temperature <= -1e4)
    # The input was not modified.
    assert np.sum(raw.temperature <= -1e4) == np.sum(blank)


def test_skymodel_equality():
    s1 = sm.SkyModel.uniform_healpix(408.0, nside=4)
    s2 = sm.SkyModel.uniform_healpix(408.0, nside=4)
    assert s1 == s2

    s3 = attrs.evolve(s1, temperature=s1.temperature * 2)
    assert s1 != s3

    s4 = sm.SkyModel.uniform_healpix(408.0, nside=8)
    assert s1 != s4

    temp = s1.temperature.copy()
    temp[0] = np.nan
    assert attrs.evolve(s1, temperature=temp) == attrs.evolve(
        s1, temperature=temp.copy()
    )

    s5 = attrs.evolve(s1, coords=s1.coords.icrs)
    assert s1 != s5


def test_at_freq_non_galactic_frame():
    """The spectral index model is evaluated in Galactic coordinates in any frame."""
    gal = sm.SkyModel.uniform_healpix(408.0, nside=4)
    icrs = attrs.evolve(gal, coords=gal.coords.icrs)
    idx = sm.GaussianIndex()
    np.testing.assert_allclose(
        icrs.at_freq(np.array([50.0, 100.0]), index_model=idx),
        gal.at_freq(np.array([50.0, 100.0]), index_model=idx),
        rtol=1e-8,
    )


@pytest.mark.parametrize("shape", [(18, 36), (36, 72)])
def test_latlon_grid_pixel_res(tmp_path, shape):
    """Pixel solid angles of a lat-lon grid sum to ~4pi for any grid shape."""
    fname = tmp_path / "grid.txt"
    np.savetxt(fname, np.full(shape, 1000.0))
    sky = sm.SkyModel.from_latlon_grid_file(fname, frequency=408.0)
    assert sky.temperature.shape == (shape[0] * shape[1],)
    np.testing.assert_allclose(np.sum(sky.pixel_res), 4 * np.pi, rtol=5e-3)
