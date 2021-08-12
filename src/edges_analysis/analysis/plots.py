"""Plotting utilities."""
import datetime as dt

import healpy as hp
import matplotlib.pyplot as plt
import numpy as np
from astropy import coordinates as apc
from astropy import time as apt
from astropy import units as apu
from scipy import interpolate as interp

from . import beams
from . import sky_models
from .. import const


def plot_sky_model():
    """Plot a Haslam sky model."""
    # Loading Haslam map
    map408, (lon, lat, gc) = sky_models.haslam_408MHz_map()
    ipole = 2.65
    icenter = 2.4
    sigma_deg = 8.5
    i1 = ipole - (ipole - icenter) * np.exp(-(1 / 2) * (np.abs(lat) / sigma_deg) ** 2)
    Tcmb = 2.725
    s = (map408 - Tcmb) * (90 / 408) ** (-i1) + Tcmb
    hp.cartview(
        np.log10(s),
        nest="true",
        coord=("G", "C"),
        flip="geo",
        title="",
        notext="true",
        min=2.7,
        max=4.3,
        unit=r"log($T_{\mathrm{sky}}$)",
        rot=[5.76667 * 15, 0, 0],
        cmap="jet",
    )

    hp.graticule(local=True)
    beam = beams.feko_read(
        "mid_band", 0, frequency_interpolation=False, az_antenna_axis=90
    )
    beam90 = beam[20, :, :]
    beam90n = beam90 / np.max(beam90)
    FWHM = np.zeros((360, 2))
    EL_raw = np.arange(0, 91, 1)
    EL_new = np.arange(0, 90.01, 0.01)
    for j in range(len(beam90[0, :])):
        func = interp.interp1d(EL_raw, beam90n[:, j])
        beam90n_interp = func(EL_new)

        min_diff = 100
        for el, beam in zip(EL_new, beam90n_interp):
            diff = np.abs(beam, -0.5)
            if diff < min_diff:
                min_diff = np.copy(diff)
                FWHM[j, 0] = j
                FWHM[j, 1] = 90 - el

    def get_ra_dec(time_iter_utc):
        time_iter_utc_dt = dt.datetime(*time_iter_utc)
        alt_az = apc.SkyCoord(
            alt=(90 - FWHM[:, 1]) * apu.deg,
            az=FWHM[:, 0] * apu.deg,
            frame="altaz",
            obstime=apt.Time(time_iter_utc_dt, format="datetime"),
            location=const.edges_location,
        )
        ra_dec = alt_az.icrs
        ra = np.asarray(ra_dec.ra)
        dec = np.asarray(ra_dec.dec)
        ra[ra > 180] -= 360
        return ra, dec

    # Converting Beam Contours from Local to Equatorial coordinates
    ra_start, dec_start = get_ra_dec([2014, 1, 1, 3, 31, 0])
    ra_middle, dec_middle = get_ra_dec([2014, 1, 1, 9, 30, 0])
    ra_end, dec_end = get_ra_dec([2014, 1, 1, 15, 29, 0])

    plt.plot(np.arange(-180, 181, 1), -26.7 * np.ones(361), "y--", linewidth=2)
    plt.plot(ra_start, dec_start, "w", linewidth=3)
    plt.plot(ra_middle, dec_middle, "w--", linewidth=3)
    plt.plot(ra_end, dec_end, "w:", linewidth=3)
    plt.plot(-6 * (360 / 24), -26.7, "x", color="1", markersize=5, mew=2)
    plt.plot(0 * (360 / 24), -26.7, "x", color="1", markersize=5, mew=2)
    plt.plot(6 * (360 / 24), -26.7, "x", color="1", markersize=5, mew=2)
    off_x = -4
    off_y = -12
    for i in range(0, 26, 2):
        plt.text(-180 + i * 15 + off_x, -90 + off_y, str(i))

    plt.text(-60, -115, "galactic hour angle [hr]")
    off_y = -3
    for j in range(90, -120, 30):
        off_x = -15 if j > 0 else (-10 if j == 0 else -19)
        plt.text(-180 + off_x, j + off_y, str(j))
    plt.text(-210, 45, "declination [degrees]", rotation=90)
