"""Plotting utilities."""
import datetime as dt

import healpy as hp
import matplotlib.pyplot as plt
import numpy as np
from astropy import coordinates as apc
from astropy import time as apt
from astropy import units as apu
from scipy import interpolate as interp
import edges_cal.modelling as mdl
from typing import Sequence

from . import beams
from . import sky_models
from . import const
from .gsdata import GSData
from .averaging.lstbin import lst_bin_with_models, lst_bin_direct, lst_bin
from .averaging import averaging

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


def plot_waterfall(
    data: GSData,
    load: int = 0,
    pol: int = 0,
    which_flags: tuple[str] = None,
    ignore_flags: tuple[str] = (),
    ax: plt.Axes | None = None,
    cbar: bool=True,
    xlab: bool=True,
    ylab: bool=True,
    title: bool | str=True,
    attribute: str = 'spectra',
    **imshow_kwargs,
):
    q = getattr(data, attribute)
    if q.shape != data.data.shape:
        raise ValueError(f"Cannot use attribute '{attribute}' as it doesn't have the same shape as data.")

    q = np.where(data.get_flagged_nsamples(which_flags, ignore_flags) == 0, np.nan, q)
    q = q[load, :, :, pol]

    if ax is None:
        ax = plt.subplots(1, 1)[1]

    # If given model parameters, assume we want to plot residuals.
    if model is not None:
        model = model.at(x=data.freq_array)
        for i, p in model_params:
            q[i] -= model(parameters=p)

        cmap = imshow_kwargs.pop("cmap", "coolwarm")
    else:
        cmap = imshow_kwargs.pop("cmap", "magma")

    times = data.time_array
    
    img = ax.imshow(
        q,
        origin="lower",
        extent=(
            data.freq_array.min(),
            data.freq_array.max(),
            times.min(),
            times.max(),
        ),
        cmap=cmap,
        aspect="auto",
        interpolation="none",
        **imshow_kwargs,
    )

    if xlab:
        ax.set_xlabel("Frequency [MHz]")
    if ylab:
        ax.set_ylabel("Hours into Observation")
    

    if title and not isinstance(title, str):
        if not data.in_lst:
            ax.set_title(f"{data.get_initial_yearday()}. LST0={data.lst_array[0]:.2f}")

    if cbar:
        cb = plt.colorbar(img, ax=ax)
        cb.set_label(data.loads[load])

    return ax

def plot_time_average(
    data: GSData,
    ax: plt.Axes | None = None, 
    logy=None, 
    lst_min: float = 0,
    lst_max: float = 24, 
    load: int= 0,
    pol: int = 0,
    attribute: str = 'spectra',
    offset: float = 0.0
):
    if ax is not None:
        fig = ax.figure
    else:
        fig, ax = plt.subplots(1, 1)

    if lst_min > 0 or lst_max < 24:
        data = data.select_times(range=(lst_min, lst_max))

    data = lst_bin(data, binsize=24.0)

    q = getattr(data, attribute)
    if q.shape != data.data.shape:
        raise ValueError(f"Cannot use attribute '{attribute}' as it doesn't have the same shape as data.")

    ax.plot(data.freq_array, q[load, pol, 0] - offset)
    ax.set_xlabel("Frequency [MHz]")
    ax.set_ylabel("Average Spectrum")

    if logy is None:
        logy = np.all(q > 0)

    if logy:
        ax.set_yscale("log")

    return ax, data

def plot_daily_residuals(
    objs: list[GSData],
    model: mdl.Model | None, 
    separation: float = 20.0,
    ax: plt.Axes | None = None,
    load: int= 0,
    pol: int=0,
    **kw
) -> plt.Axes:
    """
    Make a single plot of residuals for each day in the dataset.

    Parameters
    ----------
    separation
        The separation between residuals in K (on the plot).
    ax
        An optional axis on which to plot.
    gha_min
        A minimum GHA to include in the averaged residuals.
    gha_max
        A maximum GHA to include in the averaged residuals.
    freq_resolution
        The frequency resolution to bin the spectra into for the plot. In same
        units as the instance frequencies.
    days
        The integer day numbers to include in the plot. Default is to include
        all days in the dataset.
    weights
        The weights to use for flagging. By default, use the weights of the object.
        If 'old' is given, use the pre-filter weights. Otherwise, must be an array
        the same size as the spectrum/resids.

    Returns
    -------
    ax
        The matplotlib Axes on which the plot is made.
    """
    if ax is None:
        fig, ax = plt.subplots(1, 1)

    for i, data in enumerate(objs):
        if data.data_model is None and model is None:
            raise ValueError("If data has no model, must provide one!")

        if data.data_model is None:
            data = data.add_model(model)

        ax, data = plot_time_average(
            data,
            attribute='resids',
            offset=separation*i,
            ax = ax,
            **kw
        )

        rms = np.sqrt(
            averaging.weighted_mean(data=data.resids[load, pol] ** 2, weights=data.nsamples[load, pol])[0]
        )
        ax.text(
            data.freq_array.max() + 5 * u.MHz,
            -i * separation,
            f"{data.get_initial_yearday()} RMS={rms:.2f}",
        )

    return ax