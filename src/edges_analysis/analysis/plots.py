import datetime as dt
import os
from typing import Optional, Sequence
from pathlib import Path

import healpy as hp
import matplotlib.pyplot as plt
import numpy as np
from astropy import coordinates as apc
from astropy import time as apt
from astropy import units as apu
from edges_cal import xrfi as rfi
from edges_cal import modelling as mdl
from scipy import interpolate as interp
import h5py

from . import beams, filters, loss, levels, coordinates as cd
from . import sky_models, tools
from ..config import config
from .. import const


def plot_level3_ancillary(level3, plot_file: [None, str, Path] = None):
    fig, ax = plt.subplots(5, 1, figsize=[4.5, 9], sharex=True)

    day = level3.meta["day"]

    #        f, t, p, r, w, rms, tp, m = io.level3read(direc + fl)
    gha = level3.ancillary["gha"]

    index_gha_6 = None
    index_gha_18 = None
    if np.any(gha < 6) and np.any(gha > 6):
        index_gha_6 = np.where(gha > 6)[0][0]
    if np.any(gha < 18) and np.any(gha > 18):
        index_gha_18 = np.where(gha > 18)[0][0]

    def plot_an_axis(thing, ax, ylim, yticks, ylabel):
        ax.plot(thing, "b", linewidth=2)
        if index_gha_6:
            ax.axvline(index_gha_6, color="g", ls="--", lw=2)
        if index_gha_18 > -1:
            ax.axvline(index_gha_18, color="r", ls="--", lw=2)
        ax.set_ylim(ylim)
        ax.yaxis.yticks(yticks)
        ax.set_ylabel(ylabel)
        ax.grid(True)

    if index_gha_6:
        ax[0].text(index_gha_6, 12, " 6hr", rotation=0, color="g", fontsize=10)
    if index_gha_18:
        ax[0].text(index_gha_18, 12, " 18hr", rotation=0, color="r", fontsize=10)

    ax[0].set_title(day)
    plot_an_axis(gha, ax[0], [-1, 26], [0, 6, 12, 18, 24], "GHA [hr]")
    plot_an_axis(
        level3.ancillary["sun_azel"][:, 1],
        ax[1],
        [-110, 110],
        [-90, -45, 0, 45, 90],
        "sun elev [deg]",
    )
    plot_an_axis(
        level3.ancillary["ambient_temp"],
        ax[2],
        [0, 40],
        [10, 20, 30],
        r"amb temp [$^{\circ}$C]",
    )
    plot_an_axis(
        level3.ancillary["ambient_humidity"],
        ax[3],
        [-30, 110],
        [-20, 0, 20, 40, 60, 80, 100],
        "amb humid [%]",
    )
    plot_an_axis(
        level3.ancillary["receiver1_temp"],
        ax[4],
        [23, 27],
        [24, 25, 26],
        r"rec temp [$^{\circ}$C]",
    )

    ax[-1].xlabel("time [number of raw spectra since start of file]")

    if plot_file:
        if not Path(plot_file).is_absolute():
            plot_file = str(level3.filename).replace(".h5", "_ancillary_plots.pdf")

        plt.savefig(str(plot_file), bbox_inches="tight")


def comparison_switch_receiver1(reflection_measurements):
    """
    Create a plot comparing multiple reflection coefficient measurements.

    Parameters
    ----------
    reflection_measurements : dict
        A dictionary where the keys are labels, and the values are lists of lists.
        The elements of each should be (s11, s12, s22), as generated from
        :func:`switch_corrections`.
    """
    f = np.arange(50, 201)
    fig, ax = plt.subplots(3, 2, figsize=(12, 8))

    for i in range(3):  # s11, s12, s21
        for label, coeffs in reflection_measurements.items():
            ax[i, 0].plot(f, 20 * np.log10(np.abs(coeffs[i])), label=label)
            ax[i, 1].plot(f, (180 / np.pi) * np.unwrap(np.angle(coeffs[i])), label=label)

    return fig, ax


def plot_season_average_residuals(
    case,
    n_fg=3,
    ddy=1.5,
    title="No Beam Correction, Residuals to 5 LINLOG terms, 61-159 MHz",
    figure_name="no_beam_correction",
):
    def path(kind):
        return (
            config["field_products"]
            + f"mid_band/spectra/level5/case2/case2_{case.split('_')+'_' if kind != 'frequency' else ''}{kind}.txt"
        )

    fb = np.genfromtxt(path("frequency"))
    ty = np.genfromtxt(path("temperature"))
    wy = np.genfromtxt(path("weights"))
    delta_hr = int(case[0])

    parameters = {
        1: {
            "f_low": 61,
            "f_high": 159,
            "plot_f_low": 30,
            "plot_f_high": 165,
            "xticks": np.arange(60, 160 + 1, 20),
        },
        2: {
            "f_low": 61,
            "f_high": 136,
            "plot_f_low": 30,
            "plot_f_high": 145,
            "xticks": np.arange(60, 140 + 1, 20),
        },
        3: {
            "f_low": 110,
            "f_high": 159,
            "plot_f_low": 70,
            "plot_f_high": 165,
            "xticks": np.arange(100, 160 + 1, 20),
        },
    }

    if int(case.split("_")[-1]) not in parameters:
        raise ValueError("case must be one of {}".format(parameters.keys()))

    parameters = parameters[case.split("_")[-1]]
    ff, rr, ww = tools.spectra_to_residuals(
        fb, ty, wy, parameters["f_low"], parameters["f_high"], n_fg
    )
    ar = np.arange(0, 25, delta_hr)
    str_ar = ["GHA=" + str(ar[i]) + "-" + str(ar[i + 1]) + " hr" for i in range(len(ar) - 1)]

    plot_residuals(
        ff,
        rr,
        ww,
        str_ar,
        FIG_SX=7,
        FIG_SY=12,
        DY=ddy,
        f_low=parameters["plot_f_low"],
        f_high=parameters["plot_f_high"],
        XTICKS=parameters["xticks"],
        XTEXT=32,
        YLABEL=str(ddy) + " K per division",
        TITLE=title,
        save=True,
        figure_name=figure_name,
        figure_format="pdf",
    )


def plot_residuals(
    f,
    r,
    w,
    list_names,
    FIG_SX=7,
    FIG_SY=12,
    DY=2,
    f_low=50,
    f_high=180,
    XTICKS=np.arange(60, 180 + 1, 20),
    XTEXT=160,
    YLABEL="ylabel",
    TITLE="",
    save=False,
    figure_path="/home/raul/Desktop/",
    figure_name="2018_150_00",
    figure_format="png",
):
    n_spec = len(r[:, 0])

    plt.figure(figsize=(FIG_SX, FIG_SY))

    for i in range(len(list_names)):
        plt.plot(f[w[i] > 0], (r[i] - i * DY)[w[i] > 0], color="rb"[i % 2])
        plt.text(XTEXT, -i * DY, list_names[i])

    plt.xlim([f_low, f_high])
    plt.ylim([-DY * n_spec, DY])
    plt.grid()
    plt.xticks(XTICKS)
    plt.yticks([])
    plt.xlabel("frequency [MHz]")
    plt.ylabel(YLABEL)
    plt.title(TITLE)

    if save:
        plt.savefig(figure_path + figure_name + "." + figure_format, bbox_inches="tight")


def plot_level3_rms(level3, ylim_lower=None, ylim_upper=None):
    """
    This function plots the RMS of residuals of Level3 data
    """
    rms_lower = level3.get_model_rms(freq_range=(level3.freq.min, level3.freq.center))
    rms_upper = level3.get_model_rms(freq_range=(level3.freq.min, level3.freq.center))

    fig, ax = plt.subplots(2, 1, sharex=True)

    ax[0].plot(level3.ancillary["lst"], rms_lower, ".")
    plt.xticks(np.arange(0, 25, 2))
    plt.grid()
    plt.xlim([0, 24])
    plt.ylim(ylim_lower)

    plt.ylabel("RMS [K]")
    ax[0].title(f"{level3.filename}: Low-frequency Half")

    ax[1].plot(level3.ancillary["lst"], rms_upper, ".r")
    plt.grid()
    plt.ylim(ylim_upper)

    plt.ylabel("RMS [K]")
    plt.xlabel("LST [Hr]")
    plt.title(f"{level3.filename}:  High-frequency Half")

    return fig, ax


def plot_sky_model():
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
    beam = beams.feko_read("mid_band", 0, frequency_interpolation=False, az_antenna_axis=90)
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


def plot_sky_model_comparison():
    # Loading Haslam map
    map1, (lon1, lat1, gc1) = sky_models.haslam_408MHz_map()
    # Loading LW map
    map2, (lon2, lat2, gc2) = sky_models.LW_150MHz_map()
    # Loading Guzman map
    map3, (lon3, lat3, gc3) = sky_models.guzman_45MHz_map()

    # Scaling sky map (the map contains the CMB, which has to be removed and then added back)
    ipole = 2.65
    icenter = 2.4
    sigma_deg = 8.5
    i1 = ipole - (ipole - icenter) * np.exp(-(1 / 2) * (np.abs(lat1) / sigma_deg) ** 2)
    i2 = ipole - (ipole - icenter) * np.exp(-(1 / 2) * (np.abs(lat2) / sigma_deg) ** 2)
    i3 = ipole - (ipole - icenter) * np.exp(-(1 / 2) * (np.abs(lat3) / sigma_deg) ** 2)
    i12 = 2.56 * np.ones(len(lat1))
    band_deg = 10
    index_inband = 2.4
    index_outband = 2.65
    i13 = np.zeros(len(lat1))
    i13[np.abs(lat1) <= band_deg] = index_inband
    i13[np.abs(lat1) > band_deg] = index_outband
    Tcmb = 2.725
    s1 = (map1 - Tcmb) * (90 / 408) ** (-i1) + Tcmb
    s2 = (map1 - Tcmb) * (90 / 408) ** (-i12) + Tcmb
    s4 = (map2 - Tcmb) * (90 / 150) ** (-i2) + Tcmb
    s5 = (map3 - Tcmb) * (90 / 45) ** (-i3) + Tcmb
    ss4 = hp.pixelfunc.ud_grade(s4, 512, order_in="NESTED")
    ss5 = hp.pixelfunc.ud_grade(s5, 512, order_in="NESTED")
    hp.cartview(s1, nest=True, min=500, max=2000, cbar=False, coord="GC")
    hp.cartview(s2, nest=True, min=500, max=2000, cbar=False, coord="GC")
    hp.cartview(ss4, nest=True, min=500, max=2000, cbar=False, coord="GC")
    hp.cartview(ss5, nest=True, min=500, max=2000, cbar=False, coord="GC")
    dLIM = 500
    hp.cartview(s2 - s1, nest=True, min=-dLIM, max=dLIM, cbar=False, coord="GC")
    hp.cartview(ss4 - s1, nest=True, min=-dLIM, max=dLIM, cbar=False, coord="GC")
    hp.cartview(ss5 - s1, nest=True, min=-dLIM, max=dLIM, cbar=False, coord="GC")


def plot_beam_power(max_norm=True):
    def get_sm(beam_file, low_band=False):
        if not low_band:
            bm_all = beams.feko_read(
                "mid_band", beam_file, frequency_interpolation=False, az_antenna_axis=90
            )
            f = np.arange(50, 201, 2)
        else:
            bm_all = beams.FEKO_low_band_blade_beam(
                beam_file, frequency_interpolation=False, az_antenna_axis=0
            )
            f = np.arange(40, 121, 2)

        el = np.arange(0, 91)
        sin_theta = np.sin((90 - el) * (np.pi / 180))
        sin_theta_2D = np.tile(sin_theta, (360, 1)).T

        sm = np.zeros(len(f))
        for i in range(len(f)):
            bm = bm_all[i, :, :]
            normalization = np.max(np.max(bm)) if max_norm else 1

            nb = bm / normalization

            s_sq_deg = np.sum(nb * sin_theta_2D)
            sm[i] = s_sq_deg / ((180 / np.pi) ** 2)
        return sm, f

    def plot_ax(sm, f, ls, label):
        fmask = (f >= 50) & (f <= 110)

        f = f[fmask]
        sm = sm[fmask]

        pm = np.polyfit(f, sm, 4)
        mm = np.polyval(pm, f)
        ax[0].plot(f, sm, ls=ls, label=label)
        ax[1].plot(f, sm - mm)

    sm, f = get_sm(0)
    smi, f = get_sm(1)
    sl, fl = get_sm(2, True)

    fig, ax = plt.subplots(2, 1)
    plot_ax(
        sm,
        f,
        "-",
        "Mid-Band 30mx30m ground plane",
    )
    plot_ax(
        smi,
        f,
        ":",
        "Mid-Band infinite ground plane",
    )
    plot_ax(
        sl,
        fl,
        "--",
        "Low-Band 30mx30m ground plane",
    )

    if max_norm:
        ax[0].set_ylabel("solid angle of\n beam above horizon [sr]")
    else:
        ax[0].set_ylabel("normalized total radiated power\n above horizon [fraction of 4pi]")
    ax[0].legend()

    ax[1].set_ylabel("residuals to\n 5-term polynomial [sr]")
    ax[1].set_xlabel("frequency [MHz]")
    return fig, ax


def beam_chromaticity_differences(files):
    # Plot
    size_x = 4.7
    size_y = 5
    x0 = 0.13
    y0 = 0.035
    dx = 0.53
    dy = 0.55
    xoff = 0.09
    dxc = 0.03

    bf = []
    for fl in files:
        beam_fac = beams.InterpolatedBeamFactor(fl)
        lst = beam_fac["lst"]
        bf_ = beam_fac["beam_factor"]
        gha = cd.lst2gha(lst)
        indx = np.argsort(gha)
        bf.append(bf_[indx])

    fig, ax = plt.subplots(1, 4, figsize=(size_x, size_y))
    scale_max = 0.0043
    scale_min = -0.0043

    cmap = plt.cm.viridis
    rgba = cmap(0.0)
    cmap.set_under(rgba)

    for i in range(4):
        im = ax[i].imshow(
            bf[i + 1] - bf[0],
            interpolation="none",
            extent=[50, 200, 24, 0],
            aspect="auto",
            vmin=scale_min,
            vmax=scale_max,
            cmap=cmap,
        )
        ax[i].axhline(6, "w--", linewidth=2)
        ax[i].axhline(18, "w--", linewidth=2)
        ax[i].set_xlim([60, 120])
        ax[i].set_xticks(np.arange(60, 121, 10))
        ax[i].set_yticks(np.arange(0, 25, 3))
        ax[i].set_xlabel(r"$\nu$ [MHz]", fontsize=14)
        if i == 0:
            ax[i].set_ylabel("GHA [hr]", fontsize=14)
        ax[i].set_title(f"({'abcd'[i]})", fontsize=18)

    cax = fig.add_axes([x0 + 3.2 * xoff + 4 * dx, y0, dxc, dy])
    fig.colorbar(im, cax=cax, orientation="vertical")
    cax.set_title(r"$\Delta C$", fontsize=14)

    return fig, ax


def plot_beam_chromaticity_correction(beam_factor_file):
    # Plot
    size_x = 4.7
    size_y = 5
    x0 = 0.13
    y0 = 0.035
    dx = 0.67
    dy = 0.6
    dy1 = 0.2
    yoff = 0.05
    dxc = 0.03
    xoffc = 0.03

    beam_fac = beams.InterpolatedBeamFactor(beam_factor_file)
    lst = beam_fac["lst"]
    f = beam_fac["frequency"]
    bf = beam_fac["beam_factor"]
    gha = cd.lst2gha(lst)
    indx = np.argsort(gha)
    bf = bf[indx]

    f1 = plt.figure(num=1, figsize=(size_x, size_y))
    ax = f1.add_axes([x0, y0 + 1 * (yoff + dy1), dx, dy])
    im = ax.imshow(
        bf,
        interpolation="none",
        extent=[50, 200, 24, 0],
        aspect="auto",
        vmin=0.979,
        vmax=1.021,
    )
    ax.plot(f, 6 * np.ones(len(f)), "w--", linewidth=1.5)
    ax.plot(f, 18 * np.ones(len(f)), "w--", linewidth=1.5)
    ax.set_xlim([60, 120])
    ax.set_xticklabels("")
    ax.set_yticks(np.arange(0, 25, 3))
    ax.set_ylabel("GHA [hr]")
    cax = f1.add_axes([x0 + 1 * dx + xoffc, y0 + 1 * (yoff + dy1), dxc, dy])
    f1.colorbar(im, cax=cax, orientation="vertical", ticks=[0.98, 0.99, 1, 1.01, 1.02])
    cax.set_title("$C$")
    ax = f1.add_axes([x0, y0, dx, dy1])
    ax.plot(f, bf[125, :], "k")
    ax.plot(f, bf[175, :], "k--")
    ax.legend(["GHA=10 hr", "GHA=14 hr"], fontsize=8, ncol=2)
    ax.set_ylim([0.9, 1.1])
    ax.set_xlim([60, 120])
    ax.set_ylim([0.975, 1.025])
    ax.set_xlabel(r"$\nu$ [MHz]")
    ax.set_yticks([0.98, 1, 1.02])
    ax.set_ylabel("$C$")

    return f1, ax


def plot_ground_loss(ax=None):
    fe = np.linspace(0, 200, 2000)

    if ax is None:
        fig, ax = plt.subplots(1, 1)

    Gg = loss.ground_loss("mid_band", fe)
    flb = np.arange(50, 121, 1)
    Gglb = loss.ground_loss("low_band", flb)
    ax.plot(fe, (1 - Gg) * 100, "b", linewidth=1.0, label="ground loss [%]")
    ax.plot(flb, (1 - Gglb) * 100, "r", linewidth=1.0, label="ground loss [%]")
    ax.set_ylabel(r"ground loss [%]")  # , fontsize=15)
    ax.set_xlim([48, 122])
    xt = np.arange(50, 121, 10)
    ax.set_xticks(xt)
    ax.set_xticklabels("")
    ax.tick_params(axis="x", direction="in")
    ax.set_ylim([0.1, 0.5])
    ax.set_yticks(np.arange(0.15, 0.46, 0.05))
    ax.text(115, 0.118, "(c)", fontsize=14)
    return fig, ax


def _get_xf(path, f_high=np.inf):
    x = np.genfromtxt(path)
    f = x[:, 0] / 1e6
    ra = x[:, 1] + 1j * x[:, 2]
    return f[f <= f_high], ra[f <= f_high]


def plot_beam_factor(
    az_above_horizon,
    el_above_horizon,
    irf,
    lst,
    path_plots,
    plot_format,
    sky_map,
    sky_ref_above_horizon,
):
    LAT_DEG = np.copy(const.edges_lat_deg)
    AZ_plot = np.copy(az_above_horizon)
    AZ_plot[AZ_plot > 180] -= 360
    EL_plot = np.copy(el_above_horizon)
    SKY_plot = np.copy(sky_ref_above_horizon)
    max_log10sky = np.max(np.log10(sky_map[:, irf]))
    min_log10sky = np.min(np.log10(sky_map[:, irf]))
    marker_size = 10
    GHA = cd.lst2gha(lst)

    if plot_format == "rect":
        plt.figure(figsize=[19, 6])
        plt.scatter(
            AZ_plot,
            EL_plot,
            edgecolors="none",
            s=marker_size,
            c=np.log10(SKY_plot),
            vmin=min_log10sky,
            vmax=max_log10sky,
        )
        plt.xticks(np.arange(-180, 181, 30))
        plt.yticks([0, 15, 30, 45, 60, 75, 90])
        cbar = plt.colorbar()
        cbar.set_label("log10( Tsky @ 50MHz [K] )", rotation=90)
        plt.xlabel("AZ [deg]")
        plt.ylabel("EL [deg]")
        plt.title(f"LAT={LAT_DEG:.3f} [deg] \n\n LST={lst:.3f} hr        GHA={GHA:.3f} hr")
    elif plot_format == "polar":
        fig = plt.figure(figsize=[11.5, 10])
        ax = fig.add_subplot(111, projection="polar")
        c = ax.scatter(
            (np.pi / 180) * AZ_plot,
            90 - EL_plot,
            edgecolors="none",
            s=marker_size,
            c=np.log10(SKY_plot),
            vmin=min_log10sky,
            vmax=5,
        )
        ax.set_theta_offset(-np.pi / 2)
        ax.set_ylim([0, 90])
        ax.set_yticks([0, 30, 60, 90])
        ax.set_yticklabels(["90", "60", "30", "0"])
        plt.text(-2 * (np.pi / 180), 101, "AZ", fontsize=14, fontweight="bold")
        plt.text(22 * (np.pi / 180), 95, "EL", fontsize=14, fontweight="bold")
        plt.text(
            45 * (np.pi / 180),
            143,
            "Raul Monsalve",
            fontsize=8,
            color=[0.5, 0.5, 0.5],
        )
        plt.title(
            f"LAT={LAT_DEG:.3f} [deg] \n\n LST={lst:.3f} hr        GHA={GHA:.3f} hr",
            fontsize=14,
            fontweight="bold",
        )
        cbar_ax = fig.add_axes([0.9, 0.3, 0.02, 0.4])
        hcbar = fig.colorbar(c, cax=cbar_ax)
        hcbar.set_label("log10( Tsky @ 50MHz [K] )", rotation=90)
    else:
        raise ValueError("plot_format must be either 'polar' or 'rect'.")

    plt.savefig(
        path_plots + f"LST_{lst:.3f} hr.png",
        bbox_inches="tight",
    )
