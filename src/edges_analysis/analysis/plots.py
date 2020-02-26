import datetime as dt
import os
from os import listdir, makedirs
from os.path import exists

import healpy as hp
import matplotlib.pyplot as plt
import numpy as np
from astropy import coordinates as apc
from astropy import time as apt
from astropy import units as apu
from edges_cal import EdgesFrequencyRange, reflection_coefficient as rc
from edges_cal import modelling as mdl
from edges_io.io import S1P
from scipy import interpolate as interp

from . import beams, filters, io, loss, rfi
from . import s11 as s11m
from . import sky_models, tools
from ..config import config


def plot_daily_residuals_nominal(f, r, w, yd, path="/home/raul/Desktop/"):
    # Filter the data.
    keep = filters.daily_nominal_filter("mid_band", 1, yd)
    r, w, yd = r[keep], w[keep], yd[keep]

    for i, (rr, ww) in enumerate(zip(r, w)):
        fb, rb, wb = tools.spectral_binning_number_of_samples(f, rr, ww)

        if i == 0:
            rb_all = np.zeros((len(r), len(fb)))
            wb_all = np.zeros((len(r), len(fb)))

        rb_all[i] = rb
        wb_all[i] = wb

    lst_text = [str(int(ydd[1])) for ydd in yd]
    dy = 0.7

    f_low_plot = 53
    f_high_plot = 122

    xticks = np.arange(60, 121, 20)
    xtext = 54
    ylabel = str(dy) + " K per division"
    title = ""

    figure_name = "daily_residuals_nominal_59_121MHz_GHA_6_18hr"

    fig_sx = 6
    fig_sy = 15

    # Plotting
    plot_residuals(
        fb,
        rb_all,
        wb_all,
        lst_text,
        FIG_SX=fig_sx,
        FIG_SY=fig_sy,
        DY=dy,
        f_low=f_low_plot,
        f_high=f_high_plot,
        XTICKS=xticks,
        XTEXT=xtext,
        YLABEL=ylabel,
        TITLE=title,
        save=True,
        figure_path=path,
        figure_name=figure_name,
    )


def plots_midband_metadata():
    direc = config["edges_folder"] + "/mid_band/spectra/level3/case_nominal/"
    list_files = os.listdir(direc)
    list_files.sort()

    fig, ax = plt.subplots(5, 1, figsize=[4.5, 9], sharex=True)

    # Processing files
    for i, fl in enumerate(list_files):

        day = fl[0:11]

        f, t, p, r, w, rms, tp, m = io.level3read(direc + fl)

        gha = m[:, 4]
        gha[gha < 0] += 24

        sun_el = m[:, 6]
        temp = m[:, 9]
        hum = m[:, 10]
        rec_temp = m[:, 11]

        index_gha_6 = -1
        index_gha_18 = -1
        for i in range(len(gha) - 1):
            if (gha[i] <= 6) and (gha[i + 1] > 6):
                index_gha_6 = i

            if (gha[i] <= 18) and (gha[i + 1] > 18):
                index_gha_18 = i + 1

        def plot_an_axis(
            thing, ax, ylim, yticks, ylabel, do_title=False, do_text=False
        ):
            ax.plot(thing, "b", linewidth=2)
            if index_gha_6 > -1:
                ax.plot([index_gha_6, index_gha_6], [-1000, 1000], "g--", linewidth=2)
                if do_text:
                    plt.text(
                        index_gha_6, 12, " 6hr", rotation=0, color="g", fontsize=10
                    )
            if index_gha_18 > -1:
                ax.plot([index_gha_18, index_gha_18], [-1000, 1000], "r--", linewidth=2)
                if do_text:
                    plt.text(
                        index_gha_18, 12, " 18hr", rotation=0, color="r", fontsize=10
                    )
            ax.set_ylim(ylim)
            ax.yaxis.yticks(yticks)
            ax.set_ylabel(ylabel)
            ax.grid()
            if do_title:
                ax.set_title(day)

        plot_an_axis(
            gha,
            ax[0],
            [-1, 26],
            [0, 6, 12, 18, 24],
            "GHA [hr]",
            do_title=True,
            do_text=True,
        )
        plot_an_axis(
            sun_el, ax[1], [-110, 110], [-90, -45, 0, 45, 90], "sun elev [deg]"
        )
        plot_an_axis(temp, ax[2], [0, 40], [10, 20, 30], r"amb temp [$^{\circ}$C]")
        plot_an_axis(
            hum, ax[3], [-30, 110], [-20, 0, 20, 40, 60, 80, 100], "amb humid [%]"
        )
        plot_an_axis(rec_temp, ax[4], [23, 27], [24, 25, 26], r"rec temp [$^{\circ}$C]")

    ax[-1].xlabel("time [number of raw spectra since start of file]")

    plt.savefig(
        config["edges_folder"]
        + "/mid_band/spectra/level3/case_nominal/metadata/"
        + day
        + ".png",
        bbox_inches="tight",
    )


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
            ax[i, 1].plot(
                f, (180 / np.pi) * np.unwrap(np.angle(coeffs[i])), label=label
            )

    return fig, ax


def receiver1_switch_crosscheck():
    path15 = (
        "/home/raul/DATA/EDGES_old/calibration/receiver_calibration/low_band1"
        "/2015_08_25C"
        "/data/s11/raw/20150903/switch25degC/"
    )
    path18 = (
        "/home/raul/DATA/EDGES/mid_band/calibration/receiver_calibration/receiver1"
        "/2018_01_25C/data/s11/raw/InternalSwitch/"
    )

    def get_correction(pth, dodgy):
        if dodgy:
            # Measurements at 25degC
            o_sw_m, f = S1P.read(pth + "open.S1P")
            s_sw_m, f = S1P.read(pth + "short.S1P")
            l_sw_m, f = S1P.read(pth + "load.S1P")

            o_sw_in, f = S1P.read(pth + "open_input.S1P")
            s_sw_in, f = S1P.read(pth + "short_input.S1P")
            l_sw_in, f = S1P.read(pth + "load_input.S1P")
        else:
            o_sw_m, f = S1P.read(pth + "Open01.s1p")
            s_sw_m, f = S1P.read(pth + "Short01.s1p")
            l_sw_m, f = S1P.read(pth + "Match01.s1p")

            o_sw_in, f = S1P.read(pth + "ExternalOpen01.s1p")
            s_sw_in, f = S1P.read(pth + "ExternalShort01.s1p")
            l_sw_in, f = S1P.read(pth + "ExternalMatch01.s1p")

        # Standards assumed at the switch
        o_sw = 1 * np.ones(len(fd))
        s_sw = -1 * np.ones(len(fd))
        l_sw = 0 * np.ones(len(fd))

        # Correction at the switch -- 25degC
        om = rc.de_embed(o_sw, s_sw, l_sw, o_sw_m, s_sw_m, l_sw_m, o_sw_in)[0]
        sm = rc.de_embed(o_sw, s_sw, l_sw, o_sw_m, s_sw_m, l_sw_m, s_sw_in)[0]
        lm = rc.de_embed(o_sw, s_sw, l_sw, o_sw_m, s_sw_m, l_sw_m, l_sw_in)[0]
        return f, om, sm, lm

    fd, om15, sm15, lm15 = get_correction(path15, True)
    f, om18, sm18, lm18 = get_correction(path18, False)

    # Plot
    fig1, ax1 = plt.subplots(2, 3)
    for i, (label, kind) in enumerate(
        zip(["Open", "Short", "Match"], [(om15, om18), (sm15, sm18), (lm15, lm18)])
    ):
        for j, (ylabel, fnc) in enumerate(
            zip(
                ["magnitude [dB]", "phase [deg]"],
                [
                    lambda x: 20 * np.log10(np.abs(x)),
                    lambda x: (180 / np.pi) * np.unwrap(np.angle(x)),
                ],
            )
        ):
            ax1[j, i].plot(fd / 1e6, fnc(om15), "k")
            ax1[j, i].plot(f / 1e6, fnc(om18), "r--")

            if i == 0:
                ax1[j, i].set_ylabel(ylabel)
            if j == 0:
                ax1[j, i].set_title(
                    "{} Standard at the Receiver Input\n(Measured from the Switch)".format(
                        label
                    )
                )

    fig2, ax2 = plt.subplots(1, 2)

    z15 = rc.gamma2impedance(lm15, 50)
    z18 = rc.gamma2impedance(lm18, 50)

    ax2[0].plot(fd / 1e6, np.real(z15), "k")
    ax2[0].plot(f / 1e6, np.real(z18), "r--")
    ax2[0].plot(fd / 1e6, np.real(z15), "k")

    ax2[0].set_ylabel(r"real(Z$_{50}$) [ohm]")
    ax2[0].set_xlabel("frequency [MHz]")

    ax2[1].plot(fd / 1e6, np.imag(z15), "k")
    ax2[1].plot(f / 1e6, np.imag(z18), "r--")
    ax2[1].plot(fd / 1e6, np.imag(z15), "k")

    ax2[1].set_ylabel(r"imag(Z$_{50}$) [ohm]")
    ax2[1].set_xlabel("frequency [MHz]")
    ax2[1].legend(["September 2015", "February 2018"])

    return (fig1, ax1), (fig2, ax2)


def plot_season_average_residuals(
    case,
    n_fg=3,
    ddy=1.5,
    title="No Beam Correction, Residuals to 5 LINLOG terms, 61-159 MHz",
    figure_name="no_beam_correction",
):
    def path(kind):
        return (
            config["edges_folder"]
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
    str_ar = [
        "GHA=" + str(ar[i]) + "-" + str(ar[i + 1]) + " hr" for i in range(len(ar) - 1)
    ]

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


def plot_residuals_simulated_antenna_temperature(model, title):

    if model == 1:
        path = (
            config["edges_folder"] + "mid_band/calibration/beam_factors/raw/mid_band_50"
            "-200MHz_90deg_alan1_haslam_2.5_2.62_reffreq_100MHz_{}.txt"
        )
    elif model == 2:
        path = (
            config["edges_folder"] + "mid_band/calibration/beam_factors/raw/mid_band_50"
            "-200MHz_90deg_alan1_haslam_2.5_2.5_reffreq_80MHz_{}.txt"
        )
    else:
        raise ValueError("model must be 1 or 2")

    t = np.genfromtxt(path.format("tant"))
    lst = np.genfromtxt(path.format("LST"))
    f = np.genfromtxt(path.format("freq"))

    # Converting LST to GHA
    gha = lst - 17.76
    gha[gha < 0] += 24

    indx = np.argsort(gha)
    gha = gha[indx]
    t = t[indx, :]

    t1 = np.zeros((24, len(f)))
    for i in range(24):
        tb = t[(gha >= i) & (gha <= i + 1), :]
        avtb = np.mean(tb, axis=0)
        t1[i, :] = avtb

    w = np.ones((len(t1[:, 0]), len(t[0, :])))
    fx, rx, wx = tools.spectra_to_residuals(f, t1, w, 61, 159, 5)
    index = np.arange(0, 24, 1)

    ar = np.arange(0, 25, 1)
    str_ar = [
        "GHA=" + str(ar[i]) + "-" + str(ar[i + 1]) + " hr" for i in range(len(ar) - 1)
    ]

    plot_residuals(
        fx,
        rx[index, :],
        wx[index, :],
        str_ar,
        FIG_SX=7,
        FIG_SY=12,
        DY=1.5,
        f_low=30,
        f_high=165,
        XTICKS=np.arange(60, 160 + 1, 20),
        XTEXT=32,
        YLABEL="1.5 K per division",
        TITLE=title,
        save=True,
        figure_name="simulation",
        figure_format="pdf",
    )


def level4_plot_integrated_residuals(case, f_low=60, f_high=150):
    direc = (
        config["edges_folder"]
        + "mid_band/spectra/level4/{}/binned_averages/GHA_every_1hr.txt"
    )

    folders = {
        2: "calibration_2019_10_no_ground_loss_no_beam_corrections",
        3: "case_nominal_50-150MHz_no_ground_loss_no_beam_corrections",
        5: "case_nominal_14_14_terms_55-150MHz_no_ground_loss_no_beam_corrections",
        406: "case_nominal_50-150MHz_LNA1_a2_h2_o2_s1_sim2",
        501: "case_nominal_50-150MHz_LNA2_a2_h2_o2_s1_sim2_all_lc_yes_bc",
    }

    if case not in folders:
        raise ValueError("case must be one of {}".format(folders.keys()))

    d = np.genfromtxt(direc.format(folders[case]))

    f = d[:, 0]

    def getstuff(r_full, w_full):
        ff, r = tools.spectrum_fit(
            f,
            d[:, i + 1],
            d[:, i + 1 + 48],
            n_poly=5,
            f1_low=f_low,
            f1_high=f_high,
            f2_low=f_low,
            f2_high=f_high,
        )
        if r_full is None:
            r_full = r
            w_full = d[:, i + 1 + 48]
        else:
            r_full = np.vstack((r_full, r))
            w_full = np.vstack((w_full, d[:, i + 1 + 48]))

        return r_full, w_full

    r_low, w_low, r_high, w_high = None, None, None, None
    for i in range(24):
        if (i >= 6) and (i <= 17):
            r_low, w_low = getstuff(r_low, w_low)
        if i < 6 or i > 17:
            r_high, w_high = getstuff(r_high, w_high)

    r_high1 = r_high[:6, :]
    r_high2 = r_high[6:, :]
    r_high = np.vstack((r_high2, r_high1))

    w_high1 = w_high[:6, :]
    w_high2 = w_high[6:, :]
    w_high = np.vstack((w_high2, w_high1))

    plt.figure(figsize=[13, 11])
    plt.subplot(1, 2, 1)
    for i in range(len(r_low[:, 0])):
        plt.plot(
            f[w_low[i, :] > 0], r_low[i, :][w_low[i, :] > 0] - 0.5 * i, "br"[i % 2]
        )
    plt.xlim([60, 150])
    plt.grid()
    plt.ylim([-6, 0.5])
    plt.xlabel("frequency [MHz]")
    plt.ylabel("GHA\n [0.5 K per division]")
    plt.yticks(np.arange(-5.5, 0.1, 0.5), np.arange(17, 5, -1))

    plt.subplot(1, 2, 2)
    for i in range(len(r_high[:, 0])):
        plt.plot(
            f[w_high[i, :] > 0], r_high[i, :][w_high[i, :] > 0] - 2 * i, "br"[i % 2]
        )
    plt.xlim([60, 150])
    plt.grid()
    plt.ylim([-24, 2])
    plt.xlabel("frequency [MHz]")
    plt.ylabel("GHA\n [2 K per division]")
    plt.yticks(np.arange(-22, 0.1, 2), np.arange(5, -7, -1))

    plt.savefig(
        config["edges_folder"]
        + "plots/20191218/data_residuals_no_ground_loss_no_beam_correction.pdf",
        bbox_inches="tight",
    )


def plot_level4(
    case,
    index_GHA,
    averaged_146,
    good,
    f_low,
    f_high,
    model,
    n_fg,
    K_per_division,
    file_name,
    fit_range=None,
    xsize=20,
    plot_title=False,
):
    """To use the plot_level4_second_approach, set fit_range to (FL1, FH1, FL2, FH2)."""

    path = "/home/raul/DATA2/EDGES_vol2/mid_band/spectra/level4/{folder}/{folder}.hdf5"

    folders = {
        101: "rcv18_sw18_nominal_GHA_every_1hr",
        200: "rcv18_ant19_nominal",
        201: "rcv18_ant19_every_1hr_GHA",
        1: "case_nominal_55_150MHz",
    }

    if case not in folders:
        raise ValueError("case must be one of {}".format(folders.keys()))

    folder = folders[case]
    f, py, ry, wy, index, gha, yy = io.level4read(path.format(folder=folder))

    save_rms_folder = config[
        "edges_folder"
    ] + "mid_band/spectra/level4/{}/rms_filters/".format(folder)

    if not exists(save_rms_folder):
        makedirs(save_rms_folder)

    px, rb, rx, wb, wx, yx = _average_days(averaged_146, gha, py, ry, wy, yy)

    # Identify indices of good and bad days
    kk = filters.daily_nominal_filter("mid_band", case, index_GHA, yx)

    p_all = px[kk == int(good), :]
    r_all = rx[kk == int(good), :]
    w_all = wx[kk == int(good), :]
    yd = yx[kk == int(good)]

    NS = 168

    fig, ax = plt.subplots(1, 1, figsize=[7, xsize] if good else [5, 5])
    ax.set_ylabel(
        "day of year 2018   [" + str(K_per_division) + " K per division]\n  \n  "
    )

    rms_all, fb, ii, rb, wb = _scroll_through_i(
        ax,
        f_high,
        f_low,
        K_per_division,
        NS,
        n_fg,
        f,
        index_GHA,
        model,
        p_all,
        r_all,
        rb,
        w_all,
        wb,
        yd,
        fit_range=fit_range,
    )

    if fit_range:
        for x in fit_range:
            ax.axvline(x, "y--", linewidth=1)

    ax.set_xlim([55, 151])
    ax.set_xlabel(r"$\nu$ [MHz]", fontsize=12)
    ax.set_ylim([-(ii + 1.5) * K_per_division, 1.5 * K_per_division])
    ax.yaxis.yticks([10], labels=[""])

    if plot_title:
        gha_integration = gha[2] - gha[1]

        gha1 = gha[index_GHA]
        gha2 = gha[index_GHA] + gha_integration
        if gha2 > 24:
            gha2 -= 24
        ax.set_title(
            "GHA="
            + str(gha1)
            + "-"
            + str(gha2)
            + " hr, "
            + model
            + ", "
            + str(int(n_fg))
            + " terms",
            fontsize=16,
        )

    plt.savefig("/home/raul/Desktop/" + file_name + ".pdf", bbox_inches="tight")

    return fb, rb, wb, rms_all


def _average_days(averaged_146, gha, py, ry, wy, yy):
    px = np.delete(py, 1, axis=0)
    rx = np.delete(ry, 1, axis=0)
    wx = np.delete(wy, 1, axis=0)
    yx = np.delete(yy, 1, axis=0)
    # Average the data from the two days 147
    for i in range(len(gha) - 1):
        p147 = np.mean(py[1:3, i, :], axis=0)
        r147, w147 = mdl.spectral_averaging(ry[1:3, i, :], wy[1:3, i, :])

        px[1, i, :] = p147
        rx[1, i, :] = r147
        wx[1, i, :] = w147

    if averaged_146:

        ldays = len(rx[:, 0, 0])
        lgha = len(rx[0, :, 0])

        pb = np.zeros((ldays - 1, lgha, len(px[0, 0, :])))
        rb = np.zeros((ldays - 1, lgha, len(rx[0, 0, :])))
        wb = np.zeros((ldays - 1, lgha, len(wx[0, 0, :])))

        for i in range(ldays - 1):
            for j in range(lgha):
                pa = np.array((px[0, j, :], px[i + 1, j, :]))
                ra = np.array((rx[0, j, :], rx[i + 1, j, :]))
                wa = np.array((wx[0, j, :], wx[i + 1, j, :]))

                h1 = np.mean(pa, axis=0)
                h2, h3 = tools.spectral_averaging(ra, wa)

                pb[i, j, :] = h1
                rb[i, j, :] = h2
                wb[i, j, :] = h3

        px = np.copy(pb)
        rx = np.copy(rb)
        wx = np.copy(wb)
        yx = np.delete(yx, 0, axis=0)

    return px, rb, rx, wb, wx, yx


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
        plt.savefig(
            figure_path + figure_name + "." + figure_format, bbox_inches="tight"
        )


def _scroll_through_i(
    ax,
    f_high,
    f_low,
    K_per_division,
    NS,
    n_fg,
    f,
    index_GHA,
    model,
    p_all,
    r_all,
    rb,
    w_all,
    wb,
    yd,
    fit_range=None,
):
    rms_all = np.zeros((len(yd[:, 0]), 7))
    ii = -1
    for i in range(len(yd)):
        avp = p_all[i, index_GHA, :]
        avr = r_all[i, index_GHA, :]
        avw = w_all[i, index_GHA, :]

        lw = len(avw[avw > 0])

        if lw > 0:
            ii = ii + 1
            rr, wr = rfi.cleaning_sweep(
                f, avr, avw, window_width=3, n_poly=2, n_bootstrap=20, n_sigma=3.5,
            )

            m = mdl.model_evaluate("LINLOG", avp, f / 200)
            tr = m + rr

            mask = (f >= f_low) & (f <= f_high)
            ft = f[mask]
            tt = tr[mask]
            wt = wr[mask]

            # ----------------------------------------------------------------------
            if fit_range:
                index_sel = ((ft >= fit_range[0]) & (ft <= fit_range[1])) | (
                    (ft >= fit_range[2]) & (ft <= fit_range[3])
                )

                fx = ft[index_sel]
                tx = tt[index_sel]
                wx = wt[index_sel]
            else:
                fx = ft
                tx = tt
                wx = wt

            if model == "LINLOG":
                p = mdl.fit_polynomial_fourier("LINLOG", fx / 200, tx, n_fg, Weights=wx)
                mt = mdl.model_evaluate("LINLOG", p[0], ft / 200)
                rt = tt - mt

                fb, rb, wb, sb = tools.spectral_binning_number_of_samples(
                    ft, rt, wt, nsamples=NS
                )
            elif model == "LOGLOG":
                p = np.polyfit(np.log(fx[wx > 0] / 200), np.log(tx[wx > 0]), n_fg - 1)
                log_ml = np.polyval(p, np.log(ft / 200))
                ml = np.exp(log_ml)
                rl = tt - ml

                fb, rb, wb, sb = tools.spectral_binning_number_of_samples(
                    ft, rl, wt, nsamples=NS
                )
            else:
                raise ValueError("model must be LINLOG or LOGLOG")
            if i % 2 == 0:
                ax.plot(fb[wb > 0], rb[wb > 0] - ii * K_per_division, "b")
            else:
                ax.plot(fb[wb > 0], rb[wb > 0] - ii * K_per_division, "r")

            RMS = 1000 * np.std(rb[wb > 0])
            RMS_text = str(int(RMS)) + " mK"

            ax.text(
                50, -ii * K_per_division - (1 / 6) * K_per_division, str(int(yd[i, 1]))
            )
            ax.text(153, -ii * K_per_division - (1 / 6) * K_per_division, RMS_text)

            rms_all[i, 0] = yd[i, 0]
            rms_all[i, 1] = yd[i, 1]
            rms_all[i, 2] = index_GHA
            rms_all[i, 3] = f_low
            rms_all[i, 4] = f_high
            rms_all[i, 5] = n_fg
            rms_all[i, 6] = RMS
    return rms_all, fb, ii, rb, wb


def plots_level3_rms_folder(
    band, case=1, y_top_low=10, y_top_high=50, y_bottom_low=10, y_bottom_high=30
):
    """
    This function plots the RMS of residuals of Level3 data

    YTOP_LOW/_HIGH:      y-limits of top panel
    YBOTTOM_LOW/_HIGH:   y-limits of bottom panel
    """

    # Case selection
    if case == 1:
        flag_folder = "nominal_60_160MHz"
    else:
        raise ValueError("case must be 1")

    # Listing files to be processed
    path_files = "/EDGES/spectra/level3/" + band + "/" + flag_folder + "/"
    list_files = os.listdir(path_files)
    list_files.sort()

    # Folder to save plots
    path_plots = path_files + "plots_residuals_rms/"
    if not exists(path_plots):
        makedirs(path_plots)

    # Loading data and plotting RMS
    lf = len(list_files)
    for i in range(lf):
        f, t, p, r, w, rms, m = io.level3read(path_files + list_files[i])

        plt.figure()

        plt.subplot(2, 1, 1)
        plt.plot(m[:, 3], rms[:, 0], ".")
        plt.xticks(np.arange(0, 25, 2))
        plt.grid()
        plt.xlim([0, 24])
        plt.ylim([y_top_low, y_top_high])

        plt.ylabel("RMS [K]")
        plt.title(list_files[i] + ":  Low-frequency Half")

        plt.subplot(2, 1, 2)
        plt.plot(m[:, 3], rms[:, 1], ".r")
        plt.xticks(np.arange(0, 25, 2))
        plt.grid()
        plt.xlim([0, 24])
        plt.ylim([y_bottom_low, y_bottom_high])

        plt.ylabel("RMS [K]")
        plt.xlabel("LST [Hr]")
        plt.title(list_files[i] + ":  High-frequency Half")

        if len(list_files[0]) > 12:
            file_name = list_files[i][0:11]
        else:
            file_name = list_files[i][0:8]

        plt.savefig(path_plots + file_name + ".png", bbox_inches="tight")


def level4_plot_residuals(case, gha_index, title, subfolder, figure_save_name, dy):
    filename = (
        config["edges_folder"]
        + "mid_band/spectra/level4/{}/binned_residuals/binned_residuals_one_hour_GHA.hdf5"
    )
    save_direc = os.path.join(os.path.dirname(filename), "binned_plots", subfolder)

    folders = {
        2: "calibration_2019_10_no_ground_loss_no_beam_corrections",
        3: "case_nominal_50-150MHz_no_ground_loss_no_beam_corrections",
        406: "case_nominal_50-150MHz_LNA1_a2_h2_o2_s1_sim2",
        5: "case_nominal_14_14_terms_55-150MHz_no_ground_loss_no_beam_corrections",
        501: "case_nominal_50-150MHz_LNA2_a2_h2_o2_s1_sim2_all_lc_yes_bc",
        5011: "case_nominal_50-150MHz_LNA2_a2_h2_o2_s1_sim2_all_lc_yes_bc",
    }

    if case not in folders:
        raise ValueError("case must be one of {}".format(folders.keys()))

    folder = folders[case]
    filename = filename.format(folder)
    figure_save_path_subfolder = save_direc.format(folder)

    if case == 5011:
        filename = filename.replace("GHA", "GHA_60-120MHz")

    if not exists(figure_save_path_subfolder):
        makedirs(figure_save_path_subfolder)

    fb, rb, wb, sb, gha, yd = io.level4_binned_read(filename)
    start = False
    el = []
    for i, (ri, wi, si, yi) in enumerate(zip(rb, wb, sb, yd)):
        if np.sum(wi[gha_index]) > 0:

            std_x = np.std(ri[gha_index][wi[gha_index] > 0])

            path_file = os.path.join(
                os.path.dirname(filename), f"{int(yi[0])}_{int(yi[1])}_00.hdf5"
            )

            f, t, p, r, w, rms, tp, m = io.level3read(path_file)

            gha_level3 = m[:, 4]
            gha_level3[gha_level3 < 0] += 24
            sun_el = m[(gha_level3 >= gha_index) & (gha_level3 <= (gha_index + 1)), 6]
            el.append(str(int(np.max(sun_el))) if len(sun_el) > 0 else "X")
            if not start:
                rb_new = ri[gha_index]
                wb_new = wi[gha_index]
                yd_new = yi
                std_new = np.copy(std_x)
                start = True
            else:
                rb_new = np.vstack((rb_new, ri[gha_index]))
                wb_new = np.vstack((wb_new, wi[gha_index]))
                yd_new = np.vstack((yd_new, yi))
                std_new = np.append(std_new, std_x)

    # Settings
    LST_text = [
        f"{int(yi[1])}: {eli} deg, {int(1000 * stdi)} mK"
        for yi, eli, stdi in zip(yd_new, el, std_new)
    ]

    FIG_SX = 10
    FIG_SY = 20

    YLABEL = str(dy) + " K per division"
    FIGURE_FORMAT = "pdf"

    if case == 5011:
        f_low_plot = 40
        f_high_plot = 122
        XTICKS = np.arange(60, 121, 10)
        XTEXT = 40.5

    else:
        f_low_plot = 35
        f_high_plot = 152
        XTICKS = np.arange(60, 151, 10)
        XTEXT = 35.5

    # Plotting
    plot_residuals(
        fb,
        rb_new,
        wb_new,
        LST_text,
        FIG_SX=FIG_SX,
        FIG_SY=FIG_SY,
        DY=dy,
        f_low=f_low_plot,
        f_high=f_high_plot,
        XTICKS=XTICKS,
        XTEXT=XTEXT,
        YLABEL=YLABEL,
        TITLE=title,
        save=True,
        figure_path=figure_save_path_subfolder,
        figure_name=figure_save_name,
        figure_format=FIGURE_FORMAT,
    )

    return rb_new, wb_new, yd_new


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
    beam = beams.feko_blade_beam(
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

    # Reference location
    # TODO: extract into data
    edges_lat_deg = -26.714778
    edges_lon_deg = 116.605528
    edges_location = apc.EarthLocation(
        lat=edges_lat_deg * apu.deg, lon=edges_lon_deg * apu.deg
    )

    def get_ra_dec(time_iter_utc):
        time_iter_utc_dt = dt.datetime(*time_iter_utc)
        alt_az = apc.SkyCoord(
            alt=(90 - FWHM[:, 1]) * apu.deg,
            az=FWHM[:, 0] * apu.deg,
            frame="altaz",
            obstime=apt.Time(time_iter_utc_dt, format="datetime"),
            location=edges_location,
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


def plot_data_stats():
    # Plot of histogram of GHA for integrated spectrum
    f, px, rx, wx, index, gha, ydx = io.level4read(
        "/home/raul/DATA2/EDGES_vol2/mid_band/spectra/level4/case26/case26.hdf5"
    )
    keep = filters.daily_nominal_filter("mid_band", 26, ydx)
    ydy = ydx[keep > 0]
    ix = np.arange(len(ydy))
    path_files = config["edges_folder"] + "mid_band/spectra/level3/case26/"
    new_list = listdir(path_files)
    new_list.sort()
    index_new_list = range(len(new_list))
    gha_all = np.array([])
    for i in index_new_list:
        if len(new_list[i]) <= 8:
            day = float(new_list[i][5:])
        else:
            day = float(new_list[i][5:8])

        Q = ix[ydy[:, 1] == day]

        if len(Q) > 0:
            f, ty, py, ry, wy, rmsy, tpy, my = io.level3read(path_files + new_list[i])
            ii = index[i, 0, 0 : len(my)]
            gha_i = my[ii > 0, 4]
            gha_i[gha_i < 0] = gha_i[gha_i < 0] + 24
            gha_all = np.append(gha_all, gha_i)
    sp = np.genfromtxt(
        config["edges_folder"]
        + "mid_band/spectra/level5/case2/integrated_spectrum_case2.txt"
    )
    fb = sp[:, 0]
    wb = sp[:, 2]
    fbb = fb[fb >= 60]
    wbb = wb[fb >= 60]
    fig = plt.figure(1, figsize=[3.7, 5])
    x0_top = 0.1
    y0_top = 0.57
    x0_bottom = 0.1
    y0_bottom = 0.1
    dx = 0.85
    dy = 0.35
    ax = fig.add_axes([x0_top, y0_top, dx, dy])
    ax.hist(gha_all, np.arange(6, 18.1, 1 / 6))
    plt.ylim([0, 400])
    plt.xlabel("GHA [hr]")
    plt.ylabel("number of raw spectra\nper 10-min GHA bin")
    plt.text(5.8, 345, "(a)", fontsize=14)
    ax = fig.add_axes([x0_bottom, y0_bottom, dx, dy])
    ax.step(fbb, wbb / np.max(wbb), linewidth=1)
    plt.yticks([0, 0.25, 0.5, 0.75, 1])
    plt.ylim([0, 1.25])
    plt.xlabel(r"$\nu$ [MHz]")
    plt.ylabel("normalized weights")
    plt.text(59, 1.09, "(b)", fontsize=14)
    plt.savefig(
        config["edges_folder"] + "plots/20190730/data_statistics.pdf",
        bbox_inches="tight",
    )


def plot_low_mid_comparison():
    filename = (
        "/home/raul/DATA1/EDGES_vol1/spectra/level3/low_band2_2017"
        "/EW_with_shield_nominal/2017_160_00.hdf5"
    )
    flx, tlx, wlx, ml = io.level3_read_raw_spectra(filename)
    filename = (
        "/home/raul/DATA2/EDGES_vol2/mid_band/spectra/level3/case2_75MHz/2018_150_00"
        ".hdf5"
    )
    fmx, tmx, pm, rmx, wmx, rmsm, tpm, mm = io.level3read(filename)
    f_low = 60
    f_high = 100
    fl = flx[(flx >= f_low) & (flx <= f_high)]
    tl = tlx[:, (flx >= f_low) & (flx <= f_high)]
    tm = tmx[:, (fmx >= f_low) & (fmx <= f_high)]
    wm = wmx[:, (fmx >= f_low) & (fmx <= f_high)]

    il = [1597, 1787, 1976, 2166, 77, 267, 547, 647, 837, 1027, 1217, 1407]
    im = [1610, 1793, 1977, 2161, 138, 322, 505, 689, 873, 1057, 1241, 1425]
    plt.figure(figsize=[4, 6])
    gg = "c"
    dy = 500  # K
    lw = 1
    k = 0.1
    for i, ili, imi in enumerate(zip(il, im)):
        tli = tl[ili, :]

        tmi = tm[imi, :]
        wmi = wm[imi, :]

        plt.plot(fl[wmi > 0], (tmi - tli)[wmi > 0] - i * dy, color=gg, linewidth=lw)
        plt.plot([50, 150], [-i * dy, -i * dy], "k")
        plt.text(53.6 if i <= 4 else 52.3, -i * dy - k * dy, str(2 * i) + " hr")

    plt.yticks([])
    plt.ylabel("GHA [" + str(dy) + " K per division]\n\n\n")
    plt.xlabel(r"$\nu$ [MHz]")
    plt.xlim([58, 102])
    plt.ylim([-12 * dy, dy])
    # Saving
    plt.savefig(
        config["edges_folder"] + "/plots/20190612/comparison_mid_low2.pdf",
        bbox_inches="tight",
    )


def plot_beam_power(max_norm=True):
    def get_sm(beam_file, low_band=False):
        if not low_band:
            bm_all = beams.feko_blade_beam(
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
        sm, f, "-", "Mid-Band 30mx30m ground plane",
    )
    plot_ax(
        smi, f, ":", "Mid-Band infinite ground plane",
    )
    plot_ax(
        sl, fl, "--", "Low-Band 30mx30m ground plane",
    )

    if max_norm:
        ax[0].set_ylabel("solid angle of\n beam above horizon [sr]")
    else:
        ax[0].set_ylabel(
            "normalized total radiated power\n above horizon [fraction of 4pi]"
        )
    ax[0].legend()

    ax[1].set_ylabel("residuals to\n 5-term polynomial [sr]")
    ax[1].set_xlabel("frequency [MHz]")
    return fig, ax


def beam_chromaticity_differences():
    # Plot
    size_x = 4.7
    size_y = 5
    x0 = 0.13
    y0 = 0.035
    dx = 0.53
    dy = 0.55
    xoff = 0.09
    dxc = 0.03
    filenames = [
        (
            "table_lores_mid_band_50-200MHz_90deg_alan0_haslam_gaussian_index_2.4_2.65_sigma_deg_8"
            ".5_reffreq_90MHz.hdf5"
        ),
        (
            "table_lores_mid_band_50-200MHz_90deg_alan1_haslam_gaussian_index_2.4_2.65_sigma_deg_8"
            ".5_reffreq_90MHz.hdf5"
        ),
        (
            "table_lores_mid_band_50-200MHz_90deg_alan0_haslam_flat_index_2"
            ".56_reffreq_90MHz.hdf5"
        ),
        (
            "table_lores_mid_band_50-200MHz_90deg_alan0_LW_gaussian_index_2"
            ".4_2.65_sigma_deg_8.5_reffreq_90MHz.hdf5"
        ),
        (
            "table_lores_mid_band_50-200MHz_90deg_alan0_guzman_gaussian_index_2.4_2.65_sigma_deg_8"
            ".5_reffreq_90MHz.hdf5"
        ),
    ]
    folder = "/home/raul/DATA2/EDGES_vol2/mid_band/calibration/beam_factors/table/"

    bx = []
    for fname in filenames:
        f, lst, bf = beams.beam_factor_table_read(folder + fname)
        gha = lst + (24 - 17.76)
        gha[gha >= 24] -= 24
        IX = np.argsort(gha)
        bx.append(bf[IX])

    fig, ax = plt.subplots(1, 4, figsize=(size_x, size_y))
    scale_max = 0.0043
    scale_min = -0.0043

    cmap = plt.cm.viridis
    rgba = cmap(0.0)
    cmap.set_under(rgba)

    for i in range(4):
        im = ax[i].imshow(
            bx[i + 1] - bx[0],
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

    path_plot_save = config["edges_folder"] + "/plots/20190729/"
    plt.savefig(
        path_plot_save + "beam_chromaticity_differences.pdf", bbox_inches="tight"
    )


def plot_beam_chromaticity_correction():
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
    beam_factor_filename = (
        "table_lores_mid_band_50-200MHz_90deg_alan0_haslam_gaussian_index_2.4_2.65_sigma_deg_8"
        ".5_reffreq_90MHz.hdf5"
    )
    f, lst, bf = beams.beam_factor_table_read(
        "/home/raul/DATA2/EDGES_vol2/mid_band/calibration/beam_factors/table/"
        + beam_factor_filename
    )
    gha = lst + (24 - 17.76)
    gha[gha >= 24] = gha[gha >= 24] - 24
    IX = np.argsort(gha)
    bx1 = bf[IX]

    f1 = plt.figure(num=1, figsize=(size_x, size_y))
    ax = f1.add_axes([x0, y0 + 1 * (yoff + dy1), dx, dy])
    im = ax.imshow(
        bx1,
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
    ax.plot(f, bx1[125, :], "k")
    ax.plot(f, bx1[175, :], "k--")
    ax.legend(["GHA=10 hr", "GHA=14 hr"], fontsize=8, ncol=2)
    ax.set_ylim([0.9, 1.1])
    ax.set_xlim([60, 120])
    ax.set_ylim([0.975, 1.025])
    ax.set_xlabel(r"$\nu$ [MHz]")
    ax.set_yticks([0.98, 1, 1.02])
    ax.set_ylabel("$C$")
    # Saving plot
    path_plot_save = config["edges_folder"] + "plots/20190729/"
    plt.savefig(
        path_plot_save + "beam_chromaticity_correction.pdf", bbox_inches="tight"
    )


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
    return Gg, Gglb, flb


def plot_beam_gain(path_plot_save):
    fmin = 50
    fmax = 120
    fmin_res = 50
    fmax_res = 120
    n_fg = 5
    el = np.arange(0, 91)
    sin_theta = np.sin((90 - el) * (np.pi / 180))
    sin_theta_2D_T = np.tile(sin_theta, (360, 1))
    sin_theta_2D = sin_theta_2D_T.T

    def get_beam(beam_file, low_band=False):
        if not low_band:
            b_all = beams.feko_blade_beam(
                "mid_band", beam_file, frequency_interpolation=False, az_antenna_axis=90
            )
            f = np.arange(50, 201, 2)
        else:
            b_all = beams.FEKO_low_band_blade_beam(
                beam_file=beam_file,
                frequency_interpolation=False,
                AZ_antenna_axis=0,
                frequency=np.array([0]) if beam_file == 5 else None,
            )
            f = np.arange(40, 121, 2)

        bint = np.zeros(len(f))
        for i in range(len(f)):
            b = b_all[i, :, :]
            bint[i] = np.sum(b * sin_theta_2D)
        fr = f[(f >= fmin) & (f <= fmax)]
        bx = bint[(f >= fmin) & (f <= fmax)]
        b = bx / np.mean(bx)
        ff = fr[(fr >= fmin_res) & (fr <= fmax_res)]
        bb = b[(fr >= fmin_res) & (fr <= fmax_res)]
        x = np.polyfit(ff, bb, n_fg - 1)
        m = np.polyval(x, ff)
        r = bb - m
        return b, r, fr, ff

    b1, r1, fr1, ff1 = get_beam(0)
    b2, r2, fr2, ff2 = get_beam(1)
    b3, r3, fr3, ff3 = get_beam(2, True)
    b4, r4, fr4, ff4 = get_beam(5, True)

    f1, axs = plt.subplots(1, 2)

    ax = axs[0]
    ax.plot(fr1, b1, "b", linewidth=1.0)
    ax.plot(fr2, b2, "b--", linewidth=0.5)
    ax.plot(fr3, b3, "r", linewidth=1.0)
    ax.plot(fr4, b4, "r--", linewidth=0.5)
    ax.set_ylabel("normalized integrated gain")
    ax.set_xlim([48, 132])
    xt = np.arange(50, 131, 10)
    ax.set_xticks(xt)
    ax.tick_params(axis="x", direction="in")
    ax.set_xticklabels(["" for i in range(len(xt))])
    ax.set_ylim([0.980, 1.015])
    ax.set_yticks(np.arange(0.985, 1.011, 0.005))

    ax = axs[1]
    ax.plot(ff1, r1, "b", linewidth=1.0)
    ax.plot(ff2, r2 - 0.0005, "b--", linewidth=5)
    ax.plot(ff3, r3 - 0.001, "r", linewidth=1.0)
    ax.plot(ff4, r4 - 0.0015, "r--", linewidth=0.5)
    ax.set_ylim([-0.002, 0.0005])
    ax.set_xlabel("$\\nu$ [MHz]")
    ax.set_ylabel("normalized integrated gain residuals\n [0.05% per division]")
    ax.set_xlim([48, 132])
    ax.set_xticks(np.arange(50, 131, 10))
    ax.set_yticklabels("")
    plt.savefig(path_plot_save + "beam_gain.pdf", bbox_inches="tight")


def plot_antenna_beam():
    path_plot_save = config["edges_folder"] + "plots/20200108/"

    size_x = 8.2
    size_y = 6
    f_low = 50
    f_high = 120

    def get_beam(beam_file, low_band=False):
        if not low_band:
            bm = beams.feko_blade_beam(
                "mid_band",
                beam_file,
                frequency_interpolation=False,
                frequency=np.array([0]),
                az_antenna_axis=0,
            )
            ff = np.arange(50, 201, 2)
        else:
            bm = beams.FEKO_low_band_blade_beam(
                beam_file=beam_file,
                frequency_interpolation=False,
                frequency=np.array([0]),
                AZ_antenna_axis=0,
            )
            ff = np.arange(40, 121, 2)

        mask = (ff >= f_low) & (ff <= f_high)
        fe = ff[mask]
        g_zenith = bm[mask, 90, 0]
        g_45_E = bm[mask, 45, 0]
        g_45_H = bm[mask, 45, 90]
        return fe, (g_zenith, g_45_E, g_45_H)

    f1, ax = plt.subplots(1, 4, figsize=(size_x, size_y))

    labels = [
        r"Mid-Band 30m x 30m GP",
        "Mid-Band infinite GP",
        r"Low-Band 30m x 30m GP",
        r"Low-Band 10m x 10m GP",
    ]

    n_fg = 4
    DY = 0.08

    for j, ((beam_file, low_band), label) in enumerate(
        zip([(0, False), (1, False), (2, True), (5, True)], labels)
    ):
        f, g = get_beam(beam_file, low_band)

        # Save fe for a later plot.
        if j == 0:
            fe = f

        for i, gg in enumerate(g):
            p = np.polyfit(f, gg, n_fg)
            m = np.polyval(p, f)

            ax[0].plot(f, gg, color=f"C{j}", label=label if not i else None)

            add = DY if not j else (-DY if j == 1 else 0)
            ax[1].plot(f, gg - m + add, color=f"C{j}")

    ax[0].set_xticklabels("")
    ax[0].set_ylabel("gain")
    ax[0].set_xlim([48, 122])
    ax[0].tick_params(axis="x", direction="in")
    ax[0].set_xticks(np.arange(50, 121, 10))
    ax[0].set_ylim([0, 9])
    ax[0].set_yticks(np.arange(1, 8.1, 1))
    ax[0].text(115, 0.4, "(a)", fontsize=14)
    ax[0].text(50, 5.7, "zenith", fontsize=10)
    ax[0].text(50, 2.9, r"$\theta=45^{\circ}$, H-plane", fontsize=10)
    ax[0].text(50, 0.9, r"$\theta=45^{\circ}$, E-plane", fontsize=10)
    ax.legend(fontsize=7, ncol=2)

    ax[1].set_xlabel(r"$\nu$ [MHz]")  # , fontsize=15)
    ax[1].set_ylabel("gain residuals\n[" + str(DY / 2) + " per division]")
    ax[1].set_xlim([48, 122])
    xt = np.arange(50, 121, 10)
    ax[1].set_xticks(xt)
    yt = np.arange(-1.5 * DY, 1.5 * DY + 0.0001, DY / 2)
    ax[1].set_ylim([-0.135, 0.135])
    ax[1].set_yticks(yt)
    ax[1].set_yticklabels([""] * len(yt))
    ax[1].text(115, -0.123, "(b)", fontsize=14)
    ax[1].text(50, 0.035, "zenith", fontsize=10)
    ax[1].text(50, -0.03, r"$\theta=45^{\circ}$, H-plane", fontsize=10)
    ax[1].text(50, -0.12, r"$\theta=45^{\circ}$, E-plane", fontsize=10)

    Gg, Gglb, flb = plot_ground_loss(ax[2])

    x1 = (1 - Gg) * 100
    x2 = (1 - Gglb) * 100
    p1 = np.polyfit(fe, x1, n_fg)
    p2flb = np.polyfit(flb, x2, n_fg)
    m1 = np.polyval(p1, fe)
    m2 = np.polyval(p2flb)
    ax[3].plot(fe, x1 - m1, "b", linewidth=1)
    ax[3].plot(flb, x2 - m2, "r", linewidth=1)
    ax[3].set_xticks(xt)
    ax[3].set_ylim([-0.006, 0.006])
    ax[3].set_yticks(np.arange(-0.004, 0.0041, 0.002))
    ax[3].set_ylabel(r"ground loss residuals [%]")
    ax[3].set_xlabel(r"$\nu$ [MHz]")
    ax[3].text(115, -0.0055, "(d)", fontsize=14)
    plt.savefig(path_plot_save + "beam_gain.pdf", bbox_inches="tight")


def _get_xf(fe, path, f_high=np.inf):
    x = np.genfromtxt(
        "/run/media/raul/SSD_4TB/EDGES_vol1/calibration/antenna_s11/low_band1/s11/corrected"
        + path
    )
    f = x[:, 0] / 1e6
    ra = x[:, 1] + 1j * x[:, 2]
    return f[f <= f_high], ra[f <= f_high]


def plot_balun_loss2(s11_path):
    # Paths
    path_plot_save = config["edges_folder"] + "plots/20190917/"
    # Plot
    size_x = 4.5
    size_y = 2.7

    fe, flb1, flb2, ra, ralb1, ralb2 = _get_ra(s11_path)

    Gb, Gc = loss.balun_and_connector_loss("mid_band", fe, ra)
    Gbc = Gb * Gc
    Gblb, Gclb = loss.balun_and_connector_loss("low_band_2015", flb1, ralb1)
    Gbclb = Gblb * Gclb
    Gblb2, Gclb2 = loss.balun_and_connector_loss("low_band2_2017", flb2, ralb2)
    Gbclb2 = Gblb2 * Gclb2

    fig, ax = plt.subplots(1, 1, figsize=(size_x, size_y))
    ax.plot(fe, (1 - Gbc) * 100, "b", linewidth=1.3)
    ax.plot(flb1, (1 - Gbclb) * 100, "r", linewidth=1.3)
    ax.plot(flb2, (1 - Gbclb2) * 100, "r--", linewidth=1.3)

    ax.set_ylabel(r"antenna loss [%]")
    ax.set_xlim([48, 132])
    ax.tick_params(axis="x", direction="in")
    ax.set_xticks(np.arange(50, 131, 10))
    ax.set_ylim([0, 1])
    ax.set_yticks(np.arange(0.2, 0.9, 0.2))
    ax.set_xlabel("$\\nu$ [MHz]", fontsize=13)
    ax.legend(["Mid-Band", "Low-Band 1", "Low-Band 2"], fontsize=9)
    plt.savefig(path_plot_save + "balun_loss.pdf", bbox_inches="tight")


def plot_antenna_calibration_params(s11_path):
    # Paths
    path_plot_save = config["edges_folder"] + "plots/20190917/"
    # Plot
    fs_labels = 12
    size_x = 4.5
    size_y = 5.5  # 10.5

    fe, flb1, flb2, ra, ralb1, ralb2 = _get_ra(s11_path)

    fig, ax = plt.subplots(2, 1, figsize=(size_x, size_y))
    ax[0].plot(fe, 20 * np.log10(np.abs(ra)), "b", linewidth=1.3, label="Mid-Band")
    ax[0].plot(
        flb1, 20 * np.log10(np.abs(ralb1)), "r", linewidth=1.3, label="Low-Band 1"
    )
    ax[0].plot(
        flb2, 20 * np.log10(np.abs(ralb2)), "r--", linewidth=1.3, label="Low-Band 2"
    )

    ax[0].legend(fontsize=9)
    ax[0].set_xticklabels("")
    ax[0].set_ylabel(r"$|\Gamma_{\mathrm{ant}}|$ [dB]", fontsize=fs_labels)
    ax[0].set_xlim([48, 132])
    ax[0].set_ylim([-17, -1])
    ax[0].set_yticks(np.arange(-16, -1, 2))
    ax[0].tick_params(axis="x", direction="in")
    ax[0].set_xticks(np.arange(50, 131, 10))
    ax[0].text(122, -15.6, "(a)", fontsize=14)

    ax[1].plot(fe, (180 / np.pi) * np.unwrap(np.angle(ra)), "b", lw=1.3)
    ax[1].plot(flb1, (180 / np.pi) * np.unwrap(np.angle(ralb1)), "r", lw=1.3)
    ax[1].plot(flb2, (180 / np.pi) * np.unwrap(np.angle(ralb2)), "r--", lw=1.3)

    ax[1].set_ylabel(
        r"$\angle\/\Gamma_{\mathrm{ant}}$ [ $^\mathrm{o}$]", fontsize=fs_labels
    )
    ax[1].set_xlim([48, 132])
    ax[1].tick_params(axis="x", direction="in")
    ax[1].set_xticks(np.arange(50, 131, 10))
    ax[1].set_ylim([-700, 300])
    ax[1].set_yticks(np.arange(-600, 201, 200))
    ax[1].text(122, -620, "(b)", fontsize=14)
    ax[1].set_xlabel(r"$\nu$ [MHz]", fontsize=13)
    plt.savefig(
        path_plot_save + "antenna_reflection_coefficients.pdf", bbox_inches="tight"
    )


def _get_ra(s11_path):
    fe = EdgesFrequencyRange(f_low=50, f_high=130).freq
    ra = s11m.antenna_s11_remove_delay(s11_path, fe, delay_0=0.17, n_fit=15)
    flb1, ralb1 = _get_xf(fe, "/2016_243/S11_blade_low_band_2016_243.txt", f_high=100)
    flb2, ralb2 = _get_xf(
        fe,
        "/2017-06-29-low2-noshield_average/S11_blade_low_band_2017_180_NO_SHIELD.txt",
        f_high=100,
    )
    return fe, flb1, flb2, ra, ralb1, ralb2


def plot_balun_loss(s11_path):

    # Paths
    path_plot_save = config["edges_folder"] + "plots/20200108/"

    # Plot
    size_x = 4.5
    size_y = 2.7  # 10.5
    plt.figure(num=1, figsize=(size_x, size_y))

    fe, flb1, flb2, ra, ralb1, ralb2 = _get_ra(s11_path)

    Gb, Gc = loss.balun_and_connector_loss("mid_band", fe, ra)
    Gbc = Gb * Gc

    Gblb, Gclb = loss.balun_and_connector_loss("low_band_2015", flb1, ralb1)
    Gbclb = Gblb * Gclb

    Gblb2, Gclb2 = loss.balun_and_connector_loss("low_band2_2017", flb2, ralb2)
    Gbclb2 = Gblb2 * Gclb2

    # Figure
    fig, ax = plt.subplots(2, 1, figsize=[6, 7])

    ax[0].plot(fe, (1 - Gbc) * 100, "b", linewidth=1.3, label="Mid-Band")
    ax[0].plot(flb1, (1 - Gbclb) * 100, "r", linewidth=1.3, label="Low-Band 1")
    ax[0].plot(flb1, (1 - Gbclb2) * 100, "r--", linewidth=1.3, label="Low-Band 2")
    ax[0].legend(fontsize=9)

    ax[0].set_xlim([48, 132])
    ax[0].xaxis.xticks(np.arange(50, 131, 10), "")

    ax[0].set_ylim([0, 1])
    ax[0].yaxis.yticks(np.arange(0.0, 0.9, 0.2))
    ax[0].set_ylabel(r"balun loss [%]")

    ax[0].text(48.5, 0.05, "(a)", fontsize=15)

    # Subplot 2
    Ga = loss.antenna_loss("mid_band", fe)
    ax[1].plot(fe, (1 - Ga) * 100, "b", linewidth=1.3)

    ax[1].set_xlim([48, 132])
    ax[1].xaxis.xticks(np.arange(50, 131, 10))
    ax[1].set_xlabel(r"$\nu$ [MHz]", fontsize=13)

    ax[1].set_ylim([0, 0.12])
    ax[1].yaxis.yticks(np.arange(0.0, 0.11, 0.02))
    ax[1].set_ylabel(r"antenna loss [%]")

    ax[1].text(48.5, 0.007, "(b)", fontsize=15)

    plt.savefig(path_plot_save + "loss.pdf", bbox_inches="tight")


def plot_vna_comparison(folders, labels, repeat_num=None):
    assert len(folders) == len(labels)

    vna = {}
    for label, folder in zip(folders, labels):
        fls = listdir(folder)
        vna[label] = {}

        for standard in ["open", "short", "match", "3db", "6db", "10db", "15db"]:
            find_standard = standard
            if repeat_num is not None:
                find_standard += f"0{repeat_num}"

            fl = [fl for fl in fls if find_standard in fl.lower()][0]

            vna[label][standard], f = S1P.read(fl)

    o_a, s_a, m_a = rc.agilent_85033E(f, 50, m=1, md_value_ps=38)

    for label, standards in vna.items():
        for standard, s11 in standards.items():
            if standard.endswith("db"):
                vna[label][standard + "_corrected"] = rc.de_embed(
                    o_a,
                    s_a,
                    m_a,
                    standards["open"],
                    standards["short"],
                    standards["match"],
                    standards[standard],
                )

    fig, ax = plt.subplots(len(folders), 2, sharex=True)

    def angle(x):
        return (180 / np.pi) * np.unwrap(np.angle(x))

    for i, (label, standards) in enumerate(vna.items()):
        for standard, s11 in standards.items():
            if not standard.endswith("corrected"):
                continue

            for j, fnc in enumerate(
                (
                    lambda x: 20 * np.log10(np.abs(x)),
                    lambda x: angle(x) - angle(vna[labels[0]][standard]),
                )
            ):
                ax[i, j].plot(f, fnc(s11), label=label)
                ax[i, j].set_ylabel(f"{standard} Attn [{'degrees' if j else 'dB'}]")
                ax[i, j].set_title(r"$\Delta$ PHASE" if j else "MAGNITUDE")

        ax[i, -1].set_xlabel("frequency [MHz]")
    ax[0, 0].legend()


def plot_calibration_term_sweep(fname):
    rms, cterms, wterms = io.calibration_rms_read(fname)

    figs = [plt.subplots(1, 1) for i in range(len(rms))]

    for r, (fig, ax) in zip(rms, figs):
        ax.imshow(np.flipud(r), interpolation="none", extent=[1, 15, 1, 15])
        plt.colorbar()

    return figs


def plot_beam_factor(
    az_above_horizon,
    edges_lat_deg,
    el_above_horizon,
    irf,
    lst,
    path_plots,
    plot_format,
    sky_map,
    sky_ref_above_horizon,
):
    LAT_DEG = np.copy(edges_lat_deg)
    AZ_plot = np.copy(az_above_horizon)
    AZ_plot[AZ_plot > 180] -= 360
    EL_plot = np.copy(el_above_horizon)
    SKY_plot = np.copy(sky_ref_above_horizon)
    max_log10sky = np.max(np.log10(sky_map[:, irf]))
    min_log10sky = np.min(np.log10(sky_map[:, irf]))
    marker_size = 10
    LST_gc = 17 + (45 / 60) + (40.04 / (60 * 60))  # LST of Galactic Center
    GHA = lst - LST_gc
    if GHA < 0:
        GHA += 24

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
        plt.title(
            f"LAT={LAT_DEG:.3f} [deg] \n\n LST={lst:.3f} hr        GHA={GHA:.3f} hr"
        )
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
            45 * (np.pi / 180), 143, "Raul Monsalve", fontsize=8, color=[0.5, 0.5, 0.5],
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
        path_plots + f"LST_{lst:.3f} hr.png", bbox_inches="tight",
    )
