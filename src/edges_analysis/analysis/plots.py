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
from edges_cal import EdgesFrequencyRange
from edges_cal import modelling as mdl
from edges_cal import reflection_coefficient as rc
from edges_io.io import S1P
from scipy import interpolate as interp

from . import beams, filters, io, loss, rfi
from . import s11 as s11m
from . import sky_models, tools

edges_folder = ""  # TODO: rmeove


def plot_daily_residuals_nominal(f, rx, wx, ydx):
    keep_index = filters.daily_nominal_filter("mid_band", 1, ydx)
    r = rx[keep_index == 1]
    w = wx[keep_index == 1]
    yd = ydx[keep_index == 1]

    lr = len(r[:, 0])
    for i in range(lr):
        fb, rb, wb = tools.spectral_binning_number_of_samples(f, r[i, :], w[i, :])
        if i == 0:
            rb_all = np.zeros((lr, len(fb)))
            wb_all = np.zeros((lr, len(fb)))

        rb_all[i, :] = rb
        wb_all[i, :] = wb

    # Settings
    # ----------------------------------

    LST_text = [
        str(int(yd[i, 1])) for i in range(lr)
    ]  # 'GHA=0-5 hr', 'GHA=5-11 hr', 'GHA=11-18 hr', 'GHA=18-24 hr']
    DY = 0.7

    FLOW_plot = 53
    FHIGH_plot = 122

    XTICKS = np.arange(60, 121, 20)
    XTEXT = 54
    YLABEL = str(DY) + " K per division"
    TITLE = ""  # '59-121 MHz, GHA=6-18 hr'

    figure_path = "/home/raul/Desktop/"
    figure_name = "daily_residuals_nominal_59_121MHz_GHA_6_18hr"

    FIG_SX = 6
    FIG_SY = 15

    # Plotting
    tools.plot_residuals(
        fb,
        rb_all,
        wb_all,
        LST_text,
        FIG_SX=FIG_SX,
        FIG_SY=FIG_SY,
        DY=DY,
        FLOW=FLOW_plot,
        FHIGH=FHIGH_plot,
        XTICKS=XTICKS,
        XTEXT=XTEXT,
        YLABEL=YLABEL,
        TITLE=TITLE,
        save="yes",
        figure_path=figure_path,
        figure_name=figure_name,
    )


def plots_midband_metadata():
    list_files = os.listdir(edges_folder + "/mid_band/spectra/level3/case_nominal/")
    list_files.sort()

    plt.close()
    plt.close()

    # Processing files
    for i in range(len(list_files)):

        day = list_files[i][0:11]

        f, t, p, r, w, rms, tp, m = io.level3read(
            edges_folder + "/mid_band/spectra/level3/case_nominal/" + list_files[i]
        )

        gha = m[:, 4]
        gha[gha < 0] = gha[gha < 0] + 24

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

        plt.close()
        plt.close()

        plt.figure(figsize=[4.5, 9])
        plt.subplot(5, 1, 1)
        plt.plot(gha, "b", linewidth=2)
        if index_gha_6 > -1:
            plt.plot([index_gha_6, index_gha_6], [-1000, 1000], "g--", linewidth=2)
            # plt.text(index_gha_6, 12, 'GHA=6hr', rotation=90, color='g', fontsize=12)
            plt.text(index_gha_6, 12, " 6hr", rotation=0, color="g", fontsize=10)
        if index_gha_18 > -1:
            plt.plot([index_gha_18, index_gha_18], [-1000, 1000], "r--", linewidth=2)
            # plt.text(index_gha_18, 14, 'GHA=18hr', rotation=90, color='r', fontsize=12)
            plt.text(index_gha_18, 12, " 18hr", rotation=0, color="r", fontsize=10)
        plt.ylim([-1, 26])
        plt.yticks([0, 6, 12, 18, 24])
        plt.ylabel("GHA [hr]")
        plt.grid()
        plt.title(day)
        # plt.legend(['GHA = 6 hr','GHA = 18 hr'], loc=0)

        plt.subplot(5, 1, 2)
        plt.plot(sun_el, "b", linewidth=2)
        if index_gha_6 > -1:
            plt.plot([index_gha_6, index_gha_6], [-1000, 1000], "g--", linewidth=2)

        if index_gha_18 > -1:
            plt.plot([index_gha_18, index_gha_18], [-1000, 1000], "r--", linewidth=2)
        plt.ylim([-110, 110])
        plt.yticks([-90, -45, 0, 45, 90])
        plt.ylabel("sun elev [deg]")
        plt.grid()

        plt.subplot(5, 1, 3)
        plt.plot(temp, "b", linewidth=2)
        if index_gha_6 > -1:
            plt.plot([index_gha_6, index_gha_6], [-1000, 1000], "g--", linewidth=2)

        if index_gha_18 > -1:
            plt.plot([index_gha_18, index_gha_18], [-1000, 1000], "r--", linewidth=2)
        plt.ylim([0, 40])
        plt.yticks([10, 20, 30])
        plt.ylabel(r"amb temp [$^{\circ}$C]")
        plt.grid()

        plt.subplot(5, 1, 4)
        plt.plot(hum, "b", linewidth=2)
        if index_gha_6 > -1:
            plt.plot([index_gha_6, index_gha_6], [-1000, 1000], "g--", linewidth=2)

        if index_gha_18 > -1:
            plt.plot([index_gha_18, index_gha_18], [-1000, 1000], "r--", linewidth=2)
        plt.ylim([-30, 110])
        plt.yticks([-20, 0, 20, 40, 60, 80, 100])
        plt.ylabel("amb humid [%]")
        plt.grid()

        plt.subplot(5, 1, 5)
        plt.plot(rec_temp, "b", linewidth=2)
        if index_gha_6 > -1:
            plt.plot([index_gha_6, index_gha_6], [-1000, 1000], "g--", linewidth=2)

        if index_gha_18 > -1:
            plt.plot([index_gha_18, index_gha_18], [-1000, 1000], "r--", linewidth=2)

        plt.ylim([23, 27])
        plt.yticks([24, 25, 26])
        plt.ylabel(r"rec temp [$^{\circ}$C]")
        plt.grid()

        plt.xlabel("time [number of raw spectra since start of file]")

        plt.savefig(
            edges_folder
            + "/mid_band/spectra/level3/case_nominal/metadata/"
            + day
            + ".png",
            bbox_inches="tight",
        )
        plt.close()
        plt.close()


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
    # ant_s11 = np.ones(len(f))
    #
    # o1 = cr1.switch_correction_receiver1(ant_s11, f_in=f, case=1)
    # o2 = cr1.switch_correction_receiver1(ant_s11, f_in=f, case=2)
    #
    # o10 = cr1.switch_correction_receiver1(ant_s11, f_in=f, case=10)
    # o11 = cr1.switch_correction_receiver1(ant_s11, f_in=f, case=11)
    # o12 = cr1.switch_correction_receiver1(ant_s11, f_in=f, case=12)
    # o13 = cr1.switch_correction_receiver1(ant_s11, f_in=f, case=13)

    # fl = np.arange(50, 101)
    # al = np.ones(len(fl))
    # ol = oeg.low_band_switch_correction(al, 25, f_in=fl)

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

    # Measurements at 25degC
    o_sw_m15, fd = S1P.read(path15 + "open.S1P")
    s_sw_m15, fd = S1P.read(path15 + "short.S1P")
    l_sw_m15, fd = S1P.read(path15 + "load.S1P")

    o_sw_in15, fd = S1P.read(path15 + "open_input.S1P")
    s_sw_in15, fd = S1P.read(path15 + "short_input.S1P")
    l_sw_in15, fd = S1P.read(path15 + "load_input.S1P")

    # Standards assumed at the switch
    o_sw = 1 * np.ones(len(fd))
    s_sw = -1 * np.ones(len(fd))
    l_sw = 0 * np.ones(len(fd))

    # Correction at the switch -- 25degC
    om15, xx1, xx2, xx3 = rc.de_embed(
        o_sw, s_sw, l_sw, o_sw_m15, s_sw_m15, l_sw_m15, o_sw_in15
    )
    sm15, xx1, xx2, xx3 = rc.de_embed(
        o_sw, s_sw, l_sw, o_sw_m15, s_sw_m15, l_sw_m15, s_sw_in15
    )
    lm15, xx1, xx2, xx3 = rc.de_embed(
        o_sw, s_sw, l_sw, o_sw_m15, s_sw_m15, l_sw_m15, l_sw_in15
    )

    # Loading measurements
    path18 = (
        "/home/raul/DATA/EDGES/mid_band/calibration/receiver_calibration/receiver1"
        "/2018_01_25C/data/s11/raw/InternalSwitch/"
    )

    o_sw_m18, f = S1P.read(path18 + "Open01.s1p")
    s_sw_m18, f = S1P.read(path18 + "Short01.s1p")
    l_sw_m18, f = S1P.read(path18 + "Match01.s1p")

    o_sw_in18, f = S1P.read(path18 + "ExternalOpen01.s1p")
    s_sw_in18, f = S1P.read(path18 + "ExternalShort01.s1p")
    l_sw_in18, f = S1P.read(path18 + "ExternalMatch01.s1p")

    # Standards assumed at the switch
    o_sw = 1 * np.ones(len(f))
    s_sw = -1 * np.ones(len(f))
    l_sw = 0 * np.ones(len(f))

    # Correction at the switch
    om18, xx1, xx2, xx3 = rc.de_embed(
        o_sw, s_sw, l_sw, o_sw_m18, s_sw_m18, l_sw_m18, o_sw_in18
    )
    sm18, xx1, xx2, xx3 = rc.de_embed(
        o_sw, s_sw, l_sw, o_sw_m18, s_sw_m18, l_sw_m18, s_sw_in18
    )
    lm18, xx1, xx2, xx3 = rc.de_embed(
        o_sw, s_sw, l_sw, o_sw_m18, s_sw_m18, l_sw_m18, l_sw_in18
    )

    # Plot

    plt.figure(1)
    plt.subplot(2, 3, 1)
    plt.plot(fd / 1e6, 20 * np.log10(np.abs(om15)), "k")
    plt.plot(f / 1e6, 20 * np.log10(np.abs(om18)), "r--")
    plt.plot(fd / 1e6, 20 * np.log10(np.abs(om15)), "k")
    plt.ylabel("magnitude [dB]")
    plt.title("Open Standard at the Receiver Input\n(Measured from the Switch)")

    plt.subplot(2, 3, 4)
    plt.plot(fd / 1e6, (180 / np.pi) * np.unwrap(np.angle(om15)), "k")
    plt.plot(f / 1e6, (180 / np.pi) * np.unwrap(np.angle(om18)), "r--")
    plt.plot(fd / 1e6, (180 / np.pi) * np.unwrap(np.angle(om15)), "k")
    plt.ylabel("phase [deg]")
    plt.xlabel("frequency [MHz]")

    plt.subplot(2, 3, 2)
    plt.plot(fd / 1e6, 20 * np.log10(np.abs(sm15)), "k")
    plt.plot(f / 1e6, 20 * np.log10(np.abs(sm18)), "r--")
    plt.plot(fd / 1e6, 20 * np.log10(np.abs(sm15)), "k")
    plt.title("Short Standard at the Receiver Input\n(Measured from the Switch)")

    plt.subplot(2, 3, 5)
    plt.plot(fd / 1e6, (180 / np.pi) * np.unwrap(np.angle(sm15)), "k")
    plt.plot(f / 1e6, (180 / np.pi) * np.unwrap(np.angle(sm18)), "r--")
    plt.plot(fd / 1e6, (180 / np.pi) * np.unwrap(np.angle(sm15)), "k")
    plt.xlabel("frequency [MHz]")

    plt.subplot(2, 3, 3)
    plt.plot(fd / 1e6, 20 * np.log10(np.abs(lm15)), "k")
    plt.plot(f / 1e6, 20 * np.log10(np.abs(lm18)), "r--")
    plt.plot(fd / 1e6, 20 * np.log10(np.abs(lm15)), "k")
    plt.title("50-ohm Load Standard at the Receiver Input\n(Measured from the Switch)")
    plt.legend(["September 2015", "February 2018"])

    plt.subplot(2, 3, 6)
    plt.plot(fd / 1e6, (180 / np.pi) * np.unwrap(np.angle(lm15)), "k")
    plt.plot(f / 1e6, (180 / np.pi) * np.unwrap(np.angle(lm18)), "r--")
    plt.plot(fd / 1e6, (180 / np.pi) * np.unwrap(np.angle(lm15)), "k")
    plt.xlabel("frequency [MHz]")

    plt.figure(2)
    z15 = rc.gamma2impedance(lm15, 50)
    z18 = rc.gamma2impedance(lm18, 50)

    plt.subplot(1, 2, 1)
    plt.plot(fd / 1e6, np.real(z15), "k")
    plt.plot(f / 1e6, np.real(z18), "r--")
    plt.plot(fd / 1e6, np.real(z15), "k")

    plt.ylabel(r"real(Z$_{50}$) [ohm]")
    plt.xlabel("frequency [MHz]")

    plt.subplot(1, 2, 2)
    plt.plot(fd / 1e6, np.imag(z15), "k")
    plt.plot(f / 1e6, np.imag(z18), "r--")
    plt.plot(fd / 1e6, np.imag(z15), "k")

    plt.ylabel(r"imag(Z$_{50}$) [ohm]")
    plt.xlabel("frequency [MHz]")
    plt.legend(["September 2015", "February 2018"])

    return fd, om15, sm15, lm15, f, om18, sm18, lm18


def plot_season_average_residuals(
    case,
    Nfg=3,
    DDY=1.5,
    TITLE="No Beam Correction, Residuals to 5 LINLOG terms, 61-159 MHz",
    figure_name="no_beam_correction",
):
    if case == "1hr_1":
        delta_HR = 1
        fb = np.genfromtxt(
            edges_folder + "mid_band/spectra/level5/case2/case2_frequency.txt"
        )
        ty = np.genfromtxt(
            edges_folder + "mid_band/spectra/level5/case2/case2_1hr_temperature.txt"
        )
        wy = np.genfromtxt(
            edges_folder + "mid_band/spectra/level5/case2/case2_1hr_weights.txt"
        )

        ff, rr, ww = tools.spectra_to_residuals(fb, ty, wy, 61, 159, Nfg)

        ar = np.arange(0, 25, delta_HR)
        str_ar = [
            "GHA=" + str(ar[i]) + "-" + str(ar[i + 1]) + " hr"
            for i in range(len(ar) - 1)
        ]

        tools.plot_residuals(
            ff,
            rr,
            ww,
            str_ar,
            FIG_SX=7,
            FIG_SY=12,
            DY=DDY,
            FLOW=30,
            FHIGH=165,
            XTICKS=np.arange(60, 160 + 1, 20),
            XTEXT=32,
            YLABEL=str(DDY) + " K per division",
            TITLE=TITLE,
            save="yes",
            figure_name=figure_name,
            figure_format="pdf",
        )
    elif case == "1hr_2":
        delta_HR = 1
        fb = np.genfromtxt(
            edges_folder + "mid_band/spectra/level5/case2/case2_frequency.txt"
        )
        ty = np.genfromtxt(
            edges_folder + "mid_band/spectra/level5/case2/case2_1hr_temperature.txt"
        )
        wy = np.genfromtxt(
            edges_folder + "mid_band/spectra/level5/case2/case2_1hr_weights.txt"
        )

        ff, rr, ww = tools.spectra_to_residuals(fb, ty, wy, 61, 136, Nfg)

        ar = np.arange(0, 25, delta_HR)
        str_ar = [
            "GHA=" + str(ar[i]) + "-" + str(ar[i + 1]) + " hr"
            for i in range(len(ar) - 1)
        ]

        tools.plot_residuals(
            ff,
            rr,
            ww,
            str_ar,
            FIG_SX=6,
            FIG_SY=12,
            DY=DDY,
            FLOW=30,
            FHIGH=145,
            XTICKS=np.arange(60, 140 + 1, 20),
            XTEXT=32,
            YLABEL=str(DDY) + " K per division",
            TITLE=TITLE,
            save="yes",
            figure_name=figure_name,
            figure_format="pdf",
        )
    elif case == "1hr_3":
        delta_HR = 1
        fb = np.genfromtxt(
            edges_folder + "mid_band/spectra/level5/case2/case2_frequency.txt"
        )
        ty = np.genfromtxt(
            edges_folder + "mid_band/spectra/level5/case2/case2_1hr_temperature.txt"
        )
        wy = np.genfromtxt(
            edges_folder + "mid_band/spectra/level5/case2/case2_1hr_weights.txt"
        )

        ff, rr, ww = tools.spectra_to_residuals(fb, ty, wy, 110, 159, Nfg)

        ar = np.arange(0, 25, delta_HR)
        str_ar = [
            "GHA=" + str(ar[i]) + "-" + str(ar[i + 1]) + " hr"
            for i in range(len(ar) - 1)
        ]

        tools.plot_residuals(
            ff,
            rr,
            ww,
            str_ar,
            FIG_SX=7,
            FIG_SY=12,
            DY=DDY,
            FLOW=70,
            FHIGH=165,
            XTICKS=np.arange(100, 160 + 1, 20),
            XTEXT=72,
            YLABEL=str(DDY) + " K per division",
            TITLE=TITLE,
            save="yes",
            figure_name=figure_name,
            figure_format="pdf",
        )
    elif case == "2hr_1":
        delta_HR = 2
        fb = np.genfromtxt(
            edges_folder + "mid_band/spectra/level5/case2/case2_frequency.txt"
        )
        ty = np.genfromtxt(
            edges_folder + "mid_band/spectra/level5/case2/case2_2hr_temperature.txt"
        )
        wy = np.genfromtxt(
            edges_folder + "mid_band/spectra/level5/case2/case2_2hr_weights.txt"
        )

        ff, rr, ww = tools.spectra_to_residuals(fb, ty, wy, 61, 159, Nfg)

        ar = np.arange(0, 25, delta_HR)
        str_ar = [
            "GHA=" + str(ar[i]) + "-" + str(ar[i + 1]) + " hr"
            for i in range(len(ar) - 1)
        ]

        tools.plot_residuals(
            ff,
            rr,
            ww,
            str_ar,
            FIG_SX=7,
            FIG_SY=12,
            DY=DDY,
            FLOW=30,
            FHIGH=165,
            XTICKS=np.arange(60, 160 + 1, 20),
            XTEXT=32,
            YLABEL=str(DDY) + " K per division",
            TITLE=TITLE,
            save="yes",
            figure_name=figure_name,
            figure_format="pdf",
        )
    elif case == "2hr_2":
        delta_HR = 2
        fb = np.genfromtxt(
            edges_folder + "mid_band/spectra/level5/case2/case2_frequency.txt"
        )
        ty = np.genfromtxt(
            edges_folder + "mid_band/spectra/level5/case2/case2_2hr_temperature.txt"
        )
        wy = np.genfromtxt(
            edges_folder + "mid_band/spectra/level5/case2/case2_2hr_weights.txt"
        )

        ff, rr, ww = tools.spectra_to_residuals(fb, ty, wy, 61, 136, Nfg)

        ar = np.arange(0, 25, delta_HR)
        str_ar = [
            "GHA=" + str(ar[i]) + "-" + str(ar[i + 1]) + " hr"
            for i in range(len(ar) - 1)
        ]

        tools.plot_residuals(
            ff,
            rr,
            ww,
            str_ar,
            FIG_SX=6,
            FIG_SY=12,
            DY=DDY,
            FLOW=30,
            FHIGH=145,
            XTICKS=np.arange(60, 140 + 1, 20),
            XTEXT=32,
            YLABEL=str(DDY) + " K per division",
            TITLE=TITLE,
            save="yes",
            figure_name=figure_name,
            figure_format="pdf",
        )
    elif case == "3hr_1":
        delta_HR = 3
        fb = np.genfromtxt(
            edges_folder + "mid_band/spectra/level5/case2/case2_frequency.txt"
        )
        ty = np.genfromtxt(
            edges_folder + "mid_band/spectra/level5/case2/case2_3hr_temperature.txt"
        )
        wy = np.genfromtxt(
            edges_folder + "mid_band/spectra/level5/case2/case2_3hr_weights.txt"
        )

        ff, rr, ww = tools.spectra_to_residuals(fb, ty, wy, 61, 159, Nfg)

        ar = np.arange(0, 25, delta_HR)
        str_ar = [
            "GHA=" + str(ar[i]) + "-" + str(ar[i + 1]) + " hr"
            for i in range(len(ar) - 1)
        ]

        tools.plot_residuals(
            ff,
            rr,
            ww,
            str_ar,
            FIG_SX=7,
            FIG_SY=12,
            DY=DDY,
            FLOW=30,
            FHIGH=165,
            XTICKS=np.arange(60, 160 + 1, 20),
            XTEXT=32,
            YLABEL=str(DDY) + " K per division",
            TITLE=TITLE,
            save="yes",
            figure_name=figure_name,
            figure_format="pdf",
        )
    elif case == "3hr_2":
        delta_HR = 3
        fb = np.genfromtxt(
            edges_folder + "mid_band/spectra/level5/case2/case2_frequency.txt"
        )
        ty = np.genfromtxt(
            edges_folder + "mid_band/spectra/level5/case2/case2_3hr_temperature.txt"
        )
        wy = np.genfromtxt(
            edges_folder + "mid_band/spectra/level5/case2/case2_3hr_weights.txt"
        )

        ff, rr, ww = tools.spectra_to_residuals(fb, ty, wy, 61, 136, Nfg)

        ar = np.arange(0, 25, delta_HR)
        str_ar = [
            "GHA=" + str(ar[i]) + "-" + str(ar[i + 1]) + " hr"
            for i in range(len(ar) - 1)
        ]

        tools.plot_residuals(
            ff,
            rr,
            ww,
            str_ar,
            FIG_SX=6,
            FIG_SY=12,
            DY=DDY,
            FLOW=30,
            FHIGH=145,
            XTICKS=np.arange(60, 140 + 1, 20),
            XTEXT=32,
            YLABEL=str(DDY) + " K per division",
            TITLE=TITLE,
            save="yes",
            figure_name=figure_name,
            figure_format="pdf",
        )
    elif case == "4hr_1":
        fb = np.genfromtxt(
            edges_folder + "mid_band/spectra/level5/case2/case2_frequency.txt"
        )
        ar = np.genfromtxt(
            edges_folder + "mid_band/spectra/level5/case2/case2_4hr_gha_edges.txt"
        )
        ty = np.genfromtxt(
            edges_folder + "mid_band/spectra/level5/case2/case2_4hr_temperature.txt"
        )
        wy = np.genfromtxt(
            edges_folder + "mid_band/spectra/level5/case2/case2_4hr_weights.txt"
        )

        ff, rr, ww = tools.spectra_to_residuals(fb, ty, wy, 61, 159, Nfg)

        # ar     = np.arange(0, 25, delta_HR)
        str_ar = [
            "GHA=" + str(int(ar[i])) + "-" + str(int(ar[i + 1])) + " hr"
            for i in range(len(ar) - 1)
        ]

        tools.plot_residuals(
            ff,
            rr,
            ww,
            str_ar,
            FIG_SX=7,
            FIG_SY=12,
            DY=DDY,
            FLOW=30,
            FHIGH=165,
            XTICKS=np.arange(60, 160 + 1, 20),
            XTEXT=32,
            YLABEL=str(DDY) + " K per division",
            TITLE=TITLE,
            save="yes",
            figure_name=figure_name,
            figure_format="pdf",
        )
    elif case == "4hr_2":
        fb = np.genfromtxt(
            edges_folder + "mid_band/spectra/level5/case2/case2_frequency.txt"
        )
        ar = np.genfromtxt(
            edges_folder + "mid_band/spectra/level5/case2/case2_4hr_gha_edges.txt"
        )
        ty = np.genfromtxt(
            edges_folder + "mid_band/spectra/level5/case2/case2_4hr_temperature.txt"
        )
        wy = np.genfromtxt(
            edges_folder + "mid_band/spectra/level5/case2/case2_4hr_weights.txt"
        )

        ff, rr, ww = tools.spectra_to_residuals(fb, ty, wy, 61, 136, Nfg)

        # ar     = np.arange(0, 25, delta_HR)
        str_ar = [
            "GHA=" + str(int(ar[i])) + "-" + str(int(ar[i + 1])) + " hr"
            for i in range(len(ar) - 1)
        ]

        tools.plot_residuals(
            ff,
            rr,
            ww,
            str_ar,
            FIG_SX=6,
            FIG_SY=12,
            DY=DDY,
            FLOW=30,
            FHIGH=145,
            XTICKS=np.arange(140 + 1, 20),
            XTEXT=32,
            YLABEL=str(DDY) + " K per division",
            TITLE=TITLE,
            save="yes",
            figure_name=figure_name,
            figure_format="pdf",
        )


def plot_residuals_simulated_antenna_temperature(model, title):
    if model == 1:
        t = np.genfromtxt(
            edges_folder + "mid_band/calibration/beam_factors/raw/mid_band_50"
            "-200MHz_90deg_alan1_haslam_2.5_2.62_reffreq_100MHz_tant.txt"
        )
        lst = np.genfromtxt(
            edges_folder + "mid_band/calibration/beam_factors/raw/mid_band_50"
            "-200MHz_90deg_alan1_haslam_2.5_2.62_reffreq_100MHz_LST.txt"
        )
        f = np.genfromtxt(
            edges_folder + "mid_band/calibration/beam_factors/raw/mid_band_50"
            "-200MHz_90deg_alan1_haslam_2.5_2.62_reffreq_100MHz_freq.txt"
        )
    elif model == 2:
        t = np.genfromtxt(
            edges_folder + "mid_band/calibration/beam_factors/raw/mid_band_50"
            "-200MHz_90deg_alan1_haslam_2.5_2.5_reffreq_80MHz_tant.txt"
        )
        lst = np.genfromtxt(
            edges_folder + "mid_band/calibration/beam_factors/raw/mid_band_50"
            "-200MHz_90deg_alan1_haslam_2.5_2.5_reffreq_80MHz_LST.txt"
        )
        f = np.genfromtxt(
            edges_folder + "mid_band/calibration/beam_factors/raw/mid_band_50"
            "-200MHz_90deg_alan1_haslam_2.5_2.5_reffreq_80MHz_freq.txt"
        )
    # Converting LST to GHA
    gha = lst - 17.76
    gha[gha < 0] = gha[gha < 0] + 24

    IX = np.argsort(gha)
    gha = gha[IX]
    t = t[IX, :]

    t1 = np.zeros((24, len(f)))
    for i in range(24):
        tb = t[(gha >= i) & (gha <= i + 1), :]
        print(gha[(gha >= i) & (gha <= i + 1)])
        avtb = np.mean(tb, axis=0)
        t1[i, :] = avtb

    w = np.ones((len(t1[:, 0]), len(t[0, :])))
    fx, rx, wx = tools.spectra_to_residuals(f, t1, w, 61, 159, 5)
    index = np.arange(0, 24, 1)

    ar = np.arange(0, 25, 1)
    str_ar = [
        "GHA=" + str(ar[i]) + "-" + str(ar[i + 1]) + " hr" for i in range(len(ar) - 1)
    ]

    plt.figure()
    tools.plot_residuals(
        fx,
        rx[index, :],
        wx[index, :],
        str_ar,
        FIG_SX=7,
        FIG_SY=12,
        DY=1.5,
        FLOW=30,
        FHIGH=165,
        XTICKS=np.arange(60, 160 + 1, 20),
        XTEXT=32,
        YLABEL="1.5 K per division",
        TITLE=title,
        save="yes",
        figure_name="simulation",
        figure_format="pdf",
    )


def level4_plot_integrated_residuals(case, FLOW=60, FHIGH=150):
    if case == 2:
        d = np.genfromtxt(
            edges_folder
            + "mid_band/spectra/level4/calibration_2019_10_no_ground_loss_no_beam_corrections"
            "/binned_averages/GHA_every_1hr.txt"
        )

    elif case == 3:
        d = np.genfromtxt(
            edges_folder
            + "mid_band/spectra/level4/case_nominal_50-150MHz_no_ground_loss_no_beam_corrections"
            "/binned_averages/GHA_every_1hr.txt"
        )

    elif case == 5:
        d = np.genfromtxt(
            edges_folder + "mid_band/spectra/level4/case_nominal_14_14_terms_55"
            "-150MHz_no_ground_loss_no_beam_corrections/binned_averages/GHA_every_1hr.txt"
        )

    elif case == 406:
        d = np.genfromtxt(
            edges_folder
            + "mid_band/spectra/level4/case_nominal_50-150MHz_LNA1_a2_h2_o2_s1_sim2"
            "/binned_averages/GHA_every_1hr.txt"
        )

    elif case == 501:
        d = np.genfromtxt(
            edges_folder
            + "mid_band/spectra/level4/case_nominal_50-150MHz_LNA2_a2_h2_o2_s1_sim2_all_lc_yes_bc"
            "/binned_averages/GHA_every_1hr.txt"
        )

    f = d[:, 0]

    start_low = 0
    start_high = 0
    for i in range(24):
        if (i >= 6) and (i <= 17):
            ff, r = tools.spectrum_fit(
                f,
                d[:, i + 1],
                d[:, i + 1 + 48],
                Nfg=5,
                F1_LOW=FLOW,
                F1_HIGH=FHIGH,
                F2_LOW=FLOW,
                F2_HIGH=FHIGH,
            )
            if start_low == 0:
                r_low = np.copy(r)
                w_low = d[:, i + 1 + 48]
                start_low = 1
            else:
                r_low = np.vstack((r_low, r))
                w_low = np.vstack((w_low, d[:, i + 1 + 48]))

        if (i < 6) or (i > 17):
            ff, r = tools.spectrum_fit(
                f,
                d[:, i + 1],
                d[:, i + 1 + 48],
                Nfg=5,
                F1_LOW=FLOW,
                F1_HIGH=FHIGH,
                F2_LOW=FLOW,
                F2_HIGH=FHIGH,
            )
            if start_high == 0:
                r_high = np.copy(r)
                w_high = d[:, i + 1 + 48]
                start_high = 1
            else:
                r_high = np.vstack((r_high, r))
                w_high = np.vstack((w_high, d[:, i + 1 + 48]))

    r_high1 = r_high[0:6, :]
    r_high2 = r_high[6::, :]
    r_high = np.vstack((r_high2, r_high1))

    w_high1 = w_high[0:6, :]
    w_high2 = w_high[6::, :]
    w_high = np.vstack((w_high2, w_high1))

    plt.figure(figsize=[13, 11])
    plt.subplot(1, 2, 1)
    c = "b"
    for i in range(len(r_low[:, 0])):
        plt.plot(f[w_low[i, :] > 0], r_low[i, :][w_low[i, :] > 0] - 0.5 * i, c)
        if c == "b":
            c = "r"
        elif c == "r":
            c = "b"
    plt.xlim([60, 150])
    plt.grid()
    plt.ylim([-6, 0.5])
    plt.xlabel("frequency [MHz]")
    plt.ylabel("GHA\n [0.5 K per division]")
    plt.yticks(np.arange(-5.5, 0.1, 0.5), np.arange(17, 5, -1))

    plt.subplot(1, 2, 2)
    c = "b"
    for i in range(len(r_high[:, 0])):
        plt.plot(f[w_high[i, :] > 0], r_high[i, :][w_high[i, :] > 0] - 2 * i, c)
        if c == "b":
            c = "r"
        elif c == "r":
            c = "b"
    plt.xlim([60, 150])
    plt.grid()
    plt.ylim([-24, 2])
    plt.xlabel("frequency [MHz]")
    plt.ylabel("GHA\n [2 K per division]")
    plt.yticks(np.arange(-22, 0.1, 2), np.arange(5, -7, -1))

    plt.savefig(
        edges_folder
        + "plots/20191218/data_residuals_no_ground_loss_no_beam_correction.pdf",
        bbox_inches="tight",
    )


def plot_level4(
    case,
    index_GHA,
    averaged_146,
    good_bad,
    FLOW,
    FHIGH,
    model,
    Nfg,
    K_per_division,
    file_name,
):
    """
    model: 'LINLOG', 'LOGLOG'

    """

    # Load Level 4 data
    if case == 101:
        f, py, ry, wy, index, gha, yy = io.level4read(
            "/home/raul/DATA2/EDGES_vol2/mid_band/spectra"
            "/level4/rcv18_sw18_nominal_GHA_every_1hr/rcv18_sw18_nominal_GHA_every_1hr.hdf5"
        )

    # save_rms_folder = edges_folder + 'mid_band/spectra/level4/rcv18_ant19_nominal/rms_filters/'
    # if not exists(save_rms_folder):
    # makedirs(save_rms_folder)

    if case == 200:
        f, py, ry, wy, index, gha, yy = io.level4read(
            "/home/raul/DATA2/EDGES_vol2/mid_band/spectra/level4/rcv18_ant19_nominal"
            "/rcv18_ant19_nominal.hdf5"
        )

        save_rms_folder = (
            edges_folder + "mid_band/spectra/level4/rcv18_ant19_nominal/rms_filters/"
        )
        if not exists(save_rms_folder):
            makedirs(save_rms_folder)

    if case == 201:
        f, py, ry, wy, index, gha, yy = io.level4read(
            "/home/raul/DATA2/EDGES_vol2/mid_band/spectra/level4/rcv18_ant19_every_1hr_GHA"
            "/rcv18_ant19_every_1hr_GHA.hdf5"
        )

        save_rms_folder = (
            edges_folder
            + "mid_band/spectra/level4/rcv18_ant19_every_1hr_GHA/rms_filters/"
        )
        if not exists(save_rms_folder):
            makedirs(save_rms_folder)

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

    if averaged_146 == "yes":

        ldays = len(rx[:, 0, 0])
        lgha = len(rx[0, :, 0])

        pb = np.zeros((ldays - 1, lgha, len(px[0, 0, :])))
        rb = np.zeros((ldays - 1, lgha, len(rx[0, 0, :])))
        wb = np.zeros((ldays - 1, lgha, len(wx[0, 0, :])))

        for i in range(ldays - 1):
            for j in range(lgha):
                print([i, j])

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

        print(px.shape)
        print(rx.shape)
        print(wx.shape)
        print(yx.shape)

    # Identify indices of good and bad days
    kk = filters.daily_nominal_filter("mid_band", case, index_GHA, yx)
    # kk = daily_strict_filter('mid_band', yx)
    # kk = np.ones(len(px[:,0]))

    if good_bad == "good":
        p_all = px[kk == 1, :]
        r_all = rx[kk == 1, :]
        w_all = wx[kk == 1, :]
        yd = yx[kk == 1]

    elif good_bad == "bad":
        p_all = px[kk == 0, :]
        r_all = rx[kk == 0, :]
        w_all = wx[kk == 0, :]
        yd = yx[kk == 0]

    # index = np.arange(len(yd))

    # Nfg = 4
    NS = 168

    plt.close()
    plt.close()
    plt.close()
    plt.close()
    plt.close()

    if good_bad == "good":
        plt.figure(1, figsize=[7, 15])
    elif good_bad == "bad":
        plt.figure(1, figsize=[5, 5])

    plt.ylabel(
        "day of year 2018   [" + str(K_per_division) + " K per division]\n  \n  "
    )

    # for i in range(2):

    RMS_all = np.zeros((len(yd[:, 0]), 7))
    ii = -1
    for i in range(len(yd)):

        print(str(i) + " " + str(yd[i, 1]))

        avp = p_all[i, index_GHA, :]
        avr = r_all[i, index_GHA, :]
        avw = w_all[i, index_GHA, :]

        lw = len(avw[avw > 0])
        print(lw)

        if lw > 0:
            ii = ii + 1
            rr, wr = rfi.cleaning_sweep(
                f,
                avr,
                avw,
                window_width_MHz=3,
                Npolyterms_block=2,
                N_choice=20,
                N_sigma=3.5,
            )

            m = mdl.model_evaluate("LINLOG", avp, f / 200)
            tr = m + rr

            ft = f[(f >= FLOW) & (f <= FHIGH)]
            tt = tr[(f >= FLOW) & (f <= FHIGH)]
            wt = wr[(f >= FLOW) & (f <= FHIGH)]

            if model == "LINLOG":
                pt = mdl.fit_polynomial_fourier("LINLOG", ft / 200, tt, Nfg, Weights=wt)
                mt = mdl.model_evaluate("LINLOG", pt[0], ft / 200)
                rt = tt - mt

                fb, rb, wb, sb = tools.spectral_binning_number_of_samples(
                    ft, rt, wt, nsamples=NS
                )

            if model == "LOGLOG":
                pl = np.polyfit(np.log(ft[wt > 0] / 200), np.log(tt[wt > 0]), Nfg - 1)
                log_ml = np.polyval(pl, np.log(ft / 200))
                ml = np.exp(log_ml)
                rl = tt - ml

                fb, rb, wb, sb = tools.spectral_binning_number_of_samples(
                    ft, rl, wt, nsamples=NS
                )

            if i % 2 == 0:
                plt.plot(fb[wb > 0], rb[wb > 0] - ii * K_per_division, "b")
            else:
                plt.plot(fb[wb > 0], rb[wb > 0] - ii * K_per_division, "r")

            RMS = 1000 * np.std(rb[wb > 0])
            RMS_text = str(int(RMS)) + " mK"
            print(RMS_text)

            plt.text(
                50, -ii * K_per_division - (1 / 6) * K_per_division, str(int(yd[i, 1]))
            )
            plt.text(153, -ii * K_per_division - (1 / 6) * K_per_division, RMS_text)

            RMS_all[i, 0] = yd[i, 0]
            RMS_all[i, 1] = yd[i, 1]
            RMS_all[i, 2] = index_GHA
            RMS_all[i, 3] = FLOW
            RMS_all[i, 4] = FHIGH
            RMS_all[i, 5] = Nfg
            RMS_all[i, 6] = RMS

    plt.xlim([55, 152])
    plt.xticks(np.arange(60, 151, 10))
    plt.xlabel(r"$\nu$ [MHz]", fontsize=12)
    plt.ylim([-(ii + 1) * K_per_division, K_per_division])
    plt.yticks([10], labels=[""])

    GHA_integration = gha[2] - gha[1]

    GHA1 = gha[index_GHA]
    GHA2 = gha[index_GHA] + GHA_integration
    if GHA2 > 24:
        GHA2 = GHA2 - 24
    plt.title(
        "GHA="
        + str(GHA1)
        + "-"
        + str(GHA2)
        + " hr, "
        + model
        + ", "
        + str(int(Nfg))
        + " terms",
        fontsize=16,
    )

    plt.savefig("/home/raul/Desktop/" + file_name + ".pdf", bbox_inches="tight")

    return fb, rb, wb, RMS_all


def plot_residuals(
    f,
    r,
    w,
    list_names,
    FIG_SX=7,
    FIG_SY=12,
    DY=2,
    FLOW=50,
    FHIGH=180,
    XTICKS=np.arange(60, 180 + 1, 20),
    XTEXT=160,
    YLABEL="ylabel",
    TITLE="hello",
    save="no",
    figure_path="/home/raul/Desktop/",
    figure_name="2018_150_00",
    figure_format="png",
):
    N_spec = len(r[:, 0])

    plt.figure(figsize=(FIG_SX, FIG_SY))

    for i in range(len(list_names)):
        print(i)

        if i % 2 == 0:
            color = "r"
        else:
            color = "b"

        plt.plot(f[w[i] > 0], (r[i] - i * DY)[w[i] > 0], color)
        plt.text(XTEXT, -i * DY, list_names[i])

    plt.xlim([FLOW, FHIGH])
    plt.ylim([-DY * (N_spec), DY])

    plt.grid()

    plt.xticks(XTICKS)
    plt.yticks([])

    plt.xlabel("frequency [MHz]")
    plt.ylabel(YLABEL)

    plt.title(TITLE)

    if save == "yes":
        plt.savefig(
            figure_path + figure_name + "." + figure_format, bbox_inches="tight"
        )


def plot_level4_second_approach(
    case,
    index_GHA,
    averaged_146,
    good_bad,
    model,
    FLOW,
    FHIGH,
    F1L,
    F1H,
    F2L,
    F2H,
    Nfg,
    K_per_division,
    file_name,
):
    """
    model: 'LINLOG', 'LOGLOG'

    """
    if case == 1:
        f, py, ry, wy, index, gha, yy = io.level4read(
            "/home/raul/DATA2/EDGES_vol2/mid_band/spectra/level4/case_nominal_55_150MHz"
            "/case_nominal_55_150MHz.hdf5"
        )

        save_rms_folder = (
            edges_folder + "mid_band/spectra/level4/case_nominal_55_150MHz/rms_filters/"
        )
        if not exists(save_rms_folder):
            makedirs(save_rms_folder)

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

    if averaged_146 == "yes":

        ldays = len(rx[:, 0, 0])
        lgha = len(rx[0, :, 0])

        pb = np.zeros((ldays - 1, lgha, len(px[0, 0, :])))
        rb = np.zeros((ldays - 1, lgha, len(rx[0, 0, :])))
        wb = np.zeros((ldays - 1, lgha, len(wx[0, 0, :])))

        for i in range(ldays - 1):
            for j in range(lgha):
                print([i, j])

                pa = np.array((px[0, j, :], px[i + 1, j, :]))
                ra = np.array((rx[0, j, :], rx[i + 1, j, :]))
                wa = np.array((wx[0, j, :], wx[i + 1, j, :]))

                h1 = np.mean(pa, axis=0)
                h2, h3 = mdl.spectral_averaging(ra, wa)

                pb[i, j, :] = h1
                rb[i, j, :] = h2
                wb[i, j, :] = h3

        px = np.copy(pb)
        rx = np.copy(rb)
        wx = np.copy(wb)
        yx = np.delete(yx, 0, axis=0)

        print(px.shape)
        print(rx.shape)
        print(wx.shape)
        print(yx.shape)

    # Identify indices of good and bad days
    kk = filters.daily_nominal_filter("mid_band", case, yx)

    if good_bad == "good":
        p_all = px[kk == 1, :]
        r_all = rx[kk == 1, :]
        w_all = wx[kk == 1, :]
        yd = yx[kk == 1]

    elif good_bad == "bad":
        p_all = px[kk == 0, :]
        r_all = rx[kk == 0, :]
        w_all = wx[kk == 0, :]
        yd = yx[kk == 0]

    NS = 168

    if good_bad == "good":
        plt.figure(1, figsize=[7, 20])
    elif good_bad == "bad":
        plt.figure(1, figsize=[5, 5])

    plt.ylabel(
        "day of year 2018   [" + str(K_per_division) + " K per division]\n  \n  "
    )

    RMS_all = np.zeros((len(yd[:, 0]), 7))
    ii = -1
    for i in range(len(yd)):

        print(str(i) + " " + str(yd[i, 1]))

        avp = p_all[i, index_GHA, :]
        avr = r_all[i, index_GHA, :]
        avw = w_all[i, index_GHA, :]

        lw = len(avw[avw > 0])
        print(lw)

        if lw > 0:
            ii = ii + 1
            rr, wr = rfi.cleaning_sweep(
                f,
                avr,
                avw,
                window_width_MHz=3,
                Npolyterms_block=2,
                N_choice=20,
                N_sigma=3.5,
            )

            m = mdl.model_evaluate("LINLOG", avp, f / 200)
            tr = m + rr

            ft = f[(f >= FLOW) & (f <= FHIGH)]
            tt = tr[(f >= FLOW) & (f <= FHIGH)]
            wt = wr[(f >= FLOW) & (f <= FHIGH)]

            # ----------------------------------------------------------------------
            index_all = np.arange(0, len(ft))
            index_sel = index_all[
                ((ft >= F1L) & (ft <= F1H)) | ((ft >= F2L) & (ft <= F2H))
            ]

            fx = ft[index_sel]
            tx = tt[index_sel]
            wx = wt[index_sel]

            if model == "LINLOG":
                p = mdl.fit_polynomial_fourier("LINLOG", fx / 200, tx, Nfg, Weights=wx)
                mt = mdl.model_evaluate("LINLOG", p[0], ft / 200)
                rt = tt - mt

                fb, rb, wb, sb = tools.spectral_binning_number_of_samples(
                    ft, rt, wt, nsamples=NS
                )

            if model == "LOGLOG":
                p = np.polyfit(np.log(fx[wx > 0] / 200), np.log(tx[wx > 0]), Nfg - 1)
                log_ml = np.polyval(p, np.log(ft / 200))
                ml = np.exp(log_ml)
                rl = tt - ml

                fb, rb, wb, sb = tools.spectral_binning_number_of_samples(
                    ft, rl, wt, nsamples=NS
                )

            if i % 2 == 0:
                plt.plot(fb[wb > 0], rb[wb > 0] - ii * K_per_division, "b")
            else:
                plt.plot(fb[wb > 0], rb[wb > 0] - ii * K_per_division, "r")

            RMS = 1000 * np.std(rb[wb > 0])
            RMS_text = str(int(RMS)) + " mK"
            print(RMS_text)

            plt.text(
                50, -ii * K_per_division - (1 / 6) * K_per_division, str(int(yd[i, 1]))
            )
            plt.text(153, -ii * K_per_division - (1 / 6) * K_per_division, RMS_text)

            RMS_all[i, 0] = yd[i, 0]
            RMS_all[i, 1] = yd[i, 1]
            RMS_all[i, 2] = index_GHA
            RMS_all[i, 3] = FLOW
            RMS_all[i, 4] = FHIGH
            RMS_all[i, 5] = Nfg
            RMS_all[i, 6] = RMS

    plt.plot([F1L, F1L], [-1000, 1000], "y--", linewidth=1)
    plt.plot([F1H, F1H], [-1000, 1000], "y--", linewidth=1)
    plt.plot([F2L, F2L], [-1000, 1000], "y--", linewidth=1)
    plt.plot([F2H, F2H], [-1000, 1000], "y--", linewidth=1)
    plt.xlim([55, 151])
    plt.xlabel(r"$\nu$ [MHz]", fontsize=12)
    plt.ylim([-(ii + 1.5) * K_per_division, 1.5 * K_per_division])
    plt.yticks([10], labels=[""])

    plt.savefig("/home/raul/Desktop/" + file_name + ".pdf", bbox_inches="tight")

    return fb, rb, wb, RMS_all


def plots_level3_rms_folder(
    band, case, YTOP_LOW=10, YTOP_HIGH=50, YBOTTOM_LOW=10, YBOTTOM_HIGH=30
):
    """
    This function plots the RMS of residuals of Level3 data

    YTOP_LOW/_HIGH:      y-limits of top panel
    YBOTTOM_LOW/_HIGH:   y-limits of bottom panel
    """

    # Case selection
    if case == 1:
        flag_folder = "nominal_60_160MHz"

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
        f, t, p, r, w, rms, m = io.level3read(
            path_files + list_files[i], print_key="no"
        )

        plt.figure()

        plt.subplot(2, 1, 1)
        plt.plot(m[:, 3], rms[:, 0], ".")
        plt.xticks(np.arange(0, 25, 2))
        plt.grid()
        plt.xlim([0, 24])
        plt.ylim([YTOP_LOW, YTOP_HIGH])

        plt.ylabel("RMS [K]")
        plt.title(list_files[i] + ":  Low-frequency Half")

        plt.subplot(2, 1, 2)
        plt.plot(m[:, 3], rms[:, 1], ".r")
        plt.xticks(np.arange(0, 25, 2))
        plt.grid()
        plt.xlim([0, 24])
        plt.ylim([YBOTTOM_LOW, YBOTTOM_HIGH])

        plt.ylabel("RMS [K]")
        plt.xlabel("LST [Hr]")
        plt.title(list_files[i] + ":  High-frequency Half")

        if len(list_files[0]) > 12:
            file_name = list_files[i][0:11]

        elif len(list_files[0]) == 12:
            file_name = list_files[i][0:8]

        plt.savefig(path_plots + file_name + ".png", bbox_inches="tight")


def level4_plot_residuals(case, GHA_index, TITLE, subfolder, figure_save_name, DY):
    if case == 2:
        filename = (
            edges_folder
            + "mid_band/spectra/level4/calibration_2019_10_no_ground_loss_no_beam_corrections"
            "/binned_residuals/binned_residuals_one_hour_GHA.hdf5"
        )
        figure_save_path_subfolder = (
            edges_folder
            + "mid_band/spectra/level4/calibration_2019_10_no_ground_loss_no_beam_corrections"
            "/binned_plots/" + subfolder + "/"
        )
    elif case == 3:
        filename = (
            edges_folder + "mid_band/spectra/level4/case_nominal_50"
            "-150MHz_no_ground_loss_no_beam_corrections"
            "/binned_residuals/binned_residuals_one_hour_GHA.hdf5"
        )
        figure_save_path_subfolder = (
            edges_folder
            + "mid_band/spectra/level4/case_nominal_50-150MHz_no_ground_loss_no_beam_corrections"
            "/binned_plots/" + subfolder + "/"
        )
    elif case == 406:
        filename = (
            edges_folder
            + "mid_band/spectra/level4/case_nominal_50-150MHz_LNA1_a2_h2_o2_s1_sim2"
            "/binned_residuals"
            "/binned_residuals_one_hour_GHA.hdf5"
        )
        figure_save_path_subfolder = (
            edges_folder
            + "mid_band/spectra/level4/case_nominal_50-150MHz_LNA1_a2_h2_o2_s1_sim2/binned_plots/"
            + subfolder
            + "/"
        )
    elif case == 5:
        filename = (
            edges_folder + "mid_band/spectra/level4/case_nominal_14_14_terms_55"
            "-150MHz_no_ground_loss_no_beam_corrections/binned_residuals"
            "/binned_residuals_one_hour_GHA.hdf5"
        )
        figure_save_path_subfolder = (
            edges_folder + "mid_band/spectra/level4/case_nominal_14_14_terms_55"
            "-150MHz_no_ground_loss_no_beam_corrections/binned_plots/" + subfolder + "/"
        )
    elif case == 501:
        filename = (
            edges_folder + "mid_band/spectra/level4/case_nominal_50"
            "-150MHz_LNA2_a2_h2_o2_s1_sim2_all_lc_yes_bc"
            "/binned_residuals/binned_residuals_one_hour_GHA.hdf5"
        )
        figure_save_path_subfolder = (
            edges_folder
            + "mid_band/spectra/level4/case_nominal_50-150MHz_LNA2_a2_h2_o2_s1_sim2_all_lc_yes_bc"
            "/binned_plots/" + subfolder + "/"
        )
    elif case == 5011:
        filename = (
            edges_folder + "mid_band/spectra/level4/case_nominal_50"
            "-150MHz_LNA2_a2_h2_o2_s1_sim2_all_lc_yes_bc"
            "/binned_residuals/binned_residuals_one_hour_GHA_60-120MHz.hdf5"
        )
        figure_save_path_subfolder = (
            edges_folder
            + "mid_band/spectra/level4/case_nominal_50-150MHz_LNA2_a2_h2_o2_s1_sim2_all_lc_yes_bc"
            "/binned_plots/" + subfolder + "/"
        )
    if not exists(figure_save_path_subfolder):
        makedirs(figure_save_path_subfolder)

    fb, rb, wb, sb, gha, yd = io.level4_binned_read(filename)
    start = 0
    for i in range(len(yd)):
        if np.sum(wb[i, GHA_index, :]) > 0:

            std_x = np.std(rb[i, GHA_index, :][wb[i, GHA_index, :] > 0])

            path_file = (
                edges_folder + "mid_band/spectra/level3/case_nominal_50"
                "-150MHz_LNA2_a2_h2_o2_s1_sim2_all_lc_yes_bc/"
                + str(int(yd[i, 0]))
                + "_"
                + str(int(yd[i, 1]))
                + "_00.hdf5"
            )
            f, t, p, r, w, rms, tp, m = io.level3read(path_file, print_key="no")

            GHA_level3 = m[:, 4]
            GHA_level3[GHA_level3 < 0] = GHA_level3[GHA_level3 < 0] + 24
            SUNEL = m[(GHA_level3 >= GHA_index) & (GHA_level3 <= (GHA_index + 1)), 6]
            EL = str(int(np.max(SUNEL))) if len(SUNEL) > 0 else "X"
            if start == 0:
                rb_new = rb[i, GHA_index, :]
                wb_new = wb[i, GHA_index, :]
                yd_new = yd[i]
                std_new = np.copy(std_x)
                EL_new = [EL]

                start = 1

            elif start == 1:
                rb_new = np.vstack((rb_new, rb[i, GHA_index, :]))
                wb_new = np.vstack((wb_new, wb[i, GHA_index, :]))
                yd_new = np.vstack((yd_new, yd[i]))
                std_new = np.append(std_new, std_x)
                EL_new.append(EL)

            print(EL_new)

    # Settings
    LST_text = [
        str(int(yd_new[i, 1]))
        + ": "
        + EL_new[i]
        + " deg, "
        + str(int(1000 * std_new[i]))
        + " mK"
        for i in range(len(yd_new))
    ]

    FIG_SX = 10
    FIG_SY = 20

    YLABEL = str(DY) + " K per division"
    FIGURE_FORMAT = "pdf"

    if case == 5011:
        FLOW_plot = 40
        FHIGH_plot = 122
        XTICKS = np.arange(60, 121, 10)
        XTEXT = 40.5

    else:
        FLOW_plot = 35
        FHIGH_plot = 152
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
        DY=DY,
        FLOW=FLOW_plot,
        FHIGH=FHIGH_plot,
        XTICKS=XTICKS,
        XTEXT=XTEXT,
        YLABEL=YLABEL,
        TITLE=TITLE,
        save="yes",
        figure_path=figure_save_path_subfolder,
        figure_name=figure_save_name,
        figure_format=FIGURE_FORMAT,
    )

    return rb_new, wb_new, yd_new


def plot_sky_model():
    # Loading Haslam map
    map408, lon, lat, gc = sky_models.haslam_408MHz_map()
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
    )  # , ,     , min=2.2, max=3.9 unit=r'log($T_{sky}$)' title='',
    # min=2.3, max=2.6, unit=r'$\beta$'), , rot=[180,0,0]
    hp.graticule(local=True)
    beam = beams.feko_blade_beam(
        "mid_band", 0, frequency_interpolation="no", AZ_antenna_axis=90
    )
    beam90 = beam[20, :, :]
    beam90n = beam90 / np.max(beam90)
    FWHM = np.zeros((360, 2))
    EL_raw = np.arange(0, 91, 1)
    EL_new = np.arange(0, 90.01, 0.01)
    for j in range(len(beam90[0, :])):  # Loop over AZ
        # print(j)

        func = interp.interp1d(EL_raw, beam90n[:, j])
        beam90n_interp = func(EL_new)

        minDiff = 100
        for i in range(len(EL_new)):
            Diff = np.abs(beam90n_interp[i] - 0.5)
            if Diff < minDiff:
                # print(90-EL_new[i])
                minDiff = np.copy(Diff)
                FWHM[j, 0] = j
                FWHM[j, 1] = 90 - EL_new[i]
    # Reference location
    EDGES_lat_deg = -26.714778
    EDGES_lon_deg = 116.605528
    EDGES_location = apc.EarthLocation(
        lat=EDGES_lat_deg * apu.deg, lon=EDGES_lon_deg * apu.deg
    )
    Time_iter_UTC_start = np.array([2014, 1, 1, 3, 31, 0])
    Time_iter_UTC_middle = np.array([2014, 1, 1, 9, 30, 0])
    Time_iter_UTC_end = np.array([2014, 1, 1, 15, 29, 0])
    Time_iter_UTC_start_dt = dt.datetime(
        Time_iter_UTC_start[0],
        Time_iter_UTC_start[1],
        Time_iter_UTC_start[2],
        Time_iter_UTC_start[3],
        Time_iter_UTC_start[4],
        Time_iter_UTC_start[5],
    )
    Time_iter_UTC_middle_dt = dt.datetime(
        Time_iter_UTC_middle[0],
        Time_iter_UTC_middle[1],
        Time_iter_UTC_middle[2],
        Time_iter_UTC_middle[3],
        Time_iter_UTC_middle[4],
        Time_iter_UTC_middle[5],
    )
    Time_iter_UTC_end_dt = dt.datetime(
        Time_iter_UTC_end[0],
        Time_iter_UTC_end[1],
        Time_iter_UTC_end[2],
        Time_iter_UTC_end[3],
        Time_iter_UTC_end[4],
        Time_iter_UTC_end[5],
    )
    # Converting Beam Contours from Local to Equatorial coordinates
    AltAz_start = apc.SkyCoord(
        alt=(90 - FWHM[:, 1]) * apu.deg,
        az=FWHM[:, 0] * apu.deg,
        frame="altaz",
        obstime=apt.Time(Time_iter_UTC_start_dt, format="datetime"),
        location=EDGES_location,
    )
    RaDec_start = AltAz_start.icrs
    Ra_start = np.asarray(RaDec_start.ra)
    Dec_start = np.asarray(RaDec_start.dec)
    RaWrap_start = np.copy(Ra_start)
    RaWrap_start[Ra_start > 180] = Ra_start[Ra_start > 180] - 360
    AltAz_middle = apc.SkyCoord(
        alt=(90 - FWHM[:, 1]) * apu.deg,
        az=FWHM[:, 0] * apu.deg,
        frame="altaz",
        obstime=apt.Time(Time_iter_UTC_middle_dt, format="datetime"),
        location=EDGES_location,
    )
    RaDec_middle = AltAz_middle.icrs
    Ra_middle = np.asarray(RaDec_middle.ra)
    Dec_middle = np.asarray(RaDec_middle.dec)
    RaWrap_middle = np.copy(Ra_middle)
    RaWrap_middle[Ra_middle > 180] = Ra_middle[Ra_middle > 180] - 360
    AltAz_end = apc.SkyCoord(
        alt=(90 - FWHM[:, 1]) * apu.deg,
        az=FWHM[:, 0] * apu.deg,
        frame="altaz",
        obstime=apt.Time(Time_iter_UTC_end_dt, format="datetime"),
        location=EDGES_location,
    )
    RaDec_end = AltAz_end.icrs
    Ra_end = np.asarray(RaDec_end.ra)
    Dec_end = np.asarray(RaDec_end.dec)
    RaWrap_end = np.copy(Ra_end)
    RaWrap_end[Ra_end > 180] = Ra_end[Ra_end > 180] - 360
    plt.plot(np.arange(-180, 181, 1), -26.7 * np.ones(361), "y--", linewidth=2)
    plt.plot(RaWrap_start, Dec_start, "w", linewidth=3)
    plt.plot(RaWrap_middle, Dec_middle, "w--", linewidth=3)
    plt.plot(RaWrap_end, Dec_end, "w:", linewidth=3)
    plt.plot(-6 * (360 / 24), -26.7, "x", color="1", markersize=5, mew=2)
    plt.plot(0 * (360 / 24), -26.7, "x", color="1", markersize=5, mew=2)
    plt.plot(6 * (360 / 24), -26.7, "x", color="1", markersize=5, mew=2)
    off_x = -4
    off_y = -12
    plt.text(-180 + off_x, -90 + off_y, "0")
    plt.text(-150 + off_x, -90 + off_y, "2")
    plt.text(-120 + off_x, -90 + off_y, "4")
    plt.text(-90 + off_x, -90 + off_y, "6")
    plt.text(-60 + off_x, -90 + off_y, "8")
    plt.text(-30 + off_x, -90 + off_y, "10")
    plt.text(-0 + off_x, -90 + off_y, "12")
    plt.text(30 + off_x, -90 + off_y, "14")
    plt.text(60 + off_x, -90 + off_y, "16")
    plt.text(90 + off_x, -90 + off_y, "18")
    plt.text(120 + off_x, -90 + off_y, "20")
    plt.text(150 + off_x, -90 + off_y, "22")
    plt.text(180 + off_x, -90 + off_y, "24")
    plt.text(-60, -115, "galactic hour angle [hr]")
    off_x1 = -15
    off_x2 = -10
    off_x3 = -19
    off_y = -3
    plt.text(-180 + off_x1, 90 + off_y, "90")
    plt.text(-180 + off_x1, 60 + off_y, "60")
    plt.text(-180 + off_x1, 30 + off_y, "30")
    plt.text(-180 + off_x2, 0 + off_y, "0")
    plt.text(-180 + off_x3, -30 + off_y, "-30")
    plt.text(-180 + off_x3, -60 + off_y, "-60")
    plt.text(-180 + off_x3, -90 + off_y, "-90")
    plt.text(-210, 45, "declination [degrees]", rotation=90)


def plot_sky_model_comparison():
    # Loading Haslam map
    map1, lon1, lat1, gc1 = sky_models.haslam_408MHz_map()
    # Loading LW map
    map2, lon2, lat2, gc2 = sky_models.LW_150MHz_map()
    # Loading Guzman map
    map3, lon3, lat3, gc3 = sky_models.guzman_45MHz_map()
    # Scaling sky map (the map contains the CMB, which has to be removed and then added back)
    # ---------------------------------------------------------------------------------------
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
    hp.cartview(s1, nest="yes", min=500, max=2000, cbar=False, coord="GC")
    hp.cartview(s2, nest="yes", min=500, max=2000, cbar=False, coord="GC")
    hp.cartview(ss4, nest="yes", min=500, max=2000, cbar=False, coord="GC")
    hp.cartview(ss5, nest="yes", min=500, max=2000, cbar=False, coord="GC")
    dLIM = 500
    hp.cartview(s2 - s1, nest="yes", min=-dLIM, max=dLIM, cbar=False, coord="GC")
    hp.cartview(ss4 - s1, nest="yes", min=-dLIM, max=dLIM, cbar=False, coord="GC")
    hp.cartview(ss5 - s1, nest="yes", min=-dLIM, max=dLIM, cbar=False, coord="GC")


def plot_data_stats():
    # Plot of histogram of GHA for integrated spectrum
    f, px, rx, wx, index, gha, ydx = io.level4read(
        "/home/raul/DATA2/EDGES_vol2/mid_band/spectra/level4/case26/case26.hdf5"
    )
    keep = filters.daily_nominal_filter("mid_band", 26, ydx)
    ydy = ydx[keep > 0]
    ix = np.arange(len(ydy))
    path_files = edges_folder + "mid_band/spectra/level3/case26/"
    new_list = listdir(path_files)
    new_list.sort()
    index_new_list = range(len(new_list))
    gha_all = np.array([])
    for i in index_new_list:

        if len(new_list[i]) == 8:
            day = float(new_list[i][5::])
        elif len(new_list[i]) > 8:
            day = float(new_list[i][5:8])

        Q = ix[ydy[:, 1] == day]

        if len(Q) > 0:
            print(new_list[i])
            f, ty, py, ry, wy, rmsy, tpy, my = io.level3read(path_files + new_list[i])
            ii = index[i, 0, 0 : len(my)]
            gha_i = my[ii > 0, 4]
            gha_i[gha_i < 0] = gha_i[gha_i < 0] + 24
            gha_all = np.append(gha_all, gha_i)
    sp = np.genfromtxt(
        edges_folder + "mid_band/spectra/level5/case2/integrated_spectrum_case2.txt"
    )
    fb = sp[:, 0]
    wb = sp[:, 2]
    fbb = fb[fb >= 60]
    wbb = wb[fb >= 60]
    plt.close()
    plt.close()
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
        edges_folder + "plots/20190730/data_statistics.pdf", bbox_inches="tight"
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
    FLOW = 60
    FHIGH = 100
    fl = flx[(flx >= FLOW) & (flx <= FHIGH)]
    tl = tlx[:, (flx >= FLOW) & (flx <= FHIGH)]
    tm = tmx[:, (fmx >= FLOW) & (fmx <= FHIGH)]
    wm = wmx[:, (fmx >= FLOW) & (fmx <= FHIGH)]
    il_0 = 1597
    il_2 = 1787
    il_4 = 1976
    il_6 = 2166
    il_8 = 77
    il_10 = 267
    il_12 = 457
    il_14 = 647
    il_16 = 837
    il_18 = 1027
    il_20 = 1217
    il_22 = 1407
    im_0 = 1610
    im_2 = 1793
    im_4 = 1977
    im_6 = 2161
    im_8 = 138
    im_10 = 322
    im_12 = 505
    im_14 = 689
    im_16 = 873
    im_18 = 1057
    im_20 = 1241
    im_22 = 1425
    il = [il_0, il_2, il_4, il_6, il_8, il_10, il_12, il_14, il_16, il_18, il_20, il_22]
    im = [im_0, im_2, im_4, im_6, im_8, im_10, im_12, im_14, im_16, im_18, im_20, im_22]
    plt.figure(figsize=[4, 6])
    # gg = [0.5, 0.5, 0.5]
    gg = "c"
    dy = 500  # K
    lw = 1
    k = 0.1
    for i in range(12):
        tli = tl[il[i], :]

        tmi = tm[im[i], :]
        wmi = wm[im[i], :]

        if i % 2 == 0:
            plt.plot(fl[wmi > 0], (tmi - tli)[wmi > 0] - i * dy, color=gg, linewidth=lw)
        else:
            plt.plot(fl[wmi > 0], (tmi - tli)[wmi > 0] - i * dy, color=gg, linewidth=lw)

        plt.plot([50, 150], [-i * dy, -i * dy], "k")
        if i <= 4:
            plt.text(53.6, -i * dy - k * dy, str(2 * i) + " hr")
        else:
            plt.text(52.3, -i * dy - k * dy, str(2 * i) + " hr")
    plt.yticks([])
    plt.ylabel("GHA [" + str(dy) + " K per division]\n\n\n")
    plt.xlabel(r"$\nu$ [MHz]")
    plt.xlim([58, 102])
    plt.ylim([-12 * dy, dy])
    # Saving
    plt.savefig(
        edges_folder + "/plots/20190612/comparison_mid_low2.pdf", bbox_inches="tight"
    )


def plot_beam_power():
    bm_all = beams.feko_blade_beam(
        "mid_band", 0, frequency_interpolation="no", AZ_antenna_axis=90
    )
    f = np.arange(50, 201, 2)
    el = np.arange(0, 91)
    sin_theta = np.sin((90 - el) * (np.pi / 180))
    sin_theta_2D_T = np.tile(sin_theta, (360, 1))
    sin_theta_2D = sin_theta_2D_T.T
    # normalization  = 1
    sm = np.zeros(len(f))
    for i in range(len(f)):
        bm = bm_all[i, :, :]
        normalization = np.max(np.max(bm))

        nb = bm / normalization

        s_sq_deg = np.sum(nb * sin_theta_2D)
        sm[i] = s_sq_deg / ((180 / np.pi) ** 2)
    bm_all = beams.feko_blade_beam(
        "mid_band", 1, frequency_interpolation="no", AZ_antenna_axis=90
    )
    f = np.arange(50, 201, 2)
    el = np.arange(0, 91)
    sin_theta = np.sin((90 - el) * (np.pi / 180))
    sin_theta_2D_T = np.tile(sin_theta, (360, 1))
    sin_theta_2D = sin_theta_2D_T.T
    # normalization  = 1
    smi = np.zeros(len(f))
    for i in range(len(f)):
        bm = bm_all[i, :, :]
        normalization = np.max(np.max(bm))

        nb = bm / normalization

        s_sq_deg = np.sum(nb * sin_theta_2D)
        smi[i] = s_sq_deg / ((180 / np.pi) ** 2)
    bmlb_all = beams.FEKO_low_band_blade_beam(
        beam_file=2, frequency_interpolation="no", AZ_antenna_axis=0
    )
    fl = np.arange(40, 121, 2)
    sl = np.zeros(len(fl))
    for i in range(len(fl)):
        bmlb = bmlb_all[i, :, :]
        normalization = np.max(np.max(bmlb))

        nb = bmlb / normalization

        s_sq_deg = np.sum(nb * sin_theta_2D)
        sl[i] = s_sq_deg / ((180 / np.pi) ** 2)
    fx = f[(f >= 50) & (f <= 110)]
    smx = sm[(f >= 50) & (f <= 110)]
    smix = smi[(f >= 50) & (f <= 110)]
    slx = sl[(fl >= 50) & (fl <= 110)]
    pm = np.polyfit(fx, smx, 4)
    mm = np.polyval(pm, fx)
    pmi = np.polyfit(fx, smix, 4)
    mmi = np.polyval(pmi, fx)
    pl = np.polyfit(fx, slx, 4)
    ml = np.polyval(pl, fx)
    plt.close()
    plt.figure(1)
    plt.subplot(2, 1, 1)
    plt.plot(fx, smx)
    plt.plot(fx, smix, ":")
    plt.plot(fx, slx, "--")
    plt.ylabel("solid angle of\n beam above horizon [sr]")
    plt.legend(
        [
            "Mid-Band 30mx30m ground plane",
            "Mid-Band infinite ground plane",
            "Low-Band 30mx30m ground plane",
        ]
    )
    plt.subplot(2, 1, 2)
    plt.plot(fx, smx - mm)
    plt.plot(fx, smix - mmi, ":")
    plt.plot(fx, slx - ml, "--")
    plt.ylabel("residuals to\n 5-term polynomial [sr]")
    plt.xlabel("frequency [MHz]")
    bm_all = beams.feko_blade_beam(
        "mid_band", 0, frequency_interpolation="no", AZ_antenna_axis=90
    )
    f = np.arange(50, 201, 2)
    el = np.arange(0, 91)
    sin_theta = np.sin((90 - el) * (np.pi / 180))
    sin_theta_2D_T = np.tile(sin_theta, (360, 1))
    sin_theta_2D = sin_theta_2D_T.T
    # normalization  = 1
    sm = np.zeros(len(f))
    for i in range(len(f)):
        bm = bm_all[i, :, :]
        normalization = 1  # np.max(np.max(bm))

        nb = bm / normalization

        s_sq_deg = np.sum(nb * sin_theta_2D)
        sm[i] = s_sq_deg / ((180 / np.pi) ** 2)
    bm_all = beams.feko_blade_beam(
        "mid_band", 1, frequency_interpolation="no", AZ_antenna_axis=90
    )
    f = np.arange(50, 201, 2)
    el = np.arange(0, 91)
    sin_theta = np.sin((90 - el) * (np.pi / 180))
    sin_theta_2D_T = np.tile(sin_theta, (360, 1))
    sin_theta_2D = sin_theta_2D_T.T
    # normalization  = 1
    smi = np.zeros(len(f))
    for i in range(len(f)):
        bm = bm_all[i, :, :]
        normalization = 1  # np.max(np.max(bm))

        nb = bm / normalization

        s_sq_deg = np.sum(nb * sin_theta_2D)
        smi[i] = s_sq_deg / ((180 / np.pi) ** 2)
    bmlb_all = beams.FEKO_low_band_blade_beam(
        beam_file=2, frequency_interpolation="no", AZ_antenna_axis=0
    )
    fl = np.arange(40, 121, 2)
    sl = np.zeros(len(fl))
    for i in range(len(fl)):
        bmlb = bmlb_all[i, :, :]
        normalization = 1  # np.max(np.max(bmlb))

        nb = bmlb / normalization

        s_sq_deg = np.sum(nb * sin_theta_2D)
        sl[i] = s_sq_deg / ((180 / np.pi) ** 2)
    fx = f[(f >= 50) & (f <= 110)]
    smx = sm[(f >= 50) & (f <= 110)]
    smix = smi[(f >= 50) & (f <= 110)]
    slx = sl[(fl >= 50) & (fl <= 110)]
    # normalized total radiated power
    smx = smx / (4 * np.pi)
    smix = smix / (4 * np.pi)
    slx = slx / (4 * np.pi)
    pm = np.polyfit(fx, smx, 4)
    mm = np.polyval(pm, fx)
    pmi = np.polyfit(fx, smix, 4)
    mmi = np.polyval(pmi, fx)
    pl = np.polyfit(fx, slx, 4)
    ml = np.polyval(pl, fx)

    plt.figure(2)
    plt.subplot(2, 1, 1)
    plt.plot(fx, smx)
    plt.plot(fx, smix, ":")
    plt.plot(fx, slx, "--")
    plt.ylabel("normalized total radiated power\n above horizon [fraction of 4pi]")
    plt.legend(
        [
            "Mid-Band 30mx30m ground plane",
            "Mid-Band infinite ground plane",
            "Low-Band 30mx30m ground plane",
        ]
    )
    plt.subplot(2, 1, 2)
    plt.plot(fx, smx - mm)
    plt.plot(fx, smix - mmi, ":")
    plt.plot(fx, slx - ml, "--")
    plt.ylabel("residuals to 5-term polynomial\n [fraction of 4pi]")
    plt.xlabel("frequency [MHz]")


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
    beam_factor_filename = (
        "table_lores_mid_band_50-200MHz_90deg_alan1_haslam_gaussian_index_2.4_2.65_sigma_deg_8"
        ".5_reffreq_90MHz.hdf5"
    )
    f, lst, bf = beams.beam_factor_table_read(
        "/home/raul/DATA2/EDGES_vol2/mid_band/calibration/beam_factors/table/"
        + beam_factor_filename
    )
    bx2 = bf[IX]
    beam_factor_filename = (
        "table_lores_mid_band_50-200MHz_90deg_alan0_haslam_flat_index_2"
        ".56_reffreq_90MHz.hdf5"
    )
    f, lst, bf = beams.beam_factor_table_read(
        "/home/raul/DATA2/EDGES_vol2/mid_band/calibration/beam_factors/table/"
        + beam_factor_filename
    )
    bx3 = bf[IX]
    beam_factor_filename = (
        "table_lores_mid_band_50-200MHz_90deg_alan0_LW_gaussian_index_2"
        ".4_2.65_sigma_deg_8.5_reffreq_90MHz.hdf5"
    )
    f, lst, bf = beams.beam_factor_table_read(
        "/home/raul/DATA2/EDGES_vol2/mid_band/calibration/beam_factors/table/"
        + beam_factor_filename
    )
    bx4 = bf[IX]
    beam_factor_filename = (
        "table_lores_mid_band_50-200MHz_90deg_alan0_guzman_gaussian_index_2.4_2.65_sigma_deg_8"
        ".5_reffreq_90MHz.hdf5"
    )
    f, lst, bf = beams.beam_factor_table_read(
        "/home/raul/DATA2/EDGES_vol2/mid_band/calibration/beam_factors/table/"
        + beam_factor_filename
    )
    bx5 = bf[IX]

    f1 = plt.figure(num=1, figsize=(size_x, size_y))
    scale_max = 0.0043
    scale_min = -0.0043
    cmap = plt.cm.viridis
    rgba = cmap(0.0)
    cmap.set_under(rgba)
    ax = f1.add_axes([x0 + 0 * (xoff + dx), y0, dx, dy])
    ax.imshow(
        bx2 - bx1,
        interpolation="none",
        extent=[50, 200, 24, 0],
        aspect="auto",
        vmin=scale_min,
        vmax=scale_max,
        cmap=cmap,
    )
    plt.plot([60, 120], [6, 6], "w--", linewidth=2)
    plt.plot([60, 120], [18, 18], "w--", linewidth=2)
    ax.set_xlim([60, 120])
    ax.set_xticks(np.arange(60, 121, 10))
    ax.set_yticks(np.arange(0, 25, 3))
    ax.set_xlabel(r"$\nu$ [MHz]", fontsize=14)
    ax.set_ylabel("GHA [hr]", fontsize=14)
    ax.set_title("(a)", fontsize=18)
    ax = f1.add_axes([x0 + 1 * (xoff + dx), y0, dx, dy])
    ax.imshow(
        bx3 - bx1,
        interpolation="none",
        extent=[50, 200, 24, 0],
        aspect="auto",
        vmin=scale_min,
        vmax=scale_max,
        cmap=cmap,
    )  # , cmap='jet')
    plt.plot([60, 120], [6, 6], "w--", linewidth=2)
    plt.plot([60, 120], [18, 18], "w--", linewidth=2)
    ax.set_xlim([60, 120])
    ax.set_yticklabels("")
    ax.set_xlabel(r"$\nu$ [MHz]", fontsize=14)
    ax.set_xticks(np.arange(60, 121, 10))
    ax.set_yticks(np.arange(0, 25, 3))
    ax.set_title("(b)", fontsize=18)
    ax = f1.add_axes([x0 + 2 * (xoff + dx), y0, dx, dy])
    ax.imshow(
        bx4 - bx1,
        interpolation="none",
        extent=[50, 200, 24, 0],
        aspect="auto",
        vmin=scale_min,
        vmax=scale_max,
        cmap=cmap,
    )  # , cmap='jet')
    plt.plot([60, 120], [6, 6], "w--", linewidth=2)
    plt.plot([60, 120], [18, 18], "w--", linewidth=2)
    ax.set_xlim([60, 120])
    ax.set_yticklabels("")
    ax.set_xlabel(r"$\nu$ [MHz]", fontsize=14)
    ax.set_xticks(np.arange(60, 121, 10))
    ax.set_yticks(np.arange(0, 25, 3))
    ax.set_title("(c)", fontsize=18)
    ax = f1.add_axes([x0 + 3 * (xoff + dx), y0, dx, dy])
    im = ax.imshow(
        bx5 - bx1,
        interpolation="none",
        extent=[50, 200, 24, 0],
        aspect="auto",
        vmin=scale_min,
        vmax=scale_max,
        cmap=cmap,
    )  # , cmap='jet')
    plt.plot([60, 120], [6, 6], "w--", linewidth=2)
    plt.plot([60, 120], [18, 18], "w--", linewidth=2)
    cax = f1.add_axes([x0 + 3.2 * xoff + 4 * dx, y0, dxc, dy])  # + xoffc
    f1.colorbar(im, cax=cax, orientation="vertical")
    cax.set_title(r"$\Delta C$", fontsize=14)
    ax.set_xlim([60, 120])
    ax.set_yticklabels("")
    ax.set_xlabel(r"$\nu$ [MHz]", fontsize=14)
    ax.set_xticks(np.arange(60, 121, 10))
    ax.set_yticks(np.arange(0, 25, 3))
    ax.set_title("(d)", fontsize=18)
    # Saving plot
    path_plot_save = edges_folder + "/plots/20190729/"
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
    print(gha[IX][175])

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
    path_plot_save = edges_folder + "plots/20190729/"
    plt.savefig(
        path_plot_save + "beam_chromaticity_correction.pdf", bbox_inches="tight"
    )


def plot_ground_loss():
    # TODO: the following paragraph is all made up!
    f1 = plt.figure()
    x0 = 0
    dx = 0
    xoff = 0
    y0 = 0
    dy = 0
    fe = np.linspace(0, 200, 2000)

    ax = f1.add_axes([x0 + 1 * dx + xoff, y0 + 1 * dy, dx, dy])
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


def plot_beam_gain(path_plot_save):
    fmin = 50
    fmax = 120
    fmin_res = 50
    fmax_res = 120
    Nfg = 5
    el = np.arange(0, 91)
    sin_theta = np.sin((90 - el) * (np.pi / 180))
    sin_theta_2D_T = np.tile(sin_theta, (360, 1))
    sin_theta_2D = sin_theta_2D_T.T
    b_all = beams.feko_blade_beam(
        "mid_band", 0, frequency_interpolation="no", AZ_antenna_axis=90
    )
    f = np.arange(50, 201, 2)
    bint = np.zeros(len(f))
    for i in range(len(f)):
        b = b_all[i, :, :]
        bint[i] = np.sum(b * sin_theta_2D)
    fr1 = f[(f >= fmin) & (f <= fmax)]
    bx1 = bint[(f >= fmin) & (f <= fmax)]
    b1 = bx1 / np.mean(bx1)
    ff1 = fr1[(fr1 >= fmin_res) & (fr1 <= fmax_res)]
    bb1 = b1[(fr1 >= fmin_res) & (fr1 <= fmax_res)]
    x = np.polyfit(ff1, bb1, Nfg - 1)
    m = np.polyval(x, ff1)
    r1 = bb1 - m
    b_all = beams.feko_blade_beam(
        "mid_band", 1, frequency_interpolation="no", AZ_antenna_axis=90
    )
    f = np.arange(50, 201, 2)
    bint = np.zeros(len(f))
    for i in range(len(f)):
        b = b_all[i, :, :]
        bint[i] = np.sum(b * sin_theta_2D)
    fr2 = f[(f >= fmin) & (f <= fmax)]
    bb = bint[(f >= fmin) & (f <= fmax)]
    b2 = bb / np.mean(bb)
    ff2 = fr2[(fr2 >= fmin_res) & (fr2 <= fmax_res)]
    bb2 = b2[(fr2 >= fmin_res) & (fr2 <= fmax_res)]
    x = np.polyfit(ff2, bb2, Nfg - 1)
    m = np.polyval(x, ff2)
    r2 = bb2 - m
    b_all = beams.FEKO_low_band_blade_beam(
        beam_file=2, frequency_interpolation="no", AZ_antenna_axis=0
    )
    f = np.arange(40, 121, 2)
    bint = np.zeros(len(f))
    for i in range(len(f)):
        b = b_all[i, :, :]
        bint[i] = np.sum(b * sin_theta_2D)
    fr3 = f[(f >= fmin) & (f <= fmax)]
    bx3 = bint[(f >= fmin) & (f <= fmax)]
    b3 = bx3 / np.mean(bx3)
    ff3 = fr3[(fr3 >= fmin_res) & (fr3 <= fmax_res)]
    bb3 = b3[(fr3 >= fmin_res) & (fr3 <= fmax_res)]
    x = np.polyfit(ff3, bb3, 4)
    m = np.polyval(x, ff3)
    r3 = bb3 - m
    b_all = beams.FEKO_low_band_blade_beam(
        beam_file=5,
        frequency_interpolation="no",
        frequency=np.array([0]),
        AZ_antenna_axis=0,
    )
    f = np.arange(50, 121, 2)
    bint = np.zeros(len(f))
    for i in range(len(f)):
        b = b_all[i, :, :]
        bint[i] = np.sum(b * sin_theta_2D)
    fr4 = f[(f >= fmin) & (f <= fmax)]
    bb = bint[(f >= fmin) & (f <= fmax)]
    b4 = bb / np.mean(bb)
    ff4 = fr4[(fr4 >= fmin_res) & (fr4 <= fmax_res)]
    bb4 = b4[(fr4 >= fmin_res) & (fr4 <= fmax_res)]
    x = np.polyfit(ff4, bb4, 4)
    m = np.polyval(x, ff4)
    r4 = bb4 - m

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
    # Paths
    path_plot_save = edges_folder + "plots/20200108/"
    # Plot
    size_x = 8.2
    size_y = 6
    x0 = 0.15
    y0 = 0.09
    dx = 0.45
    dy = 0.4
    xoff = x0 / 1.3
    f1 = plt.figure(num=1, figsize=(size_x, size_y))
    flow = 50
    fhigh = 120
    bm = beams.feko_blade_beam(
        "mid_band",
        0,
        frequency_interpolation="no",
        frequency=np.array([0]),
        AZ_antenna_axis=0,
    )
    ff = np.arange(50, 201, 2)
    fe = ff[(ff >= flow) & (ff <= fhigh)]
    g_zenith = bm[:, 90, 0][(ff >= flow) & (ff <= fhigh)]
    g_45_E = bm[:, 45, 0][(ff >= flow) & (ff <= fhigh)]
    g_45_H = bm[:, 45, 90][(ff >= flow) & (ff <= fhigh)]
    bm_inf = beams.feko_blade_beam(
        "mid_band",
        1,
        frequency_interpolation="no",
        frequency=np.array([0]),
        AZ_antenna_axis=0,
    )
    ff = np.arange(50, 201, 2)
    g_inf_zenith = bm_inf[:, 90, 0][(ff >= flow) & (ff <= fhigh)]
    g_inf_45_E = bm_inf[:, 45, 0][(ff >= flow) & (ff <= fhigh)]
    g_inf_45_H = bm_inf[:, 45, 90][(ff >= flow) & (ff <= fhigh)]
    bm_lb = beams.FEKO_low_band_blade_beam(
        beam_file=2,
        frequency_interpolation="no",
        frequency=np.array([0]),
        AZ_antenna_axis=0,
    )
    fcuec = np.arange(40, 121, 2)
    flb = fcuec[(fcuec >= flow) & (fcuec <= fhigh)]
    glb_zenith = bm_lb[:, 90, 0][(fcuec >= flow) & (fcuec <= fhigh)]
    glb_45_E = bm_lb[:, 45, 0][(fcuec >= flow) & (fcuec <= fhigh)]
    glb_45_H = bm_lb[:, 45, 90][(fcuec >= flow) & (fcuec <= fhigh)]
    bm_10 = beams.FEKO_low_band_blade_beam(
        beam_file=5,
        frequency_interpolation="no",
        frequency=np.array([0]),
        AZ_antenna_axis=0,
    )
    fcuec = np.arange(50, 121, 2)
    f10 = fcuec[(fcuec >= flow) & (fcuec <= fhigh)]
    g10_zenith = bm_10[:, 90, 0][(fcuec >= flow) & (fcuec <= fhigh)]
    g10_45_E = bm_10[:, 45, 0][(fcuec >= flow) & (fcuec <= fhigh)]
    g10_45_H = bm_10[:, 45, 90][(fcuec >= flow) & (fcuec <= fhigh)]
    print(glb_zenith[-1])
    print(flb[-1])
    print(g10_zenith[-1])
    print(f10[-1])
    ax = f1.add_axes([x0, y0 + 1 * dy, dx, dy])
    ax.plot(fe, g_zenith, "b", linewidth=1.0, label="")
    ax.plot(fe, g_inf_zenith, "b--", linewidth=0.5, label="")
    ax.plot(flb, glb_zenith, "r", linewidth=1.0, label="")
    ax.plot(f10, g10_zenith, "r--", linewidth=0.5, label="")
    ax.plot(fe, g_zenith, "b", linewidth=1.0, label="")
    ax.plot(fe, g_45_E, "b", linewidth=1.0, label="")
    ax.plot(fe, g_45_H, "b", linewidth=1.0, label="")
    ax.plot(fe, g_inf_zenith, "b--", linewidth=0.5, label="")
    ax.plot(fe, g_inf_45_E, "b--", linewidth=0.5, label="")
    ax.plot(fe, g_inf_45_H, "b--", linewidth=0.5, label="")
    ax.plot(flb, glb_zenith, "r", linewidth=1.0, label="")
    ax.plot(flb, glb_45_E, "r", linewidth=1.0, label="")
    ax.plot(flb, glb_45_H, "r", linewidth=1.0, label="")
    ax.plot(f10, g10_zenith, "r--", linewidth=0.5, label="")
    ax.plot(f10, g10_45_E, "r--", linewidth=0.5, label="")
    ax.plot(f10, g10_45_H, "r--", linewidth=0.5, label="")
    ax.set_xticklabels("")
    ax.set_ylabel("gain")
    ax.set_xlim([48, 122])
    ax.tick_params(axis="x", direction="in")
    ax.set_xticks(np.arange(50, 121, 10))
    ax.set_ylim([0, 9])
    ax.set_yticks(np.arange(1, 8.1, 1))
    ax.text(115, 0.4, "(a)", fontsize=14)
    ax.text(50, 5.7, "zenith", fontsize=10)
    ax.text(50, 2.9, r"$\theta=45^{\circ}$, H-plane", fontsize=10)
    ax.text(50, 0.9, r"$\theta=45^{\circ}$, E-plane", fontsize=10)
    ax.legend(
        [
            r"Mid-Band 30m x 30m GP",
            "Mid-Band infinite GP",
            r"Low-Band 30m x 30m GP",
            r"Low-Band 10m x 10m GP",
        ],
        fontsize=7,
        ncol=2,
    )
    Nfg = 4
    p1 = np.polyfit(fe, g_zenith, Nfg)
    p2 = np.polyfit(fe, g_45_E, Nfg)
    p3 = np.polyfit(fe, g_45_H, Nfg)
    m1 = np.polyval(p1, fe)
    m2 = np.polyval(p2, fe)
    m3 = np.polyval(p3, fe)
    pi1 = np.polyfit(fe, g_inf_zenith, Nfg)
    pi2 = np.polyfit(fe, g_inf_45_E, Nfg)
    pi3 = np.polyfit(fe, g_inf_45_H, Nfg)
    mi1 = np.polyval(pi1, fe)
    mi2 = np.polyval(pi2, fe)
    mi3 = np.polyval(pi3, fe)
    p1 = np.polyfit(flb, glb_zenith, Nfg)
    p2 = np.polyfit(flb, glb_45_E, Nfg)
    p3 = np.polyfit(flb, glb_45_H, Nfg)
    mlb1 = np.polyval(p1, flb)
    mlb2 = np.polyval(p2, flb)
    mlb3 = np.polyval(p3, flb)
    p1 = np.polyfit(f10, g10_zenith, Nfg)
    p2 = np.polyfit(f10, g10_45_E, Nfg)
    p3 = np.polyfit(f10, g10_45_H, Nfg)
    m10_1 = np.polyval(p1, f10)
    m10_2 = np.polyval(p2, f10)
    m10_3 = np.polyval(p3, f10)
    ax = f1.add_axes([x0, y0 + 0 * dy, dx, dy])
    DY = 0.08
    ax.plot(fe, g_zenith - m1 + DY, "b", linewidth=1.0, label=r"$T_{\mathrm{unc}}$")
    ax.plot(fe, g_45_E - m2 - DY, "b", linewidth=1.0, label=r"$T_{\mathrm{unc}}$")
    ax.plot(fe, g_45_H - m3 - 0.0, "b", linewidth=1.0, label=r"$T_{\mathrm{unc}}$")
    ax.plot(
        fe, g_inf_zenith - mi1 + DY, "b--", linewidth=0.5, label=r"$T_{\mathrm{unc}}$"
    )
    ax.plot(
        fe, g_inf_45_E - mi2 - DY, "b--", linewidth=0.5, label=r"$T_{\mathrm{unc}}$"
    )
    ax.plot(
        fe, g_inf_45_H - mi3 - 0.0, "b--", linewidth=0.5, label=r"$T_{\mathrm{unc}}$"
    )
    ax.plot(
        flb, glb_zenith - mlb1 + DY, "r", linewidth=1.0, label=r"$T_{\mathrm{unc}}$"
    )
    ax.plot(flb, glb_45_E - mlb2 - DY, "r", linewidth=1.0, label=r"$T_{\mathrm{unc}}$")
    ax.plot(flb, glb_45_H - mlb3 - 0.0, "r", linewidth=1.0, label=r"$T_{\mathrm{unc}}$")
    ax.plot(
        f10, g10_zenith - m10_1 + DY, "r--", linewidth=0.5, label=r"$T_{\mathrm{unc}}$"
    )
    ax.plot(
        f10, g10_45_E - m10_2 - DY, "r--", linewidth=0.5, label=r"$T_{\mathrm{unc}}$"
    )
    ax.plot(f10, g10_45_H - m10_3 - 0.0, "r--", linewidth=0.5, el=r"$T_{\mathrm{unc}}$")
    ax.set_xlabel(r"$\nu$ [MHz]")  # , fontsize=15)
    ax.set_ylabel(
        "gain residuals\n[" + str(DY / 2) + " per division]"
    )  # , fontsize=14)
    ax.set_xlim([48, 122])
    xt = np.arange(50, 121, 10)
    ax.set_xticks(xt)
    yt = np.arange(-1.5 * DY, 1.5 * DY + 0.0001, DY / 2)
    ax.set_ylim([-0.135, 0.135])
    ax.set_yticks(yt)
    ax.set_yticklabels([""] * len(yt))
    ax.text(115, -0.123, "(b)", fontsize=14)
    ax.text(50, 0.035, "zenith", fontsize=10)
    ax.text(50, -0.03, r"$\theta=45^{\circ}$, H-plane", fontsize=10)
    ax.text(50, -0.12, r"$\theta=45^{\circ}$, E-plane", fontsize=10)
    ax = f1.add_axes([x0 + 1 * dx + xoff, y0 + 1 * dy, dx, dy])
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
    ax = f1.add_axes([x0 + 1 * dx + xoff, y0 + 0 * dy, dx, dy])
    x1 = (1 - Gg) * 100
    x2 = (1 - Gglb) * 100
    p1 = np.polyfit(fe, x1, Nfg)
    p2flb = np.polyfit(flb, x2, Nfg)
    m1 = np.polyval(p1, fe)
    m2 = np.polyval(p2flb)
    ax.plot(fe, x1 - m1, "b", linewidth=1)
    ax.plot(flb, x2 - m2, "r", linewidth=1)
    ax.set_xticks(xt)
    ax.set_ylim([-0.006, 0.006])
    ax.set_yticks(np.arange(-0.004, 0.0041, 0.002))
    ax.set_ylabel(r"ground loss residuals [%]")
    ax.set_xlabel(r"$\nu$ [MHz]")
    ax.text(115, -0.0055, "(d)", fontsize=14)
    plt.savefig(path_plot_save + "beam_gain.pdf", bbox_inches="tight")
    return dx, dy, f1, fe, path_plot_save, x0, xoff, y0


def plot_balun_loss2(s11_path):
    # Paths
    path_plot_save = edges_folder + "plots/20190917/"
    # Plot
    size_x = 4.5
    size_y = 2.7  # 10.5
    x0 = 0.15
    y0 = 0.09
    dx = 0.8
    dy = 0.8  # 18
    f1 = plt.figure(num=1, figsize=(size_x, size_y))
    # Frequency
    fe = EdgesFrequencyRange(f_low=50, f_high=130).freq
    # Antenna S11
    # -----------
    ra = s11m.antenna_s11_remove_delay(
        s11_path, fe, delay_0=0.17, model_type="polynomial", Nfit=15
    )
    xlb1 = np.genfromtxt(
        "/run/media/raul/SSD_4TB/EDGES_vol1/calibration/antenna_s11/low_band1/s11/corrected"
        "/2016_243/S11_blade_low_band_2016_243.txt"
    )
    flb1 = xlb1[:, 0] / 1e6
    ralb1 = xlb1[:, 1] + 1j * xlb1[:, 2]
    xlb2 = np.genfromtxt(
        "/run/media/raul/SSD_4TB/EDGES_vol1/calibration/antenna_s11/low_band2/s11/corrected"
        "/2017-06-29-low2-noshield_average/S11_blade_low_band_2017_180_NO_SHIELD.txt"
    )
    flb2 = xlb2[:, 0] / 1e6
    ralb2 = xlb2[:, 1] + 1j * xlb2[:, 2]
    Gb, Gc = loss.balun_and_connector_loss("mid_band", fe, ra)
    Gbc = Gb * Gc
    Gblb, Gclb = loss.balun_and_connector_loss("low_band_2015", flb1, ralb1)
    Gbclb = Gblb * Gclb
    Gblb2, Gclb2 = loss.balun_and_connector_loss("low_band2_2017", flb2, ralb2)
    Gbclb2 = Gblb2 * Gclb2
    ax = f1.add_axes([x0, y0 + 0 * dy, dx, dy])
    ax.plot(fe, (1 - Gbc) * 100, "b", linewidth=1.3, label="")
    ax.plot(
        flb1[flb1 <= 100], (1 - Gbclb)[flb1 <= 100] * 100, "r", linewidth=1.3, label=""
    )
    ax.plot(
        flb1[flb1 <= 100],
        (1 - Gbclb2)[flb1 <= 100] * 100,
        "r--",
        linewidth=1.3,
        label="",
    )
    ax.set_ylabel(r"antenna loss [%]")
    ax.set_xlim([48, 132])
    ax.tick_params(axis="x", direction="in")
    ax.set_xticks(np.arange(50, 131, 10))
    ax.set_ylim([0, 1])
    ax.set_yticks(np.arange(0.2, 0.9, 0.2))
    # ax.text(114, 0.07, '(c)', fontsize=14)
    ax.set_xlabel("$\\nu$ [MHz]", fontsize=13)
    ax.legend(["Mid-Band", "Low-Band 1", "Low-Band 2"], fontsize=9)
    plt.savefig(path_plot_save + "balun_loss.pdf", bbox_inches="tight")
    return dx, dy, f1, fe, path_plot_save, x0, y0


def plot_antenna_calibration_params(s11_path):
    # Paths
    path_plot_save = edges_folder + "plots/20190917/"
    # Plot
    FS_LABELS = 12
    size_x = 4.5
    size_y = 5.5  # 10.5
    x0 = 0.15
    y0 = 0.09
    dx = 0.8
    dy = 0.4  # 18
    f1 = plt.figure(num=1, figsize=(size_x, size_y))
    # Frequency
    fe = EdgesFrequencyRange(f_low=50, f_high=130).freq
    # f, il, ih = ba.frequency_edges(50, 130)
    # fe = f[il : ih + 1]
    # Antenna S11
    # -----------
    ra = s11m.antenna_s11_remove_delay(
        s11_path, fe, delay_0=0.17, model_type="polynomial", Nfit=15
    )
    xlb1 = np.genfromtxt(
        "/run/media/raul/SSD_4TB/EDGES_vol1/calibration/antenna_s11/low_band1/s11/corrected"
        "/2016_243/S11_blade_low_band_2016_243.txt"
    )
    flb1 = xlb1[:, 0] / 1e6
    ralb1 = xlb1[:, 1] + 1j * xlb1[:, 2]
    xlb2 = np.genfromtxt(
        "/run/media/raul/SSD_4TB/EDGES_vol1/calibration/antenna_s11/low_band2/s11/corrected"
        "/2017-06-29-low2-noshield_average/S11_blade_low_band_2017_180_NO_SHIELD.txt"
    )
    flb2 = xlb2[:, 0] / 1e6
    ralb2 = xlb2[:, 1] + 1j * xlb2[:, 2]
    ax = f1.add_axes([x0, y0 + 1 * dy, dx, dy])
    ax.plot(fe, 20 * np.log10(np.abs(ra)), "b", linewidth=1.3, label="")
    ax.plot(
        flb1[flb1 <= 100],
        20 * np.log10(np.abs(ralb1[flb1 <= 100])),
        "r",
        linewidth=1.3,
        label="",
    )
    ax.plot(
        flb2[flb2 <= 100],
        20 * np.log10(np.abs(ralb2[flb2 <= 100])),
        "r--",
        linewidth=1.3,
        label="",
    )
    ax.legend(["Mid-Band", "Low-Band 1", "Low-Band 2"], fontsize=9)
    ax.set_xticklabels("")
    ax.set_ylabel(r"$|\Gamma_{\mathrm{ant}}|$ [dB]", fontsize=FS_LABELS)
    ax.set_xlim([48, 132])
    ax.set_ylim([-17, -1])
    ax.set_yticks(np.arange(-16, -1, 2))
    ax.tick_params(axis="x", direction="in")
    ax.set_xticks(np.arange(50, 131, 10))
    ax.text(122, -15.6, "(a)", fontsize=14)
    ax = f1.add_axes([x0, y0 + 0 * dy, dx, dy])
    ax.plot(fe, (180 / np.pi) * np.unwrap(np.angle(ra)), "b", linewidth=1.3, label=r"")
    ax.plot(
        flb1[flb1 <= 100],
        (180 / np.pi) * np.unwrap(np.angle(ralb1[flb1 <= 100])),
        "r",
        linewidth=1.3,
        label=r"",
    )
    ax.plot(
        flb2[flb2 <= 100],
        (180 / np.pi) * np.unwrap(np.angle(ralb2[flb2 <= 100])),
        "r--",
        linewidth=1.3,
        label=r"",
    )
    ax.set_ylabel(
        r"$\angle\/\Gamma_{\mathrm{ant}}$ [ $^\mathrm{o}$]", fontsize=FS_LABELS
    )
    ax.set_xlim([48, 132])
    ax.tick_params(axis="x", direction="in")
    ax.set_xticks(np.arange(50, 131, 10))
    ax.set_ylim([-700, 300])
    ax.set_yticks(np.arange(-600, 201, 200))
    ax.text(122, -620, "(b)", fontsize=14)
    ax.set_xlabel(r"$\nu$ [MHz]", fontsize=13)
    plt.savefig(
        path_plot_save + "antenna_reflection_coefficients.pdf", bbox_inches="tight"
    )


def plot_receiver_calibration_params():
    # Paths
    path_plot_save = edges_folder + "plots/20190917/"

    # Calibration parameters
    rcv_file = (
        edges_folder + "mid_band/calibration/receiver_calibration/receiver1"
        "/2018_01_25C/results/nominal/calibration_files"
        "/calibration_file_receiver1_cterms7_wterms8.txt"
    )

    # mid_band/calibration/receiver_calibration/receiver1/2018_01_25C/results/nominal
    # /calibration_files/calibration_file_receiver1_50_150MHz_cterms7_wterms7.txt'

    rcv = np.genfromtxt(rcv_file)

    FLOW = 50
    FHIGH = 130

    fX = rcv[:, 0]
    rcv2 = rcv[(fX >= FLOW) & (fX <= FHIGH), :]

    fe = rcv2[:, 0]
    rl = rcv2[:, 1] + 1j * rcv2[:, 2]
    sca = rcv2[:, 3]
    off = rcv2[:, 4]
    TU = rcv2[:, 5]
    TC = rcv2[:, 6]
    TS = rcv2[:, 7]

    # Low-Band
    rcv1 = np.genfromtxt(
        "/home/raul/DATA1/EDGES_vol1/calibration/receiver_calibration/low_band1/2015_08_25C"
        "/results/nominal/calibration_files/calibration_file_low_band_2015_nominal.txt"
    )

    # Plot

    size_x = 4.5
    size_y = 7.5  # 10.5
    x0 = 0.15
    y0 = 0.09
    dx = 0.7
    dy = 0.3

    FS_LABELS = 12
    FS_PANELS = 14

    f1 = plt.figure(num=1, figsize=(size_x, size_y))

    ax = f1.add_axes([x0, y0 + 2 * dy, dx, dy])
    h1 = ax.plot(
        fe,
        20 * np.log10(np.abs(rl)),
        "b",
        linewidth=1.3,
        label=r"$|\Gamma_{\mathrm{rec}}|$",
    )
    ax.plot(
        rcv1[:, 0],
        20 * np.log10(np.abs(rcv1[:, 1] + 1j * rcv1[:, 2])),
        "r",
        linewidth=1.3,
    )

    ax2 = ax.twinx()
    h2 = ax2.plot(
        fe,
        (180 / np.pi) * np.unwrap(np.angle(rl)),
        "b--",
        linewidth=1.3,
        label=r"$\angle\/\Gamma_{\mathrm{rec}}$",
    )
    ax2.plot(
        rcv1[:, 0],
        (180 / np.pi) * np.unwrap(np.angle(rcv1[:, 1] + 1j * rcv1[:, 2])),
        "r--",
        linewidth=1.3,
    )

    h = h1 + h2
    labels = [l.get_label() for l in h]
    ax.legend(h, labels, loc=0, fontsize=10, ncol=2)

    ax.set_ylim([-40, -30])
    ax.set_xticklabels("")
    ax.set_yticks(np.arange(-39, -30, 2))
    ax.set_ylabel(r"$|\Gamma_{\mathrm{rec}}|$ [dB]", fontsize=FS_LABELS)
    ax.text(48.5, -39.55, "(a)", fontsize=FS_PANELS)
    ax.text(105, -38.9, "Mid-Band", fontweight="bold", color="b")
    ax.text(105, -39.5, "Low-Band 1", fontweight="bold", color="r")

    ax2.set_ylim([70, 130])
    ax2.set_xticklabels("")
    ax2.set_yticks(np.arange(80, 121, 10))
    ax2.set_ylabel(
        r"$\angle\/\Gamma_{\mathrm{rec}}$ [ $^\mathrm{o}$]", fontsize=FS_LABELS
    )

    ax.set_xlim([48, 132])
    ax.tick_params(axis="x", direction="in")
    ax.set_xticks(np.arange(50, 131, 10))

    ax = f1.add_axes([x0, y0 + 1 * dy, dx, dy])
    h1 = ax.plot(fe, sca, "b", linewidth=1.3, label="$C_1$")
    ax.plot(
        rcv1[:, 0], rcv1[:, 3], "r", linewidth=1.3
    )  # <----------------------------- Low-Band
    ax2 = ax.twinx()
    h2 = ax2.plot(fe, off, "b--", linewidth=1.3, label="$C_2$")
    ax2.plot(
        rcv1[:, 0], rcv1[:, 4], "r--", linewidth=1.3
    )  # <-----------------------------
    # Low-Band
    h = h1 + h2
    labels = [l.get_label() for l in h]
    ax.legend(h, labels, loc=0, fontsize=10, ncol=2)

    ax.set_ylim([3.3, 5.2])
    ax.set_xticklabels("")
    ax.set_yticks(np.arange(3.5, 5.1, 0.5))
    ax.set_ylabel("$C_1$", fontsize=FS_LABELS)
    ax.text(48.5, 3.38, "(b)", fontsize=FS_PANELS)

    ax2.set_ylim([-2.75, -0.75])
    ax2.set_xticklabels("")
    ax2.set_yticks(np.arange(-2.5, -0.85, 0.5))
    ax2.set_ylabel("$C_2$ [K]", fontsize=FS_LABELS)

    ax.set_xlim([48, 132])
    ax.tick_params(axis="x", direction="in")
    ax.set_xticks(np.arange(50, 131, 10))

    ax = f1.add_axes([x0, y0 + 0 * dy, dx, dy])
    h1 = ax.plot(fe, TU, "b", linewidth=1.3, label=r"$T_{\mathrm{U}}$")
    ax.plot(rcv1[:, 0], rcv1[:, 5], "r", linewidth=1.3)

    ax2 = ax.twinx()
    h2 = ax2.plot(fe, TC, "b--", linewidth=1.3, label=r"$T_{\mathrm{C}}$")
    ax2.plot(rcv1[:, 0], rcv1[:, 6], "r--", linewidth=1.3)

    h3 = ax2.plot(fe, TS, "b:", linewidth=1.3, label=r"$T_{\mathrm{S}}$")
    ax2.plot(rcv1[:, 0], rcv1[:, 7], "r:", linewidth=1.3)

    h = h1 + h2 + h3
    labels = [l.get_label() for l in h]
    ax.legend(h, labels, loc=0, fontsize=10, ncol=3)

    ax.set_ylim([178, 190])
    ax.set_yticks(np.arange(180, 189, 2))
    ax.set_ylabel(r"$T_{\mathrm{U}}$ [K]", fontsize=FS_LABELS)
    ax.set_xlabel(r"$\nu$ [MHz]", fontsize=FS_LABELS)
    ax.text(48.5, 178.5, "(c)", fontsize=FS_PANELS)

    ax2.set_ylim([-55, 35])
    ax2.set_yticks(np.arange(-40, 21, 20))
    ax2.set_ylabel(r"$T_{\mathrm{C}}, T_{\mathrm{S}}$ [K]", fontsize=FS_LABELS)

    ax.set_xlim([48, 132])
    ax.set_xticks(np.arange(50, 131, 10))

    plt.savefig(path_plot_save + "receiver_calibration.pdf", bbox_inches="tight")


def plot_balun_loss(s11_path):

    # Paths
    path_plot_save = edges_folder + "plots/20200108/"

    # Plot
    size_x = 4.5
    size_y = 2.7  # 10.5
    plt.figure(num=1, figsize=(size_x, size_y))

    # Frequency
    fe = EdgesFrequencyRange(f_low=50, f_high=130).freq

    # Antenna S11
    # -----------
    ra = s11m.antenna_s11_remove_delay(
        s11_path, fe, delay_0=0.17, model_type="polynomial", Nfit=15
    )

    xlb1 = np.genfromtxt(
        "/run/media/raul/SSD_4TB/EDGES_vol1/calibration/antenna_s11/low_band1/s11/corrected"
        "/2016_243/S11_blade_low_band_2016_243.txt"
    )
    flb1 = xlb1[:, 0] / 1e6
    ralb1 = xlb1[:, 1] + 1j * xlb1[:, 2]

    xlb2 = np.genfromtxt(
        "/run/media/raul/SSD_4TB/EDGES_vol1/calibration/antenna_s11/low_band2/s11/corrected"
        "/2017-06-29-low2-noshield_average/S11_blade_low_band_2017_180_NO_SHIELD.txt"
    )
    flb2 = xlb2[:, 0] / 1e6
    ralb2 = xlb2[:, 1] + 1j * xlb2[:, 2]

    Gb, Gc = loss.balun_and_connector_loss("mid_band", fe, ra)
    Gbc = Gb * Gc

    Gblb, Gclb = loss.balun_and_connector_loss("low_band_2015", flb1, ralb1)
    Gbclb = Gblb * Gclb

    Gblb2, Gclb2 = loss.balun_and_connector_loss("low_band2_2017", flb2, ralb2)
    Gbclb2 = Gblb2 * Gclb2

    # Figure
    # ---------------------
    plt.figure(figsize=[6, 7])

    # Subplot 1
    # ---------
    plt.subplot(2, 1, 1)
    plt.plot(fe, (1 - Gbc) * 100, "b", linewidth=1.3, label="")
    plt.plot(
        flb1[flb1 <= 100], (1 - Gbclb)[flb1 <= 100] * 100, "r", linewidth=1.3, label=""
    )
    plt.plot(
        flb1[flb1 <= 100],
        (1 - Gbclb2)[flb1 <= 100] * 100,
        "r--",
        linewidth=1.3,
        label="",
    )
    plt.legend(["Mid-Band", "Low-Band 1", "Low-Band 2"], fontsize=9)

    plt.xlim([48, 132])
    plt.xticks(np.arange(50, 131, 10), "")

    plt.ylim([0, 1])
    plt.yticks(np.arange(0.0, 0.9, 0.2))
    plt.ylabel(r"balun loss [%]")

    plt.text(48.5, 0.05, "(a)", fontsize=15)

    Ga = loss.antenna_loss("mid_band", fe)

    # Subplot 2
    # ---------
    plt.subplot(2, 1, 2)
    plt.plot(fe, (1 - Ga) * 100, "b", linewidth=1.3)

    plt.xlim([48, 132])
    plt.xticks(np.arange(50, 131, 10))
    plt.xlabel(r"$\nu$ [MHz]", fontsize=13)

    plt.ylim([0, 0.12])
    plt.yticks(np.arange(0.0, 0.11, 0.02))
    plt.ylabel(r"antenna loss [%]")

    plt.text(48.5, 0.007, "(b)", fontsize=15)

    plt.savefig(path_plot_save + "loss.pdf", bbox_inches="tight")


def plot_calibration_parameters(s11_path):

    # Paths
    path_plot_save = edges_folder + "plots/20200108/"

    FS_LABELS = 12
    FS_PANELS = 14

    plt.close()
    plt.figure(figsize=[12, 12])

    # Antenna S11
    # -----------
    freq = EdgesFrequencyRange(f_low=50, f_high=130)
    fe = freq.freq

    ra = s11m.antenna_s11_remove_delay(
        s11_path,
        fe,
        delay_0=0.17,
        model_type="polynomial",
        Nfit=15,
        plot_fit_residuals="no",
    )

    xlb1 = np.genfromtxt(
        "/run/media/raul/SSD_4TB/EDGES_vol1/calibration/antenna_s11/low_band1/s11/corrected"
        "/2016_243/S11_blade_low_band_2016_243.txt"
    )
    flb1 = xlb1[:, 0] / 1e6
    ralb1 = xlb1[:, 1] + 1j * xlb1[:, 2]

    xlb2 = np.genfromtxt(
        "/run/media/raul/SSD_4TB/EDGES_vol1/calibration/antenna_s11/low_band2/s11/corrected"
        "/2017-06-29-low2-noshield_average/S11_blade_low_band_2017_180_NO_SHIELD.txt"
    )
    flb2 = xlb2[:, 0] / 1e6
    ralb2 = xlb2[:, 1] + 1j * xlb2[:, 2]

    # Subplot 1
    # ---------
    plt.subplot(4, 2, 1)
    plt.plot(fe, 20 * np.log10(np.abs(ra)), "b", linewidth=1.3, label="")
    plt.plot(
        flb1[flb1 <= 100],
        20 * np.log10(np.abs(ralb1[flb1 <= 100])),
        "r",
        linewidth=1.3,
        label="",
    )
    plt.plot(
        flb2[flb2 <= 100],
        20 * np.log10(np.abs(ralb2[flb2 <= 100])),
        "r--",
        linewidth=1.3,
        label="",
    )

    plt.ylabel(r"$|\Gamma_{\mathrm{ant}}|$ [dB]", fontsize=FS_LABELS)

    plt.xlim([48, 132])
    plt.ylim([-17, -1])
    plt.xticks(np.arange(50, 131, 10), "")
    plt.yticks(np.arange(-16, -1, 2))
    plt.text(48.5, -16.2, "(a)", fontsize=14)

    # Subplot 2
    # ---------
    plt.subplot(4, 2, 2)
    plt.plot(fe, (180 / np.pi) * np.unwrap(np.angle(ra)), "b", linewidth=1.3, label=r"")
    plt.plot(
        flb1[flb1 <= 100],
        (180 / np.pi) * np.unwrap(np.angle(ralb1[flb1 <= 100])),
        "r",
        linewidth=1.3,
        label=r"",
    )
    plt.plot(
        flb2[flb2 <= 100],
        (180 / np.pi) * np.unwrap(np.angle(ralb2[flb2 <= 100])),
        "r--",
        linewidth=1.3,
        label=r"",
    )

    plt.ylabel(r"$\angle\/\Gamma_{\mathrm{ant}}$ [ $^\mathrm{o}$]", fontsize=FS_LABELS)
    plt.legend(["Mid-Band", "Low-Band 1", "Low-Band 2"], fontsize=9)

    plt.xlim([48, 132])
    plt.ylim([-700, 300])
    plt.xticks(np.arange(50, 131, 10), "")
    plt.yticks(np.arange(-600, 201, 200))
    plt.text(48.5, -650, "(b)", fontsize=FS_PANELS)

    # Receiver calibration parameters
    # -------------------------------
    rcv_file = (
        edges_folder + "mid_band/calibration/receiver_calibration/receiver1"
        "/2018_01_25C/results/nominal/calibration_files"
        "/calibration_file_receiver1_cterms7_wterms8.txt"
    )

    rcv = np.genfromtxt(rcv_file)

    FLOW = 50
    FHIGH = 130

    fX = rcv[:, 0]
    rcv2 = rcv[(fX >= FLOW) & (fX <= FHIGH), :]

    fe = rcv2[:, 0]
    rl = rcv2[:, 1] + 1j * rcv2[:, 2]
    sca = rcv2[:, 3]
    off = rcv2[:, 4]
    TU = rcv2[:, 5]
    TC = rcv2[:, 6]
    TS = rcv2[:, 7]

    # Low-Band
    rcv1 = np.genfromtxt(
        "/home/raul/DATA1/EDGES_vol1/calibration/receiver_calibration/low_band1/2015_08_25C"
        "/results/nominal/calibration_files/calibration_file_low_band_2015_nominal.txt"
    )

    # Subplot 3
    # ---------
    plt.subplot(4, 2, 3)
    plt.plot(fe, 20 * np.log10(np.abs(rl)), "b", linewidth=1.3)
    plt.plot(rcv1[:, 0], 20 * np.log10(np.abs(rcv1[:, 1] + 1j * rcv1[:, 2])), "r")

    plt.xlim([48, 132])
    plt.ylim([-40, -30])
    plt.xticks(np.arange(50, 131, 10), "")
    plt.yticks(np.arange(-39, -30, 2))
    plt.ylabel(r"$|\Gamma_{\mathrm{rec}}|$ [dB]", fontsize=FS_LABELS)
    plt.text(48.5, -39.55, "(c)", fontsize=FS_PANELS)

    # Subplot 4
    # ---------
    plt.subplot(4, 2, 4)
    plt.plot(fe, (180 / np.pi) * np.unwrap(np.angle(rl)), "b", linewidth=1.3)
    plt.plot(
        rcv1[:, 0],
        (180 / np.pi) * np.unwrap(np.angle(rcv1[:, 1] + 1j * rcv1[:, 2])),
        "r",
        linewidth=1.3,
    )

    plt.xlim([48, 132])
    plt.ylim([70, 130])
    plt.xticks(np.arange(50, 131, 10), "")
    plt.yticks(np.arange(80, 121, 10))
    plt.ylabel(r"$\angle\/\Gamma_{\mathrm{rec}}$ [ $^\mathrm{o}$]", fontsize=FS_LABELS)
    plt.text(48.5, 73, "(d)", fontsize=FS_PANELS)

    # Subplot 5
    # ---------
    plt.subplot(4, 2, 5)
    plt.plot(fe, sca, "b", linewidth=1.3)
    plt.plot(rcv1[:, 0], rcv1[:, 3], "r", linewidth=1.3)

    plt.xlim([48, 132])
    plt.xticks(np.arange(50, 131, 10), "")
    plt.ylim([3.3, 5.2])
    plt.yticks(np.arange(3.5, 5.1, 0.5))
    plt.ylabel("$C_1$", fontsize=FS_LABELS)
    plt.text(48.5, 3.38, "(e)", fontsize=FS_PANELS)

    # Subplot 6
    # ---------
    plt.subplot(4, 2, 6)
    plt.plot(fe, off, "b", linewidth=1.3)
    plt.plot(rcv1[:, 0], rcv1[:, 4], "r", linewidth=1.3)

    plt.xlim([48, 132])
    plt.xticks(np.arange(50, 131, 10), "")
    plt.ylim([-2.75, -0.75])
    plt.yticks(np.arange(-2.5, -0.85, 0.5))
    plt.ylabel("$C_2$ [K]", fontsize=FS_LABELS)
    plt.text(48.5, -2.65, "(f)", fontsize=FS_PANELS)

    # Subplot 7
    # ---------
    plt.subplot(4, 2, 7)
    plt.plot(fe, TU, "b", linewidth=1.3)
    plt.plot(rcv1[:, 0], rcv1[:, 5], "r", linewidth=1.3)

    plt.xlim([48, 132])
    plt.xticks(np.arange(50, 131, 10))
    plt.ylim([178, 190])
    plt.yticks(np.arange(180, 189, 2))
    plt.ylabel(r"$T_{\mathrm{U}}$ [K]", fontsize=FS_LABELS)
    plt.xlabel(r"$\nu$ [MHz]", fontsize=FS_LABELS)
    plt.text(48.5, 178.5, "(g)", fontsize=FS_PANELS)

    # Subplot 8
    # ---------
    plt.subplot(4, 2, 8)
    plt.plot(fe, TC, "b", linewidth=1.3)
    plt.plot(rcv1[:, 0], rcv1[:, 6], "r", linewidth=1.3)

    plt.plot(fe, TS, "b", linewidth=1.3)
    plt.plot(rcv1[:, 0], rcv1[:, 7], "r", linewidth=1.3)

    plt.xlim([48, 132])
    plt.xticks(np.arange(50, 131, 10))
    plt.ylim([-55, 35])
    plt.yticks(np.arange(-40, 21, 20))
    plt.ylabel(r"$T_{\mathrm{C}}, T_{\mathrm{S}}$ [K]", fontsize=FS_LABELS)
    plt.xlabel(r"$\nu$ [MHz]", fontsize=FS_LABELS)
    plt.text(48.5, -50, "(h)", fontsize=FS_PANELS)

    plt.text(80, 18, r"$T_{\mathrm{S}}$", fontsize=FS_PANELS)
    plt.text(90, -10, r"$T_{\mathrm{C}}$", fontsize=FS_PANELS)

    plt.savefig(path_plot_save + "calibration_parameters.pdf", bbox_inches="tight")
