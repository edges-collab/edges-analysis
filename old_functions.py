"""
This file has a bunch of functions that were in the original edges-analysis, mostly
as scripts that aren't core functionality. They are here for posterity.
"""
import numpy as np
from edges_cal import s11_correction as s11c
from edges_io.io import SwitchingState
import matplotlib.pyplot as plt

from edges_cal import modelling as mdl
from edges_cal.cal_coefficients import EdgesFrequencyRange
from edges_analysis.analysis import beams, filters, io, loss, plots, tools

from edges_analysis.simulation import data_models as dm
from edges_analysis.config import config
from edges_analysis import s11 as s11m
from edges_analysis.analysis.scripts import (
    models_antenna_s11_remove_delay,
    batch_level1_to_level2,
)
from edges_analysis.estimation.plots import triangle_plot, load_samples
from edges_analysis.estimation import plots as eplt


def plots_midband_paper(plot_number, s11_path="antenna_s11_2018_147_17_04_33.txt"):
    # TODO: move to old
    if plot_number == 1:
        eplt.plot_receiver_calibration_params()
    elif plot_number == 10:
        eplt.plot_data_stats()
    elif plot_number == 11:
        eplt.plot_low_mid_comparison()
    elif plot_number == 12:
        eplt.plot_sky_model_comparison()
    elif plot_number == 121:
        eplt.plot_beam_gain()
    elif plot_number == 13:
        eplt.plot_sky_model()
    elif plot_number == 14:
        eplt.plot_absorption_model_comparison()
    elif plot_number == 2:
        eplt.plot_antenna_calibration_params(s11_path)
    elif plot_number == 3:
        eplt.plot_balun_loss2(s11_path)
    elif plot_number == 4:
        eplt.plot_antenna_beam()
    elif plot_number == 5:
        eplt.plot_beam_chromaticity_correction()
    elif plot_number == 50:
        eplt.plot_ground_loss()
    elif plot_number == 500:
        eplt.plot_calibration_parameters(s11_path)
    elif plot_number == 501:
        eplt.plot_balun_loss()
    elif plot_number == 6:
        eplt.beam_chromaticity_differences()
    elif plot_number == 7:
        eplt.plot_beam_power()
    elif plot_number == 8:
        eplt.plot_foreground_polychord_fit(
            datafile=(
                config["edges_folder"] + "mid_band/spectra/level5/case_nominal"
                "/integrated_spectrum_case_nominal_days_186_219_58-120MHz.txt"
            ),
            fg_chain_root=(
                config["edges_folder"]
                + "mid_band/polychord/20190815/case_nominal/foreground_powerlog_5par/chain"
            ),
            full_chain_root=(
                config["edges_folder"] + "mid_band/polychord/20190815/case_nominal"
                "/foreground_powerlog_5par_signal_exp_4par/chain"
            ),
            f_low=58,
            f_high=120,
            save_path=config["edges_folder"] + "plots/20190815/",
        )
    elif plot_number == 9:
        triangle_plot(
            file_root=(
                config["edges_folder"] + "mid_band/polychord/20190617/case2"
                "/foreground_exp_signal_exp_4par_60_120MHz/chain"
            ),
            output_file=(
                config["edges_folder"]
                + "plots/20190617/triangle_plot_exp_exp_4par_60_120MHz.pdf"
            ),
        )


def plots_midband_polychord(fig):
    # TODO: move to old

    if fig == 0:
        folder = (
            config["edges_folder"]
            + "mid_band/polychord/20190508/case1_nominal/foreground_model_exp/"
        )
        getdist_samples, ww, ll, best_fit, covariance_matrix = load_samples(
            folder + "chain.txt",
            0,
            label_names=[
                r"T_{100}\;[{\rm K}]",
                r"\beta",
                r"\gamma",
                r"\delta",
                r"\epsilon",
            ],
        )
        triangle_plot(
            getdist_samples, folder + "result.pdf", legend_FS=10, label_FS=10, axes_FS=5
        )

        v, t, w, sigma, inv_sigma, det_sigma = dm.real_data(10, 60, 120)

        model = dm.full_model(
            best_fit,
            v,
            100,
            model_type_signal="exp",
            model_type_foreground="exp",
            n_21=0,
            n_fgpar=5,
        )
    elif fig == 1:
        folder = (
            config["edges_folder"]
            + "mid_band/polychord/20190508/case1_nominal/foreground_model_linlog/"
        )
        getdist_samples, ww, ll, best_fit, covariance_matrix = load_samples(
            folder + "chain.txt",
            0,
            label_names=[r"a_0", r"a_1", r"a_2", r"a_3", r"a_4"],
        )
        triangle_plot(
            getdist_samples, folder + "result.pdf", legend_FS=10, label_FS=12, axes_FS=8
        )

        v, t, w, sigma, inv_sigma, det_sigma = dm.real_data(10, 60, 120)

        model = dm.full_model(
            best_fit,
            v,
            100,
            model_type_signal="exp",
            model_type_foreground="linlog",
            n_21=0,
            n_fgpar=5,
        )
    elif fig == 10:
        # Data used:  60-120
        folder = (
            config["edges_folder"] + "mid_band/polychord/20190508/case1_nominal"
            "/foreground_model_exp_signal_model_tanh_60_120MHz/"
        )
        getdist_samples, ww, ll, best_fit, covariance_matrix = load_samples(
            folder + "chain.txt",
            0,
            label_names=[
                r"A\;[{\rm K}]",
                r"\nu_0\;[{\rm MHz}]",
                r"w\;[{\rm MHz}]",
                r"\tau_1",
                r"\tau_2",
                r"T_{100}\;[{\rm K}]",
                r"\beta",
                r"\gamma",
                r"\delta",
                r"\epsilon",
            ],
        )
        triangle_plot(
            getdist_samples, folder + "result.pdf", legend_FS=10, label_FS=13, axes_FS=7
        )

        v, t, w, sigma, inv_sigma, det_sigma = dm.real_data(10, 60, 120)

        model = dm.full_model(
            best_fit,
            v,
            100,
            model_type_signal="tanh",
            model_type_foreground="exp",
            n_21=5,
            n_fgpar=5,
        )
    elif fig == 11:
        # Data used:  60-120, CASE 2, cterms7, wterms8
        folder = (
            config["edges_folder"] + "mid_band/polychord/20190508/case1_nominal"
            "/foreground_model_exp_signal_model_tanh_60_120MHz_case2/"
        )
        getdist_samples, ww, ll, best_fit, covariance_matrix = load_samples(
            folder + "chain.txt",
            0,
            label_names=[
                r"A\;[{\rm K}]",
                r"\nu_0\;[{\rm MHz}]",
                r"w\;[{\rm MHz}]",
                r"\tau_1",
                r"\tau_2",
                r"T_{100}\;[{\rm K}]",
                r"\beta",
                r"\gamma",
                r"\delta",
                r"\epsilon",
            ],
        )
        triangle_plot(
            getdist_samples, folder + "result.pdf", legend_FS=10, label_FS=13, axes_FS=7
        )

        v, t, w, sigma, inv_sigma, det_sigma = dm.real_data(2, 60, 120)

        model = dm.full_model(
            best_fit,
            v,
            100,
            model_type_signal="tanh",
            model_type_foreground="exp",
            n_21=5,
            n_fgpar=5,
        )
    elif fig == 12:
        # Data used:  60-120
        folder = (
            config["edges_folder"] + "mid_band/polychord/20190508/case1_nominal"
            "/foreground_model_exp_signal_model_exp_60_120MHz/"
        )
        getdist_samples, ww, ll, best_fit, covariance_matrix = load_samples(
            folder + "chain.txt",
            2000,
            label_names=[
                r"A\;[{\rm K}]",
                r"\nu_0\;[{\rm MHz}]",
                r"w\;[{\rm MHz}]",
                r"\tau",
                r"\chi",
                r"T_{100}\;[{\rm K}]",
                r"\beta",
                r"\gamma",
                r"\delta",
                r"\epsilon",
            ],
        )
        triangle_plot(
            getdist_samples, folder + "result.pdf", legend_FS=10, label_FS=13, axes_FS=7
        )

        v, t, w, sigma, inv_sigma, det_sigma = dm.real_data(10, 60, 120)

        model = dm.full_model(
            best_fit,
            v,
            100,
            model_type_signal="exp",
            model_type_foreground="exp",
            n_21=5,
            n_fgpar=5,
        )
    elif fig == 2:
        # Data used:  60-67, 103-119.5
        folder = (
            config["edges_folder"]
            + "mid_band/polychord/20190508/case1_nominal/foreground_model_exp_gap/"
        )
        getdist_samples, ww, ll, best_fit, covariance_matrix = load_samples(
            folder + "chain.txt",
            0,
            label_names=[
                r"T_{100}\;[{\rm K}]",
                r"\beta",
                r"\gamma",
                r"\delta",
                r"\epsilon",
            ],
        )
        triangle_plot(
            getdist_samples, folder + "result.pdf", legend_FS=10, label_FS=10, axes_FS=5
        )

        v, t, w, sigma, inv_sigma, det_sigma = dm.real_data(10, 60, 119.5)

        model = dm.full_model(
            best_fit,
            v,
            100,
            model_type_signal="exp",
            model_type_foreground="exp",
            n_21=0,
            n_fgpar=5,
        )
    elif fig == 3:
        # Data used:  60-65, 103-119.5
        folder = (
            config["edges_folder"]
            + "mid_band/polychord/20190508/case1_nominal/foreground_model_exp_gap2/"
        )
        getdist_samples, ww, ll, best_fit, covariance_matrix = load_samples(
            folder + "chain.txt",
            0,
            label_names=[
                r"T_{100}\;[{\rm K}]",
                r"\beta",
                r"\gamma",
                r"\delta",
                r"\epsilon",
            ],
        )
        triangle_plot(
            getdist_samples, folder + "result.pdf", legend_FS=10, label_FS=10, axes_FS=5
        )

        v, t, w, sigma, inv_sigma, det_sigma = dm.real_data(10, 60, 119.5)

        model = dm.full_model(
            best_fit,
            v,
            100,
            model_type_signal="exp",
            model_type_foreground="exp",
            n_21=0,
            n_fgpar=5,
        )
    elif fig == 4:
        # Data used:  60-65, 95-119.5
        folder = (
            config["edges_folder"]
            + "mid_band/polychord/20190508/case1_nominal/foreground_model_exp_gap3/"
        )
        getdist_samples, ww, ll, best_fit, covariance_matrix = load_samples(
            folder + "chain.txt",
            0,
            label_names=[
                r"T_{100}\;[{\rm K}]",
                r"\beta",
                r"\gamma",
                r"\delta",
                r"\epsilon",
            ],
        )
        triangle_plot(
            getdist_samples, folder + "result.pdf", legend_FS=10, abel_FS=10, axes_FS=5
        )

        v, t, w, sigma, inv_sigma, det_sigma = dm.real_data(10, 60, 119.5)

        model = dm.full_model(
            best_fit,
            v,
            100,
            model_type_signal="exp",
            model_type_foreground="exp",
            n_21=0,
            n_fgpar=5,
        )
    elif fig == 5:
        # Data used:  60-65, 95-115
        folder = (
            config["edges_folder"]
            + "mid_band/polychord/20190508/case1_nominal/foreground_model_exp_gap4/"
        )
        getdist_samples, ww, ll, best_fit, covariance_matrix = load_samples(
            folder + "chain.txt",
            0,
            label_names=[
                r"T_{100}\;[{\rm K}]",
                r"\beta",
                r"\gamma",
                r"\delta",
                r"\epsilon",
            ],
        )
        triangle_plot(
            getdist_samples, folder + "result.pdf", legend_FS=10, label_FS=10, axes_FS=5
        )

        v, t, w, sigma, inv_sigma, det_sigma = dm.real_data(10, 60, 115)

        model = dm.full_model(
            best_fit,
            v,
            100,
            model_type_signal="exp",
            model_type_foreground="exp",
            n_21=0,
            n_fgpar=5,
        )
    elif fig == 6:
        # Data used:  60-65, 100-115
        folder = (
            config["edges_folder"]
            + "mid_band/polychord/20190508/case1_nominal/foreground_model_exp_gap5/"
        )
        getdist_samples, ww, ll, best_fit, covariance_matrix = load_samples(
            folder + "chain.txt",
            0,
            label_names=[
                r"T_{100}\;[{\rm K}]",
                r"\beta",
                r"\gamma",
                r"\delta",
                r"\epsilon",
            ],
        )
        triangle_plot(
            getdist_samples, folder + "result.pdf", legend_FS=10, label_FS=10, axes_FS=5
        )

        v, t, w, sigma, inv_sigma, det_sigma = dm.real_data(10, 60, 115)

        model = dm.full_model(
            best_fit,
            v,
            100,
            model_type_signal="exp",
            model_type_foreground="exp",
            n_21=0,
            n_fgpar=5,
        )
    elif fig == 7:
        # Data used:  60-65, 97-115
        folder = (
            config["edges_folder"]
            + "mid_band/polychord/20190508/case1_nominal/foreground_model_exp_gap6/"
        )
        getdist_samples, ww, ll, best_fit, covariance_matrix = load_samples(
            folder + "chain.txt",
            0,
            label_names=[
                r"T_{100}\;[{\rm K}]",
                r"\beta",
                r"\gamma",
                r"\delta",
                r"\epsilon",
            ],
        )
        triangle_plot(
            getdist_samples, folder + "result.pdf", legend_FS=10, label_FS=10, axes_FS=5
        )

        v, t, w, sigma, inv_sigma, det_sigma = dm.real_data(10, 60, 115)

        model = dm.full_model(
            best_fit,
            v,
            100,
            model_type_signal="exp",
            model_type_foreground="exp",
            n_21=0,
            n_fgpar=5,
        )
    elif fig == 8:
        # Data used:  60-65, 100-115, CASE 2, cterms7, wterms8
        folder = (
            config["edges_folder"]
            + "mid_band/polychord/20190508/case1_nominal/foreground_model_exp_gap7/"
        )
        getdist_samples, ww, ll, best_fit, covariance_matrix = load_samples(
            folder + "chain.txt",
            0,
            label_names=[
                r"T_{100}\;[{\rm K}]",
                r"\beta",
                r"\gamma",
                r"\delta",
                r"\epsilon",
            ],
        )
        triangle_plot(
            getdist_samples, folder + "result.pdf", legend_FS=10, label_FS=10, axes_FS=5
        )

        v, t, w, sigma, inv_sigma, det_sigma = dm.real_data(2, 60, 115)

        model = dm.full_model(
            best_fit,
            v,
            100,
            model_type_signal="exp",
            model_type_foreground="exp",
            n_21=0,
            n_fgpar=5,
        )
    elif fig == 9:
        # Data used:  60-115
        folder = (
            config["edges_folder"]
            + "mid_band/polychord/20190508/case1_nominal/foreground_model_exp_signal_model_tanh/"
        )
        getdist_samples, ww, ll, best_fit, covariance_matrix = load_samples(
            folder + "chain.txt",
            0,
            label_names=[
                r"A\;[{\rm K}]",
                r"\nu_0\;[{\rm MHz}]",
                r"w\;[{\rm MHz}]",
                r"\tau_1",
                r"\tau_2",
                r"T_{100}\;[{\rm K}]",
                r"\beta",
                r"\gamma",
                r"\delta",
                r"\epsilon",
            ],
        )
        triangle_plot(
            getdist_samples, folder + "result.pdf", legend_FS=10, label_FS=10, axes_FS=5
        )

        v, t, w, sigma, inv_sigma, det_sigma = dm.real_data(10, 60, 115)

        model = dm.full_model(
            best_fit,
            v,
            100,
            model_type_signal="tanh",
            model_type_foreground="exp",
            n_21=5,
            n_fgpar=5,
        )
    return v, t, w, model


def plot_foreground_analysis():
    # TODO: potentially general plotting functionality, but otherwise old.

    path_file = (
        "/media/raul/DATA/EDGES_vol2/mid_band/spectra/level4/case_nominal_50"
        "-150MHz_LNA2_a2_h2_o2_s1_sim2_all_lc_yes_bc_20min/foreground_fits.hdf5"
    )

    fref, fit2, fit3, fit4, fit5 = io.level4_foreground_fits_read(path_file)

    plt.figure(1)

    fit2[:, :, 2] += 17.76
    fit3[:, :, 2] += 17.76
    fit4[:, :, 2] += 17.76
    fit5[:, :, 2] += 17.76

    fit2[:, :, 2][fit2[:, :, 2] > 24] -= 24
    fit3[:, :, 2][fit3[:, :, 2] > 24] -= 24
    fit4[:, :, 2][fit4[:, :, 2] > 24] -= 24
    fit5[:, :, 2][fit5[:, :, 2] > 24] -= 24

    plt.subplot(2, 1, 1)
    for k in range(len(fit2[0, :, 0])):
        for i in range(len(fit2[:, 0, 0])):
            if (
                (fit2[i, k, 3] < 0)
                and (fit3[i, k, 3] < 0)
                and (fit4[i, k, 3] < 0)
                and (fit5[i, k, 3] < 0)
            ):
                plt.plot(fit2[i, k, 2], np.abs(fit2[i, k, 6]), "b.")
                plt.plot(fit3[i, k, 2], np.abs(fit3[i, k, 6]), "g.")
                plt.plot(fit4[i, k, 2], np.abs(fit4[i, k, 6]), "y.")
                plt.plot(fit5[i, k, 2], np.abs(fit5[i, k, 6]), "r.")

    plt.grid()
    plt.legend(["2par", "3par", "4par", "5par"])
    plt.ylabel("beta")
    plt.ylim([2.48, 2.62])
    plt.xticks(np.arange(0, 25, 2))
    plt.xlim([0, 24])
    plt.title("Mid-Band Normalized at 90 MHz")

    plt.subplot(2, 1, 2)
    for k in range(len(fit2[0, :, 0])):
        for i in range(len(fit2[:, 0, 0])):
            if (
                (fit2[i, k, 3] < 0)
                and (fit3[i, k, 3] < 0)
                and (fit4[i, k, 3] < 0)
                and (fit5[i, k, 3] < 0)
            ):

                plt.plot(fit3[i, k, 2], fit3[i, k, 7], "g.")
                plt.plot(fit4[i, k, 2], fit4[i, k, 7], "y.")
                plt.plot(fit5[i, k, 2], fit5[i, k, 7], "r.")

    plt.grid()
    plt.ylabel("gamma")
    plt.xlabel("LST [hr]")
    plt.xticks(np.arange(0, 25, 2))
    plt.xlim([0, 24])


def batch_mid_band_level1_to_level2():
    batch_level1_to_level2(
        "mid_band",
        path=config["home_folder"] + "/EDGES/spectra/level1/mid_band/300_350/",
        omit_days=range(170, 175),
    )


def batch_low_band3_level1_to_level2():
    batch_level1_to_level2(
        "low_band3",
        path=config["home_folder"] + "/EDGES/spectra/level1/low_band3/300_350/",
        omit_days=[225],
        day_indx=12,
    )


def plot_residuals_GHA_1hr_bin(direc, f, r, w, fname="linlog_5terms_60-120MHz"):
    # TODO: move to old
    GHA_edges = list(np.arange(0, 25))
    DY = 0.4

    # Plotting
    plots.plot_residuals(
        f,
        r,
        w,
        [f"GHA={gha}-{GHA_edges[i + 1]} hr" for i, gha in enumerate(GHA_edges[:-1])],
        DY=DY,
        f_low=40,
        f_high=125,
        XTICKS=np.arange(60, 121, 20),
        XTEXT=42,
        YLABEL=f"{DY} K per division",
        TITLE="5 LINLOG terms, 60-120 MHz",
        save=True,
        figure_path=direc,
        figure_name=fname,
    )


def plot_residuals_GHA_Xhr_bin(direc, fname, f, r, w):
    # TODO: move to old
    DY = 0.5

    # Plotting
    plots.plot_residuals(
        f,
        r,
        w,
        ["GHA=0-5 hr", "GHA=5-11 hr", "GHA=11-18 hr", "GHA=18-24 hr"],
        FIG_SX=8,
        FIG_SY=7,
        DY=DY,
        f_low=35,
        f_high=165,
        XTICKS=np.arange(60, 161, 20),
        XTEXT=38,
        YLABEL=f"{DY} K per division",
        TITLE="4 LINLOG terms, 62-120 MHz",
        save=True,
        figure_path=direc,
        figure_name=fname,
    )


def vna_comparison():
    # TODO: move to old
    paths = [
        config["edges_folder"] + "others/vna_comparison/keysight_e5061a/",
        config["edges_folder"] + "others/vna_comparison/copper_mountain_r60/",
        config["edges_folder"] + "others/vna_comparison/tektronix_ttr506a/",
        config["edges_folder"] + "others/vna_comparison/copper_mountain_tr1300/",
    ]

    labels = ["Keysight E5061A", "CM R60", "Tektronix", "CM TR1300"]

    plots.plot_vna_comparison(paths, labels)


def VNA_comparison2():
    # TODO: move to old
    paths = [
        config["edges_folder"] + "others/vna_comparison/again/ks_e5061a/",
        config["edges_folder"] + "others/vna_comparison/again/cm_tr1300/",
    ]

    labels = ["Keysight E5061A", "CM TR1300"]
    plots.plot_vna_comparison(paths, labels)


def VNA_comparison3():
    # TODO: move to old
    paths = (
        config["edges_folder"]
        + "others/vna_comparison/fieldfox_N9923A/agilent_E5061A_male/",
        config["edges_folder"]
        + "others/vna_comparison/fieldfox_N9923A/agilent_E5061A_female/",
        config["edges_folder"]
        + "others/vna_comparison/fieldfox_N9923A/fieldfox_N9923A_male/",
        config["edges_folder"]
        + "others/vna_comparison/fieldfox_N9923A/fieldfox_N9923A_female/",
    )
    labels = ["Male E5061A", "Male N9923A", "Female E5061A", "Female N9923A"]
    plots.plot_vna_comparison(paths, labels, repeat_num=2)


def plots_for_memo148(plot_number):
    # TODO: move to old

    # Receiver calibration parameters
    if plot_number == 1:
        # Paths
        path_plot_save = config["edges_folder"] + "plots/20190828/"

        # Calibration parameters
        rcv_file = (
            config["edges_folder"]
            + "mid_band/calibration/receiver_calibration/receiver1"
            "/2018_01_25C/results/nominal/calibration_files"
            "/calibration_file_receiver1_cterms7_wterms8.txt"
        )

        rcv = np.genfromtxt(rcv_file)

        f_low = 50
        f_high = 150

        fX = rcv[:, 0]
        rcv2 = rcv[(fX >= f_low) & (fX <= f_high), :]

        fe = rcv2[:, 0]
        rl = rcv2[:, 1] + 1j * rcv2[:, 2]
        sca = rcv2[:, 3]
        off = rcv2[:, 4]
        TU = rcv2[:, 5]
        TC = rcv2[:, 6]
        TS = rcv2[:, 7]

        rcv1 = np.genfromtxt(
            "/home/raul/DATA2/EDGES_vol2/mid_band/calibration/receiver_calibration/receiver1"
            "/2019_04_25C/results/nominal/calibration_files"
            "/calibration_file_receiver1_50_150MHz_cterms8_wterms10.txt"
        )

        # Low-Band
        rcv_lb = np.genfromtxt(
            "/home/raul/DATA1/EDGES_vol1/calibration/receiver_calibration/low_band1/2015_08_25C"
            "/results/nominal/calibration_files/calibration_file_low_band_2015_nominal.txt"
        )

        # Plot

        size_x = 6
        size_y = 8  # 10.5
        x0 = 0.15
        y0 = 0.09
        dx = 0.7
        dy = 0.3

        f1 = plt.figure(num=1, figsize=(size_x, size_y))

        ax = f1.add_axes([x0, y0 + 2 * dy, dx, dy])
        h1 = ax.plot(
            fe,
            20 * np.log10(np.abs(rl)),
            "b",
            linewidth=1.5,
            label=r"$|\Gamma_{\mathrm{rec}}|$",
        )
        ax.plot(
            rcv1[:, 0],
            20 * np.log10(np.abs(rcv1[:, 1] + 1j * rcv1[:, 2])),
            "r",
            linewidth=1.5,
        )
        ax.plot(
            rcv_lb[:, 0],
            20 * np.log10(np.abs(rcv_lb[:, 1] + 1j * rcv_lb[:, 2])),
            "g",
            linewidth=1.5,
        )

        ax2 = ax.twinx()
        h2 = ax2.plot(
            fe,
            (180 / np.pi) * np.unwrap(np.angle(rl)),
            "b--",
            linewidth=1.5,
            label=r"$\angle\/\Gamma_{\mathrm{rec}}$",
        )
        ax2.plot(
            rcv1[:, 0],
            (180 / np.pi) * np.unwrap(np.angle(rcv1[:, 1] + 1j * rcv1[:, 2])),
            "r--",
            linewidth=1.5,
        )
        ax2.plot(
            rcv_lb[:, 0],
            (180 / np.pi) * np.unwrap(np.angle(rcv_lb[:, 1] + 1j * rcv_lb[:, 2])),
            "g--",
            linewidth=1.5,
        )

        h = h1 + h2
        labels = [l.get_label() for l in h]
        ax.legend(h, labels, loc=0, fontsize=10, ncol=2)

        ax.set_ylim([-40, -28])
        ax.set_xticklabels("")
        ax.set_yticks(np.arange(-39, -28, 2))
        ax.set_ylabel(r"$|\Gamma_{\mathrm{rec}}|$ [dB]", fontsize=14)
        ax.text(48.5, -39.5, "(a)", fontsize=16)
        ax.text(113, -37, "2015-Aug", fontweight="bold", color="g")
        ax.text(113, -38, "2018-Jan", fontweight="bold", color="b")
        ax.text(113, -39, "2019-Apr", fontweight="bold", color="r")

        ax2.set_ylim([70, 130])
        ax2.set_xticklabels("")
        ax2.set_yticks(np.arange(80, 121, 10))
        ax2.set_ylabel(r"$\angle\/\Gamma_{\mathrm{rec}}$ [ $^\mathrm{o}$]", fontsize=14)

        ax.set_xlim([48, 152])
        ax.tick_params(axis="x", direction="in")
        ax.set_xticks(np.arange(50, 151, 10))

        ax = f1.add_axes([x0, y0 + 1 * dy, dx, dy])
        h1 = ax.plot(fe, sca, "b", linewidth=1.5, label="$C_1$")
        ax.plot(rcv1[:, 0], rcv1[:, 3], "r", linewidth=1.5)
        ax.plot(
            rcv_lb[:, 0], rcv_lb[:, 3], "g", linewidth=1.5
        )  # <----------------------------- Low-Band
        ax2 = ax.twinx()
        h2 = ax2.plot(fe, off, "b--", linewidth=1.5, label="$C_2$")
        ax2.plot(rcv1[:, 0], rcv1[:, 4], "r--", linewidth=1.5)
        ax2.plot(
            rcv_lb[:, 0], rcv_lb[:, 4], "g--", linewidth=1.5
        )  # <----------------------------- Low-Band
        h = h1 + h2
        labels = [l.get_label() for l in h]
        ax.legend(h, labels, loc=0, fontsize=10, ncol=2)

        ax.set_ylim([3.3, 5.2])
        ax.set_xticklabels("")
        ax.set_yticks(np.arange(3.5, 5.1, 0.5))
        ax.set_ylabel("$C_1$", fontsize=14)
        ax.text(48.5, 3.38, "(b)", fontsize=16)

        ax2.set_ylim([-2.75, -0])
        ax2.set_xticklabels("")
        ax2.set_yticks(np.arange(-2.5, -0.05, 0.5))
        ax2.set_ylabel("$C_2$ [K]", fontsize=14)

        ax.set_xlim([48, 152])
        ax.tick_params(axis="x", direction="in")
        ax.set_xticks(np.arange(50, 151, 10))

        ax = f1.add_axes([x0, y0 + 0 * dy, dx, dy])
        h1 = ax.plot(fe, TU, "b", linewidth=1.5, label=r"$T_{\mathrm{unc}}$")
        ax.plot(rcv1[:, 0], rcv1[:, 5], "r", linewidth=1.5)  #
        ax.plot(rcv_lb[:, 0], rcv_lb[:, 5], "g", linewidth=1.5)

        ax2 = ax.twinx()
        h2 = ax2.plot(fe, TC, "b--", linewidth=1.5, label=r"$T_{\mathrm{cos}}$")
        ax2.plot(rcv1[:, 0], rcv1[:, 6], "r--", linewidth=1.5)
        ax2.plot(rcv_lb[:, 0], rcv_lb[:, 6], "g--", linewidth=1.5)

        h3 = ax2.plot(fe, TS, "b:", linewidth=1.5, label=r"$T_{\mathrm{sin}}$")
        ax2.plot(rcv1[:, 0], rcv1[:, 7], "r:", linewidth=1.5)
        ax2.plot(rcv_lb[:, 0], rcv_lb[:, 7], "g:", linewidth=1.5)

        h = h1 + h2 + h3
        labels = [l.get_label() for l in h]
        ax.legend(h, labels, loc=0, fontsize=10, ncol=3)

        ax.set_ylim([178, 190])
        ax.set_yticks(np.arange(180, 189, 2))
        ax.set_ylabel(r"$T_{\mathrm{unc}}$ [K]", fontsize=14)
        ax.set_xlabel(r"$\nu$ [MHz]", fontsize=14)
        ax.text(48.5, 178.5, "(c)", fontsize=16)

        ax2.set_ylim([-55, 35])
        ax2.set_yticks(np.arange(-40, 21, 20))
        ax2.set_ylabel(r"$T_{\mathrm{cos}}, T_{\mathrm{sin}}$ [K]", fontsize=14)

        ax.set_xlim([48, 152])
        ax.set_xticks(np.arange(50, 151, 10))

        plt.savefig(path_plot_save + "receiver_calibration.pdf", bbox_inches="tight")
    if plot_number == 2:
        # Paths
        path_plot_save = config["edges_folder"] + "plots/20190828/"

        flb = np.arange(50, 100, 1)
        alb = np.ones(len(flb))

        f = np.arange(50, 151, 1)
        ant_s11 = np.ones(len(f))

        corrections = [s11c.low_band_switch_correction(alb, 27.16, f_in=flb)[1]]

        paths = [
            (
                config["edges_folder"]
                + "calibration/receiver_calibration/mid_band/2017_11_15C_25C_35C/data/s11/raw/25C"
                "/receiver_MRO_fieldfox_40-200MHz/"
            ),
            (
                config["edges_folder"]
                + "mid_band/calibration/receiver_calibration/receiver1/2018_01_25C/data/s11/raw"
                "/InternalSwitch/"
            ),
            (
                config["edges_folder"]
                + "mid_band/calibration/receiver_calibration/receiver1/2019_03_25C/data/s11/raw"
                "/SwitchingState01/"
            ),
        ]

        for path in paths:
            corrections.append(
                s11c.low_band_switch_correction(
                    ant_s11,
                    internal_switch=SwitchingState(path, run_num=1),
                    f_in=f,
                    poly_order=23,
                )[1]
            )

        size_x = 11
        size_y = 13.0

        fig, ax = plt.subplots(3, 2, figsize=(size_x, size_y))

        for i, (correction, style) in enumerate(
            zip(corrections, ["k--", "k:", "b", "r--"])
        ):
            for k, label in enumerate(["s11", "s12s21", "s22"]):
                for j, fnc in enumerate(
                    (
                        lambda x: 20 * np.log10(np.abs(x)),
                        lambda x: (180 / np.pi) * np.unwrap(np.angle(x)),
                    )
                ):
                    ax[k, j].plot(flb if not i else f, fnc(correction[label]), style)

                    ax[k, j].xaxis.set_xticks(np.arange(50, 151, 10), labels="")
                    ax[k, j].grid()

                    if j == 0:
                        ax[k, j].set_ylabel(r"$|{}|$ [dB]".format(label))
                    else:
                        ax[k, j].set_ylabel(r"$\angle {}$ [deg]".format(label))

        ax[0, 0].legend(
            [
                r"sw 2015, 50.12$\Omega$",
                r"sw 2017, 49.85$\Omega$",
                r"sw 2018, 50.027$\Omega$",
                r"sw 2019, 50.15$\Omega$",
            ]
        )

        plt.savefig(path_plot_save + "sw_parameters.pdf", bbox_inches="tight")

    if plot_number == 3:
        # Paths
        path_plot_save = config["edges_folder"] + "plots/20190828/"

        size_x = 11
        size_y = 13.0

        plt.figure(num=1, figsize=(size_x, size_y))

        # Frequency
        fe = EdgesFrequencyRange(f_low=50, f_high=150).freq
        # f, il, ih = ba.frequency_edges(50, 150)
        # fe = f[il : ih + 1]

        ra1 = s11m.antenna_s11_remove_delay(
            "antenna_s11_2018_147_16_52_34.txt", fe, delay_0=0.17, n_fit=15
        )
        ra2 = s11m.antenna_s11_remove_delay(
            "antenna_s11_mid_band_2018_147_switch_calibration_2019_03_case6.txt",
            fe,
            delay_0=0.17,
            n_fit=15,
        )

        ra3 = s11m.antenna_s11_remove_delay(
            "antenna_s11_mid_band_2018_147_switch_calibration_2018_01_5012_case3.txt",
            fe,
            delay_0=0.17,
            n_fit=15,
        )
        ra4 = s11m.antenna_s11_remove_delay(
            "antenna_s11_mid_band_2018_147_switch_calibration_2019_03_5012_case3.txt",
            fe,
            delay_0=0.17,
            n_fit=15,
        )

        ra11 = s11m.antenna_s11_remove_delay(
            "antenna_s11_mid_band_2018_222_case1_switch_calibration_2018_01.txt",
            fe,
            delay_0=0.17,
            n_fit=15,
        )
        ra31 = s11m.antenna_s11_remove_delay(
            "antenna_s11_mid_band_2018_222_case1_switch_calibration_2019_03.txt",
            fe,
            delay_0=0.17,
            n_fit=15,
        )

        plt.subplot(3, 2, 1)
        plt.plot(fe, 20 * np.log10(np.abs(ra1)), "b")
        plt.xticks(np.arange(50, 151, 10), labels="")
        plt.grid()
        plt.ylabel("magnitude [dB]")

        plt.subplot(3, 2, 2)
        plt.plot(fe, (180 / np.pi) * np.unwrap(np.angle(ra1)), "b")
        plt.xticks(np.arange(50, 151, 10), labels="")
        plt.grid()
        plt.ylabel("phase [deg]")

        plt.subplot(3, 2, 3)
        plt.plot(fe, 20 * np.log10(np.abs(ra2)) - 20 * np.log10(np.abs(ra1)), "r")
        plt.plot(fe, 20 * np.log10(np.abs(ra3)) - 20 * np.log10(np.abs(ra1)), "r--")
        plt.plot(fe, 20 * np.log10(np.abs(ra4)) - 20 * np.log10(np.abs(ra1)), "r:")
        plt.xticks(np.arange(50, 151, 10))
        plt.ylim([-0.04, 0.06])
        plt.grid()
        plt.ylabel(r"$\Delta$ magnitude [dB]")
        plt.legend(
            [
                r"Day 147(sw 2019, 50.15$\Omega$) - Day 147(sw 2018, 50.027$\Omega$)",
                r"Day 147(sw 2018, 50.12$\Omega$) - Day 147(sw 2018, 50.027$\Omega$)",
                r"Day 147(sw 2019, 50.12$\Omega$) - Day 147(sw 2018, 50.027$\Omega$)",
            ],
            fontsize=8,
        )

        plt.subplot(3, 2, 4)
        plt.plot(
            fe,
            (180 / np.pi) * np.unwrap(np.angle(ra2))
            - (180 / np.pi) * np.unwrap(np.angle(ra1)),
            "r",
        )
        plt.plot(
            fe,
            (180 / np.pi) * np.unwrap(np.angle(ra3))
            - (180 / np.pi) * np.unwrap(np.angle(ra1)),
            "r--",
        )
        plt.plot(
            fe,
            (180 / np.pi) * np.unwrap(np.angle(ra4))
            - (180 / np.pi) * np.unwrap(np.angle(ra1)),
            "r:",
        )
        plt.xticks(np.arange(50, 151, 10))
        plt.ylim([-0.3, 0.3])
        plt.grid()
        plt.ylabel(r"$\Delta$ phase [deg]")

        plt.subplot(3, 2, 5)
        plt.plot(fe, 20 * np.log10(np.abs(ra11)) - 20 * np.log10(np.abs(ra1)), "g")
        plt.plot(fe, 20 * np.log10(np.abs(ra31)) - 20 * np.log10(np.abs(ra1)), "g--")
        plt.xticks(np.arange(50, 151, 10))
        plt.ylim([-0.2, 0.2])
        plt.grid()
        plt.xlabel(r"$\nu$ [MHz]", fontsize=14)
        plt.ylabel(r"$\Delta$ magnitude [dB]")
        plt.legend(
            [
                r"Day 222(sw 2018, 50.027$\Omega$) - Day 147(sw 2018, 50.027$\Omega$)",
                r"Day 222(sw 2019, 50.15$\Omega$) - Day 147(sw 2018, 50.027$\Omega$)",
            ],
            fontsize=8,
        )

        plt.subplot(3, 2, 6)
        plt.plot(
            fe,
            (180 / np.pi) * np.unwrap(np.angle(ra11))
            - (180 / np.pi) * np.unwrap(np.angle(ra1)),
            "g",
        )
        plt.plot(
            fe,
            (180 / np.pi) * np.unwrap(np.angle(ra31))
            - (180 / np.pi) * np.unwrap(np.angle(ra1)),
            "g--",
        )
        plt.xticks(np.arange(50, 151, 10))
        plt.ylim([-2, 2])
        plt.grid()
        plt.xlabel(r"$\nu$ [MHz]", fontsize=14)
        plt.ylabel(r"$\Delta$ phase [deg]")

        plt.savefig(path_plot_save + "antenna_s11.pdf", bbox_inches="tight")
        plt.close()
        plt.close()
        plt.close()
        plt.close()

    if plot_number == 4:
        # Paths
        path_plot_save = config["edges_folder"] + "plots/20190828/"

        GHA1 = 6
        GHA2 = 18

        f, t150_low_case1, w, s150_low_case1 = io.level3_single_file_test(
            config["edges_folder"] + "mid_band/spectra"
            "/level3/tests_55_150MHz/rcv18_sw18/2018_150_00.hdf5",
            GHA1,
            GHA2,
            60,
            150,
            False,
            "name",
        )
        f, t150_low_case2, w, s150_low_case2 = io.level3_single_file_test(
            config["edges_folder"]
            + "mid_band/spectra/level3/tests_55_150MHz/rcv18_sw19/2018_150_00"
            ".hdf5",
            GHA1,
            GHA2,
            60,
            150,
            False,
            "name",
        )
        f, t150_low_case3, w, s150_low_case3 = io.level3_single_file_test(
            config["edges_folder"]
            + "mid_band/spectra/level3/tests_55_150MHz/rcv19_sw18/2018_150_00"
            ".hdf5",
            GHA1,
            GHA2,
            60,
            150,
            False,
            "name",
        )
        f, t150_low_case4, w, s150_low_case4 = io.level3_single_file_test(
            config["edges_folder"]
            + "mid_band/spectra/level3/tests_55_150MHz/rcv19_sw19/2018_150_00"
            ".hdf5",
            GHA1,
            GHA2,
            60,
            150,
            False,
            "name",
        )

        plt.figure(figsize=[6, 5])

        plt.subplot(2, 1, 1)
        plt.plot(f, t150_low_case2 - t150_low_case1, "b")
        plt.xticks(np.arange(50, 151, 10), labels="")
        plt.xlim([58, 152])
        plt.ylim([-1, 2])
        plt.title("Day 2018-150, GHA=6-18 hr")
        plt.xlabel(r"$\nu$ [MHz]", fontsize=14)
        plt.ylabel(r"$\Delta$ temperature [K]")
        plt.legend(["Case 2 - Case 1"], fontsize=9)

        plt.subplot(2, 1, 2)
        plt.plot(f, t150_low_case3 - t150_low_case1, "b--")
        plt.plot(f, t150_low_case4 - t150_low_case1, "b:")
        plt.xticks(np.arange(50, 151, 10))
        plt.xlim([58, 152])
        # plt.ylim([-3,3])
        # plt.title('Day 150, GHA=6-18 hr')
        plt.xlabel(r"$\nu$ [MHz]", fontsize=14)
        plt.ylabel(r"$\Delta$ temperature [K]")
        plt.legend(["Case 3 - Case 1", "Case 4 - Case 1"], fontsize=9)

        plt.savefig(path_plot_save + "delta_temperature.pdf", bbox_inches="tight")
        plt.close()
        plt.close()
        plt.close()
        plt.close()

    if plot_number == 5:
        # Paths
        path_plot_save = config["edges_folder"] + "plots/20190828/"

        GHA1 = GHA2 = 18

        fx, t150_low_case1, w, s150_low_case1 = io.level3_single_file_test(
            config["edges_folder"]
            + "mid_band/spectra/level3/tests_55_150MHz/rcv18_sw18/2018_150_00.hdf5",
            GHA1,
            GHA2,
            55,
            150,
            False,
            "name",
        )
        fx, t150_low_case2, w, s150_low_case2 = io.level3_single_file_test(
            config["edges_folder"]
            + "mid_band/spectra/level3/tests_55_150MHz/rcv18_sw19/2018_150_00.hdf5",
            GHA1,
            GHA2,
            55,
            150,
            False,
            "name",
        )
        fx, t150_low_case3, w, s150_low_case3 = io.level3_single_file_test(
            config["edges_folder"]
            + "mid_band/spectra/level3/tests_55_150MHz/rcv19_sw18/2018_150_00.hdf5",
            GHA1,
            GHA2,
            55,
            150,
            False,
            "name",
        )
        fx, t150_low_case4, w, s150_low_case4 = io.level3_single_file_test(
            config["edges_folder"]
            + "mid_band/spectra/level3/tests_55_150MHz/rcv19_sw19/2018_150_00.hdf5",
            GHA1,
            GHA2,
            55,
            150,
            False,
            "name",
        )

    f_low = 60
    f_high = 150

    f = fx[(fx >= f_low) & (fx <= f_high)]
    t150_low_case1 = t150_low_case1[(fx >= f_low) & (fx <= f_high)]
    t150_low_case2 = t150_low_case2[(fx >= f_low) & (fx <= f_high)]
    t150_low_case3 = t150_low_case3[(fx >= f_low) & (fx <= f_high)]
    t150_low_case4 = t150_low_case4[(fx >= f_low) & (fx <= f_high)]

    s150_low_case1 = s150_low_case1[(fx >= f_low) & (fx <= f_high)]
    s150_low_case2 = s150_low_case2[(fx >= f_low) & (fx <= f_high)]
    s150_low_case3 = s150_low_case3[(fx >= f_low) & (fx <= f_high)]
    s150_low_case4 = s150_low_case4[(fx >= f_low) & (fx <= f_high)]

    p = mdl.fit_polynomial_fourier(
        "LINLOG", f, t150_low_case1, 5, Weights=1 / (s150_low_case1 ** 2)
    )
    r1 = t150_low_case1 - p[1]

    p = mdl.fit_polynomial_fourier(
        "LINLOG", f, t150_low_case2, 5, Weights=1 / (s150_low_case2 ** 2)
    )
    r2 = t150_low_case2 - p[1]

    p = mdl.fit_polynomial_fourier(
        "LINLOG", f, t150_low_case3, 5, Weights=1 / (s150_low_case3 ** 2)
    )
    r3 = t150_low_case3 - p[1]

    p = mdl.fit_polynomial_fourier(
        "LINLOG", f, t150_low_case4, 5, Weights=1 / (s150_low_case4 ** 2)
    )
    r4 = t150_low_case4 - p[1]

    f1 = plt.figure(figsize=[8, 8])

    plt.subplot(4, 1, 1)
    plt.plot(f, r1, "b")
    plt.xticks(np.arange(50, 151, 10), labels="")
    plt.xlim([55, 152])
    plt.ylim([-1, 1])
    plt.title("Day 2018-150, GHA=6-18 hr")
    # plt.xlabel(r'$\nu$ [MHz]', fontsize=14)
    plt.ylabel(r"$\Delta$ T [K]")
    plt.legend(["Case 1"], fontsize=9)

    plt.subplot(4, 1, 2)
    plt.plot(f, r2, "b")
    plt.xticks(np.arange(50, 151, 10), abels="")
    plt.xlim([55, 152])
    plt.ylim([-1, 1])
    plt.ylabel(r"$\Delta$ T [K]")
    plt.legend(["Case 2"], fontsize=9)

    plt.subplot(4, 1, 3)
    plt.plot(f, r3, "b")
    plt.xticks(np.arange(50, 151, 10), labels="")
    plt.xlim([55, 152])
    plt.ylim([-1, 1])
    plt.ylabel(r"$\Delta$ T [K]")
    plt.legend(["Case 3"], fontsize=9)

    plt.subplot(4, 1, 4)
    plt.plot(f, r4, "b")
    plt.xticks(np.arange(50, 151, 10))
    plt.xlim([55, 152])
    plt.ylim([-1, 1])
    plt.xlabel(r"$\nu$ [MHz]", fontsize=14)
    plt.ylabel(r"$\Delta$ T		K]")
    plt.legend(["Case 4"], fontsize=9)

    plt.savefig(path_plot_save + "residuals1.pdf", bbox_inches="tight")
    if plot_number == 6:
        # Paths
        path_plot_save = config["edges_folder"] + "plots/20190828/"

    GHA1 = 6
    GHA2 = 18

    fx, t188_low_case1, w, s188_low_case1 = io.level3_single_file_test(
        config["edges_folder"]
        + "mid_band/spectra/level3/tests_55_150MHz/rcv18_sw18/2018_188_00.hdf5",
        GHA1,
        GHA2,
        55,
        150,
        False,
        "name",
    )
    fx, t188_low_case2, w, s188_low_case2 = io.level3_single_file_test(
        config["edges_folder"]
        + "mid_band/spectra/level3/tests_55_150MHz/rcv18_sw19/2018_188_00.hdf5",
        GHA1,
        GHA2,
        55,
        150,
        False,
        "name",
    )
    fx, t188_low_case3, w, s188_low_case3 = io.level3_single_file_test(
        config["edges_folder"] + "mid_band/spectra/level3"
        "/tests_55_150MHz/rcv19_sw18/2018_188_00.hdf5",
        GHA1,
        GHA2,
        55,
        150,
        False,
        "name",
    )
    fx, t188_low_case4, w, s188_low_case4 = io.level3_single_file_test(
        config["edges_folder"]
        + "mid_band/spectra/level3/tests_55_150MHz/rcv19_sw19/2018_188_00.hdf5",
        GHA1,
        GHA2,
        55,
        150,
        False,
        "name",
    )

    f_low = 60
    f_high = 150

    f = fx[(fx >= f_low) & (fx <= f_high)]

    t188_low_case1 = t188_low_case1[(fx >= f_low) & (fx <= f_high)]
    t188_low_case2 = t188_low_case2[(fx >= f_low) & (fx <= f_high)]
    t188_low_case3 = t188_low_case3[(fx >= f_low) & (fx <= f_high)]
    t188_low_case4 = t188_low_case4[(fx >= f_low) & (fx <= f_high)]

    s188_low_case1 = s188_low_case1[(fx >= f_low) & (fx <= f_high)]
    s188_low_case2 = s188_low_case2[(fx >= f_low) & (fx <= f_high)]
    s188_low_case3 = s188_low_case3[(fx >= f_low) & (fx <= f_high)]
    s188_low_case4 = s188_low_case4[(fx >= f_low) & (fx <= f_high)]

    p = mdl.fit_polynomial_fourier(
        "LINLOG", f, t188_low_case1, 5, Weights=1 / (s188_low_case1 ** 2)
    )
    r1 = t188_low_case1 - p[1]

    p = mdl.fit_polynomial_fourier(
        "LINLOG", f, t188_low_case2, 5, Weights=1 / (s188_low_case2 ** 2)
    )
    r2 = t188_low_case2 - p[1]

    p = mdl.fit_polynomial_fourier(
        "LINLOG", f, t188_low_case3, 5, Weights=1 / (s188_low_case3 ** 2)
    )
    r3 = t188_low_case3 - p[1]

    p = mdl.fit_polynomial_fourier(
        "LINLOG", f, t188_low_case4, 5, Weights=1 / (s188_low_case4 ** 2)
    )
    r4 = t188_low_case4 - p[1]

    f1 = plt.figure(figsize=[8, 8])

    plt.subplot(4, 1, 1)
    plt.plot(f, r1, "b")
    plt.xticks(np.arange(50, 151, 10), labels="")
    plt.xlim([55, 152])
    plt.ylim([-1, 1])
    plt.title("Day 2018-188, GHA=6-18 hr")
    plt.ylabel(r"$\Delta$ T [K]")
    plt.legend(["Case 1"], fontsize=9)

    plt.subplot(4, 1, 2)
    plt.plot(f, r2, "b")
    plt.xticks(np.arange(50, 151, 10), labels="")
    plt.xlim([55, 152])
    plt.ylim([-1, 1])
    plt.ylabel(r"$\Delta$ T [K]")
    plt.legend(["Case 2"], fontsize=9)

    plt.subplot(4, 1, 3)
    plt.plot(f, r3, "b")
    plt.xticks(np.arange(50, 151, 10), labels="")
    plt.xlim([55, 152])
    plt.ylim([-1, 1])
    plt.ylabel(r"$\Delta$ T [K]")
    plt.legend(["Case 3"], fontsize=9)

    plt.subplot(4, 1, 4)
    plt.plot(f, r4, "b")
    plt.xticks(np.arange(50, 151, 10))
    plt.xlim([55, 152])
    plt.ylim([-1, 1])
    plt.xlabel(r"$\nu$ [MHz]", fontsize=14)
    plt.ylabel(r"$\Delta$ T [K]")
    plt.legend(["Case 4"], fontsize=9)

    plt.savefig(path_plot_save + "residuals2.pdf", bbox_inches="tight")

    if plot_number == 7:
        # Paths
        path_plot_save = config["edges_folder"] + "plots/20190828/"

    GHA1 = 6
    GHA2 = 18

    fx, t150_low_case1, w, s150_low_case1 = io.level3_single_file_test(
        config["edges_folder"]
        + "mid_band/spectra/level3/tests_55_150MHz/rcv18_sw18/2018_150_00.hdf5",
        GHA1,
        GHA2,
        55,
        150,
        False,
        "name",
    )
    fx, t150_low_case2, w, s150_low_case2 = io.level3_single_file_test(
        config["edges_folder"]
        + "mid_band/spectra/level3/tests_55_150MHz/rcv18_sw19/2018_150_00.hdf5",
        GHA1,
        GHA2,
        55,
        150,
        False,
        "name",
    )
    fx, t150_low_case3, w, s150_low_case3 = io.level3_single_file_test(
        config["edges_folder"]
        + "mid_band/spectra/level3/tests_55_150MHz/rcv18_sw18and19_ant147/2018_150_00.hdf5",
        GHA1,
        GHA2,
        55,
        150,
        False,
        "name",
    )

    f_low = 60
    f_high = 150

    f = fx[(fx >= f_low) & (fx <= f_high)]

    t150_low_case1 = t150_low_case1[(fx >= f_low) & (fx <= f_high)]
    t150_low_case2 = t150_low_case2[(fx >= f_low) & (fx <= f_high)]
    t150_low_case3 = t150_low_case3[(fx >= f_low) & (fx <= f_high)]

    s150_low_case1 = s150_low_case1[(fx >= f_low) & (fx <= f_high)]
    s150_low_case2 = s150_low_case2[(fx >= f_low) & (fx <= f_high)]
    s150_low_case3 = s150_low_case3[(fx >= f_low) & (fx <= f_high)]

    p = mdl.fit_polynomial_fourier(
        "LINLOG", f, t150_low_case1, 5, Weights=1 / (s150_low_case1 ** 2)
    )
    r1 = t150_low_case1 - p[1]

    p = mdl.fit_polynomial_fourier(
        "LINLOG", f, t150_low_case2, 5, Weights=1 / (s150_low_case2 ** 2)
    )
    r2 = t150_low_case2 - p[1]

    p = mdl.fit_polynomial_fourier(
        "LINLOG", f, t150_low_case3, 5, Weights=1 / (s150_low_case3 ** 2)
    )
    r3 = t150_low_case3 - p[1]

    f1 = plt.figure(figsize=[8, 8])

    plt.subplot(3, 1, 1)
    plt.plot(f, r1, "b")
    plt.xticks(np.arange(50, 151, 10), labels="")
    plt.xlim([55, 152])
    plt.ylim([-1, 1])
    plt.title("Day 2018-150, GHA=6-18 hr")
    plt.ylabel(r"$\Delta$ T [K]")
    plt.legend(["Case 1"], fontsize=9)

    plt.subplot(3, 1, 2)
    plt.plot(f, r2, "b")
    plt.xticks(np.arange(50, 151, 10), labels="")
    plt.xlim([55, 152])
    plt.ylim([-1, 1])
    plt.ylabel(r"$\Delta$ T [K]")
    plt.legend(["Case 2"], fontsize=9)

    plt.subplot(3, 1, 3)
    plt.plot(f, r3, "b")
    plt.xticks(np.arange(50, 151, 10))
    plt.xlim([55, 152])
    plt.ylim([-1, 1])
    plt.xlabel(r"$\nu$ [MHz]", fontsize=14)
    plt.ylabel(r"$\Delta$ T [K]")
    plt.legend(["average sw 2018 & 2019"], fontsize=9)

    plt.savefig(path_plot_save + "residuals3.pdf", bbox_inches="tight")
    plt.close()
    plt.close()
    plt.close()
    plt.close()

    if plot_number == 8:
        # Paths
        path_plot_save = config["edges_folder"] + "plots/20190828/"

    GHA1 = 6
    GHA2 = 18

    fx, t188_low_case1, w, s188_low_case1 = io.level3_single_file_test(
        config["edges_folder"]
        + "mid_band/spectra/level3/tests_55_150MHz/rcv18_sw18/2018_188_00.hdf5",
        GHA1,
        GHA2,
        55,
        150,
        False,
        "name",
    )
    fx, t188_low_case2, w, s188_low_case2 = io.level3_single_file_test(
        config["edges_folder"]
        + "mid_band/spectra/level3/tests_55_150MHz/rcv18_sw19/2018_188_00.hdf5",
        GHA1,
        GHA2,
        55,
        150,
        False,
        "name",
    )
    fx, t188_low_case3, w, s188_low_case3 = io.level3_single_file_test(
        config["edges_folder"]
        + "mid_band/spectra/level3/tests_55_150MHz/rcv18_sw18and19_ant147"
        "/2018_188_00.hdf5",
        GHA1,
        GHA2,
        55,
        150,
        False,
        "name",
    )

    f_low = 60
    f_high = 150

    f = fx[(fx >= f_low) & (fx <= f_high)]

    t188_low_case1 = t188_low_case1[(fx >= f_low) & (fx <= f_high)]
    t188_low_case2 = t188_low_case2[(fx >= f_low) & (fx <= f_high)]
    t188_low_case3 = t188_low_case3[(fx >= f_low) & (fx <= f_high)]

    s188_low_case1 = s188_low_case1[(fx >= f_low) & (fx <= f_high)]
    s188_low_case2 = s188_low_case2[(fx >= f_low) & (fx <= f_high)]
    s188_low_case3 = s188_low_case3[(fx >= f_low) & (fx <= f_high)]

    p = mdl.fit_polynomial_fourier(
        "LINLOG", f, t188_low_case1, 5, Weights=1 / (s188_low_case1 ** 2)
    )
    r1 = t188_low_case1 - p[1]

    p = mdl.fit_polynomial_fourier(
        "LINLOG", f, t188_low_case2, 5, Weights=1 / (s188_low_case2 ** 2)
    )
    r2 = t188_low_case2 - p[1]

    p = mdl.fit_polynomial_fourier(
        "LINLOG", f, t188_low_case3, 5, Weights=1 / (s188_low_case3 ** 2)
    )
    r3 = t188_low_case3 - p[1]

    f1 = plt.figure(figsize=[8, 8])

    plt.subplot(3, 1, 1)
    plt.plot(f, r1, "b")
    plt.xticks(np.arange(50, 151, 10), labels="")
    plt.xlim([55, 152])
    plt.ylim([-1, 1])
    plt.title("Day 2018-188, GHA=6-18 hr")
    # plt.xlabel(r'$\nu$ [MHz]', fontsize=14)
    plt.ylabel(r"$\Delta$ T [K]")
    plt.legend(["Case 1"], fontsize=9)

    plt.subplot(3, 1, 2)
    plt.plot(f, r2, "b")
    plt.xticks(np.arange(50, 151, 10), labels="")
    plt.xlim([55, 152])
    plt.ylim([-1, 1])
    plt.ylabel(r"$\Delta$ T [K]")
    plt.legend(["Case 2"], fontsize=9)

    plt.subplot(3, 1, 3)
    plt.plot(f, r3, "b")
    plt.xticks(np.arange(50, 151, 10))
    plt.xlim([55, 152])
    plt.ylim([-1, 1])
    # plt.title('Day 2018-150, GHA=6-18 hr')
    plt.xlabel(r"$\nu$ [MHz]", fontsize=14)
    plt.ylabel(r"$\Delta$ T [K]")
    plt.legend(["average sw 2018 & 2019"], fontsize=9)

    plt.savefig(path_plot_save + "residuals4.pdf", bbox_inches="tight")


def plots_of_absorption_glitch(part_number):
    # TODO: move to old.

    if part_number == 1:
        fmin = 50
        fmax = 200

        fmin_res = 50
        fmax_res = 120

        n_fg = 5

        el = np.arange(0, 91)
        sin_theta = np.sin((90 - el) * (np.pi / 180))
        sin_theta_2D_T = np.tile(sin_theta, (360, 1))
        sin_theta_2D = sin_theta_2D_T.T

        # High-Band Blade on Soil, no GP
        b_all = beams.FEKO_high_band_blade_beam_plus_shaped_finite_ground_plane(
            beam_file=21,
            frequency_interpolation=False,
            frequency=np.array([0]),
            AZ_antenna_axis=0,
        )
        f = np.arange(65, 201, 1)

        bint = np.zeros(len(f))
        for i in range(len(f)):
            b = b_all[i, :, :]
            bint[i] = np.sum(b * sin_theta_2D)

        ft = f[(f >= fmin) & (f <= fmax)]
        bx = bint[(f >= fmin) & (f <= fmax)]
        bt = (1 / (4 * np.pi)) * ((np.pi / 180) ** 2) * bx  # /np.mean(bx)

        fr = ft[(ft >= fmin_res) & (ft <= fmax_res)]
        br = bt[(ft >= fmin_res) & (ft <= fmax_res)]

        x = np.polyfit(fr, br, n_fg - 1)
        m = np.polyval(x, fr)
        rr = br - m

        ft1 = np.copy(ft)
        bt1 = np.copy(bt)

        fr1 = np.copy(fr)
        rr1 = np.copy(rr)

        # High-Band Fourpoint on Plus-Sign GP
        b_all = beams.FEKO_high_band_fourpoint_beam(
            2, frequency_interpolation=False, frequency=np.array([0]), AZ_antenna_axis=0
        )
        f = np.arange(65, 201, 1)

        bint = np.zeros(len(f))
        for i in range(len(f)):
            b = b_all[i, :, :]
            bint[i] = np.sum(b * sin_theta_2D)

        ft = f[(f >= fmin) & (f <= fmax)]
        bx = bint[(f >= fmin) & (f <= fmax)]
        bt = (1 / (4 * np.pi)) * ((np.pi / 180) ** 2) * bx  # /np.mean(bx)

        fr = ft[(ft >= fmin_res) & (ft <= fmax_res)]
        br = bt[(ft >= fmin_res) & (ft <= fmax_res)]

        x = np.polyfit(fr, br, n_fg - 1)
        m = np.polyval(x, fr)
        rr = br - m

        ft2 = np.copy(ft)
        bt2 = np.copy(bt)

        fr2 = np.copy(fr)
        rr2 = np.copy(rr)

        # High-Band Blade on Plus-Sign GP
        b_all = beams.FEKO_high_band_blade_beam_plus_shaped_finite_ground_plane(
            beam_file=20,
            frequency_interpolation=False,
            frequency=np.array([0]),
            AZ_antenna_axis=0,
        )
        f = np.arange(65, 201, 1)

        bint = np.zeros(len(f))
        for i in range(len(f)):
            b = b_all[i, :, :]
            bint[i] = np.sum(b * sin_theta_2D)

        ft = f[(f >= fmin) & (f <= fmax)]
        bx = bint[(f >= fmin) & (f <= fmax)]
        bt = (1 / (4 * np.pi)) * ((np.pi / 180) ** 2) * bx  # /np.mean(bx)

        fr = ft[(ft >= fmin_res) & (ft <= fmax_res)]
        br = bt[(ft >= fmin_res) & (ft <= fmax_res)]

        x = np.polyfit(fr, br, n_fg - 1)
        m = np.polyval(x, fr)
        rr = br - m

        ft3 = np.copy(ft)
        bt3 = np.copy(bt)

        fr3 = np.copy(fr)
        rr3 = np.copy(rr)
        # Low-Band 3, Blade on Plus-Sign GP
        b_all = beams.feko_blade_beam(
            "low_band3", 1, frequency_interpolation=False, az_antenna_axis=90
        )
        f = np.arange(50, 121, 2)

        bint = np.zeros(len(f))
        for i in range(len(f)):
            b = b_all[i, :, :]
            bint[i] = np.sum(b * sin_theta_2D)

        ft = f[(f >= fmin) & (f <= fmax)]
        bx = bint[(f >= fmin) & (f <= fmax)]
        bt = (1 / (4 * np.pi)) * ((np.pi / 180) ** 2) * bx  # /np.mean(bx)

        fr = ft[(ft >= fmin_res) & (ft <= fmax_res)]
        br = bt[(ft >= fmin_res) & (ft <= fmax_res)]

        x = np.polyfit(fr, br, n_fg - 1)
        m = np.polyval(x, fr)
        rr = br - m

        ft4 = np.copy(ft)
        bt4 = np.copy(bt)

        fr4 = np.copy(fr)
        rr4 = np.copy(rr)

        # Figure 2
        # #########################################################################################

        # Mid-Band, Blade, infinite GP
        b_all = beams.feko_blade_beam(
            "mid_band", 1, frequency_interpolation=False, az_antenna_axis=90
        )
        f = np.arange(50, 201, 2)

        bint = np.zeros(len(f))
        for i in range(len(f)):
            b = b_all[i, :, :]
            bint[i] = np.sum(b * sin_theta_2D)

        ft = f[(f >= fmin) & (f <= fmax)]
        bx = bint[(f >= fmin) & (f <= fmax)]
        bt = (1 / (4 * np.pi)) * ((np.pi / 180) ** 2) * bx  # /np.mean(bx)

        fr = ft[(ft >= fmin_res) & (ft <= fmax_res)]
        br = bt[(ft >= fmin_res) & (ft <= fmax_res)]

        x = np.polyfit(fr, br, n_fg - 1)
        m = np.polyval(x, fr)
        rr = br - m

        ft5 = np.copy(ft)
        bt5 = np.copy(bt)

        fr5 = np.copy(fr)
        rr5 = np.copy(rr)

        # Low-Band, Blade on 10m x 10m GP
        b_all = beams.FEKO_low_band_blade_beam(
            beam_file=5,
            frequency_interpolation=False,
            frequency=np.array([0]),
            AZ_antenna_axis=0,
        )
        f = np.arange(50, 121, 2)

        bint = np.zeros(len(f))
        for i in range(len(f)):
            b = b_all[i, :, :]
            bint[i] = np.sum(b * sin_theta_2D)

        ft = f[(f >= fmin) & (f <= fmax)]
        bx = bint[(f >= fmin) & (f <= fmax)]
        bt = (1 / (4 * np.pi)) * ((np.pi / 180) ** 2) * bx  # /np.mean(bx)

        fr = ft[(ft >= fmin_res) & (ft <= fmax_res)]
        br = bt[(ft >= fmin_res) & (ft <= fmax_res)]

        x = np.polyfit(fr, br, n_fg - 1)
        m = np.polyval(x, fr)
        rr = br - m

        ft6 = np.copy(ft)
        bt6 = np.copy(bt)

        fr6 = np.copy(fr)
        rr6 = np.copy(rr)

        # Low-Band, Blade on 30m x 30m GP
        b_all = beams.FEKO_low_band_blade_beam(
            beam_file=2, frequency_interpolation=False, AZ_antenna_axis=0
        )
        f = np.arange(40, 121, 2)

        bint = np.zeros(len(f))
        for i in range(len(f)):
            b = b_all[i, :, :]
            bint[i] = np.sum(b * sin_theta_2D)

        ft = f[(f >= fmin) & (f <= fmax)]
        bx = bint[(f >= fmin) & (f <= fmax)]
        bt = (1 / (4 * np.pi)) * ((np.pi / 180) ** 2) * bx  # /np.mean(bx)

        fr = ft[(ft >= fmin_res) & (ft <= fmax_res)]
        br = bt[(ft >= fmin_res) & (ft <= fmax_res)]

        x = np.polyfit(fr, br, n_fg - 1)
        m = np.polyval(x, fr)
        rr = br - m

        ft7 = np.copy(ft)
        bt7 = np.copy(bt)

        fr7 = np.copy(fr)
        rr7 = np.copy(rr)

        # Low-Band, Blade on 30m x 30m GP, NIVEDITA
        b_all = beams.FEKO_low_band_blade_beam(
            beam_file=0, frequency_interpolation=False, AZ_antenna_axis=0
        )
        f = np.arange(40, 101, 2)

        bint = np.zeros(len(f))
        for i in range(len(f)):
            b = b_all[i, :, :]
            bint[i] = np.sum(b * sin_theta_2D)

        ft = f[(f >= fmin) & (f <= fmax)]
        bx = bint[(f >= fmin) & (f <= fmax)]
        bt = (1 / (4 * np.pi)) * ((np.pi / 180) ** 2) * bx  # /np.mean(bx)

        fr = ft[(ft >= fmin_res) & (ft <= fmax_res)]
        br = bt[(ft >= fmin_res) & (ft <= fmax_res)]

        x = np.polyfit(fr, br, n_fg - 1)
        m = np.polyval(x, fr)
        rr = br - m

        ft8 = np.copy(ft)
        bt8 = np.copy(bt)

        fr8 = np.copy(fr)
        rr8 = np.copy(rr)

        # Mid-Band, Blade, on 30m x 30m GP
        b_all = beams.feko_blade_beam(
            "mid_band", 0, frequency_interpolation=False, az_antenna_axis=90
        )
        f = np.arange(50, 201, 2)

        bint = np.zeros(len(f))
        for i in range(len(f)):
            b = b_all[i, :, :]
            bint[i] = np.sum(b * sin_theta_2D)

        ft = f[(f >= fmin) & (f <= fmax)]
        bx = bint[(f >= fmin) & (f <= fmax)]
        bt = (1 / (4 * np.pi)) * ((np.pi / 180) ** 2) * bx  # /np.mean(bx)

        fr = ft[(ft >= fmin_res) & (ft <= fmax_res)]
        br = bt[(ft >= fmin_res) & (ft <= fmax_res)]

        x = np.polyfit(fr, br, n_fg - 1)
        m = np.polyval(x, fr)
        rr = br - m

        ft9 = np.copy(ft)
        bt9 = np.copy(bt)

        fr9 = np.copy(fr)
        rr9 = np.copy(rr)

        # Mid-Band, Blade, on 30m x 30m GP, NIVEDITA
        b_all = beams.feko_blade_beam(
            "mid_band", 100, frequency_interpolation=False, az_antenna_axis=90
        )
        f = np.arange(60, 201, 2)

        bint = np.zeros(len(f))
        for i in range(len(f)):
            b = b_all[i, :, :]
            bint[i] = np.sum(b * sin_theta_2D)

        ft = f[(f >= fmin) & (f <= fmax)]
        bx = bint[(f >= fmin) & (f <= fmax)]
        bt = (1 / (4 * np.pi)) * ((np.pi / 180) ** 2) * bx  # /np.mean(bx)

        fr = ft[(ft >= fmin_res) & (ft <= fmax_res)]
        br = bt[(ft >= fmin_res) & (ft <= fmax_res)]

        x = np.polyfit(fr, br, n_fg - 1)
        m = np.polyval(x, fr)
        rr = br - m

        ft10 = np.copy(ft)
        bt10 = np.copy(bt)

        fr10 = np.copy(fr)
        rr10 = np.copy(rr)

        plt.figure(1)

        plt.subplot(4, 2, 1)
        plt.plot(ft1, bt1)
        plt.xlim([45, 205])
        plt.title(r"Integrated gain above horizon / $4\pi$")
        plt.ylabel("High-Band Blade\n no GP")

        plt.subplot(4, 2, 2)
        plt.plot(fr1, rr1)
        plt.xlim([45, 125])
        plt.ylim(-0.00029, 0.00029)
        plt.title("Residuals")

        plt.subplot(4, 2, 3)
        plt.plot(ft2, bt2)
        plt.xlim([45, 205])
        plt.ylabel("High-Band Fourpoint\n Plus-sign GP")

        plt.subplot(4, 2, 4)
        plt.plot(fr2, rr2)
        plt.xlim([45, 125])
        plt.ylim(-0.00029, 0.00029)

        plt.subplot(4, 2, 5)
        plt.plot(ft3, bt3)
        plt.xlim([45, 205])
        plt.ylabel("High-Band Blade\n Plus-sign GP")

        plt.subplot(4, 2, 6)
        plt.plot(fr3, rr3)
        plt.xlim([45, 125])
        plt.ylim(-0.00029, 0.00029)

        plt.subplot(4, 2, 7)
        plt.plot(ft4, bt4)
        plt.xlim([45, 205])
        plt.ylabel("Low-Band 3 Blade\n Plus-sign GP")
        plt.xlabel("frequency [MHz]")

        plt.subplot(4, 2, 8)
        plt.plot(fr4, rr4)
        plt.xlim([45, 125])
        plt.ylim(-0.00029, 0.00029)
        plt.xlabel("frequency [MHz]")

        plt.figure(2)

        plt.subplot(6, 2, 1)
        plt.plot(ft5, bt5)
        plt.xlim([45, 205])
        plt.title(r"Integrated gain above horizon / $4\pi$")
        plt.ylabel("Mid-Band Blade\n Infinite GP")

        plt.subplot(6, 2, 2)
        plt.plot(fr5, rr5)
        plt.xlim([45, 125])
        plt.ylim(-0.00029, 0.00029)
        plt.title("Residuals")

        plt.subplot(6, 2, 3)
        plt.plot(ft6, bt6)
        plt.xlim([45, 205])
        plt.ylabel("Low-Band Blade\n 10m x 10m GP")

        plt.subplot(6, 2, 4)
        plt.plot(fr6, rr6)
        plt.ylim(-0.00029, 0.00029)
        plt.xlim([45, 125])

        plt.subplot(6, 2, 5)
        plt.plot(ft7, bt7)
        plt.xlim([45, 205])
        plt.ylabel("Low-Band Blade\n 30m x 30m GP")

        plt.subplot(6, 2, 6)
        plt.plot(fr7, rr7)
        plt.ylim(-0.00029, 0.00029)
        plt.xlim([45, 125])

        plt.subplot(6, 2, 7)
        plt.plot(ft8, bt8)
        plt.xlim([45, 205])
        plt.ylabel("Low-Band Blade\n 30m x 30m GP\n NIVEDITA")

        plt.subplot(6, 2, 8)
        plt.plot(fr8, rr8)
        plt.ylim(-0.00029, 0.00029)
        plt.xlim([45, 125])
        plt.subplot(6, 2, 9)
        plt.plot(ft9, bt9)
        plt.xlim([45, 205])
        plt.ylabel("Mid-Band Blade\n 30m x 30m GP")

        plt.subplot(6, 2, 10)
        plt.plot(fr9, rr9)
        plt.ylim(-0.00029, 0.00029)
        plt.xlim([45, 125])

        plt.subplot(6, 2, 11)
        plt.plot(ft10, bt10)
        plt.xlim([45, 205])
        plt.xlabel("frequency [MHz]")
        plt.ylabel("Mid-Band Blade\n 30m x 30m GP\n NIVEDITA")

        plt.subplot(6, 2, 12)
        plt.plot(fr10, rr10)
        plt.ylim(-0.00029, 0.00029)
        plt.xlim([45, 125])
        plt.xlabel("frequency [MHz]")
    elif part_number == 2:
        f = np.genfromtxt(
            "/run/media/raul/WD_RED_6TB/EDGES_vol2/low_band3/calibration/beam_factors/raw"
            "/gain_glitch_test_freq.txt"
        )
        t = np.genfromtxt(
            "/run/media/raul/WD_RED_6TB/EDGES_vol2/low_band3/calibration/beam_factors/raw"
            "/gain_glitch_test_tant.txt"
        )

        gah = np.genfromtxt(
            "/run/media/raul/WD_RED_6TB/EDGES_vol2/low_band3/calibration/beam_factors/raw"
            "/gain_glitch_test_int_gain_above_horizon.txt"
        )
        bf = np.genfromtxt(
            "/run/media/raul/WD_RED_6TB/EDGES_vol2/low_band3/calibration/beam_factors/raw"
            "/gain_glitch_test_data.txt"
        )

        t10 = t[11]
        t53 = t[53]

        bf10 = bf[11]
        bf53 = bf[53]

        fr = f[(f >= 60) & (f <= 120)]
        t10 = t10[(f >= 60) & (f <= 120)]
        t53 = t53[(f >= 60) & (f <= 120)]

        gg = gah[(f >= 60) & (f <= 120)]

        bf10 = bf10[(f >= 60) & (f <= 120)]
        bf53 = bf53[(f >= 60) & (f <= 120)]

        t10_corr = t10 * gg / 3145728 + 300 * (1 - (gg / 3145728))
        t53_corr = t53 * gg / 3145728 + 300 * (1 - (gg / 3145728))

        bf10_corr = bf10 * (gg / gg[13])
        bf53_corr = bf53 * (gg / gg[13])

        p10_1 = mdl.fit_polynomial_fourier("LINLOG", fr, t10_corr, 6)
        p10_2 = mdl.fit_polynomial_fourier("LINLOG", fr, t10, 6)

        p53_1 = mdl.fit_polynomial_fourier("LINLOG", fr, t53_corr, 6)
        p53_2 = mdl.fit_polynomial_fourier("LINLOG", fr, t53, 6)

        p_bf_10_1 = mdl.fit_polynomial_fourier("LINLOG", fr, bf10_corr, 6)
        p_bf_10_2 = mdl.fit_polynomial_fourier("LINLOG", fr, bf10, 6)

        p_bf_53_1 = mdl.fit_polynomial_fourier("LINLOG", fr, bf53_corr, 6)
        p_bf_53_2 = mdl.fit_polynomial_fourier("LINLOG", fr, bf53, 6)

        k10_corr = (t10_corr - 300 * (1 - (gg / 3145728))) / bf10_corr
        k10 = t10_corr / bf10

        k53_corr = (t53_corr - 300 * (1 - (gg / 3145728))) / bf53_corr
        k53 = t53_corr / bf53

        p_k10_corr = mdl.fit_polynomial_fourier("LINLOG", fr, k10_corr, 6)
        p_k10 = mdl.fit_polynomial_fourier("LINLOG", fr, k10, 6)

        p_k53_corr = mdl.fit_polynomial_fourier("LINLOG", fr, k53_corr, 6)
        p_k53 = mdl.fit_polynomial_fourier("LINLOG", fr, k53, 6)

        plt.close()

        plt.figure(1)

        plt.subplot(2, 2, 1)
        plt.plot(fr, t10_corr)
        plt.plot(fr, t10, "--")
        plt.ylabel("temperature [K]")
        plt.title("Low Foreground (GHA = 10 hr)")

        plt.subplot(2, 2, 2)
        plt.plot(fr, t53_corr)
        plt.plot(fr, t53, "--")
        plt.title("High Foreground (GHA = 0 hr)")

        plt.subplot(2, 2, 3)
        plt.plot(fr, t10_corr - p10_1[1])
        plt.plot(fr, t10 - p10_2[1], "--")
        plt.ylim([-0.5, 0.5])
        plt.xlabel("frequency [MHz]")
        plt.ylabel(r"$\Delta$ temperature [K]")
        plt.legend(["correct", "incorrect"])

        plt.subplot(2, 2, 4)
        plt.plot(fr, t53_corr - p53_1[1])
        plt.plot(fr, t53 - p53_2[1], "--")
        plt.ylim([-0.5, 0.5])
        plt.xlabel("frequency [MHz]")
        plt.figure(2)

        plt.subplot(2, 2, 1)
        plt.plot(fr, bf10_corr)
        plt.plot(fr, bf10, "--")
        plt.ylabel("beam correction factor")
        plt.title("Low Foreground (GHA = 10 hr)")

        plt.subplot(2, 2, 2)
        plt.plot(fr, bf53_corr)
        plt.plot(fr, bf53, "--")
        plt.title("High Foreground (GHA = 0 hr)")

        plt.subplot(2, 2, 3)
        plt.plot(fr, bf10_corr - p_bf_10_1[1])
        plt.plot(fr, bf10 - p_bf_10_2[1], "--")
        plt.ylim([-0.0004, 0.0004])
        plt.xlabel("frequency [MHz]")
        plt.ylabel(r"$\Delta$ temperature [K]")
        plt.legend(["correct", "incorrect"])

        plt.subplot(2, 2, 4)
        plt.plot(fr, bf53_corr - p_bf_53_1[1])
        plt.plot(fr, bf53 - p_bf_53_2[1], "--")
        plt.ylim([-0.0004, 0.0004])
        plt.xlabel("frequency [MHz]")

        plt.figure(3)

        plt.subplot(2, 2, 1)
        plt.plot(fr, k10_corr)
        plt.ylabel("temperature [K]")
        plt.title("Low Foreground (GHA = 10 hr)")

        plt.subplot(2, 2, 2)
        plt.plot(fr, t53_corr)
        plt.title("High Foreground (GHA = 0 hr)")

        plt.subplot(2, 2, 3)
        plt.plot(fr, k10_corr - p_k10_corr[1])
        plt.plot(fr, k10 - p_k10[1], "--")
        plt.ylim([-0.5, 0.5])
        plt.xlabel("frequency [MHz]")
        plt.ylabel(r"$\Delta$ temperature [K]")
        plt.legend(["correct", "incorrect"])

        plt.subplot(2, 2, 4)
        plt.plot(fr, k53_corr - p_k53_corr[1])
        plt.plot(fr, k53 - p_k53[1], "--")
        plt.ylim([-0.5, 0.5])
        plt.xlabel("frequency [MHz]")


def high_band_2015_reanalysis():
    # TODO: move to old
    LST1 = 1
    LST2 = 11

    f_low = 80
    f_high = 140

    n_fg = 5

    filename_list = [
        "2015_251_00.hdf5",
        "2015_252_00.hdf5",
        "2015_253_00.hdf5",
        "2015_254_00.hdf5",
        "2015_256_00.hdf5",
        "2015_257_00.hdf5",
        "2015_258_00.hdf5",
        "2015_259_00.hdf5",
    ]

    for i in range(len(filename_list)):
        f, r, p, w, rms, m = io.level3read(
            "/run/media/raul/WD_BLACK_6TB/EDGES_vol1/spectra/level3/high_band_2015/2018_analysis"
            "/case101/" + filename_list[i]
        )

        index = np.arange(len(r[:, 0]))
        index_selected = index[(m[:, 3] >= LST1) & (m[:, 3] <= LST2)]

        avr, avw = tools.spectral_averaging(r[index_selected, :], w[index_selected, :])
        avp = np.mean(p[index, :], axis=0)

        if i == 0:
            r_all = np.zeros((len(filename_list), len(f)))
            w_all = np.zeros((len(filename_list), len(f)))
            p_all = np.zeros((len(filename_list), len(avp)))

        r_all[i, :] = avr
        w_all[i, :] = avw
        p_all[i, :] = avp

    rr, ww = tools.spectral_averaging(r_all, w_all)
    pp = np.mean(p_all, axis=0)

    fb, rb, wb, sb = tools.spectral_binning_number_of_samples(f, rr, ww, nsamples=128)

    av_mf = np.polyval(pp, fb / 200)
    avmb = av_mf * ((fb / 200) ** (-2.5))

    tb = rb + avmb

    fk = fb[(fb >= f_low) & (fb <= f_high)]
    tk = tb[(fb >= f_low) & (fb <= f_high)]
    sk = sb[(fb >= f_low) & (fb <= f_high)]

    p1 = mdl.fit_polynomial_fourier("LINLOG", fk, tk, n_fg, Weights=(1 / sk) ** 2)

    signal = dm.signal_model("tanh", [-0.7, 79, 22, 7, 8], fk)
    p2 = mdl.fit_polynomial_fourier(
        "LINLOG", fk, tk - signal, n_fg, Weights=(1 / sk) ** 2
    )

    plt.close()
    plt.close()
    plt.plot(fk, tk - p1[1])
    plt.plot(fk, tk - signal - p2[1] - 0.6)
    plt.ylim([-0.8, 0.4])
    plt.yticks([-0.6, -0.4, -0.2, 0, 0.2], labels=["", "", ""])

    plt.xlabel("frequency [MHz]")
    plt.ylabel(r"$\Delta$ temperature [0.2 K per division]")
    plt.legend(["Foreground", "Foreground + Signal"])

    return fk, tk


def comparison_FEKO_HFSS():
    # TODO: this went into a memo. Move to old.
    theta, phi, b60 = beams.hfss_read(
        "/run/media/raul/WD_RED_6TB/EDGES_vol2/mid_band/calibration/beam/raul/hfss/20191002"
        "/pec_60MHz.csv",
        "linear",
        theta_min=0,
        theta_max=180,
        theta_resolution=1,
        phi_min=0,
        phi_max=359,
        phi_resolution=1,
    )
    H60 = b60[theta <= 90, :]

    theta, phi, b90 = beams.hfss_read(
        "/run/media/raul/WD_RED_6TB/EDGES_vol2/mid_band/calibration/beam/raul/hfss/20191002"
        "/pec_90MHz.csv",
        "linear",
        theta_min=0,
        theta_max=180,
        theta_resolution=1,
        phi_min=0,
        phi_max=359,
        phi_resolution=1,
    )
    H90 = b90[theta <= 90, :]

    theta, phi, b120 = beams.hfss_read(
        "/run/media/raul/WD_RED_6TB/EDGES_vol2/mid_band/calibration/beam/raul/hfss/20191002"
        "/pec_120MHz.csv",
        "linear",
        theta_min=0,
        theta_max=180,
        theta_resolution=1,
        phi_min=0,
        phi_max=359,
        phi_resolution=1,
    )
    H120 = b120[theta <= 90, :]

    bm = beams.feko_blade_beam(
        "mid_band",
        1,
        frequency_interpolation=False,
        frequency=np.array([0]),
        az_antenna_axis=0,
    )
    F60 = np.flipud(bm[5])
    F90 = np.flipud(bm[20])
    F120 = np.flipud(bm[35])

    plt.figure(1)
    plt.subplot(2, 1, 1)
    plt.imshow(F60, interpolation=None, aspect="auto")
    cbar = plt.colorbar()
    cbar.set_label("directive gain [linear]", rotation=90)
    plt.title("Feko, 60 MHz")
    plt.ylabel("theta [deg]")

    plt.subplot(2, 1, 2)
    plt.imshow(H60 - F60, interpolation=None, aspect="auto")
    cbar = plt.colorbar()
    cbar.set_label(r"$\Delta$ directive gain [linear]", rotation=90)
    plt.title("(HFSS - Feko), 60 MHz")
    plt.ylabel("theta [deg]")
    plt.xlabel("phi [deg]")

    plt.figure(2)
    plt.subplot(2, 1, 1)
    plt.imshow(F90, interpolation=None, aspect="auto")
    cbar = plt.colorbar()
    cbar.set_label("directive gain [linear]", rotation=90)
    plt.title("Feko, 90 MHz")
    plt.ylabel("theta [deg]")

    plt.subplot(2, 1, 2)
    plt.imshow(H90 - F90, interpolation=None, aspect="auto")
    cbar = plt.colorbar()
    cbar.set_label(r"$\Delta$ directive gain [linear]", rotation=90)
    plt.title("(HFSS - Feko), 90 MHz")
    plt.ylabel("theta [deg]")
    plt.xlabel("phi [deg]")

    plt.figure(3)
    plt.subplot(2, 1, 1)
    plt.imshow(F120, interpolation=None, aspect="auto")
    cbar = plt.colorbar()
    cbar.set_label("directive gain [linear]", rotation=90)
    plt.title("Feko, 120 MHz")
    plt.ylabel("theta [deg]")

    plt.subplot(2, 1, 2)
    plt.imshow(H120 - F120, interpolation=None, aspect="auto")
    cbar = plt.colorbar()
    cbar.set_label(r"$\Delta$ directive gain [linear]", rotation=90)
    plt.title("(HFSS - Feko), 120 MHz")
    plt.ylabel("theta [deg]")
    plt.xlabel("phi [deg]")

    return H60, H90, H120, F60, F90, F120


def comparison_FEKO_WIPLD():
    # TODO: move to old.
    path_plot_save = config["edges_folder"] + "plots/20191015/"

    # WIPL-D
    filename = (
        "/run/media/raul/WD_RED_6TB/EDGES_vol2/others/beam_simulations/wipl-d/20191012"
        "/blade_dipole_infinite_PEC.ra1"
    )
    f, thetaX, phiX, beam = beams.wipld_read(filename)

    theta = thetaX[thetaX <= 90]
    m = beam[:, thetaX <= 90, :]
    W60 = m[f == 60, :, :][0]
    W90 = m[f == 90, :, :][0]
    W120 = m[f == 120, :, :][0]

    sin_theta = np.sin(theta * (np.pi / 180))
    sin_theta_2D_T = np.tile(sin_theta, (360, 1))
    sin_theta_2D = sin_theta_2D_T.T

    bint = np.zeros(len(f))
    for i in range(len(f)):
        b = m[i, :, :]
        bint[i] = np.sum(b * sin_theta_2D)

    btW = (1 / (4 * np.pi)) * ((np.pi / 180) ** 2) * bint  # /np.mean(bx)

    n_fg = 7
    f_high = 150
    fX = f[f <= f_high]
    btWX = btW[f <= f_high]

    x = np.polyfit(fX, btWX, n_fg - 1)
    model = np.polyval(x, fX)
    rtWX = btWX - model

    deltaW = np.zeros((len(m[:, 0, 0]) - 1, len(m[0, :, 0]), len(m[0, 0, :])))
    for i in range(len(f) - 1):
        deltaW[i, :, :] = m[i + 1, :, :] - m[i, :, :]

    # HFSS
    thetaX, phi, b60 = beams.hfss_read(
        config["edges_folder"]
        + "others/beam_simulations/hfss/20191002/mid_band_infinite_pec/60MHz.csv",
        "linear",
        theta_min=0,
        theta_max=180,
        theta_resolution=1,
        phi_min=0,
        phi_max=359,
        phi_resolution=1,
    )
    theta = thetaX[thetaX <= 90]
    H60 = b60[thetaX <= 90, :]

    thetaX, phi, b90 = beams.hfss_read(
        config["edges_folder"]
        + "others/beam_simulations/hfss/20191002/mid_band_infinite_pec/90MHz.csv",
        "linear",
        theta_min=0,
        theta_max=180,
        theta_resolution=1,
        phi_min=0,
        phi_max=359,
        phi_resolution=1,
    )
    H90 = b90[thetaX <= 90, :]

    thetaX, phi, b120 = beams.hfss_read(
        config["edges_folder"]
        + "others/beam_simulations/hfss/20191002/mid_band_infinite_pec/120MHz.csv",
        "linear",
        theta_min=0,
        theta_max=180,
        theta_resolution=1,
        phi_min=0,
        phi_max=359,
        phi_resolution=1,
    )
    H120 = b120[thetaX <= 90, :]

    thetaX, phi, b65 = beams.hfss_read(
        config["edges_folder"]
        + "others/beam_simulations/hfss/20191002/mid_band_infinite_pec/65MHz.csv",
        "linear",
        theta_min=0,
        theta_max=90,
        theta_resolution=1,
        phi_min=0,
        phi_max=359,
        phi_resolution=1,
    )
    H65 = b65[thetaX <= 90, :]

    thetaX, phi, b66 = beams.hfss_read(
        config["edges_folder"]
        + "others/beam_simulations/hfss/20191002/mid_band_infinite_pec/66MHz.csv",
        "linear",
        theta_min=0,
        theta_max=90,
        theta_resolution=1,
        phi_min=0,
        phi_max=359,
        phi_resolution=1,
    )
    H66 = b66[thetaX <= 90, :]

    thetaX, phi, b67 = beams.hfss_read(
        config["edges_folder"]
        + "others/beam_simulations/hfss/20191002/mid_band_infinite_pec/67MHz.csv",
        "linear",
        theta_min=0,
        theta_max=90,
        theta_resolution=1,
        phi_min=0,
        phi_max=359,
        phi_resolution=1,
    )
    H67 = b67[thetaX <= 90, :]

    thetaX, phi, b68 = beams.hfss_read(
        config["edges_folder"]
        + "others/beam_simulations/hfss/20191002/mid_band_infinite_pec/68MHz.csv",
        "linear",
        theta_min=0,
        theta_max=90,
        theta_resolution=1,
        phi_min=0,
        phi_max=359,
        phi_resolution=1,
    )
    H68 = b68[thetaX <= 90, :]

    thetaX, phi, b69 = beams.hfss_read(
        config["edges_folder"]
        + "others/beam_simulations/hfss/20191002/mid_band_infinite_pec/69MHz.csv",
        "linear",
        theta_min=0,
        theta_max=90,
        theta_resolution=1,
        phi_min=0,
        phi_max=359,
        phi_resolution=1,
    )
    H69 = b69[thetaX <= 90, :]

    thetaX, phi, b70 = beams.hfss_read(
        config["edges_folder"]
        + "others/beam_simulations/hfss/20191002/mid_band_infinite_pec/70MHz.csv",
        "linear",
        theta_min=0,
        theta_max=90,
        theta_resolution=1,
        phi_min=0,
        phi_max=359,
        phi_resolution=1,
    )
    H70 = b70[thetaX <= 90, :]

    thetaX, phi, b71 = beams.hfss_read(
        config["edges_folder"]
        + "others/beam_simulations/hfss/20191002/mid_band_infinite_pec/71MHz.csv",
        "linear",
        theta_min=0,
        theta_max=90,
        theta_resolution=1,
        phi_min=0,
        phi_max=359,
        phi_resolution=1,
    )
    H71 = b71[thetaX <= 90, :]

    thetaX, phi, b72 = beams.hfss_read(
        config["edges_folder"]
        + "others/beam_simulations/hfss/20191002/mid_band_infinite_pec/72MHz.csv",
        "linear",
        theta_min=0,
        theta_max=90,
        theta_resolution=1,
        phi_min=0,
        phi_max=359,
        phi_resolution=1,
    )
    H72 = b72[thetaX <= 90, :]

    thetaX, phi, b73 = beams.hfss_read(
        config["edges_folder"]
        + "others/beam_simulations/hfss/20191002/mid_band_infinite_pec/73MHz.csv",
        "linear",
        theta_min=0,
        theta_max=90,
        theta_resolution=1,
        phi_min=0,
        phi_max=359,
        phi_resolution=1,
    )
    H73 = b73[thetaX <= 90, :]

    thetaX, phi, b74 = beams.hfss_read(
        config["edges_folder"]
        + "others/beam_simulations/hfss/20191002/mid_band_infinite_pec/74MHz.csv",
        "linear",
        theta_min=0,
        theta_max=90,
        theta_resolution=1,
        phi_min=0,
        phi_max=359,
        phi_resolution=1,
    )
    H74 = b74[thetaX <= 90, :]

    thetaX, phi, b75 = beams.hfss_read(
        config["edges_folder"]
        + "others/beam_simulations/hfss/20191002/mid_band_infinite_pec/75MHz.csv",
        "linear",
        theta_min=0,
        theta_max=90,
        theta_resolution=1,
        phi_min=0,
        phi_max=359,
        phi_resolution=1,
    )
    H75 = b75[thetaX <= 90, :]

    sin_theta = np.sin(theta * (np.pi / 180))
    sin_theta_2D_T = np.tile(sin_theta, (360, 1))
    sin_theta_2D = sin_theta_2D_T.T

    bint60 = np.sum(H60 * sin_theta_2D)
    btH60 = (1 / (4 * np.pi)) * ((np.pi / 180) ** 2) * bint60

    bint90 = np.sum(H90 * sin_theta_2D)
    btH90 = (1 / (4 * np.pi)) * ((np.pi / 180) ** 2) * bint90

    bint120 = np.sum(H120 * sin_theta_2D)
    btH120 = (1 / (4 * np.pi)) * ((np.pi / 180) ** 2) * bint120

    bint65 = np.sum(H65 * sin_theta_2D)
    btH65 = (1 / (4 * np.pi)) * ((np.pi / 180) ** 2) * bint65

    bint66 = np.sum(H66 * sin_theta_2D)
    btH66 = (1 / (4 * np.pi)) * ((np.pi / 180) ** 2) * bint66

    bint67 = np.sum(H67 * sin_theta_2D)
    btH67 = (1 / (4 * np.pi)) * ((np.pi / 180) ** 2) * bint67

    bint68 = np.sum(H68 * sin_theta_2D)
    btH68 = (1 / (4 * np.pi)) * ((np.pi / 180) ** 2) * bint68

    bint69 = np.sum(H69 * sin_theta_2D)
    btH69 = (1 / (4 * np.pi)) * ((np.pi / 180) ** 2) * bint69

    bint70 = np.sum(H70 * sin_theta_2D)
    btH70 = (1 / (4 * np.pi)) * ((np.pi / 180) ** 2) * bint70
    bint71 = np.sum(H71 * sin_theta_2D)
    btH71 = (1 / (4 * np.pi)) * ((np.pi / 180) ** 2) * bint71

    bint72 = np.sum(H72 * sin_theta_2D)
    btH72 = (1 / (4 * np.pi)) * ((np.pi / 180) ** 2) * bint72

    bint73 = np.sum(H73 * sin_theta_2D)
    btH73 = (1 / (4 * np.pi)) * ((np.pi / 180) ** 2) * bint73

    bint74 = np.sum(H74 * sin_theta_2D)
    btH74 = (1 / (4 * np.pi)) * ((np.pi / 180) ** 2) * bint74

    bint75 = np.sum(H75 * sin_theta_2D)
    btH75 = (1 / (4 * np.pi)) * ((np.pi / 180) ** 2) * bint75

    fHX = np.array([60, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 90, 120])
    btHX = np.array(
        [
            btH60,
            btH65,
            btH66,
            btH67,
            btH68,
            btH69,
            btH70,
            btH71,
            btH72,
            btH73,
            btH74,
            btH75,
            btH90,
            btH120,
        ]
    )

    n_fg = 4
    x = np.polyfit(fHX, btHX, n_fg - 1)
    model = np.polyval(x, fHX)
    rtHX = btHX - model

    # FEKO
    bm = beams.feko_blade_beam(
        "mid_band",
        1,
        frequency_interpolation=False,
        frequency=np.array([0]),
        az_antenna_axis=0,
    )
    F60 = np.flipud(bm[5])
    F90 = np.flipud(bm[20])
    F120 = np.flipud(bm[35])

    f = np.arange(50, 201, 2)
    el = np.arange(0, 91)
    sin_theta = np.sin((90 - el) * (np.pi / 180))
    sin_theta_2D_T = np.tile(sin_theta, (360, 1))
    sin_theta_2D = sin_theta_2D_T.T

    bint = np.zeros(len(f))
    for i in range(len(f)):
        b = bm[i, :, :]
        bint[i] = np.sum(b * sin_theta_2D)

    btF = (1 / (4 * np.pi)) * ((np.pi / 180) ** 2) * bint  # /np.mean(bx)

    n_fg = 7
    fX = f[f <= f_high]
    btFX = btF[f <= f_high]

    x = np.polyfit(fX, btFX, n_fg - 1)
    model = np.polyval(x, fX)
    rtFX = btFX - model

    deltaF = np.zeros((len(bm[:, 0, 0]) - 1, len(bm[0, :, 0]), len(bm[0, 0, :])))
    for i in range(len(f) - 1):
        XX = bm[i + 1, :, :] - bm[i, :, :]
        XXX = np.flipud(XX)
        deltaF[i, :, :] = XXX

    plt.figure(6, figsize=(8, 10))
    plt.subplot(2, 1, 1)
    plt.plot(f, btF, "b")
    plt.plot(f, btW, "r--")
    plt.plot(fHX, btHX, "g.-")
    plt.xticks([50, 75, 100, 125, 150, 175, 200], labels=[])
    plt.ylabel("integrated gain above horizon [fraction of 4 pi]")
    plt.legend(["FEKO", "WIPL-D", "HFSS-IE"], loc=3)

    plt.subplot(2, 1, 2)
    plt.plot([70, 70], [-1.5e-5, 1.5e-5], "c:")
    plt.plot([95, 95], [-1.5e-5, 1.5e-5], "c:")
    plt.text(67, -0.5e-5, "70 MHz", rotation=90)
    plt.text(92, -0.5e-5, "95 MHz", rotation=90)
    plt.plot(fX, rtWX + 0e-5, "r--")
    plt.plot(fX, rtFX + 1e-5, "b")
    plt.plot(fHX, rtHX - 1e-5, "g.-")
    plt.ylim([-1.5e-5, 1.5e-5])
    plt.xticks([50, 75, 100, 125, 150, 175, 200])
    plt.yticks(np.arange(-1e-5, 1.1e-5, 5e-6), labels=[])
    plt.xlabel("frequency [MHz]")
    plt.ylabel("fit residuals [fraction of 4 pi]\n(5e-6 per division)")

    plt.savefig(path_plot_save + "fig6.pdf", bbox_inches="tight")
    plt.close()
    plt.close()

    return (
        W60,
        W90,
        W120,
        F60,
        F90,
        F120,
        f,
        btW,
        btF,
        fX,
        rtWX,
        rtFX,
        btH60,
        btH90,
        btH120,
        deltaF,
        deltaW,
    )


def integrated_antenna_gain_WIPLD(case, n_fg):
    # TODO: mark as old

    # WIPL-D
    filename0 = (
        "/run/media/raul/WD_RED_6TB/EDGES_vol2/others/beam_simulations/wipl-d/20191023"
        "/blade_dipole_free_space.ra1"
    )

    filename1 = (
        "/run/media/raul/WD_RED_6TB/EDGES_vol2/others/beam_simulations/wipl-d/20191023"
        "/blade_dipole_infinite_soil_3.5_0.02.ra1"
    )
    filename2 = (
        "/run/media/raul/WD_RED_6TB/EDGES_vol2/others/beam_simulations/wipl-d/20191023"
        "/blade_dipole_infinite_soil_2.5_0.02.ra1"
    )
    filename3 = (
        "/run/media/raul/WD_RED_6TB/EDGES_vol2/others/beam_simulations/wipl-d/20191023"
        "/blade_dipole_infinite_soil_4.5_0.02.ra1"
    )
    filename4 = (
        "/run/media/raul/WD_RED_6TB/EDGES_vol2/others/beam_simulations/wipl-d/20191023"
        "/blade_dipole_infinite_soil_3.5_0.002.ra1"
    )
    filename5 = (
        "/run/media/raul/WD_RED_6TB/EDGES_vol2/others/beam_simulations/wipl-d/20191023"
        "/blade_dipole_infinite_soil_3.5_0.2.ra1"
    )

    filename11 = (
        "/run/media/raul/WD_RED_6TB/EDGES_vol2/others/beam_simulations/wipl-d/20191023"
        "/blade_dipole_infinite_soil_metal_GP_1m_side_0.0000001m_height.ra1"
    )
    filename12 = (
        "/run/media/raul/WD_RED_6TB/EDGES_vol2/others/beam_simulations/wipl-d/20191023"
        "/blade_dipole_infinite_soil_metal_GP_1m_side_0.01m_height.ra1"
    )
    filename13 = (
        "/run/media/raul/WD_RED_6TB/EDGES_vol2/others/beam_simulations/wipl-d/20191023"
        "/blade_dipole_infinite_soil_metal_GP_1m_side_0.0000001m_height_2.5_0.02.ra1"
    )
    filename14 = (
        "/run/media/raul/WD_RED_6TB/EDGES_vol2/others/beam_simulations/wipl-d/20191023"
        "/blade_dipole_infinite_soil_metal_GP_1m_side_0.0000001m_height_4.5_0.02.ra1"
    )
    filename15 = (
        "/run/media/raul/WD_RED_6TB/EDGES_vol2/others/beam_simulations/wipl-d/20191023"
        "/blade_dipole_infinite_soil_metal_GP_1m_side_0.0000001m_height_3.5_0.002.ra1"
    )
    filename16 = (
        "/run/media/raul/WD_RED_6TB/EDGES_vol2/others/beam_simulations/wipl-d/20191023"
        "/blade_dipole_infinite_soil_metal_GP_1m_side_0.0000001m_height_3.5_0.2.ra1"
    )

    filename51 = (
        "/run/media/raul/WD_RED_6TB/EDGES_vol2/others/beam_simulations/wipl-d/20191023"
        "/blade_dipole_infinite_soil_metal_GP_5m_side_0.0000001m_height.ra1"
    )
    filename52 = (
        "/run/media/raul/WD_RED_6TB/EDGES_vol2/others/beam_simulations/wipl-d/20191023"
        "/blade_dipole_infinite_soil_metal_GP_5m_side_0.01m_height.ra1"
    )
    filename53 = (
        "/run/media/raul/WD_RED_6TB/EDGES_vol2/others/beam_simulations/wipl-d/20191023"
        "/blade_dipole_infinite_soil_metal_GP_5m_side_0.0000001m_height_2.5_0.02.ra1"
    )
    filename54 = (
        "/run/media/raul/WD_RED_6TB/EDGES_vol2/others/beam_simulations/wipl-d/20191023"
        "/blade_dipole_infinite_soil_metal_GP_5m_side_0.0000001m_height_4.5_0.02.ra1"
    )
    filename55 = (
        "/run/media/raul/WD_RED_6TB/EDGES_vol2/others/beam_simulations/wipl-d/20191023"
        "/blade_dipole_infinite_soil_metal_GP_5m_side_0.0000001m_height_3.5_0.002.ra1"
    )
    filename56 = (
        "/run/media/raul/WD_RED_6TB/EDGES_vol2/others/beam_simulations/wipl-d/20191023"
        "/blade_dipole_infinite_soil_metal_GP_5m_side_0.0000001m_height_3.5_0.2.ra1"
    )

    filename101 = (
        "/run/media/raul/WD_RED_6TB/EDGES_vol2/others/beam_simulations/wipl-d/20191023"
        "/blade_dipole_infinite_soil_metal_GP_10m_side_0.0000001m_height.ra1"
    )
    filename102 = (
        "/run/media/raul/WD_RED_6TB/EDGES_vol2/others/beam_simulations/wipl-d/20191023"
        "/blade_dipole_infinite_soil_metal_GP_10m_side_0.01m_height.ra1"
    )
    filename103 = (
        "/run/media/raul/WD_RED_6TB/EDGES_vol2/others/beam_simulations/wipl-d/20191023"
        "/blade_dipole_infinite_soil_metal_GP_10m_side_0.0000001m_height_2.5_0.02.ra1"
    )
    filename104 = (
        "/run/media/raul/WD_RED_6TB/EDGES_vol2/others/beam_simulations/wipl-d/20191023"
        "/blade_dipole_infinite_soil_metal_GP_10m_side_0.0000001m_height_4.5_0.02.ra1"
    )
    filename105 = (
        "/run/media/raul/WD_RED_6TB/EDGES_vol2/others/beam_simulations/wipl-d/20191023"
        "/blade_dipole_infinite_soil_metal_GP_10m_side_0.0000001m_height_3.5_0.002.ra1"
    )
    filename106 = (
        "/run/media/raul/WD_RED_6TB/EDGES_vol2/others/beam_simulations/wipl-d/20191023"
        "/blade_dipole_infinite_soil_metal_GP_10m_side_0.0000001m_height_3.5_0.2.ra1"
    )

    filename200 = (
        "/run/media/raul/WD_RED_6TB/EDGES_vol2/others/beam_simulations/wipl-d/20191030"
        "/blade_dipole_infinite_soil_metal_GP_30mx30m_0.0000001m_height_3.5_0.02.ra1"
    )
    filename201 = (
        "/run/media/raul/WD_RED_6TB/EDGES_vol2/others/beam_simulations/wipl-d/20191030"
        "/blade_dipole_infinite_soil_metal_GP_single_precision.ra1"
    )
    filename202 = (
        "/run/media/raul/WD_RED_6TB/EDGES_vol2/others/beam_simulations/wipl-d/20191030"
        "/blade_dipole_infinite_soil_metal_GP_double_precision.ra1"
    )
    filename203 = (
        "/run/media/raul/WD_RED_6TB/EDGES_vol2/others/beam_simulations/wipl-d/20191030"
        "/blade_dipole_infinite_soil_metal_GP_0.01_single_precision.ra1"
    )
    filename204 = (
        "/run/media/raul/WD_RED_6TB/EDGES_vol2/others/beam_simulations/wipl-d/20191030"
        "/blade_dipole_infinite_soil_metal_GP_0.01_double_precision.ra1"
    )
    filename205 = (
        "/run/media/raul/WD_RED_6TB/EDGES_vol2/others/beam_simulations/wipl-d/20191030"
        "/blade_dipole_infinite_soil_metal_GP_190-200MHz_single_precision.ra1"
    )
    filename206 = (
        "/run/media/raul/WD_RED_6TB/EDGES_vol2/others/beam_simulations/wipl-d/20191030"
        "/blade_dipole_infinite_soil_metal_GP_190-200MHz_double_precision.ra1"
    )

    filename300 = (
        "/run/media/raul/WD_RED_6TB/EDGES_vol2/others/beam_simulations/wipl-d/20191030"
        "/blade_dipole_infinite_PEC.ra1"
    )
    filename301 = (
        "/run/media/raul/WD_RED_6TB/EDGES_vol2/others/beam_simulations/wipl-d/20191030"
        "/blade_dipole_infinite_soil_metal_GP_10mx10m.ra1"
    )
    filename302 = (
        "/run/media/raul/WD_RED_6TB/EDGES_vol2/others/beam_simulations/wipl-d/20191030"
        "/blade_dipole_infinite_soil_metal_GP_15mx15m.ra1"
    )
    filename303 = (
        "/run/media/raul/WD_RED_6TB/EDGES_vol2/others/beam_simulations/wipl-d/20191030"
        "/blade_dipole_infinite_soil_metal_GP_20mx20m.ra1"
    )
    filename304 = (
        "/run/media/raul/WD_RED_6TB/EDGES_vol2/others/beam_simulations/wipl-d/20191030"
        "/blade_dipole_infinite_soil_metal_GP_30mx30m.ra1"
    )
    filename305 = (
        "/run/media/raul/WD_RED_6TB/EDGES_vol2/others/beam_simulations/wipl-d/20191030"
        "/blade_dipole_infinite_soil_real_metal_GP_15mx15m.ra1"
    )
    filename306 = (
        "/run/media/raul/WD_RED_6TB/EDGES_vol2/others/beam_simulations/wipl-d/20191030"
        "/blade_dipole_infinite_soil_real_metal_GP_30mx30m.ra1"
    )

    filename400 = (
        "/run/media/raul/WD_RED_6TB/EDGES_vol2/others/beam_simulations/wipl-d/20191101"
        "/blade_dipole_infinite_PEC_50-120MHz.ra1"
    )
    filename401 = (
        "/run/media/raul/WD_RED_6TB/EDGES_vol2/others/beam_simulations/wipl-d/20191101"
        "/blade_dipole_infinite_soil_metal_GP_10mx10m_50-120MHz.ra1"
    )
    filename402 = (
        "/run/media/raul/WD_RED_6TB/EDGES_vol2/others/beam_simulations/wipl-d/20191101"
        "/blade_dipole_infinite_soil_metal_GP_15mx15m_50-120MHz.ra1"
    )
    filename403 = (
        "/run/media/raul/WD_RED_6TB/EDGES_vol2/others/beam_simulations/wipl-d/20191101"
        "/blade_dipole_infinite_soil_metal_GP_20mx20m_50-120MHz.ra1"
    )
    filename404 = (
        "/run/media/raul/WD_RED_6TB/EDGES_vol2/others/beam_simulations/wipl-d/20191101"
        "/blade_dipole_infinite_soil_metal_GP_30mx30m_50-120MHz.ra1"
    )
    filename405 = (
        "/run/media/raul/WD_RED_6TB/EDGES_vol2/others/beam_simulations/wipl-d/20191101"
        "/blade_dipole_infinite_soil_real_metal_GP_15mx15m_50-120MHz.ra1"
    )
    filename406 = (
        "/run/media/raul/WD_RED_6TB/EDGES_vol2/others/beam_simulations/wipl-d/20191101"
        "/blade_dipole_infinite_soil_real_metal_GP_30mx30m_50-120MHz.ra1"
    )

    filename500 = (
        "/run/media/raul/WD_RED_6TB/EDGES_vol2/others/beam_simulations/wipl-d/20191101"
        "/EDGES_low_band_30mx30m.ra1"
    )
    filename501 = (
        "/run/media/raul/WD_RED_6TB/EDGES_vol2/others/beam_simulations/wipl-d/20191101"
        "/EDGES_low_band_10mx10m.ra1"
    )

    filename601 = (
        "/run/media/raul/WD_RED_6TB/EDGES_vol2/others/beam_simulations/wipl-d/20191112"
        "/blade_dipole_infinite_soil_metal_GP_auto_mesh_MLFMM.ra1"
    )
    # filename602 = '/run/media/raul/WD_RED_6TB/EDGES_vol2/others/beam_simulations/wipl-d
    # /20191112/blade_dipole_infinite_soil_metal_GP_auto_mesh_MLFMM_200MHz.ra1'
    filename602 = (
        "/run/media/raul/WD_RED_6TB/EDGES_vol2/others/beam_simulations/wipl-d/20191112"
        "/blade_dipole_infinite_soil_metal_GP_0.1m.ra1"
    )

    filename701 = (
        "/run/media/raul/WD_RED_6TB/EDGES_vol2/others/beam_simulations/wipl-d/20191114"
        "/blade_dipole_infinite_soil_metal_GP_1cm.ra1"
    )
    filename702 = (
        "/run/media/raul/WD_RED_6TB/EDGES_vol2/others/beam_simulations/wipl-d/20191114"
        "/blade_dipole_infinite_soil_metal_GP_1cm_double.ra1"
    )
    filename703 = (
        "/run/media/raul/WD_RED_6TB/EDGES_vol2/others/beam_simulations/wipl-d/20191114"
        "/blade_dipole_infinite_soil_metal_GP_integral_accuracy_enhanced3_matrix_precision_double.ra1"
    )

    filename801 = (
        "/run/media/raul/WD_RED_6TB/EDGES_vol2/others/beam_simulations/wipl-d/20191122"
        "/mid_band_perf_30x30.ra1"
    )

    if case == 0:
        filename = filename0

    if case == 1:
        filename = filename1

    if case == 2:
        filename = filename2

    if case == 3:
        filename = filename3

    if case == 4:
        filename = filename4

    if case == 5:
        filename = filename5

    if case == 11:
        filename = filename11

    if case == 12:
        filename = filename12

    if case == 13:
        filename = filename13

    if case == 14:
        filename = filename14

    if case == 15:
        filename = filename15

    if case == 16:
        filename = filename16

    if case == 51:
        filename = filename51

    if case == 52:
        filename = filename52

    if case == 53:
        filename = filename53

    if case == 54:
        filename = filename54

    if case == 55:
        filename = filename55

    if case == 56:
        filename = filename56

    if case == 101:
        filename = filename101

    if case == 102:
        filename = filename102

    if case == 103:
        filename = filename103

    if case == 104:
        filename = filename104

    if case == 105:
        filename = filename105

    if case == 106:
        filename = filename106

    if case == 200:
        filename = filename200

    if case == 201:
        filename = filename201

    if case == 202:
        filename = filename202

    if case == 203:
        filename = filename203

    if case == 204:
        filename = filename204

    if case == 205:
        filename = filename205

    if case == 206:
        filename = filename206

    if case == 300:
        filename = filename300

    if case == 301:
        filename = filename301

    if case == 302:
        filename = filename302

    if case == 303:
        filename = filename303

    if case == 304:
        filename = filename304

    if case == 305:
        filename = filename305

    if case == 306:
        filename = filename306

    if case == 400:
        filename = filename400

    if case == 401:
        filename = filename401

    if case == 402:
        filename = filename402

    if case == 403:
        filename = filename403

    if case == 404:
        filename = filename404

    if case == 405:
        filename = filename405

    if case == 406:
        filename = filename406

    if case == 500:
        filename = filename500

    if case == 501:
        filename = filename501

    if case == 601:
        filename = filename601

    if case == 602:
        filename = filename602

    if case == 701:
        filename = filename701

    if case == 702:
        filename = filename702

    if case == 703:
        filename = filename703

    if case == 801:
        filename = filename801

    f, thetaX, phiX, beam = beams.wipld_read(filename)

    if case == 0:
        theta = np.copy(thetaX)
        m = np.copy(beam)
    else:
        theta = thetaX[thetaX < 90]
        m = beam[:, thetaX < 90, :]

    sin_theta = np.sin(theta * (np.pi / 180))
    sin_theta_2D_T = np.tile(sin_theta, (360, 1))
    sin_theta_2D = sin_theta_2D_T.T

    bint = np.zeros(len(f))
    for i in range(len(f)):
        bt = m[i, :, :]
        bint[i] = np.sum(bt * sin_theta_2D)
        b = (1 / (4 * np.pi)) * ((np.pi / 180) ** 2) * bint  # /np.mean(bx)

    # n_fg   = 5
    f_low = 50
    f_high = 200
    fX = f[(f >= f_low) & (f <= f_high)]
    bX = b[(f >= f_low) & (f <= f_high)]

    x = np.polyfit(fX, bX, n_fg - 1)
    model = np.polyval(x, fX)
    rX = bX - model

    return f, b, fX, rX, thetaX, phiX, beam


def plots_for_memo_153(figname):
    # TODO: move to old
    path_plot_save = config["edges_folder"] + "plots/20191025/"
    sx = 8
    sy = 10

    if figname == 1:
        n_fg = 5
        f, b0, f0, r0 = integrated_antenna_gain_WIPLD(0, n_fg)

        plt.close()
        plt.close()
        plt.close()
        plt.figure(figsize=[sx, sy])

        plt.subplot(2, 1, 1)
        plt.plot(f, b0, "k")
        plt.ylim([0.999, 1.001])
        plt.ylabel("integrated gain over the full sphere\n [fraction of 4pi]")

        plt.subplot(2, 1, 2)
        plt.plot(f0, r0, "k")
        plt.ylim([-0.000005, 0.000005])
        plt.ylabel("residuals")
        plt.xlabel("frequency [MHz]")

        plt.savefig(path_plot_save + "fig1.pdf", bbox_inches="tight")
        plt.close()
        plt.close()

    if figname == 2:
        n_fg = 5
        f, b1, f1, r1 = integrated_antenna_gain_WIPLD(1, n_fg)
        f, b2, f2, r2 = integrated_antenna_gain_WIPLD(2, n_fg)
        f, b3, f3, r3 = integrated_antenna_gain_WIPLD(3, n_fg)
        f, b4, f4, r4 = integrated_antenna_gain_WIPLD(4, n_fg)
        f, b5, f5, r5 = integrated_antenna_gain_WIPLD(5, n_fg)

        plt.close()
        plt.close()
        plt.close()
        plt.figure(figsize=[sx, sy])

        plt.subplot(2, 1, 1)
        plt.plot(f, b1, "k")
        plt.plot(f, b2, "b--")
        plt.plot(f, b3, "b:")
        plt.plot(f, b4, "r--")
        plt.plot(f, b5, "r:")
        plt.ylabel("integrated gian above horizon\n [fraction of 4pi]")
        plt.legend(
            [
                r"$\epsilon_r$=3.5, $\sigma$=0.02",
                r"$\epsilon_r$=2.5, $\sigma$=0.02",
                r"$\epsilon_r$=4.5, $\sigma$=0.02",
                r"$\epsilon_r$=3.5, $\sigma$=0.002",
                r"$\epsilon_r$=3.5, $\sigma$=0.2",
            ],
            ncol=2,
            loc=0,
        )

        plt.subplot(2, 1, 2)
        plt.plot(f1, r1, "k")
        plt.plot(f2, r2, "b--")
        plt.plot(f3, r3, "b:")
        plt.plot(f4, r4, "r--")
        plt.plot(f5, r5, "r:")
        plt.ylim([-0.00002, 0.00002])
        plt.ylabel("residuals")
        plt.xlabel("frequency [MHz]")

        plt.savefig(path_plot_save + "fig2.pdf", bbox_inches="tight")
        plt.close()
        plt.close()

    if figname == 3:
        n_fg = 7
        f, b1, f1, r1 = integrated_antenna_gain_WIPLD(11, n_fg)
        f, b2, f2, r2 = integrated_antenna_gain_WIPLD(12, n_fg)
        f, b3, f3, r3 = integrated_antenna_gain_WIPLD(13, n_fg)
        f, b4, f4, r4 = integrated_antenna_gain_WIPLD(14, n_fg)
        f, b5, f5, r5 = integrated_antenna_gain_WIPLD(15, n_fg)
        f, b6, f6, r6 = integrated_antenna_gain_WIPLD(16, n_fg)

        plt.figure(figsize=[sx, sy])

        plt.subplot(2, 1, 1)
        plt.plot(f, b1, "k")
        plt.plot(f, b2, "k:")
        plt.plot(f, b3, "b--")
        plt.plot(f, b4, "b:")
        plt.plot(f, b5, "r--")
        plt.plot(f, b6, "r:")
        plt.ylabel("integrated gian above horizon\n [fraction of 4pi]")
        plt.legend(
            [
                r"$\epsilon_r$=3.5, $\sigma$=0.02",
                r"$\epsilon_r$=2.5, $\sigma$=0.02",
                r"$\epsilon_r$=4.5, $\sigma$=0.02",
                r"$\epsilon_r$=3.5, $\sigma$=0.002",
                r"$\epsilon_r$=3.5, $\sigma$=0.2",
            ],
            ncol=2,
            loc=0,
        )

        plt.subplot(2, 1, 2)
        plt.plot(f1, r1, "k")
        plt.plot(f2, r2, "k:")
        plt.plot(f3, r3, "b--")
        plt.plot(f4, r4, "b:")
        plt.plot(f5, r5, "r--")
        plt.plot(f6, r6, "r:")
        plt.ylim([-0.0006, 0.0006])
        plt.ylabel("residuals")
        plt.xlabel("frequency [MHz]")

        plt.savefig(path_plot_save + "fig3.pdf", bbox_inches="tight")
    if figname == 4:
        n_fg = 7
        f, b1, f1, r1 = integrated_antenna_gain_WIPLD(51, n_fg)
        f, b2, f2, r2 = integrated_antenna_gain_WIPLD(52, n_fg)
        f, b3, f3, r3 = integrated_antenna_gain_WIPLD(53, n_fg)
        f, b4, f4, r4 = integrated_antenna_gain_WIPLD(54, n_fg)
        f, b5, f5, r5 = integrated_antenna_gain_WIPLD(55, n_fg)
        f, b6, f6, r6 = integrated_antenna_gain_WIPLD(56, n_fg)

        plt.figure(figsize=[sx, sy])

        plt.subplot(2, 1, 1)
        plt.plot(f, b1, "k")
        plt.plot(f, b2, "k:")
        plt.plot(f, b3, "b--")
        plt.plot(f, b4, "b:")
        plt.plot(f, b5, "r--")
        plt.plot(f, b6, "r:")
        plt.ylabel("integrated gian above horizon\n [fraction of 4pi]")
        plt.legend(
            [
                r"$\epsilon_r$=3.5, $\sigma$=0.02",
                r"$\epsilon_r$=2.5, $\sigma$=0.02",
                r"$\epsilon_r$=4.5, $\sigma$=0.02",
                r"$\epsilon_r$=3.5, $\sigma$=0.002",
                r"$\epsilon_r$=3.5, $\sigma$=0.2",
            ],
            ncol=2,
            loc=0,
        )

        plt.subplot(2, 1, 2)
        plt.plot(f1, r1, "k")
        plt.plot(f2, r2, "k:")
        plt.plot(f3, r3, "b--")
        plt.plot(f4, r4, "b:")
        plt.plot(f5, r5, "r--")
        plt.plot(f6, r6, "r:")
        plt.ylim([-0.0004, 0.0004])
        plt.ylabel("residuals")
        plt.xlabel("frequency [MHz]")

        plt.savefig(path_plot_save + "fig4.pdf", bbox_inches="tight")

    if figname == 5:
        n_fg = 7
        f, b1, f1, r1 = integrated_antenna_gain_WIPLD(101, n_fg)
        f, b2, f2, r2 = integrated_antenna_gain_WIPLD(102, n_fg)
        f, b3, f3, r3 = integrated_antenna_gain_WIPLD(103, n_fg)
        f, b4, f4, r4 = integrated_antenna_gain_WIPLD(104, n_fg)
        f, b5, f5, r5 = integrated_antenna_gain_WIPLD(105, n_fg)
        f, b6, f6, r6 = integrated_antenna_gain_WIPLD(106, n_fg)

        plt.figure(figsize=[sx, sy])

        plt.subplot(2, 1, 1)
        plt.plot(f, b1, "k")
        plt.plot(f, b2, "k:")
        plt.plot(f, b3, "b--")
        plt.plot(f, b4, "b:")
        plt.plot(f, b5, "r--")
        plt.plot(f, b6, "r:")
        plt.ylabel("integrated gian above horizon\n [fraction of 4pi]")
        plt.legend(
            [
                r"$\epsilon_r$=3.5, $\sigma$=0.02",
                r"$\epsilon_r$=2.5, $\sigma$=0.02",
                r"$\epsilon_r$=4.5, $\sigma$=0.02",
                r"$\epsilon_r$=3.5, $\sigma$=0.002",
                r"$\epsilon_r$=3.5, $\sigma$=0.2",
            ],
            ncol=2,
            loc=0,
        )

        plt.subplot(2, 1, 2)
        plt.plot(f1, r1, "k")
        plt.plot(f2, r2, "k:")
        plt.plot(f3, r3, "b--")
        plt.plot(f4, r4, "b:")
        plt.plot(f5, r5, "r--")
        plt.plot(f6, r6, "r:")
        plt.ylim([-0.0002, 0.0002])
        plt.ylabel("residuals")
        plt.xlabel("frequency [MHz]")

        plt.savefig(path_plot_save + "fig5.pdf", bbox_inches="tight")


def plots_for_memo_155():
    # TODO: move to old
    path_plot_save = config["edges_folder"] + "plots/20191105/"

    fA, b300, fx, rx, theta, phi, beam300 = integrated_antenna_gain_WIPLD(300, 2)
    fA, b301, fx, rx, theta, phi, beam301 = integrated_antenna_gain_WIPLD(301, 2)
    fA, b302, fx, rx, theta, phi, beam302 = integrated_antenna_gain_WIPLD(302, 2)
    fA, b303, fx, rx, theta, phi, beam303 = integrated_antenna_gain_WIPLD(303, 2)
    fA, b304, fx, rx, theta, phi, beam304 = integrated_antenna_gain_WIPLD(304, 2)
    fA, b305, fx, rx, theta, phi, beam305 = integrated_antenna_gain_WIPLD(305, 2)
    fA, b306, fx, rx, theta, phi, beam306 = integrated_antenna_gain_WIPLD(306, 2)

    fB, b400, fx, rx, theta, phi, beam400 = integrated_antenna_gain_WIPLD(400, 2)
    fB, b401, fx, rx, theta, phi, beam401 = integrated_antenna_gain_WIPLD(401, 2)
    fB, b402, fx, rx, theta, phi, beam402 = integrated_antenna_gain_WIPLD(402, 2)
    fB, b403, fx, rx, theta, phi, beam403 = integrated_antenna_gain_WIPLD(403, 2)
    fB, b404, fx, rx, theta, phi, beam404 = integrated_antenna_gain_WIPLD(404, 2)
    fB, b405, fx, rx, theta, phi, beam405 = integrated_antenna_gain_WIPLD(405, 2)
    fB, b406, fx, rx, theta, phi, beam406 = integrated_antenna_gain_WIPLD(406, 2)

    fC, b500, fx, rx, theta, phi, beam500 = integrated_antenna_gain_WIPLD(500, 2)
    fC, b501, fx, rx, theta, phi, beam501 = integrated_antenna_gain_WIPLD(501, 2)

    # --------------------------------------------
    # FEKO
    # --------------------------------------------

    fmin = 1
    fmax = 300

    el = np.arange(0, 91)
    sin_theta = np.sin((90 - el) * (np.pi / 180))
    sin_theta_2D_T = np.tile(sin_theta, (360, 1))
    sin_theta_2D = sin_theta_2D_T.T

    # Low-Band, Blade on 10m x 10m GP
    b_all = beams.FEKO_low_band_blade_beam(
        beam_file=5,
        frequency_interpolation=False,
        frequency=np.array([0]),
        AZ_antenna_axis=0,
    )
    f = np.arange(50, 121, 2)
    # f      = np.arange(40,101,2.5)

    bint = np.zeros(len(f))
    for i in range(len(f)):
        b = b_all[i, :, :]
        bint[i] = np.sum(b * sin_theta_2D)

    ft = f[(f >= fmin) & (f <= fmax)]
    bx = bint[(f >= fmin) & (f <= fmax)]
    bt = (1 / (4 * np.pi)) * ((np.pi / 180) ** 2) * bx  # /np.mean(bx)

    f_low10 = np.copy(ft)
    blow10 = np.copy(bt)
    beam_low10 = np.copy(b_all)

    # Low-Band, Blade on 30m x 30m GP
    b_all = beams.FEKO_low_band_blade_beam(
        beam_file=2, frequency_interpolation=False, AZ_antenna_axis=0
    )
    f = np.arange(40, 121, 2)

    bint = np.zeros(len(f))
    for i in range(len(f)):
        b = b_all[i, :, :]
        bint[i] = np.sum(b * sin_theta_2D)

    ft = f[(f >= fmin) & (f <= fmax)]
    bx = bint[(f >= fmin) & (f <= fmax)]
    bt = (1 / (4 * np.pi)) * ((np.pi / 180) ** 2) * bx  # /np.mean(bx)

    f_low30 = np.copy(ft)
    blow30 = np.copy(bt)
    beam_low30 = np.copy(b_all)

    # Mid-Band, Blade, on 30m x 30m GP
    b_all = beams.feko_blade_beam(
        "mid_band", 0, frequency_interpolation=False, az_antenna_axis=90
    )
    f = np.arange(50, 201, 2)

    bint = np.zeros(len(f))
    for i in range(len(f)):
        b = b_all[i, :, :]
        bint[i] = np.sum(b * sin_theta_2D)

    ft = f[(f >= fmin) & (f <= fmax)]
    bx = bint[(f >= fmin) & (f <= fmax)]
    bt = (1 / (4 * np.pi)) * ((np.pi / 180) ** 2) * bx  # /np.mean(bx)

    fmid30 = np.copy(ft)
    bmid30 = np.copy(bt)
    beam_mid30 = np.copy(b_all)

    # Figure 1
    # --------------------------------------------

    plt.figure(1, figsize=[10, 12])
    plt.plot(fA, b301, "r")
    plt.plot(fA, b305, "y")
    plt.plot(fA, b302, "m")
    plt.plot(fA, b303, "c")
    plt.plot(fA, b306, "b")
    plt.plot(fA, b304, "g")
    plt.plot(fA, b300, "k")

    plt.plot(fB, b401, "r--")
    plt.plot(fB, b405, "y--")
    plt.plot(fB, b402, "m--")
    plt.plot(fB, b403, "c--")
    plt.plot(fB, b406, "b--")
    plt.plot(fB, b404, "g--")
    plt.plot(fB, b400, "k--")

    plt.ylim([0.988, 1.002])

    plt.xlabel("frequency [MHz]")
    plt.ylabel("integrated gain above horizon\n [fraction of 4pi]")

    plt.legend(
        [
            "10mx10m",
            "perf 15mx15m",
            "15mx15m",
            "20mx20m",
            "perf 30mx30m",
            "30mx30m",
            "Inf PEC",
        ],
        loc=0,
    )

    plt.savefig(path_plot_save + "fig1.pdf", bbox_inches="tight")
    plt.close()
    plt.close()

    # Figure 2
    # --------------------------------------------

    plt.figure(2, figsize=[10, 8])
    plt.plot(fA, b301, "r")
    plt.plot(fC, b501, "r--")

    plt.plot(fA, b306, "b")
    plt.plot(fC, b500, "b--")

    plt.xlabel("frequency [MHz]")
    plt.ylabel("integrated gain above horizon\n [fraction of 4pi]")

    plt.legend(
        [
            "10mx10m, mid-band",
            "10mx10m, low-band",
            "perf 30mx30m, mid-band",
            "perf 30mx30m, low-band",
        ],
        loc=0,
    )

    plt.savefig(path_plot_save + "fig2.pdf", bbox_inches="tight")
    plt.close()
    plt.close()

    # Figure 3
    # --------------------------------------------

    plt.figure(3, figsize=[10, 8])
    plt.plot(fA, 100 * (1 - b301), "r")
    plt.plot(fC, 100 * (1 - b501), "r--")

    plt.plot(fA, 100 * (1 - b306), "b")
    plt.plot(fC, 100 * (1 - b500), "b--")

    plt.xlabel("frequency [MHz]")
    plt.ylabel("loss [%]")

    plt.legend(
        [
            "10mx10m, mid-band",
            "10mx10m, low-band",
            "perf 30mx30m, mid-band",
            "perf 30mx30m, low-band",
        ],
        loc=0,
    )

    plt.savefig(path_plot_save + "fig3.pdf", bbox_inches="tight")
    plt.close()
    plt.close()

    # Figure 4
    # --------------------------------------------

    plt.figure(4, figsize=[10, 13])
    plt.plot(fA, b301, "r")
    plt.plot(fC, b501, "r--")
    plt.plot(f_low10, blow10, "m--")

    plt.plot(fA, b306, "b")
    plt.plot(fmid30, bmid30, "c")
    plt.plot(fC, b500, "b--")
    plt.plot(f_low30, blow30, "c--")

    plt.xlabel("frequency [MHz]")
    plt.ylabel("integrated gain above horizon\n [fraction of 4pi]")

    plt.legend(
        [
            "10mx10m, mid-band",
            "10mx10m, low-band",
            "10mx10m, low-band FEKO",
            "perf 30mx30m, mid-band",
            "perf 30mx30m, mid-band FEKO",
            "perf 30mx30m, low-band",
            "perf 30mx30m, low-band FEKO",
        ],
        loc=0,
    )

    plt.savefig(path_plot_save + "fig4.pdf", bbox_inches="tight")
    plt.close()
    plt.close()

    # Figure 5
    # --------------------------------------------

    plt.figure(5, figsize=[10, 8])
    plt.plot(fA, beam301[:, 0, 0], "r")
    plt.plot(fC, beam501[:, 0, 0], "r--")
    plt.plot(fA, beam306[:, 0, 0], "b")
    plt.plot(fC, beam500[:, 0, 0], "b--")

    plt.plot(fA, beam301[:, 45, 0], "r")
    plt.plot(fC, beam501[:, 45, 0], "r--")
    plt.plot(fA, beam306[:, 45, 0], "b")
    plt.plot(fC, beam500[:, 45, 0], "b--")

    plt.plot(fA, beam301[:, 45, 90], "r")
    plt.plot(fC, beam501[:, 45, 90], "r--")
    plt.plot(fA, beam306[:, 45, 90], "b")
    plt.plot(fC, beam500[:, 45, 90], "b--")

    plt.xlim([50, 140])
    plt.ylim([0, 8])

    plt.xlabel("frequency [MHz]")
    plt.ylabel("gain")

    plt.legend(
        [
            "10mx10m, mid-band",
            "10mx10m, low-band",
            "perf 30mx30m, mid-band",
            "perf 30mx30m, low-band",
        ],
        loc=0,
    )

    plt.text(60, 7.3, "Zenith", fontsize=16)
    plt.text(60, 4.3, r"$\theta=45^{\circ}$, $\phi=90^{\circ}$", fontsize=16)
    plt.text(60, 2.1, r"$\theta=45^{\circ}$, $\phi=0^{\circ}$", fontsize=16)

    plt.savefig(path_plot_save + "fig5.pdf", bbox_inches="tight")
    plt.close()
    plt.close()

    # Figure 6
    # --------------------------------------------

    plt.figure(6, figsize=[10, 8])
    plt.plot(f_low10, beam_low10[:, -1, 0], "r--")
    plt.plot(fmid30, beam_mid30[:, -1, 0], "b")
    plt.plot(f_low30, beam_low30[:, -1, 0], "b--")

    plt.plot(f_low10, beam_low10[:, 45, 0], "r--")
    plt.plot(fmid30, beam_mid30[:, 45, 0], "b")
    plt.plot(f_low30, beam_low30[:, 45, 0], "b--")

    plt.plot(f_low10, beam_low10[:, 45, 90], "r--")
    plt.plot(fmid30, beam_mid30[:, 45, 90], "b")
    plt.plot(f_low30, beam_low30[:, 45, 90], "b--")

    plt.xlim([50, 140])
    plt.ylim([0, 8])

    plt.xlabel("frequency [MHz]")
    plt.ylabel("gain")

    plt.legend(
        ["10mx10m, low-band", "perf 30mx30m, mid-band", "perf 30mx30m, low-band"], loc=0
    )

    plt.text(60, 7.3, "Zenith", fontsize=16)
    plt.text(60, 4.3, r"$\theta=45^{\circ}$, $\phi=90^{\circ}$", fontsize=16)
    plt.text(60, 2.1, r"$\theta=45^{\circ}$, $\phi=0^{\circ}$", fontsize=16)

    plt.savefig(path_plot_save + "fig6.pdf", bbox_inches="tight")
    plt.close()
    plt.close()

    # Figure 7
    # --------------------------------------------
    # Mid-Band, 10mx10m

    delta = np.zeros(
        (len(beam301[:, 0, 0]) - 1, len(beam301[0, :, 0]), len(beam301[0, 0, :]))
    )
    for i in range(len(fA) - 1):
        XX = beam301[i + 1, :, :] - beam301[i, :, :]
        XXX = np.flipud(XX)
        delta[i, :, :] = XXX

    plt.figure(7, figsize=(7, 7))

    plt.subplot(2, 1, 1)
    plt.imshow(
        np.flipud(delta[:, :, 0].T),
        interpolation="none",
        aspect="auto",
        extent=[50, 200, 90, 0],
        vmin=-0.05,
        vmax=0.05,
    )
    cbar = plt.colorbar()
    plt.yticks([0, 30, 60, 90])
    plt.xticks([50, 75, 100, 125, 150, 175, 200], labels=[])
    cbar.set_label(r"$\Delta$ directive gain per MHz", rotation=90)
    plt.ylabel("theta [deg]")
    plt.title("phi=0 [deg]")

    plt.subplot(2, 1, 2)
    plt.imshow(
        np.flipud(delta[:, :, 90].T),
        interpolation="none",
        aspect="auto",
        extent=[50, 200, 90, 0],
        vmin=-0.05,
        vmax=0.05,
    )
    cbar = plt.colorbar()
    plt.yticks([0, 30, 60, 90])
    plt.xticks([50, 75, 100, 125, 150, 175, 200])
    cbar.set_label(r"$\Delta$ directive gain per MHz", rotation=90)
    plt.ylabel("theta [deg]")
    plt.xlabel("frequency [MHz]")
    plt.title("phi=90 [deg]")

    plt.savefig(path_plot_save + "fig7.pdf", bbox_inches="tight")
    plt.close()
    plt.close()

    # Figure 8
    # --------------------------------------------
    # Mid-Band, 30mx30m

    delta = np.zeros(
        (len(beam306[:, 0, 0]) - 1, len(beam306[0, :, 0]), len(beam306[0, 0, :]))
    )
    for i in range(len(fA) - 1):
        XX = beam306[i + 1, :, :] - beam306[i, :, :]
        XXX = np.flipud(XX)
        delta[i, :, :] = XXX

    plt.figure(8, figsize=(7, 7))

    plt.subplot(2, 1, 1)
    plt.imshow(
        np.flipud(delta[:, :, 0].T),
        interpolation="none",
        aspect="auto",
        extent=[50, 200, 90, 0],
        vmin=-0.05,
        vmax=0.05,
    )
    cbar = plt.colorbar()
    plt.yticks([0, 30, 60, 90])
    plt.xticks([50, 75, 100, 125, 150, 175, 200], labels=[])
    cbar.set_label(r"$\Delta$ directive gain per MHz", rotation=90)
    plt.ylabel("theta [deg]")
    plt.title("phi=0 [deg]")

    plt.subplot(2, 1, 2)
    plt.imshow(
        np.flipud(delta[:, :, 90].T),
        interpolation="none",
        aspect="auto",
        extent=[50, 200, 90, 0],
        vmin=-0.05,
        vmax=0.05,
    )
    cbar = plt.colorbar()
    plt.yticks([0, 30, 60, 90])
    plt.xticks([50, 75, 100, 125, 150, 175, 200])
    cbar.set_label(r"$\Delta$ directive gain per MHz", rotation=90)
    plt.ylabel("theta [deg]")
    plt.xlabel("frequency [MHz]")
    plt.title("phi=90 [deg]")

    plt.savefig(path_plot_save + "fig8.pdf", bbox_inches="tight")
    plt.close()
    plt.close()

    # Figure 9
    # --------------------------------------------
    # Low-Band, 10mx10m

    delta = np.zeros(
        (len(beam501[:, 0, 0]) - 1, len(beam501[0, :, 0]), len(beam501[0, 0, :]))
    )
    for i in range(len(fC) - 1):
        XX = beam501[i + 1, :, :] - beam501[i, :, :]
        XXX = np.flipud(XX)
        delta[i, :, :] = XXX

    plt.figure(9, figsize=(7, 7))

    plt.subplot(2, 1, 1)
    plt.imshow(
        np.flipud(delta[:, :, 0].T),
        interpolation="none",
        aspect="auto",
        extent=[50, 200, 90, 0],
        vmin=-0.05,
        vmax=0.05,
    )
    cbar = plt.colorbar()
    plt.yticks([0, 30, 60, 90])
    plt.xticks([50, 75, 100, 125, 150, 175, 200], labels=[])
    cbar.set_label(r"$\Delta$ directive gain per MHz", rotation=90)
    plt.ylabel("theta [deg]")
    plt.title("phi=0 [deg]")

    plt.subplot(2, 1, 2)
    plt.imshow(
        np.flipud(delta[:, :, 90].T),
        interpolation="none",
        aspect="auto",
        extent=[50, 200, 90, 0],
        vmin=-0.05,
        vmax=0.05,
    )
    cbar = plt.colorbar()
    plt.yticks([0, 30, 60, 90])
    plt.xticks([50, 75, 100, 125, 150, 175, 200])
    cbar.set_label(r"$\Delta$ directive gain per MHz", rotation=90)
    plt.ylabel("theta [deg]")
    plt.xlabel("frequency [MHz]")
    plt.title("phi=90 [deg]")

    plt.savefig(path_plot_save + "fig9.pdf", bbox_inches="tight")
    plt.close()
    plt.close()

    # Figure 10
    # --------------------------------------------
    # Low-Band, 30mx30m

    delta = np.zeros(
        (len(beam500[:, 0, 0]) - 1, len(beam500[0, :, 0]), len(beam500[0, 0, :]))
    )
    for i in range(len(fC) - 1):
        XX = beam500[i + 1, :, :] - beam500[i, :, :]
        XXX = np.flipud(XX)
        delta[i, :, :] = XXX

    plt.figure(10, figsize=(7, 7))

    plt.subplot(2, 1, 1)
    plt.imshow(
        np.flipud(delta[:, :, 0].T),
        interpolation="none",
        aspect="auto",
        extent=[50, 200, 90, 0],
        vmin=-0.05,
        vmax=0.05,
    )
    cbar = plt.colorbar()
    plt.yticks([0, 30, 60, 90])
    plt.xticks([50, 75, 100, 125, 150, 175, 200], labels=[])
    cbar.set_label(r"$\Delta$ directive gain per MHz", rotation=90)
    plt.ylabel("theta [deg]")
    plt.title("phi=0 [deg]")

    plt.subplot(2, 1, 2)
    plt.imshow(
        np.flipud(delta[:, :, 90].T),
        interpolation="none",
        aspect="auto",
        extent=[50, 200, 90, 0],
        vmin=-0.05,
        vmax=0.05,
    )
    cbar = plt.colorbar()
    plt.yticks([0, 30, 60, 90])
    plt.xticks([50, 75, 100, 125, 150, 175, 200])
    cbar.set_label(r"$\Delta$ directive gain per MHz", rotation=90)
    plt.ylabel("theta [deg]")
    plt.xlabel("frequency [MHz]")
    plt.title("phi=90 [deg]")

    plt.savefig(path_plot_save + "fig10.pdf", bbox_inches="tight")
    plt.close()
    plt.close()

    # Figure 11
    # --------------------------------------------
    # Mid-Band, 30mx30m

    delta = np.zeros(
        (
            len(beam_mid30[:, 0, 0]) - 1,
            len(beam_mid30[0, :, 0]),
            len(beam_mid30[0, 0, :]),
        )
    )
    for i in range(len(fmid30) - 1):
        XX = beam_mid30[i + 1, :, :] - beam_mid30[i, :, :]
        XXX = np.flipud(XX)
        delta[i, :, :] = XXX

    plt.figure(11, figsize=(7, 7))

    plt.subplot(2, 1, 1)
    plt.imshow(
        delta[:, :, 0].T,
        interpolation="none",
        aspect="auto",
        extent=[50, 200, 90, 0],
        vmin=-0.05,
        vmax=0.05,
    )
    cbar = plt.colorbar()
    plt.yticks([0, 30, 60, 90])
    plt.xticks([50, 75, 100, 125, 150, 175, 200], labels=[])
    cbar.set_label(r"$\Delta$ directive gain per MHz", rotation=90)
    plt.ylabel("theta [deg]")
    plt.title("phi=0 [deg]")

    plt.subplot(2, 1, 2)
    plt.imshow(
        delta[:, :, 90].T,
        interpolation="none",
        aspect="auto",
        extent=[50, 200, 90, 0],
        vmin=-0.05,
        vmax=0.05,
    )
    cbar = plt.colorbar()
    plt.yticks([0, 30, 60, 90])
    plt.xticks([50, 75, 100, 125, 150, 175, 200])
    cbar.set_label(r"$\Delta$ directive gain per MHz", rotation=90)
    plt.ylabel("theta [deg]")
    plt.xlabel("frequency [MHz]")
    plt.title("phi=90 [deg]")

    plt.savefig(path_plot_save + "fig11.pdf", bbox_inches="tight")
    plt.close()
    plt.close()

    # Figure 12
    # --------------------------------------------
    # Low-Band, 30mx30m

    delta = np.zeros(
        (
            len(beam_low30[:, 0, 0]) - 1,
            len(beam_low30[0, :, 0]),
            len(beam_low30[0, 0, :]),
        )
    )
    for i in range(len(f_low30) - 1):
        XX = beam_low30[i + 1, :, :] - beam_low30[i, :, :]
        XXX = np.flipud(XX)
        delta[i, :, :] = XXX

    plt.figure(12, figsize=(7, 7))

    plt.subplot(2, 1, 1)
    plt.imshow(
        delta[:, :, 0].T,
        interpolation="none",
        aspect="auto",
        extent=[40, 120, 90, 0],
        vmin=-0.05,
        vmax=0.05,
    )
    cbar = plt.colorbar()
    plt.yticks([0, 30, 60, 90])
    plt.xticks([40, 60, 80, 100, 120], labels=[])
    cbar.set_label(r"$\Delta$ directive gain per MHz", rotation=90)
    plt.ylabel("theta [deg]")
    plt.title("phi=0 [deg]")

    plt.subplot(2, 1, 2)
    plt.imshow(
        delta[:, :, 90].T,
        interpolation="none",
        aspect="auto",
        extent=[40, 120, 90, 0],
        vmin=-0.05,
        vmax=0.05,
    )
    cbar = plt.colorbar()
    plt.yticks([0, 30, 60, 90])
    plt.xticks([40, 60, 80, 100, 120])
    cbar.set_label(r"$\Delta$ directive gain per MHz", rotation=90)
    plt.ylabel("theta [deg]")
    plt.xlabel("frequency [MHz]")
    plt.title("phi=90 [deg]")

    plt.savefig(path_plot_save + "fig12.pdf", bbox_inches="tight")
    plt.close()
    plt.close()

    # Figure 13
    # --------------------------------------------
    # Low-Band, 10mx10m

    delta = np.zeros(
        (
            len(beam_low10[:, 0, 0]) - 1,
            len(beam_low10[0, :, 0]),
            len(beam_low10[0, 0, :]),
        )
    )
    for i in range(len(f_low10) - 1):
        XX = beam_low10[i + 1, :, :] - beam_low10[i, :, :]
        XXX = np.flipud(XX)
        delta[i, :, :] = XXX

    plt.figure(13, figsize=(7, 7))

    plt.subplot(2, 1, 1)
    plt.imshow(
        delta[:, :, 0].T,
        interpolation="none",
        aspect="auto",
        extent=[50, 120, 90, 0],
        vmin=-0.05,
        vmax=0.05,
    )
    cbar = plt.colorbar()
    plt.yticks([0, 30, 60, 90])
    plt.xticks([50, 60, 70, 80, 90, 100, 110, 120], labels=[])
    cbar.set_label(r"$\Delta$ directive gain per MHz", rotation=90)
    plt.ylabel("theta [deg]")
    plt.title("phi=0 [deg]")

    plt.subplot(2, 1, 2)
    plt.imshow(
        delta[:, :, 90].T,
        interpolation="none",
        aspect="auto",
        extent=[50, 120, 90, 0],
        vmin=-0.05,
        vmax=0.05,
    )
    cbar = plt.colorbar()
    plt.yticks([0, 30, 60, 90])
    plt.xticks([50, 60, 70, 80, 90, 100, 110, 120])
    cbar.set_label(r"$\Delta$ directive gain per MHz", rotation=90)
    plt.ylabel("theta [deg]")
    plt.xlabel("frequency [MHz]")
    plt.title("phi=90 [deg]")

    plt.savefig(path_plot_save + "fig13.pdf", bbox_inches="tight")
    plt.close()
    plt.close()

    delta = np.zeros(
        (
            len(beam_low10[:, 0, 0]) - 1,
            len(beam_low10[0, :, 0]),
            len(beam_low10[0, 0, :]),
        )
    )
    for i in range(len(f) - 1):
        XX = beam_low10[i + 1, :, :] - beam_low10[i, :, :]
        XXX = np.flipud(XX)
        delta[i, :, :] = XXX

    return fA, beam301, delta, f_low10, beam_low10


def beam_correction_check(f_low, f_high):
    # TODO: move to old.

    bb = np.genfromtxt(
        config["edges_folder"] + "mid_band/calibration/beam_factors/raw/mid_band_50"
        "-200MHz_90deg_alan1_haslam_2.5_2.62_reffreq_100MHz_data.txt"
    )
    ff = np.genfromtxt(
        config["edges_folder"] + "mid_band/calibration/beam_factors/raw/mid_band_50"
        "-200MHz_90deg_alan1_haslam_2.5_2.62_reffreq_100MHz_freq.txt"
    )

    b = bb[:, (ff >= f_low) & (ff <= f_high)]
    f = ff[(ff >= f_low) & (ff <= f_high)]

    plt.figure(1)
    plt.imshow(b, interpolation="none", aspect="auto", vmin=0.99, vmax=1.01)
    plt.colorbar()

    f_t, lst_t, bf_t = beams.beam_factor_table_read(
        config["edges_folder"]
        + "mid_band/calibration/beam_factors/table/table_hires_mid_band_50"
        "-200MHz_90deg_alan1_haslam_2.5_2.62_reffreq_100MHz.hdf5"
    )

    plt.figure(2)
    plt.imshow(bf_t, interpolation="none", aspect="auto", vmin=0.99, vmax=1.01)
    plt.colorbar()

    lin = np.arange(0, 24, 1)
    bf = beams.beam_factor_table_evaluate(f_t, lst_t, bf_t, lin)

    plt.figure(3)
    plt.imshow(bf, interpolation="none", aspect="auto", vmin=0.99, vmax=1.01)
    plt.colorbar()

    return f, f_t


def plots_for_midband_verification_paper(
    antenna_reflection_loss=False, beam_factor=False
):
    # TODO: move to old.

    if antenna_reflection_loss:
        # Plot
        # ---------------------------------------
        f1 = plt.figure(num=1, figsize=(5.5, 6))

        # Generating reflection coefficient and loss
        f = np.arange(60, 160.1, 0.1)

        s11_ant = models_antenna_s11_remove_delay(
            "mid_band",
            f,
            year=2018,
            day=147,
            delay_0=0.17,
            model_type="polynomial",
            Nfit=14,
            plot_fit_residuals=False,
        )

        Gb, Gc = loss.balun_and_connector_loss(f, s11_ant)

        # Nominal S11
        # -----------
        ax = f1.add_axes([0.11, 0.1 + 0.4, 0.73, 0.4])
        h1 = ax.plot(
            f, 20 * np.log10(np.abs(s11_ant)), "b", linewidth=2, label="magnitude"
        )
        ax2 = ax.twinx()
        h2 = ax2.plot(
            f,
            (180 / np.pi) * np.unwrap(np.angle(s11_ant)),
            "r--",
            linewidth=2,
            label="phase",
        )

        h = h1 + h2
        labels = [l.get_label() for l in h]
        ax.legend(h, labels, loc=0, fontsize=12)

        ax.set_xlim([60, 160])
        ax.set_ylim([-18, 2])
        ax.set_yticks(np.arange(-16, 1, 4))
        ax.set_xticks([60, 80, 100, 120, 140, 160])
        ax.set_xticklabels([])

        ax2.set_ylim([-800 - 100, 0 + 100])
        ax2.set_yticks(np.arange(-800, 1, 200))

        ax.grid()
        ax.set_ylabel("magnitude [dB]", fontsize=14)
        ax2.set_ylabel("phase [degrees]", fontsize=14)
        ax.text(63, -3, "(a)", fontsize=18)

        # Losses
        # ------
        ax = f1.add_axes([0.11, 0.1, 0.73, 0.4])
        h1 = ax.plot(f, 100 * (1 - Gb), "g", linewidth=2, label="balun loss")
        h2 = ax.plot(f, 100 * (1 - Gc), "r", linewidth=2, label="connector loss")
        h3 = ax.plot(
            f, 100 * (1 - Gb * Gc), "k", linewidth=2, label="balun + connector loss"
        )

        h = h1 + h2 + h3
        labels = [l.get_label() for l in h]
        ax.legend(h, labels, loc=0, fontsize=12)

        ax.set_xlim([60, 160])
        ax.set_ylim([-0.1, 1.1])
        ax.set_xticks([60, 80, 100, 120, 140, 160])
        # ax.set_xticklabels([])

        ax.grid()
        ax.set_ylabel("loss [%]", fontsize=14)
        ax.set_xlabel("frequency [MHz]", fontsize=14)
        ax.text(63, 0.87, "(b)", fontsize=18)

    plt.savefig(
        "/data5/raul/EDGES/results/plots/20181022/antenna_reflection_loss.pdf",
        bbox_inches="tight",
    )

    if beam_factor:
        plt.figure(figsize=(11, 4))

        # ---------------------------------------
        plt.subplot(1, 2, 1)

        f, lst_table, bf_table = beams.beam_factor_table_read(
            "/data5/raul/EDGES/calibration/beam_factors/mid_band/beam_factor_table_hires.hdf5"
        )

        lst_in = np.arange(0, 24, 24 / 144)
        bf = beams.beam_factor_table_evaluate(f, lst_table, bf_table, lst_in)

        plt.imshow(
            bf,
            interpolation="none",
            aspect="auto",
            vmin=0.95,
            vmax=1.05,
            extent=[60, 160, 24, 0],
        )
        plt.yticks(np.arange(0, 25, 4))
        plt.grid()
        plt.colorbar()
        plt.xlabel("frequency [MHz]")
        plt.ylabel("LST [hr]")

        # ---------------------------------------
        plt.subplot(1, 2, 2)

        lst_in = np.arange(0, 24, 4)
        bf = beams.beam_factor_table_evaluate(f, lst_table, bf_table, lst_in)

        plt.plot(f, bf.T)
        plt.legend(["0 hr", "4 hr", "8 hr", "12 hr", "16 hr", "20 hr"], loc=3)
        plt.xlabel("frequency MHz]")
        plt.ylabel(r"correction factor, $C$")

        plt.savefig(
            "/data5/raul/EDGES/results/plots/20181022/beam_factor.pdf",
            bbox_inches="tight",
        )
