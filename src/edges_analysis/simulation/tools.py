import matplotlib.pyplot as plt
import numpy as np
from edges_cal import modelling as mdl
from src.edges_analysis import basic as ba

edges_folder = ""  # TODO: remove


def residuals_of_simulations(case, name_flag, Nfg=5, FLOW=60, FHIGH=150):
    folder_plot = edges_folder + "plots/20191227/"

    cases = {
        "inf ground plane": 11,
        "30x30 gaussian": 12,
        "30x30 flat index": 13,
        "30x30 gaussian guzman": 14,
        "30x30 gaussian LW": 15,
        "WIPL-D 101 haslam gaussian": 21,
        "WIPL-D 102 haslam gaussian": 22,
        "WIPL-D 103 haslam gaussian": 23,
        "30x30 gaussian corrected": 31,
        "30x30 gaussian loss-normalized": 51,
    }

    if case in cases:
        case = cases[case]

    if case not in cases.values():
        raise ValueError("case must be one of {}".format(cases.values()))

    if case == 11:
        t_all = np.genfromtxt(
            "/run/media/raul/WD_RED_6TB/EDGES_vol2/mid_band/calibration/beam_factors/raw"
            "/mid_band_50-200MHz_90deg_alan1_haslam_gaussian_index_2.4_2.65_sigma_deg_8"
            ".5_reffreq_90MHz_tant.txt"
        )
        f = np.genfromtxt(
            "/run/media/raul/WD_RED_6TB/EDGES_vol2/mid_band/calibration/beam_factors/raw"
            "/mid_band_50-200MHz_90deg_alan1_haslam_gaussian_index_2.4_2.65_sigma_deg_8"
            ".5_reffreq_90MHz_freq.txt"
        )
        LST = np.genfromtxt(
            "/run/media/raul/WD_RED_6TB/EDGES_vol2/mid_band/calibration/beam_factors/raw"
            "/mid_band_50-200MHz_90deg_alan1_haslam_gaussian_index_2.4_2.65_sigma_deg_8"
            ".5_reffreq_90MHz_LST.txt"
        )
        bf_all = np.genfromtxt(
            "/run/media/raul/WD_RED_6TB/EDGES_vol2/mid_band/calibration/beam_factors/raw"
            "/mid_band_50-200MHz_90deg_alan1_haslam_gaussian_index_2.4_2.65_sigma_deg_8"
            ".5_reffreq_90MHz_data.txt"
        )
    elif case == 12:
        t_all = np.genfromtxt(
            "/run/media/raul/WD_RED_6TB/EDGES_vol2/mid_band/calibration/beam_factors/raw"
            "/mid_band_50-200MHz_90deg_alan0_haslam_gaussian_index_2.4_2.65_sigma_deg_8"
            ".5_reffreq_90MHz_tant.txt"
        )
        f = np.genfromtxt(
            "/run/media/raul/WD_RED_6TB/EDGES_vol2/mid_band/calibration/beam_factors/raw"
            "/mid_band_50-200MHz_90deg_alan0_haslam_gaussian_index_2.4_2.65_sigma_deg_8"
            ".5_reffreq_90MHz_freq.txt"
        )
        LST = np.genfromtxt(
            "/run/media/raul/WD_RED_6TB/EDGES_vol2/mid_band/calibration/beam_factors/raw"
            "/mid_band_50-200MHz_90deg_alan0_haslam_gaussian_index_2.4_2.65_sigma_deg_8"
            ".5_reffreq_90MHz_LST.txt"
        )
        bf_all = np.genfromtxt(
            "/run/media/raul/WD_RED_6TB/EDGES_vol2/mid_band/calibration/beam_factors/raw"
            "/mid_band_50-200MHz_90deg_alan0_haslam_gaussian_index_2.4_2.65_sigma_deg_8"
            ".5_reffreq_90MHz_data.txt"
        )
    elif case == 12:
        t_all = np.genfromtxt(
            "/run/media/raul/WD_RED_6TB/EDGES_vol2/mid_band/calibration/beam_factors/raw"
            "/mid_band_50-200MHz_90deg_alan0_haslam_flat_index_2.56_reffreq_90MHz_tant.txt"
        )
        f = np.genfromtxt(
            "/run/media/raul/WD_RED_6TB/EDGES_vol2/mid_band/calibration/beam_factors/raw"
            "/mid_band_50-200MHz_90deg_alan0_haslam_flat_index_2.56_reffreq_90MHz_freq.txt"
        )
        LST = np.genfromtxt(
            "/run/media/raul/WD_RED_6TB/EDGES_vol2/mid_band/calibration/beam_factors/raw"
            "/mid_band_50-200MHz_90deg_alan0_haslam_flat_index_2.56_reffreq_90MHz_LST.txt"
        )
        bf_all = np.genfromtxt(
            "/run/media/raul/WD_RED_6TB/EDGES_vol2/mid_band/calibration/beam_factors/raw"
            "/mid_band_50-200MHz_90deg_alan0_haslam_flat_index_2.56_reffreq_90MHz_data.txt"
        )
    elif case == 14:
        t_all = np.genfromtxt(
            "/run/media/raul/WD_RED_6TB/EDGES_vol2/mid_band/calibration/beam_factors/raw"
            "/mid_band_50-200MHz_90deg_alan0_guzman_gaussian_index_2.4_2.65_sigma_deg_8"
            ".5_reffreq_90MHz_tant.txt"
        )
        f = np.genfromtxt(
            "/run/media/raul/WD_RED_6TB/EDGES_vol2/mid_band/calibration/beam_factors/raw"
            "/mid_band_50-200MHz_90deg_alan0_guzman_gaussian_index_2.4_2.65_sigma_deg_8"
            ".5_reffreq_90MHz_freq.txt"
        )
        LST = np.genfromtxt(
            "/run/media/raul/WD_RED_6TB/EDGES_vol2/mid_band/calibration/beam_factors/raw"
            "/mid_band_50-200MHz_90deg_alan0_guzman_gaussian_index_2.4_2.65_sigma_deg_8"
            ".5_reffreq_90MHz_LST.txt"
        )
        bf_all = np.genfromtxt(
            "/run/media/raul/WD_RED_6TB/EDGES_vol2/mid_band/calibration/beam_factors/raw"
            "/mid_band_50-200MHz_90deg_alan0_guzman_gaussian_index_2.4_2.65_sigma_deg_8"
            ".5_reffreq_90MHz_data.txt"
        )
    elif case == 15:
        t_all = np.genfromtxt(
            "/run/media/raul/WD_RED_6TB/EDGES_vol2/mid_band/calibration/beam_factors/raw"
            "/mid_band_50-200MHz_90deg_alan0_LW_gaussian_index_2.4_2.65_sigma_deg_8"
            ".5_reffreq_90MHz_tant.txt"
        )
        f = np.genfromtxt(
            "/run/media/raul/WD_RED_6TB/EDGES_vol2/mid_band/calibration/beam_factors/raw"
            "/mid_band_50-200MHz_90deg_alan0_LW_gaussian_index_2.4_2.65_sigma_deg_8"
            ".5_reffreq_90MHz_freq.txt"
        )
        LST = np.genfromtxt(
            "/run/media/raul/WD_RED_6TB/EDGES_vol2/mid_band/calibration/beam_factors/raw"
            "/mid_band_50-200MHz_90deg_alan0_LW_gaussian_index_2.4_2.65_sigma_deg_8"
            ".5_reffreq_90MHz_LST.txt"
        )
        bf_all = np.genfromtxt(
            "/run/media/raul/WD_RED_6TB/EDGES_vol2/mid_band/calibration/beam_factors/raw"
            "/mid_band_50-200MHz_90deg_alan0_LW_gaussian_index_2.4_2.65_sigma_deg_8"
            ".5_reffreq_90MHz_data.txt"
        )
    elif case == 21:
        t_all = np.genfromtxt(
            "/run/media/raul/WD_RED_6TB/EDGES_vol2/mid_band/calibration/beam_factors/raw"
            "/mid_band_50-200MHz_WIPL-D_101_90deg_haslam_gaussian_index_2.4_2.65_sigma_deg_8"
            ".5_reffreq_90MHz_tant.txt"
        )
        f = np.genfromtxt(
            "/run/media/raul/WD_RED_6TB/EDGES_vol2/mid_band/calibration/beam_factors/raw"
            "/mid_band_50-200MHz_WIPL-D_101_90deg_haslam_gaussian_index_2.4_2.65_sigma_deg_8"
            ".5_reffreq_90MHz_freq.txt"
        )
        LST = np.genfromtxt(
            "/run/media/raul/WD_RED_6TB/EDGES_vol2/mid_band/calibration/beam_factors/raw"
            "/mid_band_50-200MHz_WIPL-D_101_90deg_haslam_gaussian_index_2.4_2.65_sigma_deg_8"
            ".5_reffreq_90MHz_LST.txt"
        )
        l_all = np.genfromtxt(
            "/run/media/raul/WD_RED_6TB/EDGES_vol2/mid_band/calibration/beam_factors/raw"
            "/mid_band_50-200MHz_WIPL-D_101_90deg_haslam_gaussian_index_2.4_2.65_sigma_deg_8"
            ".5_reffreq_90MHz_loss.txt"
        )
        bf_all = np.genfromtxt(
            "/run/media/raul/WD_RED_6TB/EDGES_vol2/mid_band/calibration/beam_factors/raw"
            "/mid_band_50-200MHz_WIPL-D_101_90deg_haslam_gaussian_index_2.4_2.65_sigma_deg_8"
            ".5_reffreq_90MHz_beam_factor.txt"
        )

        # Adding the ground loss
        t_all = t_all + 300 * l_all

    elif case == 22:
        t_all = np.genfromtxt(
            "/run/media/raul/WD_RED_6TB/EDGES_vol2/mid_band/calibration/beam_factors/raw"
            "/mid_band_50-200MHz_WIPL-D_102_90deg_haslam_gaussian_index_2.4_2.65_sigma_deg_8"
            ".5_reffreq_90MHz_tant.txt"
        )
        f = np.genfromtxt(
            "/run/media/raul/WD_RED_6TB/EDGES_vol2/mid_band/calibration/beam_factors/raw"
            "/mid_band_50-200MHz_WIPL-D_102_90deg_haslam_gaussian_index_2.4_2.65_sigma_deg_8"
            ".5_reffreq_90MHz_freq.txt"
        )
        LST = np.genfromtxt(
            "/run/media/raul/WD_RED_6TB/EDGES_vol2/mid_band/calibration/beam_factors/raw"
            "/mid_band_50-200MHz_WIPL-D_102_90deg_haslam_gaussian_index_2.4_2.65_sigma_deg_8"
            ".5_reffreq_90MHz_LST.txt"
        )
        l_all = np.genfromtxt(
            "/run/media/raul/WD_RED_6TB/EDGES_vol2/mid_band/calibration/beam_factors/raw"
            "/mid_band_50-200MHz_WIPL-D_102_90deg_haslam_gaussian_index_2.4_2.65_sigma_deg_8"
            ".5_reffreq_90MHz_loss.txt"
        )
        bf_all = np.genfromtxt(
            "/run/media/raul/WD_RED_6TB/EDGES_vol2/mid_band/calibration/beam_factors/raw"
            "/mid_band_50-200MHz_WIPL-D_102_90deg_haslam_gaussian_index_2.4_2.65_sigma_deg_8"
            ".5_reffreq_90MHz_beam_factor.txt"
        )

        # Adding the ground loss
        t_all = t_all + 300 * l_all
    elif case == 23:
        t_all = np.genfromtxt(
            "/run/media/raul/WD_RED_6TB/EDGES_vol2/mid_band/calibration/beam_factors/raw"
            "/mid_band_50-200MHz_WIPL-D_103_90deg_haslam_gaussian_index_2.4_2.65_sigma_deg_8"
            ".5_reffreq_90MHz_tant.txt"
        )
        f = np.genfromtxt(
            "/run/media/raul/WD_RED_6TB/EDGES_vol2/mid_band/calibration/beam_factors/raw"
            "/mid_band_50-200MHz_WIPL-D_103_90deg_haslam_gaussian_index_2.4_2.65_sigma_deg_8"
            ".5_reffreq_90MHz_freq.txt"
        )
        LST = np.genfromtxt(
            "/run/media/raul/WD_RED_6TB/EDGES_vol2/mid_band/calibration/beam_factors/raw"
            "/mid_band_50-200MHz_WIPL-D_103_90deg_haslam_gaussian_index_2.4_2.65_sigma_deg_8"
            ".5_reffreq_90MHz_LST.txt"
        )
        l_all = np.genfromtxt(
            "/run/media/raul/WD_RED_6TB/EDGES_vol2/mid_band/calibration/beam_factors/raw"
            "/mid_band_50-200MHz_WIPL-D_103_90deg_haslam_gaussian_index_2.4_2.65_sigma_deg_8"
            ".5_reffreq_90MHz_loss.txt"
        )
        bf_all = np.genfromtxt(
            "/run/media/raul/WD_RED_6TB/EDGES_vol2/mid_band/calibration/beam_factors/raw"
            "/mid_band_50-200MHz_WIPL-D_103_90deg_haslam_gaussian_index_2.4_2.65_sigma_deg_8"
            ".5_reffreq_90MHz_beam_factor.txt"
        )

        # Adding the ground loss
        t_all = t_all + 300 * l_all

    # 30x30 m,   SAME AS CASE 12 BUT WITH CORRECT COMPUTATION OF ANTENNA
    # TEMPERATURE AND BEAM FACTOR
    elif case == 31:
        t_all = np.genfromtxt(
            "/run/media/raul/WD_RED_6TB/EDGES_vol2/mid_band/calibration/beam_factors/raw"
            "/CORRECT_mid_band_50-200MHz_90deg_alan0_haslam_gaussian_index_2.4_2.65_sigma_deg_8"
            ".5_reffreq_90MHz_tant.txt"
        )
        f = np.genfromtxt(
            "/run/media/raul/WD_RED_6TB/EDGES_vol2/mid_band/calibration/beam_factors/raw"
            "/CORRECT_mid_band_50-200MHz_90deg_alan0_haslam_gaussian_index_2.4_2.65_sigma_deg_8"
            ".5_reffreq_90MHz_freq.txt"
        )
        LST = np.genfromtxt(
            "/run/media/raul/WD_RED_6TB/EDGES_vol2/mid_band/calibration/beam_factors/raw"
            "/CORRECT_mid_band_50-200MHz_90deg_alan0_haslam_gaussian_index_2.4_2.65_sigma_deg_8"
            ".5_reffreq_90MHz_LST.txt"
        )
        bf_all = np.genfromtxt(
            "/run/media/raul/WD_RED_6TB/EDGES_vol2/mid_band/calibration/beam_factors/raw"
            "/CORRECT_mid_band_50-200MHz_90deg_alan0_haslam_gaussian_index_2.4_2.65_sigma_deg_8"
            ".5_reffreq_90MHz_beam_factor.txt"
        )

    # 30x30 m,   SAME AS CASE 12 BUT WITH CORRECT AND "LOSS-NORMALIZED"
    # COMPUTATION OF ANTENNA TEMPERATURE AND BEAM FACTOR
    elif case == 51:
        t_all = np.genfromtxt(
            "/run/media/raul/WD_RED_6TB/EDGES_vol2/mid_band/calibration/beam_factors/raw"
            "/NORMALIZED_mid_band_50-150MHz_90deg_alan0_haslam_gaussian_index_2.4_2"
            ".65_sigma_deg_8.5_reffreq_90MHz_tant.txt"
        )
        f = np.genfromtxt(
            "/run/media/raul/WD_RED_6TB/EDGES_vol2/mid_band/calibration/beam_factors/raw"
            "/NORMALIZED_mid_band_50-150MHz_90deg_alan0_haslam_gaussian_index_2.4_2"
            ".65_sigma_deg_8.5_reffreq_90MHz_freq.txt"
        )
        LST = np.genfromtxt(
            "/run/media/raul/WD_RED_6TB/EDGES_vol2/mid_band/calibration/beam_factors/raw"
            "/NORMALIZED_mid_band_50-150MHz_90deg_alan0_haslam_gaussian_index_2.4_2"
            ".65_sigma_deg_8.5_reffreq_90MHz_LST.txt"
        )
        bf_all = np.genfromtxt(
            "/run/media/raul/WD_RED_6TB/EDGES_vol2/mid_band/calibration/beam_factors/raw"
            "/NORMALIZED_mid_band_50-150MHz_90deg_alan0_haslam_gaussian_index_2.4_2"
            ".65_sigma_deg_8.5_reffreq_90MHz_beam_factor.txt"
        )

    fc = f[(f >= FLOW) & (f <= FHIGH)]
    tc = t_all[:, (f >= FLOW) & (f <= FHIGH)]
    bc = bf_all[:, (f >= FLOW) & (f <= FHIGH)]

    GHA = LST - 17.76
    GHA[GHA < 0] += 24

    index_sort = np.argsort(GHA)

    GHA_inc = GHA[index_sort]
    tc_inc = tc[index_sort, :]
    bc_inc = bc[index_sort, :]

    res = np.zeros((24, len(fc)))
    resb = np.zeros((24, len(fc)))
    bf_all = np.zeros((24, len(fc)))

    for i in range(24):
        index = np.arange(len(GHA_inc))

        ix = index[(GHA_inc >= i) & (GHA_inc < (i + 1))]
        tx = tc_inc[ix, :]
        bx = bc_inc[ix, :]

        # int(ix)
        avt = np.mean(tx, axis=0)
        avb = np.mean(bx, axis=0)

        pc = mdl.fit_polynomial_fourier("LINLOG", fc / 200, avt, Nfg)
        mc = mdl.model_evaluate("LINLOG", pc[0], fc / 200)
        rc = avt - mc
        res[i, :] = rc

        pc = mdl.fit_polynomial_fourier("LINLOG", fc / 200, avb, Nfg)
        mbc = mdl.model_evaluate("LINLOG", pc[0], fc / 200)
        rbc = avb - mbc
        resb[i, :] = rbc

        bf_all[i, :] = avb

    gg = np.arange(24)

    r_low = res[(gg >= 6) & (gg <= 17), :]

    r_high = res[(gg < 6) | (gg > 17), :]
    r_high1 = r_high[0:6, :]
    r_high2 = r_high[6::, :]
    r_high = np.vstack((r_high2, r_high1))

    rb_low = resb[(gg >= 6) & (gg <= 17), :]

    rb_high = resb[(gg < 6) | (gg > 17), :]
    rb_high1 = rb_high[0:6, :]
    rb_high2 = rb_high[6::, :]
    rb_high = np.vstack((rb_high2, rb_high1))

    b_low = bf_all[(gg >= 6) & (gg <= 17), :]

    b_high = bf_all[(gg < 6) | (gg > 17), :]
    b_high1 = b_high[0:6, :]
    b_high2 = b_high[6::, :]
    b_high = np.vstack((b_high2, b_high1))

    f = np.copy(fc)

    plt.figure(figsize=[13, 11])
    plt.subplot(1, 2, 1)
    c = "b"
    for i in range(len(r_low[:, 0])):
        plt.plot(f, r_low[i, :] - 0.5 * i, c)
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
        plt.plot(f, r_high[i, :] - 2 * i, c)
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
        folder_plot + name_flag + "_simulated_residuals.pdf", bbox_inches="tight"
    )

    plt.figure(figsize=[13, 11])
    plt.subplot(1, 2, 1)
    c = "b"
    for i in range(len(b_low[:, 0])):
        plt.plot(f, b_low[i, :] - 0.1 * i, c)
        if c == "b":
            c = "r"
        elif c == "r":
            c = "b"
    plt.xlim([60, 150])
    plt.grid()
    plt.ylim([-0.2, 1.1])
    plt.xlabel("frequency [MHz]")
    plt.ylabel("GHA\n [0.1 per division]")
    plt.yticks(np.arange(-0.1, 1.01, 0.1), np.arange(17, 5, -1))

    plt.subplot(1, 2, 2)
    c = "b"
    for i in range(len(b_high[:, 0])):
        plt.plot(f, b_high[i, :] - 0.1 * i, c)
        if c == "b":
            c = "r"
        elif c == "r":
            c = "b"
    plt.xlim([60, 150])
    plt.grid()
    plt.ylim([-0.2, 1.1])
    plt.xlabel("frequency [MHz]")
    plt.ylabel("GHA\n [0.1 per division]")
    plt.yticks(np.arange(-0.1, 1.01, 0.1), np.arange(5, -7, -1))

    plt.savefig(
        folder_plot + name_flag + "_simulated_correction.pdf", bbox_inches="tight"
    )

    plt.figure(figsize=[13, 11])
    plt.subplot(1, 2, 1)
    c = "b"
    for i in range(len(rb_low[:, 0])):
        plt.plot(f, rb_low[i, :] - 0.005 * i, c)
        if c == "b":
            c = "r"
        elif c == "r":
            c = "b"
    plt.xlim([60, 150])
    plt.grid()
    plt.ylim([-0.06, 0.005])
    plt.xlabel("frequency [MHz]")
    plt.ylabel("GHA\n [0.05 per division]")
    plt.yticks(np.arange(-0.055, 0.0025, 0.005), np.arange(17, 5, -1))

    plt.subplot(1, 2, 2)
    c = "b"
    for i in range(len(rb_high[:, 0])):
        plt.plot(f, rb_high[i, :] - 0.005 * i, c)
        if c == "b":
            c = "r"
        elif c == "r":
            c = "b"
    plt.xlim([60, 150])
    plt.grid()
    plt.ylim([-0.06, 0.005])
    plt.xlabel("frequency [MHz]")
    plt.ylabel("GHA\n [0.05 per division]")
    plt.yticks(np.arange(-0.055, 0.0025, 0.005), np.arange(5, -7, -1))

    plt.savefig(
        folder_plot + name_flag + "_simulated_correction_residuals.pdf",
        bbox_inches="tight",
    )

    return fc, GHA_inc, res
