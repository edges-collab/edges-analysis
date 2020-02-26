import matplotlib.pyplot as plt
import numpy as np
from edges_cal import modelling as mdl

edges_folder = ""  # TODO: remove


def residuals_of_simulations(case, name_flag, n_fg=5, f_low=60, f_high=150):
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

    prefix = (
        "/run/media/raul/WD_RED_6TB/EDGES_vol2/mid_band/calibration/beam_factors/raw/"
    )

    files = {
        11: "mid_band_50-200MHz_90deg_alan1_haslam_gaussian_index_2.4_2.65_sigma_deg_8"
        ".5_reffreq_90MHz_{}.txt",
        12: "mid_band_50-200MHz_90deg_alan0_haslam_gaussian_index_2.4_2.65_sigma_deg_8"
        ".5_reffreq_90MHz_{}.txt",
        13: "mid_band_50-200MHz_90deg_alan0_haslam_flat_index_2.56_reffreq_90MHz_{}.txt",
        14: "mid_band_50-200MHz_90deg_alan0_guzman_gaussian_index_2.4_2.65_sigma_deg_8"
        ".5_reffreq_90MHz_{}.txt",
        15: "mid_band_50-200MHz_90deg_alan0_LW_gaussian_index_2.4_2.65_sigma_deg_8"
        ".5_reffreq_90MHz_{}.txt",
        21: "mid_band_50-200MHz_WIPL-D_101_90deg_haslam_gaussian_index_2.4_2.65_sigma_deg_8"
        ".5_reffreq_90MHz_{}.txt",
        22: "mid_band_50-200MHz_WIPL-D_102_90deg_haslam_gaussian_index_2.4_2.65_sigma_deg_8"
        ".5_reffreq_90MHz_{}.txt",
        23: "mid_band_50-200MHz_WIPL-D_103_90deg_haslam_gaussian_index_2.4_2.65_sigma_deg_8"
        ".5_reffreq_90MHz_{}.txt",
        31: "CORRECT_mid_band_50-200MHz_90deg_alan0_haslam_gaussian_index_2.4_2.65_sigma_deg_8"
        ".5_reffreq_90MHz_{}.txt",
        51: "NORMALIZED_mid_band_50-150MHz_90deg_alan0_haslam_gaussian_index_2.4_2"
        ".65_sigma_deg_8.5_reffreq_90MHz_{}.txt",
    }

    fl = prefix + files[case]
    t_all = np.genfromtxt(fl.format("tant"))
    f = np.genfromtxt(fl.format("freq"))
    LST = np.genfromtxt(fl.format("LST"))
    bf_all = np.genfromtxt(fl.format("data"))

    if case in [21, 22, 23]:
        loss = np.genfromtxt(fl.format("loss"))
        t_all += 300 * loss

    mask = (f >= f_low) & (f <= f_high)
    fc = f[mask]
    tc = t_all[:, mask]
    bc = bf_all[:, mask]

    GHA = LST - 17.76
    GHA[GHA < 0] += 24

    index_sort = np.argsort(GHA)

    GHA_inc = GHA[index_sort]
    tc_inc = tc[index_sort]
    bc_inc = bc[index_sort]

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

        pc = mdl.fit_polynomial_fourier("LINLOG", fc / 200, avt, n_fg)
        mc = mdl.model_evaluate("LINLOG", pc[0], fc / 200)
        rc = avt - mc
        res[i, :] = rc

        pc = mdl.fit_polynomial_fourier("LINLOG", fc / 200, avb, n_fg)
        mbc = mdl.model_evaluate("LINLOG", pc[0], fc / 200)
        rbc = avb - mbc
        resb[i, :] = rbc

        bf_all[i, :] = avb

    gg = np.arange(24)

    out = {}
    for name, x in zip(
        ["residuals", "correction", "correction_residuals"], [res, resb, bf_all]
    ):
        out[name]["low"] = x[(gg >= 6) & (gg <= 17), :]
        high = x[(gg < 6) | (gg > 17), :]
        high1 = high[:6]
        high2 = high[6:]
        out[name]["high"] = np.vstack((high2, high1))

    f = np.copy(fc)

    def plot_it(key, ylim_1, yticks_1, div_1, ylim_2=None, yticks_2=None, div_2=None):
        fig, ax = plt.subplots(1, 2, figsize=[13, 11], sharex=True)

        for i, val in enumerate(out[key]["low"]):
            ax[0].plot(f, val - div_1 * i, "br"[i % 2])

        for i, val in enumerate(out[key]["high"]):
            ax[1].plot(f, val - (div_2 or div_1) * i, "br"[i % 2])

        for axx in ax:
            axx.set_xlim([60, 150])
            axx.grid()
            axx.set_xlabel("frequency [MHz]")

        ax[0].set_ylim(ylim_1)
        ax[0].set_ylabel(f"GHA\n [{div_1} K per division]")
        ax[0].yaxis.yticks(yticks_1, np.arange(17, 5, -1))

        ax[1].set_ylim(ylim_2 or ylim_1)
        ax[1].set_ylabel(f"GHA\n [{div_2 or div_1} K per division]")
        ax[1].yaxis.yticks(yticks_2 or yticks_1, np.arange(5, -7, -1))

        plt.savefig(
            folder_plot + name_flag + f"_simulated_{key}.pdf", bbox_inches="tight"
        )

    plot_it(
        "residuals",
        ylim_1=(-6, 0.5),
        yticks_1=np.arange(-5.5, 0.1, 0.5),
        div_1=0.5,
        ylim_2=(-24, 2),
        yticks_2=np.arange(-22, 0.1, 2),
        div_2=2,
    )
    plot_it(
        "correction", ylim_1=(-0.2, 1.1), yticks_1=np.arange(-0.1, 1.01, 0.1), div_1=0.1
    )
    plot_it(
        "correction_residuals",
        ylim_1=(-0.06, 0.005),
        yticks_1=np.arange(-0.055, 0.0025, 0.005),
        div_1=0.005,
    )

    return fc, GHA_inc, res
