from os import listdir, makedirs
from os.path import exists

import h5py
import numpy as np
import src.edges_analysis
import src.edges_analysis.analysis.levels
import src.edges_analysis.analysis.tools
from edges_cal import modelling as mdl
from edges_cal import receiver_calibration_func as rcf
from edges_cal import reflection_coefficient as rc
from edges_cal import s11_correction as s11c
from edges_cal.cal_coefficients import EdgesFrequencyRange, HotLoadCorrection
from edges_io.io import SwitchingState
from matplotlib import pyplot as plt

from . import beams, filters, io, loss, plots, rfi
from . import s11 as s11m
from . import tools
from .levels import level1_to_level2, level2_to_level3
from ..simulation import data_models as dm

edges_folder = ""  # TODO: remove
home_folder = ""  # TODO: remove
MRO_folder = ""  # TODO: remove
edges_folder_v1 = ""  # TODO: remove

CALFILES = {
    1: "nominal/calibration_files/original/calibration_file_receiver1_cterms7_wterms7.txt",
    2: "nominal/calibration_files/original/calibration_file_receiver1_cterms7_wterms8.txt",
    3: "nominal/calibration_files/original/calibration_file_receiver1_cterms7_wterms15.txt",
    4: "nominal/calibration_files/original/calibration_file_receiver1_cterms8_wterms8.txt",
    5: "nominal/calibration_files/original/calibration_file_receiver1_cterms9_wterms9.txt",
    6: "nominal/calibration_files/original/calibration_file_receiver1_cterms10_wterms10.txt",
    7: "nominal_50-150MHz_no_rfi/calibration_files/calibration_file_receiver1_cterms9_wterms9_50-150MHz_no_rfi.txt",
    8: "nominal_50-150MHz_no_rfi/calibration_files/calibration_file_receiver1_cterms8_wterms8_50-150MHz_no_rfi.txt",
    30: "nominal_cleaned_60_120MHz/calibration_files/calibration_file_receiver1_cterms7_wterms6.txt",
    31: "nominal_cleaned_60_120MHz/calibration_files/calibration_file_receiver1_cterms7_wterms8.txt",
    32: "nominal_cleaned_60_120MHz/calibration_files/calibration_file_receiver1_cterms7_wterms9.txt",
    33: "nominal_cleaned_60_120MHz/calibration_files/calibration_file_receiver1_cterms6_wterms4.txt",
    34: "nominal_cleaned_60_120MHz/calibration_files/calibration_file_receiver1_cterms6_wterms8.txt",
    35: "nominal_cleaned_60_120MHz/calibration_files/calibration_file_receiver1_cterms6_wterms9.txt",
    39: "nominal_cleaned_60_120MHz/calibration_files/calibration_file_receiver1_cterms5_wterms9.txt",
    21: "nominal/calibration_files/calibration_file_receiver1_cterms7_wterms7.txt",
    22: "nominal/calibration_files/calibration_file_receiver1_cterms7_wterms8.txt",
    23: "nominal/calibration_files/calibration_file_receiver1_cterms7_wterms9.txt",
    24: "nominal/calibration_files/calibration_file_receiver1_cterms7_wterms10.txt",
    25: "nominal/calibration_files/calibration_file_receiver1_cterms8_wterms8.txt",
    26: "nominal/calibration_files/calibration_file_receiver1_cterms8_wterms11.txt",
    40: "nominal/calibration_files/60_85MHz/calibration_file_receiver1_60_85MHz_cterms4_wterms6.txt",
    88: "nominal/calibration_files/calibration_file_receiver1_50_150MHz_cterms8_wterms8.txt",
    810: "nominal/calibration_files/calibration_file_receiver1_50_150MHz_cterms8_wterms10.txt",
    811: "nominal/calibration_files/calibration_file_receiver1_50_150MHz_cterms8_wterms11.txt",
    100: "nominal/calibration_files/calibration_file_receiver1_50_190MHz_cterms10_wterms13.txt",
    200: "/nominal_2019_12_50-150MHz_try1/calibration_files/calibration_file_receiver1_55_150MHz_cterms14_wterms14.txt",
    201: "/nominal_2019_12_50-150MHz_try1/calibration_files/calibration_file_receiver1_50_150MHz_cterms7_wterms8.txt",
    202: "/nominal_2019_12_50-150MHz_try1/calibration_files/calibration_file_receiver1_50_150MHz_cterms7_wterms11.txt",
    203: "/nominal_2019_12_50-150MHz_try1_LNA_rep1/calibration_files/calibration_file_receiver1_50_150MHz_cterms7_wterms8.txt",
    204: "/nominal_2019_12_50-150MHz_try1_LNA_rep2/calibration_files/calibration_file_receiver1_50_150MHz_cterms7_wterms8.txt",
    205: "/nominal_2019_12_50-150MHz_try1_LNA_rep12/calibration_files/calibration_file_receiver1_50_150MHz_cterms7_wterms8.txt",
    301: "/nominal_2019_12_50-150MHz_try1/calibration_files/calibration_file_receiver1_60_120MHz_cterms7_wterms4.txt",
    302: "/nominal_2019_12_50-150MHz_try1/calibration_files/calibration_file_receiver1_60_120MHz_cterms7_wterms5.txt",
    303: "/nominal_2019_12_50-150MHz_try1/calibration_files/calibration_file_receiver1_60_120MHz_cterms7_wterms9.txt",
    304: "/nominal_2019_12_50-150MHz_try1/calibration_files/calibration_file_receiver1_60_120MHz_cterms8_wterms4.txt",
    305: "/nominal_2019_12_50-150MHz_try1/calibration_files/calibration_file_receiver1_60_120MHz_cterms8_wterms5.txt",
    306: "/nominal_2019_12_50-150MHz_try1/calibration_files/calibration_file_receiver1_60_120MHz_cterms8_wterms9.txt",
    307: "/nominal_2019_12_50-150MHz_try1/calibration_files/calibration_file_receiver1_60_120MHz_cterms5_wterms4.txt",
    308: "/nominal_2019_12_50-150MHz_try1/calibration_files/calibration_file_receiver1_60_120MHz_cterms6_wterms4.txt",
    401: "/nominal_2019_12_50-150MHz_LNA1_a1_h1_o1_s1_sim2/calibration_files/calibration_file_receiver1_50_150MHz_cterms7_wterms8.txt",
    402: "/nominal_2019_12_50-150MHz_LNA1_a1_h2_o1_s1_sim2/calibration_files/calibration_file_receiver1_50_150MHz_cterms7_wterms8.txt",
    403: "/nominal_2019_12_50-150MHz_LNA1_a2_h1_o1_s1_sim2/calibration_files/calibration_file_receiver1_50_150MHz_cterms7_wterms8.txt",
    404: "/nominal_2019_12_50-150MHz_LNA1_a2_h2_o1_s1_sim2/calibration_files/calibration_file_receiver1_50_150MHz_cterms7_wterms8.txt",
    405: "/nominal_2019_12_50-150MHz_LNA1_a2_h2_o1_s2_sim2/calibration_files/calibration_file_receiver1_50_150MHz_cterms7_wterms8.txt",
    406: "/nominal_2019_12_50-150MHz_LNA1_a2_h2_o2_s1_sim2/calibration_files/calibration_file_receiver1_50_150MHz_cterms7_wterms8.txt",
    407: "/nominal_2019_12_50-150MHz_LNA1_a2_h2_o2_s2_sim2/calibration_files/calibration_file_receiver1_50_150MHz_cterms7_wterms8.txt",
}


def models_antenna_s11_remove_delay(**kwargs):
    # TODO: figure out where this came from.
    raise NotImplementedError(
        "this is not the same as the remove_delay function from s11 "
        "(which was from original calibrate.py). This was supposed to"
        " be from edges.py but doesn't exist"
    )


def calibration_file_computation(
    calibration_date,
    folder,
    f_low,
    f_high,
    cterms_nominal,
    wterms_nominal,
    save_nominal=False,
    save_nominal_flag="",
    term_sweep=False,
    panels=4,
    plot_nominal=False,
):
    prefix = (
        edges_folder
        + "mid_band/calibration/receiver_calibration/receiver1/"
        + calibration_date
        + "/results/"
        + folder
    )
    # Location of saved results
    path_save = prefix + "/calibration_files/"

    # Spectra
    TuncV = np.genfromtxt(prefix + "/data/average_spectra_300_350.txt")

    # Frequency selection
    Tunc = TuncV[(TuncV[:, 0] >= f_low) & (TuncV[:, 0] <= f_high), :]

    ff = Tunc[:, 0]
    WW_all = np.zeros(len(ff))

    if calibration_date == "2018_01_25C" and folder in (
        "nominal_cleaned_60_120MHz",
        "nominal_cleaned_60_90MHz",
    ):
        TTae, WWae, TThe, WWhe, TToe, WWoe, TTse, WWse, TTqe, WWqe = Tunc.T[1:]
    elif calibration_date == "2018_01_25C" and folder in (
        "nominal_2019_12_50-150MHz_try1",
        "nominal_2019_12_50-150MHz_try1_LNA_rep1",
        "nominal_2019_12_50-150MHz_try1_LNA_rep2",
        "nominal_2019_12_50-150MHz_try1_LNA_rep12",
        "nominal_2019_12_50-150MHz_LNA1_a1_h1_o1_s1_sim2",
        "nominal_2019_12_50-150MHz_LNA1_a1_h2_o1_s1_sim2",
        "nominal_2019_12_50-150MHz_LNA1_a2_h1_o1_s1_sim2",
        "nominal_2019_12_50-150MHz_LNA1_a2_h2_o1_s1_sim2",
        "nominal_2019_12_50-150MHz_LNA1_a2_h2_o1_s2_sim2",
        "nominal_2019_12_50-150MHz_LNA1_a2_h2_o2_s1_sim2",
        "nominal_2019_12_50-150MHz_LNA1_a2_h2_o2_s2_sim2",
    ):
        TTae, TThe, TToe, TTse, TTqe = Tunc.T[1:6]
        WWae, WWhe, WWoe, WWse, WWqe = np.ones((5, len(ff)))
    else:
        TTae, TThe, TToe, TTse, TTqe = Tunc.T[[1, 2, 3, 4, 6]]
        WWae, WWhe, WWoe, WWse, WWqe = np.ones((5, len(ff)))

    raw_data = {
        "amb": {"temp": TTae, "weights": WWae},
        "hot": {"temp": TThe, "weights": WWhe},
        "shorted": {"temp": TTse, "weights": WWse},
        "open": {"temp": TToe, "weights": WWoe},
        "simu": {"temp": TTqe, "weights": WWqe},
    }

    # RFI cleaning
    n_sigma = 4
    index_good = np.zeros(len(ff), dtype=bool)
    for d in raw_data.values():
        wghts = rfi.cleaning_sweep(
            ff,
            d["temp"],
            d["weights"],
            window_width=4,
            n_poly=4,
            n_bootstrap=20,
            n_sigma=n_sigma,
            flip=True,
        )[1]
        index_good &= wghts != 0

    # Removing data points with zero weight. But for the rest, not using the weights
    # because they are pretty even across frequency
    f = ff[index_good]
    good_data = {key: val["temp"][index_good] for key, val in raw_data.items()}
    WW_all[index_good] = 1

    # Physical temperature
    Tphys = np.genfromtxt(prefix + "/temp/physical_temperatures.txt")
    phys_temp = {
        name: t for name, t in zip(["amb", "hot", "open", "shorted"], Tphys[:4])
    }

    #    Ta, Th, To, Ts = Tphys[:4]

    if calibration_date == "2018_01_25C" and folder in (
        "nominal_2019_12_50-150MHz_try1",
        "nominal_2019_12_50-150MHz_try1_LNA_rep1",
        "nominal_2019_12_50-150MHz_try1_LNA_rep2",
        "nominal_2019_12_50-150MHz_try1_LNA_rep12",
        "nominal_2019_12_50-150MHz_LNA1_a1_h1_o1_s1_sim2",
        "nominal_2019_12_50-150MHz_LNA1_a1_h2_o1_s1_sim2",
        "nominal_2019_12_50-150MHz_LNA1_a2_h1_o1_s1_sim2",
        "nominal_2019_12_50-150MHz_LNA1_a2_h2_o1_s1_sim2",
        "nominal_2019_12_50-150MHz_LNA1_a2_h2_o1_s2_sim2",
        "nominal_2019_12_50-150MHz_LNA1_a2_h2_o2_s1_sim2",
        "nominal_2019_12_50-150MHz_LNA1_a2_h2_o2_s2_sim2",
    ):
        phys_temp["simu"] = Tphys[4]
    else:
        phys_temp["simu"] = Tphys[5]

    # S11
    path_s11 = prefix + "/s11/"

    # S11 at original frequency array
    if calibration_date == "2019_10_25C":
        ffn = (ff / 200) - 0.5
        fn = (f / 200) - 0.5
        sim_name = "simu3"
        sim_basis = "fourier"
    elif (calibration_date == "2018_01_25C") and (
        (folder == "nominal_2019_12_50-150MHz_try1")
        or (folder == "nominal_2019_12_50-150MHz_try1_LNA_rep1")
        or (folder == "nominal_2019_12_50-150MHz_try1_LNA_rep2")
        or (folder == "nominal_2019_12_50-150MHz_try1_LNA_rep12")
        or (folder == "nominal_2019_12_50-150MHz_LNA1_a1_h1_o1_s1_sim2")
        or (folder == "nominal_2019_12_50-150MHz_LNA1_a1_h2_o1_s1_sim2")
        or (folder == "nominal_2019_12_50-150MHz_LNA1_a2_h1_o1_s1_sim2")
        or (folder == "nominal_2019_12_50-150MHz_LNA1_a2_h2_o1_s1_sim2")
        or (folder == "nominal_2019_12_50-150MHz_LNA1_a2_h2_o1_s2_sim2")
        or (folder == "nominal_2019_12_50-150MHz_LNA1_a2_h2_o2_s1_sim2")
        or (folder == "nominal_2019_12_50-150MHz_LNA1_a2_h2_o2_s2_sim2")
    ):
        ffn = (ff / 200) - 0.5
        fn = (f / 200) - 0.5

        sim_name = "simu"
        sim_basis = "fourier"
    else:
        ffn = (ff - 120) / 60
        fn = (f - 120) / 60
        sim_name = "simu"
        sim_basis = "polynomial"

    loads_full = {
        name: s11m.get_s11_model_from_file(path_s11, name, ffn)
        for name in ["LNA", "amb", "hot", "open", "shorted"]
    }
    sr_full = {
        kind: s11m.get_s11_model_from_file(
            path_s11, "sr", ffn, coeff=kind, basis="polynomial"
        )
        for kind in ["s11", "s12s21", "s22"]
    }
    rrsimu = s11m.get_s11_model_from_file(path_s11, sim_name, ffn, basis=sim_basis)
    GG = HotLoadCorrection.get_power_gain(sr_full, loads_full["hot"])

    loads = {
        name: s11m.get_s11_model_from_file(path_s11, name, fn)
        for name in ["LNA", "amb", "hot", "open", "shorted"]
    }
    sr = {
        kind: s11m.get_s11_model_from_file(
            path_s11, "sr", fn, coeff=kind, basis="polynomial"
        )
        for kind in ["s11", "s12s21", "s22"]
    }
    # rsimu = s11m.get_s11_model_from_file(path_s11, sim_name, fn, basis=sim_basis)
    G = HotLoadCorrection.get_power_gain(sr, loads["hot"])

    # temperature
    phys_temp["hot_corrected"] = G * phys_temp["hot"] + (1 - G) * phys_temp["amb"]

    # Sweeping parameters and computing RMS
    Tamb_internal = 300
    cterms = range(1, 15)
    wterms = range(1, 15)

    if term_sweep:
        RMS = np.zeros((4, len(wterms), len(cterms)))
        for j in cterms:
            for i in wterms:
                fig, ax = plt.subplots(
                    panels, 1, figsize=(6, panels * 3 - 6), sharex=True
                )
                fig_raw, ax_raw = plt.subplots(
                    panels, 1, figsize=(6, panels * 3 - 6), sharex=True
                )

                C1, C2, TU, TC, TS = rcf.get_calibration_quantities_iterative(
                    fn,
                    T_raw={
                        "ambient": good_data["amb"],
                        "hot_load": good_data["hot"],
                        "short": good_data["shorted"],
                        "open": good_data["open"],
                    },
                    gamma_rec=loads["LNA"],
                    gamma_ant={
                        "ambient": loads["amb"],
                        "hot_load": loads["hot"],
                        "short": loads["shorted"],
                        "open": loads["open"],
                    },
                    T_ant={
                        "ambient": phys_temp["amb"],
                        "hot_load": phys_temp["hot_corrected"],
                        "short": phys_temp["shorted"],
                        "open": phys_temp["open"],
                    },
                    cterms=j,
                    wterms=i,
                    Tamb_internal=Tamb_internal,
                )

                # Only open cable
                print("---------------------------------------------------")
                print(j + 1, i + 1)

                # Cross-check
                tcal = {}
                for k, (load, s11) in enumerate(
                    {**loads_full, **{"simu": rrsimu}}.items()
                ):
                    if load == "LNA":
                        continue

                    TT = rcf.calibrated_antenna_temperature(
                        raw_data[load],
                        s11,
                        loads_full["LNA"],
                        C1,
                        C2,
                        TU,
                        TC,
                        TS,
                        T_load=Tamb_internal,
                    )

                    if load == "hot":
                        TT = (TT - (1 - GG) * phys_temp["amb"]) / GG

                    fb, tb, wb, stdb = tools.spectral_binning_number_of_samples(
                        ff, TT, WW_all, nsamples=64
                    )
                    tcal[load] = {
                        "temp_cal": TT,
                        "f_binned": fb,
                        "t_binned": tb,
                        "w_binned": wb,
                        "std_binned": stdb,
                    }

                    RMS[k, i - min(wterms), j - min(cterms)] = np.std(
                        tb - phys_temp[load]
                    )

                    if panels < 5 and load == "simu":
                        # Don't plot the simulation if panels < 5
                        continue

                    # Save figure 1 (BINNED)
                    # ----------------------
                    ax[k].plot(fb, tb)
                    ax[k].plot(fb, phys_temp[load] * np.ones(len(fb)))
                    ax[k].set_ylabel(
                        r"T_{\rm %s}\nRMS=%s K"
                        % (load, round(np.std(tb - phys_temp[load]), 3))
                    )
                    ax[k].set_title(f"CTerms={j}, Wterms={i}")

                    ax_raw[k].plot(ff[WW_all > 0], TT[WW_all > 0])
                    ax_raw[k].plot(fb, phys_temp[load] * np.ones(len(fb)))
                    ax_raw[k].set_ylabel(
                        r"T_{\rm %s}\nRMS=%s K"
                        % (load, round(np.std(TT[WW_all > 0] - phys_temp[load]), 3))
                    )
                    ax_raw[k].set_title(f"CTerms={j}, Wterms={i}")

                if j == cterms_nominal and i == wterms_nominal:
                    # Save some nominal stuff.
                    # Saving results
                    if save_nominal:
                        # Array
                        save_array = np.array(
                            [
                                ff,
                                np.real(loads_full["LNA"]),
                                np.imag(loads_full["LNA"]),
                                C1,
                                C2,
                                TU,
                                TC,
                                TS,
                            ]
                        )

                        # Save
                        np.savetxt(
                            path_save
                            + "calibration_file_receiver1"
                            + save_nominal_flag
                            + ".txt",
                            save_array,
                            fmt="%1.8f",
                        )

                    if plot_nominal:
                        fig_nom, ax_nom = plt.subplots(
                            5, 1, sharex=True, figsize=(10, 10)
                        )
                        for k, (load, caltemp) in tcal.items():
                            rms = np.std(
                                caltemp["temp_cal"][WW_all > 0] - phys_temp[load]
                            )
                            ax_nom[k].plot(
                                ff[WW_all > 0],
                                caltemp["temp_cal"][WW_all > 0],
                                "g",
                                lw=1,
                            )
                            ax_nom[k].plot(fb, phys_temp[load] * np.ones(len(fb)), "k")
                            plt.ylabel(
                                rf"T$_{load[0].upper()}$ [K]\nRMS={rms:.3f} K",
                                fontsize=13,
                            )

                        # Creating folder if necessary
                        plt.savefig(
                            path_save
                            + f"calibration_crosscheck_{f_low}_{f_high}MHz_cterms{j}_wterms{i}.pdf",
                            bbox_inches="tight",
                        )
                # Creating folder if necessary
                path_save_term_sweep = (
                    f"{path_save}/binned_calibration_term_sweep_{f_low}_{f_high}MHz/"
                )
                if not exists(path_save_term_sweep):
                    makedirs(path_save_term_sweep)

                fig.savefig(
                    path_save_term_sweep
                    + f"calibration_term_sweep_{f_low}_{f_high}MHz_cterms{j}_wterms{i}.png",
                    bbox_inches="tight",
                )

                raw_path_save_term_sweep = (
                    f"{path_save}/raw_binned_calibration_term_sweep_{f_low}_"
                    f"{f_high}MHz/"
                )
                if not exists(path_save_term_sweep):
                    makedirs(path_save_term_sweep)

                fig_raw.savefig(
                    raw_path_save_term_sweep
                    + f"calibration_term_sweep_{f_low}_{f_high}MHz_cterms{j}_wterms{i}.png",
                    bbox_inches="tight",
                )

        # Save
        with h5py.File(
            path_save_term_sweep + f"calibration_term_sweep_{f_low}_{f_high}MHz.hdf5"
            "w",
        ) as hf:
            hf.create_dataset("RMS", data=RMS)
            hf.create_dataset("index_cterms", data=cterms)
            hf.create_dataset("index_wterms", data=wterms)

    return f, good_data, WW_all


def daily_integrations_and_residuals():
    f, pz, rz, wz, index, gha, ydz = io.level4read(
        edges_folder + "mid_band/spectra/level4" "/case_nominal/case_nominal.hdf5"
    )

    px = np.delete(pz, 1, axis=0)
    rx = np.delete(rz, 1, axis=0)
    wx = np.delete(wz, 1, axis=0)
    ydx = np.delete(ydz, 1, axis=0)

    # Average the data from the two days 147
    for i in range(len(gha) - 1):
        p147 = np.mean(pz[1:3, i, :], axis=0)
        r147, w147 = tools.spectral_averaging(rz[1:3, i, :], wz[1:3, i, :])

        px[1, i, :] = p147
        rx[1, i, :] = r147
        wx[1, i, :] = w147

    bad_days = np.array(
        [
            [2018, 159],
            [2018, 169],
            [2018, 184],
            [2018, 185],
            [2018, 191],
            [2018, 193],
            [2018, 195],
            [2018, 196],
            [2018, 204],
            [2018, 208],
            [2018, 216],
            [2018, 220],
        ]
    )  #

    #
    f_low = 57  # 60
    f_high = 120
    n_fg = 5
    Nsp = 1

    ll = len(px[:, 0, 0])
    j = -1

    for i in range(ll):

        if (ydx[i, 0] in bad_days[:, 0]) and (ydx[i, 1] in bad_days[:, 1]):  # print(i)

            print(ydx[i, :])

        else:
            j = j + 1

            mx = mdl.model_evaluate("LINLOG", px[i, 0, :], f / 200)
            tx = mx + rx[i, 0, :]

            fy = f[(f >= f_low) & (f <= f_high)]
            ty = tx[(f >= f_low) & (f <= f_high)]
            wy = wx[i, 0, (f >= f_low) & (f <= f_high)]

            p = mdl.fit_polynomial_fourier("LINLOG", fy / 200, ty, n_fg, Weights=wy)
            my = mdl.model_evaluate("LINLOG", p[0], fy / 200)
            ry = ty - my

            (
                fb,
                rb,
                wb,
                sb,
            ) = src.edges_analysis.analysis.tools.spectral_binning_number_of_samples(
                fy, ry, wy, nsamples=128
            )

            mb = mdl.model_evaluate("LINLOG", p[0], fb / 200)
            tb = mb + rb

            if j == 0:
                rb_all = np.zeros((ll - len(bad_days), len(fb)))
                tb_all = np.zeros((ll - len(bad_days), len(fb)))
                wb_all = np.zeros((ll - len(bad_days), len(fb)))
                sb_all = np.zeros((ll - len(bad_days), len(fb)))
                yd_all = np.zeros((ll - len(bad_days), 2))

            tb_all[j, :] = tb
            rb_all[j, :] = rb
            wb_all[j, :] = wb
            sb_all[j, :] = sb
            yd_all[j, :] = ydx[i, :]

    K = 0.5
    lb = int(np.floor(len(tb_all[:, 0]) / Nsp))
    plt.figure(figsize=[6.5, 10])
    for i in range(lb):

        rb_i, wb_i = tools.spectral_averaging(
            rb_all[(Nsp * i) : (Nsp * i + Nsp), :],
            wb_all[(Nsp * i) : (Nsp * i + Nsp), :],
        )

        if i < 26 / Nsp:
            cc = "b"
            lw = 2
        else:
            cc = "r"
            lw = 1
        plt.plot(fb, rb_i - K * i, color=cc, linewidth=lw)

        if Nsp == 1:
            plt.text(54, -K * i - 0.3 * K, str(int(yd_all[i, 1])))
        elif Nsp > 1:
            plt.text(
                52,
                -K * i - 0.3 * K,
                str(int(yd_all[Nsp * i, 1]))
                + "-"
                + str(int(yd_all[Nsp * i + Nsp - 1, 1])),
            )

    plt.ylim([-K * lb, K])
    plt.yticks([])
    plt.xlabel(r"$\nu$ [MHz]", fontsize=13)
    plt.xlim([52, 120])

    return fb, tb_all, rb_all, wb_all, sb_all, yd_all


def integrated_spectrum_level4(
    case,
    index_GHA,
    f_low,
    f_high,
    day_range,
    day_min1,
    day_max1,
    day_min2,
    day_max2,
    n_fg,
    rms_threshold,
    save,
    filename_flag,
):
    if case == 100:
        f, px, rx, wx, index, gha, ydx = io.level4read(
            edges_folder + "mid_band/spectra/level4/rcv18_sw18_nominal_GHA_6_18hr"
            "/rcv18_sw18_nominal_GHA_6_18hr.hdf5"
        )
    if case == 101:
        f, px, rx, wx, index, gha, ydx = io.level4read(
            edges_folder + "mid_band/spectra/level4/rcv18_sw18_nominal_GHA_every_1hr"
            "/rcv18_sw18_nominal_GHA_every_1hr.hdf5"
        )
        save_path = (
            edges_folder + "mid_band/spectra/level5/rcv18_sw18_nominal_GHA_every_1hr/"
        )
        save_spectrum = (
            "integrated_spectrum_rcv18_sw18_every_1hr_GHA" + filename_flag + ".txt"
        )

    if case == 21:
        f, px, rx, wx, index, gha, ydx = io.level4read(
            edges_folder
            + "mid_band/spectra/level4/rcv18_ant19_nominal/rcv18_ant19_nominal.hdf5"
        )

    if case == 22:

        f, px, rx, wx, index, gha, ydx = io.level4read(
            edges_folder + "mid_band/spectra/level4/rcv18_ant19_every_1hr_GHA"
            "/rcv18_ant19_every_1hr_GHA.hdf5"
        )

        save_path = (
            edges_folder + "mid_band/spectra/level5" "/rcv18_ant19_every_1hr_GHA/"
        )
        save_spectrum = (
            "integrated_spectrum_rcv18_ant19_every_1hr_GHA" + filename_flag + ".txt"
        )

    # Producing integrated spectrum
    # ------------------------------------------------

    if len(index_GHA) == 1:
        keep_index = filters.daily_nominal_filter("mid_band", case, index_GHA[0], ydx)

        p = px[
            (keep_index == 1)
            & (
                ((ydx[:, 1] >= day_min1) & (ydx[:, 1] <= day_max1))
                | ((ydx[:, 1] >= day_min2) & (ydx[:, 1] <= day_max2))
            ),
            index_GHA[0],
            :,
        ]
        r = rx[
            (keep_index == 1)
            & (
                ((ydx[:, 1] >= day_min1) & (ydx[:, 1] <= day_max1))
                | ((ydx[:, 1] >= day_min2) & (ydx[:, 1] <= day_max2))
            ),
            index_GHA[0],
            :,
        ]
        w = wx[
            (keep_index == 1)
            & (
                ((ydx[:, 1] >= day_min1) & (ydx[:, 1] <= day_max1))
                | ((ydx[:, 1] >= day_min2) & (ydx[:, 1] <= day_max2))
            ),
            index_GHA[0],
            :,
        ]
    elif len(index_GHA) > 1:
        for i in range(len(index_GHA)):
            keep_index = filters.daily_nominal_filter(
                "mid_band", case, index_GHA[i], ydx
            )

            p_i = px[
                (keep_index == 1)
                & (
                    ((ydx[:, 1] >= day_min1) & (ydx[:, 1] <= day_max1))
                    | ((ydx[:, 1] >= day_min2) & (ydx[:, 1] <= day_max2))
                ),
                index_GHA[i],
                :,
            ]
            r_i = rx[
                (keep_index == 1)
                & (
                    ((ydx[:, 1] >= day_min1) & (ydx[:, 1] <= day_max1))
                    | ((ydx[:, 1] >= day_min2) & (ydx[:, 1] <= day_max2))
                ),
                index_GHA[i],
                :,
            ]
            w_i = wx[
                (keep_index == 1)
                & (
                    ((ydx[:, 1] >= day_min1) & (ydx[:, 1] <= day_max1))
                    | ((ydx[:, 1] >= day_min2) & (ydx[:, 1] <= day_max2))
                ),
                index_GHA[i],
                :,
            ]

            if i == 0:
                p = np.copy(p_i)
                r = np.copy(r_i)
                w = np.copy(w_i)

            elif i > 0:
                p = np.vstack((p, p_i))
                r = np.vstack((r, r_i))
                w = np.vstack((w, w_i))

    print(p.shape)
    avp = np.mean(p, axis=0)
    m = mdl.model_evaluate("LINLOG", avp, f / 200)

    avr, avw = tools.spectral_averaging(r, w)
    rr, wr = rfi.cleaning_sweep(
        f, avr, avw, window_width=3, n_poly=2, n_bootstrap=20, n_sigma=3
    )

    tr = m + rr

    fr1 = 136
    fr2 = 139
    tr[(f >= fr1) & (f <= fr2)] = 0
    wr[(f >= fr1) & (f <= fr2)] = 0

    p = mdl.fit_polynomial_fourier("LINLOG", f / 200, tr, 7, Weights=wr)
    m = mdl.model_evaluate("LINLOG", p[0], f / 200)
    r = tr - m

    NS = 64
    fb, rb, wb, sb = tools.spectral_binning_number_of_samples(f, r, wr, nsamples=NS)

    mb = mdl.model_evaluate("LINLOG", p[0], fb / 200)
    tb = mb + rb
    tb[wb == 0] = 0
    sb[wb == 0] = 0

    outT = np.array([fb, tb, wb, sb])
    out = outT.T

    # Saving spectrum
    if save and (day_range != "daily"):
        np.savetxt(
            save_path + save_spectrum,
            out,
            header="freq [MHz], temp [K], weight [K], std dev [K]",
        )

    # Computing residuals for plot
    # ----------------------------------------
    fx = fb[(fb >= f_low) & (fb <= f_high)]
    tx = tb[(fb >= f_low) & (fb <= f_high)]
    wx = wb[(fb >= f_low) & (fb <= f_high)]
    sx = sb[(fb >= f_low) & (fb <= f_high)]

    ft = fx[wx > 0]
    tt = tx[wx > 0]
    # wt = wx[wx > 0]
    st = sx[wx > 0]

    pt = mdl.fit_polynomial_fourier(
        "LINLOG", ft / 200, tt, n_fg, Weights=(1 / (st ** 2))
    )
    mt = mdl.model_evaluate("LINLOG", pt[0], ft / 200)
    rt = tt - mt

    pl = np.polyfit(np.log(ft / 200), np.log(tt), n_fg - 1)
    log_ml = np.polyval(pl, np.log(ft / 200))
    ml = np.exp(log_ml)
    rl = tt - ml

    return ft, tt, st, rt, rl


def integrated_half_hour_level4(band, case, first_day, last_day, GHA_start=13.5):
    """
    Feb 6, 2019
    """

    if band == "mid_band":
        if case == 4:
            f, p_all, r_all, w_all, gha, yd = io.level4read(
                "/home/raul/DATA/EDGES/mid_band/spectra/level4/case4/case4.hdf5"
            )
        elif case == 41:
            f, p_all, r_all, w_all, gha, yd = io.level4read(
                "/home/raul/DATA/EDGES/mid_band/spectra/level4/case41/case41.hdf5"
            )
        else:
            raise ValueError("for mid_band, case must be 4 or 41")

        index = np.arange(0, len(gha))
        IX = int(index[gha == GHA_start])

        p = p_all[:, IX, :]
        r = r_all[:, IX, :]
        w = w_all[:, IX, :]

        day = yd[:, 1]
        flag = 0

        if GHA_start == 6.0:  # Looks good over 58-118 MHz
            discarded_days = [146, 164, 167, 169]
        if GHA_start == 6.5:  # Looks good over 58-118 MHz
            discarded_days = [146, 147, 174, 179, 181, 198, 211, 215]
        if GHA_start == 7.0:  # Looks good over 58-118 MHz
            discarded_days = [146, 147, 157, 166]
        if GHA_start == 7.5:  # Looks good up to 120 MHz
            discarded_days = [146, 159]
        if GHA_start == 8.0:
            discarded_days = [146, 151, 159]
        if GHA_start == 8.5:
            discarded_days = [146, 159]
        if GHA_start == 9.0:
            discarded_days = [146, 151, 152, 157, 159, 163, 185]
        if GHA_start == 9.5:
            discarded_days = [146, 157, 159, 167, 196]
            w[day == 149, (f > 104.5) & (f < 110)] = 0
            w[day == 150, (f > 104.5) & (f < 110)] = 0
            w[day == 152, (f > 104.5) & (f < 110)] = 0
            w[day == 163, (f > 104.5) & (f < 110)] = 0

            w[day == 150, (f > 129.0) & (f < 135)] = 0
            w[day == 160, (f > 129.0) & (f < 135)] = 0
            w[day == 161, (f > 129.0) & (f < 135)] = 0
            w[day == 162, (f > 129.0) & (f < 135)] = 0
            w[day == 166, (f > 129.0) & (f < 135)] = 0

            w[:, (f > 134.5) & (f < 140.5)] = 0

        if GHA_start == 10.0:
            discarded_days = [152, 157, 166, 159, 196]  # , 176, 180, 181, 185]

        if GHA_start == 10.5:
            discarded_days = [174, 176, 204, 218]  # , 151, 162, 189, 210
            w[:, (f > 101.52) & (f < 101.53)] = 0
            w[:, (f > 102.50) & (f < 102.53)] = 0
            w[:, (f > 153.02) & (f < 153.04)] = 0

            w[:, (f > 111.7) & (f < 115.4)] = 0
            w[:, (f > 121.47) & (f < 121.55)] = 0
            w[:, (f > 146.5) & (f < 148.0)] = 0
            w[:, (f > 150) & (f < 150.5)] = 0

            w[:, (f > 105.72) & (f < 105.74)] = 0
            w[:, (f > 106.05) & (f < 106.15)] = 0
            w[:, (f > 106.42) & (f < 106.55)] = 0

        if GHA_start == 11.0:
            discarded_days = [149, 165, 176, 204]

            # This is the right way of doing it!
            w[day == 151, (f > 129.0) & (f < 135.0)] = 0
            w[day == 161, (f > 129.0) & (f < 135.0)] = 0

            w[:, (f > 109) & (f < 114.2)] = 0
            w[:, (f > 105.72) & (f < 105.74)] = 0
            w[:, (f > 106.05) & (f < 106.15)] = 0
            w[:, (f > 106.42) & (f < 106.55)] = 0
            w[:, (f > 138) & (f < 138.4)] = 0

        if GHA_start == 11.5:
            discarded_days = [175, 176, 177, 200, 204, 216]
            w[day == 149, (f > 110.0) & (f < 118.0)] = 0
            w[:, (f > 137.0) & (f < 140.0)] = 0

        if GHA_start == 12.0:
            discarded_days = [146, 147, 170, 175, 176, 193, 195, 204, 205, 220]
        if GHA_start == 12.5:
            discarded_days = [146, 170, 176, 185, 195, 198, 204, 220]
        if GHA_start == 13.0:
            discarded_days = [152, 174, 176, 182, 185, 195, 204, 208, 214]
        if GHA_start == 13.5:
            discarded_days = [
                151,
                163,
                164,
                176,
                185,
                187,
                189,
                192,
                195,
                200,
                208,
                215,
                219,
            ]
        if GHA_start == 14:
            discarded_days = [176, 184, 185, 193, 199, 208, 210]
        if GHA_start == 14.5:
            discarded_days = [166, 174, 177, 185, 199, 200, 201, 208]
        if GHA_start == 15:
            discarded_days = [
                150,
                157,
                178,
                182,
                185,
                187,
                198,
                208,
            ]  # , 210, 215, 217, 218, 219]
        if GHA_start == 15.5:
            discarded_days = [185]
        if GHA_start == 16.0:
            discarded_days = [184]
        if GHA_start == 16.5:
            discarded_days = [191, 192]
        if GHA_start == 17.0:
            discarded_days = [184, 186, 192, 216]
        if GHA_start == 17.5:
            discarded_days = [186, 192, 196]
        if GHA_start == 18.0:
            discarded_days = [192, 197]
        if GHA_start == 18.5:
            discarded_days = [192, 209, 211]

    for i in range(len(p[:, 0])):
        if (day[i] >= first_day) & (day[i] <= last_day):
            if (np.sum(w[i, :]) > 0) and (day[i] not in discarded_days):

                print(day[i])

                t = mdl.model_evaluate("LINLOG", p[i, :], f / 200) + r[i, :]
                par = mdl.fit_polynomial_fourier(
                    "LINLOG", f / 200, t, 4, Weights=w[i, :]
                )
                r_raw = t - par[1]
                w_raw = w[i, :]

                fb, rb, wb = tools.spectral_binning_number_of_samples(f, r_raw, w_raw)

                if flag == 0:
                    rr_all = np.copy(r_raw)
                    wr_all = np.copy(w_raw)
                    pr_all = np.copy(par[0])

                    rb_all = np.copy(rb)
                    wb_all = np.copy(wb)
                    d_all = np.copy(day[i])
                    flag = 1

                elif flag > 0:
                    rr_all = np.vstack((rr_all, r_raw))
                    wr_all = np.vstack((wr_all, w_raw))
                    pr_all = np.vstack((pr_all, par[0]))

                    rb_all = np.vstack((rb_all, rb))
                    wb_all = np.vstack((wb_all, wb))
                    d_all = np.append(d_all, day[i])

    avrn, avwn = tools.spectral_averaging(rr_all, wr_all)
    avp = np.mean(pr_all, axis=0)

    # For the 10.5 and 11 averages, DO NOT USE THIS CLEANING. ONLY use the 2.5sigma filter in the
    # INTEGRATED 10.5-11.5 spectrum, AFTER integration
    avrn, avwn = rfi.cleaning_sweep(
        f, avrn, avwn, window_width=3, n_poly=2, n_bootstrap=20, n_sigma=3.0
    )

    avtn = mdl.model_evaluate("LINLOG", avp, f / 200) + avrn

    # avrn, avwn = rfi.cleaning_sweep(f, avrn, avwn, window_width=5, n_poly=2,
    # n_bootstrap=20, n_sigma=2)
    fb, rbn, wbn = tools.spectral_binning_number_of_samples(f, avrn, avwn)

    tbn = mdl.model_evaluate("LINLOG", avp, fb / 200) + rbn

    return fb, rb_all, wb_all, d_all, tbn, wbn, f, rr_all, wr_all, avrn, avwn, avp, avtn


def integrated_half_hour_level4_many(
    band, case, GHA_start=[13.5, 14.0], save=False, filename="test.txt"
):
    for i in range(len(GHA_start)):

        print("------------------------------- " + str(GHA_start[i]))
        (
            fb,
            rb_all,
            wb_all,
            d_all,
            tbn,
            wbn,
            f,
            rr_all,
            wr_all,
            avr,
            avw,
            avp,
            avt,
        ) = integrated_half_hour_level4(band, case, 140, 170, GHA_start=GHA_start[i])

        if i == 0:
            avr_all = np.copy(avr)
            avw_all = np.copy(avw)
            avp_all = np.copy(avp)

        else:
            avr_all = np.vstack((avr_all, avr))
            avw_all = np.vstack((avw_all, avw))
            avp_all = np.vstack((avp_all, avp))

    avr, avw = tools.spectral_averaging(avr_all, avw_all)
    avp = np.mean(avp_all, axis=0)
    avt = mdl.model_evaluate("LINLOG", avp, f / 200) + avr

    avrn, avwn = rfi.cleaning_sweep(
        f, avr, avw, window_width=3, n_poly=2, n_bootstrap=20, n_sigma=2.5
    )
    fb, rb, wbn = tools.spectral_binning_number_of_samples(f, avrn, avwn)
    tbn = mdl.model_evaluate("LINLOG", avp, fb / 200) + rb

    return f, avr, avw, avt, avp, fb, tbn, wbn


def season_integrated_spectra_GHA(
    band, case, new_gha_edges=np.arange(0, 25, 2), data_save_name_flag="2hr"
):
    case_str = "case{}".format(case)
    data_save_path = edges_folder + band + "/spectra/level5/" + case_str + "/"

    # Loading level4 data
    f, p_all, r_all, w_all, gha_edges, yd = io.level4read(
        edges_folder + band + "/spectra/level4/" + case_str + "/" + case_str + ".hdf5"
    )

    # Creating intermediate 1hr-average arrays
    pr_all = np.zeros((len(gha_edges) - 1, len(p_all[0, 0, :])))
    rr_all = np.zeros((len(gha_edges) - 1, len(f)))
    wr_all = np.zeros((len(gha_edges) - 1, len(f)))

    # Looping over every original GHA edges
    for j in range(len(gha_edges) - 1):

        # Looping over day
        counter = 0
        for i in range(len(yd)):  # range(38): range(4): #

            # Returns a 1 if the 1hr average tested is good quality, and a 0 if it is not
            keep = filters.one_hour_filter(band, case, yd[i, 0], yd[i, 1], gha_edges[j])
            print(yd[i, 1])
            print(gha_edges[j])

            # Index of good spectra
            if keep == 1:
                if counter == 0:
                    index_good = np.array([i])
                    counter = counter + 1

                elif counter > 0:
                    index_good = np.append(index_good, i)

        # Selecting good parameters and spectra
        pp = p_all[index_good, j, :]
        rr = r_all[index_good, j, :]
        ww = w_all[index_good, j, :]

        # Average parameters and spectra
        avp = np.mean(pp, axis=0)
        avr, avw = tools.weighted_mean(rr, ww)

        # RFI cleaning of 1-hr season average spectra
        avr_no_rfi, avw_no_rfi = rfi.cleaning_sweep(
            f, avr, avw, window_width=3, n_poly=2, n_bootstrap=20, n_sigma=2.5,
        )  # 3

        # Storing season 1hr-average spectra
        pr_all[j, :] = avp
        rr_all[j, :] = avr_no_rfi
        wr_all[j, :] = avw_no_rfi

        # Frequency binning
        fb, rb, wb = tools.spectral_binning_number_of_samples(f, avr_no_rfi, avw_no_rfi)
        mb = mdl.model_evaluate("LINLOG", avp, fb / 200)
        tb = mb + rb
        tb[wb == 0] = 0

        # Storing binned average spectra
        if j == 0:
            tb_all = np.zeros((len(gha_edges) - 1, len(fb)))
            wb_all = np.zeros((len(gha_edges) - 1, len(fb)))

        tb_all[j, :] = tb
        wb_all[j, :] = wb

    print("-------------------------------")

    # Averaging data within new GHA edges
    for j in range(len(new_gha_edges) - 1):
        new_gha_start = new_gha_edges[j]
        new_gha_end = new_gha_edges[j + 1]

        print(str(new_gha_start) + " " + str(new_gha_end))

        counter = 0
        for i in range(len(gha_edges) - 1):
            if new_gha_start < new_gha_end:
                if (gha_edges[i] >= new_gha_start) and (gha_edges[i] < new_gha_end):

                    print(gha_edges[i])
                    if counter == 0:
                        px_all = pr_all[i, :]
                        rx_all = rr_all[i, :]
                        wx_all = wr_all[i, :]
                        counter = counter + 1

                    elif counter > 0:
                        px_all = np.vstack((px_all, pr_all[i, :]))
                        rx_all = np.vstack((rx_all, rr_all[i, :]))
                        wx_all = np.vstack((wx_all, wr_all[i, :]))

            elif new_gha_start > new_gha_end:
                if (gha_edges[i] >= new_gha_start) or (gha_edges[i] < new_gha_end):
                    print(gha_edges[i])
                    if counter == 0:
                        px_all = pr_all[i, :]
                        rx_all = rr_all[i, :]
                        wx_all = wr_all[i, :]
                        counter = counter + 1

                    elif counter > 0:
                        px_all = np.vstack((px_all, pr_all[i, :]))
                        rx_all = np.vstack((rx_all, rr_all[i, :]))
                        wx_all = np.vstack((wx_all, wr_all[i, :]))

        if len(px_all.shape) == 1:
            avpx = np.copy(px_all)
            avrx = np.copy(rx_all)
            avwx = np.copy(wx_all)

        elif len(px_all.shape) == 2:
            avpx = np.mean(px_all, axis=0)
            avrx, avwx = tools.weighted_mean(rx_all, wx_all)

        avrx_no_rfi, avwx_no_rfi = rfi.cleaning_sweep(
            f, avrx, avwx, window_width=3, n_poly=2, n_bootstrap=20, n_sigma=2.5,
        )

        # Frequency binning
        fb, rbx, wbx = tools.spectral_binning_number_of_samples(
            f, avrx_no_rfi, avwx_no_rfi
        )
        modelx = mdl.model_evaluate("LINLOG", avpx, fb / 200)
        tbx = modelx + rbx
        tbx[wbx == 0] = 0

        # Storing binned average spectra
        if j == 0:
            tbx_all = np.zeros((len(new_gha_edges) - 1, len(fb)))
            wbx_all = np.zeros((len(new_gha_edges) - 1, len(fb)))

        tbx_all[j, :] = tbx
        wbx_all[j, :] = wbx

        # Saving data
        np.savetxt(
            data_save_path + case_str + "_frequency.txt", fb, header="Frequency [MHz]."
        )
        np.savetxt(
            data_save_path + case_str + "_1hr_gha_edges.txt",
            gha_edges,
            header="GHA edges of integrated spectra from 0hr to 23hr in steps of 1hr [hr].",
        )
        np.savetxt(
            data_save_path + case_str + "_1hr_temperature.txt",
            tb_all,
            header="Rows correspond to different GHAs from 0hr to 23hr in steps of 1hr. Columns correspond to frequency.",
        )
        np.savetxt(
            data_save_path + case_str + "_1hr_weights.txt",
            wb_all,
            header="Rows correspond to different GHAs from 0hr to 23hr in steps of 1hr. Columns correspond to frequency.",
        )
        np.savetxt(
            data_save_path + case_str + "_" + data_save_name_flag + "_gha_edges.txt",
            new_gha_edges,
            header="GHA edges of integrated spectra [hr].",
        )
        np.savetxt(
            data_save_path + case_str + "_" + data_save_name_flag + "_temperature.txt",
            tbx_all,
            header="Rows correspond to different GHAs. Columns correspond to frequency.",
        )
        np.savetxt(
            data_save_path + case_str + "_" + data_save_name_flag + "_weights.txt",
            wbx_all,
            header="Rows correspond to different GHAs. Columns correspond to frequency.",
        )

    return fb, tb_all, wb_all, tbx_all, wbx_all


def batch_low_band_level1_to_level2(set_number):
    # Original 10x10 m^2 ground plane
    if set_number == 1:
        level1_to_level2("low_band", "2015", "286_02")
        level1_to_level2("low_band", "2015", "287_00")
        level1_to_level2("low_band", "2015", "288_00")
        level1_to_level2("low_band", "2015", "289_00")

        level1_to_level2("low_band", "2015", "291_00")
        level1_to_level2("low_band", "2015", "292_00")
        level1_to_level2("low_band", "2015", "293_00")
        level1_to_level2("low_band", "2015", "294_00")
        level1_to_level2("low_band", "2015", "295_00")
        level1_to_level2("low_band", "2015", "296_00")
        level1_to_level2("low_band", "2015", "297_00")
        level1_to_level2("low_band", "2015", "298_00")
        level1_to_level2("low_band", "2015", "299_00")

        level1_to_level2("low_band", "2015", "300_00")
        level1_to_level2("low_band", "2015", "301_00")
        level1_to_level2("low_band", "2015", "302_00")
        level1_to_level2("low_band", "2015", "303_00")

        level1_to_level2("low_band", "2015", "310_18")
        level1_to_level2("low_band", "2015", "311_00")
        level1_to_level2("low_band", "2015", "312_00")
        level1_to_level2("low_band", "2015", "313_00")
        level1_to_level2("low_band", "2015", "314_00")
        level1_to_level2("low_band", "2015", "315_00")
        level1_to_level2("low_band", "2015", "316_00")
        level1_to_level2("low_band", "2015", "317_00")
        level1_to_level2("low_band", "2015", "318_00")
        level1_to_level2("low_band", "2015", "319_00")

        level1_to_level2("low_band", "2015", "320_00")
        level1_to_level2("low_band", "2015", "321_00")
        level1_to_level2("low_band", "2015", "322_00")
        level1_to_level2("low_band", "2015", "323_00")
        level1_to_level2("low_band", "2015", "324_00")
        level1_to_level2("low_band", "2015", "325_00")
        level1_to_level2("low_band", "2015", "326_00")
        level1_to_level2("low_band", "2015", "327_00")
        level1_to_level2("low_band", "2015", "328_00")
        level1_to_level2("low_band", "2015", "329_00")

        level1_to_level2("low_band", "2015", "330_00")
        level1_to_level2("low_band", "2015", "331_00")
        level1_to_level2("low_band", "2015", "332_00")
        level1_to_level2("low_band", "2015", "333_00")
        level1_to_level2("low_band", "2015", "334_00")
        level1_to_level2("low_band", "2015", "335_00")
        level1_to_level2("low_band", "2015", "336_00")
        level1_to_level2("low_band", "2015", "337_00")
        level1_to_level2("low_band", "2015", "338_00")
        level1_to_level2("low_band", "2015", "339_00")

        level1_to_level2("low_band", "2015", "340_00")
        level1_to_level2("low_band", "2015", "341_00")
        level1_to_level2("low_band", "2015", "342_00")
        level1_to_level2("low_band", "2015", "343_14")
        level1_to_level2("low_band", "2015", "344_00")
        level1_to_level2("low_band", "2015", "344_21")
        level1_to_level2("low_band", "2015", "345_00")
        level1_to_level2("low_band", "2015", "346_00")

        level1_to_level2("low_band", "2015", "347_00")
        level1_to_level2("low_band", "2015", "348_00")
        level1_to_level2("low_band", "2015", "349_00")

        level1_to_level2("low_band", "2015", "350_00")
        level1_to_level2("low_band", "2015", "351_00")
        level1_to_level2("low_band", "2015", "352_00")
        level1_to_level2("low_band", "2015", "353_00")
        level1_to_level2("low_band", "2015", "354_00")

        level1_to_level2("low_band", "2015", "362_00")
        level1_to_level2("low_band", "2015", "363_00")
        level1_to_level2("low_band", "2015", "364_00")
        level1_to_level2("low_band", "2015", "365_00")

        level1_to_level2("low_band", "2016", "001_00")
        level1_to_level2("low_band", "2016", "002_00")
        level1_to_level2("low_band", "2016", "003_00")
        level1_to_level2("low_band", "2016", "004_00")
        level1_to_level2("low_band", "2016", "005_00")
        level1_to_level2("low_band", "2016", "006_00")
        level1_to_level2("low_band", "2016", "007_00")
        level1_to_level2("low_band", "2016", "008_00")
        level1_to_level2("low_band", "2016", "009_00")

        level1_to_level2("low_band", "2016", "010_00")
        level1_to_level2("low_band", "2016", "011_00")
        level1_to_level2("low_band", "2016", "012_00")
        level1_to_level2("low_band", "2016", "013_00")
        level1_to_level2("low_band", "2016", "014_00")
        level1_to_level2("low_band", "2016", "015_00")
        level1_to_level2("low_band", "2016", "016_00")
        level1_to_level2("low_band", "2016", "017_00")
        level1_to_level2("low_band", "2016", "018_00")
        level1_to_level2("low_band", "2016", "019_00")

        level1_to_level2("low_band", "2016", "020_00")
        level1_to_level2("low_band", "2016", "028_00")
        level1_to_level2("low_band", "2016", "029_00")

        level1_to_level2("low_band", "2016", "030_00")
        level1_to_level2("low_band", "2016", "031_00")
        level1_to_level2("low_band", "2016", "032_00")
        level1_to_level2("low_band", "2016", "033_00")
        level1_to_level2("low_band", "2016", "034_00")
        level1_to_level2("low_band", "2016", "035_00")
        level1_to_level2("low_band", "2016", "036_00")
        level1_to_level2("low_band", "2016", "037_00")
        level1_to_level2("low_band", "2016", "038_00")
        level1_to_level2("low_band", "2016", "039_00")

        level1_to_level2("low_band", "2016", "040_00")
        level1_to_level2("low_band", "2016", "041_00")
        level1_to_level2("low_band", "2016", "042_00")
        level1_to_level2("low_band", "2016", "043_00")
        level1_to_level2("low_band", "2016", "044_00")
        level1_to_level2("low_band", "2016", "045_00")
        level1_to_level2("low_band", "2016", "046_00")
        level1_to_level2("low_band", "2016", "047_00")
        level1_to_level2("low_band", "2016", "048_00")
        level1_to_level2("low_band", "2016", "049_00")

        level1_to_level2("low_band", "2016", "050_00")
        level1_to_level2("low_band", "2016", "051_00")
        level1_to_level2("low_band", "2016", "052_00")
        level1_to_level2("low_band", "2016", "053_00")
        level1_to_level2("low_band", "2016", "055_21")
        level1_to_level2("low_band", "2016", "056_00")
        level1_to_level2("low_band", "2016", "057_00")
        level1_to_level2("low_band", "2016", "058_00")
        level1_to_level2("low_band", "2016", "059_00")

        level1_to_level2("low_band", "2016", "060_00")
        level1_to_level2("low_band", "2016", "061_00")
        level1_to_level2("low_band", "2016", "062_00")
        level1_to_level2("low_band", "2016", "063_00")
        level1_to_level2("low_band", "2016", "064_00")
        level1_to_level2("low_band", "2016", "065_00")
        level1_to_level2("low_band", "2016", "066_00")
        level1_to_level2("low_band", "2016", "067_00")
        level1_to_level2("low_band", "2016", "068_00")
        level1_to_level2("low_band", "2016", "069_00")

        level1_to_level2("low_band", "2016", "070_00")
        level1_to_level2("low_band", "2016", "071_00")
        level1_to_level2("low_band", "2016", "072_00")
        level1_to_level2("low_band", "2016", "073_00")
        level1_to_level2("low_band", "2016", "074_00")
        level1_to_level2("low_band", "2016", "075_00")
        level1_to_level2("low_band", "2016", "076_00")
        level1_to_level2("low_band", "2016", "077_00")
        level1_to_level2("low_band", "2016", "078_00")
        level1_to_level2("low_band", "2016", "079_00")

        level1_to_level2("low_band", "2016", "080_00")
        level1_to_level2("low_band", "2016", "081_00")
        level1_to_level2("low_band", "2016", "082_00")

        level1_to_level2("low_band", "2016", "083_00")
        level1_to_level2("low_band", "2016", "084_00")
        level1_to_level2("low_band", "2016", "085_00")
        level1_to_level2("low_band", "2016", "086_00")
        level1_to_level2("low_band", "2016", "087_00")
        level1_to_level2("low_band", "2016", "088_00")
        level1_to_level2("low_band", "2016", "089_00")

        level1_to_level2("low_band", "2016", "090_00")
        level1_to_level2("low_band", "2016", "091_00")
        level1_to_level2("low_band", "2016", "092_00")
        level1_to_level2("low_band", "2016", "093_00")
        level1_to_level2("low_band", "2016", "094_00")
        level1_to_level2("low_band", "2016", "095_00")
        level1_to_level2("low_band", "2016", "096_00")
        level1_to_level2("low_band", "2016", "097_00")
        level1_to_level2("low_band", "2016", "098_00")
        level1_to_level2("low_band", "2016", "099_00")
        level1_to_level2("low_band", "2016", "100_00")

        level1_to_level2("low_band", "2016", "101_00")
        level1_to_level2("low_band", "2016", "102_00")
        level1_to_level2("low_band", "2016", "103_00")
        level1_to_level2("low_band", "2016", "104_00")

        level1_to_level2("low_band", "2016", "106_13")
        level1_to_level2("low_band", "2016", "107_00")
        level1_to_level2("low_band", "2016", "108_00")
        level1_to_level2("low_band", "2016", "109_00")
        level1_to_level2("low_band", "2016", "110_00")

        level1_to_level2("low_band", "2016", "111_00")
        level1_to_level2("low_band", "2016", "112_00")
        level1_to_level2("low_band", "2016", "113_00")
        level1_to_level2("low_band", "2016", "114_00")
        level1_to_level2("low_band", "2016", "115_00")
        level1_to_level2("low_band", "2016", "116_00")
        level1_to_level2("low_band", "2016", "117_00")
        level1_to_level2("low_band", "2016", "118_00")

        level1_to_level2("low_band", "2016", "122_16")
        level1_to_level2("low_band", "2016", "123_00")
        level1_to_level2("low_band", "2016", "124_00")
        level1_to_level2("low_band", "2016", "125_00")
        level1_to_level2("low_band", "2016", "126_00")
        level1_to_level2("low_band", "2016", "127_00")

        level1_to_level2("low_band", "2016", "128_00")
        level1_to_level2("low_band", "2016", "129_00")

        level1_to_level2("low_band", "2016", "130_00")
        level1_to_level2("low_band", "2016", "131_00")
        level1_to_level2("low_band", "2016", "132_00")
        level1_to_level2("low_band", "2016", "133_00")
        level1_to_level2("low_band", "2016", "134_00")
        level1_to_level2("low_band", "2016", "135_00")
        level1_to_level2("low_band", "2016", "136_00")
        level1_to_level2("low_band", "2016", "137_00")
        level1_to_level2("low_band", "2016", "138_00")
        level1_to_level2("low_band", "2016", "139_00")

        level1_to_level2("low_band", "2016", "140_00")
        level1_to_level2("low_band", "2016", "141_00")
        level1_to_level2("low_band", "2016", "142_00")
        level1_to_level2("low_band", "2016", "143_00")
        level1_to_level2("low_band", "2016", "144_00")
        level1_to_level2("low_band", "2016", "145_00")
        level1_to_level2("low_band", "2016", "146_00")
        level1_to_level2("low_band", "2016", "147_00")
        level1_to_level2("low_band", "2016", "148_00")
        level1_to_level2("low_band", "2016", "149_00")

        level1_to_level2("low_band", "2016", "150_00")
        level1_to_level2("low_band", "2016", "151_00")
        level1_to_level2("low_band", "2016", "152_00")
        level1_to_level2("low_band", "2016", "153_00")
        level1_to_level2("low_band", "2016", "154_00")
        level1_to_level2("low_band", "2016", "155_00")
        level1_to_level2("low_band", "2016", "156_00")
        level1_to_level2("low_band", "2016", "157_00")
        level1_to_level2("low_band", "2016", "158_00")
        level1_to_level2("low_band", "2016", "159_00")

        level1_to_level2("low_band", "2016", "160_00")
        level1_to_level2("low_band", "2016", "167_00")
        level1_to_level2("low_band", "2016", "168_00")
        level1_to_level2("low_band", "2016", "169_00")

        level1_to_level2("low_band", "2016", "170_00")
        level1_to_level2("low_band", "2016", "171_00")
        level1_to_level2("low_band", "2016", "172_00")
        level1_to_level2("low_band", "2016", "173_00")

        level1_to_level2("low_band", "2016", "180_15")
        level1_to_level2("low_band", "2016", "181_00")
        level1_to_level2("low_band", "2016", "182_00")
        level1_to_level2("low_band", "2016", "183_00")
        level1_to_level2("low_band", "2016", "184_00")
        level1_to_level2("low_band", "2016", "185_00")
        level1_to_level2("low_band", "2016", "186_00")
        level1_to_level2("low_band", "2016", "187_00")
        level1_to_level2("low_band", "2016", "188_00")
        level1_to_level2("low_band", "2016", "189_00")

        level1_to_level2("low_band", "2016", "190_00")
        level1_to_level2("low_band", "2016", "191_00")
        level1_to_level2("low_band", "2016", "192_00")
        level1_to_level2("low_band", "2016", "193_00")
        level1_to_level2("low_band", "2016", "194_00")
        level1_to_level2("low_band", "2016", "195_00")
        level1_to_level2("low_band", "2016", "196_00")
        level1_to_level2("low_band", "2016", "197_00")
        level1_to_level2("low_band", "2016", "198_00")
        level1_to_level2("low_band", "2016", "199_00")

        level1_to_level2("low_band", "2016", "200_00")
        level1_to_level2("low_band", "2016", "201_00")
        level1_to_level2("low_band", "2016", "202_00")
        level1_to_level2("low_band", "2016", "203_00")
        level1_to_level2("low_band", "2016", "204_00")
        level1_to_level2("low_band", "2016", "210_14")
        level1_to_level2("low_band", "2016", "211_00")
        level1_to_level2("low_band", "2016", "212_00")

        level1_to_level2("low_band", "2016", "217_00")
        level1_to_level2("low_band", "2016", "218_00")
        level1_to_level2("low_band", "2016", "219_00")
        level1_to_level2("low_band", "2016", "220_00")

        level1_to_level2("low_band", "2016", "226_19")
        level1_to_level2("low_band", "2016", "227_00")
        level1_to_level2("low_band", "2016", "228_00")
        level1_to_level2("low_band", "2016", "229_00")

        level1_to_level2("low_band", "2016", "230_00")

        level1_to_level2("low_band", "2016", "238_00")

        level1_to_level2("low_band", "2016", "246_07")
        level1_to_level2("low_band", "2016", "247_00")
        level1_to_level2("low_band", "2016", "248_00")
        level1_to_level2("low_band", "2016", "249_00")

        level1_to_level2("low_band", "2016", "250_02")
        level1_to_level2("low_band", "2016", "251_00")
        level1_to_level2("low_band", "2016", "252_00")
        level1_to_level2("low_band", "2016", "253_13")
        level1_to_level2("low_band", "2016", "254_00")

    if set_number == 2:
        # Low Band with NEW GOOD SWITCH and EXTENDED GROUND PLANE
        # -------------------------------------------------------
        level1_to_level2("low_band", "2016", "258_13")
        level1_to_level2("low_band", "2016", "259_00")

        level1_to_level2("low_band", "2016", "260_00")
        level1_to_level2("low_band", "2016", "261_00")
        level1_to_level2("low_band", "2016", "262_00")
        level1_to_level2("low_band", "2016", "263_00")
        level1_to_level2("low_band", "2016", "264_00")
        level1_to_level2("low_band", "2016", "265_00")
        level1_to_level2("low_band", "2016", "266_00")
        level1_to_level2("low_band", "2016", "267_00")
        level1_to_level2("low_band", "2016", "268_00")
        level1_to_level2("low_band", "2016", "269_00")

        level1_to_level2("low_band", "2016", "270_00")
        level1_to_level2("low_band", "2016", "271_00")
        level1_to_level2("low_band", "2016", "273_15")
        level1_to_level2("low_band", "2016", "274_00")
        level1_to_level2("low_band", "2016", "275_00")
        level1_to_level2("low_band", "2016", "276_00")
        level1_to_level2("low_band", "2016", "277_00")
        level1_to_level2("low_band", "2016", "278_00")
        level1_to_level2("low_band", "2016", "279_00")

        level1_to_level2("low_band", "2016", "280_00")
        level1_to_level2("low_band", "2016", "281_00")
        level1_to_level2("low_band", "2016", "282_00")
        level1_to_level2("low_band", "2016", "283_00")
        level1_to_level2("low_band", "2016", "284_00")
        level1_to_level2("low_band", "2016", "285_00")
        level1_to_level2("low_band", "2016", "286_00")
        level1_to_level2("low_band", "2016", "287_00")
        level1_to_level2("low_band", "2016", "288_00")
        level1_to_level2("low_band", "2016", "289_00")

        level1_to_level2("low_band", "2016", "290_00")
        level1_to_level2("low_band", "2016", "291_00")
        level1_to_level2("low_band", "2016", "292_00")
        level1_to_level2("low_band", "2016", "293_00")
        level1_to_level2("low_band", "2016", "294_00")
        level1_to_level2("low_band", "2016", "295_00")
        level1_to_level2("low_band", "2016", "296_00")
        level1_to_level2("low_band", "2016", "297_00")
        level1_to_level2("low_band", "2016", "298_00")
        level1_to_level2("low_band", "2016", "299_00")

        level1_to_level2("low_band", "2016", "302_14")
        level1_to_level2("low_band", "2016", "303_00")
        level1_to_level2("low_band", "2016", "304_00")
        level1_to_level2("low_band", "2016", "305_00")

        level1_to_level2("low_band", "2016", "314_15")
        level1_to_level2("low_band", "2016", "315_00")
        level1_to_level2("low_band", "2016", "316_00")
        level1_to_level2("low_band", "2016", "317_00")
        level1_to_level2("low_band", "2016", "318_00")
        level1_to_level2("low_band", "2016", "319_00")

        level1_to_level2("low_band", "2016", "320_00")
        level1_to_level2("low_band", "2016", "321_00")
        level1_to_level2("low_band", "2016", "322_00")
        level1_to_level2("low_band", "2016", "323_00")
        level1_to_level2("low_band", "2016", "324_00")
        level1_to_level2("low_band", "2016", "325_00")
        level1_to_level2("low_band", "2016", "326_00")
        level1_to_level2("low_band", "2016", "327_00")
        level1_to_level2("low_band", "2016", "328_00")
        level1_to_level2("low_band", "2016", "329_00")

        level1_to_level2("low_band", "2016", "330_00")
        level1_to_level2("low_band", "2016", "331_00")
        level1_to_level2("low_band", "2016", "332_00")
        level1_to_level2("low_band", "2016", "333_00")
        level1_to_level2("low_band", "2016", "334_00")
        level1_to_level2("low_band", "2016", "335_00")
        level1_to_level2("low_band", "2016", "336_00")
        level1_to_level2("low_band", "2016", "337_00")
        level1_to_level2("low_band", "2016", "338_00")
        level1_to_level2("low_band", "2016", "339_00")

        level1_to_level2("low_band", "2016", "340_00")
        level1_to_level2("low_band", "2016", "341_00")
        level1_to_level2("low_band", "2016", "342_00")
        level1_to_level2("low_band", "2016", "343_00")
        level1_to_level2("low_band", "2016", "344_00")
        level1_to_level2("low_band", "2016", "345_00")
        level1_to_level2("low_band", "2016", "346_00")
        level1_to_level2("low_band", "2016", "347_00")
        level1_to_level2("low_band", "2016", "348_00")
        level1_to_level2("low_band", "2016", "349_00")

        level1_to_level2("low_band", "2016", "350_00")
        level1_to_level2("low_band", "2016", "351_00")
        level1_to_level2("low_band", "2016", "352_00")
        level1_to_level2("low_band", "2016", "353_00")
        level1_to_level2("low_band", "2016", "354_00")
        level1_to_level2("low_band", "2016", "355_00")
        level1_to_level2("low_band", "2016", "356_00")
        level1_to_level2("low_band", "2016", "356_06")
        level1_to_level2("low_band", "2016", "357_00")
        level1_to_level2("low_band", "2016", "357_07")
        level1_to_level2("low_band", "2016", "358_00")
        level1_to_level2("low_band", "2016", "359_00")

        level1_to_level2("low_band", "2016", "360_00")
        level1_to_level2("low_band", "2016", "361_00")
        level1_to_level2("low_band", "2016", "362_00")
        level1_to_level2("low_band", "2016", "363_00")
        level1_to_level2("low_band", "2016", "364_00")
        level1_to_level2("low_band", "2016", "365_00")
        level1_to_level2("low_band", "2016", "366_00")

        level1_to_level2("low_band", "2017", "001_15")
        level1_to_level2("low_band", "2017", "002_00")
        level1_to_level2("low_band", "2017", "003_00")
        level1_to_level2("low_band", "2017", "005_00")
        level1_to_level2("low_band", "2017", "006_00")
        level1_to_level2("low_band", "2017", "007_00")
        level1_to_level2("low_band", "2017", "008_00")
        level1_to_level2("low_band", "2017", "009_00")

        level1_to_level2("low_band", "2017", "010_00")
        level1_to_level2("low_band", "2017", "011_07")
        level1_to_level2("low_band", "2017", "012_00")
        level1_to_level2("low_band", "2017", "013_00")
        level1_to_level2("low_band", "2017", "014_00")
        level1_to_level2("low_band", "2017", "015_00")
        level1_to_level2("low_band", "2017", "016_00")
        level1_to_level2("low_band", "2017", "017_00")
        level1_to_level2("low_band", "2017", "018_00")
        level1_to_level2("low_band", "2017", "019_00")

        level1_to_level2("low_band", "2017", "023_00")

        level1_to_level2("low_band", "2017", "077_07")
        level1_to_level2("low_band", "2017", "078_00")
        level1_to_level2("low_band", "2017", "079_00")

        level1_to_level2("low_band", "2017", "080_00")
        level1_to_level2("low_band", "2017", "081_00")
        level1_to_level2("low_band", "2017", "081_12")
        level1_to_level2("low_band", "2017", "082_00")
        level1_to_level2("low_band", "2017", "082_08")
        level1_to_level2("low_band", "2017", "083_00")
        level1_to_level2("low_band", "2017", "084_00")
        level1_to_level2("low_band", "2017", "085_00")
        level1_to_level2("low_band", "2017", "086_00")
        level1_to_level2("low_band", "2017", "087_00")
        level1_to_level2("low_band", "2017", "087_21")
        level1_to_level2("low_band", "2017", "088_00")
        level1_to_level2("low_band", "2017", "089_00")

        level1_to_level2("low_band", "2017", "090_00")
        level1_to_level2("low_band", "2017", "091_00")

        level1_to_level2("low_band", "2017", "092_00")
        level1_to_level2("low_band", "2017", "093_00")
        level1_to_level2("low_band", "2017", "093_17")
        level1_to_level2("low_band", "2017", "094_00")
        level1_to_level2("low_band", "2017", "095_00")
        level1_to_level2("low_band", "2017", "095_15")

        level1_to_level2("low_band", "2017", "153_12")
        level1_to_level2("low_band", "2017", "154_00")
        level1_to_level2("low_band", "2017", "155_00")
        level1_to_level2("low_band", "2017", "156_00")

        level1_to_level2("low_band", "2017", "157_00")
        level1_to_level2("low_band", "2017", "158_03")
        level1_to_level2("low_band", "2017", "159_00")
        level1_to_level2("low_band", "2017", "160_00")
        level1_to_level2("low_band", "2017", "161_00")

        level1_to_level2("low_band", "2017", "162_00")
        level1_to_level2("low_band", "2017", "163_00")
        level1_to_level2("low_band", "2017", "164_00")
        level1_to_level2("low_band", "2017", "165_00")
        level1_to_level2("low_band", "2017", "166_00")
        level1_to_level2("low_band", "2017", "167_00")

        level1_to_level2("low_band", "2017", "168_00")
        level1_to_level2("low_band", "2017", "169_00")
        level1_to_level2("low_band", "2017", "170_00")
        level1_to_level2("low_band", "2017", "171_00")


def batch_low_band2_level1_to_level2(set_number):
    # NS with shield
    if set_number == 1:
        level1_to_level2("low_band2", "2017", "082_03", low2_flag="")
        level1_to_level2("low_band2", "2017", "082_08", low2_flag="")
        level1_to_level2("low_band2", "2017", "083_00", low2_flag="")
        level1_to_level2("low_band2", "2017", "084_00")
        level1_to_level2("low_band2", "2017", "085_00")
        level1_to_level2("low_band2", "2017", "086_00")
        level1_to_level2("low_band2", "2017", "086_14")
        level1_to_level2("low_band2", "2017", "087_00")
        level1_to_level2("low_band2", "2017", "087_21")
        level1_to_level2("low_band2", "2017", "088_00")
        level1_to_level2("low_band2", "2017", "089_00")
        level1_to_level2("low_band2", "2017", "090_00")
        level1_to_level2("low_band2", "2017", "091_00")
        level1_to_level2("low_band2", "2017", "092_00")
        level1_to_level2("low_band2", "2017", "093_00")
        level1_to_level2("low_band2", "2017", "093_17")
        level1_to_level2("low_band2", "2017", "094_00")
        level1_to_level2("low_band2", "2017", "095_00")
        level1_to_level2("low_band2", "2017", "096_00")
        level1_to_level2("low_band2", "2017", "097_00")
        level1_to_level2("low_band2", "2017", "098_00")
        level1_to_level2("low_band2", "2017", "099_00")
        level1_to_level2("low_band2", "2017", "100_00")
        level1_to_level2("low_band2", "2017", "101_00")
        level1_to_level2("low_band2", "2017", "102_00")
        level1_to_level2("low_band2", "2017", "102_15")
        level1_to_level2("low_band2", "2017", "103_00")
        level1_to_level2("low_band2", "2017", "103_15")
        level1_to_level2("low_band2", "2017", "104_00")
        level1_to_level2("low_band2", "2017", "105_00")
        level1_to_level2("low_band2", "2017", "106_00")
        level1_to_level2("low_band2", "2017", "107_00")
        level1_to_level2("low_band2", "2017", "108_00")
        level1_to_level2("low_band2", "2017", "109_00")
        level1_to_level2("low_band2", "2017", "110_00")
        level1_to_level2("low_band2", "2017", "111_00")
        level1_to_level2("low_band2", "2017", "112_00")
        level1_to_level2("low_band2", "2017", "113_00")
        level1_to_level2("low_band2", "2017", "114_00")
        level1_to_level2("low_band2", "2017", "115_00")
        level1_to_level2("low_band2", "2017", "116_00")
        level1_to_level2("low_band2", "2017", "117_00")
        level1_to_level2("low_band2", "2017", "117_16")
        level1_to_level2("low_band2", "2017", "118_00")
        level1_to_level2("low_band2", "2017", "119_00")
        level1_to_level2("low_band2", "2017", "120_00")
        level1_to_level2("low_band2", "2017", "121_00")
        level1_to_level2("low_band2", "2017", "122_00")
        level1_to_level2("low_band2", "2017", "123_00")
        level1_to_level2("low_band2", "2017", "124_00")
        level1_to_level2("low_band2", "2017", "125_00")
        level1_to_level2("low_band2", "2017", "126_00")
        level1_to_level2("low_band2", "2017", "127_00")
        level1_to_level2("low_band2", "2017", "128_00")
        level1_to_level2("low_band2", "2017", "129_00")
        level1_to_level2("low_band2", "2017", "130_00")
        level1_to_level2("low_band2", "2017", "131_00")
        level1_to_level2("low_band2", "2017", "132_00")
        level1_to_level2("low_band2", "2017", "133_00")
        level1_to_level2("low_band2", "2017", "134_00")
        level1_to_level2("low_band2", "2017", "135_00")
        level1_to_level2("low_band2", "2017", "136_00")
        level1_to_level2("low_band2", "2017", "137_00")
        level1_to_level2("low_band2", "2017", "138_00")
        level1_to_level2("low_band2", "2017", "139_00")
        level1_to_level2("low_band2", "2017", "140_00")
        level1_to_level2("low_band2", "2017", "141_00")
        level1_to_level2("low_band2", "2017", "142_00")

    # Rotation of antenna to EW
    if set_number == 2:
        level1_to_level2("low_band2", "2017", "154_00")
        level1_to_level2("low_band2", "2017", "155_00")
        level1_to_level2("low_band2", "2017", "156_00")
        level1_to_level2("low_band2", "2017", "157_01")
        level1_to_level2("low_band2", "2017", "158_03")
        level1_to_level2("low_band2", "2017", "159_00")
        level1_to_level2("low_band2", "2017", "160_00")
        level1_to_level2("low_band2", "2017", "161_00")
        level1_to_level2("low_band2", "2017", "162_00")
        level1_to_level2("low_band2", "2017", "163_00")
        level1_to_level2("low_band2", "2017", "164_00")
        level1_to_level2("low_band2", "2017", "165_00")
        level1_to_level2("low_band2", "2017", "166_00")
        level1_to_level2("low_band2", "2017", "167_00")
        level1_to_level2("low_band2", "2017", "168_00")
        level1_to_level2("low_band2", "2017", "169_00")
        level1_to_level2("low_band2", "2017", "170_00")
        level1_to_level2("low_band2", "2017", "171_00")

    # Removing the Balun Shield
    if set_number == 3:
        level1_to_level2("low_band2", "2017", "181_00")
        level1_to_level2("low_band2", "2017", "182_00")
        level1_to_level2("low_band2", "2017", "183_00")
        level1_to_level2("low_band2", "2017", "184_00")
        level1_to_level2("low_band2", "2017", "184_17")
        level1_to_level2("low_band2", "2017", "185_00")
        level1_to_level2("low_band2", "2017", "186_00")
        level1_to_level2("low_band2", "2017", "187_00")
        level1_to_level2("low_band2", "2017", "188_00")
        level1_to_level2("low_band2", "2017", "189_00")
        level1_to_level2("low_band2", "2017", "190_00")
        level1_to_level2("low_band2", "2017", "191_00")
        level1_to_level2("low_band2", "2017", "192_00")
        level1_to_level2("low_band2", "2017", "193_00")
        level1_to_level2("low_band2", "2017", "194_00")
        level1_to_level2("low_band2", "2017", "195_00")
        level1_to_level2("low_band2", "2017", "196_00")
        level1_to_level2("low_band2", "2017", "197_00")
        level1_to_level2("low_band2", "2017", "198_00")
        level1_to_level2("low_band2", "2017", "199_00")
        level1_to_level2("low_band2", "2017", "200_00")
        level1_to_level2("low_band2", "2017", "201_00")
        level1_to_level2("low_band2", "2017", "202_00")
        level1_to_level2("low_band2", "2017", "203_00")
        level1_to_level2("low_band2", "2017", "204_00")
        level1_to_level2("low_band2", "2017", "205_00")
        level1_to_level2("low_band2", "2017", "206_00")
        level1_to_level2("low_band2", "2017", "207_00")
        level1_to_level2("low_band2", "2017", "208_00")
        level1_to_level2("low_band2", "2017", "209_00")
        level1_to_level2("low_band2", "2017", "210_00")
        level1_to_level2("low_band2", "2017", "211_00")
        level1_to_level2("low_band2", "2017", "212_00")
        level1_to_level2("low_band2", "2017", "213_00")
        level1_to_level2("low_band2", "2017", "214_00")
        level1_to_level2("low_band2", "2017", "215_00")
        level1_to_level2("low_band2", "2017", "216_00")
        level1_to_level2("low_band2", "2017", "217_00")
        level1_to_level2("low_band2", "2017", "218_16")
        level1_to_level2("low_band2", "2017", "219_00")
        level1_to_level2("low_band2", "2017", "220_00")
        level1_to_level2("low_band2", "2017", "221_00")
        level1_to_level2("low_band2", "2017", "222_00")
        level1_to_level2("low_band2", "2017", "223_00")
        level1_to_level2("low_band2", "2017", "224_00")
        level1_to_level2("low_band2", "2017", "225_00")
        level1_to_level2("low_band2", "2017", "226_00")
        level1_to_level2("low_band2", "2017", "227_00")
        level1_to_level2("low_band2", "2017", "228_00")
        level1_to_level2("low_band2", "2017", "229_00")
        level1_to_level2("low_band2", "2017", "230_00")
        level1_to_level2("low_band2", "2017", "231_00")
        level1_to_level2("low_band2", "2017", "232_00")
        level1_to_level2("low_band2", "2017", "233_00")
        level1_to_level2("low_band2", "2017", "234_00")
        level1_to_level2("low_band2", "2017", "235_00")
        level1_to_level2("low_band2", "2017", "236_00")
        level1_to_level2("low_band2", "2017", "237_00")
        level1_to_level2("low_band2", "2017", "238_00")
        level1_to_level2("low_band2", "2017", "239_00")
        level1_to_level2("low_band2", "2017", "240_00")
        level1_to_level2("low_band2", "2017", "241_00")
        level1_to_level2("low_band2", "2017", "242_00")
        level1_to_level2("low_band2", "2017", "243_00")
        level1_to_level2("low_band2", "2017", "244_00")
        level1_to_level2("low_band2", "2017", "245_00")
        level1_to_level2("low_band2", "2017", "246_00")
        level1_to_level2("low_band2", "2017", "247_00")
        level1_to_level2("low_band2", "2017", "248_00")
        level1_to_level2("low_band2", "2017", "249_00")
        level1_to_level2("low_band2", "2017", "250_00")
        level1_to_level2("low_band2", "2017", "251_00")
        level1_to_level2("low_band2", "2017", "252_00")
        level1_to_level2("low_band2", "2017", "253_00")
        level1_to_level2("low_band2", "2017", "254_00")
        level1_to_level2("low_band2", "2017", "255_00")
        level1_to_level2("low_band2", "2017", "256_00")
        level1_to_level2("low_band2", "2017", "257_00")
        level1_to_level2("low_band2", "2017", "258_00")
        level1_to_level2("low_band2", "2017", "259_00")
        level1_to_level2("low_band2", "2017", "260_00")
        level1_to_level2("low_band2", "2017", "261_00")
        level1_to_level2("low_band2", "2017", "262_00")
        level1_to_level2("low_band2", "2017", "263_00")
        level1_to_level2("low_band2", "2017", "264_00")
        level1_to_level2("low_band2", "2017", "265_00")
        level1_to_level2("low_band2", "2017", "266_00")
        level1_to_level2("low_band2", "2017", "267_00")
        level1_to_level2("low_band2", "2017", "268_00")
        level1_to_level2("low_band2", "2017", "269_00")
        level1_to_level2("low_band2", "2017", "270_00")
        level1_to_level2("low_band2", "2017", "271_00")
        level1_to_level2("low_band2", "2017", "272_00")
        level1_to_level2("low_band2", "2017", "273_00")
        level1_to_level2("low_band2", "2017", "274_00")
        level1_to_level2("low_band2", "2017", "275_00")
        level1_to_level2("low_band2", "2017", "276_00")
        level1_to_level2("low_band2", "2017", "277_00")
        level1_to_level2("low_band2", "2017", "278_00")
        level1_to_level2("low_band2", "2017", "279_00")
        level1_to_level2("low_band2", "2017", "280_00")
        level1_to_level2("low_band2", "2017", "281_00")
        level1_to_level2("low_band2", "2017", "282_00")
        level1_to_level2("low_band2", "2017", "283_00")
        level1_to_level2("low_band2", "2017", "284_00")
        level1_to_level2("low_band2", "2017", "285_00")
        level1_to_level2("low_band2", "2017", "286_00")
        level1_to_level2("low_band2", "2017", "287_00")
        level1_to_level2("low_band2", "2017", "288_00")
        level1_to_level2("low_band2", "2017", "289_00")
        level1_to_level2("low_band2", "2017", "290_00")
        level1_to_level2("low_band2", "2017", "291_00")
        level1_to_level2("low_band2", "2017", "291_21")
        level1_to_level2("low_band2", "2017", "292_00")
        level1_to_level2("low_band2", "2017", "293_00")
        level1_to_level2("low_band2", "2017", "294_00")
        level1_to_level2("low_band2", "2017", "295_00")
        level1_to_level2("low_band2", "2017", "296_00")
        level1_to_level2("low_band2", "2017", "297_00")
        level1_to_level2("low_band2", "2017", "298_00")
        level1_to_level2("low_band2", "2017", "300_18")
        level1_to_level2("low_band2", "2017", "301_00")
        level1_to_level2("low_band2", "2017", "302_00")
        level1_to_level2("low_band2", "2017", "303_00")
        level1_to_level2("low_band2", "2017", "310_04")
        level1_to_level2("low_band2", "2017", "311_00")
        level1_to_level2("low_band2", "2017", "312_00")
        level1_to_level2("low_band2", "2017", "313_00")
        level1_to_level2("low_band2", "2017", "314_19")
        level1_to_level2("low_band2", "2017", "315_00")
        level1_to_level2("low_band2", "2017", "316_00")
        level1_to_level2("low_band2", "2017", "317_00")
        level1_to_level2("low_band2", "2017", "332_04")
        level1_to_level2("low_band2", "2017", "333_00")
        level1_to_level2("low_band2", "2017", "334_00")
        level1_to_level2("low_band2", "2017", "335_00")
        level1_to_level2("low_band2", "2017", "336_00")
        level1_to_level2("low_band2", "2017", "337_20")
        level1_to_level2("low_band2", "2017", "338_00")
        level1_to_level2("low_band2", "2017", "339_03")
        level1_to_level2("low_band2", "2017", "340_00")
        level1_to_level2("low_band2", "2017", "341_00")
        level1_to_level2("low_band2", "2017", "342_00")
        level1_to_level2("low_band2", "2017", "343_00")
        level1_to_level2("low_band2", "2017", "344_00")
        level1_to_level2("low_band2", "2017", "345_00")
        level1_to_level2("low_band2", "2017", "346_00")
        level1_to_level2("low_band2", "2017", "347_00")
        level1_to_level2("low_band2", "2017", "348_00")
        level1_to_level2("low_band2", "2017", "349_00")
        level1_to_level2("low_band2", "2017", "350_00")
        level1_to_level2("low_band2", "2017", "351_00")
        level1_to_level2("low_band2", "2017", "352_00")
        level1_to_level2("low_band2", "2017", "353_00")
        level1_to_level2("low_band2", "2017", "354_00")
        level1_to_level2("low_band2", "2017", "355_00")
        level1_to_level2("low_band2", "2017", "356_00")
        level1_to_level2("low_band2", "2017", "357_00")
        level1_to_level2("low_band2", "2017", "358_00")
        level1_to_level2("low_band2", "2017", "359_00")
        level1_to_level2("low_band2", "2017", "360_00")
        level1_to_level2("low_band2", "2017", "361_00")
        level1_to_level2("low_band2", "2017", "362_00")
        level1_to_level2("low_band2", "2017", "363_00")
        level1_to_level2("low_band2", "2017", "364_00")
        level1_to_level2("low_band2", "2017", "365_00")

    if set_number == 4:
        level1_to_level2("low_band2", "2018", "001_00")
        level1_to_level2("low_band2", "2018", "002_00")
        level1_to_level2("low_band2", "2018", "003_00")
        level1_to_level2("low_band2", "2018", "004_00")
        level1_to_level2("low_band2", "2018", "005_00")
        level1_to_level2("low_band2", "2018", "006_00")
        level1_to_level2("low_band2", "2018", "007_00")
        level1_to_level2("low_band2", "2018", "008_00")
        level1_to_level2("low_band2", "2018", "009_00")
        level1_to_level2("low_band2", "2018", "010_00")
        level1_to_level2("low_band2", "2018", "011_00")
        level1_to_level2("low_band2", "2018", "012_00")
        level1_to_level2("low_band2", "2018", "013_00")
        level1_to_level2("low_band2", "2018", "014_00")
        level1_to_level2("low_band2", "2018", "015_00")
        level1_to_level2("low_band2", "2018", "016_00")
        level1_to_level2("low_band2", "2018", "017_00")
        level1_to_level2("low_band2", "2018", "018_00")
        level1_to_level2("low_band2", "2018", "019_00")
        level1_to_level2("low_band2", "2018", "020_00")
        level1_to_level2("low_band2", "2018", "021_00")
        level1_to_level2("low_band2", "2018", "022_00")
        level1_to_level2("low_band2", "2018", "023_00")
        level1_to_level2("low_band2", "2018", "024_00")
        level1_to_level2("low_band2", "2018", "025_00")
        level1_to_level2("low_band2", "2018", "026_00")
        level1_to_level2("low_band2", "2018", "027_00")
        level1_to_level2("low_band2", "2018", "028_00")
        level1_to_level2("low_band2", "2018", "029_00")
        level1_to_level2("low_band2", "2018", "030_00")
        level1_to_level2("low_band2", "2018", "031_00")
        level1_to_level2("low_band2", "2018", "032_00")
        level1_to_level2("low_band2", "2018", "033_00")
        level1_to_level2("low_band2", "2018", "034_00")
        level1_to_level2("low_band2", "2018", "035_00")
        level1_to_level2("low_band2", "2018", "036_00")
        level1_to_level2("low_band2", "2018", "037_00")
        level1_to_level2("low_band2", "2018", "038_00")
        level1_to_level2("low_band2", "2018", "040_00")
        level1_to_level2("low_band2", "2018", "041_00")
        level1_to_level2("low_band2", "2018", "042_00")
        level1_to_level2("low_band2", "2018", "043_00")
        level1_to_level2("low_band2", "2018", "044_00")
        level1_to_level2("low_band2", "2018", "045_00")
        level1_to_level2("low_band2", "2018", "046_00")
        level1_to_level2("low_band2", "2018", "047_00")
        level1_to_level2("low_band2", "2018", "048_00")
        level1_to_level2("low_band2", "2018", "049_00")
        level1_to_level2("low_band2", "2018", "050_00")
        level1_to_level2("low_band2", "2018", "051_00")
        level1_to_level2("low_band2", "2018", "052_00")
        level1_to_level2("low_band2", "2018", "053_00")
        level1_to_level2("low_band2", "2018", "054_00")
        level1_to_level2("low_band2", "2018", "060_17")
        level1_to_level2("low_band2", "2018", "061_00")
        level1_to_level2("low_band2", "2018", "062_14")
        level1_to_level2("low_band2", "2018", "063_00")
        level1_to_level2("low_band2", "2018", "064_00")
        level1_to_level2("low_band2", "2018", "065_00")
        level1_to_level2("low_band2", "2018", "066_00")
        level1_to_level2("low_band2", "2018", "067_00")
        level1_to_level2("low_band2", "2018", "068_00")
        level1_to_level2("low_band2", "2018", "073_19")
        level1_to_level2("low_band2", "2018", "074_00")
        level1_to_level2("low_band2", "2018", "079_04")
        level1_to_level2("low_band2", "2018", "080_00")
        level1_to_level2("low_band2", "2018", "084_00")
        level1_to_level2("low_band2", "2018", "085_00")
        level1_to_level2("low_band2", "2018", "085_06")
        level1_to_level2("low_band2", "2018", "086_00")
        level1_to_level2("low_band2", "2018", "087_00")
        level1_to_level2("low_band2", "2018", "088_00")
        level1_to_level2("low_band2", "2018", "089_00")
        level1_to_level2("low_band2", "2018", "095_04")
        level1_to_level2("low_band2", "2018", "096_00")
        level1_to_level2("low_band2", "2018", "097_00")
        level1_to_level2("low_band2", "2018", "097_15")
        level1_to_level2("low_band2", "2018", "098_00")
        level1_to_level2("low_band2", "2018", "099_00")
        level1_to_level2("low_band2", "2018", "100_00")
        level1_to_level2("low_band2", "2018", "101_00")
        level1_to_level2("low_band2", "2018", "102_00")
        level1_to_level2("low_band2", "2018", "103_00")
        level1_to_level2("low_band2", "2018", "104_00")
        level1_to_level2("low_band2", "2018", "105_00")
        level1_to_level2("low_band2", "2018", "106_00")
        level1_to_level2("low_band2", "2018", "107_00")
        level1_to_level2("low_band2", "2018", "108_00")
        level1_to_level2("low_band2", "2018", "109_00")
        level1_to_level2("low_band2", "2018", "110_00")
        level1_to_level2("low_band2", "2018", "111_00")
        level1_to_level2("low_band2", "2018", "112_00")
        level1_to_level2("low_band2", "2018", "113_00")
        level1_to_level2("low_band2", "2018", "114_00")
        level1_to_level2("low_band2", "2018", "115_00")
        level1_to_level2("low_band2", "2018", "116_00")
        level1_to_level2("low_band2", "2018", "117_00")
        level1_to_level2("low_band2", "2018", "118_00")
        level1_to_level2("low_band2", "2018", "119_00")
        level1_to_level2("low_band2", "2018", "120_00")
        level1_to_level2("low_band2", "2018", "121_00")
        level1_to_level2("low_band2", "2018", "122_00")
        level1_to_level2("low_band2", "2018", "123_00")
        level1_to_level2("low_band2", "2018", "124_00")
        level1_to_level2("low_band2", "2018", "125_00")
        level1_to_level2("low_band2", "2018", "126_00")
        level1_to_level2("low_band2", "2018", "127_00")
        level1_to_level2("low_band2", "2018", "128_00")
        level1_to_level2("low_band2", "2018", "129_00")
        level1_to_level2("low_band2", "2018", "130_00")
        level1_to_level2("low_band2", "2018", "131_00")
        level1_to_level2("low_band2", "2018", "132_00")
        level1_to_level2("low_band2", "2018", "133_00")
        level1_to_level2("low_band2", "2018", "134_00")
        level1_to_level2("low_band2", "2018", "135_00")
        level1_to_level2("low_band2", "2018", "136_00")
        level1_to_level2("low_band2", "2018", "137_00")
        level1_to_level2("low_band2", "2018", "138_00")
        level1_to_level2("low_band2", "2018", "139_00")
        level1_to_level2("low_band2", "2018", "140_00")
        level1_to_level2("low_band2", "2018", "141_00")
        level1_to_level2("low_band2", "2018", "142_00")
        level1_to_level2("low_band2", "2018", "143_00")
        level1_to_level2("low_band2", "2018", "144_00")
        level1_to_level2("low_band2", "2018", "145_00")
        level1_to_level2("low_band2", "2018", "146_00")
        level1_to_level2("low_band2", "2018", "147_00")
        level1_to_level2("low_band2", "2018", "147_17")
        level1_to_level2("low_band2", "2018", "148_00")
        level1_to_level2("low_band2", "2018", "149_00")
        level1_to_level2("low_band2", "2018", "150_00")
        level1_to_level2("low_band2", "2018", "151_00")
        level1_to_level2("low_band2", "2018", "152_00")
        level1_to_level2("low_band2", "2018", "152_19")
        level1_to_level2("low_band2", "2018", "153_00")
        level1_to_level2("low_band2", "2018", "154_00")
        level1_to_level2("low_band2", "2018", "155_00")
        level1_to_level2("low_band2", "2018", "156_00")
        level1_to_level2("low_band2", "2018", "157_00")
        level1_to_level2("low_band2", "2018", "158_00")
        level1_to_level2("low_band2", "2018", "159_00")
        level1_to_level2("low_band2", "2018", "160_00")
        level1_to_level2("low_band2", "2018", "161_00")
        level1_to_level2("low_band2", "2018", "162_00")
        level1_to_level2("low_band2", "2018", "163_00")
        level1_to_level2("low_band2", "2018", "164_00")
        level1_to_level2("low_band2", "2018", "165_00")
        level1_to_level2("low_band2", "2018", "166_00")
        level1_to_level2("low_band2", "2018", "167_00")
        level1_to_level2("low_band2", "2018", "168_00")
        level1_to_level2("low_band2", "2018", "169_00")
        level1_to_level2("low_band2", "2018", "170_00")
        level1_to_level2("low_band2", "2018", "171_00")
        level1_to_level2("low_band2", "2018", "181")
        level1_to_level2("low_band2", "2018", "182")
        level1_to_level2("low_band2", "2018", "183")
        level1_to_level2("low_band2", "2018", "184")
        level1_to_level2("low_band2", "2018", "185")
        level1_to_level2("low_band2", "2018", "186")
        level1_to_level2("low_band2", "2018", "187")
        level1_to_level2("low_band2", "2018", "188")
        level1_to_level2("low_band2", "2018", "189")
        level1_to_level2("low_band2", "2018", "190")
        level1_to_level2("low_band2", "2018", "191")
        level1_to_level2("low_band2", "2018", "192")
        level1_to_level2("low_band2", "2018", "193")
        level1_to_level2("low_band2", "2018", "194")
        level1_to_level2("low_band2", "2018", "195")
        level1_to_level2("low_band2", "2018", "196")
        level1_to_level2("low_band2", "2018", "197")
        level1_to_level2("low_band2", "2018", "198")
        level1_to_level2("low_band2", "2018", "199")
        level1_to_level2("low_band2", "2018", "200")
        level1_to_level2("low_band2", "2018", "201")
        level1_to_level2("low_band2", "2018", "202")
        level1_to_level2("low_band2", "2018", "203")
        level1_to_level2("low_band2", "2018", "204")
        level1_to_level2("low_band2", "2018", "205")
        level1_to_level2("low_band2", "2018", "206")
        level1_to_level2("low_band2", "2018", "207")
        level1_to_level2("low_band2", "2018", "208")
        level1_to_level2("low_band2", "2018", "209")
        level1_to_level2("low_band2", "2018", "210")
        level1_to_level2("low_band2", "2018", "211")
        level1_to_level2("low_band2", "2018", "212")
        level1_to_level2("low_band2", "2018", "213")
        level1_to_level2("low_band2", "2018", "214")
        level1_to_level2("low_band2", "2018", "215")
        level1_to_level2("low_band2", "2018", "216")
        level1_to_level2("low_band2", "2018", "217")
        level1_to_level2("low_band2", "2018", "218")


def batch_mid_band_level1_to_level2():
    # Listing files to be processed
    path_files = home_folder + "/EDGES/spectra/level1/mid_band/300_350/"
    new_list = listdir(path_files)
    new_list.sort()

    for i in range(26, len(new_list)):

        day = new_list[i][12:18]
        print(day)

        if (int(day[0:3]) <= 170) or (
            int(day[0:3]) >= 174
        ):  # files in this range have problems
            src.edges_analysis.analysis.levels.level1_to_level2("mid_band", "2018", day)


def batch_low_band3_level1_to_level2():
    # Listing files to be processed
    path_files = home_folder + "/EDGES/spectra/level1/low_band3/300_350/"
    new_list = listdir(path_files)
    new_list.sort()

    for i in range(len(new_list)):  # range(26, len(new_list)):

        year = new_list[i][7:11]
        day = new_list[i][12:15]

        if (int(year) == 2018) and (
            int(day) == 225
        ):  # files in this range have problems
            print(year + " " + day + ": bad file")

        else:
            print(year + " " + day)
            src.edges_analysis.analysis.levels.level1_to_level2("low_band3", year, day)


def batch_mid_band_level2_to_level3(case, first_day, last_day):
    # Case selection
    # --------------------------------

    if case == 0:
        flag_folder = "case_nominal"

        receiver_cal_file = 2  # cterms=7, wterms=8 terms over 50-150 MHz

        antenna_s11_day = 147
        antenna_s11_case = 3  # taken 2+ minutes after turning on the switch
        antenna_s11_Nfit = 13  # 13 terms over 55-120 MHz

        balun_correction = 1
        ground_correction = 1
        beam_correction = 1

        f_low = 55
        f_high = 120
        n_fg = 5

    if case == 1:
        flag_folder = "test_rcv18_sw18_no_beam_correction"  # 'case_nominal_55_150MHz'

        receiver_cal_file = 2  # cterms=7, wterms=8 terms over 50-150 MHz

        antenna_s11_day = 147
        antenna_s11_case = 3  # taken 2+ minutes after turning on the switch
        antenna_s11_Nfit = 14  # 14 terms over 55-150 MHz

        balun_correction = 1
        ground_correction = 1
        beam_correction = 0  # 1

        f_low = 55
        f_high = 150
        n_fg = 5

    if case == 2:
        flag_folder = "calibration_2019_10_no_ground_loss_no_beam_corrections"

        receiver_cal_file = 100  # cterms=10, wterms=13 terms over 50-190 MHz

        antenna_s11_day = 147
        antenna_s11_case = 3  # taken 2+ minutes after turning on the switch
        antenna_s11_Nfit = 14  # 14 terms over 55-150 MHz, 13 terms over 55-120 MHz

        balun_correction = 1
        ground_correction = 0
        beam_correction = 0

        f_low = 55
        f_high = 150
        n_fg = 5

    if case == 3:
        flag_folder = "case_nominal_50-150MHz_no_ground_loss_no_beam_corrections"

        receiver_cal_file = 2  # cterms=7, wterms=8 terms over 50-150 MHz

        antenna_s11_day = 147
        antenna_s11_case = 3  # taken 2+ minutes after turning on the switch
        antenna_s11_Nfit = 14  # 14 terms over 55-150 MHz

        balun_correction = 1
        ground_correction = 0
        beam_correction = 0

        f_low = 50
        f_high = 150
        n_fg = 5

    if case == 4:
        flag_folder = (
            "case_nominal_8_11_terms_50-150MHz_no_ground_loss_no_beam_corrections"
        )

        receiver_cal_file = 26  # cterms=8, wterms=11 terms over 50-150 MHz

        antenna_s11_day = 147
        antenna_s11_case = 3  # taken 2+ minutes after turning on the switch
        antenna_s11_Nfit = 14  # 14 terms over 55-150 MHz

        balun_correction = 1
        ground_correction = 0
        beam_correction = 0

        f_low = 50
        f_high = 150
        n_fg = 5

    if case == 5:
        flag_folder = (
            "case_nominal_14_14_terms_55-150MHz_no_ground_loss_no_beam_corrections"
        )

        receiver_cal_file = 200  # cterms=8, wterms=11 terms over 50-150 MHz

        antenna_s11_day = 147
        antenna_s11_case = 3  # taken 2+ minutes after turning on the switch
        antenna_s11_Nfit = 14  # 14 terms over 55-150 MHz

        balun_correction = 1
        ground_correction = 0
        beam_correction = 0

        f_low = 55
        f_high = 150
        n_fg = 5

    if case == 61:
        flag_folder = "case_nominal_60-120MHz_7_4"

        receiver_cal_file = 301  # cterms=8, wterms=11 terms over 50-150 MHz

        antenna_s11_day = 147
        antenna_s11_case = 3  # taken 2+ minutes after turning on the switch
        antenna_s11_Nfit = 13  # 13 terms over 60-120 MHz

        balun_correction = 1
        ground_correction = 0
        beam_correction = 0

        f_low = 60
        f_high = 120
        n_fg = 5

    if case == 62:
        flag_folder = "case_nominal_60-120MHz_7_5"

        receiver_cal_file = 302  # cterms=8, wterms=11 terms over 50-150 MHz

        antenna_s11_day = 147
        antenna_s11_case = 3  # taken 2+ minutes after turning on the switch
        antenna_s11_Nfit = 13  # 13 terms over 60-120 MHz

        balun_correction = 1
        ground_correction = 0
        beam_correction = 0

        f_low = 60
        f_high = 120
        n_fg = 5

    if case == 63:
        flag_folder = "case_nominal_60-120MHz_7_9"

        receiver_cal_file = 303  # cterms=8, wterms=11 terms over 50-150 MHz

        antenna_s11_day = 147
        antenna_s11_case = 3  # taken 2+ minutes after turning on the switch
        antenna_s11_Nfit = 13  # 13 terms over 60-120 MHz

        balun_correction = 1
        ground_correction = 0
        beam_correction = 0

        f_low = 60
        f_high = 120
        n_fg = 5

    if case == 64:
        flag_folder = "case_nominal_60-120MHz_8_4"

        receiver_cal_file = 304  # cterms=8, wterms=11 terms over 50-150 MHz

        antenna_s11_day = 147
        antenna_s11_case = 3  # taken 2+ minutes after turning on the switch
        antenna_s11_Nfit = 13  # 13 terms over 60-120 MHz

        balun_correction = 1
        ground_correction = 0
        beam_correction = 0

        f_low = 60
        f_high = 120
        n_fg = 5

    if case == 65:
        flag_folder = "case_nominal_60-120MHz_8_5"

        receiver_cal_file = 305  # cterms=8, wterms=11 terms over 50-150 MHz

        antenna_s11_day = 147
        antenna_s11_case = 3  # taken 2+ minutes after turning on the switch
        antenna_s11_Nfit = 13  # 13 terms over 60-120 MHz

        balun_correction = 1
        ground_correction = 0
        beam_correction = 0

        f_low = 60
        f_high = 120
        n_fg = 5

    if case == 66:
        flag_folder = "case_nominal_60-120MHz_8_9"

        receiver_cal_file = 306  # cterms=8, wterms=11 terms over 50-150 MHz

        antenna_s11_day = 147
        antenna_s11_case = 3  # taken 2+ minutes after turning on the switch
        antenna_s11_Nfit = 13  # 13 terms over 60-120 MHz

        balun_correction = 1
        ground_correction = 0
        beam_correction = 0

        f_low = 60
        f_high = 120
        n_fg = 5

    if case == 67:
        flag_folder = "case_nominal_60-120MHz_5_4"

        receiver_cal_file = 307  # cterms=8, wterms=11 terms over 50-150 MHz

        antenna_s11_day = 147
        antenna_s11_case = 3  # taken 2+ minutes after turning on the switch
        antenna_s11_Nfit = 13  # 13 terms over 60-120 MHz

        balun_correction = 1
        ground_correction = 0
        beam_correction = 0

        f_low = 60
        f_high = 120
        n_fg = 5

    if case == 68:
        flag_folder = "case_nominal_60-120MHz_6_4"

        receiver_cal_file = 308  # cterms=8, wterms=11 terms over 50-150 MHz

        antenna_s11_day = 147
        antenna_s11_case = 3  # taken 2+ minutes after turning on the switch
        antenna_s11_Nfit = 13  # 13 terms over 60-120 MHz

        balun_correction = 1
        ground_correction = 0
        beam_correction = 0

        f_low = 60
        f_high = 120
        n_fg = 5

    if case == 69:
        flag_folder = "case_nominal_55-150MHz_7_7"

        receiver_cal_file = 21  # cterms=8, wterms=11 terms over 50-150 MHz

        antenna_s11_day = 147
        antenna_s11_case = 3  # taken 2+ minutes after turning on the switch
        antenna_s11_Nfit = 14  # 13 terms over 60-120 MHz

        balun_correction = 1
        ground_correction = 0
        beam_correction = 0

        f_low = 55
        f_high = 150
        n_fg = 5

    if case == 70:
        flag_folder = "case_nominal_55-150MHz_7_10"

        receiver_cal_file = 24  # cterms=8, wterms=11 terms over 50-150 MHz

        antenna_s11_day = 147
        antenna_s11_case = 3  # taken 2+ minutes after turning on the switch
        antenna_s11_Nfit = 14  # 13 terms over 60-120 MHz

        balun_correction = 1
        ground_correction = 0
        beam_correction = 0

        f_low = 55
        f_high = 150
        n_fg = 5

    if case == 71:
        flag_folder = "case_nominal_50-150MHz_7_8"

        receiver_cal_file = 201  # cterms=7, wterms=8 terms over 50-150 MHz

        antenna_s11_day = 147
        antenna_s11_case = 3  # taken 2+ minutes after turning on the switch
        antenna_s11_Nfit = 14  # 14 terms over 55-150 MHz

        balun_correction = 1
        ground_correction = 0
        beam_correction = 0

        f_low = 50
        f_high = 150
        n_fg = 5

    if case == 72:
        flag_folder = "case_nominal_50-150MHz_7_11"

        receiver_cal_file = 202  # cterms=7, wterms=8 terms over 50-150 MHz

        antenna_s11_day = 147
        antenna_s11_case = 3  # taken 2+ minutes after turning on the switch
        antenna_s11_Nfit = 14  # 14 terms over 55-150 MHz

        balun_correction = 1
        ground_correction = 0
        beam_correction = 0

        f_low = 50
        f_high = 150
        n_fg = 5

    if case == 73:
        flag_folder = "case_nominal_50-150MHz_7_8_LNA_rep1"

        receiver_cal_file = 203  # cterms=7, wterms=8 terms over 50-150 MHz

        antenna_s11_day = 147
        antenna_s11_case = 3  # taken 2+ minutes after turning on the switch
        antenna_s11_Nfit = 14  # 14 terms over 55-150 MHz

        balun_correction = 1
        ground_correction = 0
        beam_correction = 0

        f_low = 50
        f_high = 150
        n_fg = 5

    if case == 74:
        flag_folder = "case_nominal_50-150MHz_7_8_LNA_rep2"

        receiver_cal_file = 204  # cterms=7, wterms=8 terms over 50-150 MHz

        antenna_s11_day = 147
        antenna_s11_case = 3  # taken 2+ minutes after turning on the switch
        antenna_s11_Nfit = 14  # 14 terms over 55-150 MHz

        balun_correction = 1
        ground_correction = 0
        beam_correction = 0

        f_low = 50
        f_high = 150
        n_fg = 5

    if case == 75:
        flag_folder = "case_nominal_50-150MHz_7_8_LNA_rep12"

        receiver_cal_file = 205  # cterms=7, wterms=8 terms over 50-150 MHz

        antenna_s11_day = 147
        antenna_s11_case = 3  # taken 2+ minutes after turning on the switch
        antenna_s11_Nfit = 14  # 14 terms over 55-150 MHz

        balun_correction = 1
        ground_correction = 0
        beam_correction = 0

        f_low = 50
        f_high = 150
        n_fg = 5

    if case == 731:
        flag_folder = "case_nominal_50-150MHz_7_8_LNA_rep1_ant1"

        receiver_cal_file = 203  # cterms=7, wterms=8 terms over 50-150 MHz

        antenna_s11_day = 147
        antenna_s11_case = 1  # taken 2+ minutes after turning on the switch
        antenna_s11_Nfit = 14  # 14 terms over 55-150 MHz

        balun_correction = 1
        ground_correction = 0
        beam_correction = 0

        f_low = 50
        f_high = 150
        n_fg = 5

    if case == 735:
        flag_folder = "case_nominal_50-150MHz_7_8_LNA_rep1_ant5"

        receiver_cal_file = 203  # cterms=7, wterms=8 terms over 50-150 MHz

        antenna_s11_day = 147
        antenna_s11_case = 5  # taken 2+ minutes after turning on the switch
        antenna_s11_Nfit = 14  # 14 terms over 55-150 MHz

        balun_correction = 1
        ground_correction = 0
        beam_correction = 0

        f_low = 50
        f_high = 150
        n_fg = 5

    if case == 736:
        flag_folder = "case_nominal_50-150MHz_7_8_LNA_rep1_ant6"

        receiver_cal_file = 203  # cterms=7, wterms=8 terms over 50-150 MHz

        antenna_s11_day = 147
        antenna_s11_case = 6  # taken 2+ minutes after turning on the switch
        antenna_s11_Nfit = 14  # 14 terms over 55-150 MHz

        balun_correction = 1
        ground_correction = 0
        beam_correction = 0

        f_low = 50
        f_high = 150
        n_fg = 5

    if case == 401:
        flag_folder = "case_nominal_50-150MHz_LNA1_a1_h1_o1_s1_sim2"

        receiver_cal_file = 401  # cterms=7, wterms=8 terms over 50-150 MHz

        antenna_s11_day = 147
        antenna_s11_case = 3  # taken 2+ minutes after turning on the switch
        antenna_s11_Nfit = 14  # 14 terms over 55-150 MHz

        balun_correction = 1
        ground_correction = 0
        beam_correction = 0

        f_low = 50
        f_high = 150
        n_fg = 5

    if case == 402:
        flag_folder = "case_nominal_50-150MHz_LNA1_a1_h2_o1_s1_sim2"

        receiver_cal_file = 402  # cterms=7, wterms=8 terms over 50-150 MHz

        antenna_s11_day = 147
        antenna_s11_case = 3  # taken 2+ minutes after turning on the switch
        antenna_s11_Nfit = 14  # 14 terms over 55-150 MHz

        balun_correction = 1
        ground_correction = 0
        beam_correction = 0

        f_low = 50
        f_high = 150
        n_fg = 5

    if case == 403:
        flag_folder = "case_nominal_50-150MHz_LNA1_a2_h1_o1_s1_sim2"

        receiver_cal_file = 403  # cterms=7, wterms=8 terms over 50-150 MHz

        antenna_s11_day = 147
        antenna_s11_case = 3  # taken 2+ minutes after turning on the switch
        antenna_s11_Nfit = 14  # 14 terms over 55-150 MHz

        balun_correction = 1
        ground_correction = 0
        beam_correction = 0

        f_low = 50
        f_high = 150
        n_fg = 5

    if case == 404:
        flag_folder = "case_nominal_50-150MHz_LNA1_a2_h2_o1_s1_sim2"

        receiver_cal_file = 404  # cterms=7, wterms=8 terms over 50-150 MHz

        antenna_s11_day = 147
        antenna_s11_case = 3  # taken 2+ minutes after turning on the switch
        antenna_s11_Nfit = 14  # 14 terms over 55-150 MHz

        balun_correction = 1
        ground_correction = 0
        beam_correction = 0

        f_low = 50
        f_high = 150
        n_fg = 5

    if case == 405:
        flag_folder = "case_nominal_50-150MHz_LNA1_a2_h2_o1_s2_sim2"

        receiver_cal_file = 405  # cterms=7, wterms=8 terms over 50-150 MHz

        antenna_s11_day = 147
        antenna_s11_case = 3  # taken 2+ minutes after turning on the switch
        antenna_s11_Nfit = 14  # 14 terms over 55-150 MHz

        balun_correction = 1
        ground_correction = 0
        beam_correction = 0

        f_low = 50
        f_high = 150
        n_fg = 5

    if case == 406:
        flag_folder = "case_nominal_50-150MHz_LNA1_a2_h2_o2_s1_sim2"

        receiver_cal_file = 406  # cterms=7, wterms=8 terms over 50-150 MHz

        antenna_s11_day = 147
        antenna_s11_case = 3  # taken 2+ minutes after turning on the switch
        antenna_s11_Nfit = 14  # 14 terms over 55-150 MHz

        balun_correction = 1
        ground_correction = 0
        beam_correction = 0

        f_low = 50
        f_high = 150
        n_fg = 5

    if case == 407:
        flag_folder = "case_nominal_50-150MHz_LNA1_a2_h2_o2_s2_sim2"

        receiver_cal_file = 407  # cterms=7, wterms=8 terms over 50-150 MHz

        antenna_s11_day = 147
        antenna_s11_case = 3  # taken 2+ minutes after turning on the switch
        antenna_s11_Nfit = 14  # 14 terms over 55-150 MHz

        balun_correction = 1
        ground_correction = 0
        beam_correction = 0

        f_low = 50
        f_high = 150
        n_fg = 5

    if case == 500:
        flag_folder = "case_nominal_50-150MHz_LNA2_a2_h2_o2_s1_sim2_all_lc_no_bc"  #

        receiver_cal_file = 2  # cterms=7, wterms=8 terms over 50-150 MHz

        antenna_s11_day = 147
        antenna_s11_case = 3  # taken 2+ minutes after turning on the switch
        antenna_s11_Nfit = 14  # 14 terms over 55-150 MHz

        antenna_correction = 1
        balun_correction = 1
        ground_correction = 1
        beam_correction = 0

        f_low = 50
        f_high = 150
        n_fg = 5

    if case == 501:
        flag_folder = "case_nominal_50-150MHz_LNA2_a2_h2_o2_s1_sim2_all_lc_yes_bc"  #
        # 'case_nominal_50-150MHz_LNA1_a2_h2_o2_s1_sim2_all_corrections'

        receiver_cal_file = 2  # cterms=7, wterms=8 terms over 50-150 MHz

        antenna_s11_day = 147
        antenna_s11_case = 3  # taken 2+ minutes after turning on the switch
        antenna_s11_Nfit = 14  # 14 terms over 55-150 MHz

        antenna_correction = 1
        balun_correction = 1
        ground_correction = 1
        beam_correction = 1

        f_low = 50
        f_high = 150
        n_fg = 5

    if case == 510:
        flag_folder = (
            "test_A"  # 'case_nominal_50-150MHz_LNA1_a2_h2_o2_s1_sim2_all_corrections'
        )

        receiver_cal_file = 2  # cterms=7, wterms=8 terms over 50-150 MHz

        antenna_s11_day = 147
        antenna_s11_case = 3  # taken 2+ minutes after turning on the switch
        antenna_s11_Nfit = 14  # 14 terms over 55-150 MHz

        antenna_correction = 1
        balun_correction = 1
        ground_correction = 1
        beam_correction = 1
        beam_correction_case = (
            0  # alan0 beam (30x30m ground plane), haslam map with gaussian
        )
        # lat-function for spectral index

        f_low = 50
        f_high = 150
        n_fg = 5

    if case == 511:
        flag_folder = (
            "test_B"  # 'case_nominal_50-150MHz_LNA1_a2_h2_o2_s1_sim2_all_corrections'
        )

        receiver_cal_file = 2  # cterms=7, wterms=8 terms over 50-150 MHz

        antenna_s11_day = 147
        antenna_s11_case = 3  # taken 2+ minutes after turning on the switch
        antenna_s11_Nfit = 14  # 14 terms over 55-150 MHz

        antenna_correction = 1
        balun_correction = 1
        ground_correction = 1
        beam_correction = 1
        beam_correction_case = (
            1  # alan0 beam (30x30m ground plane), haslam map with gaussian
        )
        # lat-function for spectral index

        f_low = 50
        f_high = 150
        n_fg = 5

    # Listing files to be processed
    path_files = edges_folder + "mid_band/spectra/level2/"
    old_list = listdir(path_files)
    old_list.sort()

    bad_files = [
        "2018_153_00.hdf5",
        "2018_154_00.hdf5",
        "2018_155_00.hdf5",
        "2018_156_00.hdf5",
        "2018_158_00.hdf5",
        "2018_168_00.hdf5",
        "2018_183_00.hdf5",
        "2018_194_00.hdf5",
        "2018_202_00.hdf5",
        "2018_203_00.hdf5",
        "2018_206_00.hdf5",
        "2018_207_00.hdf5",
        "2018_213_00.hdf5",
        "2018_214_00.hdf5",
        "2018_221_00.hdf5",
        "2018_222_00.hdf5",
    ]

    new_list = []
    for i in range(len(old_list)):
        if old_list[i] not in bad_files:
            new_list.append(old_list[i])

    # Processing files
    for i in range(len(new_list)):

        day = int(new_list[i][5:8])

        if (day >= first_day) & (day <= last_day):
            print(day)

            level2_to_level3(
                "mid_band",
                new_list[i],
                flag_folder=flag_folder,
                receiver_cal_file=edges_folder
                + "mid_band/calibration/receiver_calibration/receiver1/2018_01_25C/results/"
                + CALFILES[receiver_cal_file],
                antenna_s11_year=2018,
                antenna_s11_day=antenna_s11_day,
                antenna_s11_case=antenna_s11_case,
                antenna_s11_Nfit=antenna_s11_Nfit,
                antenna_correction=antenna_correction,
                balun_correction=balun_correction,
                ground_correction=ground_correction,
                beam_correction=beam_correction,
                beam_correction_case=beam_correction_case,
                f_low=f_low,
                f_high=f_high,
                n_fg=n_fg,
            )

    return 0


def batch_low_band3_level2_to_level3(case):
    if case == 2:
        flag_folder = "case2"
        receiver_cal_file = (
            edges_folder
            + "mid_band/calibration/receiver_calibration/receiver1/2018_01_25C/results/nominal"
            "/calibration_files/original/calibration_file_receiver1_cterms7_wterms8.txt"
        )
        antenna_s11_day = 227
        antenna_s11_Nfit = 14
        beam_correction = 1
        balun_correction = 1
        f_low = 50
        f_high = 120
        n_fg = 7

    # Listing files to be processed
    path_files = edges_folder + "low_band3/spectra/level2/"
    new_list = listdir(path_files)
    new_list.sort()

    for i in range(len(new_list)):

        level2_to_level3(
            "low_band3",
            new_list[i],
            flag_folder=flag_folder,
            receiver_cal_file=receiver_cal_file,
            antenna_s11_year=2018,
            antenna_s11_day=antenna_s11_day,
            antenna_s11_Nfit=antenna_s11_Nfit,
            beam_correction=beam_correction,
            balun_correction=balun_correction,
            f_low=f_low,
            f_high=f_high,
            n_fg=n_fg,
        )

    return new_list


def batch_plot_daily_residuals_LST():
    # Listing files to be processed
    path_files = (
        home_folder + "/EDGES/spectra/level3/mid_band/nominal_60_160MHz_fullcal/"
    )
    new_list = listdir(path_files)
    new_list.sort()

    # Global settings
    # ----------------------------------

    # Computation of residuals
    LST_boundaries = np.arange(0, 25, 1)
    f_low = 60
    f_high = 150
    SUN_EL_max = -20
    MOON_EL_max = 90

    # Plotting
    LST_centers = list(np.arange(0.5, 24))
    LST_text = ["LST=" + str(LST_centers[i]) + " hr" for i in range(len(LST_centers))]
    DY = 4
    f_low_plot = 35
    f_high_plot = 155
    XTICKS = np.arange(60, 156, 20)
    XTEXT = 38
    YLABEL = "4 K per division"

    # 3 foreground terms
    n_fg = 5
    figure_path = (
        "/data5/raul/EDGES/results/plots/20181031/midband_residuals_nighttime_5terms/"
    )

    for i in range(len(new_list)):  # [len(new_list)-1]:
        print(new_list[i])

        # Computing residuals
        fb, rb, wb = tools.daily_residuals_LST(
            new_list[i],
            LST_boundaries=LST_boundaries,
            f_low=f_low,
            f_high=f_high,
            n_fg=n_fg,
            SUN_EL_max=SUN_EL_max,
            MOON_EL_max=MOON_EL_max,
        )

        # Plotting
        plots.plot_daily_residuals_LST(
            fb,
            rb,
            LST_text,
            DY=DY,
            f_low=f_low_plot,
            f_high=f_high_plot,
            XTICKS=XTICKS,
            XTEXT=XTEXT,
            YLABEL=YLABEL,
            TITLE=str(n_fg) + " TERMS:  " + new_list[i][0:-5],
            save=True,
            figure_path=figure_path,
            figure_name=new_list[i][0:-5],
        )


def plot_residuals_GHA_1hr_bin(f, r, w):
    # Settings
    # ----------------------------------

    GHA_edges = list(np.arange(0, 25))
    GHA_text = [
        "GHA=" + str(GHA_edges[i]) + "-" + str(GHA_edges[i + 1]) + " hr"
        for i in range(len(GHA_edges) - 1)
    ]
    DY = 0.4
    f_low_plot = 40
    f_high_plot = 125
    XTICKS = np.arange(60, 121, 20)
    XTEXT = 42
    YLABEL = str(DY) + " K per division"
    TITLE = "5 LINLOG terms, 60-120 MHz"
    figure_path = "/home/raul/Desktop/"
    figure_name = "linlog_5terms_60-120MHz"

    # Plotting
    src.edges_analysis.analysis.tools.plot_residuals(
        f,
        r,
        w,
        GHA_text,
        DY=DY,
        f_low=f_low_plot,
        f_high=f_high_plot,
        XTICKS=XTICKS,
        XTEXT=XTEXT,
        YLABEL=YLABEL,
        TITLE=TITLE,
        save=True,
        figure_path=figure_path,
        figure_name=figure_name,
    )


def plot_residuals_GHA_Xhr_bin(f, r, w):
    # Settings
    # ----------------------------------
    LST_text = ["GHA=0-5 hr", "GHA=5-11 hr", "GHA=11-18 hr", "GHA=18-24 hr"]
    DY = 0.5
    f_low_plot = 35
    f_high_plot = 165
    XTICKS = np.arange(60, 161, 20)
    XTEXT = 38
    YLABEL = str(DY) + " K per division"
    TITLE = "4 LINLOG terms, 62-120 MHz"
    figure_path = "/home/raul/Desktop/"
    figure_name = "CASE2_linlog_4terms_62-120MHz"
    FIG_SX = 8
    FIG_SY = 7

    # Plotting
    src.edges_analysis.analysis.tools.plot_residuals(
        f,
        r,
        w,
        LST_text,
        FIG_SX=FIG_SX,
        FIG_SY=FIG_SY,
        DY=DY,
        f_low=f_low_plot,
        f_high=f_high_plot,
        XTICKS=XTICKS,
        XTEXT=XTEXT,
        YLABEL=YLABEL,
        TITLE=TITLE,
        save=True,
        figure_path=figure_path,
        figure_name=figure_name,
    )


def vna_comparison():
    path_folder1 = edges_folder + "others/vna_comparison/keysight_e5061a/"
    path_folder2 = edges_folder + "others/vna_comparison/copper_mountain_r60/"
    path_folder3 = edges_folder + "others/vna_comparison/tektronix_ttr506a/"
    path_folder4 = edges_folder + "others/vna_comparison/copper_mountain_tr1300/"

    o_K, f = rc.s1p_read(path_folder1 + "AGILENT_E5061A_OPEN.s1p")
    s_K, f = rc.s1p_read(path_folder1 + "AGILENT_E5061A_SHORT.s1p")
    m_K, f = rc.s1p_read(path_folder1 + "AGILENT_E5061A_MATCH.s1p")
    at3_K, f = rc.s1p_read(path_folder1 + "AGILENT_E5061A_3dB_ATTENUATOR.s1p")
    at6_K, f = rc.s1p_read(path_folder1 + "AGILENT_E5061A_6dB_ATTENUATOR.s1p")
    at10_K, f = rc.s1p_read(path_folder1 + "AGILENT_E5061A_10dB_ATTENUATOR.s1p")
    at15_K, f = rc.s1p_read(path_folder1 + "AGILENT_E5061A_15dB_ATTENUATOR.s1p")

    o_R, f = rc.s1p_read(path_folder2 + "OPEN.s1p")
    s_R, f = rc.s1p_read(path_folder2 + "SHORT.s1p")
    m_R, f = rc.s1p_read(path_folder2 + "MATCH.s1p")
    at3_R, f = rc.s1p_read(path_folder2 + "3dB_ATTENUATOR.s1p")
    at6_R, f = rc.s1p_read(path_folder2 + "6dB_ATTENUATOR.s1p")
    at10_R, f = rc.s1p_read(path_folder2 + "10dB_ATTENUATOR.s1p")
    at15_R, f = rc.s1p_read(path_folder2 + "15dB_ATTENUATOR.s1p")

    o_T, f = rc.s1p_read(path_folder3 + "uncalibrated_Open02.s1p")
    s_T, f = rc.s1p_read(path_folder3 + "uncalibrated_Short02.s1p")
    m_T, f = rc.s1p_read(path_folder3 + "uncalibrated_Match02.s1p")
    at3_T, f = rc.s1p_read(path_folder3 + "uncalibrated_3dB_Measurment2.s1p")
    at6_T, f = rc.s1p_read(path_folder3 + "uncalibrated_6dB_Measurment2.s1p")
    at10_T, f = rc.s1p_read(path_folder3 + "uncalibrated_10dB_Measurment2.s1p")
    at15_T, f = rc.s1p_read(path_folder3 + "uncalibrated_15dB_Measurment2.s1p")

    o_C, f = rc.s1p_read(path_folder4 + "Open_Measurment_01.s1p")
    s_C, f = rc.s1p_read(path_folder4 + "Short_Measurment_01.s1p")
    m_C, f = rc.s1p_read(path_folder4 + "Match_Measurment_01.s1p")
    at3_C, f = rc.s1p_read(path_folder4 + "3dB_Measurment_01.s1p")
    at6_C, f = rc.s1p_read(path_folder4 + "6dB_Measurment_01.s1p")
    at10_C, f = rc.s1p_read(path_folder4 + "10dB_Measurment_01.s1p")
    at15_C, f = rc.s1p_read(path_folder4 + "15dB_Measurment_01.s1p")

    xx = rc.agilent_85033E(f, 50, m=1, md_value_ps=38)
    o_a = xx[0]
    s_a = xx[1]
    m_a = xx[2]

    # Correction
    at3_Rc, xx1, xx2, xx3 = rc.de_embed(o_a, s_a, m_a, o_R, s_R, m_R, at3_R)
    at6_Rc, xx1, xx2, xx3 = rc.de_embed(o_a, s_a, m_a, o_R, s_R, m_R, at6_R)
    at10_Rc, xx1, xx2, xx3 = rc.de_embed(o_a, s_a, m_a, o_R, s_R, m_R, at10_R)
    at15_Rc, xx1, xx2, xx3 = rc.de_embed(o_a, s_a, m_a, o_R, s_R, m_R, at15_R)

    at3_Tc, xx1, xx2, xx3 = rc.de_embed(o_a, s_a, m_a, o_T, s_T, m_T, at3_T)
    at6_Tc, xx1, xx2, xx3 = rc.de_embed(o_a, s_a, m_a, o_T, s_T, m_T, at6_T)
    at10_Tc, xx1, xx2, xx3 = rc.de_embed(o_a, s_a, m_a, o_T, s_T, m_T, at10_T)
    at15_Tc, xx1, xx2, xx3 = rc.de_embed(o_a, s_a, m_a, o_T, s_T, m_T, at15_T)

    at3_Kc, xx1, xx2, xx3 = rc.de_embed(o_a, s_a, m_a, o_K, s_K, m_K, at3_K)
    at6_Kc, xx1, xx2, xx3 = rc.de_embed(o_a, s_a, m_a, o_K, s_K, m_K, at6_K)
    at10_Kc, xx1, xx2, xx3 = rc.de_embed(o_a, s_a, m_a, o_K, s_K, m_K, at10_K)
    at15_Kc, xx1, xx2, xx3 = rc.de_embed(o_a, s_a, m_a, o_K, s_K, m_K, at15_K)

    at3_Cc, xx1, xx2, xx3 = rc.de_embed(o_a, s_a, m_a, o_C, s_C, m_C, at3_C)
    at6_Cc, xx1, xx2, xx3 = rc.de_embed(o_a, s_a, m_a, o_C, s_C, m_C, at6_C)
    at10_Cc, xx1, xx2, xx3 = rc.de_embed(o_a, s_a, m_a, o_C, s_C, m_C, at10_C)
    at15_Cc, xx1, xx2, xx3 = rc.de_embed(o_a, s_a, m_a, o_C, s_C, m_C, at15_C)

    # Plot

    plt.figure(1)

    plt.subplot(4, 2, 1)
    plt.plot(f / 1e6, 20 * np.log10(np.abs(at3_Kc)))
    plt.plot(f / 1e6, 20 * np.log10(np.abs(at3_Rc)))
    plt.plot(f / 1e6, 20 * np.log10(np.abs(at3_Tc)))
    plt.plot(f / 1e6, 20 * np.log10(np.abs(at3_Cc)))
    plt.ylabel("3-dB Attn [dB]")
    plt.title("MAGNITUDE")
    plt.legend(["Keysight E5061A", "CM R60", "Tektronix", "CM TR1300"])

    plt.subplot(4, 2, 2)
    plt.plot(
        f / 1e6,
        (180 / np.pi) * np.unwrap(np.angle(at3_Rc))
        - (180 / np.pi) * np.unwrap(np.angle(at3_Kc)),
    )
    plt.plot(
        f / 1e6,
        (180 / np.pi) * np.unwrap(np.angle(at3_Rc))
        - (180 / np.pi) * np.unwrap(np.angle(at3_Kc)),
    )
    plt.plot(
        f / 1e6,
        (180 / np.pi) * np.unwrap(np.angle(at3_Tc))
        - (180 / np.pi) * np.unwrap(np.angle(at3_Kc)),
    )
    plt.plot(
        f / 1e6,
        (180 / np.pi) * np.unwrap(np.angle(at3_Cc))
        - (180 / np.pi) * np.unwrap(np.angle(at3_Kc)),
    )
    plt.ylabel("3-dB Attn [degrees]")
    plt.title(r"$\Delta$ PHASE")

    plt.subplot(4, 2, 3)
    plt.plot(f / 1e6, 20 * np.log10(np.abs(at6_Kc)))
    plt.plot(f / 1e6, 20 * np.log10(np.abs(at6_Rc)))
    plt.plot(f / 1e6, 20 * np.log10(np.abs(at6_Tc)))
    plt.plot(f / 1e6, 20 * np.log10(np.abs(at6_Cc)))
    plt.ylabel("6-dB Attn [dB]")

    plt.subplot(4, 2, 4)
    plt.plot(
        f / 1e6,
        (180 / np.pi) * np.unwrap(np.angle(at6_Rc))
        - (180 / np.pi) * np.unwrap(np.angle(at6_Kc)),
    )
    plt.plot(
        f / 1e6,
        (180 / np.pi) * np.unwrap(np.angle(at6_Rc))
        - (180 / np.pi) * np.unwrap(np.angle(at6_Kc)),
    )
    plt.plot(
        f / 1e6,
        (180 / np.pi) * np.unwrap(np.angle(at6_Tc))
        - (180 / np.pi) * np.unwrap(np.angle(at6_Kc)),
    )
    plt.plot(
        f / 1e6,
        (180 / np.pi) * np.unwrap(np.angle(at6_Cc))
        - (180 / np.pi) * np.unwrap(np.angle(at6_Kc)),
    )
    plt.ylabel("6-dB Attn [degrees]")

    plt.subplot(4, 2, 5)
    plt.plot(f / 1e6, 20 * np.log10(np.abs(at10_Kc)))
    plt.plot(f / 1e6, 20 * np.log10(np.abs(at10_Rc)))
    plt.plot(f / 1e6, 20 * np.log10(np.abs(at10_Tc)))
    plt.plot(f / 1e6, 20 * np.log10(np.abs(at10_Cc)))
    plt.ylabel("10-dB Attn [dB]")

    plt.subplot(4, 2, 6)
    plt.plot(
        f / 1e6,
        (180 / np.pi) * np.unwrap(np.angle(at10_Rc))
        - (180 / np.pi) * np.unwrap(np.angle(at10_Kc)),
    )
    plt.plot(
        f / 1e6,
        (180 / np.pi) * np.unwrap(np.angle(at10_Rc))
        - (180 / np.pi) * np.unwrap(np.angle(at10_Kc)),
    )
    plt.plot(
        f / 1e6,
        (180 / np.pi) * np.unwrap(np.angle(at10_Tc))
        - (180 / np.pi) * np.unwrap(np.angle(at10_Kc)),
    )
    plt.plot(
        f / 1e6,
        (180 / np.pi) * np.unwrap(np.angle(at10_Cc))
        - (180 / np.pi) * np.unwrap(np.angle(at10_Kc)),
    )
    plt.ylabel("10-dB Attn [degrees]")

    plt.subplot(4, 2, 7)
    plt.plot(f / 1e6, 20 * np.log10(np.abs(at15_Kc)))
    plt.plot(f / 1e6, 20 * np.log10(np.abs(at15_Rc)))
    plt.plot(f / 1e6, 20 * np.log10(np.abs(at15_Tc)))
    plt.plot(f / 1e6, 20 * np.log10(np.abs(at15_Cc)))
    plt.xlabel("frequency [MHz]")
    plt.ylabel("15-dB Attn [dB]")

    plt.subplot(4, 2, 8)
    plt.plot(
        f / 1e6,
        (180 / np.pi) * np.unwrap(np.angle(at15_Rc))
        - (180 / np.pi) * np.unwrap(np.angle(at15_Kc)),
    )
    plt.plot(
        f / 1e6,
        (180 / np.pi) * np.unwrap(np.angle(at15_Rc))
        - (180 / np.pi) * np.unwrap(np.angle(at15_Kc)),
    )
    plt.plot(
        f / 1e6,
        (180 / np.pi) * np.unwrap(np.angle(at15_Tc))
        - (180 / np.pi) * np.unwrap(np.angle(at15_Kc)),
    )
    plt.plot(
        f / 1e6,
        (180 / np.pi) * np.unwrap(np.angle(at15_Cc))
        - (180 / np.pi) * np.unwrap(np.angle(at15_Kc)),
    )
    plt.xlabel("frequency [MHz]")
    plt.ylabel("15-dB Attn [degrees]")

    return (
        f,
        at3_K,
        at6_K,
        at10_K,
        at15_K,
        at3_Kc,
        at6_Kc,
        at10_Kc,
        at15_Kc,
        at3_R,
        at6_R,
        at10_R,
        at15_R,
        at3_Rc,
        at6_Rc,
        at10_Rc,
        at15_Rc,
    )


def VNA_comparison2():
    path_folder1 = edges_folder + "others/vna_comparison/again/ks_e5061a/"
    path_folder2 = edges_folder + "others/vna_comparison/again/cm_tr1300/"

    o_K1, f = rc.s1p_read(path_folder1 + "Open_Measurment_01.s1p")
    s_K1, f = rc.s1p_read(path_folder1 + "Short_Measurment_01.s1p")
    m_K1, f = rc.s1p_read(path_folder1 + "Match_Measurment_01.s1p")
    at3_K1, f = rc.s1p_read(path_folder1 + "3dB_Mearsurment_01.s1p")
    at6_K1, f = rc.s1p_read(path_folder1 + "6dB_Mearsurment_01.s1p")
    at10_K1, f = rc.s1p_read(path_folder1 + "10dB_Mearsurment_01.s1p")
    at15_K1, f = rc.s1p_read(path_folder1 + "15dB_Mearsurment_01.s1p")

    o_K2, f = rc.s1p_read(path_folder1 + "Open_Measurment_02.s1p")
    s_K2, f = rc.s1p_read(path_folder1 + "Short_Measurment_02.s1p")
    m_K2, f = rc.s1p_read(path_folder1 + "Match_Measurment_02.s1p")
    at3_K2, f = rc.s1p_read(path_folder1 + "3dB_Mearsurment_02.s1p")
    at6_K2, f = rc.s1p_read(path_folder1 + "6dB_Mearsurment_02.s1p")
    at10_K2, f = rc.s1p_read(path_folder1 + "10dB_Mearsurment_02.s1p")
    at15_K2, f = rc.s1p_read(path_folder1 + "15dB_Mearsurment_02.s1p")

    o_C1, f = rc.s1p_read(path_folder2 + "Open_Measurment_01.s1p")
    s_C1, f = rc.s1p_read(path_folder2 + "Short_Measurment_01.s1p")
    m_C1, f = rc.s1p_read(path_folder2 + "Match_Measurment_01.s1p")
    at3_C1, f = rc.s1p_read(path_folder2 + "3dB_Measurment_01.s1p")
    at6_C1, f = rc.s1p_read(path_folder2 + "6dB_Measurment_01.s1p")
    at10_C1, f = rc.s1p_read(path_folder2 + "10dB_Measurment_01.s1p")
    at15_C1, f = rc.s1p_read(path_folder2 + "15dB_Measurment_01.s1p")

    o_C2, f = rc.s1p_read(path_folder2 + "Open_Measurment_02.s1p")
    s_C2, f = rc.s1p_read(path_folder2 + "Short_Measurment_02.s1p")
    m_C2, f = rc.s1p_read(path_folder2 + "Match_Measurment_02.s1p")
    at3_C2, f = rc.s1p_read(path_folder2 + "3dB_Measurment_02.s1p")
    at6_C2, f = rc.s1p_read(path_folder2 + "6dB_Measurment_02.s1p")
    at10_C2, f = rc.s1p_read(path_folder2 + "10dB_Measurment_02.s1p")
    at15_C2, f = rc.s1p_read(path_folder2 + "15dB_Measurment_02.s1p")

    xx = rc.agilent_85033E(f, 50, m=1, md_value_ps=38)
    o_a = xx[0]
    s_a = xx[1]
    m_a = xx[2]

    # Correction
    at3_Kc, xx1, xx2, xx3 = rc.de_embed(o_a, s_a, m_a, o_K2, s_K2, m_K2, at3_K2)
    at6_Kc, xx1, xx2, xx3 = rc.de_embed(o_a, s_a, m_a, o_K2, s_K2, m_K2, at6_K2)
    at10_Kc, xx1, xx2, xx3 = rc.de_embed(o_a, s_a, m_a, o_K2, s_K2, m_K2, at10_K2)
    at15_Kc, xx1, xx2, xx3 = rc.de_embed(o_a, s_a, m_a, o_K2, s_K2, m_K2, at15_K2)

    at3_Cc, xx1, xx2, xx3 = rc.de_embed(o_a, s_a, m_a, o_C2, s_C2, m_C2, at3_C2)
    at6_Cc, xx1, xx2, xx3 = rc.de_embed(o_a, s_a, m_a, o_C2, s_C2, m_C2, at6_C2)
    at10_Cc, xx1, xx2, xx3 = rc.de_embed(o_a, s_a, m_a, o_C2, s_C2, m_C2, at10_C2)
    at15_Cc, xx1, xx2, xx3 = rc.de_embed(o_a, s_a, m_a, o_C2, s_C2, m_C2, at15_C2)

    # Plot

    plt.figure(1, figsize=(15, 10))

    plt.subplot(4, 2, 1)
    plt.plot(f / 1e6, 20 * np.log10(np.abs(at3_Kc)), "k")
    plt.plot(f / 1e6, 20 * np.log10(np.abs(at3_Cc)), "r")

    plt.ylabel("3-dB Attn [dB]")
    plt.title("MAGNITUDE")
    plt.legend(["Keysight E5061A", "CM TR1300"])

    plt.subplot(4, 2, 2)
    plt.plot(
        f / 1e6,
        (180 / np.pi) * np.unwrap(np.angle(at3_Cc))
        - (180 / np.pi) * np.unwrap(np.angle(at3_Kc)),
        "r",
    )
    plt.ylabel("3-dB Attn [degrees]")
    plt.title(r"$\Delta$ PHASE")

    plt.subplot(4, 2, 3)
    plt.plot(f / 1e6, 20 * np.log10(np.abs(at6_Kc)), "k")
    plt.plot(f / 1e6, 20 * np.log10(np.abs(at6_Cc)), "r")
    plt.ylabel("6-dB Attn [dB]")

    plt.subplot(4, 2, 4)
    plt.plot(
        f / 1e6,
        (180 / np.pi) * np.unwrap(np.angle(at6_Cc))
        - (180 / np.pi) * np.unwrap(np.angle(at6_Kc)),
        "r",
    )
    plt.ylabel("6-dB Attn [degrees]")

    plt.subplot(4, 2, 5)
    plt.plot(f / 1e6, 20 * np.log10(np.abs(at10_Kc)), "k")
    plt.plot(f / 1e6, 20 * np.log10(np.abs(at10_Cc)), "r")
    plt.ylabel("10-dB Attn [dB]")

    plt.subplot(4, 2, 6)
    plt.plot(
        f / 1e6,
        (180 / np.pi) * np.unwrap(np.angle(at10_Cc))
        - (180 / np.pi) * np.unwrap(np.angle(at10_Kc)),
        "r",
    )
    plt.ylabel("10-dB Attn [degrees]")

    plt.subplot(4, 2, 7)
    plt.plot(f / 1e6, 20 * np.log10(np.abs(at15_Kc)), "k")
    plt.plot(f / 1e6, 20 * np.log10(np.abs(at15_Cc)), "r")
    plt.xlabel("frequency [MHz]")
    plt.ylabel("15-dB Attn [dB]")

    plt.subplot(4, 2, 8)
    plt.plot(
        f / 1e6,
        (180 / np.pi) * np.unwrap(np.angle(at15_Cc))
        - (180 / np.pi) * np.unwrap(np.angle(at15_Kc)),
        "r",
    )
    plt.xlabel("frequency [MHz]")
    plt.ylabel("15-dB Attn [degrees]")

    plt.savefig(
        edges_folder + "/results/plots/20190415/vna_comparison.pdf", bbox_inches="tight"
    )


def VNA_comparison3():
    path_folder1 = (
        edges_folder + "others/vna_comparison/fieldfox_N9923A/agilent_E5061A_male/"
    )
    path_folder2 = (
        edges_folder + "others/vna_comparison/fieldfox_N9923A/agilent_E5061A_female/"
    )
    path_folder3 = (
        edges_folder + "others/vna_comparison/fieldfox_N9923A/fieldfox_N9923A_male/"
    )
    path_folder4 = (
        edges_folder + "others/vna_comparison/fieldfox_N9923A/fieldfox_N9923A_female/"
    )

    REP = "02"

    A_o, f = rc.s1p_read(path_folder1 + "Open" + REP + ".s1p")
    A_s, f = rc.s1p_read(path_folder1 + "Short" + REP + ".s1p")
    A_m, f = rc.s1p_read(path_folder1 + "Match" + REP + ".s1p")
    A_at3, f = rc.s1p_read(path_folder1 + "3dB_" + REP + ".s1p")
    A_at6, f = rc.s1p_read(path_folder1 + "6dB_" + REP + ".s1p")
    A_at10, f = rc.s1p_read(path_folder1 + "10dB_" + REP + ".s1p")
    A_at15, f = rc.s1p_read(path_folder1 + "15dB_" + REP + ".s1p")

    B_o, f = rc.s1p_read(path_folder2 + "Open_" + REP + ".s1p")
    B_s, f = rc.s1p_read(path_folder2 + "Short_" + REP + ".s1p")
    B_m, f = rc.s1p_read(path_folder2 + "Match_" + REP + ".s1p")
    B_at3, f = rc.s1p_read(path_folder2 + "3dB_" + REP + ".s1p")
    B_at6, f = rc.s1p_read(path_folder2 + "6dB_02.s1p")
    B_at10, f = rc.s1p_read(path_folder2 + "10dB_" + REP + ".s1p")
    B_at15, f = rc.s1p_read(path_folder2 + "15dB_" + REP + ".s1p")

    C_o, f = rc.s1p_read(path_folder3 + "OPEN" + REP + ".s1p")
    C_s, f = rc.s1p_read(path_folder3 + "SHORT" + REP + ".s1p")
    C_m, f = rc.s1p_read(path_folder3 + "MATCH" + REP + ".s1p")
    C_at3, f = rc.s1p_read(path_folder3 + "3DB_" + REP + ".s1p")
    C_at6, f = rc.s1p_read(path_folder3 + "6DB_" + REP + ".s1p")
    C_at10, f = rc.s1p_read(path_folder3 + "10DB_" + REP + ".s1p")
    C_at15, f = rc.s1p_read(path_folder3 + "15DB_" + REP + ".s1p")

    D_o, f = rc.s1p_read(path_folder4 + "OPEN" + REP + ".s1p")
    D_s, f = rc.s1p_read(path_folder4 + "SHORT" + REP + ".s1p")
    D_m, f = rc.s1p_read(path_folder4 + "MATCH" + REP + ".s1p")
    D_at3, f = rc.s1p_read(path_folder4 + "3DB_" + REP + ".s1p")
    D_at6, f = rc.s1p_read(path_folder4 + "6DB_02.s1p")
    D_at10, f = rc.s1p_read(path_folder4 + "10DB_" + REP + ".s1p")
    D_at15, f = rc.s1p_read(path_folder4 + "15DB_" + REP + ".s1p")

    xx = rc.agilent_85033E(f, 50, m=1, md_value_ps=38)
    o_a = xx[0]
    s_a = xx[1]
    m_a = xx[2]

    # Correction
    A_at3c, xx1, xx2, xx3 = rc.de_embed(o_a, s_a, m_a, A_o, A_s, A_m, A_at3)
    A_at6c, xx1, xx2, xx3 = rc.de_embed(o_a, s_a, m_a, A_o, A_s, A_m, A_at6)
    A_at10c, xx1, xx2, xx3 = rc.de_embed(o_a, s_a, m_a, A_o, A_s, A_m, A_at10)
    A_at15c, xx1, xx2, xx3 = rc.de_embed(o_a, s_a, m_a, A_o, A_s, A_m, A_at15)

    B_at3c, xx1, xx2, xx3 = rc.de_embed(o_a, s_a, m_a, B_o, B_s, B_m, B_at3)
    B_at6c, xx1, xx2, xx3 = rc.de_embed(o_a, s_a, m_a, B_o, B_s, B_m, B_at6)
    B_at10c, xx1, xx2, xx3 = rc.de_embed(o_a, s_a, m_a, B_o, B_s, B_m, B_at10)
    B_at15c, xx1, xx2, xx3 = rc.de_embed(o_a, s_a, m_a, B_o, B_s, B_m, B_at15)

    C_at3c, xx1, xx2, xx3 = rc.de_embed(o_a, s_a, m_a, C_o, C_s, C_m, C_at3)
    C_at6c, xx1, xx2, xx3 = rc.de_embed(o_a, s_a, m_a, C_o, C_s, C_m, C_at6)
    C_at10c, xx1, xx2, xx3 = rc.de_embed(o_a, s_a, m_a, C_o, C_s, C_m, C_at10)
    C_at15c, xx1, xx2, xx3 = rc.de_embed(o_a, s_a, m_a, C_o, C_s, C_m, C_at15)

    D_at3c, xx1, xx2, xx3 = rc.de_embed(o_a, s_a, m_a, D_o, D_s, D_m, D_at3)
    D_at6c, xx1, xx2, xx3 = rc.de_embed(o_a, s_a, m_a, D_o, D_s, D_m, D_at6)
    D_at10c, xx1, xx2, xx3 = rc.de_embed(o_a, s_a, m_a, D_o, D_s, D_m, D_at10)
    D_at15c, xx1, xx2, xx3 = rc.de_embed(o_a, s_a, m_a, D_o, D_s, D_m, D_at15)

    # Plot

    plt.figure(1)

    plt.subplot(4, 2, 1)
    plt.plot(f / 1e6, 20 * np.log10(np.abs(A_at3c)), "b")
    plt.plot(f / 1e6, 20 * np.log10(np.abs(C_at3c)), "b--")
    plt.plot(f / 1e6, 20 * np.log10(np.abs(B_at3c)), "r")
    plt.plot(f / 1e6, 20 * np.log10(np.abs(D_at3c)), "r--")
    plt.ylabel("3-dB Attn [dB]")
    plt.title("MAGNITUDE")

    plt.subplot(4, 2, 2)
    plt.plot(
        f / 1e6,
        (180 / np.pi) * np.unwrap(np.angle(C_at3c))
        - (180 / np.pi) * np.unwrap(np.angle(A_at3c)),
        "b--",
    )
    plt.plot(
        f / 1e6,
        (180 / np.pi) * np.unwrap(np.angle(D_at3c))
        - (180 / np.pi) * np.unwrap(np.angle(B_at3c)),
        "r--",
    )
    plt.ylabel("3-dB Attn [degrees]")
    plt.title(r"$\Delta$ PHASE")

    plt.subplot(4, 2, 3)
    plt.plot(f / 1e6, 20 * np.log10(np.abs(A_at6c)), "b")
    plt.plot(f / 1e6, 20 * np.log10(np.abs(C_at6c)), "b--")
    plt.plot(f / 1e6, 20 * np.log10(np.abs(B_at6c)), "r")
    plt.plot(f / 1e6, 20 * np.log10(np.abs(D_at6c)), "r--")
    plt.ylabel("6-dB Attn [dB]")

    plt.subplot(4, 2, 4)
    plt.plot(
        f / 1e6,
        (180 / np.pi) * np.unwrap(np.angle(C_at6c))
        - (180 / np.pi) * np.unwrap(np.angle(A_at6c)),
        "b--",
    )
    plt.plot(
        f / 1e6,
        (180 / np.pi) * np.unwrap(np.angle(D_at6c))
        - (180 / np.pi) * np.unwrap(np.angle(B_at6c)),
        "r--",
    )
    plt.ylabel("6-dB Attn [degrees]")

    plt.subplot(4, 2, 5)
    plt.plot(f / 1e6, 20 * np.log10(np.abs(A_at10c)), "b")
    plt.plot(f / 1e6, 20 * np.log10(np.abs(C_at10c)), "b--")
    plt.plot(f / 1e6, 20 * np.log10(np.abs(B_at10c)), "r")
    plt.plot(f / 1e6, 20 * np.log10(np.abs(D_at10c)), "r--")
    plt.ylabel("10-dB Attn [dB]")

    plt.subplot(4, 2, 6)
    plt.plot(
        f / 1e6,
        (180 / np.pi) * np.unwrap(np.angle(C_at10c))
        - (180 / np.pi) * np.unwrap(np.angle(A_at10c)),
        "b--",
    )
    plt.plot(
        f / 1e6,
        (180 / np.pi) * np.unwrap(np.angle(D_at10c))
        - (180 / np.pi) * np.unwrap(np.angle(B_at10c)),
        "r--",
    )
    plt.ylabel("10-dB Attn [degrees]")

    plt.subplot(4, 2, 7)
    plt.plot(f / 1e6, 20 * np.log10(np.abs(A_at15c)), "b")
    plt.plot(f / 1e6, 20 * np.log10(np.abs(C_at15c)), "b--")
    plt.plot(f / 1e6, 20 * np.log10(np.abs(B_at15c)), "r")
    plt.plot(f / 1e6, 20 * np.log10(np.abs(D_at15c)), "r--")
    plt.xlabel("frequency [MHz]")
    plt.ylabel("15-dB Attn [dB]")
    plt.legend(["Male E5061A", "Male N9923A", "Female E5061A", "Female N9923A"])

    plt.subplot(4, 2, 8)
    plt.plot(
        f / 1e6,
        (180 / np.pi) * np.unwrap(np.angle(C_at15c))
        - (180 / np.pi) * np.unwrap(np.angle(A_at15c)),
        "b--",
    )
    plt.plot(
        f / 1e6,
        (180 / np.pi) * np.unwrap(np.angle(D_at15c))
        - (180 / np.pi) * np.unwrap(np.angle(B_at15c)),
        "r--",
    )
    plt.xlabel("frequency [MHz]")
    plt.ylabel("15-dB Attn [degrees]")


def VNA_comparison4():
    path_folder = edges_folder + "others/vna_comparison/keysight_P9370A/"

    o_p1, fx = rc.s1p_read(path_folder + "open.s1p")
    s_p1, fx = rc.s1p_read(path_folder + "short.s1p")
    m_p1, fx = rc.s1p_read(path_folder + "load.s1p")
    at3_p1, fx = rc.s1p_read(path_folder + "attn3db.s1p")
    at6_p1, fx = rc.s1p_read(path_folder + "attn6db.s1p")
    at10_p1, fx = rc.s1p_read(path_folder + "attn10db.s1p")

    o_p2, fx = rc.s1p_read(path_folder + "port2_open.s1p")
    s_p2, fx = rc.s1p_read(path_folder + "port2_short.s1p")
    m_p2, fx = rc.s1p_read(path_folder + "port2_load.s1p")
    at3_p2, fx = rc.s1p_read(path_folder + "port2_attn3db.s1p")
    at6_p2, fx = rc.s1p_read(path_folder + "port2_attn6db.s1p")
    at10_p2, fx = rc.s1p_read(path_folder + "port2_attn10db.s1p")

    f_low = 15e6
    f = fx[(fx >= f_low)]

    o_p1 = o_p1[(fx >= f_low)]
    s_p1 = s_p1[(fx >= f_low)]
    m_p1 = m_p1[(fx >= f_low)]
    at3_p1 = at3_p1[(fx >= f_low)]
    at6_p1 = at6_p1[(fx >= f_low)]
    at10_p1 = at10_p1[(fx >= f_low)]

    o_p2 = o_p2[(fx >= f_low)]
    s_p2 = s_p2[(fx >= f_low)]
    m_p2 = m_p2[(fx >= f_low)]
    at3_p2 = at3_p2[(fx >= f_low)]
    at6_p2 = at6_p2[(fx >= f_low)]
    at10_p2 = at10_p2[(fx >= f_low)]

    Leads = 0.004
    R50 = 48.785 - Leads
    R3 = 163.70 - Leads
    R6 = 85.04 - Leads
    R10 = 61.615 - Leads

    g3 = rc.impedance2gamma(R3, 50) * np.ones(len(f))
    g6 = rc.impedance2gamma(R6, 50) * np.ones(len(f))
    g10 = rc.impedance2gamma(R10, 50) * np.ones(len(f))

    xx = rc.agilent_85033E(f, R50, m=0, md_value_ps=38)
    o_a = xx[0]
    s_a = xx[1]
    m_a = xx[2]

    # Correction
    at3_p1c, xx1, xx2, xx3 = rc.de_embed(o_a, s_a, m_a, o_p1, s_p1, m_p1, at3_p1)
    at6_p1c, xx1, xx2, xx3 = rc.de_embed(o_a, s_a, m_a, o_p1, s_p1, m_p1, at6_p1)
    at10_p1c, xx1, xx2, xx3 = rc.de_embed(o_a, s_a, m_a, o_p1, s_p1, m_p1, at10_p1)

    at3_p2c, xx1, xx2, xx3 = rc.de_embed(o_a, s_a, m_a, o_p2, s_p2, m_p2, at3_p2)
    at6_p2c, xx1, xx2, xx3 = rc.de_embed(o_a, s_a, m_a, o_p2, s_p2, m_p2, at6_p2)
    at10_p2c, xx1, xx2, xx3 = rc.de_embed(o_a, s_a, m_a, o_p2, s_p2, m_p2, at10_p2)

    # Plot

    plt.figure(1, figsize=(12, 10))

    plt.subplot(3, 2, 1)
    plt.plot(f / 1e6, 20 * np.log10(np.abs(at3_p1c)), "k")
    plt.plot(f / 1e6, 20 * np.log10(np.abs(at3_p2c)), "r")
    plt.plot(f / 1e6, 20 * np.log10(np.abs(g3)), "g--")

    plt.ylabel("3-dB Attn [dB]")
    plt.title("MAGNITUDE")
    plt.legend(["Port 1", "Port 2", "From DC resistance"])

    plt.subplot(3, 2, 2)
    plt.plot(
        f / 1e6,
        (180 / np.pi) * np.unwrap(np.angle(at3_p2c))
        - (180 / np.pi) * np.unwrap(np.angle(at3_p1c)),
        "r",
    )
    plt.ylabel("3-dB Attn [degrees]")
    plt.title(r"$\Delta$ PHASE")

    plt.subplot(3, 2, 3)
    plt.plot(f / 1e6, 20 * np.log10(np.abs(at6_p1c)), "k")
    plt.plot(f / 1e6, 20 * np.log10(np.abs(at6_p2c)), "r")
    plt.plot(f / 1e6, 20 * np.log10(np.abs(g6)), "g--")

    plt.ylabel("6-dB Attn [dB]")

    plt.subplot(3, 2, 4)
    plt.plot(
        f / 1e6,
        (180 / np.pi) * np.unwrap(np.angle(at6_p2c))
        - (180 / np.pi) * np.unwrap(np.angle(at6_p1c)),
        "r",
    )
    plt.ylabel("6-dB Attn [degrees]")

    plt.subplot(3, 2, 5)
    plt.plot(f / 1e6, 20 * np.log10(np.abs(at10_p1c)), "k")
    plt.plot(f / 1e6, 20 * np.log10(np.abs(at10_p2c)), "r")
    plt.plot(f / 1e6, 20 * np.log10(np.abs(g10)), "g--")

    plt.ylabel("10-dB Attn [dB]")
    plt.xlabel("frequency [MHz]")

    plt.subplot(3, 2, 6)
    plt.plot(
        f / 1e6,
        (180 / np.pi) * np.unwrap(np.angle(at10_p2c))
        - (180 / np.pi) * np.unwrap(np.angle(at10_p1c)),
        "r",
    )
    plt.ylabel("10-dB Attn [degrees]")
    plt.xlabel("frequency [MHz]")

    plt.savefig(edges_folder + "plots/20190612/vna_comparison.pdf", bbox_inches="tight")


def plot_number_of_cterms_wterms():
    rms, cterms, wterms = io.calibration_rms_read(
        "/run/media/raul/WD_RED_6TB/EDGES_vol2/mid_band/calibration/receiver_calibration"
        "/receiver1/2018_01_25C/results/nominal/calibration_files"
        "/calibration_term_sweep_50_150MHz/calibration_term_sweep_50_150MHz.hdf5"
    )

    plt.figure()
    plt.imshow(
        np.flipud(rms[0, :, :]),
        interpolation="none",
        vmin=0.016,
        vmax=0.02,
        extent=[1, 15, 1, 15],
    )
    plt.colorbar()
    plt.figure()
    plt.imshow(
        np.flipud(rms[1, :, :]),
        interpolation="none",
        vmin=0.016,
        vmax=0.02,
        extent=[1, 15, 1, 15],
    )
    plt.colorbar()
    plt.figure()
    plt.imshow(
        np.flipud(rms[2, :, :]),
        interpolation="none",
        vmin=0.3,
        vmax=0.6,
        extent=[1, 15, 1, 15],
    )
    plt.colorbar()
    plt.figure()
    plt.imshow(
        np.flipud(rms[3, :, :]),
        interpolation="none",
        vmin=0.3,
        vmax=0.6,
        extent=[1, 15, 1, 15],
    )
    plt.colorbar()


def antsim3_calibration():
    f_low = 50
    f_high = 150

    # Spectra
    d = np.genfromtxt(
        edges_folder
        + "mid_band/calibration/receiver_calibration/receiver1/2018_01_25C/results"
        "/nominal/data/average_spectra_300_350.txt"
    )
    ff = d[:, 0]
    tx = d[:, 5]

    tunc = tx[(ff >= f_low) & (ff <= f_high)]

    (
        f,
        s11_LNA,
        C1,
        C2,
        TU,
        TC,
        TS,
    ) = src.edges_analysis.simulation.data_models.MC_receiver(
        "mid_band",
        MC_spectra_noise=np.zeros(4),
        MC_s11_syst=[1, 1, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0],
        MC_temp=np.zeros(4),
    )

    # AntSim3 S11
    # -----------
    path_s11 = (
        edges_folder + "mid_band/calibration/receiver_calibration/receiver1/2018_01_25C"
        "/results/nominal/s11/"
    )
    fn = (f - 120) / 60

    par = np.genfromtxt(path_s11 + "par_s11_simu_mag.txt")
    rsimu_mag = mdl.model_evaluate("polynomial", par, fn)
    par = np.genfromtxt(path_s11 + "par_s11_simu_ang.txt")
    rsimu_ang = mdl.model_evaluate("polynomial", par, fn)
    rsimu = rsimu_mag * (np.cos(rsimu_ang) + 1j * np.sin(rsimu_ang))

    rsimu_MC = src.edges_analysis.simulation.data_models.MC_antenna_s11(
        f, rsimu, s11_Npar_max=14
    )

    # Calibrated antenna temperature with losses and beam chromaticity
    # ----------------------------------------------------------------
    tcal = rcf.calibrated_antenna_temperature(
        tunc, rsimu_MC, s11_LNA, C1, C2, TU, TC, TS
    )

    fb, tb, wb = tools.spectral_binning_number_of_samples(f, tcal, np.ones(len(f)))

    return f, rsimu, rsimu_MC, fb, tb


def plots_for_memo148(plot_number):
    # Receiver calibration parameters
    if plot_number == 1:
        # Paths
        path_plot_save = edges_folder + "plots/20190828/"

        # Calibration parameters
        rcv_file = (
            edges_folder + "mid_band/calibration/receiver_calibration/receiver1"
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
        path_plot_save = edges_folder + "plots/20190828/"

        flb = np.arange(50, 100, 1)
        alb = np.ones(len(flb))

        f = np.arange(50, 151, 1)
        ant_s11 = np.ones(len(f))

        corrections = [s11c.low_band_switch_correction(alb, 27.16, f_in=flb)[1]]

        paths = [
            (
                edges_folder_v1
                + "calibration/receiver_calibration/mid_band/2017_11_15C_25C_35C/data/s11/raw/25C"
                "/receiver_MRO_fieldfox_40-200MHz/"
            ),
            (
                edges_folder
                + "mid_band/calibration/receiver_calibration/receiver1/2018_01_25C/data/s11/raw"
                "/InternalSwitch/"
            ),
            (
                edges_folder
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
        path_plot_save = edges_folder + "plots/20190828/"

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
        path_plot_save = edges_folder + "plots/20190828/"

        GHA1 = 6
        GHA2 = 18

        f, t150_low_case1, w, s150_low_case1 = io.level3_single_file_test(
            edges_folder
            + "mid_band/spectra/level3/tests_55_150MHz/rcv18_sw18/2018_150_00.hdf5",
            GHA1,
            GHA2,
            60,
            150,
            False,
            "LINLOG",
            5,
            False,
            "name",
        )
        f, t150_low_case2, w, s150_low_case2 = io.level3_single_file_test(
            edges_folder
            + "mid_band/spectra/level3/tests_55_150MHz/rcv18_sw19/2018_150_00"
            ".hdf5",
            GHA1,
            GHA2,
            60,
            150,
            False,
            "LINLOG",
            5,
            False,
            "name",
        )
        f, t150_low_case3, w, s150_low_case3 = io.level3_single_file_test(
            edges_folder
            + "mid_band/spectra/level3/tests_55_150MHz/rcv19_sw18/2018_150_00"
            ".hdf5",
            GHA1,
            GHA2,
            60,
            150,
            False,
            "LINLOG",
            5,
            False,
            "name",
        )
        f, t150_low_case4, w, s150_low_case4 = io.level3_single_file_test(
            edges_folder
            + "mid_band/spectra/level3/tests_55_150MHz/rcv19_sw19/2018_150_00"
            ".hdf5",
            GHA1,
            GHA2,
            60,
            150,
            False,
            "LINLOG",
            5,
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
        path_plot_save = edges_folder + "plots/20190828/"

        GHA1 = GHA2 = 18

        fx, t150_low_case1, w, s150_low_case1 = io.level3_single_file_test(
            edges_folder
            + "mid_band/spectra/level3/tests_55_150MHz/rcv18_sw18/2018_150_00.hdf5",
            GHA1,
            GHA2,
            55,
            150,
            False,
            "LINLOG",
            5,
            False,
            "name",
        )
        fx, t150_low_case2, w, s150_low_case2 = io.level3_single_file_test(
            edges_folder
            + "mid_band/spectra/level3/tests_55_150MHz/rcv18_sw19/2018_150_00.hdf5",
            GHA1,
            GHA2,
            55,
            150,
            False,
            "LINLOG",
            5,
            False,
            "name",
        )
        fx, t150_low_case3, w, s150_low_case3 = io.level3_single_file_test(
            edges_folder
            + "mid_band/spectra/level3/tests_55_150MHz/rcv19_sw18/2018_150_00.hdf5",
            GHA1,
            GHA2,
            55,
            150,
            False,
            "LINLOG",
            5,
            False,
            "name",
        )
        fx, t150_low_case4, w, s150_low_case4 = io.level3_single_file_test(
            edges_folder
            + "mid_band/spectra/level3/tests_55_150MHz/rcv19_sw19/2018_150_00.hdf5",
            GHA1,
            GHA2,
            55,
            150,
            False,
            "LINLOG",
            5,
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
        path_plot_save = edges_folder + "plots/20190828/"

    GHA1 = 6
    GHA2 = 18

    fx, t188_low_case1, w, s188_low_case1 = io.level3_single_file_test(
        edges_folder
        + "mid_band/spectra/level3/tests_55_150MHz/rcv18_sw18/2018_188_00.hdf5",
        GHA1,
        GHA2,
        55,
        150,
        False,
        "LINLOG",
        5,
        False,
        "name",
    )
    fx, t188_low_case2, w, s188_low_case2 = io.level3_single_file_test(
        edges_folder
        + "mid_band/spectra/level3/tests_55_150MHz/rcv18_sw19/2018_188_00.hdf5",
        GHA1,
        GHA2,
        55,
        150,
        False,
        "LINLOG",
        5,
        False,
        "name",
    )
    fx, t188_low_case3, w, s188_low_case3 = io.level3_single_file_test(
        edges_folder + "mid_band/spectra/level3"
        "/tests_55_150MHz/rcv19_sw18/2018_188_00.hdf5",
        GHA1,
        GHA2,
        55,
        150,
        False,
        "LINLOG",
        5,
        False,
        "name",
    )
    fx, t188_low_case4, w, s188_low_case4 = io.level3_single_file_test(
        edges_folder
        + "mid_band/spectra/level3/tests_55_150MHz/rcv19_sw19/2018_188_00.hdf5",
        GHA1,
        GHA2,
        55,
        150,
        False,
        "LINLOG",
        5,
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
        path_plot_save = edges_folder + "plots/20190828/"

    GHA1 = 6
    GHA2 = 18

    fx, t150_low_case1, w, s150_low_case1 = io.level3_single_file_test(
        edges_folder
        + "mid_band/spectra/level3/tests_55_150MHz/rcv18_sw18/2018_150_00.hdf5",
        GHA1,
        GHA2,
        55,
        150,
        False,
        "LINLOG",
        5,
        False,
        "name",
    )
    fx, t150_low_case2, w, s150_low_case2 = io.level3_single_file_test(
        edges_folder
        + "mid_band/spectra/level3/tests_55_150MHz/rcv18_sw19/2018_150_00.hdf5",
        GHA1,
        GHA2,
        55,
        150,
        False,
        "LINLOG",
        5,
        False,
        "name",
    )
    fx, t150_low_case3, w, s150_low_case3 = io.level3_single_file_test(
        edges_folder
        + "mid_band/spectra/level3/tests_55_150MHz/rcv18_sw18and19_ant147/2018_150_00.hdf5",
        GHA1,
        GHA2,
        55,
        150,
        False,
        "LINLOG",
        5,
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
        path_plot_save = edges_folder + "plots/20190828/"

    GHA1 = 6
    GHA2 = 18

    fx, t188_low_case1, w, s188_low_case1 = io.level3_single_file_test(
        edges_folder
        + "mid_band/spectra/level3/tests_55_150MHz/rcv18_sw18/2018_188_00.hdf5",
        GHA1,
        GHA2,
        55,
        150,
        False,
        "LINLOG",
        5,
        False,
        "name",
    )
    fx, t188_low_case2, w, s188_low_case2 = io.level3_single_file_test(
        edges_folder
        + "mid_band/spectra/level3/tests_55_150MHz/rcv18_sw19/2018_188_00.hdf5",
        GHA1,
        GHA2,
        55,
        150,
        False,
        "LINLOG",
        5,
        False,
        "name",
    )
    fx, t188_low_case3, w, s188_low_case3 = io.level3_single_file_test(
        edges_folder + "mid_band/spectra/level3/tests_55_150MHz/rcv18_sw18and19_ant147"
        "/2018_188_00.hdf5",
        GHA1,
        GHA2,
        55,
        150,
        False,
        "LINLOG",
        5,
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
    # import old_high_band_edges as ohb

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
        print(filename_list[i])

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
    path_plot_save = edges_folder + "plots/20191015/"

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
        edges_folder
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
        edges_folder
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
        edges_folder
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
        edges_folder
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
        edges_folder
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
        edges_folder
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
        edges_folder
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
        edges_folder
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
        edges_folder
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
        edges_folder
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
        edges_folder
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
        edges_folder
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
        edges_folder
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
        edges_folder
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


def plots_beam_gain_derivative():
    # Low-Band, Blade on 10m x 10m GP

    b_all = beams.FEKO_low_band_blade_beam(
        beam_file=5,
        frequency_interpolation=False,
        frequency=np.array([0]),
        AZ_antenna_axis=0,
    )

    f = np.arange(50, 121, 2)

    delta = np.zeros(
        (len(b_all[:, 0, 0]) - 1, len(b_all[0, :, 0]), len(b_all[0, 0, :]))
    )
    for i in range(len(f) - 1):
        XX = b_all[i + 1, :, :] - b_all[i, :, :]
        XXX = np.flipud(XX)
        delta[i, :, :] = XXX

    return delta, b_all, f


def integrated_antenna_gain_WIPLD_try1():
    filename = (
        "/run/media/raul/WD_RED_6TB/EDGES_vol2/others/beam_simulations/wipl-d/20191022"
        "/blade_dipole_infinite_soil_metal_GP_0.00001m_60-78MHz.ra1"
    )

    f1, thetaX, phiX, beam1 = beams.wipld_read(filename)

    filename = (
        "/run/media/raul/WD_RED_6TB/EDGES_vol2/others/beam_simulations/wipl-d/20191022"
        "/blade_dipole_infinite_soil_metal_GP_0.00001m_80-98MHz.ra1"
    )

    f2, thetaX, phiX, beam2 = beams.wipld_read(filename)

    filename = (
        "/run/media/raul/WD_RED_6TB/EDGES_vol2/others/beam_simulations/wipl-d/20191022"
        "/blade_dipole_infinite_soil_metal_GP_0.00001m_100-108MHz.ra1"
    )

    f3, thetaX, phiX, beam3 = beams.wipld_read(filename)

    f = np.concatenate((f1, f2, f3))
    beam = np.concatenate((beam1, beam2, beam3))

    theta = thetaX[thetaX < 90]
    m = beam[:, thetaX < 90, :]

    sin_theta = np.sin(theta * (np.pi / 180))
    sin_theta_2D_T = np.tile(sin_theta, (360, 1))
    sin_theta_2D = sin_theta_2D_T.T

    bint = np.zeros(len(f))
    for i in range(len(f)):
        b = m[i, :, :]
        bint[i] = np.sum(b * sin_theta_2D)

    btW = (1 / (4 * np.pi)) * ((np.pi / 180) ** 2) * bint  # /np.mean(bx)

    n_fg = 5
    f_low = 50
    f_high = 120
    fX = f[(f >= f_low) & (f <= f_high)]
    btWX = btW[(f >= f_low) & (f <= f_high)]

    x = np.polyfit(fX, btWX, n_fg - 1)
    model = np.polyval(x, fX)
    rtWX = btWX - model

    deltaW = np.zeros((len(m[:, 0, 0]) - 1, len(m[0, :, 0]), len(m[0, 0, :])))
    for i in range(len(f) - 1):
        deltaW[i, :, :] = m[i + 1, :, :] - m[i, :, :]

    return f, btW, fX, rtWX


def integrated_antenna_gain_WIPLD(case, n_fg):
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
    path_plot_save = edges_folder + "plots/20191025/"
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
    path_plot_save = edges_folder + "plots/20191105/"

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


def plot_calibration_s11():
    path_plot_save = edges_folder + "plots/20191112/"

    # Main folder
    main_folder = (
        edges_folder
        + "mid_band/calibration/receiver_calibration/receiver1/2019_10_25C/"
    )

    # Paths for source data
    path_s11 = main_folder + "data/s11/corrected/"

    d1 = np.genfromtxt(
        path_s11 + "s11_calibration_mid_band_LNA25degC_2019-11-09-19-13"
        "-48_second_set_of_measurements.txt"
    )
    d2 = np.genfromtxt(
        path_s11 + "s11_calibration_mid_band_LNA25degC_2019-11-12-21-08"
        "-17_first_set_of_measurements.txt"
    )

    plt.figure(1, figsize=(12, 14))

    plt.subplot(3, 2, 1)
    plt.plot(d2[:, 0], 20 * np.log10(np.abs(d2[:, 3] + 1j * d2[:, 4])), "b")
    plt.plot(d1[:, 0], 20 * np.log10(np.abs(d1[:, 3] + 1j * d1[:, 4])), "b--")
    plt.plot(d2[:, 0], 20 * np.log10(np.abs(d2[:, 5] + 1j * d2[:, 6])), "r")
    plt.plot(d1[:, 0], 20 * np.log10(np.abs(d1[:, 5] + 1j * d1[:, 6])), "r--")
    plt.ylabel("magnitude [dB]")
    plt.legend(["ambient rep1", "ambient rep2", "hot rep1", "hot rep2"])

    plt.subplot(3, 2, 2)
    plt.plot(
        d1[:, 0], (180 / np.pi) * np.unwrap(np.angle(d1[:, 3] + 1j * d1[:, 4])), "b--"
    )
    plt.plot(
        d2[:, 0], (180 / np.pi) * np.unwrap(np.angle(d2[:, 3] + 1j * d2[:, 4])), "b"
    )
    plt.plot(
        d1[:, 0], (180 / np.pi) * np.unwrap(np.angle(d1[:, 5] + 1j * d1[:, 6])), "r--"
    )
    plt.plot(
        d2[:, 0], (180 / np.pi) * np.unwrap(np.angle(d2[:, 5] + 1j * d2[:, 6])), "r"
    )
    plt.ylabel("phase [deg]")

    plt.subplot(3, 2, 3)
    plt.plot(d2[:, 0], 20 * np.log10(np.abs(d2[:, 7] + 1j * d2[:, 8])), "b")
    plt.plot(d1[:, 0], 20 * np.log10(np.abs(d1[:, 7] + 1j * d1[:, 8])), "b--")
    plt.plot(d2[:, 0], 20 * np.log10(np.abs(d2[:, 9] + 1j * d2[:, 10])), "r")
    plt.plot(d1[:, 0], 20 * np.log10(np.abs(d1[:, 9] + 1j * d1[:, 10])), "r--")
    plt.legend(["open rep1", "open rep2", "shorted rep1", "shorted rep2"])
    plt.ylabel("magnitude [dB]")

    plt.subplot(3, 2, 4)
    plt.plot(
        d1[:, 0], (180 / np.pi) * np.unwrap(np.angle(d1[:, 7] + 1j * d1[:, 8])), "b--"
    )
    plt.plot(
        d2[:, 0], (180 / np.pi) * np.unwrap(np.angle(d2[:, 7] + 1j * d2[:, 8])), "b"
    )
    plt.plot(
        d1[:, 0], (180 / np.pi) * np.unwrap(np.angle(d1[:, 9] + 1j * d1[:, 10])), "r--"
    )
    plt.plot(
        d2[:, 0], (180 / np.pi) * np.unwrap(np.angle(d2[:, 9] + 1j * d2[:, 10])), "r"
    )
    plt.ylabel("phase [deg]")

    plt.subplot(3, 2, 5)
    plt.plot(d2[:, 0], 20 * np.log10(np.abs(d2[:, 17] + 1j * d2[:, 18])), "b")
    plt.plot(d1[:, 0], 20 * np.log10(np.abs(d1[:, 17] + 1j * d1[:, 18])), "b--")
    plt.plot(d2[:, 0], 20 * np.log10(np.abs(d2[:, 19] + 1j * d2[:, 20])), "r")
    plt.plot(d1[:, 0], 20 * np.log10(np.abs(d1[:, 19] + 1j * d1[:, 20])), "r--")
    plt.legend(["sim2 rep1", "sim2 rep2", "sim3 rep1", "sim3 rep2"])
    plt.ylabel("magnitude [dB]")
    plt.xlabel("frequency [MHz]")

    plt.subplot(3, 2, 6)
    plt.plot(
        d2[:, 0], (180 / np.pi) * np.unwrap(np.angle(d2[:, 17] + 1j * d2[:, 18])), "b"
    )
    plt.plot(
        d1[:, 0], (180 / np.pi) * np.unwrap(np.angle(d1[:, 17] + 1j * d1[:, 18])), "b--"
    )
    plt.plot(
        d1[:, 0], (180 / np.pi) * np.unwrap(np.angle(d1[:, 19] + 1j * d1[:, 20])), "r--"
    )
    plt.plot(
        d2[:, 0], (180 / np.pi) * np.unwrap(np.angle(d2[:, 19] + 1j * d2[:, 20])), "r"
    )
    plt.ylabel("phase [deg]")
    plt.xlabel("frequency [MHz]")

    plt.savefig(path_plot_save + "fig1.pdf", bbox_inches="tight")
    plt.close()
    plt.close()

    plt.figure(2, figsize=(12, 14))

    plt.subplot(3, 2, 1)
    plt.plot(
        d1[:, 0],
        20 * np.log10(np.abs(d1[:, 3] + 1j * d1[:, 4]))
        - 20 * np.log10(np.abs(d2[:, 3] + 1j * d2[:, 4])),
        "b",
    )
    plt.plot(
        d1[:, 0],
        20 * np.log10(np.abs(d1[:, 5] + 1j * d1[:, 6]))
        - 20 * np.log10(np.abs(d2[:, 5] + 1j * d2[:, 6])),
        "r",
    )
    plt.ylabel(r"$\Delta$ magnitude [dB]")
    plt.legend(["ambient", "hot"])

    plt.subplot(3, 2, 2)
    plt.plot(
        d1[:, 0],
        (180 / np.pi) * np.unwrap(np.angle(d1[:, 3] + 1j * d1[:, 4]))
        - (180 / np.pi) * np.unwrap(np.angle(d2[:, 3] + 1j * d2[:, 4])),
        "b",
    )
    plt.plot(
        d1[:, 0],
        (180 / np.pi) * np.unwrap(np.angle(d1[:, 5] + 1j * d1[:, 6]))
        - (180 / np.pi) * np.unwrap(np.angle(d2[:, 5] + 1j * d2[:, 6])),
        "r",
    )
    plt.ylabel(r"$\Delta$ phase [deg]")

    plt.subplot(3, 2, 3)
    plt.plot(
        d1[:, 0],
        20 * np.log10(np.abs(d1[:, 7] + 1j * d1[:, 8]))
        - 20 * np.log10(np.abs(d2[:, 7] + 1j * d2[:, 8])),
        "b",
    )
    plt.plot(
        d1[:, 0],
        20 * np.log10(np.abs(d1[:, 9] + 1j * d1[:, 10]))
        - 20 * np.log10(np.abs(d2[:, 9] + 1j * d2[:, 10])),
        "r",
    )
    plt.ylabel(r"$\Delta$ magnitude [dB]")
    plt.legend(["open", "shorted"])
    plt.ylim([-0.006, 0.06])

    plt.subplot(3, 2, 4)
    plt.plot(
        d1[:, 0],
        (180 / np.pi) * np.unwrap(np.angle(d1[:, 7] + 1j * d1[:, 8]))
        - (180 / np.pi) * np.unwrap(np.angle(d2[:, 7] + 1j * d2[:, 8])),
        "b",
    )
    plt.plot(
        d1[:, 0],
        (180 / np.pi) * np.unwrap(np.angle(d1[:, 9] + 1j * d1[:, 10]))
        - (180 / np.pi) * np.unwrap(np.angle(d2[:, 9] + 1j * d2[:, 10])),
        "r",
    )
    plt.ylabel(r"$\Delta$ phase [deg]")

    plt.subplot(3, 2, 5)
    plt.plot(
        d1[:, 0],
        20 * np.log10(np.abs(d1[:, 17] + 1j * d1[:, 18]))
        - 20 * np.log10(np.abs(d2[:, 17] + 1j * d2[:, 18])),
        "b",
    )
    plt.plot(
        d1[:, 0],
        20 * np.log10(np.abs(d1[:, 19] + 1j * d1[:, 20]))
        - 20 * np.log10(np.abs(d2[:, 19] + 1j * d2[:, 20])),
        "r",
    )
    plt.ylabel(r"$\Delta$ magnitude [dB]")
    plt.xlabel("frequency [MHz]")
    plt.legend(["sim2", "sim3"])
    plt.ylim([-0.02, 0.05])

    plt.subplot(3, 2, 6)
    plt.plot(
        d1[:, 0],
        (180 / np.pi) * np.unwrap(np.angle(d1[:, 17] + 1j * d1[:, 18]))
        - (180 / np.pi) * np.unwrap(np.angle(d2[:, 17] + 1j * d2[:, 18])),
        "b",
    )
    plt.plot(
        d1[:, 0],
        (180 / np.pi) * np.unwrap(np.angle(d1[:, 19] + 1j * d1[:, 20]))
        - (180 / np.pi) * np.unwrap(np.angle(d2[:, 19] + 1j * d2[:, 20])),
        "r",
    )
    plt.ylabel(r"$\Delta$ phase [deg]")
    plt.xlabel("frequency [MHz]")

    plt.savefig(path_plot_save + "fig2.pdf", bbox_inches="tight")
    plt.close()
    plt.close()


def plot_mid_band_GHA_14_16():
    fb, tb, rb, wb, sb = tools.level4_integration(3, [14, 15], 147, 200, 60, 135, 5)

    plt.figure(1, figsize=[9, 4])

    plt.plot(fb[wb > 0], rb[wb > 0], "b")

    sig = dm.signal_model("tanh", [-0.7, 79, 19.5, 7.5, 4.5], fb)
    model = mdl.fit_polynomial_fourier("LINLOG", fb, tb - sig, 5, Weights=wb)
    mb = mdl.model_evaluate("LINLOG", model[0], fb)

    plt.plot(fb[wb > 0], (tb - sig - mb)[wb > 0] - 0.8, "b")

    plt.xticks(np.arange(60, 135, 10), fontsize=12)
    plt.ylim([-1.1, 0.4])
    plt.yticks(
        [-1, -0.8, -0.6, -0.2, 0, 0.2],
        ["-0.2", "0", "0.2", "-0.2", "0", "0.2"],
        fontsize=12,
    )
    plt.xlabel(r"$\nu$ [MHz]", fontsize=14)
    plt.ylabel(r"$T_b$ [K]", fontsize=14)

    plt.savefig("/home/raul/Desktop/GHA_14-16hr.pdf", bbox_inches="tight")


def plot_signal_residuals(f_low, f_high, A21, model_type, Ntotal):
    f = np.arange(f_low, f_high + 1, 1)
    A = 1500
    f0 = 75
    model_fg = A * (f / f0) ** ((-2.5) + 0.1 * np.log(f / f0))

    sig = dm.signal_model("exp", [A21, 79, 19, 7], f)

    total = model_fg + sig

    pc = mdl.fit_polynomial_fourier(model_type, f / 200, total, Ntotal)
    m = mdl.model_evaluate(model_type, pc[0], f / 200)
    r = total - m

    return f, r, model_fg


def plot_foreground_analysis():
    path_file = (
        "/media/raul/DATA/EDGES_vol2/mid_band/spectra/level4/case_nominal_50"
        "-150MHz_LNA2_a2_h2_o2_s1_sim2_all_lc_yes_bc_20min/foreground_fits.hdf5"
    )

    fref, fit2, fit3, fit4, fit5 = io.level4_foreground_fits_read(path_file)

    plt.figure(1)

    fit2[:, :, 2] = fit2[:, :, 2] + 17.76
    fit3[:, :, 2] = fit3[:, :, 2] + 17.76
    fit4[:, :, 2] = fit4[:, :, 2] + 17.76
    fit5[:, :, 2] = fit5[:, :, 2] + 17.76

    fit2[:, :, 2][fit2[:, :, 2] > 24] = fit2[:, :, 2][fit2[:, :, 2] > 24] - 24
    fit3[:, :, 2][fit3[:, :, 2] > 24] = fit3[:, :, 2][fit3[:, :, 2] > 24] - 24
    fit4[:, :, 2][fit4[:, :, 2] > 24] = fit4[:, :, 2][fit4[:, :, 2] > 24] - 24
    fit5[:, :, 2][fit5[:, :, 2] > 24] = fit5[:, :, 2][fit5[:, :, 2] > 24] - 24

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


def beam_correction_check(f_low, f_high):
    bb = np.genfromtxt(
        edges_folder + "mid_band/calibration/beam_factors/raw/mid_band_50"
        "-200MHz_90deg_alan1_haslam_2.5_2.62_reffreq_100MHz_data.txt"
    )
    ff = np.genfromtxt(
        edges_folder + "mid_band/calibration/beam_factors/raw/mid_band_50"
        "-200MHz_90deg_alan1_haslam_2.5_2.62_reffreq_100MHz_freq.txt"
    )

    b = bb[:, (ff >= f_low) & (ff <= f_high)]
    f = ff[(ff >= f_low) & (ff <= f_high)]

    plt.figure(1)
    plt.imshow(b, interpolation="none", aspect="auto", vmin=0.99, vmax=1.01)
    plt.colorbar()

    f_t, lst_t, bf_t = beams.beam_factor_table_read(
        edges_folder + "mid_band/calibration/beam_factors/table/table_hires_mid_band_50"
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
