from os import listdir, makedirs
from os.path import exists, dirname

import h5py
import numpy as np
import yaml
from edges_cal import modelling as mdl
from edges_cal import receiver_calibration_func as rcf
from edges_cal import s11_correction as s11c
from edges_cal import xrfi as rfi
from edges_cal.cal_coefficients import EdgesFrequencyRange, HotLoadCorrection
from edges_io.io import SwitchingState
from matplotlib import pyplot as plt

from . import beams, filters, io, loss, plots
from . import s11 as s11m
from . import tools
from .levels import level1_to_level2, level2_to_level3
from ..simulation import data_models as dm
from ..config import config


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
    # TODO: this function is only for lab data

    prefix = (
        config["edges_folder"]
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
        index_good &= ~rfi.cleaning_sweep(
            d["temp"],
            d["weights"],
            window_width=int(4 / (ff[1] - ff[0])),
            n_poly=4,
            n_bootstrap=20,
            n_sigma=n_sigma,
            flip=True,
        )

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
    # TODO: might be old? not very general.

    f, pz, rz, wz, index, gha, ydz = io.level4read(
        config["edges_folder"]
        + "mid_band/spectra/level4/case_nominal/case_nominal.hdf5"
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
    )

    f_low = 57
    f_high = 120
    n_fg = 5
    Nsp = 1

    ll = len(px)
    j = 0
    for i in range(ll):
        if not (ydx[i, 0] in bad_days[:, 0]) and (ydx[i, 1] in bad_days[:, 1]):
            mx = mdl.model_evaluate("LINLOG", px[i, 0, :], f / 200)
            tx = mx + rx[i, 0, :]

            fy = f[(f >= f_low) & (f <= f_high)]
            ty = tx[(f >= f_low) & (f <= f_high)]
            wy = wx[i, 0, (f >= f_low) & (f <= f_high)]

            p = mdl.fit_polynomial_fourier("LINLOG", fy / 200, ty, n_fg, Weights=wy)
            my = mdl.model_evaluate("LINLOG", p[0], fy / 200)
            ry = ty - my

            fb, rb, wb, sb = tools.spectral_binning_number_of_samples(
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
            else:
                tb_all[j] = tb
                rb_all[j] = rb
                wb_all[j] = wb
                sb_all[j] = sb
                yd_all[j] = ydx[i, :]

            j += 1

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
        else:
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
    save,
    filename_flag,
):
    """
    Important high level function that averages level4 field data. Used consistently.
    TODO: make this into an actual script. Make it a little bit more general.
    The idea here is just to average together a bunch of level data over many days
    and GHA. Can choose the days.
    """
    cases = {
        100: "rcv18_sw18_nominal_GHA_6_18hr",
        101: "rcv18_sw18_nominal_GHA_every_1hr",
        21: "rcv18_ant19_nominal",
        22: "rcv18_ant19_every_1hr_GHA",
    }

    in_file = f"{config['edges_folder']}mid_band/spectra/level4/{cases[case]}/{cases[case]}.hdf5"
    save_path = dirname(in_file).replace("level4", "level5")
    f, px, rx, wx, index, gha, ydx = io.level4read(in_file)

    if case == 101:
        save_spectrum = (
            "integrated_spectrum_rcv18_sw18_every_1hr_GHA" + filename_flag + ".txt"
        )
    elif case == 22:
        save_spectrum = (
            "integrated_spectrum_rcv18_ant19_every_1hr_GHA" + filename_flag + ".txt"
        )

    # Produce integrated spectrum
    for i in range(len(index_GHA)):
        keep_index = filters.daily_nominal_filter("mid_band", case, index_GHA[i], ydx)

        mask = (keep_index == 1) & (
            ((ydx[:, 1] >= day_min1) & (ydx[:, 1] <= day_max1))
            | ((ydx[:, 1] >= day_min2) & (ydx[:, 1] <= day_max2))
        )

        p_i = px[mask, index_GHA[i]]
        r_i = rx[mask, index_GHA[i]]
        w_i = wx[mask, index_GHA[i]]

        if i == 0:
            p = np.copy(p_i)
            r = np.copy(r_i)
            w = np.copy(w_i)
        else:
            p = np.vstack((p, p_i))
            r = np.vstack((r, r_i))
            w = np.vstack((w, w_i))

    avp = np.mean(p, axis=0)
    m = mdl.model_evaluate("LINLOG", avp, f / 200)

    avr, avw = tools.spectral_averaging(r, w)
    flags = rfi.cleaning_sweep(
        avr,
        avw,
        window_width=int(3 / (f[1] - f[0])),
        n_poly=2,
        n_bootstrap=20,
        n_sigma=3,
    )
    rr = np.where(flags, 0, avr)
    wr = np.where(flags, 0, avw)

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

    # Saving spectrum
    if save and day_range != "daily":
        np.savetxt(
            save_path + save_spectrum,
            np.array([fb, tb, wb, sb]).T,
            header="freq [MHz], temp [K], weight [K], std dev [K]",
        )

    # Computing residuals for plot
    mask = (fb >= f_low) & (fb <= f_high)
    fx, tx, wx, sx = fb[mask], tb[mask], wb[mask], sb[mask]
    ft, tt, st = fx[wx > 0], tx[wx > 0], sx[wx > 0]

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


def integrated_half_hour_level4_many(band, case, GHA_starts=[(13, 1), (14, 0)]):
    # TODO: this should be a script.
    discard = {
        (6, 0): [146, 164, 167, 169],
        (6, 1): [146, 147, 174, 179, 181, 198, 211, 215],
        (7, 0): [146, 147, 157, 166],
        (7, 1): [146, 159],
        (8, 0): [146, 151, 159],
        (8, 1): [146, 159],
        (9, 0): [146, 151, 152, 157, 159, 163, 185],
        (9, 1): [
            146,
            157,
            159,
            167,
            196,
            ([149, 150, 152, 163], (104.5, 110)),
            ([150, 160, 161, 162, 166], (129, 135)),
            (None, (134.5, 140.5)),
        ],
        (10, 0): [152, 157, 166, 159, 196],
        (10, 1): [
            174,
            176,
            204,
            218,
            (None, 101.52, 101.53),
            (None, (102.5, 102.53)),
            (None, (153.02, 153.04)),
            (None, (111.7, 115.4)),
            (None, (121.47, 121.55)),
            (None, (146.5, 148)),
            (None, (150, 150.5)),
            (None, (105.72, 105.74)),
            (None, (106.05, 106.15)),
            (None, (106.42, 106.55)),
        ],
        (11, 0): [
            149,
            165,
            176,
            204,
            ([151, 161], (129, 135)),
            (None, (109, 114.2)),
            (None, (105.72, 105.74)),
            (None, (106.05, 106.15)),
            (None, (106.42, 106.55)),
            (None, (138, 138.4)),
        ],
        (11, 1): [175, 176, 177, 200, 204, 216, (149, (110, 118)), (None, (137, 140))],
        (12, 0): [146, 147, 170, 175, 176, 193, 195, 204, 205, 220],
        (12, 1): [146, 170, 176, 185, 195, 198, 204, 220],
        (13, 0): [152, 174, 176, 182, 185, 195, 204, 208, 214],
        (13, 1): [151, 163, 164, 176, 185, 187, 189, 192, 195, 200, 208, 215, 219],
        (14, 0): [176, 184, 185, 193, 199, 208, 210],
        (14, 1): [166, 174, 177, 185, 199, 200, 201, 208],
        (15, 0): [150, 157, 178, 182, 185, 187, 198, 208],
        (15, 1): [185],
        (16, 0): [184],
        (16, 1): [191, 192],
        (17, 0): [184, 186, 192, 216],
        (17, 1): [186, 192, 196],
        (18, 0): [192, 197],
        (18, 1): [192, 209, 211],
    }

    for i, GHA_start in enumerate(GHA_starts):
        discarded_days = discard[GHA_start]

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
        ) = tools.integrate_level4_half_hour(
            band, case, 140, 170, discarded_days, GHA_start=GHA_start
        )

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

    flags = rfi.cleaning_sweep(
        avr,
        avw,
        window_width=int(3 / (f[1] - f[0])),
        n_poly=2,
        n_bootstrap=20,
        n_sigma=2.5,
    )
    avrn = np.where(flags, 0, avr)
    avwn = np.where(flags, 0, avw)

    fb, rb, wbn = tools.spectral_binning_number_of_samples(f, avrn, avwn)
    tbn = mdl.model_evaluate("LINLOG", avp, fb / 200) + rb

    return f, avr, avw, avt, avp, fb, tbn, wbn


def season_integrated_spectra_GHA(
    band, case, new_gha_edges=np.arange(0, 25, 2), data_save_name_flag="2hr"
):
    """
    Take a whole season of spectra and produce season averages in GHA, with
    changeable range. Not being as strict for cleaning.

    # TODO: move to scripts.
    """
    data_save_path = config["edges_folder"] + f"{band}/spectra/level5/case{case}/"

    # Loading level4 data
    f, p_all, r_all, w_all, gha_edges, yd = io.level4read(
        config["edges_folder"] + f"{band}/spectra/level4/case{case}/case{case}.hdf5"
    )

    # Creating intermediate 1hr-average arrays
    pr_all = np.zeros((len(gha_edges) - 1, len(p_all[0, 0, :])))
    rr_all = np.zeros((len(gha_edges) - 1, len(f)))
    wr_all = np.zeros((len(gha_edges) - 1, len(f)))

    # Looping over every original GHA edges
    for j in range(len(gha_edges) - 1):
        # Looping over day
        index_good = [
            i
            for i, (y, d) in enumerate(yd)
            if filters.one_hour_filter(band, case, y, d, gha_edges[j])
        ]

        # Selecting good parameters and spectra
        pp = p_all[index_good, j]
        rr = r_all[index_good, j]
        ww = w_all[index_good, j]

        # Average parameters and spectra
        avp = np.mean(pp, axis=0)
        avr, avw = tools.weighted_mean(rr, ww)

        # RFI cleaning of 1-hr season average spectra
        flags = rfi.cleaning_sweep(
            avr,
            avw,
            window_width=int(3 / (f[1] - f[0])),
            n_poly=2,
            n_bootstrap=20,
            n_sigma=2.5,
        )
        avr_no_rfi = np.where(flags, 0, avr)
        avw_no_rfi = np.where(flags, 0, avw)

        # Storing season 1hr-average spectra
        pr_all[j] = avp
        rr_all[j] = avr_no_rfi
        wr_all[j] = avw_no_rfi

        # Frequency binning
        fb, rb, wb = tools.spectral_binning_number_of_samples(f, avr_no_rfi, avw_no_rfi)
        mb = mdl.model_evaluate("LINLOG", avp, fb / 200)
        tb = mb + rb
        tb[wb == 0] = 0

        # Storing binned average spectra
        if j == 0:
            tb_all = np.zeros((len(gha_edges) - 1, len(fb)))
            wb_all = np.zeros((len(gha_edges) - 1, len(fb)))
        else:
            tb_all[j] = tb
            wb_all[j] = wb

    # Averaging data within new GHA edges
    for j in range(len(new_gha_edges) - 1):
        new_gha_start = new_gha_edges[j]
        new_gha_end = new_gha_edges[j + 1]

        flag = True
        for i in range(len(gha_edges) - 1):
            if (
                new_gha_start < new_gha_end
                and ((gha_edges[i] >= new_gha_start) and (gha_edges[i] < new_gha_end))
            ) or ((gha_edges[i] >= new_gha_start) or (gha_edges[i] < new_gha_end)):
                if flag:
                    px_all = pr_all[i]
                    rx_all = rr_all[i]
                    wx_all = wr_all[i]
                    flag = False
                else:
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

        flags = rfi.cleaning_sweep(
            avrx,
            avwx,
            window_width=int(3 / (f[1] - f[0])),
            n_poly=2,
            n_bootstrap=20,
            n_sigma=2.5,
        )
        avrx_no_rfi = np.where(flags, 0, avrx)
        avwx_no_rfi = np.where(flags, 0, avwx)

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
            data_save_path + f"case{case}_frequency.txt", fb, header="Frequency [MHz]."
        )
        np.savetxt(
            data_save_path + f"case{case}_1hr_gha_edges.txt",
            gha_edges,
            header="GHA edges of integrated spectra from 0hr to 23hr in steps of 1hr [hr].",
        )
        np.savetxt(
            data_save_path + f"case{case}_1hr_temperature.txt",
            tb_all,
            header="Rows correspond to different GHAs from 0hr to 23hr in steps of 1hr. Columns correspond to frequency.",
        )
        np.savetxt(
            data_save_path + f"case{case}_1hr_weights.txt",
            wb_all,
            header="Rows correspond to different GHAs from 0hr to 23hr in steps of 1hr. Columns correspond to frequency.",
        )
        np.savetxt(
            data_save_path + f"case{case}_{data_save_name_flag}_gha_edges.txt",
            new_gha_edges,
            header="GHA edges of integrated spectra [hr].",
        )
        np.savetxt(
            data_save_path + f"case{case}_{data_save_name_flag}_temperature.txt",
            tbx_all,
            header="Rows correspond to different GHAs. Columns correspond to frequency.",
        )
        np.savetxt(
            data_save_path + f"case{case}_{data_save_name_flag}_weights.txt",
            wbx_all,
            header="Rows correspond to different GHAs. Columns correspond to frequency.",
        )

    return fb, tb_all, wb_all, tbx_all, wbx_all


def batch_level1_to_level2(
    band, batch_file=None, path=None, omit_days=None, day_indx=12
):
    if batch_file:
        with open(batch_file) as fl:
            days = yaml.load(fl, Loader=yaml.FullLoader)["days"]
    else:
        fl_list = sorted(listdir(path))
        days = [fl[day_indx : day_indx + 6] for fl in fl_list]
        if omit_days:
            days = [d for d in days if int(d[:3]) not in omit_days]

    for year, day in days:
        level1_to_level2(band, year, day)


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


def batch_level2_to_level3(
    band,
    flag_folder,
    first_day,
    last_day,
    receiver_cal_file=2,
    antenna_s11_Nfit=13,
    balun_correction=True,
    ground_correction=True,
    beam_correction=True,
    antenna_correction=True,
    f_low=55,
    beam_correction_case=1,
    f_high=120,
    n_fg=5,
    bad_files=None,
):

    # Listing files to be processed
    path_files = config["edges_folder"] + f"{band}/spectra/level2/"
    files = sorted(listdir(path_files))

    if bad_files is None and band == "mid_band":
        bad_files = dirname(__file__) + "data/bad_files_mid_band_2to3.yaml"

    if isinstance(bad_files, str):
        with open(bad_files) as fl:
            bad_files = yaml.load(fl, Loader=yaml.FullLoader)["bad_files"]

    bad_files = bad_files or []

    files = [f for f in files if f not in bad_files]

    # Processing files
    for fl in files:
        day = int(fl[5:8])

        if (day >= first_day) & (day <= last_day):
            level2_to_level3(
                band,
                fl,
                flag_folder=flag_folder,
                rcv_file=(
                    config["edges_folder"]
                    + "mid_band/calibration/receiver_calibration/receiver1/2018_01_25C/results/"
                    + CALFILES[receiver_cal_file]
                ),
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


def plots_beam_gain_derivative():
    # Low-Band, Blade on 10m x 10m GP
    # TODO: this is fairly general and could be moved to beams.py
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
