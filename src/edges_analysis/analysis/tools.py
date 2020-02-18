import os
from os.path import exists

import h5py
import numpy as np
from edges_cal import modelling as mdl

from . import tools
from .io import data_selection, level3read, level4_binned_read, level4read

edges_folder = ""  # TODO: remove


def level4_binned_residuals(case, FLOW, FHIGH, output_file_name_hdf5):
    if case == 2:
        f, p, r, w, index, gha, yy = level4read(
            edges_folder
            + "mid_band/spectra/level4/calibration_2019_10_no_ground_loss_no_beam_corrections"
            "/calibration_2019_10_no_ground_loss_no_beam_corrections.hdf5"
        )
        save_folder = (
            edges_folder
            + "mid_band/spectra/level4/calibration_2019_10_no_ground_loss_no_beam_corrections"
            "/binned_residuals/"
        )
    elif case == 3:
        f, p, r, w, index, gha, yy = level4read(
            edges_folder
            + "mid_band/spectra/level4/case_nominal_50-150MHz_no_ground_loss_no_beam_corrections"
            "/case_nominal_50-150MHz_no_ground_loss_no_beam_corrections.hdf5"
        )
        save_folder = (
            edges_folder + "mid_band/spectra/level4/case_nominal_50"
            "-150MHz_no_ground_loss_no_beam_corrections"
            "/binned_residuals/"
        )
    elif case == 406:
        f, p, r, w, index, gha, yy = level4read(
            edges_folder
            + "mid_band/spectra/level4/case_nominal_50-150MHz_LNA1_a2_h2_o2_s1_sim2"
            "/case_nominal_50-150MHz_LNA1_a2_h2_o2_s1_sim2.hdf5"
        )
        save_folder = (
            edges_folder
            + "mid_band/spectra/level4/case_nominal_50-150MHz_LNA1_a2_h2_o2_s1_sim2"
            "/binned_residuals/"
        )
    elif case == 5:
        f, p, r, w, index, gha, yy = level4read(
            edges_folder + "mid_band/spectra/level4/case_nominal_14_14_terms_55"
            "-150MHz_no_ground_loss_no_beam_corrections/case_nominal_14_14_terms_55"
            "-150MHz_no_ground_loss_no_beam_corrections.hdf5"
        )
        save_folder = (
            edges_folder + "mid_band/spectra/level4/case_nominal_14_14_terms_55"
            "-150MHz_no_ground_loss_no_beam_corrections/binned_residuals/"
        )
    elif case == 501:
        f, p, r, w, index, gha, yy = level4read(
            edges_folder
            + "mid_band/spectra/level4/case_nominal_50-150MHz_LNA2_a2_h2_o2_s1_sim2_all_lc_yes_bc"
            "/case_nominal_50-150MHz_LNA2_a2_h2_o2_s1_sim2_all_lc_yes_bc.hdf5"
        )
        save_folder = (
            edges_folder + "mid_band/spectra/level4/case_nominal_50"
            "-150MHz_LNA2_a2_h2_o2_s1_sim2_all_lc_yes_bc"
            "/binned_residuals/"
        )
    # Computing the residuals
    # -----------------------
    start = 0
    for i in range(len(r[:, 0, 0])):
        for k in range(len(r[0, :, 0])):

            if np.sum(w[i, k, :]) > 100:

                model_k = mdl.model_evaluate("LINLOG", p[i, k, :], f / 200)
                t_k = r[i, k, :] + model_k
                w_k = w[i, k, :]

                fc = f[(f > FLOW) & (f < FHIGH)]
                tc_k = t_k[(f > FLOW) & (f < FHIGH)]
                wc_k = w_k[(f > FLOW) & (f < FHIGH)]

                Nfg = 5 if gha[k] >= 6 and gha[k] <= 17 else 6
                pc_k = mdl.fit_polynomial_fourier(
                    "LINLOG", fc / 200, tc_k, Nfg, Weights=wc_k
                )
                mc_k = mdl.model_evaluate("LINLOG", pc_k[0], fc / 200)

                rc_k = tc_k - mc_k

                fb, rb, wb, sb = spectral_binning_number_of_samples(fc, rc_k, wc_k)

                if start == 0:
                    binned_residuals = np.zeros(
                        (len(r[:, 0, 0]), len(r[0, :, 0]), len(fb))
                    )
                    binned_weights = np.zeros(
                        (len(r[:, 0, 0]), len(r[0, :, 0]), len(fb))
                    )
                    binned_stddev = np.zeros(
                        (len(r[:, 0, 0]), len(r[0, :, 0]), len(fb))
                    )
                    start = 1

                binned_residuals[i, k, :] = rb
                binned_weights[i, k, :] = wb
                binned_stddev[i, k, :] = sb

    # Save
    # ----
    with h5py.File(save_folder + output_file_name_hdf5, "w") as hf:
        hf.create_dataset("frequency", data=fb)
        hf.create_dataset("residuals", data=binned_residuals)
        hf.create_dataset("weights", data=binned_weights)
        hf.create_dataset("stddev", data=binned_stddev)
        hf.create_dataset("gha_edges", data=gha)
        hf.create_dataset("year_day", data=yy)

    return fb, binned_residuals


def level4_good_days_GHA(GHA, first_day, last_day):
    if GHA in [0, 2, 3, 4, 5, 16, 17, 18, 19, 20, 21, 22]:
        good_days = np.arange(140, 300, 1)
    elif GHA == 1:
        good_days = np.concatenate((np.arange(148, 160, 1), np.arange(161, 220, 1)))
    elif GHA == 10:
        good_days = np.concatenate(
            (
                np.arange(148, 168, 1),
                np.arange(177, 194, 1),
                np.arange(197, 202, 1),
                np.arange(205, 216, 1),
            )
        )
    elif GHA == 11:
        good_days = np.arange(187, 202, 1)
    elif GHA == 12:
        good_days = np.arange(147, 150, 1)
    elif GHA == 13:
        good_days = np.array([147, 149, 157, 159])
    elif GHA == 14:
        good_days = np.arange(148, 183, 1)
    elif GHA == 15:
        good_days = np.concatenate(
            (np.arange(140, 183, 1), np.arange(187, 206, 1), np.arange(210, 300, 1))
        )
    elif GHA == 23:
        good_days = np.arange(148, 300, 1)
    elif GHA == 6:
        good_days = np.arange(147, 300, 1)
    elif GHA == 7:
        good_days = np.concatenate(
            (
                np.arange(147, 153, 1),
                np.arange(160, 168, 1),
                np.arange(174, 202, 1),
                np.arange(210, 300, 1),
            )
        )
    elif GHA == 8:
        good_days = np.concatenate(
            (np.arange(147, 151, 1), np.arange(160, 168, 1), np.arange(174, 300, 1))
        )
    elif GHA == 9:
        good_days = np.concatenate(
            (
                np.arange(147, 153, 1),
                np.arange(160, 168, 1),
                np.arange(174, 194, 1),
                np.arange(197, 202, 1),
                np.arange(210, 300, 1),
            )
        )
    else:
        raise ValueError("GHA must be between 0-23.")
    return good_days[(good_days >= first_day) & (good_days <= last_day)]


def level4_integration(case, GHA_list, first_day, last_day, FLOW, FHIGH, Nfg):
    if case == 2:
        f, p, r, w, index, gha, yy = level4read(
            edges_folder
            + "mid_band/spectra/level4/calibration_2019_10_no_ground_loss_no_beam_corrections"
            "/calibration_2019_10_no_ground_loss_no_beam_corrections.hdf5"
        )
    elif case == 3:
        f, p, r, w, index, gha, yy = level4read(
            edges_folder
            + "mid_band/spectra/level4/case_nominal_50-150MHz_no_ground_loss_no_beam_corrections"
            "/case_nominal_50-150MHz_no_ground_loss_no_beam_corrections.hdf5"
        )
    elif case == 406:
        f, p, r, w, index, gha, yy = level4read(
            edges_folder
            + "mid_band/spectra/level4/case_nominal_50-150MHz_LNA1_a2_h2_o2_s1_sim2"
            "/case_nominal_50-150MHz_LNA1_a2_h2_o2_s1_sim2.hdf5"
        )
    elif case == 5:
        f, p, r, w, index, gha, yy = level4read(
            edges_folder + "mid_band/spectra/level4/case_nominal_14_14_terms_55"
            "-150MHz_no_ground_loss_no_beam_corrections/case_nominal_14_14_terms_55"
            "-150MHz_no_ground_loss_no_beam_corrections.hdf5"
        )
    elif case == 501:
        f, p, r, w, index, gha, yy = level4read(
            edges_folder
            + "mid_band/spectra/level4/case_nominal_50-150MHz_LNA2_a2_h2_o2_s1_sim2_all_lc_yes_bc"
            "/case_nominal_50-150MHz_LNA2_a2_h2_o2_s1_sim2_all_lc_yes_bc.hdf5"
        )
    # Computing the residuals
    # -----------------------
    start = 0

    for GHA in GHA_list:
        GHA_index = GHA
        good_days = level4_good_days_GHA(GHA_index, first_day, last_day)

        for i in range(len(yy)):
            if yy[i, 1] in good_days:

                print([GHA_index, yy[i, 1]])
                if start == 0:
                    p_new = p[i, GHA_index, :]
                    r_new = r[i, GHA_index, :]
                    w_new = w[i, GHA_index, :]

                    start = 1

                elif start == 1:
                    p_new = np.vstack((p_new, p[i, GHA_index, :]))
                    r_new = np.vstack((r_new, r[i, GHA_index, :]))
                    w_new = np.vstack((w_new, w[i, GHA_index, :]))

    avr, avw = tools.spectral_averaging(r_new, w_new)
    avp = np.mean(p_new, axis=0)

    model = mdl.model_evaluate("LINLOG", avp, f / 200)
    avt = avr + model

    fc = f[(f > FLOW) & (f < FHIGH)]
    tc = avt[(f > FLOW) & (f < FHIGH)]
    wc = avw[(f > FLOW) & (f < FHIGH)]

    model_type = "LINLOG"
    pc = mdl.fit_polynomial_fourier(model_type, fc / 200, tc, Nfg, Weights=wc)
    mc = mdl.model_evaluate(model_type, pc[0], fc / 200)
    rc = tc - mc
    fb, rb, wb, sb = spectral_binning_number_of_samples(fc, rc, wc)
    mb = mdl.model_evaluate(model_type, pc[0], fb / 200)
    tb = rb + mb
    tb[wb == 0] = 0

    return fb, tb, rb, wb, sb


def spectrum_fit(f, t, w, Nfg=5, F1_LOW=60, F1_HIGH=65, F2_LOW=95, F2_HIGH=140):
    fc = f[((f >= F1_LOW) & (f <= F1_HIGH)) | ((f >= F2_LOW) & (f <= F2_HIGH))]
    tc = t[((f >= F1_LOW) & (f <= F1_HIGH)) | ((f >= F2_LOW) & (f <= F2_HIGH))]
    wc = w[((f >= F1_LOW) & (f <= F1_HIGH)) | ((f >= F2_LOW) & (f <= F2_HIGH))]

    pc = mdl.fit_polynomial_fourier("LINLOG", fc / 200, tc, Nfg, Weights=wc)
    m = mdl.model_evaluate("LINLOG", pc[0], f / 200)
    r = t - m

    return f, r


def level4_foreground_fits(case, FLOW, FHIGH, FNORM):
    if case == 501:
        f, p, r, w, index, gha, yy = level4read(
            edges_folder
            + "mid_band/spectra/level4/case_nominal_50-150MHz_LNA2_a2_h2_o2_s1_sim2_all_lc_yes_bc"
            "/case_nominal_50-150MHz_LNA2_a2_h2_o2_s1_sim2_all_lc_yes_bc.hdf5"
        )

    if case == 502:
        f, p, r, w, index, gha, yy = level4read(
            edges_folder + "mid_band/spectra/level4/case_nominal_50"
            "-150MHz_LNA2_a2_h2_o2_s1_sim2_all_lc_yes_bc_20min/case_nominal_50"
            "-150MHz_LNA2_a2_h2_o2_s1_sim2_all_lc_yes_bc_20min.hdf5"
        )

    # Computing the foreground parameters
    # -----------------------------------
    fit2 = np.zeros((len(r[:, 0, 0]), len(r[0, :, 0]), 11))
    fit3 = np.zeros((len(r[:, 0, 0]), len(r[0, :, 0]), 12))
    fit4 = np.zeros((len(r[:, 0, 0]), len(r[0, :, 0]), 13))
    fit5 = np.zeros((len(r[:, 0, 0]), len(r[0, :, 0]), 14))

    for i in range(len(r[:, 0, 0])):
        # Loading data
        if int(yy[i, 1]) == 147:
            path_level3_file = (
                edges_folder
                + "mid_band/spectra/level3/case_nominal_50-150MHz_LNA2_a2_h2_o2_s1_sim2_all_lc_yes_bc"
                "/2018_147_00.hdf5"
            )
        else:
            path_level3_file = (
                edges_folder
                + "mid_band/spectra/level3/case_nominal_50-150MHz_LNA2_a2_h2_o2_s1_sim2_all_lc_yes_bc"
                "/2018_" + str(int(yy[i, 1])) + "_00.hdf5"
            )

        xf, xt, xp, xr, xw, xrms, xtp, xm = level3read(path_level3_file)
        print("----------------------------------------------")

        for k in range(len(r[0, :, 0])):
            IX = np.arange(0, len(index[0, 0, :]))
            new_meta = xm[IX[index[i, k, :] == 1], :]

            # If there are enough data points
            if (np.sum(w[i, k, :]) > 100) and (len(new_meta) > 0):

                av_SUNEL = np.mean(new_meta[:, 6])
                av_MOONEL = np.mean(new_meta[:, 8])

                fb, rb, wb, sb = spectral_binning_number_of_samples(
                    f, r[i, k, :], w[i, k, :]
                )

                mb = mdl.model_evaluate("LINLOG", p[i, k, :], fb / 200)
                tb = rb + mb

                fc = fb[(fb > FLOW) & (fb < FHIGH)]
                tc = tb[(fb > FLOW) & (fb < FHIGH)]
                wc = wb[(fb > FLOW) & (fb < FHIGH)]
                sc = sb[(fb > FLOW) & (fb < FHIGH)]

                for Nfg in [2, 3, 4, 5]:

                    logf = np.log(fc / FNORM)
                    logt = np.log(tc)

                    par = np.polyfit(logf[wc > 0], logt[wc > 0], Nfg - 1)
                    logm = np.polyval(par, logf)
                    m = np.exp(logm)

                    rms = np.std((tc - m)[wc > 0])

                    chi2 = np.sum(((tc - m)[wc > 0] / sc[wc > 0]) ** 2)
                    LEN = len(fc[wc > 0])
                    BIC = chi2 + Nfg * np.log(LEN)

                    if Nfg == 2:
                        fit2[i, k, 0] = yy[i, 0]
                        fit2[i, k, 1] = yy[i, 1]
                        fit2[i, k, 2] = gha[k]
                        fit2[i, k, 3] = av_SUNEL
                        fit2[i, k, 4] = av_MOONEL

                        fit2[i, k, 5] = np.exp(par[1])
                        fit2[i, k, 6] = par[0]

                        fit2[i, k, 7] = rms
                        fit2[i, k, 8] = chi2
                        fit2[i, k, 9] = LEN
                        fit2[i, k, 10] = BIC

                    if Nfg == 3:
                        fit3[i, k, 0] = yy[i, 0]
                        fit3[i, k, 1] = yy[i, 1]
                        fit3[i, k, 2] = gha[k]
                        fit3[i, k, 3] = av_SUNEL
                        fit3[i, k, 4] = av_MOONEL

                        fit3[i, k, 5] = np.exp(par[2])
                        fit3[i, k, 6] = par[1]
                        fit3[i, k, 7] = par[0]

                        fit3[i, k, 8] = rms
                        fit3[i, k, 9] = chi2
                        fit3[i, k, 10] = LEN
                        fit3[i, k, 11] = BIC

                    if Nfg == 4:
                        fit4[i, k, 0] = yy[i, 0]
                        fit4[i, k, 1] = yy[i, 1]
                        fit4[i, k, 2] = gha[k]
                        fit4[i, k, 3] = av_SUNEL
                        fit4[i, k, 4] = av_MOONEL

                        fit4[i, k, 5] = np.exp(par[3])
                        fit4[i, k, 6] = par[2]
                        fit4[i, k, 7] = par[1]
                        fit4[i, k, 8] = par[0]

                        fit4[i, k, 9] = rms
                        fit4[i, k, 10] = chi2
                        fit4[i, k, 11] = LEN
                        fit4[i, k, 12] = BIC

                    if Nfg == 5:
                        fit5[i, k, 0] = yy[i, 0]
                        fit5[i, k, 1] = yy[i, 1]
                        fit5[i, k, 2] = gha[k]
                        fit5[i, k, 3] = av_SUNEL
                        fit5[i, k, 4] = av_MOONEL

                        fit5[i, k, 5] = np.exp(par[4])
                        fit5[i, k, 6] = par[3]
                        fit5[i, k, 7] = par[2]
                        fit5[i, k, 8] = par[1]
                        fit5[i, k, 9] = par[0]

                        fit5[i, k, 10] = rms
                        fit5[i, k, 11] = chi2
                        fit5[i, k, 12] = LEN
                        fit5[i, k, 13] = BIC

                    print(str(int(yy[i, 1])))


def integrated_residuals_GHA(file_data, flow, fhigh, Nfg):
    d = np.genfromtxt(file_data)

    fx = d[:, 0]

    for i in range(int((len(d[0, :]) - 1) / 2)):
        index_t = 2 * (i + 1) - 1
        index_w = 2 * (i + 1)
        tx = d[:, index_t]
        wx = d[:, index_w]

        f = fx[(fx >= flow) & (fx <= fhigh)]
        t = tx[(fx >= flow) & (fx <= fhigh)]
        w = wx[(fx >= flow) & (fx <= fhigh)]

        par = mdl.fit_polynomial_fourier("LINLOG", f / 200, t, Nfg, Weights=w)

        r = t - par[1]

        if i == 0:
            r_all = np.copy(r)
            w_all = np.copy(w)

        elif i > 0:
            r_all = np.vstack((r_all, r))
            w_all = np.vstack((w_all, w))

    return f, r_all, w_all


def spectra_to_residuals(fx, tx_2D, wx_2D, flow, fhigh, Nfg, model_type="LINLOG"):
    f = fx[(fx >= flow) & (fx <= fhigh)]
    t_2D = tx_2D[:, (fx >= flow) & (fx <= fhigh)]
    w_2D = wx_2D[:, (fx >= flow) & (fx <= fhigh)]

    for i in range(len(t_2D[:, 0])):
        t = t_2D[i, :]
        w = w_2D[i, :]

        par = mdl.fit_polynomial_fourier(model_type, f / 200, t, Nfg, Weights=w)

        r = t - par[1]

        if i == 0:
            r_all = np.copy(r)
            w_all = np.copy(w)

        elif i > 0:
            r_all = np.vstack((r_all, r))
            w_all = np.vstack((w_all, w))

    return f, r_all, w_all


def daily_residuals_LST(
    file_name,
    LST_boundaries=np.arange(0, 25, 2),
    FLOW=60,
    FHIGH=150,
    Nfg=5,
    SUN_EL_max=90,
    MOON_EL_max=90,
):
    flag_folder = "nominal_60_160MHz_fullcal"

    # Listing files to be processed
    path_files = "/EDGES/spectra/level3/mid_band/" + flag_folder + "/"

    f, t, p, r, w, rms, m = level3read(path_files + file_name)

    flag = 0
    for i in range(len(LST_boundaries) - 1):

        IX = data_selection(
            m,
            LST_1=LST_boundaries[i],
            LST_2=LST_boundaries[i + 1],
            sun_el_max=SUN_EL_max,
            moon_el_max=MOON_EL_max,
            amb_hum_max=200,
            min_receiver_temp=0,
            max_receiver_temp=100,
        )

        if len(IX) > 0:
            RX = r[IX, :]
            WX = w[IX, :]
            PX = p[IX, :]

            avr, avw = spectral_averaging(RX, WX)
            fb, rb, wb = spectral_binning_number_of_samples(f, avr, avw, nsamples=64)

            avp = np.mean(PX, axis=0)

            mb = mdl.model_evaluate("LINLOG", avp, fb / 200)
            tb = mb + rb

            fb_x = fb[(fb >= FLOW) & (fb <= FHIGH)]
            tb_x = tb[(fb >= FLOW) & (fb <= FHIGH)]
            wb_x = wb[(fb >= FLOW) & (fb <= FHIGH)]

            par_x = mdl.fit_polynomial_fourier(
                "LINLOG", fb_x / 200, tb_x, Nfg, Weights=wb_x
            )
            rb_x = tb_x - par_x[1]

            if flag == 0:
                rb_x_all = np.zeros((len(LST_boundaries) - 1, len(fb_x)))
                wb_x_all = np.zeros((len(LST_boundaries) - 1, len(fb_x)))

            rb_x_all[i, :] = rb_x
            wb_x_all[i, :] = wb_x

            flag += 1

    if flag == 0:
        fb, rb, wb = spectral_binning_number_of_samples(
            f, r[0, :], w[0, :], nsamples=64
        )
        fb_x = fb[(fb >= FLOW) & (fb <= FHIGH)]
        rb_x_all = np.zeros((len(LST_boundaries) - 1, len(fb_x)))
        wb_x_all = np.zeros((len(LST_boundaries) - 1, len(fb_x)))

    return fb_x, rb_x_all, wb_x_all


def spectral_fit_two_ranges(model_type, fx, tx, wx, sx, F1L, F1H, F2L, F2H, Nfg):
    f = fx[wx > 0]
    t = tx[wx > 0]
    s = sx[wx > 0]

    index_all = np.arange(0, len(f))
    index_sel = index_all[((f >= F1L) & (f <= F1H)) | ((f >= F2L) & (f <= F2H))]

    ff = f[index_sel]
    tt = t[index_sel]
    ss = s[index_sel]

    if model_type == "LINLOG":
        pp = mdl.fit_polynomial_fourier("LINLOG", ff, tt, Nfg, Weights=1 / (ss ** 2))
        model = mdl.model_evaluate("LINLOG", pp[0], fx)

    elif model_type == "LOGLOG":
        pp = np.polyfit(np.log(ff), np.log(tt), Nfg - 1)
        log_model = np.polyval(pp, np.log(fx))
        model = np.exp(log_model)

    else:
        raise ValueError("model_type must be LINLOG or LOGLOG")
    return model


def average_level3_mid_band(case, LST_1=0, LST_2=24, sun_el_max=90, moon_el_max=90):
    if case == 1:
        flag_folder = "nominal_60_160MHz"
    elif case == 3:
        flag_folder = "nominal_60_160MHz_fullcal"
    else:
        raise ValueError("case must be 1 or 3")

    # Listing files to be processed
    path_files = "/EDGES/spectra/level3/mid_band/" + flag_folder + "/"
    list_files = os.listdir(path_files)
    list_files.sort()
    lf = len(list_files)

    flag = 0

    for i in range(10):  # (lf):
        print(str(i + 1) + " of " + str(lf))
        f, t, p, r, w, rms, m = level3read(path_files + list_files[i], print_key="no")

        if i == 0:
            RX_all = np.zeros((0, len(f)))
            WX_all = np.zeros((0, len(f)))
            PX_all = np.zeros((0, len(p[0, :])))

        IX = data_selection(
            m,
            LST_1=LST_1,
            LST_2=LST_2,
            sun_el_max=sun_el_max,
            moon_el_max=moon_el_max,
            amb_hum_max=200,
            min_receiver_temp=0,
            max_receiver_temp=100,
        )

        if len(IX) > 0:
            RX = r[IX, :]
            WX = w[IX, :]
            PX = p[IX, :]

            avr, avw = spectral_averaging(RX, WX)

            fb, rb, wb = spectral_binning_number_of_samples(f, avr, avw, nsamples=64)

            RX_all = np.vstack((RX_all, RX))
            WX_all = np.vstack((WX_all, WX))
            PX_all = np.vstack((PX_all, PX))

            if flag == 0:
                rb_all = np.zeros((lf, len(fb)))
                wb_all = np.zeros((lf, len(fb)))

                flag = 1

            rb_all[i, :] = rb
            wb_all[i, :] = wb

    return fb, rb_all, wb_all, list_files, f, RX_all, WX_all, PX_all


def spectral_binning_number_of_samples(freq_in, spectrum_in, weights_in, nsamples=64):
    # TODO: perhaps replace with simple convolution
    flag_start = 0
    i = 0
    for j in range(len(freq_in)):
        if i == 0:
            sum_fre = 0
            sum_num = 0
            sum_den = 0
            samples_spectrum_in = []
            samples_weights_in = []

        if (i >= 0) and (i < nsamples):

            sum_fre = sum_fre + freq_in[j]
            # if spectrum_in[j]  0:
            sum_num = sum_num + spectrum_in[j] * weights_in[j]
            sum_den = sum_den + weights_in[j]

            av_fr_temp = sum_fre / nsamples
            av_sp_temp = sum_num / sum_den if sum_den > 0 else 0
            if weights_in[j] > 0:
                samples_spectrum_in = np.append(samples_spectrum_in, spectrum_in[j])
                samples_weights_in = np.append(samples_weights_in, weights_in[j])

        if i < (nsamples - 1):
            i += 1

        elif i == (nsamples - 1):

            if len(samples_spectrum_in) <= 1:
                std_of_the_mean = 1e6

            if len(samples_spectrum_in) > 1:
                sample_variance = np.sum(
                    ((samples_spectrum_in - av_sp_temp) ** 2) * samples_weights_in
                ) / np.sum(samples_weights_in)

                # sample_variance = (np.std(samples_spectrum_in))**2
                std_of_the_mean = np.sqrt(sample_variance / len(samples_spectrum_in))

            if flag_start == 0:
                av_fr = av_fr_temp
                av_sp = av_sp_temp
                av_we = sum_den
                av_std = std_of_the_mean
                flag_start = 1

            elif flag_start > 0:
                av_fr = np.append(av_fr, av_fr_temp)
                av_sp = np.append(av_sp, av_sp_temp)
                av_we = np.append(av_we, sum_den)
                av_std = np.append(av_std, std_of_the_mean)

            i = 0

    return av_fr, av_sp, av_we, av_std


def weighted_mean(data_array, weights_array):
    # TODO: replace with simpler numpy function (if anything still uses it)
    # Number of frequency channels
    lf = len(data_array[0, :])

    # Number of spectra
    ls = len(data_array[:, 0])

    # Initializing arrays
    av = np.zeros(lf)
    w = np.zeros(lf)

    for k in range(lf):
        num = 0
        den = 0
        for j in range(ls):
            if weights_array[j, k] > 0:  # (data_array[j,k] > 0) and
                num += data_array[j, k] * weights_array[j, k]
                den += weights_array[j, k]

        if num != 0 and den != 0:
            av[k] = num / den
            w[k] = den

    return av, w


def weighted_standard_deviation(av, data_array, std_array):
    # TODO: replace with numpy function if anything still uses it.
    ls = len(data_array[0, :])
    la = len(data_array[:, 0])

    std_sq = np.zeros(ls)

    for k in range(ls):
        num = 0
        den = 0
        for j in range(la):
            num += ((data_array[j, k] - av[k]) / std_array[j, k]) ** 2
            den += 1 / (std_array[j, k] ** 2)

        if num != 0 and den != 0:
            std_sq[k] = num / den

    return np.sqrt(std_sq)


spectral_averaging = weighted_mean
