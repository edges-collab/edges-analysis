import os
from os.path import exists

import h5py
import numpy as np
from edges_cal import modelling as mdl

from . import tools
from .io import data_selection, level3read, level4read

edges_folder = ""  # TODO: remove


def _get_level4_case(case):
    _level4_cases = {
        2: "calibration_2019_10_no_ground_loss_no_beam_corrections",
        3: "case_nominal_50-150MHz_no_ground_loss_no_beam_corrections",
        406: "case_nominal_50-150MHz_LNA1_a2_h2_o2_s1_sim2",
        5: "case_nominal_14_14_terms_55-150MHz_no_ground_loss_no_beam_corrections",
        501: "case_nominal_50-150MHz_LNA2_a2_h2_o2_s1_sim2_all_lc_yes_bc",
        502: "case_nominal_50-150MHz_LNA2_a2_h2_o2_s1_sim2_all_lc_yes_bc_20min",
    }
    folder = edges_folder + "mid_band/spectra/level4/{}/"

    if case not in _level4_cases:
        raise ValueError("case must be one of {}".format(_level4_cases.keys()))

    case = _level4_cases[case]
    folder = folder.format(case)

    return case, folder


def level4_binned_residuals(case, f_low, f_high, output_file_name_hdf5):
    case, folder = _get_level4_case(case)

    f, p, r, w, index, gha, yy = level4read(folder + case + ".hdf5")
    save_folder = folder + "binned_residuals/"

    # Compute the residuals
    start = True
    for i in range(len(r[:, 0, 0])):
        for k in range(len(r[0, :, 0])):

            if np.sum(w[i, k, :]) > 100:

                model_k = mdl.model_evaluate("LINLOG", p[i, k, :], f / 200)
                t_k = r[i, k, :] + model_k
                w_k = w[i, k, :]

                fc = f[(f > f_low) & (f < f_high)]
                tc_k = t_k[(f > f_low) & (f < f_high)]
                wc_k = w_k[(f > f_low) & (f < f_high)]

                Nfg = 5 if 6 <= gha[k] <= 17 else 6
                pc_k = mdl.fit_polynomial_fourier(
                    "LINLOG", fc / 200, tc_k, Nfg, Weights=wc_k
                )
                mc_k = mdl.model_evaluate("LINLOG", pc_k[0], fc / 200)

                rc_k = tc_k - mc_k

                fb, rb, wb, sb = spectral_binning_number_of_samples(fc, rc_k, wc_k)

                if start:
                    binned_residuals = np.zeros((len(r), len(r[0]), len(fb)))
                    binned_weights = np.zeros_like(binned_residuals)
                    binned_stddev = np.zeros_like(binned_residuals)
                    start = False

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
    else:
        good_day_dct = {
            1: np.concatenate((np.arange(148, 160), np.arange(161, 220))),
            10: np.concatenate(
                (
                    np.arange(148, 168),
                    np.arange(177, 194),
                    np.arange(197, 202),
                    np.arange(205, 216),
                )
            ),
            11: np.arange(187, 202),
            12: np.arange(147, 150),
            13: np.array([147, 149, 157, 159]),
            14: np.arange(148, 183),
            15: np.concatenate(
                (np.arange(140, 183), np.arange(187, 206), np.arange(210, 300))
            ),
            23: np.arange(148, 300),
            6: np.arange(147, 300),
            7: np.concatenate(
                (
                    np.arange(147, 153),
                    np.arange(160, 168),
                    np.arange(174, 202),
                    np.arange(210, 300),
                )
            ),
            8: np.concatenate(
                (np.arange(147, 151), np.arange(160, 168), np.arange(174, 300))
            ),
            9: np.concatenate(
                (
                    np.arange(147, 153),
                    np.arange(160, 168),
                    np.arange(174, 194),
                    np.arange(197, 202),
                    np.arange(210, 300),
                )
            ),
        }
        good_days = good_day_dct[GHA]

    return good_days[(good_days >= first_day) & (good_days <= last_day)]


def level4_integration(case, GHA_list, first_day, last_day, f_low, f_high, Nfg):
    case, folder = _get_level4_case(case)

    f, p, r, w, index, gha, yy = level4read(folder + case + ".hdf5")

    # Compute the residuals
    p_new, r_new, w_new = [], [], []
    for GHA in GHA_list:
        good_days = level4_good_days_GHA(GHA, first_day, last_day)

        for i in range(len(yy)):
            if yy[i, 1] in good_days:
                p_new.append(p[i, GHA])
                r_new.append(r[i, GHA])
                w_new.append(w[i, GHA])

    avr, avw = tools.spectral_averaging(np.array(r_new), np.array(w_new))
    avp = np.mean(p_new, axis=0)

    model = mdl.model_evaluate("LINLOG", avp, f / 200)
    avt = avr + model

    mask = (f > f_low) & (f < f_high)
    fc = f[mask]
    tc = avt[mask]
    wc = avw[mask]

    model_type = "LINLOG"
    pc = mdl.fit_polynomial_fourier(model_type, fc / 200, tc, Nfg, Weights=wc)
    mc = mdl.model_evaluate(model_type, pc[0], fc / 200)
    rc = tc - mc
    fb, rb, wb, sb = spectral_binning_number_of_samples(fc, rc, wc)
    mb = mdl.model_evaluate(model_type, pc[0], fb / 200)
    tb = rb + mb
    tb[wb == 0] = 0

    return fb, tb, rb, wb, sb


def spectrum_fit(f, t, w, n_poly=5, f1_low=60, f1_high=65, f2_low=95, f2_high=140):
    fc = f[((f >= f1_low) & (f <= f1_high)) | ((f >= f2_low) & (f <= f2_high))]
    tc = t[((f >= f1_low) & (f <= f1_high)) | ((f >= f2_low) & (f <= f2_high))]
    wc = w[((f >= f1_low) & (f <= f1_high)) | ((f >= f2_low) & (f <= f2_high))]

    pc = mdl.fit_polynomial_fourier("LINLOG", fc / 200, tc, n_poly, Weights=wc)
    m = mdl.model_evaluate("LINLOG", pc[0], f / 200)
    r = t - m

    return f, r


def level4_foreground_fits(
    case, f_low, f_high, f_norm, output_file_name_hdf5="foreground_fits.hdf5"
):
    folder, case = _get_level4_case(case)
    save_folder = folder + "binned_residuals/"

    f, p, r, w, index, gha, yy = level4read(folder + case + ".hdf5")

    # Computing the foreground parameters
    # -----------------------------------
    fits = [np.zeros((len(r), len(r[0]), n)) for n in range(11, 15)]

    for i, yyi in enumerate(yy):
        # Loading data
        path_level3_file = (
            edges_folder
            + "mid_band/spectra/level3/case_nominal_50-150MHz_LNA2_a2_h2_o2_s1_sim2_all_lc_yes_bc"
            "/2018_{}_00.hdf5".format(yyi)
        )

        xf, xt, xp, xr, xw, xrms, xtp, xm = level3read(path_level3_file)
        print("----------------------------------------------")

        for k in range(len(r[0, :, 0])):
            IX = np.arange(0, len(index[0, 0, :]))
            new_meta = xm[IX[index[i, k, :] == 1], :]

            # If there are enough data points
            if (np.sum(w[i, k, :]) > 100) and (len(new_meta) > 0):

                av_sunel = np.mean(new_meta[:, 6])
                av_moonel = np.mean(new_meta[:, 8])

                fb, rb, wb, sb = spectral_binning_number_of_samples(
                    f, r[i, k, :], w[i, k, :]
                )

                mb = mdl.model_evaluate("LINLOG", p[i, k, :], fb / 200)
                tb = rb + mb

                mask = (fb > f_low) & (fb < f_high)
                fc = fb[mask]
                tc = tb[mask]
                wc = wb[mask]
                sc = sb[mask]

                for i_fg, Nfg in enumerate([2, 3, 4, 5]):

                    logf = np.log(fc / f_norm)
                    logt = np.log(tc)

                    par = np.polyfit(logf[wc > 0], logt[wc > 0], Nfg - 1)
                    logm = np.polyval(par, logf)
                    m = np.exp(logm)

                    rms = np.std((tc - m)[wc > 0])

                    chi2 = np.sum(((tc - m)[wc > 0] / sc[wc > 0]) ** 2)
                    LEN = len(fc[wc > 0])
                    BIC = chi2 + Nfg * np.log(LEN)

                    fits[i_fg][i, k, 0] = yyi[0]
                    fits[i_fg][i, k, 1] = yyi[1]
                    fits[i_fg][i, k, 2] = gha[k]
                    fits[i_fg][i, k, 3] = av_sunel
                    fits[i_fg][i, k, 4] = av_moonel

                    fits[i_fg][i, k, 5] = np.exp(par[-1])
                    fits[i_fg][i, k, 6 : 6 + Nfg - 1] = par[:-1][::-1]

                    fits[i_fg][i, k, -4] = rms
                    fits[i_fg][i, k, -3] = chi2
                    fits[i_fg][i, k, -2] = LEN
                    fits[i_fg][i, k, -1] = BIC

                    print(yyi[1])

    # Save to file.
    with h5py.File(save_folder + output_file_name_hdf5, "w") as hf:
        hf.create_dataset("fref", data=np.array([f_norm]))
        hf.create_dataset("fit2", data=fits[0])
        hf.create_dataset("fit3", data=fits[1])
        hf.create_dataset("fit4", data=fits[2])
        hf.create_dataset("fit5", data=fits[3])

    return fits


def _get_model_resid(
    f,
    t,
    w,
    n_fg,
    mask=None,
    f_low=-np.inf,
    f_high=np.inf,
    f_norm=200,
    model_type="LINLOG",
):
    if mask is None:
        mask = (f >= f_low) & (f <= f_high)

    par = mdl.fit_polynomial_fourier(
        model_type, f[mask] / f_norm, t[mask], n_fg, Weights=w[mask]
    )
    return t[mask] - par[1], w[mask], f[mask]


def integrated_residuals_GHA(file_data, f_low, f_high, n_fg):
    d = np.genfromtxt(file_data)

    fx = d[:, 0]
    tx = d[:, 1::2]
    wx = d[:, 2::2]

    return spectra_to_residuals(fx, tx, wx, f_low, f_high, n_fg)


def spectra_to_residuals(fx, tx_2D, wx_2D, f_low, f_high, n_fg, model_type="LINLOG"):
    mask = (fx >= f_low) & (fx <= f_high)

    r_all, w_all = [], []
    for t, w in zip(tx_2D, wx_2D):
        r, w, f = _get_model_resid(fx, t, w, n_fg, mask=mask, model_type=model_type)

        r_all.append(r)
        w_all.append(w)

    return f, np.array(r_all), np.array(w_all)


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

        indx = data_selection(
            m,
            sun_el_max=SUN_EL_max,
            moon_el_max=MOON_EL_max,
            amb_hum_max=200,
            min_receiver_temp=0,
            max_receiver_temp=100,
        )

        if len(indx):
            avr, avw = spectral_averaging(r[indx], w[indx])
            fb, rb, wb = spectral_binning_number_of_samples(f, avr, avw, nsamples=64)

            avp = np.mean(p[indx], axis=0)

            mb = mdl.model_evaluate("LINLOG", avp, fb / 200)
            tb = mb + rb

            rb_x, wb_x, fb_x = _get_model_resid(
                fb, tb, wb, Nfg, f_low=FLOW, f_high=FHIGH
            )

            if flag == 0:
                rb_x_all = np.zeros((len(LST_boundaries) - 1, len(fb_x)))
                wb_x_all = np.zeros((len(LST_boundaries) - 1, len(fb_x)))

            rb_x_all[i, :] = rb_x
            wb_x_all[i, :] = wb_x

            flag += 1

    if flag == 0:
        fb, rb, wb = spectral_binning_number_of_samples(f, r[0], w[0], nsamples=64)
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


def average_level3_mid_band(case, sun_el_max=90, moon_el_max=90):
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

    rx_all, wx_all, px_all = [], [], []
    rb_all, wb_all = [], []
    for i in range(10):
        print(str(i + 1) + " of " + str(lf))
        f, t, p, r, w, rms, m = level3read(path_files + list_files[i], print_key=False)

        indx = data_selection(
            m,
            sun_el_max=sun_el_max,
            moon_el_max=moon_el_max,
            amb_hum_max=200,
            min_receiver_temp=0,
            max_receiver_temp=100,
        )

        if len(indx):
            avr, avw = spectral_averaging(r[indx], w[indx])
            fb, rb, wb = spectral_binning_number_of_samples(f, avr, avw, nsamples=64)

            rx_all.append(r[indx])
            wx_all.append(w[indx])
            px_all.append(p[indx])

            rb_all.append(rb)
            wb_all.append(wb)

    return (
        fb,
        np.array(rb_all),
        np.array(wb_all),
        list_files,
        f,
        np.array(rx_all),
        np.array(wx_all),
        np.array(px_all),
    )


def spectral_binning_number_of_samples(freq_in, spectrum_in, weights_in, nsamples=64):
    # TODO: perhaps replace with simple convolution
    i = 0
    av_fr, av_sp, av_we, av_std = [], [], [], []

    for j in range(len(freq_in)):
        if i == 0:
            sum_fre, sum_num, sum_den = 0, 0, 0
            samples_spectrum_in = []
            samples_weights_in = []

        elif i < nsamples:
            sum_fre = sum_fre + freq_in[j]
            sum_num = sum_num + spectrum_in[j] * weights_in[j]
            sum_den = sum_den + weights_in[j]

            av_fr_temp = sum_fre / nsamples
            av_sp_temp = sum_num / sum_den if sum_den > 0 else 0
            if weights_in[j] > 0:
                samples_spectrum_in = np.append(samples_spectrum_in, spectrum_in[j])
                samples_weights_in = np.append(samples_weights_in, weights_in[j])

        if i < (nsamples - 1):
            i += 1
        else:
            if len(samples_spectrum_in) <= 1:
                std_of_the_mean = 1e6

            if len(samples_spectrum_in) > 1:
                sample_variance = np.sum(
                    ((samples_spectrum_in - av_sp_temp) ** 2) * samples_weights_in
                ) / np.sum(samples_weights_in)

                # sample_variance = (np.std(samples_spectrum_in))**2
                std_of_the_mean = np.sqrt(sample_variance / len(samples_spectrum_in))

            av_fr.append(av_fr_temp)
            av_sp.append(av_sp_temp)
            av_we.append(sum_den)
            av_std.append(std_of_the_mean)

            i = 0

    return np.array(av_fr), np.array(av_sp), np.array(av_we), np.array(av_std)


def weighted_mean(data, weights, axis=0):
    """A careful weighted mean where zero-weights don't error.

    In this function, if the total weight is zero, zero is returned.

    Parameters
    ----------
    data : array-like
        The data over which the weighted mean is to be taken.
    weights : array-like
        Same shape as data, giving the weights of each datum.
    axis : int, optional
        The axis over which to take the mean.

    Returns
    -------
    array-like :
        The weighted mean over `axis`, where elements with zero total weight are
        set to zero.
    """
    sum = np.sum(data * weights, axis=axis)
    weights = np.sum(weights, axis=axis)

    av = np.where(weights > 0, sum / weights, 0)
    return av, weights


def weighted_standard_deviation(av, data, std, axis=0):
    return np.sqrt(weighted_mean((data - av) ** 2, 1 / std ** 2, axis=axis)[0])


spectral_averaging = weighted_mean
