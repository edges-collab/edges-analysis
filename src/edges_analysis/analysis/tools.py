import os
from typing import Tuple
from os.path import join, dirname

import h5py
import numpy as np
from edges_cal import modelling as mdl, xrfi as rfi

from . import tools, io
from .io import level3read, level4read
from .filters import time_filter_auxiliary
from ..config import config
from . import filters


def _get_level4_case(case):
    _level4_cases = {
        2: "calibration_2019_10_no_ground_loss_no_beam_corrections",
        3: "case_nominal_50-150MHz_no_ground_loss_no_beam_corrections",
        406: "case_nominal_50-150MHz_LNA1_a2_h2_o2_s1_sim2",
        5: "case_nominal_14_14_terms_55-150MHz_no_ground_loss_no_beam_corrections",
        501: "case_nominal_50-150MHz_LNA2_a2_h2_o2_s1_sim2_all_lc_yes_bc",
        502: "case_nominal_50-150MHz_LNA2_a2_h2_o2_s1_sim2_all_lc_yes_bc_20min",
    }
    folder = config["edges_folder"] + "mid_band/spectra/level4/{}/"

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

                n_fg = 5 if 6 <= gha[k] <= 17 else 6
                pc_k = mdl.fit_polynomial_fourier(
                    "LINLOG", fc / 200, tc_k, n_fg, Weights=wc_k
                )
                mc_k = mdl.model_evaluate("LINLOG", pc_k[0], fc / 200)

                rc_k = tc_k - mc_k

                fb, rb, wb, sb = average_in_frequency(rc_k, fc, wc_k, n_samples=16)

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


def level4_integration(case, GHA_list, first_day, last_day, f_low, f_high, n_fg):
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
    pc = mdl.fit_polynomial_fourier(model_type, fc / 200, tc, n_fg, Weights=wc)
    mc = mdl.model_evaluate(model_type, pc[0], fc / 200)
    rc = tc - mc
    fb, rb, wb, sb = average_in_frequency(rc, fc, wc, n_samples=16)
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
        path_level3_file = config[
            "edges_folder"
        ] + "mid_band/spectra/level3/case_nominal_50-150MHz_LNA2_a2_h2_o2_s1_sim2_all_lc_yes_bc" "/2018_{}_00.hdf5".format(
            yyi
        )

        xf, xt, xp, xr, xw, xrms, xtp, xm = level3read(path_level3_file)

        for k in range(len(r[0, :, 0])):
            IX = np.arange(0, len(index[0, 0, :]))
            new_meta = xm[IX[index[i, k, :] == 1], :]

            # If there are enough data points
            if (np.sum(w[i, k, :]) > 100) and (len(new_meta) > 0):

                av_sunel = np.mean(new_meta[:, 6])
                av_moonel = np.mean(new_meta[:, 8])

                fb, rb, wb, sb = average_in_frequency(
                    r[i, k, :], f, w[i, k, :], n_samples=16
                )

                mb = mdl.model_evaluate("LINLOG", p[i, k, :], fb / 200)
                tb = rb + mb

                mask = (fb > f_low) & (fb < f_high)
                fc = fb[mask]
                tc = tb[mask]
                wc = wb[mask]
                sc = sb[mask]

                for i_fg, n_fg in enumerate([2, 3, 4, 5]):

                    logf = np.log(fc / f_norm)
                    logt = np.log(tc)

                    par = np.polyfit(logf[wc > 0], logt[wc > 0], n_fg - 1)
                    logm = np.polyval(par, logf)
                    m = np.exp(logm)

                    rms = np.std((tc - m)[wc > 0])

                    chi2 = np.sum(((tc - m)[wc > 0] / sc[wc > 0]) ** 2)
                    LEN = len(fc[wc > 0])
                    BIC = chi2 + n_fg * np.log(LEN)

                    fits[i_fg][i, k, 0] = yyi[0]
                    fits[i_fg][i, k, 1] = yyi[1]
                    fits[i_fg][i, k, 2] = gha[k]
                    fits[i_fg][i, k, 3] = av_sunel
                    fits[i_fg][i, k, 4] = av_moonel

                    fits[i_fg][i, k, 5] = np.exp(par[-1])
                    fits[i_fg][i, k, 6 : 6 + n_fg - 1] = par[:-1][::-1]

                    fits[i_fg][i, k, -4] = rms
                    fits[i_fg][i, k, -3] = chi2
                    fits[i_fg][i, k, -2] = LEN
                    fits[i_fg][i, k, -1] = BIC

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
    f_low=60,
    f_high=150,
    n_fg=5,
    SUN_EL_max=90,
    MOON_EL_max=90,
):
    flag_folder = "nominal_60_160MHz_fullcal"

    # Listing files to be processed
    path_files = "/EDGES/spectra/level3/mid_band/" + flag_folder + "/"

    f, t, p, r, w, rms, m = level3read(path_files + file_name)

    flag = 0
    for i in range(len(LST_boundaries) - 1):

        indx = time_filter_auxiliary(
            m,
            sun_el_max=SUN_EL_max,
            moon_el_max=MOON_EL_max,
            amb_hum_max=200,
            min_receiver_temp=0,
            max_receiver_temp=100,
        )

        if len(indx):
            avr, avw = spectral_averaging(r[indx], w[indx])
            fb, rb, wb, _ = average_in_frequency(avr, f, avw, n_samples=64)

            avp = np.mean(p[indx], axis=0)

            mb = mdl.model_evaluate("LINLOG", avp, fb / 200)
            tb = mb + rb

            rb_x, wb_x, fb_x = _get_model_resid(
                fb, tb, wb, n_fg, f_low=f_low, f_high=f_high
            )

            if flag == 0:
                rb_x_all = np.zeros((len(LST_boundaries) - 1, len(fb_x)))
                wb_x_all = np.zeros((len(LST_boundaries) - 1, len(fb_x)))

            rb_x_all[i, :] = rb_x
            wb_x_all[i, :] = wb_x

            flag += 1

    if flag == 0:
        fb, rb, wb, _ = average_in_frequency(r[0], f, w[0], n_samples=64)
        fb_x = fb[(fb >= f_low) & (fb <= f_high)]
        rb_x_all = np.zeros((len(LST_boundaries) - 1, len(fb_x)))
        wb_x_all = np.zeros((len(LST_boundaries) - 1, len(fb_x)))

    return fb_x, rb_x_all, wb_x_all


def spectral_fit_two_ranges(model_type, fx, tx, wx, sx, F1L, F1H, F2L, F2H, n_fg):
    f = fx[wx > 0]
    t = tx[wx > 0]
    s = sx[wx > 0]

    index_all = np.arange(0, len(f))
    index_sel = index_all[((f >= F1L) & (f <= F1H)) | ((f >= F2L) & (f <= F2H))]

    ff = f[index_sel]
    tt = t[index_sel]
    ss = s[index_sel]

    if model_type == "LINLOG":
        pp = mdl.fit_polynomial_fourier("LINLOG", ff, tt, n_fg, Weights=1 / (ss ** 2))
        model = mdl.model_evaluate("LINLOG", pp[0], fx)

    elif model_type == "LOGLOG":
        pp = np.polyfit(np.log(ff), np.log(tt), n_fg - 1)
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

    rx_all, wx_all, px_all = [], [], []
    rb_all, wb_all = [], []
    for i in range(10):
        f, t, p, r, w, rms, m = level3read(path_files + list_files[i])

        indx = time_filter_auxiliary(
            m,
            sun_el_max=sun_el_max,
            moon_el_max=moon_el_max,
            amb_hum_max=200,
            min_receiver_temp=0,
            max_receiver_temp=100,
        )

        if len(indx):
            avr, avw = spectral_averaging(r[indx], w[indx])
            fb, rb, wb = average_in_frequency(avr, f, avw, n_samples=64)

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


def average_in_frequency(
    spectrum: [list, np.ndarray],
    freq: [list, np.ndarray, None] = None,
    weights: [list, np.ndarray, None] = None,
    resolution: [float, None] = None,
    n_samples: [int, None] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Average a spectrum, with weights, in frequency.

    The average is optionally taken within bins along the frequency axis.

    Parameters
    ----------
    spectrum : array-like
        The spectrum to average. Must be 1D.
    freq : array-like, optional
        The frequencies along which to average. If provided, must be the same shape
        as ``spectrum``. Must be provided if either ``resolution`` or ``n_samples``
        is provided.
    weights : array-like, optional
        The weights of the weighted averaged. If provided, same shape as ``spectrum``.
        If not provided, all weights are considered to be one.
    resolution : float, optional
        The (frequency) resolution with which to perform the average, in same units
        as ``freq``. For example, if an array of frequencies with resolution 0.1 MHz is
        passed in, and ``resolution`` is 0.2, the output array will contain half the
        number of bins. Default is to average the whole array.
    n_samples : int, optional
        The number of samples to average into each frequency bin. Used only if
        ``resolution`` is not provided. By default, averages all frequencies.

    Returns
    -------
    f : array
        An array with length determined automatically by the routine, giving the
        mean frequency in each output bin.
    s : array
        Array of same length as ``f`` containing the weighted-average spectrum
    w : array
        Array of same length as ``f`` containing the total weight in each bin.
    std : array
        Array of same length as ``f`` contianing the standard deviation about the mean
        for each bin.
    Examples
    --------
    >>> freq = np.linspace(0.1, 1, 10)
    >>> spectrum = [0, 2] * 5
    >>> f, s, w = average_in_frequency(spectrum, freq=freq, resolution=0.2)
    >>> f
    [0.15, 0.35, 0.55, 0.75, 0.95]
    >>> s
    [1, 1, 1, 1, 1]
    >>> w
    [1, 1, 1, 1, 1]

    """
    if resolution is not None:
        n_samples = int((freq[1] - freq[0]) / resolution)

    if resolution or n_samples and freq is None:
        raise ValueError(
            "You must provide freq if resolution or n_samples is provided!"
        )

    nf = len(spectrum)

    if resolution is None and n_samples is None:
        n_samples = nf

    if freq is None:
        freq = np.ones(nf)

    if weights is None:
        weights = np.ones(weights)

    mod = nf % n_samples
    if mod:
        last_f = freq[-mod:]
        last_s = spectrum[-mod:]
        last_w = weights[-mod:]

    f = freq[: nf // n_samples]
    s = spectrum[: nf // n_samples]
    w = weights[: nf // n_samples]

    f = np.reshape(f, (-1, n_samples))
    s = np.reshape(s, (-1, n_samples))
    w = np.reshape(w, (-1, n_samples))

    f = np.mean(f, axis=1)
    s_tmp, w_tmp = weighted_mean(s, weights=w, axis=1)
    std = weighted_standard_deviation(
        np.atleast_2d(s_tmp), s, std=np.sqrt(1 / w), axis=-1
    )
    s = s_tmp
    w = w_tmp

    if mod:
        f = np.concatenate((f, np.mean(last_f)))
        ss, ww = weighted_mean(last_s, last_w)
        s = np.concatenate((s, ss))
        w = np.concatenate((w, ww))
        std = np.concatenate(
            (std, weighted_standard_deviation(ss, last_s, std=np.sqrt(1 / ww)))
        )

    return f, s, w, std


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


def integrate_level4_half_hour(
    band, case, first_day, last_day, discarded_days, GHA_start=13.5
):
    if band == "mid_band":
        f, p_all, r_all, w_all, gha, yd = io.level4read(
            f"/home/raul/DATA/EDGES/mid_band/spectra/level4/case{case}/case{case}.hdf5"
        )

        index = np.arange(0, len(gha))
        IX = int(index[gha == GHA_start])

        p = p_all[:, IX]
        r = r_all[:, IX]
        w = w_all[:, IX]

        day = yd[:, 1]
    else:
        raise ValueError("band must be mid_band")

    flag = False
    for disc in discarded_days:
        if hasattr(disc, "__len__"):
            # The discard is frequency-dependent
            indx = disc[0] or slice(None)
            w[indx, disc[1]] = 0

    # Remove non-wholly-discarded days
    discarded_days = [d for d in discarded_days if isinstance(d, int)]

    for i, (d, ww, pp, rr) in enumerate(day, w, p, r):
        if np.sum(ww) == 0 or d in discarded_days or d < first_day or d > last_day:
            continue

        t = mdl.model_evaluate("LINLOG", pp, f / 200) + rr
        par = mdl.fit_polynomial_fourier("LINLOG", f / 200, t, 4, Weights=ww)
        r_raw = t - par[1]
        w_raw = ww

        fb, rb, wb, _ = average_in_frequency(r_raw, f, w_raw, n_samples=16)

        if not flag:
            rr_all = np.copy(r_raw)
            wr_all = np.copy(w_raw)
            pr_all = np.copy(par[0])

            rb_all = np.copy(rb)
            wb_all = np.copy(wb)
            d_all = np.copy(d)
            flag = True
        else:
            rr_all = np.vstack((rr_all, r_raw))
            wr_all = np.vstack((wr_all, w_raw))
            pr_all = np.vstack((pr_all, par[0]))

            rb_all = np.vstack((rb_all, rb))
            wb_all = np.vstack((wb_all, wb))
            d_all = np.append(d_all, d)

    avrn, avwn = spectral_averaging(rr_all, wr_all)
    avp = np.mean(pr_all, axis=0)

    # For the 10.5 and 11 averages, DO NOT USE THIS CLEANING. ONLY use the 2.5sigma filter in the
    # INTEGRATED 10.5-11.5 spectrum, AFTER integration
    flags = rfi.cleaning_sweep(
        avrn,
        avwn,
        window_width=int(3 / (f[1] - f[0])),
        n_poly=2,
        n_bootstrap=20,
        n_sigma=3.0,
    )
    avrn[flags] = 0
    avwn[flags] = 0

    avtn = mdl.model_evaluate("LINLOG", avp, f / 200) + avrn
    fb, rbn, wbn, _ = average_in_frequency(avrn, f, avwn)
    tbn = mdl.model_evaluate("LINLOG", avp, fb / 200) + rbn

    return fb, rb_all, wb_all, d_all, tbn, wbn, f, rr_all, wr_all, avrn, avwn, avp, avtn


def integrate_spectrum(
    fname, p, r, w, index_GHA, f_low, f_high, bad_days, n_fg,
):
    """
    Important high level function that averages level4 field data. Used consistently.
    TODO: make this into an actual script. Make it a little bit more general.
    The idea here is just to average together a bunch of level data over many days
    and GHA. Can choose the days.
    """
    in_file = f"{config['edges_folder']}/mid_band/spectra/level4/{fname}/{fname}.hdf5"
    f, px, rx, wx, index, gha, ydx = io.level4read(in_file)

    # Produce integrated spectrum
    for i in range(len(index_GHA)):
        keep = filters.explicit_filter(ydx, bad=bad_days)

        p_i = px[keep, index_GHA[i]]
        r_i = rx[keep, index_GHA[i]]
        w_i = wx[keep, index_GHA[i]]

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
    flags = rfi.xrfi_poly_filter(
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
    fb, rb, wb, sb = tools.average_in_frequency(r, f, wr, n_samples=NS)

    mb = mdl.model_evaluate("LINLOG", p[0], fb / 200)
    tb = mb + rb
    tb[wb == 0] = 0
    sb[wb == 0] = 0

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
        index_good = filters.explicit_filter(
            yd,
            bad=join(
                dirname(__file__),
                f"data/bad_hours_{band}{case if case is not None else ''}.yaml",
            ),
        )

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
        fb, rb, wb, _ = tools.average_in_frequency(
            avr_no_rfi, f, avw_no_rfi, n_samples=16
        )
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
        fb, rbx, wbx, _ = tools.average_in_frequency(
            avrx_no_rfi, f, avwx_no_rfi, n_samples=16
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
