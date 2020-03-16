from typing import Tuple

import numpy as np
from edges_cal import modelling as mdl

#
# def _get_level4_case(case):
#     _level4_cases = {
#         2: "calibration_2019_10_no_ground_loss_no_beam_corrections",
#         3: "case_nominal_50-150MHz_no_ground_loss_no_beam_corrections",
#         406: "case_nominal_50-150MHz_LNA1_a2_h2_o2_s1_sim2",
#         5: "case_nominal_14_14_terms_55-150MHz_no_ground_loss_no_beam_corrections",
#         501: "case_nominal_50-150MHz_LNA2_a2_h2_o2_s1_sim2_all_lc_yes_bc",
#         502: "case_nominal_50-150MHz_LNA2_a2_h2_o2_s1_sim2_all_lc_yes_bc_20min",
#     }
#     folder = config["edges_folder"] + "mid_band/spectra/level4/{}/"
#
#     if case not in _level4_cases:
#         raise ValueError("case must be one of {}".format(_level4_cases.keys()))
#
#     case = _level4_cases[case]
#     folder = folder.format(case)
#
#     return case, folder


# def level4_binned_residuals(case, f_low, f_high, output_file_name_hdf5):
#     case, folder = _get_level4_case(case)
#
#     f, p, r, w, index, gha, yy = level4read(folder + case + ".hdf5")
#     save_folder = folder + "binned_residuals/"
#
#     # Compute the residuals
#     start = True
#     for i in range(len(r[:, 0, 0])):
#         for k in range(len(r[0, :, 0])):
#
#             if np.sum(w[i, k, :]) > 100:
#
#                 model_k = mdl.model_evaluate("LINLOG", p[i, k, :], f / 200)
#                 t_k = r[i, k, :] + model_k
#                 w_k = w[i, k, :]
#
#                 fc = f[(f > f_low) & (f < f_high)]
#                 tc_k = t_k[(f > f_low) & (f < f_high)]
#                 wc_k = w_k[(f > f_low) & (f < f_high)]
#
#                 n_fg = 5 if 6 <= gha[k] <= 17 else 6
#                 pc_k = mdl.fit_polynomial_fourier(
#                     "LINLOG", fc / 200, tc_k, n_fg, Weights=wc_k
#                 )
#                 mc_k = mdl.model_evaluate("LINLOG", pc_k[0], fc / 200)
#
#                 rc_k = tc_k - mc_k
#
#                 fb, rb, wb, sb = average_in_frequency(rc_k, fc, wc_k, n_samples=16)
#
#                 if start:
#                     binned_residuals = np.zeros((len(r), len(r[0]), len(fb)))
#                     binned_weights = np.zeros_like(binned_residuals)
#                     binned_stddev = np.zeros_like(binned_residuals)
#                     start = False
#
#                 binned_residuals[i, k, :] = rb
#                 binned_weights[i, k, :] = wb
#                 binned_stddev[i, k, :] = sb
#
#     # Save
#     # ----
#     with h5py.File(save_folder + output_file_name_hdf5, "w") as hf:
#         hf.create_dataset("frequency", data=fb)
#         hf.create_dataset("residuals", data=binned_residuals)
#         hf.create_dataset("weights", data=binned_weights)
#         hf.create_dataset("stddev", data=binned_stddev)
#         hf.create_dataset("gha_edges", data=gha)
#         hf.create_dataset("year_day", data=yy)
#
#     return fb, binned_residuals


def spectrum_fit(f, t, w, n_poly=5, f1_low=60, f1_high=65, f2_low=95, f2_high=140):
    fc = f[((f >= f1_low) & (f <= f1_high)) | ((f >= f2_low) & (f <= f2_high))]
    tc = t[((f >= f1_low) & (f <= f1_high)) | ((f >= f2_low) & (f <= f2_high))]
    wc = w[((f >= f1_low) & (f <= f1_high)) | ((f >= f2_low) & (f <= f2_high))]

    pc = mdl.fit_polynomial_fourier("LINLOG", fc / 200, tc, n_poly, Weights=wc)
    m = mdl.model_evaluate("LINLOG", pc[0], f / 200)
    r = t - m

    return f, r


# def level4_foreground_fits(
#     case, f_low, f_high, f_norm, output_file_name_hdf5="foreground_fits.hdf5"
# ):
#     folder, case = _get_level4_case(case)
#     save_folder = folder + "binned_residuals/"
#
#     f, p, r, w, index, gha, yy = level4read(folder + case + ".hdf5")
#
#     # Computing the foreground parameters
#     # -----------------------------------
#     fits = [np.zeros((len(r), len(r[0]), n)) for n in range(11, 15)]
#
#     for i, yyi in enumerate(yy):
#         # Loading data
#         path_level3_file = config[
#             "edges_folder"
#         ] + "mid_band/spectra/level3/case_nominal_50-150MHz_LNA2_a2_h2_o2_s1_sim2_all_lc_yes_bc" "/2018_{}_00.hdf5".format(
#             yyi
#         )
#
#         xf, xt, xp, xr, xw, xrms, xtp, xm = level3read(path_level3_file)
#
#         for k in range(len(r[0, :, 0])):
#             IX = np.arange(0, len(index[0, 0, :]))
#             new_meta = xm[IX[index[i, k, :] == 1], :]
#
#             # If there are enough data points
#             if (np.sum(w[i, k, :]) > 100) and (len(new_meta) > 0):
#
#                 av_sunel = np.mean(new_meta[:, 6])
#                 av_moonel = np.mean(new_meta[:, 8])
#
#                 fb, rb, wb, sb = average_in_frequency(
#                     r[i, k, :], f, w[i, k, :], n_samples=16
#                 )
#
#                 mb = mdl.model_evaluate("LINLOG", p[i, k, :], fb / 200)
#                 tb = rb + mb
#
#                 mask = (fb > f_low) & (fb < f_high)
#                 fc = fb[mask]
#                 tc = tb[mask]
#                 wc = wb[mask]
#                 sc = sb[mask]
#
#                 for i_fg, n_fg in enumerate([2, 3, 4, 5]):
#
#                     logf = np.log(fc / f_norm)
#                     logt = np.log(tc)
#
#                     par = np.polyfit(logf[wc > 0], logt[wc > 0], n_fg - 1)
#                     logm = np.polyval(par, logf)
#                     m = np.exp(logm)
#
#                     rms = np.std((tc - m)[wc > 0])
#
#                     chi2 = np.sum(((tc - m)[wc > 0] / sc[wc > 0]) ** 2)
#                     LEN = len(fc[wc > 0])
#                     BIC = chi2 + n_fg * np.log(LEN)
#
#                     fits[i_fg][i, k, 0] = yyi[0]
#                     fits[i_fg][i, k, 1] = yyi[1]
#                     fits[i_fg][i, k, 2] = gha[k]
#                     fits[i_fg][i, k, 3] = av_sunel
#                     fits[i_fg][i, k, 4] = av_moonel
#
#                     fits[i_fg][i, k, 5] = np.exp(par[-1])
#                     fits[i_fg][i, k, 6 : 6 + n_fg - 1] = par[:-1][::-1]
#
#                     fits[i_fg][i, k, -4] = rms
#                     fits[i_fg][i, k, -3] = chi2
#                     fits[i_fg][i, k, -2] = LEN
#                     fits[i_fg][i, k, -1] = BIC
#
#     # Save to file.
#     with h5py.File(save_folder + output_file_name_hdf5, "w") as hf:
#         hf.create_dataset("fref", data=np.array([f_norm]))
#         hf.create_dataset("fit2", data=fits[0])
#         hf.create_dataset("fit3", data=fits[1])
#         hf.create_dataset("fit4", data=fits[2])
#         hf.create_dataset("fit5", data=fits[3])
#
#     return fits


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


def average_in_gha(
    spectrum: np.ndarray, gha: np.ndarray, gha_bins, weights: np.ndarray = None
):
    """
    Average a spectrum into bins of GHA.

    Parameters
    ----------
    spectrum : ndarray
        The spectrum to average. The first axis is assumed to be GHA.
    gha : array
        A 1D array of GHAs corresponding to the first axis of `spectrum`.
    gha_bins : array-like
        The bin-edges of GHA to bin in.
    weights : array-like, optional
        Weights for each of the spectrum points. Assumed to be one if not given.

    Returns
    -------
    spectrum : array-like
        An array of the same shape as the input ``spectrum``, except with the first axis
        reduced to the size of ``gha_bins``. The mean spectrum in each GHA bin.
    weights : array-like
        An array of the same shape as the output spectrum, containing the sum of all
        weights in each bin.
    """
    if weights is None:
        weights = np.ones_like(spectrum)

    orig_shape = spectrum.shape[1:]

    spectrum = np.reshape(spectrum, (spectrum.shape[0], -1))
    weights = np.reshape(weights, (weights.shape[0], -1))

    out_spectrum = np.zeros((len(gha_bins) - 1, spectrum.shape[1]))
    out_weights = np.zeros((len(gha_bins) - 1, spectrum.shape[1]))

    for i, (spec, wght) in enumerate(zip(spectrum.T, weights.T)):
        out_spectrum[:, i] = np.histogram(gha, bins=gha_bins, weights=wght * spec)[0]
        out_weights[:, i] = np.histogram(gha, bins=gha_bins, weights=wght)[0]

    out_spectrum /= out_weights

    out_spectrum = np.reshape(out_spectrum, (-1,) + orig_shape)
    out_weights = np.reshape(out_weights, (-1,) + orig_shape)

    return out_spectrum, out_weights
