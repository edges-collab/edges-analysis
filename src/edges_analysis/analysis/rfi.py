import numpy as np
import pathlib
import yaml


def excision_raw_frequency(f, d, w, rfi_file=None, extra_rfi=None, in_place=False):
    """
    Excise RFI from given data using a explicitly set list of flag ranges.

    Parameters
    ----------
    f : array-like
        Frequencies, in MHz, of the data.
    d : array-like
        A 1D or 2D array, where the last axis corresponds to frequency. The data
        measured at those frequencies.
    w : array-like
        The weights associated with the data (same shape as d).
    rfi_file : str, optional
        A YAML file containing the key 'rfi_ranges', which should be a list of 2-tuples
        giving the (min, max) frequency range of known RFI channels (in MHz). By default,
        uses a file included in `edges-analysis` with known RFI channels from the MRO.
    extra_rfi : list, optional
        A list of extra RFI channels (in the format of the `rfi_ranges` from the `rfi_file`).
    in_place : bool, optional
        Whether to perform the masking in-place (i.e. overwriting the input `d` and `w`).

    Returns
    -------
    d, w : array-like
        The same shape as the input d and w. The data and weights arrays, with RFI-affected
        channels set to zero.
    """
    flatten = len(d.shape == 1)

    assert d.shape == w.shape
    assert d.shape[-1] == f.shape[0]

    d = np.atleast_2d(d)
    w = np.atleast_2d(w)

    if not in_place:
        d = np.copy(d)
        w = np.copy(w)

    rfi_freqs = []
    if rfi_file is None:
        rfi_file = pathlib.Path(__file__).parent / "data/known_rfi_channels.yaml"

    if rfi_file:
        with open(rfi_file, "r") as fl:
            rfi_freqs.append(yaml.load(fl, Loader=yaml.FullLoader)["rfi_ranges"])

    if extra_rfi:
        rfi_freqs.append(extra_rfi)

    for low, high in rfi_freqs:
        d[:, (f > low) & (f < high)] = 0
        w[:, (f > low) & (f < high)] = 0

    if flatten:
        d = d.flatten()
        w = w.flatten()

    return d, w


def cleaning_sweep(f, d, w, window_width=4, n_poly=4, n_bootstrap=20, n_sigma=2.5):
    """
    Clean sweep of RFI ???

    Parameters
    ----------
    f : array-like
        Frequencies, in MHz, of the data.
    d : array-like
        A 1D or 2D array, where the last axis corresponds to frequency. The data
        measured at those frequencies.
    w : array-like
        The weights associated with the data (same shape as d).
    window_width : float, optional
        The width of the moving window in MHz.
    n_poly : int, optional
        Number of polynomial terms to fit in each sliding window.
    n_bootstrap : int, optional
        Number of bootstrap samples to take to estimate the standard deviation of
        the data without RFI.
    n_sigma : float, optional
        The number of sigma at which to threshold RFI.
    """
    # Initialization of output arrays
    d_out = np.copy(d)
    w_out = np.copy(w)
    index = np.arange(len(f))

    # Initial section of data of width "window_width"
    f_start_block = f[f <= (f[0] + window_width)]
    d_start_block = d[f <= (f[0] + window_width)]
    w_start_block = w[f <= (f[0] + window_width)]

    # Computing residuals for initial section
    par = np.polyfit(
        f_start_block[w_start_block > 0], d_start_block[w_start_block > 0], n_poly - 1,
    )
    m_start_block = np.polyval(par, f_start_block)
    r_start_block = d_start_block - m_start_block

    # Computation of STD for initial section using the median statistic
    # number datapoints drawn for repetitions
    small_sample_size = len(r_start_block) // 2
    r_choice_std = []
    for _ in range(n_bootstrap):
        r_choice = np.random.choice(r_start_block[w_start_block > 0], small_sample_size)
        r_choice_std.append(np.std(r_choice))
    r_std = np.median(r_choice_std)

    # Initial window limits
    index_low = 0
    index_high = len(f_start_block) - 1

    # Sweeping in frequency
    while index_high < (len(f) - 1):

        index_low += 1
        index_high += 1

        # Selecting section of data of width "window_width"
        if index_high < (len(f) - 1):
            mask = slice(index_low, (index_high + 1))
        else:  # index_high == (len(f) - 1)
            mask = slice(index_low, None)

        f_block = f[mask]
        d_block = d_out[mask]
        w_block = w_out[mask]
        i_block = index[mask]

        # Computing residuals within window
        if np.sum(w_block) > 0:
            par = np.polyfit(f_block[w_block > 0], d_block[w_block > 0], n_poly - 1)
            m_block = np.polyval(par, f_block)
            r_block = d_block - m_block

            # Delete new point if it falls outside n_sigma x previous STD of noise
            if np.abs(r_block[-1]) > (n_sigma * r_std):
                w_block[-1] = 0

                d_out[i_block[-1]] = 0
                w_out[i_block[-1]] = 0

            # Compute STD for the current window using only good data
            r_std = np.std(r_block[w_block > 0])

    return d_out, w_out


def cleaning_polynomial(
    fin, tin, win, model_type="loglog", Nterms_fg=10, Nterms_std=3, Nstd=5
):
    """
    This function fits a log-log polynomial to spectrum, and then, iteratively, removes the points
    that fall outside the allowed range.
    The allowed range is determined as n_sigma x a polynomial with terms Nterms_std
    """

    f = np.copy(fin)
    t = np.copy(tin)
    w = np.copy(win)

    sum_rfi = 2
    sum_rfi_old = 1
    while sum_rfi > sum_rfi_old:

        # Remove points with temperature equal to zero, NaN, and +/- Inf
        mask = (t > 0) & (w > 0) & (~np.isnan(t)) & (~np.isinf(t))
        ff = f[mask]
        tt = t[mask]
        ww = w[mask]

        if model_type == "loglog":

            # Log of frequency and temperature, for data with non-zero weights
            log_f = np.log10(ff / 200)
            log_t = np.log10(tt)

            # Remove points with Log_t equal to NaN and +/- Inf
            mask = (~np.isnan(log_t)) & (~np.isinf(log_t))
            log_ff = log_f[mask]
            log_tt = log_t[mask]

            par = np.polyfit(log_ff, log_tt, Nterms_fg - 1)
            log_model = np.polyval(par, np.log10(f / 200))
            model = 10 ** log_model
        elif model_type == "poly":

            par = np.polyfit(ff, tt, Nterms_fg - 1)
            model = np.polyval(par, f)
        else:
            raise ValueError("model_type must be 'loglog' or 'poly'")

        res = t - model
        rr = res[(t > 0) & (w > 0) & (~np.isnan(t)) & (~np.isinf(t))]

        par = np.polyfit(ff[ww > 0] / 200, np.abs(rr)[ww > 0], Nterms_std - 1)
        model_std = np.polyval(par, f / 200)

        RFI = np.zeros(len(fin))
        t[np.abs(res) > Nstd * model_std] = 0
        w[np.abs(res) > Nstd * model_std] = 0
        RFI[np.abs(res) > Nstd * model_std] = 1

        sum_rfi_old = np.copy(sum_rfi)
        sum_rfi = np.sum(RFI)

    tout = np.copy(tin)
    tout[w == 0] = 0

    return tout, w
