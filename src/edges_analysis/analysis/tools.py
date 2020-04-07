from typing import Tuple
from datetime import datetime, timedelta

import numpy as np
from edges_cal import modelling as mdl


def join_struct_arrays(arrays):
    """Join a list of structured numpy arrays (make new columns)."""
    dtype = sum((a.dtype.descr for a in arrays), [])
    out = np.empty(len(arrays[0]), dtype=dtype)
    for a in arrays:
        for name in a.dtype.names:
            out[name] = a[name]
    return out


def dt_from_year_day(year, day, *args):
    return datetime(year, 1, 1, *args) + timedelta(days=day - 1)


def spectrum_fit(f, t, w, n_poly=5, f1_low=60, f1_high=65, f2_low=95, f2_high=140):
    fc = f[((f >= f1_low) & (f <= f1_high)) | ((f >= f2_low) & (f <= f2_high))]
    tc = t[((f >= f1_low) & (f <= f1_high)) | ((f >= f2_low) & (f <= f2_high))]
    wc = w[((f >= f1_low) & (f <= f1_high)) | ((f >= f2_low) & (f <= f2_high))]

    m = mdl.ModelFit("linlog", fc / 200, tc, weights=wc, n_terms=n_poly).evaluate(
        f / 200
    )
    r = t - m

    return f, r


def average_in_frequency(
    spectrum: [list, np.ndarray],
    freq: [list, np.ndarray, None] = None,
    weights: [list, np.ndarray, None] = None,
    resolution: [float, None] = None,
    n_samples: [int, None] = None,
    axis: int = -1,
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
    axis : int, optional
        The axis along which to do the binning.

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
    if axis < 0:
        axis += spectrum.ndim

    if resolution is not None:
        n_samples = max(int(resolution / (freq[1] - freq[0])), 1)

    if (resolution or n_samples) and freq is None:
        raise ValueError(
            "You must provide freq if resolution or n_samples is provided!"
        )

    nf = spectrum.shape[axis]

    if resolution is None and n_samples is None:
        n_samples = nf

    if freq is None:
        freq = np.ones(nf)

    if weights is None:
        weights = np.ones_like(spectrum)

    mod = nf % n_samples
    if mod:
        rng = range(-mod, -1)
        last_f = freq[rng]
        last_s = spectrum.take(rng, axis=axis)
        last_w = weights.take(rng, axis=axis)

        ss, ww = weighted_mean(last_s, last_w, axis=axis)
        last_std = weighted_standard_deviation(
            np.expand_dims(ss, axis), last_s, std=np.sqrt(1 / last_w), axis=axis
        )
        last_s = ss
        last_w = ww

    rng = range(nf - mod)
    # Get the main part of the array (without trailing bin)
    f = freq[rng]
    s = spectrum.take(rng, axis=axis)
    w = weights.take(rng, axis=axis)

    # Reshape the array so that the binning axis is split.
    f = np.reshape(f, (-1, n_samples))
    s = np.reshape(s, s.shape[:axis] + (-1, n_samples) + s.shape[(axis + 1) :])
    w = np.reshape(w, w.shape[:axis] + (-1, n_samples) + w.shape[(axis + 1) :])

    f = np.mean(f, axis=1)
    s_tmp, w_tmp = weighted_mean(s, w, axis=axis + 1)
    std = weighted_standard_deviation(
        np.expand_dims(s_tmp, axis + 1), s, std=np.sqrt(1 / w), axis=axis + 1
    )
    s = s_tmp
    w = w_tmp

    if mod:
        f = np.concatenate((f, [np.mean(last_f)]))
        s = np.concatenate((s, np.expand_dims(last_s, axis)), axis=axis)
        w = np.concatenate((w, np.expand_dims(last_w, axis)), axis=axis)
        std = np.concatenate((std, np.expand_dims(last_std, axis)), axis=axis)

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

    av = np.zeros_like(sum)
    mask = weights > 0
    av[mask] = sum[mask] / weights[mask]
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
