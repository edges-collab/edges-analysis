"""Functions for binning and decimating GSData objects in frequency."""

import itertools

import numpy as np
from astropy import units as un
from pygsdata import GSData, GSFlag, gsregister
from scipy.ndimage import convolve1d, correlate1d

from . import averaging as avg


@gsregister("reduce")
def freq_bin(
    data: GSData,
    bins: np.ndarray | un.Quantity | float | None = None,
    debias: bool | None = None,
) -> GSData:
    """
    Bin a GSData object over the frequency/channel axis.

    Parameters
    ----------
    data
        The input GSData object to be binned.
    bins
        The bin *edges* (lower inclusive, upper not inclusive, except for the last bin,
        whose upper edge is also inclusive, as in :func:`numpy.histogram`). Each
        channel is therefore in at most one bin. If an ``int``, simply
        use ``bins`` coords per bin, starting from the first bin. If a float or
        Quantity, use equi-spaced bin edges, starting from the start of coords, and
        ending past the end of coords. If an array, assumed to be the bin edges.
        If not provided, assume a single bin encompassing all the data.
    model
        A model to be used for debiasing, by default None.
    debias
        Whether to debias the data using the provided model or residuals, by default
        None.

    Returns
    -------
    GSData
        The binned and decimated GSData object.

    Notes
    -----
    If the data object has residuals, they will be set appropriately on the returned
    object. Furthermore, the frequencies of the returned object will be the mean of the
    frequencies in each bin, and therefore may not be regular. Finally, flags will only
    be maintained if they have no frquency axis (though flags will be utilized
    appropriately in the averaging process).
    """
    edges = avg.get_bin_edges(data.freqs, bins)
    nbins = len(edges) - 1
    # Each bin includes its lower edge and excludes its upper edge, except the last
    # bin, which includes both (like numpy.histogram), so that every channel within
    # [edges[0], edges[-1]] is in exactly one bin.
    bins = [
        (data.freqs >= lo)
        & ((data.freqs < hi) | ((i == nbins - 1) & (data.freqs == hi)))
        for i, (lo, hi) in enumerate(itertools.pairwise(edges))
    ]

    if debias is None:
        debias = data.residuals is not None

    if debias and data.residuals is None:
        raise ValueError("Cannot debias without data residuals!")

    d, w, r = avg.bin_data(
        data=data.data,
        weights=data.flagged_nsamples,
        residuals=data.residuals if debias else None,
        bins=bins,
        axis=-1,
    )

    return data.update(
        data=d,
        nsamples=w,
        flags={k: v for k, v in data.flags.items() if "freq" not in v.axes},
        freqs=np.array([np.mean(data.freqs.value[b]) for b in bins]) * data.freqs.unit,
        residuals=r,
    )


def _correlated_terms(
    f_nsamples: np.ndarray,
    true_nsamples: np.ndarray,
    window: np.ndarray,
    channel_correlation: np.ndarray,
) -> np.ndarray:
    """Off-diagonal terms of sum_ij w_i w_j rho_|i-j| / sqrt(n_i n_j) in each window.

    With ``w = k * f`` (``k`` the kernel centred on the output channel), this returns
    ``sum_{l>=1} 2 rho_l sum_j k_j k_{j+l} s_x s_{x+l}`` for every output channel, where
    ``s = f / sqrt(n)`` and ``x`` runs over the window.
    """
    rho = np.asarray(channel_correlation, dtype=float)
    if rho.ndim != 1 or rho.size == 0 or not np.isclose(rho[0], 1):
        raise ValueError("channel_correlation must be 1D with channel_correlation[0]=1")
    with np.errstate(divide="ignore", invalid="ignore"):
        s = np.where(true_nsamples > 0, f_nsamples / np.sqrt(true_nsamples), 0)
    nwin = window.size
    out = np.zeros(s.shape)
    for lag in range(1, min(rho.size, nwin)):
        if rho[lag] == 0:
            continue
        pairs = np.zeros(s.shape)
        pairs[..., :-lag] = s[..., :-lag] * s[..., lag:]
        kernel = np.zeros(nwin)
        kernel[: nwin - lag] = window[: nwin - lag] * window[lag:]
        out += 2 * rho[lag] * correlate1d(pairs, kernel, mode="constant", cval=0)
    return out


@gsregister("reduce")
def gauss_smooth(
    data: GSData,
    size: int,
    decimate: bool = True,
    decimate_at: int | None = None,
    flag_threshold: float = 0,
    maintain_flags: int = 0,
    use_residuals: bool | None = None,
    nsmooth: int = 4,
    use_nsamples: bool = False,
    channel_correlation: np.ndarray | None = None,
) -> GSData:
    """Smooth data with a Gaussian function, and optionally decimate.

    Parameters
    ----------
    data
        The :class:`GSData` object over which to smooth.
    size
        The size of the Gaussian smoothing kernel. The ultimate size of the kernel will
        be ``nsmooth*size``. The final array will be decimated by a factor of ``size``,
        if the ``decimate`` option is set.
        Thus, to maintain the same number of samples, set ``size`` to unity and
        ``nsmooth`` larger.
    decimate
        Whether to decimate the array by a factor of ``size``.
    decimate_at
        The first index to *keep* when decimating. If ``None``, this will be
        ``size//2``.
    flag_threshold
        The threshold of flagged samples to flag a channel. Set to 0.25 to flag in the
        same way as Alan's C-Code. In detail, the weights that are convolved (the
        flagged nsamples if ``use_nsamples`` is True, otherwise unity for each
        unflagged channel) are normalised by their maximum over frequency (for each
        load, polarization and time), and any smoothed channel whose kernel-weighted
        sum of these normalised weights is less than or equal to
        ``size*flag_threshold`` is flagged. For a dataset with uniform weights (but
        potentially some flagged bins), this is the integrated kernel window *not*
        counting flagged channels. The result is independent of the overall scale of
        nsamples.
    maintain_flags
        Whether to maintain the flags in the data. If ``True``, any fine-channels
        that were originally flagged will be flagged in the output. The default
        behaviour is to simply sum all weights within the window, but Alan's code
        uses this feature.
    use_residuals
        Whether to smooth the residuals to a model fit instead of the spectrum
        itself. By default, this is ``True`` if a model is present in the data.
    nsmooth
        The ratio of the size of the smoothing kernel (in pixels) to the decimation
        length (i.e. ``size``).
    use_nsamples
        Whether to weight the data by nsamples when performing the smoothing kernel
        convolution. If ``False``, each unflagged channel is given unit weight (times
        the kernel). Either way, the output nsamples account for the input nsamples
        (see Notes).
    channel_correlation
        The correlation between the noise of input channels ``0, 1, 2, ...`` apart
        (so ``channel_correlation[0] == 1``), used for the output nsamples. By default
        the input channels are taken to be independent. Neighbouring channels of a
        windowed FFT are correlated: for the four-term Blackman-Harris window of the
        EDGES-2 spectrometer the correlation of the channel powers is 0.666, 0.192 and
        0.023 at lags 1-3 (see :func:`edges.cal.noise_model.channel_power_correlation`),
        and ignoring it over-estimates the output nsamples of the B18 smoothing
        (``size=8``, ``nsmooth=4``) by a factor of 2.7.

    Returns
    -------
    GSData
        The smoothed and potentially decimated GSData object. Output channels with no
        unflagged data in their window, or flagged by ``flag_threshold`` or
        ``maintain_flags``, are *empty*: they have zero nsamples and are flagged by a
        ``"gauss_smooth"`` flag. Where the window holds some unflagged data, the data
        of an empty channel is the smoothed value of that data (the flagged channel
        itself does not contribute). Where it holds none, the residual is zero (so the
        data equals the model) if ``use_residuals``, and the data is NaN otherwise.
        The data are kept finite wherever a model is present because
        :class:`pygsdata.GSData` derives the model as ``data - residuals``, so a NaN
        in the data would also delete the model.

    Raises
    ------
    ValueError
        If the residuals are to be smoothed and are not present in the data, or if
        the model is NaN at an output channel that has data (the model cannot then
        be added back to the smoothed residuals).

    Notes
    -----
    Each output channel is a weighted mean of the input channels, with weights
    ``w_i = k_i * u_i``, where ``k`` is the Gaussian kernel and ``u_i`` is the
    flagged nsamples ``n_i`` (if ``use_nsamples``) or unity for unflagged channels.
    If each input channel has variance ``sigma^2 / n_i``, the variance of the output
    is ``sigma^2 * sum(w_i^2 / n_i) / sum(w_i)^2``. The output nsamples is the
    corresponding effective number of samples, ``sum(w_i)^2 / sum(w_i^2 / n_i)``
    (for ``use_nsamples=True`` this is ``sum(k n)^2 / sum(k^2 n)``), so that it can be
    used as an inverse-variance weight downstream. Note that it is *not* the plain
    kernel-weighted sum of nsamples, ``sum(k n)``, which under-estimates the inverse
    variance (by a factor of ~1.4 in the interior of the band, for the kernel
    used here, whose peak is unity). Adjacent output
    channels are correlated if the kernel is wider than the decimation factor.

    If the input channels are correlated, with correlation ``rho_|i-j|`` between
    channels ``i`` and ``j`` (``channel_correlation``), the variance of the output is
    ``sigma^2 * sum_ij(w_i w_j rho_|i-j| / sqrt(n_i n_j)) / sum(w_i)^2``, and the
    output nsamples is the corresponding effective number of samples.
    """
    if use_residuals is None:
        use_residuals = data.residuals is not None

    if use_residuals and data.residuals is None:
        raise ValueError("Cannot smooth residuals without a model in the data!")

    assert isinstance(size, int)
    if decimate:
        if decimate_at is None:
            decimate_at = size // 2

        assert isinstance(decimate_at, int)
        assert decimate_at < size
    else:
        decimate_at = 0
    # This choice of size scaling corresponds to Alan's C code.
    y = np.arange(-size * nsmooth, size * nsmooth + 1) * 2 / size
    window = np.exp(-(y**2) * 0.69)
    decimate = size if decimate else 1

    # mask data and flagged samples wherever it is NaN
    dd = data.residuals if use_residuals else data.data

    inflags = (data.flagged_nsamples == 0) | np.isnan(dd)

    data_mask = np.where(np.isnan(dd), 0, dd)
    # The true number of samples in each channel (zero where flagged or NaN).
    true_nsamples = np.where(inflags, 0, data.flagged_nsamples)
    nsamples = true_nsamples if use_nsamples else (~inflags).astype(float)

    f_nsamples = np.where(np.isnan(dd), 0, nsamples)

    sums = convolve1d(f_nsamples * data_mask, window, mode="constant", cval=0)[
        ..., decimate_at::decimate
    ]
    nsamples = convolve1d(f_nsamples, window, mode="constant", cval=0)
    # The total kernel weight of the unflagged data in each output window, before
    # any channels are emptied by maintain_flags or flag_threshold.
    window_w = nsamples[..., decimate_at::decimate].copy()

    # The output is a weighted mean with weights w = k * f_nsamples, so its variance
    # is sigma^2 * sum(w^2 / n) / sum(w)^2 (for per-sample variance sigma^2 and n
    # samples in each channel). The effective number of samples is the inverse.
    with np.errstate(divide="ignore", invalid="ignore"):
        w2_over_n = np.where(true_nsamples > 0, f_nsamples**2 / true_nsamples, 0)
    sum_w2_over_n = convolve1d(w2_over_n, window**2, mode="constant", cval=0)
    if channel_correlation is not None:
        sum_w2_over_n = sum_w2_over_n + _correlated_terms(
            f_nsamples, true_nsamples, window, channel_correlation
        )
    sum_w2_over_n = sum_w2_over_n[..., decimate_at::decimate]

    if maintain_flags > 0:
        if maintain_flags == 1:
            nsamples[inflags] = 0
        else:
            flags = convolve1d(
                (~inflags).astype(int), np.ones(maintain_flags), mode="constant", cval=0
            )
            nsamples[flags == 0] = 0

    nsamples = nsamples[..., decimate_at::decimate]
    sums_of_w = nsamples.copy()

    # Normalise the smoothed weights by the maximum weight that was convolved, so
    # that the threshold does not depend on the overall scale of the weights.
    maxw = np.max(f_nsamples, axis=-1, keepdims=True)
    with np.errstate(divide="ignore", invalid="ignore"):
        frac = np.where(maxw > 0, nsamples / maxw, 0)
    nsamples[frac <= size * flag_threshold] = 0
    mask = (nsamples == 0) | (sum_w2_over_n <= 0)

    # Empty channels whose window holds some data take the smoothed value of that
    # data, so that the data stay finite (and any model, which GSData derives as
    # data - residuals, survives).
    # Windows with no data at all have no estimate: the residual is zero (data equal
    # to the model) if smoothing residuals, and NaN otherwise.
    has_data = window_w > 0
    sums[has_data] /= window_w[has_data]
    sums[~has_data] = 0.0 if use_residuals else np.nan

    nsamples = np.zeros_like(nsamples)
    nsamples[~mask] = sums_of_w[~mask] ** 2 / sum_w2_over_n[~mask]

    if use_residuals:
        models = data.model[..., decimate_at::decimate]
        if np.any(np.isnan(models) & ~mask):
            raise ValueError(
                "The model is NaN at output channels that have data, so it cannot be "
                "added back to the smoothed residuals. GSData derives the model as "
                "data - residuals: the input data are NaN where the model is needed "
                "(e.g. from an earlier gauss_smooth of edges-analysis 8.5.0, which "
                "returned NaN for empty channels)."
            )
    else:
        models = 0

    flags = {k: v for k, v in data.flags.items() if "freq" not in v.axes}
    flags["gauss_smooth"] = GSFlag(flags=mask, axes=("load", "pol", "time", "freq"))

    return data.update(
        data=sums + models,
        nsamples=nsamples,
        residuals=sums if use_residuals else None,
        flags=flags,
        freqs=data.freqs[decimate_at::decimate],
        data_unit=data.data_unit,
    )
