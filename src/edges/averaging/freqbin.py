"""Functions for binning and decimating GSData objects in frequency."""

import itertools

import numpy as np
from astropy import units as un
from pygsdata import GSData, gsregister
from scipy.ndimage import convolve1d

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

    Returns
    -------
    GSData
        The smoothed and potentially decimated GSData object. Output channels with no
        unflagged data in their window (or flagged by ``flag_threshold`` or
        ``maintain_flags``) have NaN data and zero nsamples.

    Raises
    ------
    ValueError
        If the residuals are to be smoothed and are not present in the data.

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

    # The output is a weighted mean with weights w = k * f_nsamples, so its variance
    # is sigma^2 * sum(w^2 / n) / sum(w)^2 (for per-sample variance sigma^2 and n
    # samples in each channel). The effective number of samples is the inverse.
    with np.errstate(divide="ignore", invalid="ignore"):
        w2_over_n = np.where(true_nsamples > 0, f_nsamples**2 / true_nsamples, 0)
    sum_w2_over_n = convolve1d(w2_over_n, window**2, mode="constant", cval=0)[
        ..., decimate_at::decimate
    ]

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
    sums[~mask] /= nsamples[~mask]
    # Channels with no (or too little) unflagged data in the window are empty.
    sums[mask] = np.nan

    nsamples = np.zeros_like(nsamples)
    nsamples[~mask] = sums_of_w[~mask] ** 2 / sum_w2_over_n[~mask]

    models = data.model[..., decimate_at::decimate] if use_residuals else 0

    return data.update(
        data=sums + models,
        nsamples=nsamples,
        residuals=sums if use_residuals else None,
        flags={k: v for k, v in data.flags.items() if "freq" not in v.axes},
        freqs=data.freqs[decimate_at::decimate],
        data_unit=data.data_unit,
    )
