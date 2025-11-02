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
    bins: np.ndarray | un.Quantity | int | float | None = None,
    debias: bool | None = None,
) -> GSData:
    """
    Bin a GSData object over the frequency/channel axis.

    Parameters
    ----------
    data
        The input GSData object to be binned.
    bins
        The bin *edges* (lower inclusive, upper not inclusive). If an ``int``, simply
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
    bins = avg.get_bin_edges(data.freqs, bins)
    bins = [
        (data.freqs >= b[0]) & (data.freqs <= b[1]) for b in itertools.pairwise(bins)
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
        same way as Alan's C-Code. In detail, for a dataset with uniform weights (but
        potentially some flagged bins), any smoothed channel whose integrated window
        *not* counting flagged channels is smaller than ``size*flag_threshold`` will be
        flagged.
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
        convolution. Note that even if this is set to ``False``, the Nsamples in the
        output will be the kernel-weighted sum of the input samples to each resulting
        channel.

    Returns
    -------
    GSData
        The smoothed and potentially decimated GSData object.

    Raises
    ------
    ValueError
        If the residuals are to be smoothed and are not present in the data.
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

    inflags = data.flagged_nsamples == 0 | np.isnan(dd)

    data_mask = np.where(np.isnan(dd), 0, dd)
    nsamples = data.flagged_nsamples if use_nsamples else (~inflags).astype(float)

    f_nsamples = np.where(np.isnan(dd), 0, nsamples)

    sums = convolve1d(f_nsamples * data_mask, window, mode="constant", cval=0)[
        ..., decimate_at::decimate
    ]
    nsamples = convolve1d(f_nsamples, window, mode="constant", cval=0)

    if maintain_flags > 0:
        if maintain_flags == 1:
            nsamples[inflags] = 0
        else:
            flags = convolve1d(
                (~inflags).astype(int), np.ones(maintain_flags), mode="constant", cval=0
            )
            nsamples[flags == 0] = 0

    nsamples = nsamples[..., decimate_at::decimate]

    maxn = np.max(data.nsamples, axis=-1)[:, :, :, None]
    nsamples[nsamples / maxn <= size * flag_threshold] = 0
    mask = nsamples == 0
    sums[~mask] /= nsamples[~mask]
    sums[mask] = dd[..., decimate_at::decimate][mask]

    if not use_nsamples:
        # We have to still get the proper nsamples for the output.
        nsamples_ = convolve1d(data.flagged_nsamples, window, mode="constant", cval=0)
        nsamples_ = nsamples_[..., decimate_at::decimate]
        nsamples_[nsamples == 0] = 0
        nsamples = nsamples_

    models = data.model[..., decimate_at::decimate] if use_residuals else 0

    return data.update(
        data=sums + models,
        nsamples=nsamples,
        residuals=sums if use_residuals else None,
        flags={k: v for k, v in data.flags.items() if "freq" not in v.axes},
        freqs=data.freqs[decimate_at::decimate],
        data_unit=data.data_unit,
    )
