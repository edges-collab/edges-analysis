"""Functions for binning GSData objects in frequency."""
from __future__ import annotations

import edges_cal.modelling as mdl
import numpy as np
from astropy import units as un
from scipy.ndimage import convolve1d

from ..datamodel import add_model
from ..gsdata import GSData, gsregister
from . import averaging as avg


@gsregister("reduce")
def freq_bin(
    data: GSData,
    resolution: int | un.Quantity[un.MHz],
    model: mdl.Model | None = None,
    debias: bool | None = None,
):
    """Bin on frequency axis."""
    bins = avg.get_bin_edges(data.freq_array, resolution)
    bins = [
        (data.freq_array >= b[0]) & (data.freq_array <= b[1])
        for b in zip(bins[:-1], bins[1:])
    ]

    if debias is None:
        debias = model is not None or data.residuals is not None

    if debias and model is None and data.residuals is None:
        raise ValueError("Cannot debias without data residuals or a model!")

    if debias:
        if data.residuals is None:
            data = add_model(data, model=model, append_to_file=False)

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
        freq_array=np.array([np.mean(data.freq_array.value[b]) for b in bins])
        * data.freq_array.unit,
        residuals=r,
    )


@gsregister("reduce")
def gauss_smooth(
    data: GSData,
    size: int,
    decimate: bool = True,
    decimate_at: int = None,
    flag_threshold: float = 0,
    maintain_flags: bool = False,
    use_residuals: bool | None = None,
) -> np.ndarray:
    """Smooth data with a Gaussian function, and reduce the size of the array.

    Parameters
    ----------
    data
        The :class:`GSData` object to smooth.
    size
        The size of the Gaussian smoothing kernel. The ultimate size of the kernel will
        be ``4*size``. The final array will be decimated by a factor of ``size``.
    decimate
        Whether to decimate the array by a factor of ``size``.
    decimate_at
        The index at which to start decimating the array. If ``None``, this will be
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
    y = np.arange(-size * 4, size * 4 + 1) * 2 / size
    window = np.exp(-(y**2) * 0.69)
    decimate = size if decimate else 1

    # mask data and flagged samples wherever it is NaN
    if use_residuals:
        dd = data.residuals
    else:
        dd = data.data

    data_mask = np.where(np.isnan(dd), 0, dd)
    f_nsamples = np.where(np.isnan(dd), 0, data.flagged_nsamples)

    sums = convolve1d(data.flagged_nsamples * data_mask, window, mode="nearest")[
        ..., decimate_at::decimate
    ]
    nsamples = convolve1d(f_nsamples, window, mode="nearest")
    if maintain_flags:
        nsamples[data.complete_flags] = 0

    nsamples = nsamples[..., decimate_at::decimate]

    nsamples[nsamples / data.nsamples.max() <= size * flag_threshold] = 0
    mask = nsamples == 0
    sums[mask] = np.nan
    sums[~mask] /= nsamples[~mask]

    if use_residuals:
        models = data.model[..., decimate_at::decimate]
    else:
        models = 0

    return data.update(
        data=sums + models,
        nsamples=nsamples,
        residuals=sums if use_residuals else None,
        flags={k: v for k, v in data.flags.items() if "freq" not in v.axes},
        freq_array=data.freq_array[decimate_at::decimate],
        data_unit=data.data_unit,
    )


# @gsregister("reduce")
# def freq_bin(
#     data: GSData,
#     resolution: int | float,
#     model: mdl.Model | None = None,
# ) -> GSData:
#     """Frequency binning that auto-selects which kind of binning to do.

#     Parameters
#     ----------
#     data
#         The :class:`GSData` object to bin.
#     resolution
#         The resolution of the binning in MHz or number of existing channels (if int).
#     model
#         A :class:`edges_cal.modelling.Model` object to use for de-biasing the mean. If
#         the ``data`` object has a ``data_model`` defined, this will be used instead.
#         If not provided, simple direct binning will be used.
#     """
#     if data.data_model is None and model is None:
#         return freq_bin_direct(data, resolution)
#     else:
#         return freq_bin_with_models(data, resolution, model)
