"""Functions for binning GSData objects in frequency."""
from __future__ import annotations

import edges_cal.modelling as mdl
import numpy as np
from astropy import units as un
from scipy.ndimage import convolve1d

from ..gsdata import GSData, add_model, gsregister
from .averaging import bin_array_biased_regular, bin_freq_unbiased_regular


def freq_bin_direct(data: GSData, resolution: int | float) -> GSData:
    """Bin the data in the LST axis.

    This function does a direct weighted averaged within each LST bin. This is biased
    if the mean spectrum is not constant within each bin *and* the weights are not
    uniform.

    Parameters
    ----------
    data
        The :class:`GSData` object to bin.
    resolution
        The resolution of the binning in MHz or number of existing channels (if int).
    """
    new_freqs, spec, wght = bin_array_biased_regular(
        data=data.data,
        weights=data.flagged_nsamples,
        coords=data.freq_array,
        axis=-1,
        bins=resolution,
    )
    print("HERE>>>", new_freqs[0], spec.shape, wght.shape)
    return data.update(freq_array=new_freqs, data=spec, nsamples=wght, flags={})


@gsregister("reduce")
def freq_bin_with_models(
    data: GSData,
    resolution: int | float,
    model: mdl.Model | None = None,
):
    """Bin the data in the LST axis using model information to de-bias the mean.

    Parameters
    ----------
    data
        The :class:`GSData` object to bin.
    resolution
        The resolution of the binning in MHz or number of existing channels (if int).
    model
        A :class:`edges_cal.modelling.Model` object to use for de-biasing the mean. If
        the ``data`` object has a ``data_model`` defined, this will be used instead.
    """
    if data.data_model is None and model is None:
        raise ValueError("Cannot bin with models without a model in the data!")

    if data.data_model is None:
        data = add_model(data, model=model, append_to_file=False)

    # Averaging data within GHA bins
    f, weights, _, resids, params = bin_freq_unbiased_regular(
        model=data.data_model.model,
        params=data.data_model.parameters.reshape((-1, data.data_model.model.n_terms)),
        freq=data.freq_array.to_value("MHz"),
        resids=data.resids.reshape((-1, data.nfreqs)),
        weights=data.flagged_nsamples.reshape((-1, data.nfreqs)),
        resolution=resolution,
    )

    params = params.reshape(data.data.shape[:-1] + (data.data_model.model.n_terms,))
    return data.update(
        data=resids.reshape(data.data.shape[:-1] + (-1,)),
        nsamples=weights.reshape(data.data.shape[:-1] + (-1,)),
        flags={},
        freq_array=f * un.MHz,
        data_model=data.data_model.update(parameters=params),
    )


@gsregister("reduce")
def gauss_smooth(
    data: GSData,
    size: int,
    decimate: bool = True,
    decimate_at: int = None,
    flag_threshold: float = 0,
    maintain_flags: bool = False,
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
    """
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
    if not decimate:
        decimate = 1
    else:
        decimate = size
    ### masking data and flagged samples whereever it is NaN 
    data_mask = np.where(np.isnan(data.data),0,data.data)
    f_nsamples = np.where(np.isnan(data.data),0,data.flagged_nsamples)
    
    sums = convolve1d(data.flagged_nsamples * data_mask, window, mode="nearest")[
        ..., decimate_at::decimate
    ]
    nsamples = convolve1d(f_nsamples, window, mode="nearest")
    if maintain_flags:
        nsamples[data.complete_flags] = 0

    nsamples = nsamples[..., decimate_at::decimate]

    nsamples[nsamples / data.nsamples.max() <= size * flag_threshold] = 0
    mask = nsamples==0
    sums[mask] = np.nan
    sums[~mask] /= nsamples[~mask]

    return data.update(
        data=sums,
        nsamples=nsamples,
        flags={},
        freq_array=data.freq_array[decimate_at::decimate],
    )


@gsregister("reduce")
def freq_bin(
    data: GSData,
    resolution: int | float,
    model: mdl.Model | None = None,
) -> GSData:
    """Frequency binning that auto-selects which kind of binning to do.

    Parameters
    ----------
    data
        The :class:`GSData` object to bin.
    resolution
        The resolution of the binning in MHz or number of existing channels (if int).
    model
        A :class:`edges_cal.modelling.Model` object to use for de-biasing the mean. If
        the ``data`` object has a ``data_model`` defined, this will be used instead.
        If not provided, simple direct binning will be used.
    """
    if data.data_model is None and model is None:
        return freq_bin_direct(data, resolution)
    else:
        return freq_bin_with_models(data, resolution, model)
