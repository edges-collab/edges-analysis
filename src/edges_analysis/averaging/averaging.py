"""Methods for averaging arrays.

There are multiple methods in this module, due to the need for careful averages/binning
done in different ways. There are ultimately three axes over which we might bin spectra:
nights, LST/GHA and frequency. Each of these in fact requires slightly different methods
for averaging, in order to make the average unbiased (given flags).
"""

from __future__ import annotations

import contextlib

import numpy as np
from astropy import units as un
from edges_cal import modelling as mdl

from ..tools import slice_along_axis


def get_binned_weights(
    x: np.ndarray, bins: np.ndarray, weights: np.ndarray | None = None
) -> np.ndarray:
    """
    Get the total weight in each bin for a given vector.

    Parameters
    ----------
    x
        The input co-ordinates (1D).
    bins
        The bins into which to bin the x.
    weights
        Array with last dimension the same length as x. Input weights.

    Returns
    -------
    weights
        Output bin weights.
    """
    out = np.zeros(weights.shape[:-1] + (len(bins) - 1,))

    indices = np.digitize(x, bins) - 1
    indices[indices >= (len(bins) - 1)] -= 1

    for indx in np.ndindex(*out.shape[:-1]):
        out[indx] = np.bincount(indices, weights=weights[indx], minlength=out.shape[-1])

    return out


def get_bin_edges(
    coords: np.ndarray,
    bins: np.ndarray | un.Quantity | int | float | None = None,
    start: float | un.Quantity | None = None,
    stop: float | un.Quantity | None = None,
) -> np.ndarray:
    """Get bin edges given input coordinates and a simple description of the binning.

    Parameters
    ----------
    coords
        The input co-ordinates to bin. These must be regular and monotonically
        increasing.
    bins
        The bin *edges* (lower inclusive, upper not inclusive). If an ``int``, simply
        use ``bins`` samples per bin, starting from the first bin. If a float, use
        equi-spaced bin edges, starting from the start of coords, and ending past the
        end of coords. If not provided, assume a single bin encompassing all the data.
    """
    unit = getattr(coords, "unit", 1)
    coords = getattr(coords, "value", coords)

    diffs = np.diff(coords)
    dx = np.median(diffs)
    if not np.all(diffs > 0):
        raise ValueError("coords must be monotonically increasing!")

    if not np.allclose(diffs, dx, rtol=1e-3):
        raise ValueError("coords must be regularly spaced!")

    start = coords[0] - dx / 2 if start is None else getattr(start, "value", start)

    stop = coords[-1] + dx / 2 if stop is None else getattr(stop, "value", stop)

    if bins is None:
        return np.array([start, stop]) * unit
    elif not isinstance(bins, un.Quantity) and hasattr(bins, "__len__"):
        return np.array(bins)
    elif isinstance(bins, un.Quantity) and not bins.isscalar:
        return bins
    elif isinstance(bins, int):
        # works if its an integer
        if len(coords) % bins == 0:
            edges = coords[::bins] - dx / 2
            return np.concatenate((edges, [coords[-1] + dx])) * unit
        else:
            return (coords[::bins] - dx / 2) * unit
    else:
        if unit != 1:
            with contextlib.suppress(AttributeError):
                bins = bins.to_value(unit)
        return np.arange(start, stop, bins) * unit


def bin_array_biased_regular(
    data: np.ndarray,
    weights: np.ndarray | None = None,
    coords: np.ndarray | None = None,
    axis: int = -1,
    bins: np.ndarray | int | float | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Bin arbitrary-dimension data carefully along an axis.

    There are multiple ways to "bin" data along an axis when provided with weights.
    It is not typically accurate to return equi-spaced bins where data is averaged
    simply via summing with the weights (and the bin coords represent the centre of each
    bin). This results in some bias when the weights are not uniform.

    One way around this is to assume some underlying model and "fill in" the
    lower-weight bins. This would allow equi-spaced estimates.

    However, this function does something simpler -- it returns non-equi-spaced bins.
    This can be a little annoying if multiple data are to be binned, because one needs
    to keep track of the coordinates of each data separately. However, it is simple
    and accurate.

    Parameters
    ----------
    data
        The data to be binned. May be of arbitrary dimension.
    weights
        The weights of the data. Must be the same shape as ``data``. If not provided,
        assume all weights are unity.
    coords
        The coordinates of the data along the axis to be averaged. If not provided,
        is taken to be the indices over the axis.
    axis
        The axis over which to bin.
    bins
        The bin *edges* (lower inclusive, upper not inclusive). If an ``int``, simply
        use ``bins`` samples per bin, starting from the first bin. If a float, use
        equi-spaced bin edges, starting from the start of coords, and ending past the
        end of coords. If not provided, assume a single bin encompassing all the data.

    Returns
    -------
    coords
        The weighted average of the coordinates in each bin. If there is no weight
        in a bin
    """
    axis %= data.ndim

    if weights is None:
        weights = np.ones(data.shape, dtype=float)

    if data.shape != weights.shape:
        raise ValueError("data and weights must have same shape")

    if coords is None:
        coords = np.arange(data.shape[axis])

    if len(coords) != data.shape[axis]:
        raise ValueError("coords must be same length as the data along the given axis.")

    bins = get_bin_edges(coords, bins)

    # Get a list of tuples of bin edges
    bins = [(b, bins[i + 1]) for i, b in enumerate(bins[:-1])]

    # Generate the shape of the outputs by contracting one axis.
    out_shape = tuple(d if i != axis else len(bins) for i, d in enumerate(data.shape))

    out_data = np.ones(out_shape) * np.nan
    out_wght = np.zeros(out_shape)

    for i, (lower, upper) in enumerate(bins):
        mask = np.where((coords >= lower) & (coords < upper))[0]
        if len(mask) > 0:
            this_data = data.take(mask, axis=axis)
            this_wght = weights.take(mask, axis=axis)

            this_slice = tuple(
                slice(None) if ax != axis else i for ax in range(data.ndim)
            )
            out_data[this_slice], out_wght[this_slice] = weighted_mean(
                this_data, this_wght, axis=axis
            )

    centres = [(b[0] + b[1]) / 2 for b in bins]
    if hasattr(centres[0], "unit"):
        centres = np.array([c.value for c in centres]) * centres[0].unit
    else:
        centres = np.array(centres)

    return centres, out_data, out_wght


def bin_array_unbiased_irregular(
    data: np.ndarray,
    weights: np.ndarray | None = None,
    coords: np.ndarray | None = None,
    axis: int = -1,
    bins: np.ndarray | int | float | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Bin arbitrary-dimension data carefully along an axis.

    There are multiple ways to "bin" data along an axis when provided with weights.
    It is not typically accurate to return equi-spaced bins where data is averaged
    simply via summing with the weights (and the bin coords represent the centre of each
    bin). This results in some bias when the weights are not uniform.

    One way around this is to assume some underlying model and "fill in" the
    lower-weight bins. This would allow equi-spaced estimates.

    However, this function does something simpler -- it returns non-equi-spaced bins.
    This can be a little annoying if multiple data are to be binned, because one needs
    to keep track of the coordinates of each data separately. However, it is simple
    and accurate.

    Parameters
    ----------
    data
        The data to be binned. May be of arbitrary dimension.
    weights
        The weights of the data. Must be the same shape as ``data``. If not provided,
        assume all weights are unity.
    coords
        The coordinates of the data along the axis to be averaged. If not provided,
        is taken to be the indices over the axis.
    axis
        The axis over which to bin.
    bins
        The bin *edges* (lower inclusive, upper not inclusive). If an ``int``, simply
        use ``bins`` samples per bin, starting from the first bin. If a float, use
        equi-spaced bin edges, starting from the start of coords, and ending past the
        end of coords. If not provided, assume a single bin encompassing all the data.

    Returns
    -------
    coords
        The weighted average of the coordinates in each bin. If there is no weight
        in a bin
    """
    axis %= data.ndim

    if weights is None:
        weights = np.ones(data.shape, dtype=float)

    if data.shape != weights.shape:
        raise ValueError("data and weights must have same shape")

    if coords is None:
        coords = np.arange(data.shape[axis])

    if len(coords) != data.shape[axis]:
        raise ValueError("coords must be same length as the data along the given axis.")

    bins = get_bin_edges(coords, bins)

    # Get a list of tuples of bin edges
    bins = [(b, bins[i + 1]) for i, b in enumerate(bins[:-1])]

    # Generate the shape of the outputs by contracting one axis.
    out_shape = tuple(d if i != axis else len(bins) for i, d in enumerate(data.shape))

    out_data = np.ones(out_shape) * np.nan
    out_wght = np.zeros(out_shape)
    out_coords = np.ones(out_shape) * np.nan
    init_shape = tuple(
        data.shape[-1] if i == axis else d for i, d in enumerate(data.shape[:-1])
    )

    for i, (lower, upper) in enumerate(bins):
        mask = np.where((coords >= lower) & (coords < upper))[0]
        if len(mask) > 0:
            this_data = data.take(mask, axis=axis)
            this_wght = weights.take(mask, axis=axis)

            this_crd = np.swapaxes(
                np.broadcast_to(coords[mask], (*init_shape, len(coords[mask]))),
                axis,
                -1,
            )

            this_slice = tuple(
                slice(None) if ax != axis else i for ax in range(data.ndim)
            )
            out_data[this_slice], out_wght[this_slice] = weighted_mean(
                this_data, this_wght, axis=axis
            )

            out_coords[this_slice], _ = weighted_mean(this_crd, this_wght, axis=axis)

    return out_coords, out_data, out_wght


def bin_freq_unbiased_irregular(
    spectrum: list | np.ndarray,
    freq: list | np.ndarray | None = None,
    weights: list | np.ndarray | None = None,
    resolution: float | int | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Average a spectrum, with weights, in frequency.

    The average is optionally taken within bins along the frequency axis.
    Under the hood, uses :func:`bin_array_unbiased_irregular`, just adding some nicer
    call parameters.

    Parameters
    ----------
    spectrum : array-like
        The spectrum to average. Frequency axis is the last axis.
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

    Returns
    -------
    freq
        An array with length determined automatically by the routine, giving the
        mean frequency in each output bin. Note that the frequencies in each row
        may be different.
    spec
        Array of same length as ``freq`` containing the weighted-average spectrum
    w : array
        Array of same length as ``freq`` containing the total weight in each bin.

    Examples
    --------
    >>> freq = np.linspace(0.1, 1, 10)
    >>> spectrum = [0, 2] * 5
    >>> f, s, w = bin_freq_unbiased_irregular(spectrum, freq=freq, resolution=0.2)
    >>> f
    [0.15, 0.35, 0.55, 0.75, 0.95]
    >>> s
    [1, 1, 1, 1, 1]
    >>> w
    [1, 1, 1, 1, 1]

    """
    if resolution and freq is None:
        raise ValueError("You must provide freq if resolution is provided!")

    if freq is not None and len(freq) != spectrum.shape[-1]:
        raise ValueError(
            f"provided freq ({len(freq)}) does not match final axis of spectrum "
            f"{spectrum.shape}!"
        )

    out_freq, out_spec, out_wght = bin_array_unbiased_irregular(
        spectrum, weights=weights, coords=freq, bins=resolution
    )

    return out_freq, out_spec, out_wght


def bin_freq_unbiased_regular(
    model: mdl.Model,
    params: np.ndarray,
    freq: np.ndarray,
    resids: np.ndarray,
    weights: np.ndarray,
    resolution: float | int | None = None,
    **fit_kwargs,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Bin an array along the frequency axis into *regular* bins.

    To make the bins regular, we use a model over the frequency axis to evaluate central
    value, and add the residuals to it.

    Parameters
    ----------
    model_type
        The model for which the residuals are defined.
    params
        2D array of parameters (parameters for each model along last axis).
    freq
        The frequencies of the input
    resids
        The residuals of the input model
    weights
        The weights of the data.
    resolution
        The resolution of the new data.
    fit_kwargs
        Anything passed to construct the :class:`ModelFit` instance, example ``method``.

    Returns
    -------
    freq
        The new frequencies
    weights
        The binned weights
    spec
        The binned spectrum
    resids
        The new residuals to the same model
    params
        The new parameters of the same model
    """
    if resolution is None or resolution:
        new_f_edges = get_bin_edges(freq, resolution)
        new_f = (new_f_edges[1:] + new_f_edges[:-1]) / 2
    else:
        new_f = freq

    model = model.at(x=new_f)

    ev = np.array([model(parameters=p) for p in params])

    if resolution != 0:
        _, r, w = bin_array_unbiased_irregular(
            resids,
            weights=weights,
            coords=freq,
            bins=resolution,
        )
    else:
        r = resids
        w = weights

    s = ev + r

    new_r = []
    new_p = []

    for ss, ww in zip(s, w):
        m = model.fit(ss, ww, **fit_kwargs)
        new_r.append(m.residual)
        new_p.append(m.model_parameters)

    return new_f, w, s, np.array(new_r), np.array(new_p)


def bin_gha_unbiased_regular(
    params: np.ndarray,
    resids: np.ndarray,
    weights: np.ndarray,
    gha: np.ndarray,
    bins: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Bin data in an unbiased way using a model fit.

    See memo #183:
    http://loco.lab.asu.edu/wp-content/uploads/2020/10/averaging_with_weights.pdf)

    Data can be in the form of multiple observations, each of which has a "model" fit
    to it (eg. multiple GHA's with a model over frequency for each).

    Parameters
    ----------
    params
        Model parameters for each data point. Shape should be (Nobs, Nterms).
    resids
        Residuals of the models to data. Shape should be (Nobs, Ncoords)
    weights
        Weights of the data/residuals. Shape should be same as resids.
    gha
        The GHA coordinates to bin.
    bins
        The bins into which the GHA should be fit


    Returns
    -------
    params
        An array giving the averaged parameters of the model
    resids
        An array giving residuals in the averaged bins.
    weights
        The new weights after averaging.
    """
    params_out = np.nan * np.ones((len(bins) - 1, params.shape[-1]))
    resids_out = np.nan * np.ones((len(bins) - 1, resids.shape[-1]))
    weights_out = np.zeros_like(resids_out)
    gha %= 24

    assert gha.ndim == 1

    for i, bin_low in enumerate(bins[:-1]):
        bin_high = bins[i + 1]
        bin_low %= 24
        bin_high %= 24

        if bin_low < bin_high:
            mask = (gha >= bin_low) & (gha < bin_high)
        else:
            mask = (gha < bin_high) | (gha >= bin_low)

        if np.sum(mask) == 0:
            # Skip this bin if nothing's in it
            continue

        these_params = params[mask]
        these_resids = resids[mask]
        these_weights = weights[mask]

        # Take the nanmean, because some entire integrations/GHA's might have been
        # flagged and therefore have no applicable model. Then the params should be NaN.
        # A nanmean of all NaNs returns NaN, so that makes sense.
        params_out[i] = np.nanmean(these_params, axis=0)
        resids_out[i], weights_out[i] = weighted_mean(
            these_resids, weights=these_weights, axis=0
        )

    return params_out, resids_out, weights_out


def bin_spectrum_unbiased_regular(
    params: np.ndarray,
    resids: np.ndarray,
    weights: np.ndarray,
    gha: np.ndarray,
    gha_bins: np.ndarray,
    model: mdl.Model,
    freq: np.ndarray,
    resolution: float | int | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Bin a spectrum in GHA and frequency in an unbiased and regular manner."""
    p, r, w = bin_gha_unbiased_regular(
        params=params, resids=resids, weights=weights, gha=gha, bins=gha_bins
    )

    new_f, w, s, new_r, new_p = bin_freq_unbiased_regular(
        model=model,
        params=p,
        freq=freq,
        resids=r,
        weights=w,
        resolution=resolution,
    )

    return new_f, new_r, w, s, new_p


def weighted_sum(data, weights=None, normalize=False, axis=0):
    """Perform a careful weighted sum.

    Parameters
    ----------
    data : array-like
        The data over which the weighted mean is to be taken.
    weights : array-like, optional
        Same shape as data, giving the weights of each datum.
    normalize : bool, optional
        If True, normalize weights so that the maximum weight is unity.
    axis : int, optional
        The axis over which to take the mean.

    Returns
    -------
    array-like :
        The weighted sum over `axis`, where elements with zero total weight are
        set to nan.
    """
    if weights is None:
        weights = np.ones_like(data)

    if normalize:
        weights = weights.copy() / weights.nanmax()

    sm = np.nansum(data * weights, axis=axis)
    weights = np.nansum(weights, axis=axis)

    if hasattr(sm, "__len__"):
        sm[weights == 0] = np.nan
    elif weights == 0:
        sm = np.nan

    return sm, weights


def weighted_mean(data, weights=None, axis=0):
    """Perform a careful weighted mean where zero-weights don't error.

    In this function, if the total weight is zero, np.nan is returned.

    Parameters
    ----------
    data : array-like
        The data over which the weighted mean is to be taken.
    weights : array-like, optional
        Same shape as data, giving the weights of each datum.
    axis : int, optional
        The axis over which to take the mean.

    Returns
    -------
    array-like :
        The weighted mean over `axis`, where elements with zero total weight are
        set to nan.
    """
    sm, weights = weighted_sum(data, weights, axis=axis)

    mask = weights > 0
    if isinstance(sm, float):
        return sm / weights if mask else np.nan, weights
    av = np.zeros_like(sm)
    av[mask] = sm[mask] / weights[mask]
    av[~mask] = np.nan
    return av, weights


def weighted_sorted_metric(data, weights=None, metric="median", **kwargs):
    """Semi-weighted integrator of data.

    This function will perform integrations of data that rely on sorting the data (eg.
    median or percentile). These are ony able to partial weighting -- i.e. weights of
    zero ensure that datum is ignored, while other weights all count the same.

    Parameters
    ----------
    data : array-like
        The data over which the weighted mean is to be taken.
    weights : array-like, optional
        Same shape as data, giving the weights of each datum.
    metric : str, optional
        One of ('median', 'argmax', 'argmin','max', 'min', 'percentile', 'quantile'),
        specifying which metric to take.
    kwargs :
        Extra arguments to the function np.nan<metric>.

    Returns
    -------
    array-like :
        The weighted mean over `axis`, where elements with zero total weight are
        set to nan.
    """
    assert metric in (
        "median",
        "argmax",
        "argmin",
        "max",
        "min",
        "percentile",
        "quantile",
    )
    d = data.copy()
    d[weights == 0] = np.nan
    return getattr(np, "nan" + metric)(d, **kwargs)


def weighted_standard_deviation(av, data, std, axis=0):
    """Calcualte a careful weighted standard deviation."""
    return np.sqrt(weighted_mean((data - av) ** 2, 1 / std**2, axis=axis)[0])


def bin_data(
    data: np.ndarray,
    residuals: np.ndarray | None = None,
    weights: np.ndarray | None = None,
    bins: list[np.ndarray | slice] | None = None,
    axis: int = -1,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Bin data, in an un-biased way if possible.

    This uses the estimator from memo #183:
    http://loco.lab.asu.edu/wp-content/uploads/2020/10/averaging_with_weights.pdf.

    Parameters
    ----------
    data
        The data to be binned.
    residuals
        The residuals of the data, if known. If not provided, and weights is non
        uniform, the average will be biased.
    weights
        The weights of the data. If not provided, assume all weights are unity.
    bins
        The bins into which to bin the data. If not provided, assume a single bin
        encompassing all the data. Each element should be either an array that
        indexes into the axis over which to bin, or a slice object.
    axis
        The axis over which to bin.

    Returns
    -------
    data
        The binned data.
    weights
        The weights of the binned data.
    residuals
        The binned residuals (if provided).
    """
    if residuals is not None:
        model = data - residuals

    if axis < 0:
        axis += data.ndim

    shape = list(data.shape)
    shape[axis] = len(bins)

    outd = np.zeros(tuple(shape), dtype=data.dtype)
    outw = np.zeros(tuple(shape), dtype=float)
    if residuals is not None:
        outr = np.zeros(tuple(shape), dtype=residuals.dtype)
    else:
        outr = None

    ell = (slice(None),) * axis

    for ibin, bn in enumerate(bins):
        w = slice_along_axis(weights, bn, axis=axis) if weights is not None else None

        if residuals is not None:
            m = slice_along_axis(model, bn, axis=axis)
            r = slice_along_axis(residuals, bn, axis=axis)

            m = weighted_mean(m, axis=axis)[0]
            r, w = weighted_mean(r, weights=w, axis=axis)
            d = r + m
        else:
            d = slice_along_axis(data, bn, axis=axis)
            d, w = weighted_mean(d, weights=w, axis=axis)

        outd[(*ell, ibin)] = d
        outw[(*ell, ibin)] = w
        if residuals is not None:
            outr[(*ell, ibin)] = r

    return outd, outw, outr
