"""Functions for averaging arrays.

There are multiple methods in this module, due to the need for careful averages/binning
done in different ways. There are ultimately three axes over which we might bin spectra:
nights, LST/GHA and frequency. Each of these in fact requires slightly different methods
for averaging, in order to make the average unbiased (given flags).
"""

import contextlib
import logging
import numbers

import numpy as np
from astropy import units as un

from ..tools import slice_along_axis

logger = logging.getLogger(__name__)


def get_binned_weights(
    x: np.ndarray,
    bins: np.ndarray,
    weights: np.ndarray | None = None,
    include_left: bool = True,
    include_right: bool = True,
) -> np.ndarray:
    """
    Get the total weight in each bin for a given vector.

    Parameters
    ----------
    x
        The input co-ordinates (1D).
    bins
        The bin edges into which to bin the x.
    weights
        Array with last dimension the same length as x. Input weights. Default is
        all ones.
    include_left
        Whether to include coordinates to the left of the minimum bin in the first bin.
        If False, such coordinates are ignored.
    include_right
        Whether to include coordinates to the right of the maximum bin in the last bin.
        Note that for historical reasons, this is True, but it should probably be set
        to False for typical cases. If False, such coordinates (and any NaN
        coordinates) are ignored.

    Returns
    -------
    weights
        Output bin weights, with shape equal to the input weights, but with the last
        dimension replaced by ``len(bins) - 1``.

    """
    bins = np.asarray(bins)
    x = np.asarray(x)

    if bins.size < 2:
        raise ValueError("Bin edges must have at least 2 elements.")

    if weights is None:
        weights = np.ones_like(x)

    if not np.all(np.diff(bins) > 0):
        raise ValueError("Bin edges must be monotonically increasing!")
    if weights.shape[-1] != len(x):
        raise ValueError("Weights must have the same last axis shape as x.")

    if include_right:
        # In this case, NaNs and Infs will get put in the right-most bin,
        # which is not what we want.
        mask = np.isfinite(x)
        x = x[mask]
        weights = weights[..., mask]

    out = np.zeros((*weights.shape[:-1], len(bins) - 1))

    indices = np.digitize(x, bins) - 1

    if include_left:
        indices[indices < 0] = 0
    if include_right:
        indices[indices >= (len(bins) - 1)] = len(bins) - 2

    # Drop any coordinates that remain outside the bins.
    in_range = (indices >= 0) & (indices < len(bins) - 1)
    indices = indices[in_range]
    weights = weights[..., in_range]

    for indx in np.ndindex(*out.shape[:-1]):
        out[indx] = np.bincount(indices, weights=weights[indx], minlength=out.shape[-1])

    return out


def get_bin_edges(
    coords: np.ndarray,
    bins: np.ndarray | un.Quantity | float | None = None,
    start: float | un.Quantity | None = None,
    stop: float | un.Quantity | None = None,
) -> np.ndarray | un.Quantity:
    """Get bin edges given input coordinates and a simple description of the binning.

    Parameters
    ----------
    coords
        The input co-ordinates to bin. These must be regular and monotonically
        increasing.
    bins
        The bin *edges* (lower inclusive, upper not inclusive). If an integer (including
        numpy integer types), simply use ``bins`` coords per bin, starting from the
        first bin. If the number of coords is not divisible by ``bins``, the trailing
        coords that do not fill a whole bin are dropped (and a warning is logged).
        If a float, use equi-spaced bin edges, starting from the start of coords, and
        ending past the end of coords. If an array, assumed to be the bin edges.
        If not provided, assume a single bin encompassing all the data.
    start
        Where to start the bin edges when ``bins`` is an int or float. Defaults to
        the first coordinate minus half of the median coordinate difference.
    stop
        Where to stop the bin edges when ``bins`` is an int or float. Defaults to
        the last coordinate plus half of the median coordinate difference.

    Returns
    -------
    np.ndarray
        The bin edges.

    Notes
    -----
    This function is robust to the input coordinates being astropy Quantities, and will
    return a quantity if the coordinates are.
    """
    unit = getattr(coords, "unit", 1)
    coords = getattr(coords, "value", coords)

    if coords.size < 2:
        raise ValueError("coords must have at least 2 elements.")

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
    if not isinstance(bins, un.Quantity) and hasattr(bins, "__len__"):
        return np.array(bins)
    if isinstance(bins, un.Quantity) and not bins.isscalar:
        return bins
    if isinstance(bins, numbers.Integral):
        if len(coords) % bins != 0:
            # This is logged rather than emitted with warnings.warn, since it is the
            # normal situation in some standard pipelines (e.g. binning 8193
            # channels by 8 in calibration).
            logger.warning(
                f"The number of coordinates per bin ({bins}) does not divide the "
                f"number of coordinates ({len(coords)}): the last {len(coords) % bins} "
                "coordinate(s) will not be in any bin."
            )
            return (coords[::bins] - dx / 2) * unit
        edges = coords[::bins] - dx / 2
        return np.concatenate((edges, [coords[-1] + dx / 2])) * unit
    if unit != 1:
        with contextlib.suppress(AttributeError):
            bins = bins.to_value(unit)
    return np.arange(start, stop + bins, bins) * unit


def weighted_sum(
    data: np.ndarray,
    weights: np.ndarray | None = None,
    normalize: bool = False,
    axis: int = -1,
    fill_value: float = np.nan,
    keepdims: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    """Perform a careful weighted sum.

    This routine is 'careful' in that it allows performing sums when the total weight
    is zero, setting those values to a user-defined filling value.

    Parameters
    ----------
    data : array-like
        The data over which the weighted mean is to be taken.
    weights : array-like, optional
        Same shape as data, giving the weights of each datum. By default,
        all weights are unity.
    normalize : bool, optional
        If True, normalize weights so that the maximum weight is unity.
    axis : int, optional
        The axis over which to take the mean.
    fill_value : float, optional
        The value to fill in where the sum of the weights is zero.
    keepdims : bool, optional
        Whether to keep the original dimensions of ``data`` (i.e. have an axis
        with length one for ``axis``).

    Returns
    -------
    datasum
        The weighted sum over `axis`, where elements with zero total weight are
        set to ``fill_value``.
    sumweights
        The sum of the weights over `axis`.
    """
    if weights is None:
        weights = np.ones_like(data)

    if data.shape != weights.shape:
        raise ValueError("data and weights must have the same shape.")

    if normalize:
        weights = weights.copy() / np.nanmax(weights)

    weights = np.where(np.isnan(data), 0, weights)
    sm = np.nansum(data * weights, axis=axis, keepdims=keepdims, dtype=float)
    weights = np.nansum(weights, axis=axis, keepdims=keepdims, dtype=float)

    if hasattr(sm, "__len__"):
        sm[weights == 0] = fill_value
    elif weights == 0:
        sm = fill_value

    return sm, weights


def weighted_mean(
    data: np.ndarray,
    weights: np.ndarray | None = None,
    axis: int = -1,
    fill_value: float = np.nan,
    keepdims: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    """Perform a careful weighted mean where zero-weights don't error.

    In this function, if the total weight is zero, fill_value is returned.

    Parameters
    ----------
    data : array-like
        The data over which the weighted mean is to be taken.
    weights : array-like, optional
        Same shape as data, giving the weights of each datum.
    axis : int, optional
        The axis over which to take the mean.
    fill_value : float, optional
        The value to fill in where the sum of the weights is zero.
    keepdims : bool, optional
        Whether to keep the original dimensions of ``data`` (i.e. have an axis
        with length one for ``axis``).

    Returns
    -------
    avg
        The weighted mean over `axis`, where elements with zero total weight are
        set to ``fill_value``.
    weights
        The sum of the weights over `axis`.
    """
    sm, weights = weighted_sum(
        data, weights, axis=axis, keepdims=keepdims, fill_value=fill_value
    )

    mask = weights > 0
    if not hasattr(sm, "__len__"):
        return (sm / weights, weights) if mask else (fill_value, weights)
    av = np.zeros_like(sm)
    av[mask] = sm[mask] / weights[mask]
    av[~mask] = fill_value
    return av, weights


def weighted_variance(
    data: np.ndarray,
    nsamples: np.ndarray | None = None,
    avg: np.ndarray | None = None,
    **kwargs,
):
    """Calculate a careful weighted variance.

    This computes::

        sum(n**2 * (data - avg)**2) / sum(n**2)

    over the given axis, where ``n`` is ``nsamples`` and ``avg`` is the
    ``nsamples``-weighted mean of the data (unless given).

    Parameters
    ----------
    data : array-like
        The data over which to calculate the variance.
    nsamples
        The number of samples corresponding to each datum. The mean is weighted by
        ``nsamples``, while the squared deviations are weighted by ``nsamples**2``.
        Default is all unity.
    avg
        The weighted average of the data over the given axis (with ``keepdims=True``).
        By default, compute this internally.
    **kwargs
        Passed to :func:`weighted_mean` (e.g. ``axis`` and ``fill_value``).

    Returns
    -------
    var
        The weighted variance of the data over the given axis. Only this array is
        returned (not the sum of the weights).

    Notes
    -----
    This is a plain weighted mean of the squared deviations. No correction is made for
    the degree of freedom used in estimating the mean (it is the analogue of
    ``np.var(..., ddof=0)``), so it is a biased estimator of the variance. Also, since
    the weights are ``nsamples**2``, it is not the inverse-variance-weighted estimate of
    the variance of a single sample when ``nsamples`` varies. With uniform
    ``nsamples``, it reduces to ``np.var(data, axis=axis)``.
    """
    if nsamples is None:
        nsamples = np.ones_like(data)

    if avg is None:
        avg, _ = weighted_mean(data, weights=nsamples, keepdims=True, **kwargs)

    return weighted_mean((data - avg) ** 2, nsamples**2, **kwargs)[0]


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

    if bins is None:
        bins = [slice(None)]

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
        outd[*ell, ibin] = d
        outw[*ell, ibin] = w
        if residuals is not None:
            outr[*ell, ibin] = r

    return outd, outw, outr


def bin_array_unweighted(x: np.ndarray, size: int = 1) -> np.ndarray:
    """Simple unweighted mean-binning of an array.

    Parameters
    ----------
    x
        The array to be binned. Only the last axis will be binned.
    size
        The size of the bins.

    Notes
    -----
    The last axis of `x` is binned. It is assumed that the coordinates corresponding
    to `x` are regularly spaced, so the final average just takes `size` values and
    averages them together.

    If the array is not divisible by `size`, the last values are left out.

    Examples
    --------
    Simple 1D example::

        >>> x = np.array([1, 1, 2, 2, 3, 3])
        >>> bin_array_unweighted(x, size=2)
        array([1., 2., 3.])

    The last remaining values are left out::

        >>> x = np.array([1, 1, 2, 2, 3, 3, 4])
        >>> bin_array_unweighted(x, size=2)
        array([1., 2., 3.])
    """
    if size == 1:
        return x

    n = x.shape[-1]
    nn = size * (n // size)
    return np.nanmean(x[..., :nn].reshape((*x.shape[:-1], -1, size)), axis=-1)
