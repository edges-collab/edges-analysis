from typing import Tuple, Optional, Union, Type
from datetime import datetime, timedelta
from multiprocess import Pool, current_process, cpu_count
from multiprocessing.sharedctypes import RawArray

import numpy as np
from edges_cal import modelling as mdl, xrfi
import warnings
import logging
from collections import defaultdict

logger = logging.getLogger(__name__)

_globals = {}


def _init_worker(spectrum, weights, shape):
    # This just shoves things into _globals so that each worker in a pool hass access
    # to them. If they are in shared memory space (such as a RawArray), then they are
    # not copied to each process, just accessed therefrom.
    _globals["spectrum"] = spectrum
    _globals["weights"] = weights
    _globals["shape"] = shape


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

    m = mdl.ModelFit("linlog", fc / 200, tc, weights=wc, n_terms=n_poly).evaluate(f / 200)
    r = t - m

    return f, r


def non_stationary_weighted_average(
    data: np.ndarray,
    x: np.ndarray,
    weights: [None, np.ndarray] = None,
    model_fit: [callable, None] = None,
    model: [mdl.Model, None, str] = "polynomial",
    quick=False,
    n_terms: int = 5,
) -> [float, np.ndarray]:
    """
    Perform a weighted average over non-stationary data.

    This function does an unbiased weighted average for data whose mean (and potentially
    variance) vary throughout the data. See LoCo Memo #183 for details:
    http://loco.lab.asu.edu/wp-content/uploads/2020/10/averaging_with_weights.pdf

    This function always averages over the last axis.

    Parameters
    ----------
    data
        The data to be averaged. Can be of arbitrary dimension.
    x
        The co-ordinate of the data along the dimension to be averaged.
    weights
        The weight of each data point. Must be the same shape as `data`.
    model_fit
        If provided, a callable which may be passed `x` and return an array of the
        same shape which models the data. Not very useful if the number of dimensions
        is more than one (unless each of the other dimensions has the same model). If
        set to 'zero', a normal weighted average is performed.
    model
        Used if `model_fit` is not provided. Defines a Model which is used to fit the
        the data.
    n_terms
        The number of terms to fit in the model.

    Returns
    -------
    avg
        An array (or float) of the weighted average.
    """
    if weights is None:
        weights = np.ones_like(data)

    assert weights.shape == data.shape, "weights and data must have the same shape"
    assert len(x) == data.shape[-1], "length of x must be the same as the last axis of data"

    if quick:
        return weighted_mean(data, weights=weights, axis=-1)[0]

    # Get a model.
    if not model_fit and isinstance(model, str):
        model = mdl.Model._models[model.lower()](n_terms=n_terms, default_x=x)

    if model_fit:
        m = model_fit(x)

    # Go through each vector.
    shape = data.shape[:-1]
    out = np.zeros(shape)
    for indx in np.ndindex(*shape):
        this_data = data[indx]
        this_weight = weights[indx]

        sw = np.sum(this_weight)
        if sw == 0:
            out[indx] = np.nan

        if not model_fit:
            m = mdl.ModelFit(model, ydata=this_data, weights=this_weight).evaluate()

        res = this_data - m

        out[indx] = np.mean(m) + np.sum(res * this_weight) / sw

    if out.size == 1:
        return float(out)
    else:
        return out


def non_stationary_bin_avg(
    data: np.ndarray,
    x: np.ndarray,
    bins: np.ndarray,
    weights: [None, np.ndarray] = None,
    model_fit: [callable, None] = None,
    model: [mdl.Model, None, str] = "polynomial",
    n_terms: int = 5,
    per_bin_model: bool = False,
    quick=False,
) -> np.ndarray:
    """
    Perform a weighted average over non-stationary data.

    This function does an unbiased weighted average for data whose mean (and potentially
    variance) vary throughout the data. See LoCo Memo #183 for details:
    http://loco.lab.asu.edu/wp-content/uploads/2020/10/averaging_with_weights.pdf

    This function always averages over the last axis.

    Parameters
    ----------
    data
        The data to be averaged. Can be of arbitrary dimension.
    x
        The co-ordinate of the data along the dimension to be averaged.
    bins
        1D array of bin edges into which to make averages.
    weights
        The weight of each data point. Must be the same shape as `data`.
    model_fit
        If provided, a callable which may be passed `x` and return an array of the
        same shape which models the data. Not very useful if the number of dimensions
        is more than one (unless each of the other dimensions has the same model).
    model
        Used if `model_fit` is not provided. Defines a Model which is used to fit the
        the data.
    n_terms
        The number of terms to fit in the model.
    per_bin_model
        Whether to make a model independently in each bin, or a single model for the
        entire data. The latter is faster, but may be prone to bias if the underlying
        data model is more complex than the input model.

    Returns
    -------
    avg
        An array of length bins - 1 with the binned average.
    """
    out = np.zeros(data.shape[:-1] + (len(bins) - 1,))

    if per_bin_model or quick:
        for i in range(len(bins[:-1])):
            mask = (x >= bins[i]) & (x < bins[i + 1])

            out[..., i] = non_stationary_weighted_average(
                data=data[..., mask],
                x=x[mask],
                weights=weights[..., mask] if weights is not None else None,
                model_fit=model_fit,
                model=model,
                n_terms=n_terms,
                quick=quick,
            )
    else:
        if not model_fit and isinstance(model, str):
            model = mdl.Model._models[model.lower()](n_terms=n_terms, default_x=x)

        if weights is None:
            weights = np.ones_like(data)

        for indx in np.ndindex(*data.shape[:-1]):
            this_model = (
                model_fit or mdl.ModelFit(model, ydata=data[indx], weights=weights[indx]).evaluate
            )
            this_data = data[indx]
            this_wght = weights[indx]

            for i in range(len(bins[:-1])):
                mask = (x >= bins[i]) & (x < bins[i + 1])
                this_indx = indx + (i,)
                if np.all(this_wght[mask] == 0):
                    out[this_indx] = np.nan
                else:
                    out[this_indx] = non_stationary_weighted_average(
                        this_data[mask],
                        x=x[mask],
                        model_fit=this_model,
                        weights=this_wght[mask],
                        quick=quick,
                    )

    return out


def get_binned_weights(
    x: np.ndarray, bins: np.ndarray, weights: [None, np.ndarray] = None
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


def bin_array(
    data: np.ndarray,
    weights: Optional[np.ndarray] = None,
    coords: Optional[np.ndarray] = None,
    axis: int = -1,
    bins: Optional[Union[np.ndarray, int, float]] = None,
) -> [np.ndarray, np.ndarray, np.ndarray]:
    """Bin arbitrary-dimension data carefully along an axis.

    There are multiple ways to "bin" data along an axis when provided with weights.
    It is not typically accurate to return equi-spaced bins where data is averaged simply
    via summing with the weights (and the bin coords represent the centre of each bin).
    This results in some bias when the weights are not uniform.

    One way around this is to assume some underlying model and "fill in" the lower-weight
    bins. This would allow equi-spaced estimates.

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
        end of coords. If not provided, assume a single bin encompassing the all the
        data.

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

    if bins is None:
        bins = np.array([coords[0], coords[-1] + 0.1])
    elif isinstance(bins, int):
        bins = np.concatenate((coords[::bins], [coords[-1] + 0.1]))
    elif isinstance(bins, float):
        bins = np.concatenate((np.arange(coords[0], coords[-1], bins), [coords[-1] + 0.1]))

    # Get a list of tuples of bin edges
    bins = [(b, bins[i + 1]) for i, b in enumerate(bins[:-1])]

    # Generate the shape of the outputs by contracting one axis.
    out_shape = tuple(d if i != axis else len(bins) for i, d in enumerate(data.shape))

    out_data = np.ones(out_shape) * np.nan
    out_wght = np.zeros(out_shape)
    out_coords = np.ones(out_shape) * np.nan
    init_shape = tuple(data.shape[-1] if i == axis else d for i, d in enumerate(data.shape[:-1]))

    for i, (lower, upper) in enumerate(bins):
        mask = np.where((coords >= lower) & (coords < upper))[0]
        if len(mask) > 0:
            this_data = data.take(mask, axis=axis)
            this_wght = weights.take(mask, axis=axis)

            this_crd = np.swapaxes(
                np.broadcast_to(coords[mask], init_shape + (len(coords[mask]),)), axis, -1
            )

            this_slice = tuple(slice(None) if ax != axis else i for ax in range(data.ndim))
            out_data[this_slice], out_wght[this_slice] = weighted_mean(
                this_data, this_wght, axis=axis
            )

            out_coords[this_slice], _ = weighted_mean(this_crd, this_wght, axis=axis)

    return out_coords, out_data, out_wght


def unbiased_freq_bin(
    model_type: Type[mdl.Model],
    params: np.ndarray,
    freq: np.ndarray,
    resids: np.ndarray,
    weights: np.ndarray,
    new_freq_edges: np.ndarray,
    **kwargs,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
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
    new_freq_edges
        The new frequency bins.
    kwargs
        Anything else passed to construct the model.

    Returns
    -------
    freq
        The new frequencies
    weights
        The binned weights
    spec
        The binned spectrum
    resid
        The new residuals to the same model
    params
        The new parameters of the same model
    """

    new_f = (new_freq_edges[1:] + new_freq_edges[:-1]) / 2

    model = model_type(default_x=new_f, **kwargs)

    ev = np.array([model(parameters=p) for p in params])

    _, r, w = bin_array(
        resids,
        weights=weights,
        coords=freq,
        bins=new_freq_edges,
    )
    s = ev + r

    new_r = []
    new_p = []

    for ss, ww in zip(s, w):
        m = model.fit(ss, ww)
        new_r.append(m.residual)
        new_p.append(m.model_parameters)

    return new_f, w, s, new_r, new_p


def average_in_frequency(
    spectrum: [list, np.ndarray],
    freq: [list, np.ndarray, None] = None,
    weights: [list, np.ndarray, None] = None,
    resolution: [float, None, int] = None,
    axis: int = -1,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Average a spectrum, with weights, in frequency.

    The average is optionally taken within bins along the frequency axis.

    Parameters
    ----------
    spectrum : array-like
        The spectrum to average. Fre
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
    >>> f, s, w, std = average_in_frequency(spectrum, freq=freq, resolution=0.2)
    >>> f
    [0.15, 0.35, 0.55, 0.75, 0.95]
    >>> s
    [1, 1, 1, 1, 1]
    >>> w
    [1, 1, 1, 1, 1]

    """
    if axis < 0:
        axis += spectrum.ndim

    if resolution and freq is None:
        raise ValueError("You must provide freq if resolution or n_samples is provided!")

    nf = spectrum.shape[axis]

    if isinstance(resolution, float):
        n_samples = max(int(resolution / (freq[1] - freq[0])), 1)
    elif isinstance(resolution, int):
        n_samples = resolution
    else:
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

        # Set zeros to NaN to avoid divide by zero error.
        last_w[last_w == 0] = np.nan

        ss, ww = weighted_mean(last_s, last_w, axis=axis)
        last_std = weighted_standard_deviation(
            np.expand_dims(ss, axis), last_s, std=np.sqrt(1.0 / last_w), axis=axis
        )
        last_s = ss
        last_w = ww

    rng = range(nf - mod)
    # Get the main part of the array (without trailing bin)
    f = freq[rng]
    s = spectrum.take(rng, axis=axis)
    w = weights.take(rng, axis=axis)

    # Reshape the array so that the binning axis is split
    f = np.reshape(f, (-1, n_samples))
    s = np.reshape(s, s.shape[:axis] + (-1, n_samples) + s.shape[(axis + 1) :])
    w = np.reshape(w, w.shape[:axis] + (-1, n_samples) + w.shape[(axis + 1) :])

    # Set zeros to NaN to avoid divide by zero error.
    w[w == 0] = np.nan

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


def weighted_sum(data, weights=None, normalize=False, axis=0):
    """A careful weighted sum.

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

    sum = np.nansum(data * weights, axis=axis)
    weights = np.nansum(weights, axis=axis)

    if hasattr(sum, "__len__"):
        sum[weights == 0] = np.nan
    elif weights == 0:
        sum = np.nan

    return sum, weights


def weighted_mean(data, weights=None, axis=0):
    """A careful weighted mean where zero-weights don't error.

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
    sum, weights = weighted_sum(data, weights, axis=axis)

    mask = weights > 0
    if isinstance(sum, float):
        return sum / weights if mask else np.nan, weights
    av = np.zeros_like(sum)
    av[mask] = sum[mask] / weights[mask]
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
    return np.sqrt(weighted_mean((data - av) ** 2, 1 / std ** 2, axis=axis)[0])


def run_xrfi_pipe(
    *,
    spectrum: np.ndarray,
    freq: np.ndarray,
    xrfi_pipe: dict,
    weights: Optional[np.ndarray] = None,
    flags: Optional[np.ndarray] = None,
    n_threads: int = cpu_count(),
    fl_id=None,
) -> np.ndarray:
    """Run an xrfi pipeline on given spectrum and weights, updating weights in place."""
    for method, kwargs in xrfi_pipe.items():
        rfi = getattr(xrfi, method)

        if weights is None:
            if flags is None:
                weights = np.ones_like(spectrum)
            else:
                weights = (~flags).astype(float)

        if flags is not None:
            weights = np.where(flags, 0, weights)

        if spectrum.ndim in rfi.ndim:
            flags = getattr(xrfi, method)(spectrum, weights=weights, **kwargs)[0]
        elif spectrum.ndim > max(rfi.ndim) + 1:
            # say we have a 3-dimensional spectrum but can only do 1D in the method.
            # then we collapse to 2D and recursively run xrfi_pipe. That will trigger
            # the *next* clause, which will do parallel mapping over the first axis.
            orig_shape = spectrum.shape
            new_shape = (-1,) + orig_shape[2:]
            flags = run_xrfi_pipe(
                spectrum=spectrum.reshape(new_shape),
                weights=weights.reshape(new_shape),
                freq=freq,
                xrfi_pipe={method: kwargs},
                n_threads=n_threads,
            )
            return flags.reshape(orig_shape)
        else:
            n_threads = min(n_threads, len(spectrum))

            # Use a parallel map unless this function itself is being called by a
            # parallel map.
            wrns = defaultdict(lambda: 0)

            def count_warnings(message, *args, **kwargs):
                wrns[str(message)] += 1

            old = warnings.showwarning
            warnings.showwarning = count_warnings

            if current_process().name == "MainProcess" and n_threads > 1:

                def fnc(i):
                    # Gets the spectrum/weights from the global var dict, which was initialized
                    # by the pool. See https://research.wmz.ninja/articles/2018/03/on-sharing-large-arrays-when-using-pythons-multiprocessing.html
                    spec = np.frombuffer(_globals["spectrum"]).reshape(_globals["shape"])[i]
                    wght = np.frombuffer(_globals["weights"]).reshape(_globals["shape"])[i]

                    if np.any(wght > 0):
                        return rfi(spec, freq=freq, weights=wght, **kwargs)[0]
                    else:
                        return np.ones_like(spec, dtype=bool)

                shared_spectrum = RawArray("d", spectrum.size)
                shared_weights = RawArray("d", spectrum.size)

                # Wrap X as an numpy array so we can easily manipulates its data.
                shared_spectrum_np = np.frombuffer(shared_spectrum).reshape(spectrum.shape)
                shared_weights_np = np.frombuffer(shared_weights).reshape(spectrum.shape)

                # Copy data to our shared array.
                np.copyto(shared_spectrum_np, spectrum)
                np.copyto(shared_weights_np, weights)

                p = Pool(
                    n_threads,
                    initializer=_init_worker,
                    initargs=(shared_spectrum, shared_weights, spectrum.shape),
                )
                m = p.map
            else:

                def fnc(i):
                    if np.any(weights[i] > 0):
                        return rfi(spectrum[i], freq=freq, weights=weights[i], **kwargs)[0]
                    else:
                        return np.ones_like(spectrum[i], dtype=bool)

                m = map

            results = m(fnc, range(len(spectrum)))
            flags = np.array(list(results))

            warnings.showwarning = old

            # clear global memory (not sure if it still exists)
            _init_worker(0, 0, 0)

            fl_id = f"{fl_id}: " if fl_id else ""

            if wrns:
                for msg, count in wrns.items():
                    msg = msg.replace("\n", " ")
                    logger.warning(f"{fl_id}Received warning '{msg}' {count}/{len(flags)} times.")

        print(flags.dtype)
        logger.info(
            f"{fl_id}After {method}, nflags={np.sum(flags)}/{flags.size}"
            f" ({100*np.sum(flags)/flags.size:.1f}%)"
        )

    return flags


def model_bin_gha(
    params: np.ndarray, resids: np.ndarray, weights: np.ndarray, gha: np.ndarray, bins: np.ndarray
) -> [np.ndarray, np.ndarray, np.ndarray]:
    """
    Bin data in an unbiased way using a model fit.

    See memo #183:
    http://loco.lab.asu.edu/wp-content/uploads/2020/10/averaging_with_weights.pdf)

    Data can be in the form of multiple observations, each of which has a "model" fit
    to it (eg. multiple GHA's with a model over frequency for each).

    See
    Parameters
    ----------
    params
        Model parameters for each data point. Shape should be (Nobs, Nterms).
    resids
        Residuals of the models to data. Shape should be (Nobs, Ncoords)
    weights
        Weights of the data/residuals. Shape should be same as resids.
    coords
        The coordinates to bin.
    bins
        The bins into which the coords should be fit
    axis
        The axis over which to bin.

    Returns
    -------
    params
        An ndarray giving the averaged parameters of the model
    resids
        An ndarray giving residuals in the averaged bins.
    weights
        The new weights after averaging.
    """
    params_out = np.nan * np.ones((len(bins) - 1, params.shape[-1]))
    resids_out = np.nan * np.ones((len(bins) - 1, resids.shape[-1]))
    weights_out = np.zeros_like(resids_out)

    for i, bin_low in enumerate(bins[:-1]):
        bin_high = bins[i + 1]

        mask = (gha >= bin_low) & (gha < bin_high)
        if np.sum(mask) == 0:
            # Skip this bin if nothing's in it
            continue

        these_params = params[mask]
        these_resids = resids[mask]
        these_weights = weights[mask]

        # Take the nanmean, because some entire integrations/GHA's might have been flagged
        # and therefore have no applicable model. Then the params should be NaN. A nanmean
        # of all NaNs returns NaN, so that makes sense.
        params_out[i] = np.nanmean(these_params, axis=0)
        resids_out[i], weights_out[i] = weighted_mean(these_resids, weights=these_weights, axis=0)

    return params_out, resids_out, weights_out
