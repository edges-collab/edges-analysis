"""Functions for excising RFI."""

from __future__ import annotations

import logging
import warnings
from dataclasses import dataclass, field
from functools import cached_property
from typing import Any, Literal

import h5py
import numpy as np
import yaml
from astropy.convolution import Box1DKernel, convolve_fft
from matplotlib import pyplot as plt
from scipy import ndimage

from .. import modelling as mdl
from .. import types as tp

logger = logging.getLogger(__name__)


class NoDataError(Exception):
    pass


def _check_convolve_dims(data, half_size: tuple[int] | None = None):
    """Check the kernel sizes to be used in various convolution-like operations.

    If the kernel sizes are too big, replace them with the largest allowable size
    and issue a warning to the user.

    .. note:: ripped from here:
              https://github.com/HERA-Team/hera_qm/blob/master/hera_qm/xrfi.py

    Parameters
    ----------
    data : array
        1- or 2-D array that will undergo convolution-like operations.
    half_size : tuple
        Tuple of ints or None's with length ``data.ndim``. They represent the half-size
        of the kernel to be used (or, rather the kernel will be 2*half_size+1 in each
        dimension). None uses half_size=data.shape.

    Returns
    -------
    size : tuple
        The kernel size in each dimension.

    Raises
    ------
    ValueError:
        If half_size does not match the number of dimensions.
    """
    if half_size is None:
        half_size = (None,) * data.ndim

    if len(half_size) != data.ndim:
        raise ValueError(
            "Number of kernel dimensions does not match number of data dimensions."
        )

    out = []
    for data_shape, hsize in zip(data.shape, half_size, strict=False):
        if hsize is None or hsize > data_shape:
            out.append(data_shape)
        elif hsize < 0:
            out.append(0)
        else:
            out.append(hsize)

    return tuple(out)


def robust_divide(num, den):
    """Prevent division by zero.

    This function will compute division between two array-like objects by setting
    values to infinity when the denominator is small for the given data type. This
    avoids floating point exception warnings that may hide genuine problems
    in the data.

    Parameters
    ----------
    num : array
        The numerator.
    den : array
        The denominator.

    Returns
    -------
    out : array
        The result of dividing num / den. Elements where b is small (or zero) are set
        to infinity.
    """
    thresh = np.finfo(den.dtype).eps

    den_mask = np.abs(den) > thresh

    out = np.true_divide(num, den, where=den_mask)
    out[~den_mask] = np.inf

    # If numerator is also small, set to zero (better for smooth stuff)
    out[~den_mask & (np.abs(num) <= thresh)] = 0
    return out


def flagged_filter(
    data: np.ndarray,
    size: int | tuple[int],
    kind: str = "median",
    flags: np.ndarray | None = None,
    mode: str | None = None,
    interp_flagged: bool = True,
    **kwargs,
):
    """
    Perform an n-dimensional filter operation on optionally flagged data.

    Parameters
    ----------
    data : np.ndarray
        The data to filter. Can be of arbitrary dimension.
    size : int or tuple
        The size of the filtering convolution kernel. If tuple, one entry per dimension
        in `data`.
    kind : str, optional
        The function to apply in each window. Typical options are `mean` and `median`.
        For this function to work, the function kind chosen here must have a
        corresponding `nan<function>` implementation in numpy.
    flags : np.ndarray, optional
        A boolean array specifying data to omit from the filtering.
    mode : str, optional
        The mode of the filter. See ``scipy.ndimage.generic_filter`` for details. By
        default, 'nearest' if size < data.size otherwise 'reflect'.
    interp_flagged : bool, optional
        Whether to fill in flagged entries with its filtered value. Otherwise,
        flagged entries are set to their original value.
    kwargs :
        Other options to pass to the generic filter function.

    Returns
    -------
    np.ndarray :
        The filtered array, of the same shape and type as ``data``.

    Notes
    -----
    This function can typically be used to implement a flagged median filter. It does
    have some limitations in this regard, which we will now describe.

    It would be expected that a perfectly smooth
    monotonic function, after median filtering, should remain identical to the input.
    This is only the case for the default 'nearest' mode. For the alternative 'reflect'
    mode, the edge-data will be corrupted from the input. On the other hand, it may be
    expected that if the kernel width is equal to or larger than the data size, that
    the operation is merely to perform a full collapse over that dimension. This is the
    case only for mode 'reflect', while again mode 'nearest' will continue to yield (a
    very slow) identity operation. By default, the mode will be set to 'reflect' if
    the size is >= the data size, with an emitted warning.

    Furthermore, a median filter is *not* an identity operation, even on monotonic
    functions, for an even-sized kernel (in this case it's the average of the two
    central values).

    Also, even for an odd-sized kernel, if using flags, some of the windows will contain
    an odd number of useable data, in which case the data surrounding the flag will not
    be identical to the input.

    Finally, flags near the edges can have strange behaviour, depending on the mode.
    """
    if mode is None:
        if (isinstance(size, int) and size >= min(data.shape)) or (
            isinstance(size, tuple)
            and any(s > d for s, d in zip(size, data.shape, strict=False))
        ):
            warnings.warn(
                "Setting default mode to reflect because a large size was set.",
                stacklevel=2,
            )
            mode = "reflect"
        else:
            mode = "nearest"

    if flags is not None and np.any(flags):
        fnc = getattr(np, "nan" + kind)
        assert flags.shape == data.shape
        orig_flagged_data = data[flags].copy()
        data[flags] = np.nan
        filtered = ndimage.generic_filter(data, fnc, size=size, mode=mode, **kwargs)
        if not interp_flagged:
            filtered[flags] = orig_flagged_data
        data[flags] = orig_flagged_data

    else:
        if kind == "mean":
            kind = "uniform"
        filtered = getattr(ndimage, kind + "_filter")(
            data, size=size, mode=mode, **kwargs
        )

    return filtered


def detrend_medfilt(
    data: np.ndarray,
    flags: np.ndarray | None = None,
    half_size: tuple[int | None] | None = None,
):
    """Detrend array using a median filter.

    .. note:: ripped from here:
              https://github.com/HERA-Team/hera_qm/blob/master/hera_qm/xrfi.py

    Parameters
    ----------
    data : array
        Data to detrend. Can be an array of any number of dimensions.
    flags : boolean array, optional
        Flags specifying data to ignore in the detrend. If not given, don't ignore
        anything.
    half_size : tuple of int/None
        The half-size of the kernel to convolve (kernel size will be 2*half_size+1).
        Value of zero (for any dimension) omits that axis from the kernel, effectively
        applying the detrending for each subarray along that axis. Value of None will
        effectively (but slowly) perform a median along the entire axis before running
        the kernel over the other axis.

    Returns
    -------
    out : array
        An array containing the outlier significance metric. Same type and size as
        `data`.

    Notes
    -----
    This detrending is very good for data with large RFI compared to the noise, but also
    reasonably large noise compared to the spectrum steepness. If the noise is small
    compared to the steepness of the spectrum, individual windows can become *almost
    always* monotonic, in which case the randomly non-monotonic bins "stick out" and get
    wrongly flagged. This can be helped three ways:

    1) Use a smaller bin width. This helps by reducing the probability that a bin will
       be randomly non-monotonic. However it also loses signal-to-noise on the RFI.
    2) Pre-fit a smooth model that "flattens" the spectrum. This helps by reducing the
       probability that bins will be monotonic (higher noise level wrt steepness). It
       has the disadvantage that fitted models can be wrong when there's RFI there.
    3) Follow the medfilt with a meanfilt: if the medfilt is able to flag most/all of
       the RFI, then a following meanfilt will tend to "unfilter" the wrongly flagged
       parts.
    """
    half_size = _check_convolve_dims(data, half_size)
    size = tuple(2 * s + 1 for s in half_size)

    d_sm = flagged_filter(data, size=size, kind="median", flags=flags)
    d_rs = data - d_sm
    d_sq = d_rs**2

    # Remember that d_sq will be zero for any window in which the data is monotonic (but
    # could also be zero for non-monotonic windows where the two halves of the window
    # are self-contained). Most smooth functions will be monotonic in small enough
    # windows. If noise is of low-enough amplitude wrt the steepness of the smooth
    # underlying function, there is a good chance the resulting data will also be
    # monotonic. Nevertheless, any RFI that is large enough will cause the value of
    # that channel to *not* be the central value, and it will have d_sq > 0.

    # Factor of .456 is to put mod-z scores on same scale as standard deviation.
    sig = np.sqrt(flagged_filter(d_sq, size=size, kind="median", flags=flags) / 0.456)

    # don't divide by zero, instead turn those entries into +inf
    return robust_divide(d_rs, sig)


def detrend_meanfilt(
    data: np.ndarray,
    flags: np.ndarray | None = None,
    half_size: tuple[int | None] | None = None,
):
    """Detrend array using a mean filter.

    Parameters
    ----------
    data : array
        Data to detrend. Can be an array of any number of dimensions.
    flags : boolean array, optional
        Flags specifying data to ignore in the detrend. If not given, don't ignore
        anything.
    half_size : tuple of int/None
        The half-size of the kernel to convolve (kernel size will be 2*half_size+1).
        Value of zero (for any dimension) omits that axis from the kernel, effectively
        applying the detrending for each subarray along that axis. Value of None will
        effectively (but slowly) perform a median along the entire axis before running
        the kernel over the other axis.

    Returns
    -------
    out : array
        An array containing the outlier significance metric. Same type and size as
        `data`.

    Notes
    -----
    This detrending is very good for data that has most of the RFI flagged already, but
    will perform very poorly when un-flagged RFI still exists. It is often useful to
    precede this with a median filter.
    """
    half_size = _check_convolve_dims(data, half_size)
    size = tuple(2 * s + 1 for s in half_size)

    d_sm = flagged_filter(data, size=size, kind="mean", flags=flags)
    d_rs = data - d_sm
    d_sq = d_rs**2

    # Factor of .456 is to put mod-z scores on same scale as standard deviation.
    sig = np.sqrt(flagged_filter(d_sq, size=size, kind="mean", flags=flags))

    # don't divide by zero, instead turn those entries into +inf
    return robust_divide(d_rs, sig)


def xrfi_medfilt(
    spectrum: np.ndarray,
    threshold: float = 6,
    flags: np.ndarray | None = None,
    kf: int = 8,
    kt: int = 8,
    inplace: bool = True,
    max_iter: int = 1,
    poly_order: int = 0,
    accumulate: bool = False,
    use_meanfilt: bool = True,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Generate RFI flags for a given spectrum using a median filter.

    Parameters
    ----------
    spectrum : array-like
        Either a 1D array of shape ``(NFREQS,)`` or a 2D array of shape
        ``(NTIMES, NFREQS)`` defining the measured raw spectrum.
        If 2D, a 2D filter in freq*time will be applied by default. One can perform
        the filter just over frequency (in the case that `NTIMES > 1`) by setting
        `kt=0`.
    threshold : float, optional
        Number of effective sigma at which to clip RFI.
    flags : array-like, optional
        Boolean array of pre-existing flagged data to ignore in the filtering.
    kt, kf : tuple of int/None
        The half-size of the kernel to convolve (eg. kernel size over frequency
        will be ``2*kt+1``).
        Value of zero (for any dimension) omits that axis from the kernel, effectively
        applying the detrending for each subarray along that axis. Value of None will
        effectively (but slowly) perform a median along the entire axis before running
        the kernel over the other axis.
    inplace : bool, optional
        If True, and flags are given, update the flags in-place instead of creating a
        new array.
    max_iter : int, optional
        Maximum number of iterations to perform. Each iteration uses the flags of the
        previous iteration to achieve a more robust estimate of the flags. Multiple
        iterations are more useful if ``poly_order > 0``.
    poly_order : int, optional
        If greater than 0, fits a polynomial to the spectrum before performing
        the median filter. Only allowed if spectrum is 1D. This is useful for getting
        the number of false positives down. If max_iter>1, the polynomial will be refit
        on each iteration (using new flags).
    accumulate : bool,optional
        If True, on each iteration, accumulate flags. Otherwise, use only flags from the
        previous iteration and then forget about them. Recommended to be False.
    use_meanfilt : bool, optional
        Whether to apply a mean filter *after* the median filter. The median filter is
        good at getting RFI, but can also pick up non-RFI if the spectrum is steep
        compared to the noise. The mean filter is better at only getting RFI if the RFI
        has already been flagged.

    Returns
    -------
    flags : array-like
        Boolean array of the same shape as ``spectrum`` indicated which channels/times
        have flagged RFI.

    Notes
    -----
    The default combination of using a median filter followed by a mean filter works
    quite well. The median filter works quite well at picking up large RFI (wrt to the
    noise level), but can also create false positives if the noise level is small wrt
    the steepness of the slope. Following by a flagged mean filter tends to remove these
    false positives (as it doesn't get pinned to zero when the function is monotonic).

    It is unclear whether performing an iterative filtering is very useful unless using
    a polynomial subtraction. With polynomial subtraction, one should likely use at
    least a few iterations, without accumulation, so that the polynomial is not skewed
    by the as-yet-unflagged RFI.

    Choice of kernel size can be important. The wider the kernel, the more
    "signal-to-noise" one will get on the RFI. Also, if there is a bunch of RFI all
    clumped together, it will definitely be missed by a kernel window of order double
    the size of the clump or less. By increasing the kernel size, these clumps are
    picked up, but edge-effects become more prevalent in this case. One option here
    would be to iterate over kernel sizes (getting smaller), such that very large blobs
    are first flagged out, then progressively finer detail is added. Use
    ``xrfi_iterative_medfilt`` for that.
    """
    ii = 0

    if flags is None:
        new_flags = np.zeros(spectrum.shape, dtype=bool)
    else:
        new_flags = flags if inplace else flags.copy()

    nflags = -1

    nflags_list = []
    resid_list = []
    assert max_iter > 0
    resid = spectrum.copy()

    size = (kf,) if spectrum.ndim == 1 else (kt, kf)
    while ii < max_iter and np.sum(new_flags) > nflags:
        nflags = np.sum(new_flags)

        if spectrum.ndim == 1 and poly_order:
            # Subtract a smooth polynomial first.
            # The point of this is that steep spectra with only a little bit of noise
            # tend to detrend to exactly zero, but randomly may detrend to something
            # non-zero. In this case, the behaviour is to set the significance to
            # infinity. This is not a problem for data in which the noise is large
            # compared to the signal. We can force this by initially detrending by some
            # flexible polynomial over the whole band. This is not guaranteed to work --
            # the poly fit itself could over-fit for RFI. Therefore the order of the fit
            # should be low. Its purpose is not to do a "good fit" to the data, but
            # rather to get the residuals "flat enough" that the median filter works.
            # TODO: the following is pretty limited (why polynomial?) but it seems to do
            # reasonably well.
            f = np.linspace(0, 1, len(spectrum))
            resid[~new_flags] = (
                spectrum[~new_flags]
                - mdl.ModelFit(
                    mdl.Polynomial(n_terms=poly_order).at(f[~new_flags]),
                    spectrum[~new_flags],
                ).evaluate()
            )
            resid_list.append(resid)
        else:
            resid = spectrum

        med_significance = detrend_medfilt(resid, half_size=size, flags=new_flags)

        if use_meanfilt:
            medfilt_flags = np.abs(med_significance) > threshold
            significance = detrend_meanfilt(resid, half_size=size, flags=medfilt_flags)
        else:
            significance = med_significance

        if accumulate:
            new_flags |= np.abs(significance) > threshold
        else:
            new_flags = np.abs(significance) > threshold

        ii += 1
        nflags_list.append(np.sum(new_flags))

    if 1 < max_iter == ii and np.sum(new_flags) > nflags:
        warnings.warn(
            "Median filter reached max_iter and is still finding new RFI.",
            stacklevel=2,
        )

    return (
        new_flags,
        {
            "significance": significance,
            "median_significance": med_significance,
            "iters": ii,
            "nflags": nflags_list,
            "residuals": resid_list,
        },
    )


def xrfi_explicit(
    spectrum: np.ndarray | None = None,
    *,
    freq: np.ndarray,
    flags: np.ndarray | None = None,
    rfi_file=None,
    extra_rfi=None,
) -> np.ndarray[bool]:
    """
    Excise RFI from given data using an explicitly set list of flag ranges.

    Parameters
    ----------
    spectrum
        This parameter is unused in this function.
    freq
        Frequencies, in MHz, of the data.
    flags
        Known flags.
    rfi_file : str, optional
        A YAML file containing the key 'rfi_ranges', which should be a list of 2-tuples
        giving the (min, max) frequency range of known RFI channels (in MHz). By
        default, uses a file included in `edges-analysis` with known RFI channels from
        the MRO.
    extra_rfi : list, optional
        A list of extra RFI channels (in the format of the `rfi_ranges` from the
        `rfi_file`).

    Returns
    -------
    flags : array-like
        Boolean array of the same shape as ``spectrum`` indicated which channels/times
        have flagged RFI.
    """
    if flags is None:
        if spectrum is None:
            flags = np.zeros(freq.shape, dtype=bool)
        else:
            flags = np.zeros(spectrum.shape, dtype=bool)

    rfi_freqs = []
    if rfi_file:
        with open(rfi_file) as fl:
            rfi_freqs += yaml.load(fl, Loader=yaml.FullLoader)["rfi_ranges"]

    if extra_rfi:
        rfi_freqs += extra_rfi

    for low, high in rfi_freqs:
        flags[..., (freq > low) & (freq < high)] = True

    return flags


xrfi_explicit.ndim = (1, 2, 3)


def _get_mad(x):
    med = np.median(x)
    # Factor of 0.456 to scale median back to Gaussian std dev.
    return np.median(np.abs(x - med)) / np.sqrt(0.456)


def xrfi_model_sweep(
    spectrum: np.ndarray,
    *,
    freq: np.ndarray | None = None,
    flags: np.ndarray | None = None,
    weights: np.ndarray | None = None,
    model: mdl.Model = mdl.Polynomial(n_terms=3),
    window_width: int = 100,
    use_median: bool = True,
    n_bootstrap: int = 20,
    threshold: float | None = 3.0,
    which_bin: str = "last",
    watershed: int = 0,
    max_iter: int = 1,
) -> tuple[np.ndarray, dict]:
    """
    Flag RFI by using a moving window and a low-order polynomial to detrend.

    This is similar to :func:`xrfi_medfilt`, except that within each sliding window,
    a low-order polynomial is fit, and the std dev of the residuals is used as the
    underlying distribution width at which to clip RFI.

    Parameters
    ----------
    spectrum : array-like
        A 1D or 2D array, where the last axis corresponds to frequency. The data
        measured at those frequencies.
    flags : array-like
        The boolean array of flags.
    weights : array-like
        The weights associated with the data (same shape as `spectrum`).
    model_type
        The kind of model to use to fit each window. If a string, it must be the name
        of a :class:`~modelling.mdl.Model`.
    window_width : int, optional
        The width of the moving window in number of channels.
    use_median : bool, optional
        Instead of using bootstrap for the initial window, use Median Absolute
        Deviation. If True, ``n_bootstrap`` is not used. Note that this is typically
        more robust than bootstrap.
    n_bootstrap : int, optional
        Number of bootstrap samples to take to estimate the standard deviation of
        the data without RFI.
    n_terms
        The number of terms in the model (if applicable).
    threshold
        The number of sigma away from the fitted model must be before it is flagged.
        Higher numbers get less false positives, but may miss some true flags.
    which_bin
        Which bin to flag in each window. May be "last" (default), "all".
        In each window, only this bin will be flagged (or all bins will be if "all").
    watershed
        The number of bins beside each flagged RFI that are assumed to also be RFI.
    max_iter
        The maximum number of iterations to use before determining the flags in a
        particular window.

    Returns
    -------
    flags : array-like
        Boolean array of the same shape as ``spectrum`` indicated which channels/times
        have flagged RFI.
    info : dict
        A dictionary of info about the fit, that can be used to inspect what happened.

    Notes
    -----
    Some notes on this algorithm. The basic idea is that a window of a given width is
    used, and within that window, a model is fit to the spectrum data. The residuals of
    that fit are used to calculate the standard deviation (or the 'noise-level'), which
    gives an indication of outliers. This standard deviation may be found either by
    bootstrap sampling, or by using the Median Absolute Deviation (MAD). Both of these
    to *some extent* account for RFI that's still in the residuals, but the MAD is
    typically a bit more robust. **NOTE:** getting the estimate of the standard
    deviation wrong is one of the easiest ways for this algorithm to fail. It relies on
    a few assumptions. Firstly, the window can't be too large, or else the residuals
    within the window aren't stationary. Secondly, while previously-defined flags are
    used to flag out what might be RFI, os that those data are NOT used in getting the
    standard deviation, any remaining RFI will severely bias the std. Obviously, if RFI
    remains in the data, the model itself might not be very accurate either.

    Note that for each window, at first the RFI in that window will likely be unflagged,
    and the std will be computed with all the channels, RFI included. This is why
    using the MAD or bootstrapping is required. Even if the std is predicted robustly
    via this method (i.e. there are more good bins than bad in the window), the model
    itself may not be very good, and so the resulting flags may not be very good. This
    is where using the option of ``max_iter>1`` is useful -- in this case, the model
    is fit to the same window repeatedly until the flags in the window don't change
    between iterations (note this is NOT cumulative).

    In the end, by default, only a single channel is actually flagged per-window. While
    inside the iterative loop, any number of flags can be set (in order to make a better
    prediction of the model and std), only the first, last or central pixel is actually
    flagged and used for the next window. This can be changed by setting
    ``which_bin='all'``.
    """
    assert spectrum.ndim == 1
    nf = len(spectrum)
    f = np.linspace(-1, 1, window_width)

    model = model.at(x=f)

    # Initialize some flags, or set them equal to the input
    orig_flags = flags if flags is not None else np.zeros(nf, dtype=bool)
    orig_flags |= np.isnan(spectrum) | np.isinf(spectrum)
    flags = orig_flags.copy()

    if weights is None:
        weights = np.ones_like(~flags, dtype=float)
    if np.sum(weights) == 0 or np.all(flags):
        return np.ones_like(spectrum, dtype=bool), {}

    # Have to get flags aligned with input weights, and also input weights aligned with
    # flags. But we don't want to overwrite the input weights...
    flags |= weights <= 0
    weights = np.where(flags, 0, weights)

    # Get which pixel will be flagged.
    if which_bin == "last":
        pixel = window_width - 1
    elif which_bin == "all":
        pixel = np.arange(window_width)

    if which_bin != "all" and watershed:
        raise ValueError("can only use watershed with which_bin='all'")

    # Get the first window that has enough unflagged data.
    window = np.arange(window_width, dtype=int)

    while np.sum(weights[window] > 0) <= model.n_terms and window[-1] < (nf - 1):
        window += 1

    if window[-1] == nf - 1:
        return np.ones_like(spectrum, dtype=bool), {}

    flg, r_std, p, n = _flag_a_window(
        window,
        flags,
        spectrum,
        max_iter,
        weights,
        model,
        n_bootstrap,
        threshold,
        watershed,
        std_estimator=int(use_median),
    )
    flags[window] |= flg
    std = [r_std]
    params = [p]
    iters = [n]

    # Slide the window across the spectrum.
    window += 1
    while window[-1] < nf:
        try:
            new_flags, r_std, p, n = _flag_a_window(
                window,
                flags,
                spectrum,
                max_iter,
                weights,
                model,
                n_bootstrap,
                threshold,
                watershed,
            )
            std.append(r_std)
            params.append(p)
            iters.append(n)
            flags[window.min() + pixel] |= new_flags[pixel]
        except NoDataError:
            std.append(None)

        window += 1

    return flags, {"std": std, "params": params, "iters": iters, "model": model.model}


xrfi_model_sweep.ndim = (1,)


def _flag_a_window(
    window,
    flags,
    spectrum,
    max_iter,
    weights,
    model,
    n_bootstrap,
    threshold,
    watershed,
    std_estimator=2,
    fit_kwargs=None,
):
    # NOTE: line profiling reveals that the fitting takes ~50% of the time of this
    #       function, and taking the std takes ~20%. The next biggest are taking the
    #       two sums, which are ~6% each.
    counter = 0
    flags_changed = 1
    new_flags = flags[window]
    d = spectrum[window].copy()
    fit_kwargs = fit_kwargs or {}

    rng = np.random.default_rng()
    while counter < max_iter and flags_changed > 0:
        w = np.where(new_flags, 0, weights[window])

        mask = ~new_flags

        if np.sum(mask) > model.n_terms:
            fit = model.fit(ydata=d, weights=w, **fit_kwargs)
        else:
            raise NoDataError

        resids = fit.residual

        # Computation of STD for initial section using the median statistic
        if std_estimator == 0:
            r_choice_std = [
                np.std(rng.choice(resids[mask], len(resids[mask]) // 2))
                for _ in range(n_bootstrap)
            ]
            r_std = np.median(r_choice_std)
        elif std_estimator == 1:
            r_std = _get_mad(resids[mask])
        elif std_estimator == 2:
            r_std = np.std(resids[mask])

        zscore = np.abs(resids) / r_std
        new_flags = zscore > threshold

        if watershed is not None:
            new_flags |= _apply_watershed(new_flags, watershed, zscore / threshold)

        flags_changed = np.sum((~mask) ^ new_flags)
        counter += 1

    if counter == max_iter and max_iter > 1:
        warnings.warn(
            "Max iterations reached without finding all xRFI. Consider increasing "
            "max_iter.",
            stacklevel=2,
        )

    return new_flags, r_std, fit.model_parameters, counter


def model_filter(
    x: np.ndarray,
    data: np.ndarray,
    *,
    model: mdl.Model = mdl.Polynomial(n_terms=3),
    resid_model: mdl.Model = mdl.Polynomial(n_terms=5),
    flags: np.ndarray | None = None,
    weights: np.ndarray | None = None,
    n_resid: int = -1,
    threshold: float | None = None,
    max_iter: int = 20,
    increase_order: bool = True,
    min_terms: int = 0,
    max_terms: int = 10,
    min_resid_terms: int = 3,
    decrement_threshold: float = 0,
    min_threshold: float = 5,
    watershed: int | dict[float, int] | None = None,
    flag_if_broken: bool = True,
    init_flags: np.ndarray | None = None,
    std_estimator: Literal["model", "medfilt", "std", "mad", "sliding_rms"] = "model",
    medfilt_width: int = 100,
    sliding_rms_width: int = 100,
    fit_kwargs: dict | None = None,
):
    """
    Flag data by subtracting a smooth model and iteratively removing outliers.

    On each iteration, a model is fit to the unflagged data, and another model is fit
    to the absolute residuals. Bins with absolute residuals greater than
    ``threshold`` are flagged, and the process is repeated until no new
    flags are found.

    Parameters
    ----------
    x
        The coordinates of the data.
    data
        The data (same shape as ``x``).
    model
        A model to fit to the data.
    resid_model
        The model to fit to the absolute residuals.
    flags : array-like, optional
        The flags associated with the data (same shape as ``spectrum``).
    weights : array-like,, optional
        The weights associated with the data (same shape as ``spectrum``).
    n_resid : int, optional
        The number of polynomial terms to use to fit the residuals.
    threshold : float, optional
        The factor by which the absolute residual model is multiplied to determine
        outliers.
    max_iter : int, optional
        The maximum number of iterations to perform.
    accumulate : bool, optional
        Whether to accumulate flags on each iteration.
    increase_order : bool, optional
        Whether to increase the order of the polynomial on each iteration.
    decrement_threshold : float, optional
        An amount to decrement the threshold by every iteration. Threshold will never
        go below ``min_threshold``.
    min_threshold : float, optional
        The minimum threshold to decrement to.
    watershed
        How many data points *on each side* of a flagged point that should be flagged.
        If a dictionary, you can give keys as the threshold above which z-scores will
        be flagged, and as values, the number of bins flagged beside it. Use 0.0
        threshold to indicate the base threshold.
    init_flags
        Initial flags that are not remembered after the first iteration. These can
        help with getting the initial model. If a tuple, should be a min and max
        frequency of a range to flag.
    std_estimator
        The estimator to use to get the standard deviation of each sample.
    medfilt_width
        Only used if `std_estimator='medfilt'`. The width (in number of bins) to use
        for the median filter.

    Returns
    -------
    flags
        Boolean array of the same shape as ``data``.
    """
    fit_kwargs = fit_kwargs or {}

    threshold = threshold or (
        min_threshold + 5 * decrement_threshold
        if decrement_threshold
        else min_threshold
    )
    if not decrement_threshold:
        min_threshold = threshold

    if decrement_threshold > 0 and min_threshold > threshold:
        warnings.warn(
            f"You've set a threshold smaller than the min_threshold of {min_threshold}."
            f"Will use threshold={min_threshold}.",
            stacklevel=2,
        )
        threshold = min_threshold

    assert threshold > 1.5

    assert data.ndim == 1
    assert x.ndim == 1
    if len(x) != len(data):
        raise ValueError("freq and spectrum must have the same length")

    nx = len(x)

    # We assume the residuals are smoother than the signal itself
    if not increase_order:
        assert n_resid <= model.n_terms

    n_flags_changed = 1
    counter = 0

    # Set up a few lists that we can update on each iteration to return info to the user
    n_flags_changed_list = []
    total_flags_list = []
    model_list = []
    res_models = []
    thresholds = []
    std_list = []
    flag_list = []

    model = model.at(x=x)
    if n_resid <= 0:
        resid_model = resid_model.with_nterms(
            max(min_resid_terms, model.n_terms + n_resid)
        )

    res_model = resid_model.at(x=x)

    # Initialize some flags, or set them equal to the input
    orig_flags = flags if flags is not None else np.zeros(nx, dtype=bool)
    orig_flags |= np.isnan(data) | np.isinf(data)

    flags = orig_flags.copy()

    if init_flags is not None:
        flags = flags | init_flags

    orig_weights = np.ones_like(data) if weights is None else weights.copy()

    # Iterate until either no flags are changed between iterations, or we get to the
    # requested maximum iterations, or until we have too few unflagged data to fit
    # appropriately. keep iterating
    n_flags_changed_all = [1]
    while counter < max_iter and (
        model.n_terms <= min_terms
        or (
            any(fl > 0 for fl in n_flags_changed_all)
            and np.sum(~flags) > model.n_terms * 2
        )
    ):
        weights = np.where(flags, 0, orig_weights)

        # Get a model fit to the unflagged data.
        # Could be polynomial or fourier (or something else...)
        mdl = model.fit(ydata=data, weights=weights, **fit_kwargs)

        if any(
            len(p.parameters) == len(mdl.model_parameters)
            and np.allclose(mdl.model_parameters, p.parameters)
            for p in model_list
        ):
            # If we're not changing the parameters significantly, just exit. This is
            # *very important* as it stops closed-loop cycles where the flags and models
            # go back and forth.
            break

        res = mdl.residual

        model_list.append(mdl.fit.model)

        if std_estimator == "medfilt":
            model_std = np.sqrt(
                flagged_filter(
                    res**2,
                    size=2 * (medfilt_width // 2) + 1,
                    kind="median",
                    flags=flags,
                )
                / 0.456
            )
        elif std_estimator == "model":
            # Now fit a model to the absolute residuals.
            # This number is "like" a local standard deviation, since the polynomial
            # does something like a local average.
            # Do it in log-space so the model doesn't ever hit zero.
            # The 0.53 term comes about because the estimate of the std here is not
            # unbiased. You can obtain it by doing
            # sigma=<any number>
            # \sqrt(exp(mean(log(Normal(0, \sigma, 1000000)^2))))/\sigma
            # it is not dependent on the value of sigma.
            absres = np.abs(res)

            if n_resid <= 0:
                res_model = res_model.with_nterms(
                    max(min_resid_terms, model.n_terms + n_resid)
                )
            res_mdl = res_model.fit(
                ydata=np.log(absres**2), weights=weights, **fit_kwargs
            ).fit
            model_std = np.sqrt(np.exp(res_mdl())) / 0.53
            res_models.append(res_mdl.model)

        elif std_estimator == "std":
            model_std = np.std(res[~flags]) * np.ones_like(x)
        elif std_estimator == "mad":
            model_std = _get_mad(res[~flags]) * np.ones_like(x)
        elif std_estimator == "sliding_rms":
            # This gets the sliding RMS by convolving a top-hat with the residuals^2
            # then taking the square root. To ensure that the mean at the edges doesn't
            # get distorted, we extend the array on both sides with NaNs.
            res2 = np.concatenate((
                np.ones(sliding_rms_width // 2) * np.nan,
                res**2,
                np.ones(sliding_rms_width // 2) * np.nan,
            ))
            fflags = np.concatenate((
                np.ones(sliding_rms_width // 2, dtype=bool),
                flags,
                np.ones(sliding_rms_width // 2, dtype=bool),
            ))
            model_std = np.sqrt(
                convolve_fft(res2, Box1DKernel(sliding_rms_width), mask=fflags)[
                    sliding_rms_width // 2 : -sliding_rms_width // 2
                ]
            )
        else:
            raise ValueError(
                "std_estimator must be one of 'medfilt', 'model','std', "
                "'sliding_rms' or 'mad'."
            )

        std_list.append(model_std)

        zscore = res / model_std

        # If we're not accumulating, we just take these flags (along with the fully
        # original flags).
        new_flags = orig_flags | (zscore > threshold)

        # Apply a watershed -- assume surrounding channels will succumb to RFI.
        if watershed is not None:
            new_flags |= _apply_watershed(new_flags, watershed, zscore / threshold)

        n_flags_changed_all = [
            np.sum(flags_f ^ new_flags) for flags_f in [*flag_list, flags]
        ]
        n_flags_changed = n_flags_changed_all[-1]

        flags = new_flags.copy()

        counter += 1
        if increase_order and model.n_terms < max_terms:
            model = model.with_nterms(model.n_terms + 1)

        thresholds.append(threshold)

        # decrease the flagging threshold if we want to for next iteration
        threshold = max(threshold - decrement_threshold, min_threshold)

        logger.info(
            f"{counter} rms {model_std[-1]} {np.sum(flags)} resid {res.min()} "
            f"{res.max()} z {zscore.min()} {zscore.max()} std {model_std.min()} "
            f"{model_std.max()}"
        )
        # Append info to lists for the user's benefit
        n_flags_changed_list.append(n_flags_changed)
        total_flags_list.append(np.sum(flags))
        flag_list.append(flags)

    if counter == max_iter and max_iter > 1 and n_flags_changed > 0:
        warnings.warn(
            f"max iterations ({max_iter}) reached, not all RFI might have been caught.",
            stacklevel=2,
        )
        if flag_if_broken:
            flags[:] = True

    elif np.sum(~flags) <= model.n_terms * 2:
        warnings.warn(
            "Termination of iterative loop due to too many flags. Reduce n_signal or "
            "check data.",
            stacklevel=2,
        )
        if flag_if_broken:
            flags[:] = True

    return (
        flags,
        ModelFilterInfo(
            n_flags_changed=n_flags_changed_list,
            total_flags=total_flags_list,
            models=model_list,
            n_iters=counter,
            res_models=res_models,
            thresholds=thresholds,
            stds=std_list,
            x=x,
            data=data,
            flags=flag_list,
        ),
    )


@dataclass
class ModelFilterInfo:
    """A simple object representing the information returned by :func:`model_filter`."""

    n_flags_changed: list[int]
    total_flags: list[int]
    models: list[mdl.Model]
    res_models: list[mdl.Model] | None
    n_iters: int
    thresholds: list[float]
    stds: list[np.ndarray[float]]
    x: np.ndarray
    data: np.ndarray
    flags: list[np.ndarray[bool]]

    def get_model(self, indx: int = -1):
        """Get the model values."""
        return self.models[indx](x=self.x)

    def get_residual(self, indx: int = -1):
        """Get the residuals."""
        return self.get_model(indx) - self.data

    def get_absres_model(self, indx: int = -1):
        """Get the *model* of the absolute residuals."""
        return self.res_models[indx](self.x)

    def write(self, fname: tp.PathLike, group: str = "/"):
        """Write the object to a HDF5 file."""
        with h5py.File(fname, "a") as fl:
            grp = fl.require_group(group)

            grp.attrs["n_iters"] = self.n_iters

            for i, (model, res_model) in enumerate(
                zip(self.models, self.res_models, strict=False)
            ):
                grp.attrs[f"model_{i}"] = yaml.dump(model)
                grp.attrs[f"res_model_{i}"] = yaml.dump(res_model)

            for k in self.__dataclass_fields__:
                if k not in ["n_iters", "models", "res_models"]:
                    try:
                        grp[k] = np.asarray(getattr(self, k))
                    except TypeError as e:
                        raise TypeError(
                            f"Key {k} with data {np.asarray(getattr(self, k))} "
                            f"failed with msg: {e}"
                        ) from e

    @classmethod
    def from_file(cls, fname: tp.PathLike, group: str = "/"):
        """Create the object by reading from a HDF5 file."""
        info = {}
        with h5py.File(fname, "r") as fl:
            grp = fl[group]

            info["n_iters"] = grp.attrs["n_iters"]

            info["models"] = [
                yaml.load(grp.attrs[f"model_{i}"], Loader=yaml.FullLoader)
                for i in range(info["n_iters"])
            ]
            info["res_models"] = [
                yaml.load(grp.attrs[f"res_model_{i}"], Loader=yaml.FullLoader)
                for i in range(info["n_iters"])
            ]

            for k in grp:
                info[k] = grp[k][...]

        return cls(**info)


@dataclass
class ModelFilterInfoContainer:
    """A container of :class:`ModelFilterInfo` objects.

    This is almost a perfect drop-in replacement for a singular :class:`ModelFilterInfo`
    instance, but combines a number of them together seamlessly. This can be useful if
    several sub-models were fit to one long stream of data.
    """

    models: list[ModelFilterInfo] = field(default_factory=list)

    def append(self, model: ModelFilterInfo) -> ModelFilterInfoContainer:
        """Create a new object by appending a set of info to the existing."""
        assert isinstance(model, ModelFilterInfo)
        models = [*self.models, model]
        return ModelFilterInfoContainer(models)

    @cached_property
    def x(self):
        """The data coordinates."""
        return np.concatenate(tuple(model.x for model in self.models))

    @cached_property
    def data(self):
        """The raw data that was filtered."""
        return np.concatenate(tuple(model.data for model in self.models))

    @cached_property
    def flags(self):
        """The returned flags on each iteration."""
        return np.concatenate(tuple(model.flags for model in self.models))

    @cached_property
    def n_iters(self):
        """The number of iterations of the filtering."""
        return max(model.n_iters for model in self.models)

    @cached_property
    def n_flags_changed(self):
        """The number of flags changed on each filtering iteration."""
        return [
            sum(
                model.n_flags_changed[min(i, model.n_iters - 1)]
                for model in self.models
            )
            for i in range(self.n_iters)
        ]

    @cached_property
    def total_flags(self):
        """The total number of flags after each iteration."""
        return [
            sum(model.total_flags[min(i, model.n_iters - 1)] for model in self.models)
            for i in range(self.n_iters)
        ]

    def get_model(self, indx: int = -1):
        """Get the model values."""
        assert indx >= -1
        return np.concatenate(
            tuple(
                model.get_model(min(indx, model.n_iters - 1)) for model in self.models
            )
        )

    def get_residual(self, indx: int = -1):
        """Get the residual values."""
        assert indx >= -1
        return np.concatenate(
            tuple(
                model.get_residual(min(indx, model.n_iters - 1))
                for model in self.models
            )
        )

    def get_absres_model(self, indx: int = -1):
        """Get the *model* of the absolute residuals."""
        assert indx >= -1
        return np.concatenate(
            tuple(
                model.get_absres_model(min(indx, model.n_iters - 1))
                for model in self.models
            )
        )

    @cached_property
    def thresholds(self):
        """The threshold at each iteration."""
        for model in self.models:
            if model.n_iters == self.n_iters:
                break

        return model.thresholds

    @cached_property
    def stds(self):
        """The standard deviations at each datum for each iteration."""
        return [
            np.concatenate(
                tuple(model.stds[min(indx, model.n_iters - 1)] for model in self.models)
            )
            for indx in self.n_iters
        ]

    @classmethod
    def from_file(cls, fname: str):
        """Create an object from a given file."""
        with h5py.File(fname, "r") as fl:
            n_models = fl.attrs["n_models"]

        models = [
            ModelFilterInfo.from_file(fname, group=f"model_{i}")
            for i in range(n_models)
        ]

        return cls(models)

    def write(self, fname: str):
        """Write the object to a file."""
        with h5py.File(fname, "w") as fl:
            fl.attrs["n_models"] = len(self.models)

        for i, model in enumerate(self.models):
            model.write(fname, group=f"model_{i}")


def xrfi_model(
    spectrum: np.ndarray,
    *,
    freq: np.ndarray,
    inplace: bool = False,
    init_flags: np.ndarray | tuple[float, float] | None = None,
    flags: np.ndarray | None = None,
    **kwargs,
):
    """
    Flag RFI by subtracting a smooth model and iteratively removing outliers.

    On each iteration, a model is fit to the unflagged data, and another model is fit
    to the absolute residuals. Bins with absolute residuals greater than
    ``n_abs_resid_threshold`` are flagged, and the process is repeated until no new
    flags are found.

    Parameters
    ----------
    spectrum : array-like
        A 1D spectrum. Note that instead of a spectrum, model residuals can be passed.
        The function does *not* assume the input is positive.
    freq
        The frequencies associated with the spectrum.
    inplace : bool, optional
        Whether to fill up given flags array with the updated flags.
    init_flags
        Initial flags that are not remembered after the first iteration. These can
        help with getting the initial model. If a tuple, should be a min and max
        frequency of a range to flag.
    **kwargs
        All other parameters passed to :func:`model_filter`

    Returns
    -------
    flags : array-like
        Boolean array of the same shape as ``spectrum`` indicated which channels/times
        have flagged RFI.
    """
    if init_flags is not None and len(init_flags) == 2:
        init_flags = (freq > init_flags[0]) & (freq < init_flags[1])

    new_flags, info = model_filter(
        x=freq, data=spectrum, init_flags=init_flags, flags=flags, **kwargs
    )

    if inplace and flags is not None:
        flags |= new_flags

    return new_flags, info


xrfi_model.ndim = (1,)


def xrfi_model_sliding_rms_single_pass(
    spectrum: np.ndarray,
    *,
    freq: np.ndarray,
    model: mdl.mdl.Model = mdl.Polynomial(n_terms=3),
    flags: np.ndarray | None = None,
    weights: np.ndarray | None = None,
    window_frac: int = 16,
    min_window_size: int = 10,
    threshold: float = 2.5,
    watershed: dict | None = None,
    fit_kwargs: dict | None = None,
):
    """
    Flag RFI using a model fit and a sliding RMS window.

    This function is algorithmically the same as that used in Bowman+2018.
    The differences between this and :func:`xrfi_model` (which is the recommended
    function to use) are:

    * This does flagging *inside* the sliding window  -- i.e. once you move the
      window up by one channel, the flags can be different in the previous bins.
      This is a bit strange, since it makes the process more non-linear. If you
      were to start from the top of the band and slide the window down, you'd
      get different results.
    * The watershedding (flagging channels around the "bad" one) only happens
      if the main central channel is far enough away from the edges of the band.
    * It only flags positive outliers.

    Parameters
    ----------
    spectrum : array-like
        The 1D spectrum to flag.
    freq
        The frequencies associated with the spectrum.
    model : :class:`edges.modelling.mdl.Model`
        The model to fit to the spectrum to get residuals.
    flags : array-like, optional
        The initial flags to use. If not given, all channels are unflagged.
    window_frac : int, optional
        The size of the sliding window as a fraction of the number of channels (i.e.
        the final window is int(Nchannels / window_frac) in size).
    min_window_size : int, optional
        The minimum size of the sliding window, in number of channels.
    max_iter : int, optional
        The maximum number of iterations to perform.
    threshold : float, optional
        The threshold for flagging a channel. The threshold is the number of standard
        deviations the residuals are from zero.
    watershed : dict, optional
        The parameters for the watershedding algorithm. If not given, no watershedding
        is performed. Each key should be a float that specifies the number of
        threshold*stds away from zero that a channel should be flagged. The value
        should be the number of channels to flag on either side of the flagged channel
        for that threshold. For example, ``{3: 2}`` would flag 2 channels on either
        side of any channel that is 3*threshold standard deviations away from zero.
    fit_kwargs : dict, optional
        Any additional keyword arguments to pass to the model fit. Use the key "method"
        with value "alan-qrd" for the closest match to the Bowman+2018 code.

    Returns
    -------
    flags : array-like
        Boolean array of the same shape as ``spectrum`` indicated which channels/times
        have flagged RFI.
    info : ModelFilterInfo
        A :class:`ModelFilterInfo` object containing information about the fit at
        each iteration.
    """
    fmod = model.at(x=freq)
    fit_kwargs = fit_kwargs or {}

    if flags is None:
        flags = np.zeros(len(spectrum), dtype=bool)

    if weights is None:
        weights = (~flags).astype(int)
    else:
        weights = weights.copy()
        weights[flags] = 0.0

    n = len(spectrum)
    m = max(n // window_frac, min_window_size)

    fit = fmod.fit(ydata=spectrum, weights=weights, **fit_kwargs)

    rms = np.zeros(n)
    avs = np.zeros(n)
    for i in range(n):
        rng = slice(max(i - m, 0), min(n, i + m + 1))
        size = np.sum(weights[rng])
        av = np.sum(fit.residual[rng] * weights[rng]) / size

        rms[i] = np.sqrt(np.sum((fit.residual[rng] - av) ** 2 * weights[rng]) / size)
        avs[i] = av
        if i == 14:
            pass
        # Now while *INSIDE* the loop over frequencies, apply new flags.
        nsig = fit.residual[i] / (threshold * rms[i])

        if nsig > 1:
            weights[i] = 0

            if watershed:
                for mult, nbins in watershed.items():
                    if nsig > mult and i + nbins < n and i - nbins >= 0:
                        weights[i - nbins : i + nbins + 1] = 0

    return weights == 0


xrfi_model_sliding_rms_single_pass.ndim = (1,)


def xrfi_watershed(
    spectrum: np.ndarray | None = None,
    *,
    freq: np.ndarray | None = None,
    flags: np.ndarray | None = None,
    weights: np.ndarray | None = None,
    tol: float | tuple[float] = 0.5,
    inplace=False,
):
    """Apply a watershed over frequencies and times for flags.

    Make sure that times/freqs with many flags are all flagged.

    Parameters
    ----------
    spectrum
        Not used in this routine.
    flags : ndarray of bool
        The existing flags.
    tol : float or tuple
        The tolerance -- i.e. the fraction of entries that must be flagged before
        flagging the whole axis. If a tuple, the first element is for the frequency
        axis, and the second for the time axis.
    inplace : bool, optional
        Whether to update the flags in-place.

    Returns
    -------
    ndarray :
        Boolean array of flags.
    dict :
        Information about the flagging procedure (empty for this function)
    """
    if flags is None:
        if weights is not None:
            flags = ~(weights.astype(bool))
        else:
            raise ValueError("You must provide flags as an ndarray")

    if weights is not None:
        flags |= weights <= 0

    fl = flags if inplace else flags.copy()

    if not hasattr(tol, "__len__"):
        tol = (tol, tol)

    freq_coll = np.sum(flags, axis=-1)
    freq_mask = freq_coll > tol[0] * flags.shape[1]
    fl[freq_mask] = True

    if flags.ndim == 2:
        time_coll = np.sum(fl, axis=0)
        time_mask = time_coll > tol[1] * flags.shape[0]
        fl[:, time_mask] = True

    return fl, {}


xrfi_watershed.ndim = (1, 2)


def _apply_watershed(
    flags: np.ndarray,
    watershed: int | dict[float, int],
    zscore_thr_ratio: np.ndarray,
):
    watershed_flags = np.zeros_like(flags)

    if isinstance(watershed, int):
        watershed = {1.0: watershed}

    for thr, nw in sorted(watershed.items()):
        this_flg = zscore_thr_ratio > thr

        for i in range(1, nw + 1):
            watershed_flags[i:] |= this_flg[:-i]
            watershed_flags[:-i] |= this_flg[i:]

    return watershed_flags


def visualise_model_info(
    info: ModelFilterInfo | ModelFilterInfoContainer, n: int = 0, fig=None, ax=None
):
    """
    Make a nice visualisation of the info output from :func:`xrfi_model`.

    Parameters
    ----------
    info
        The output ``info`` from :func:`xrfi_model`.
    n
        The number of iterations to plot. Default is to plot them all. Negative numbers
        will plot the last n, and positive will plot the first n.
    """
    if fig is None or ax is None:
        fig, ax = plt.subplots(2, 3, figsize=(10, 6))

    ax[0, 0].plot(info.data, label="Data", color="k")

    if not n:
        n = info.n_iters

    counter = 0
    for i, (model, res_model, nchange, tflags, thr, flags) in enumerate(
        zip(
            info.models,
            info.res_models,
            info.n_flags_changed,
            info.total_flags,
            info.thresholds,
            info.flags,
            strict=False,
        )
    ):
        if (n < 0 and i < info.n_iters + n) or (n > 0 and i >= n):
            continue

        if np.all(flags):
            continue

        m = model(info.x)
        res = info.data - m

        ax[0, 0].plot(m, label=f"{model.n_terms}: {nchange}/{tflags}")
        ax[0, 0].set_title("Spectrum [K]")
        ax[0, 0].axes.xaxis.set_ticklabels([])

        if counter == 0:
            ax[1, 0].scatter(
                np.arange(len(flags)),
                np.where(flags, np.nan, np.abs(res)),
                alpha=0.1,
                edgecolor="none",
                s=5,
                color="k",
            )
            ax[1, 0].set_xlabel("Freq Channel")

            rm = rm0 = np.sqrt(np.exp(res_model(info.x))) / 0.53
            ax[1, 0].plot(rm0)
            ax[1, 0].set_title("Abs mdl.Model Residuals")
        else:
            rm = np.sqrt(np.exp(res_model(info.x))) / 0.53
            ax[1, 0].plot(rm)

        ax[0, 1].axes.xaxis.set_ticklabels([])

        if counter == 0:
            resres = np.abs(res) - rm0
            med = np.nanmedian(resres)
            mad = _get_mad(resres[~np.isnan(resres)])

            ax[0, 1].scatter(
                np.arange(len(flags)),
                np.where(flags, np.nan, resres),
                alpha=0.1,
                edgecolor="none",
                s=5,
                color="k",
            )
            ax[0, 1].set_ylim(med - 7 * mad, med + 7 * mad)
            ax[0, 1].set_title("Residuals of AbsResids")
        else:
            ax[0, 1].plot(rm - rm0)

        ax[1, 1].plot(res / rm, color=f"C{counter}")
        ax[1, 1].axhline(thr, color=f"C{counter}")
        ax[1, 1].axhline(-thr, color=f"C{counter}")

        ax[1, 1].set_ylim(-thr * 3, thr * 3)
        ax[1, 1].set_title("Scaled Residuals and Thresholds")
        ax[1, 1].set_xlabel("Freq Channel")

        ax[1, 2].hist(
            np.where(flags, np.nan, res / rm), bins=50, histtype="step", density=True
        )

        x = np.linspace(-4, 4, 200)
        ax[1, 2].plot(
            x,
            np.exp(-(x**2)) / np.sqrt(2 * np.pi),
            color="k",
            label="Normal Dist." if not i else None,
        )
        ax[1, 2].set_title("Scaled Residuals Distribution")

        ax[1, 2].set_xlabel("Residual")

        ax[0, 2].axis("off")
        counter += 1

    ax[0, 0].legend(title="N: Changed/Tot")
    ax[1, 2].legend()
    plt.tight_layout()
