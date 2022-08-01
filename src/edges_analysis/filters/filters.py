"""Functions that identify and flag bad data in various ways."""
from __future__ import annotations

import logging
from multiprocessing import cpu_count
from typing import Sequence, Callable, Literal
from edges_cal import types as tp
import numpy as np
import yaml
from edges_cal.xrfi import (
    ModelFilterInfoContainer,
    model_filter,
)
from astropy import units as u
import functools
from ..data import DATA_PATH
from .. import tools
from ..averaging import averaging
from edges_cal import xrfi as rfi
import tqdm
from pathos.multiprocessing import ProcessPool as Pool

from ..gsdata import GSData, register_gsprocess

logger = logging.getLogger(__name__)

FILTERS = {}


def get_filter(filt: str) -> Callable:
    """Obtain a registered step filter function from a string name."""
    if filt not in FILTERS:
        raise KeyError(f"'{filt}' does not exist as a filter.")
    return FILTERS[filt]


def gsdata_filter(
    axis: Literal["time", "freq", "day", "all"],
    multi_data: bool = False,
):
    """A decorator to register a filtering function as a potential filter.

    Any function that is wrapped by :func:`gsdata_filter` must implement the following
    signature::

        def fnc(
            data: GSData | Sequence[GSData],
            use_existing_flags: bool,
            **kwargs
        ) -> np.ndarray

    Where the ``data`` is either a single GSData object, or sequence of such
    objects. 

    The returned array should be a 1D or 2D boolean array of flags that may or may
    not include the input flags. 

    Parameters
    ----------
    axis
        The axis over which the filter works. If either 'time' or 'freq', the returned
        array from the wrapped function should be 1D, corresponding to either the
        time or frequency axis of the data. If 'both', the return should be 2D. In the
        case of 'time' or 'freq' filters, the flags will be broadcast across the other
        dimensions.
    multi_data
        Whether the filter accepts multiple objects at the same time to filter. This
        is *usually* so as to enable more accurate filtering when comparing different
        days for instance, rather than just performing a loop over the days and flagging
        each independently.
    """

    def inner(
        fnc: Callable[
            [GSData | Sequence[GSData], bool],
            np.ndarray,
        ]
    ):

        @functools.wraps(fnc)
        def wrapper(
            *,
            data: Sequence[tp.PathLike | GSData],
            flags: Sequence[np.ndarray] | None | list[None] = None,
            in_place: bool = False,
            n_threads: int = 1,
            **kwargs,
        ) -> list[np.ndarray]:
            logger.info(f"Running {fnc.__name__} filter.")

            # Read all the data, in case they haven't been turned into objects yet.
            # And check that everything is the right type.
            if not hasattr(data, "__len__"):
                data = [data]
            data = [GSData.from_file(d) if not isinstance(d, GSData) else d for d in data]
            
            def per_file_processing(data: GSData, flags: np.ndarray):
                
                n = np.sum(data.complete_flags)

                outflags = np.zeros_like(data.complete_flags)

                # Broadcast the computed flags to the correct shape.
                if axis == 'freq':
                    outflags[..., :] = flags
                elif axis == 'time':
                    outflags[..., flags, :] = True
                elif axis == "all":
                    outflags[flags] = True

                if np.all(flags):
                    logger.warning(
                        f"{data.get_initial_yearday()} was fully flagged during {fnc.__name__} "
                        "filter"
                    )
                else:
                    logger.info(
                        f"'{data.get_initial_yearday()}': {100 * n / outflags.size:.2f} → "
                        f"{100 * np.sum(outflags) / outflags.size:.2f}% [red]<+"
                        f"{100 * (np.sum(outflags) - n) / outflags.size:.2f}%>[/] "
                        f"flagged after '{fnc.__name__}' filter"
                    )

                data = data.add_flags(outflags, append_to_file=in_place)
                return data

            if multi_data:
                this_flag = fnc(data=data, **kwargs)

                # TODO: this is probably wrong
                for d, out_flg in zip(data, this_flag):
                    per_file_processing(d, out_flg)

            else:
                def fnc_(data):
                    out = fnc(data=data, **kwargs)
                    return per_file_processing(data, out)

                if n_threads > 1:
                    pool = Pool(n_threads)
                    flg = list(
                        tqdm.tqdm(
                            pool.map(
                                fnc_,
                                data,
                            ),
                            unit="files",
                            total=len(data),
                        )
                    )
                else:
                    flg = list(
                        tqdm.tqdm(map(fnc_, data), unit="files", total=len(data))
                    )

            return flg

        FILTERS[fnc.__name__] = wrapper
        return wrapper

    return inner



def chunked_iterative_model_filter(
    *,
    x: np.ndarray,
    data: np.ndarray,
    flags: np.ndarray | None = None,
    init_flags: np.ndarray | None = None,
    chunk_size: float = np.inf,
    **kwargs,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Perform a chunk-wise iterative model filter.

    This breaks the given data into smaller chunks and then calls
    :func:`edges_cal.xrfi.model_filter` on each chunk, returning the full 1D array of
    flags after all the chunks have been processed.

    Parameters
    ----------
    chunk_size
        The size of the chunks to process, in units of the input coordinates, ``x``.
    **kwargs
        Everything else is passed to :func:`edges_cal.xrfi.model_filter`.

    Returns
    -------
    flags
        The 1D array of flags corresponding to the data. Note that input flags are not
        modified in the course of this function, but the output does already contain
        those flags.
    resid
        Residuals to the model
    std
        Estimates of the standard deviation of the data at each data point.
    """
    if flags is None:
        flags = np.zeros(len(x), dtype=bool)
    if init_flags is None:
        init_flags = np.zeros(len(x), dtype=bool)

    out_flags = flags | np.isnan(data)
    resids = np.zeros_like(data)
    std = np.zeros_like(data)

    xmin = x.min()
    infos = ModelFilterInfoContainer()
    while xmin < x.max():
        mask = (x >= xmin) & (x < xmin + chunk_size)

        out_flags[mask], info = model_filter(
            x=x[mask],
            data=data[mask],
            flags=out_flags[mask],
            init_flags=init_flags[mask],
            **kwargs,
        )
        resids[mask] = info.get_residual()
        std[mask] = info.stds[-1]
        infos = infos.append(info)
        xmin += chunk_size

    return out_flags, resids, std, infos


def explicit_filter(times, bad, ret_times=False):
    """
    Explicitly filter out certain times.

    Parameters
    ----------
    times : array-like
        The input times. This can be either a recarray, a list of tuples, a list
        of ints, or a 2D array of ints. The columns of the recarray (or the entries
        of the tuples) should correspond to `year`, 'day` and `hour`. The last two
        are not required, eg. 2-tuples will be interpreted as ``(year, hour)``, and a
        list of ints will be interpreted as just years.
    bad : str or array-like
        Like `times`, but specifying the bad entries. Need not have the same columns
        as `times`. If any bad exists within a given time frame, it will be considered
        bad. Likewise, if bad has higher scope than times, then it will also be bad.
        Eg.: ``times = [2018], bad=[(2018, 125)]``, times will be considered bad.
        Also, ``times=[(2018, 125)], bad=[2018]``, times will be considered bad.
        If a str, reads the bad times from a properly configured YAML file.
    ret_times : bool, optional
        If True, return the good times as well as the indices of such in original array.

    Returns
    -------
    keep :
        indices marking which times are not bad if inplace=False.
    times :
        Only if `ret_times=True`. An array of the times that are good.
    """
    if isinstance(bad, str):
        with open(bad) as fl:
            bad = yaml.load(fl, Loader=yaml.FullLoader)["bad_days"]

    try:
        nt = len(times[0])
    except AttributeError:
        nt = 1

    try:
        nb = len(bad[0])
    except AttributeError:
        nb = 1

    assert nt in {1, 2, 3}, "times must be an array of 1,2 or 3-tuples"
    assert nb in {1, 2, 3}, "bad must be an array of 1,2 or 3-tuples"

    if nt < nb:
        bad = {b[:nt] for b in bad}
        nb = nt

    keep = [t[:nb] not in bad for t in times]

    return (keep, times[keep]) if ret_times else keep

@register_gsprocess
@gsdata_filter(axis="time")
def aux_filter(
    *,
    data: GSData,
    minima: dict[str, float] | None = None,
    maxima: dict[str, float] | None = None,
) -> np.ndarray:
    """
    Perform an auxiliary filter on the object.

    Parameters
    ----------
    minima 
        Dictionary mapping auxiliary data keys to minimum allowed values.
    maxima
        Dictionary mapping auxiliary data keys to maximum allowed values.

    Returns
    -------
    flags
        Boolean array giving which entries are bad.
    """
    flags = np.zeros(len(gha), dtype=bool)

    def filt(condition, message, flags):
        nflags = np.sum(flags)

        flags |= condition
        if nnew := np.sum(flags) - nflags:
            logger.debug(f"{nnew}/{len(flags) - nflags} times flagged due to {message}")

    for k, v in minima.items():
        if k not in data.auxiliary_measurements:
            raise ValueError(
                f"{k} not in data.auxiliary_measurements. "
                f"Allowed: {data.auxiliary_measurements.keys()}"
            )
        filt(data.auxiliary_measurements[k] < v, f"{k} minimum", flags)

    for k, v in maxima.items():
        if k not in data.auxiliary_measurements:
            raise ValueError(
                f"{k} not in data.auxiliary_measurements. "
                f"Allowed: {data.auxiliary_measurements.keys()}"
            )
        filt(data.auxiliary_measurements[k] > v, f"{k} maximum", flags)

    return flags


def _rfi_filter_factory(method: str):
    def fnc(
        *,
        data: GSData,
        # flags: np.ndarray[bool],
        n_threads: int = cpu_count(),
        freq_range: tuple[float, float] = (40, 200),
        **kwargs,
    ) -> np.ndarray:

        mask = (
            (data.freq_array >= freq_range[0]) & 
            (data.freq_array <= freq_range[1])
        )

        flags = data.complete_flags

        out_flags = tools.run_xrfi(
            method=method,
            spectrum=data.spectrum[..., mask],
            freq=data.freq_array[mask],
            flags=flags[..., mask],
            weights=data.weights[..., mask],
            n_threads=n_threads,
            **kwargs,
        )

        out = np.zeros_like(flags)
        out[..., mask] = out_flags

        return out

    fnc.__name__ = f"rfi_{method}_filter"

    return register_gsprocess(gsdata_filter(axis="all")(fnc))


rfi_model_filter = _rfi_filter_factory("model")
rfi_model_sweep_filter = _rfi_filter_factory("model_sweep")
rfi_watershed_filter = _rfi_filter_factory("watershed")

@register_gsprocess
@gsdata_filter(axis="freq")
def rfi_explicit_filter(*, data: GSData, file: tp.PathLike | None = None):
    """A filter of explicit channels of RFI."""
    if file is None:
        file = DATA_PATH / "known_rfi_channels.yaml"

    return rfi.xrfi_explicit(
        data.freq_array,
        rfi_file=file,
    )



@register_gsprocess
@gsdata_filter(axis="time")
def negative_power_filter(*, data: GSData):
    """Filter out integrations that have *any* negative/zero power.

    These integrations obviously have some weird stuff going on.
    """
    return np.array([np.any(spec <= 0) for spec in data.spectra])


def _peak_power_filter(
    *,
    data: GSData,
    threshold: float = 40.0,
    peak_freq_range: tuple[float, float] = (80, 200),
    mean_freq_range: tuple[float, float] | None = None,
):
    """
    Filters out whole integrations that have high power > 80 MHz.

    Parameters
    ----------
    threshold
        This is the threshold beyond which the peak power causes the integration to be
        flagged. The units of the threhsold are 10*log10(peak_power / mean), where the
        mean is the mean power of spectrum in the same frequency range (omitting
        power spikes > peak_power/10)
    peak_freq_range
        The range of frequencies over which to search for the peak.
    mean_freq_range
        The range of frequencies over which to take a mean to compare to the peak.
        By default, the same as the ``peak_freq_range``.
    """
    if peak_freq_range[0] >= peak_freq_range[1]:
        raise ValueError(
            f"The frequency range of the peak must be non-zero, got {peak_freq_range}"
        )

    if mean_freq_range is not None and mean_freq_range[0] >= mean_freq_range[1]:
        raise ValueError(
            f"The frequency range of the peak must be non-zero, got {peak_freq_range}"
        )

    mask = (
        (data.freq_array > peak_freq_range[0]) & 
        (data.freq_array <= peak_freq_range[1])
    )

    if not np.any(mask):
        return np.zeros(data.ntimes, dtype=bool)

    spec = data.spectra[..., mask]
    peak_power = spec.max(axis=-1)

    if mean_freq_range is not None:
        mask = (
            (data.freq_array > mean_freq_range[0]) & 
            (data.freq_array <= mean_freq_range[1])
        )
        if not np.any(mask):
            return np.zeros(data.ntimes, dtype=bool)

        spec = data.spectrum[:, mask]

    mean, _ = averaging.weighted_mean(
        spec,
        weights=((spec > 0) & ((spec.transpose(0, 1, 3, 2) < peak_power / 10).transpose(0, 1, 3, 2))).astype(float),
        axis=-1,
    )
    peak_power = 10 * np.log10(peak_power / mean)
    return peak_power > threshold

@register_gsprocess
@gsdata_filter(axis="time")
def peak_power_filter(
    *,
    data: GSData,
    threshold: float = 40.0,
    peak_freq_range: tuple[float, float] = (80, 200),
    mean_freq_range: tuple[float, float] | None = None,
):
    """
    Filters out whole integrations that have high power > 80 MHz.

    Parameters
    ----------
    threshold
        This is the threshold beyond which the peak power causes the integration to be
        flagged. The units of the threhsold are 10*log10(peak_power / mean), where the
        mean is the mean power of spectrum in the same frequency range (omitting
        power spikes > peak_power/10)
    peak_freq_range
        The range of frequencies over which to search for the peak.
    mean_freq_range
        The range of frequencies over which to take a mean to compare to the peak.
        By default, the same as the ``peak_freq_range``.
    """
    return _peak_power_filter(
        data=data,
        threshold=threshold,
        peak_freq_range=peak_freq_range,
        mean_freq_range=mean_freq_range,
    )

@register_gsprocess
@gsdata_filter(axis="time")
def peak_orbcomm_filter(
    *,
    data: GSData,
    threshold: float = 40.0,
    mean_freq_range: tuple[float, float] | None = (80, 200),
):
    """
    Filters out whole integrations that have high power between (137, 138) MHz.

    Parameters
    ----------
    threshold
        This is the threshold beyond which the peak power causes the integration to be
        flagged. The units of the threhsold are 10*log10(peak_power / mean), where the
        mean is the mean power of spectrum in the ``mean_freq_range`` (omitting
        power spikes > peak_power/10)
    mean_freq_range
        The range of frequencies over which to take a mean to compare to the peak.
        By default, the same as the ``peak_freq_range``.
    """
    return _peak_power_filter(
        data=data,
        threshold=threshold,
        peak_freq_range=(137.0, 138.0),
        mean_freq_range=mean_freq_range,
    )

@register_gsprocess
@gsdata_filter(axis="time")
def maxfm_filter(*, data: GSData, threshold: float = 200):
    """Max FM power filter.

    This takes power of the spectrum between 80 MHz and 120 MHz(the fm range).
    In that range, it checks each frequency bin to the estimated values..
    using the mean from the side bins.
    And then takes the max of all the all values that exceeded its expected..
    value (from mean).
    Compares the max exceeded power with the threshold and if it is greater
    than the threshold given, the integration will be flagged.
    """
    fm_freq = (data.freq_array >= 88) & (data.freq_array <= 120)
    # freq mask between 80 and 120 MHz for the FM range

    if not np.any(fm_freq):
        return np.zeros(data.ntimes, dtype=bool)

    fm_power = data.spectra[..., fm_freq]

    avg = (fm_power[:, 2:] + fm_power[:, :-2]) / 2
    fm_deviation_power = np.abs(fm_power[:, 1:-1] - avg)
    maxfm = np.max(fm_deviation_power, axis=1)

    return maxfm > threshold

@register_gsprocess
@gsdata_filter(axis="time")
def rmsf_filter(
    *,
    data: GSData,
    threshold: float = 200,
    freq_range: tuple[float, float] = (60, 80),
    tload: float = 1000,
    tcal: float = 300,
):
    """
    Rmsf filter - filters out based on rms calculated between 60 and 80 MHz.

    An initial powerlaw model is calculated using the normalized frequency range.
    Data between the freq_range is clipped.
    A standard deviation is calculated using the data and the init_model.
    Then rms is calculated from the mean that is eatimated
    using the standard deviation times initmodel.
    """
    freq_mask = (data.freq_array >= freq_range[0]) & (
        data.freq_array <= freq_range[1]
    )

    if not np.any(freq_mask):
        return np.zeros(data.ntimes, dtype=bool)

    semi_calibrated_data = (data.spectra * tload) + tcal
    freq = data.freq_array[freq_mask]
    init_model = (freq / 75.0) ** -2.5

    T75 = np.sum(init_model * semi_calibrated_data[..., freq_mask], axis=-1) / np.sum(
        init_model**2
    )

    rms = np.sqrt(
        np.mean(
            (semi_calibrated_data[..., freq_mask] - np.outer(T75, init_model)) ** 2,
            axis=-1,
        )
    )

    return rms > threshold

@register_gsprocess
@gsdata_filter(axis="time")
def filter_150mhz(*, data: GSData, threshold: float):
    """Filter based on power around 150 MHz.

    This takes the RMS of the power around 153.5 MHz (in a 1.5 MHz bin), after
    subtracting the mean, then compares this to the mean power of a 1.5 MHz bin around
    157 MHz (which is expected to be cleaner). If this ratio (RMS to mean) is greater
    than 200 times the threshold given, the integration will be flagged.
    """
    if data.freq_array.max() < 157 * u.MHz:
        return np.zeros(data.ntimes, dtype=bool)

    freq_mask = (data.freq_array >= 152.75) & (data.freq_array <= 154.25)
    mean = np.mean(data.spectra[..., freq_mask], axis=-1)
    rms = np.sqrt(np.mean((data.spectra[..., freq_mask] - mean) ** 2))

    freq_mask2 = (data.freq_array >= 156.25) & (data.freq_array <= 157.75)
    av = np.mean(data.spectrum[..., freq_mask2], axis=-1)
    d = 200.0 * np.sqrt(rms) / av

    return d > threshold

@register_gsprocess
@gsdata_filter(axis="time")
def power_percent_filter(
    *,
    data: GSData,
    freq_range: tuple[float, float] = (100, 200),
    min_threshold: float = -0.7,
    max_threshold: float = 3,
):
    """Filter for the power above 100 MHz seen in swpos 0.

    Calculates the percentage of power between 100 and 200 MHz
    & when the switch is in position 0.
    And flags integrations if the percentage is above or below the given threshold.
    """
    if data.data_unit != "power" or data.nloads != 3 or 'ant' not in data.loads:
        raise ValueError("Cannot perform power percent filter on non-power data!")

    p0 = data.spectra[data.loads.index('ant')]

    mask = (
        (data.freq_array > freq_range[0]) & 
        (data.freq_array <= freq_range[1])
    )

    if not np.any(mask):
        return np.zeros(data.ntimes, dtype=bool)

    ppercent = 100 * np.sum(p0[..., mask], axis=-1) / np.sum(p0, axis=-1)
    return (ppercent < min_threshold) | (ppercent > max_threshold)


# @gsdata_filter(axis="day", data_type=GSData)
# def day_filter(
#     *, data: CombinedData, dates: Sequence[datetime.datetime | tuple[int, int]]
# ):
#     """Filter out specific days."""
#     filter_dates = []
#     for date in dates:
#         if isinstance(date, datetime.date):
#             date = (date.year, ymd_to_jd(date.year, date.month, date.day))
#             filter_dates.append(date)
#         elif len(date) == 3:
#             filter_dates.append(tuple(date)[:2])
#         elif len(date) == 2:
#             filter_dates.append(tuple(date))
#         else:
#             raise ValueError(f"date '{date}' cannot be parsed as a date.")

#     return np.array([date[:2] in filter_dates for date in data.dates])


# @gsdata_filter(axis="day", data_type=(CombinedData, CombinedBinnedData))
# def day_rms_filter(
#     *,
#     data: CombinedData | CombinedBinnedData,
#     gha_min: float = 0,
#     gha_max: float = 24,
#     f_low: float = 0.0,
#     f_high: float = np.inf,
#     rms_threshold: float,
#     weighted: bool = False,
# ):
#     """Filter out days based on the rms of the residuals."""
#     gha = (data.ancillary["gha_edges"][1:] + data.ancillary["gha_edges"][:-1]) / 2
#     filter_dates = []
#     mask = (gha > gha_min) & (gha < gha_max)
#     for param, resid, weight, day in zip(
#         data.model_params, data.resids, data.weights, data.dates
#     ):
#         if np.sum(weight[mask]) == 0:
#             continue

#         mean_p, mean_r, mean_w = averaging.bin_gha_unbiased_regular(
#             params=param[mask],
#             resids=resid[mask],
#             weights=weight[mask],
#             gha=gha[mask],
#             bins=np.array([gha_min, gha_max]),
#         )
#         freq_mask = (data.raw_frequencies >= f_low) & (data.raw_frequencies <= f_high)
#         if weighted:
#             rms = np.sqrt(
#                 averaging.weighted_mean(
#                     data=mean_r[0, freq_mask] ** 2, weights=mean_w[0, freq_mask]
#                 )[0]
#             )
#         else:
#             rms = np.sqrt(np.mean(mean_r[0, freq_mask] ** 2))

#         if rms > rms_threshold:
#             filter_dates.append(day)
#     return np.array([date in filter_dates for date in data.dates])
