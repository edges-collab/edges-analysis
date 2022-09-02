"""Functions that identify and flag bad data in various ways."""
from __future__ import annotations

import functools
import logging
import numpy as np
import yaml
from astropy import units as u
from attrs import define
from edges_cal import types as tp
from edges_cal import xrfi as rfi
from edges_cal.xrfi import ModelFilterInfoContainer, model_filter
from pathlib import Path
from typing import Callable, Literal, Sequence

from .. import tools
from ..averaging import averaging, lstbin
from ..data import DATA_PATH
from ..gsdata import GSData, gsregister

logger = logging.getLogger(__name__)


class _GSDataFilter:
    def __init__(
        self,
        func: Callable,
        axis: Literal["time", "freq", "day", "all", "object"],
        multi_data: bool = False,
    ):
        self.func = func
        self.axis = axis
        self.multi_data = multi_data
        functools.update_wrapper(self, func, updated=())

    def __call__(
        self,
        data: Sequence[tp.PathLike | GSData],
        *,
        write: bool = False,
        flag_id: str = None,
        **kwargs,
    ) -> GSData | Sequence[GSData]:
        # Read all the data, in case they haven't been turned into objects yet.
        # And check that everything is the right type.
        if isinstance(data, (Path, str)):
            data = GSData.from_file(data)

        if self.multi_data and isinstance(data, (GSData, Path, str)):
            data = [data if isinstance(data, GSData) else GSData.from_file(data)]
        elif not self.multi_data and not isinstance(data, GSData):
            raise TypeError(
                f"'{self.func.__name__}' only accepts single GSData objects as data."
            )

        def per_file_processing(data: GSData, flags: np.ndarray):
            old = np.sum(data.complete_flags)

            outflags = np.zeros_like(data.complete_flags)

            # Broadcast the computed flags to the correct shape.
            if self.axis == "freq":
                outflags |= flags
            elif self.axis == "time":
                outflags[..., flags, :] = True
            elif self.axis == "all":
                outflags[flags] = True
            elif self.axis == "object":
                outflags |= flags

            data = data.add_flags(
                flag_id or self.func.__name__, outflags, append_to_file=write
            )

            if np.all(flags):
                logger.warning(
                    f"{data.get_initial_yearday(hours=True)} was fully flagged "
                    f"during {self.func.__name__} filter"
                )
            else:
                sz = outflags.size / 100
                new = np.sum(outflags)
                tot = np.sum(data.complete_flags)

                logger.info(
                    f"'{data.get_initial_yearday(hours=True)}': "
                    f"{old / sz:.2f} + {new / sz:.2f} → "
                    f"{tot / sz:.2f}% [bold]<+{(tot - old) / sz:.2f}%>[/] "
                    f"flagged after [blue]{self.func.__name__}[/]"
                )

            return data

        this_flag = self.func(data=data, **kwargs)

        if self.multi_data:
            data = [
                per_file_processing(d, out_flg) for d, out_flg in zip(data, this_flag)
            ]
        else:
            data = per_file_processing(data, this_flag)

        return data


@define
class gsdata_filter:  # noqa: N801
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

    axis: Literal["time", "freq", "day", "all"]
    multi_data: bool = False

    def __call__(self, func: Callable) -> Callable:
        """Wrap the function in a GSDataFilter instance."""
        return _GSDataFilter(func, self.axis, self.multi_data)


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


@gsregister("filter")
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
    minima = minima or {}
    maxima = maxima or {}

    flags = np.zeros(len(data.time_array), dtype=bool)

    def filt(condition, message, flags):
        nflags = np.sum(flags)

        # Sometimes, the auxiliary data will be shape (Ntimes, Nloads)
        # In this case, if any load is bad, all should be flagged.
        if condition.ndim == 2:
            condition = np.any(condition, axis=1)

        flags |= condition
        if nnew := np.sum(flags) - nflags:
            logger.info(f"{nnew}/{len(flags) - nflags} times flagged due to {message}")

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


@gsregister("filter")
@gsdata_filter(axis="time")
def sun_filter(
    *,
    data: GSData,
    elevation_range: tuple[float, float],
) -> np.ndarray:
    """
    Perform a filter based on sun position.

    Parameters
    ----------
    elevation_range
        The minimum and maximum allowed sun elevation in degrees
    """
    _, el = data.get_sun_azel()
    return (el < elevation_range[0]) | (el > elevation_range[1])


@gsregister("filter")
@gsdata_filter(axis="time")
def moon_filter(
    *,
    data: GSData,
    elevation_range: tuple[float, float],
) -> np.ndarray:
    """
    Perform a filter based on sun position.

    Parameters
    ----------
    elevation_range
        The minimum and maximum allowed sun elevation.
    """
    _, el = data.get_moon_azel()
    return (el < elevation_range[0]) | (el > elevation_range[1])


@define
class _RFIFilterFactory:
    method: str

    @property
    def __name__(self):
        return f"rfi_{self.method}_filter"

    @property
    def __docstring__(self):
        return getattr(rfi, self.method).__doc__

    def __call__(
        self,
        data: GSData,
        *,
        n_threads: int = 1,
        freq_range: tuple[float, float] = (40, 200),
        **kwargs,
    ):
        mask = (data.freq_array.to_value("MHz") >= freq_range[0]) & (
            data.freq_array.to_value("MHz") <= freq_range[1]
        )

        flags = data.complete_flags

        out_flags = tools.run_xrfi(
            method=self.method,
            spectrum=data.spectra[..., mask],
            freq=data.freq_array[mask].to_value("MHz"),
            flags=flags[..., mask],
            weights=data.nsamples[..., mask],
            n_threads=n_threads,
            **kwargs,
        )

        out = np.zeros_like(flags)
        out[..., mask] = out_flags

        return out


rfi_model_filter = gsregister("filter")(
    gsdata_filter(axis="all")(_RFIFilterFactory("model"))
)
rfi_model_sweep_filter = gsregister("filter")(
    gsdata_filter(axis="all")(_RFIFilterFactory("model_sweep"))
)
rfi_watershed_filter = gsregister("filter")(
    gsdata_filter(axis="all")(_RFIFilterFactory("watershed"))
)


@gsregister("filter")
@gsdata_filter(axis="freq")
def rfi_explicit_filter(*, data: GSData, file: tp.PathLike | None = None):
    """A filter of explicit channels of RFI."""
    if file is None:
        file = DATA_PATH / "known_rfi_channels.yaml"

    return rfi.xrfi_explicit(
        data.freq_array,
        rfi_file=file,
    )


@gsregister("filter")
@gsdata_filter(axis="time")
def negative_power_filter(*, data: GSData):
    """Filter out integrations that have *any* negative/zero power.

    These integrations obviously have some weird stuff going on.
    """
    return np.array([np.any(data.spectra[slc] <= 0) for slc in data.time_iter()])


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
            "The freq range of the mean must be a tuple with first less than second "
            f"value, got {mean_freq_range}"
        )

    freqs = data.freq_array.to_value("MHz")
    mask = (freqs > peak_freq_range[0]) & (freqs <= peak_freq_range[1])

    if not np.any(mask):
        return np.zeros(data.ntimes, dtype=bool)

    spec = data.spectra[..., mask]
    peak_power = spec.max(axis=-1)

    if mean_freq_range is not None:
        mask = (freqs > mean_freq_range[0]) & (freqs <= mean_freq_range[1])

        if not np.any(mask):
            return np.zeros(data.ntimes, dtype=bool)

        spec = data.spectrum[:, mask]

    mean, _ = averaging.weighted_mean(
        spec,
        weights=(
            (spec > 0)
            & ((spec.transpose(0, 1, 3, 2) < peak_power / 10).transpose(0, 1, 3, 2))
        ).astype(float),
        axis=-1,
    )
    peak_power = 10 * np.log10(peak_power / mean)
    return peak_power > threshold


@gsregister("filter")
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


@gsregister("filter")
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


@gsregister("filter")
@gsdata_filter(axis="all")
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
    freqs = data.freq_array.to_value("MHz")
    fm_freq = (freqs >= 88) & (freqs <= 120)
    # freq mask between 80 and 120 MHz for the FM range

    if not np.any(fm_freq):
        return np.zeros(data.ntimes, dtype=bool)

    fm_power = data.spectra[..., fm_freq]

    avg = (fm_power[..., 2:] + fm_power[..., :-2]) / 2
    fm_deviation_power = np.abs(fm_power[..., 1:-1] - avg)
    maxfm = np.max(fm_deviation_power, axis=-1)

    return maxfm > threshold


@gsregister("filter")
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
    freqs = data.freq_array.to_value("MHz")
    freq_mask = (freqs >= freq_range[0]) & (freqs <= freq_range[1])

    if not np.any(freq_mask):
        return np.zeros(data.ntimes, dtype=bool)

    semi_calibrated_data = (data.spectra * tload) + tcal
    freq = data.freq_array.value[freq_mask]
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


@gsregister("filter")
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

    freq_mask = (data.freq_array >= 152.75 * u.MHz) & (
        data.freq_array <= 154.25 * u.MHz
    )
    mean = np.mean(data.spectra[..., freq_mask], axis=-1)
    rms = np.sqrt(np.mean((data.spectra[..., freq_mask] - mean) ** 2))

    freq_mask2 = (data.freq_array >= 156.25 * u.MHz) & (
        data.freq_array <= 157.75 * u.MHz
    )
    av = np.mean(data.spectrum[..., freq_mask2], axis=-1)
    d = 200.0 * np.sqrt(rms) / av

    return d > threshold


@gsregister("filter")
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
    if data.data_unit != "power" or data.nloads != 3 or "ant" not in data.loads:
        raise ValueError("Cannot perform power percent filter on non-power data!")

    p0 = data.spectra[data.loads.index("ant")]

    freqs = data.freq_array.to_value("MHz")
    mask = (freqs > freq_range[0]) & (freqs <= freq_range[1])

    if not np.any(mask):
        return np.zeros(data.ntimes, dtype=bool)

    ppercent = 100 * np.sum(p0[..., mask], axis=-1) / np.sum(p0, axis=-1)
    return (ppercent < min_threshold) | (ppercent > max_threshold)


@gsdata_filter(axis="object")
def object_rms_filter(
    data: GSData,
    rms_threshold: float,
    gha_min: float = 0,
    gha_max: float = 24,
    f_low: float = 0.0,
    f_high: float = np.inf,
    weighted: bool = False,
) -> bool:
    """Filter out an entire object based on the rms of the residuals."""
    if data.ntimes > 1:
        data = lstbin.lst_bin(data, first_edge=gha_min, binsize=gha_max - gha_min)
    data = data.select_freqs(range=(f_low * u.MHz, f_high * u.MHz))

    if weighted:
        rms = np.sqrt(
            averaging.weighted_mean(data=data.resids**2, weights=data.nsamples)[0]
        )
    else:
        rms = np.sqrt(np.mean(data.resids**2))

    return rms > rms_threshold
