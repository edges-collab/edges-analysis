"""Functions that identify and flag bad data in various ways."""

import functools
import logging
import numbers
import warnings
from collections.abc import Callable, Sequence

import deprecation
import hickle
import numpy as np
from astropy import units as un
from astropy.coordinates import AltAz, Angle, SkyCoord
from astropy.time import Time
from attrs import define
from pygsdata import GSData, GSFlag, gsregister
from pygsdata.select import _mask_times, select_freqs

from edges import __version__

from .. import modeling as mdl
from .. import types as tp
from ..averaging import NsamplesStrategy, averaging, get_weights_from_strategy
from .runners import run_xrfi

logger = logging.getLogger(__name__)


def gsdata_filter(multi_data: bool = False):
    """A decorator to register a filtering function as a potential filter.

    Any function that is wrapped by :func:`gsdata_filter` must implement the following
    signature::

        def fnc(
            *,
            data: GSData | Sequence[GSData],
            **kwargs
        ) -> GSFlag | Sequence[GSFlag]

    Where the ``data`` is either a single GSData object, or sequence of such
    objects.

    The return value should be a :class:`GSFlag` object, which contains the flags (or
    a sequence of them, one per input object, if ``multi_data`` is True).

    The returned wrapper has the signature ``wrapper(data, *, flag_id=None, **kwargs)``.
    It calls the filter function and adds the resulting flags to the data (under the
    name ``flag_id``, which defaults to the name of the filter function), returning
    the new :class:`GSData` object(s). Existing flags on the data are kept: the new
    flags are added alongside them.

    Parameters
    ----------
    multi_data
        Whether the filter accepts multiple objects at the same time to filter. This
        is *usually* so as to enable more accurate filtering when comparing different
        days for instance, rather than just performing a loop over the days and flagging
        each independently.
    """

    def inner(func: Callable) -> Callable:
        func.func = func  # type: ignore

        @functools.wraps(func)
        def wrapper(
            data: GSData | Sequence[GSData],
            *,
            flag_id: str | None = None,
            **kwargs,
        ) -> GSData | Sequence[GSData]:
            # Read all the data, in case they haven't been turned into objects yet.
            # And check that everything is the right type.
            if multi_data and isinstance(data, GSData):
                data = [data]
            elif not multi_data and not isinstance(data, GSData):
                raise TypeError(
                    f"'{func.__name__}' only accepts single GSData objects as data."
                )

            def per_file_processing(data: GSData, flags: GSFlag):
                old = np.sum(data.flagged_nsamples == 0)

                data = data.add_flags(flag_id or func.__name__, flags)

                if np.all(flags.flags):
                    logger.warning(
                        f"{data.name} was fully flagged during {func.__name__} filter"
                    )
                else:
                    # All numbers are reported as percentages: the percentage of the
                    # data flagged before this filter, the percentage of the new flag
                    # array that is flagged, the percentage of data flagged after this
                    # filter, and the increase.
                    new = 100 * np.sum(flags.flags) / flags.flags.size
                    tot = np.sum(data.flagged_nsamples == 0)
                    totsz = data.complete_flags.size / 100

                    rep = data.get_initial_yearday(hours=True)

                    logger.info(
                        f"'{rep}': "
                        f"{old / totsz:.2f}% + {new:.2f}% → "
                        f"{tot / totsz:.2f}% [bold]<+{(tot - old) / totsz:.2f}%>[/] "
                        f"flagged after [blue]{func.__name__}[/]"
                    )

                return data

            this_flag = func(data=data, **kwargs)

            if multi_data:
                data = [
                    per_file_processing(d, out_flg)
                    for d, out_flg in zip(data, this_flag, strict=False)
                ]
            else:
                data = per_file_processing(data, this_flag)

            return data

        return wrapper

    return inner


@gsregister("filter")
@gsdata_filter()
def aux_filter(
    *,
    data: GSData,
    minima: dict[str, float] | None = None,
    maxima: dict[str, float] | None = None,
) -> GSFlag:
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

    flags = np.zeros(data.ntimes, dtype=bool)

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
        if k not in data.auxiliary_measurements.keys():  # ruff: ignore[in-dict-keys]
            raise ValueError(
                f"{k} not in data.auxiliary_measurements. "
                f"Allowed: {data.auxiliary_measurements.keys()}"
            )
        filt(data.auxiliary_measurements[k] < v, f"{k} minimum", flags)

    for k, v in maxima.items():
        if k not in data.auxiliary_measurements.keys():  # ruff: ignore[in-dict-keys]
            raise ValueError(
                f"{k} not in data.auxiliary_measurements. "
                f"Allowed: {data.auxiliary_measurements.keys()}"
            )
        filt(data.auxiliary_measurements[k] > v, f"{k} maximum", flags)

    return GSFlag(flags=flags, axes=("time",))


@gsregister("filter")
@gsdata_filter()
def sun_filter(
    *,
    data: GSData,
    elevation_range: tuple[float, float],
) -> GSFlag:
    """
    Perform a filter based on sun position.

    Parameters
    ----------
    elevation_range
        The minimum and maximum allowed sun elevation in degrees
    """
    _, el = data.get_sun_azel()
    return GSFlag(
        flags=(el < elevation_range[0]) | (el > elevation_range[1]), axes=("time",)
    )


@gsregister("filter")
@gsdata_filter()
def moon_filter(
    *,
    data: GSData,
    elevation_range: tuple[float, float],
) -> np.ndarray:
    """
    Perform a filter based on moon position.

    Parameters
    ----------
    elevation_range
        The minimum and maximum allowed sun elevation.
    """
    _, el = data.get_moon_azel()
    return GSFlag(
        flags=(el < elevation_range[0]) | (el > elevation_range[1]), axes=("time",)
    )


@gsregister("filter")
@gsdata_filter()
def sky_coord_filter(
    *,
    data: GSData,
    coord: str | SkyCoord,
    elevation_range: tuple[Angle, Angle],
) -> GSFlag:
    """
    Perform a filter based on a sky coordinate position.

    Parameters
    ----------
    coord
        The sky coordinate to filter on.
    elevation_range
        The minimum and maximum allowed elevation (as an astropy Angle).
    """
    if isinstance(coord, str):
        coord = SkyCoord.from_name(coord)

    # Use the times of the first load, assuming that this is the antenna data.
    azalt = coord.transform_to(
        AltAz(location=data.telescope.location, obstime=Time(data.times[:, 0]))
    )
    alt = azalt.alt

    return GSFlag(
        flags=(alt < elevation_range[0]) | (alt > elevation_range[1]), axes=("time",)
    )


@gsregister("filter")
@gsdata_filter()
def galaxy_filter(
    *,
    data: GSData,
    elevation_range: tuple[Angle, Angle] = (-90 * un.deg, 0 * un.deg),
) -> GSFlag:
    """
    Perform a filter based on the Galactic center position.

    Parameters
    ----------
    elevation_range
        The minimum and maximum allowed elevation (as an astropy Angle).
    """
    return sky_coord_filter.func(
        data=data, coord="Galactic Center", elevation_range=elevation_range
    )


_RFI_FILTER_DOC = """Flag RFI in a GSData object with an xRFI method.

    This uses :func:`edges.filters.xrfi.xrfi_{method}`, applied independently to
    each spectrum (i.e. each load, polarization and time) via
    :func:`edges.filters.runners.run_xrfi`.

    Parameters
    ----------
    data : GSData
        The data to flag.
    n_threads : int
        The number of processes to use.
    freq_range : tuple[float, float]
        The range of frequencies (in MHz, inclusive) over which to flag. Data outside
        this range is not flagged.
    nsamples_strategy : NsamplesStrategy
        The strategy used to obtain the weights passed to the xRFI method.
    **kwargs
        Passed through to :func:`edges.filters.xrfi.xrfi_{method}`.
    """


@define(frozen=False, slots=False)
class _RFIFilterFactory:
    method: str

    def __attrs_post_init__(self):
        # Set a real docstring, so that the wrapped filter function has one.
        self.__doc__ = _RFI_FILTER_DOC.format(method=self.method)

    @property
    def __name__(self):
        return f"rfi_{self.method}_filter"

    def __call__(
        self,
        data: GSData,
        *,
        n_threads: int = 1,
        freq_range: tuple[float, float] = (40, 200),
        nsamples_strategy: NsamplesStrategy = NsamplesStrategy.FLAGGED_NSAMPLES,
        **kwargs,
    ):
        mask = (data.freqs.to_value("MHz") >= freq_range[0]) & (
            data.freqs.to_value("MHz") <= freq_range[1]
        )

        flags = data.complete_flags

        wgt, _ = get_weights_from_strategy(data, nsamples_strategy)

        out_flags = run_xrfi(
            method=self.method,
            spectrum=data.data[..., mask],
            freqs=data.freqs[mask].to_value("MHz"),
            weights=wgt[..., mask],
            n_threads=n_threads,
            **kwargs,
        )

        out = np.zeros_like(flags)
        out[..., mask] = out_flags

        return GSFlag(
            flags=out,
            axes=("load", "pol", "time", "freq")[-out.ndim :],
        )


rfi_iterative_filter = gsregister("filter")(
    gsdata_filter()(_RFIFilterFactory("iterative"))
)
rfi_model_filter = rfi_iterative_filter  # Backwards compatibility

rfi_watershed_filter = gsregister("filter")(
    gsdata_filter()(_RFIFilterFactory("watershed"))
)
rfi_iterative_sliding_window = gsregister("filter")(
    gsdata_filter()(_RFIFilterFactory("iterative_sliding_window"))
)


@gsregister("filter")
@gsdata_filter()
def apply_flags(*, data: GSData, flags: tp.PathLike | GSFlag):
    """Apply flags from a file."""
    if not isinstance(flags, GSFlag):
        flags = hickle.load(flags)

    return flags


@gsregister("filter")
@gsdata_filter()
def flag_frequency_ranges(
    *, data: GSData, freq_ranges: list[tuple[float, float]], invert: bool = False
):
    """Flag explicit frequency ranges.

    Parameters
    ----------
    data
        The data to flag.
    freq_ranges
        A list of tuples, each containing the start and end of a frequency range to flag
        in MHz.
    invert
        If True, invert the flagging (i.e. only *keep* the data inside the ranges
        given).
    """
    if invert:
        flags = np.ones(data.nfreqs, dtype=bool)
    else:
        flags = np.zeros(data.nfreqs, dtype=bool)

    fmhz = data.freqs.to_value("MHz")
    for fmin, fmax in freq_ranges:
        if invert:
            flags[(fmhz >= fmin) & (fmhz < fmax)] = False
        else:
            flags |= (fmhz >= fmin) & (fmhz < fmax)

    return GSFlag(
        flags=flags,
        axes=("freq",),
    )


@gsregister("filter")
@gsdata_filter()
def negative_power_filter(*, data: GSData):
    """Filter out integrations that have *any* negative/zero power.

    These integrations obviously have some weird stuff going on.
    """
    flags = np.any(data.data < 0, axis=(0, 1, 3))

    return GSFlag(flags=flags, axes=("time",))


def _peak_power_filter(
    *,
    data: GSData,
    threshold: float = 40.0,
    peak_freq_range: tuple[float, float] = (80, 200),
    mean_freq_range: tuple[float, float] | None = None,
):
    """
    Filter out whole integrations that have high power in a given frequency range.

    The statistic is that of the ``-pkpwrm`` option of Alan's C-code: the ratio
    (in dB) of the peak in ``peak_freq_range`` to the mean in ``mean_freq_range``.
    The mean only includes channels that are positive and below a tenth of the peak
    in ``mean_freq_range``. If there are no such channels, the mean is unity, so
    the statistic is just the peak in dB (in the units of the data). Non-finite
    channels are treated as zero.

    Parameters
    ----------
    threshold
        The threshold (in dB) at or beyond which the peak power causes the
        integration to be flagged. If negative, integrations are instead flagged if
        their peak power is at or below the magnitude of the threshold (and not
        below the threshold itself), i.e. only integrations with high peak power are
        kept.
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

    freqs = data.freqs.to_value("MHz")
    mask = (freqs > peak_freq_range[0]) & (freqs <= peak_freq_range[1])

    if not np.any(mask):
        return np.zeros(shape=(data.nloads, data.npols, data.ntimes), dtype=bool)

    # As in the C-code, the maxima start at zero and ignore undefined channels.
    spec = np.where(np.isfinite(data.data), data.data, 0.0)
    peak_power = np.max(spec[..., mask], axis=-1, initial=0.0)

    if mean_freq_range is not None:
        mask = (freqs > mean_freq_range[0]) & (freqs <= mean_freq_range[1])

        if not np.any(mask):
            return np.zeros(data.ntimes, dtype=bool)

    spec = spec[..., mask]
    ref = np.max(spec, axis=-1, initial=0.0)
    use = (spec > 0) & (spec < ref[..., None] / 10)

    # The tiny offsets make the mean unity if there are no usable channels.
    mean = (1e-99 + np.sum(spec * use, axis=-1)) / (1e-99 + np.sum(use, axis=-1))
    with np.errstate(divide="ignore"):
        peak_power = 10 * np.log10(peak_power / mean)

    keep = (peak_power < threshold) | ((threshold < 0) & (peak_power > abs(threshold)))
    return ~keep


@gsregister("filter")
@gsdata_filter()
def peak_power_filter(
    *,
    data: GSData,
    threshold: float = 40.0,
    peak_freq_range: tuple[float, float] = (80, 200),
    mean_freq_range: tuple[float, float] | None = None,
):
    """
    Filter out whole integrations that have high power > 80 MHz.

    Parameters
    ----------
    threshold
        This is the threshold (in dB) at or beyond which the peak power causes the
        integration to be flagged. The units of the threshold are
        10*log10(peak_power / mean), where the mean is the mean power of the
        spectrum in ``mean_freq_range``, omitting power spikes above a tenth of the
        peak in that range (and unity if there are no other channels). If negative,
        only the integrations with peak power above its magnitude are kept.
    peak_freq_range
        The range of frequencies over which to search for the peak.
    mean_freq_range
        The range of frequencies over which to take a mean to compare to the peak.
        By default, the same as the ``peak_freq_range``.
    """
    flags = _peak_power_filter(
        data=data,
        threshold=threshold,
        peak_freq_range=peak_freq_range,
        mean_freq_range=mean_freq_range,
    )

    return GSFlag(
        flags=flags,
        axes=(
            "load",
            "pol",
            "time",
        )[-flags.ndim :],
    )


@gsregister("filter")
@gsdata_filter()
def peak_orbcomm_filter(
    *,
    data: GSData,
    threshold: float = 40.0,
    mean_freq_range: tuple[float, float] | None = (80, 200),
):
    """
    Filter out whole integrations that have high power in the ORBCOMM band.

    This is the ``-pkpwrm`` cut of Alan's C-code: the peak is searched for within
    1 MHz of 137.5 MHz.

    Parameters
    ----------
    threshold
        This is the threshold (in dB) at or beyond which the peak power causes the
        integration to be flagged. The units of the threshold are
        10*log10(peak_power / mean), where the mean is the mean power of the
        spectrum in ``mean_freq_range``, omitting power spikes above a tenth of the
        peak in that range (and unity if there are no other channels). If negative,
        only the integrations with peak power above its magnitude are kept.
    mean_freq_range
        The range of frequencies over which to take a mean to compare to the peak.
        If None, the same as the ORBCOMM band.
    """
    flags = _peak_power_filter(
        data=data,
        threshold=threshold,
        peak_freq_range=(136.5, 138.5),
        mean_freq_range=mean_freq_range,
    )
    return GSFlag(
        flags=flags,
        axes=(
            "load",
            "pol",
            "time",
        )[-flags.ndim :],
    )


@gsregister("filter")
@gsdata_filter()
def single_channel_spike_filter(
    *,
    data: GSData,
    threshold: float = 200,
    freq_range: tuple[tp.FreqType, tp.FreqType] = (88 * un.MHz, 120 * un.MHz),
    absolute: bool = True,
):
    """Filter data based on single channel spikes.

    This filter detrends the data using a simple convolution kernel with weights
    [0.5, 0, 0.5], which makes single channel spikes stand out.
    The entire spectrum is flagged if the residual of the original spectrum to the
    de-trended is larger than the threshold.

    Parameters
    ----------
    data
        The data to filter.
    threshold
        The threshold on the residual beyond which the spectrum is flagged.
    freq_range
        The range of frequencies in which to search for spikes. The channels at the
        edges of the range are compared to their neighbours outside it, if any.
    absolute
        Whether to flag on the absolute value of the residual (so that dips are
        also flagged), or only on positive residuals (spikes).
    """
    freqs = data.freqs
    idx = np.flatnonzero((freqs >= freq_range[0]) & (freqs <= freq_range[1]))
    idx = idx[(idx > 0) & (idx < data.nfreqs - 1)]

    if idx.size == 0:
        return GSFlag(flags=np.zeros(data.ntimes, dtype=bool), axes=("time",))

    power = data.data
    deviation_power = power[..., idx] - (power[..., idx + 1] + power[..., idx - 1]) / 2
    if absolute:
        deviation_power = np.abs(deviation_power)

    return GSFlag(
        flags=np.max(deviation_power, axis=-1) > threshold,
        axes=("load", "pol", "time"),
    )


@gsregister("filter")
@gsdata_filter()
def maxfm_filter(*, data: GSData, threshold: float = 200):
    """Filter data based on large single-channel spikes in FM band.

    This function is only provided as a convenience when comparing to the legacy code
    that had the same filter with this name (``-maxfm``). It is really just a very
    thin wrapper around `single_channel_spike_filter`, focusing on the FM band, and
    (as in the legacy code) only flagging positive spikes.
    """
    return single_channel_spike_filter.func(
        data=data,
        threshold=threshold,
        freq_range=(88 * un.MHz, 120 * un.MHz),
        absolute=False,
    )


@gsregister("filter")
@gsdata_filter()
def filter_150mhz(*, data: GSData, threshold: float):
    """Filter data based on power around 150 MHz.

    For each integration (and each load and polarization), this computes the RMS of
    the data in the band 152.75--154.25 MHz about its mean in that band (i.e. its
    standard deviation over frequency), and the mean of the data in the band
    156.25--157.75 MHz (which is expected to be cleaner). The integration is
    flagged if::

        200 * sqrt(rms) / mean > threshold

    Note that, since the square root of the RMS is used, this statistic is not
    dimensionless: the threshold depends on the units (and scale) of the data. If the
    data does not extend to 157 MHz, nothing is flagged.

    Parameters
    ----------
    data
        The data to filter.
    threshold
        The threshold on the statistic above, beyond which integrations are flagged.
    """
    if data.freqs.max() < 157 * un.MHz:
        return GSFlag(flags=np.zeros(data.ntimes, dtype=bool), axes=("time",))

    freq_mask = (data.freqs >= 152.75 * un.MHz) & (data.freqs <= 154.25 * un.MHz)
    band = data.data[..., freq_mask]
    mean = np.mean(band, axis=-1, keepdims=True)
    rms = np.sqrt(np.mean((band - mean) ** 2, axis=-1))

    freq_mask2 = (data.freqs >= 156.25 * un.MHz) & (data.freqs <= 157.75 * un.MHz)
    av = np.mean(data.data[..., freq_mask2], axis=-1)
    d = 200.0 * np.sqrt(rms) / av

    return GSFlag(
        flags=d > threshold,
        axes=(
            "load",
            "pol",
            "time",
        ),
    )


@gsregister("filter")
@gsdata_filter()
def power_percent_filter(
    *,
    data: GSData,
    freq_range: tuple[float, float] = (100, 200),
    min_threshold: float = 0,
    max_threshold: float = 3,
):
    """Filter data based on the ratio of power in a band compared to entire dataset.

    This filter computes the sum of power from the input connected to the antenna
    within a given band, and finds the ratio within that band compared to the entired
    dataset. If that ratio is outside the thresholds given for a given timestamp, then
    the entired integration is flagged.

    Note: this is a very bespoke filter. Thresholds that make sense will depend on
    both the ``freq_range`` given, and the frequency range of the data itself. In this
    regard, it is very flexible, but care must be taken to set the parameters
    appropriately.

    Parameters
    ----------
    data : GSData
        The data to be flagged.
    freq_range : tuple[float, float]
        The frequency range of the power to be summed in the numerator, in MHz.
    min_threshold : float
        Threshold of the ratio below which the integration will be flagged.
    max_threshold : float
        Threshold of the ratio above which the integration will be flagged.
    """
    if data.data_unit != "power" or data.nloads != 3 or "ant" not in data.loads:
        raise ValueError("Cannot perform power percent filter on non-power data!")

    p0 = data.data[data.loads.index("ant")]

    freqs = data.freqs.to_value("MHz")
    mask = (freqs > freq_range[0]) & (freqs <= freq_range[1])

    if not np.any(mask):
        return GSFlag(flags=np.zeros(data.ntimes, dtype=bool), axes=("time",))

    ppercent = 100 * np.sum(p0[..., mask], axis=-1) / np.sum(p0, axis=-1)
    return GSFlag(
        flags=(ppercent < min_threshold) | (ppercent > max_threshold),
        axes=(
            "pol",
            "time",
        ),
    )


@gsregister("filter")
@gsdata_filter()
def rms_filter(
    data: GSData,
    threshold: float,
    freq_range: tuple[tp.FreqType, tp.FreqType] = (0.0 * un.MHz, np.inf * un.MHz),
    nsamples_strategy: NsamplesStrategy = NsamplesStrategy.FLAGGED_NSAMPLES,
    model: mdl.Model | None = None,
) -> bool:
    """Filter integrations based on the rms of the residuals.

    Parameters
    ----------
    data
        The data to be filtered.
    threshold
        The threshold at which to flag integrations.
    freq_range
        The frequency range to use in calculating the RMS.
    nsamples_strategy
        The strategy to use to infer weights for computing RMS.
    model
        A model to be used to fit each integration. Not required if a model
        already exists on the data.
    """
    from ..analysis.datamodel import add_model

    data = select_freqs(data, freq_range=freq_range)

    if data.data.size == 0:
        # No data in the given frequency range, so nothing to flag.
        return GSFlag(
            flags=np.zeros(shape=(data.nloads, data.npols, data.ntimes), dtype=bool),
            axes=("load", "pol", "time"),
        )

    if data.residuals is None:
        if model is None:
            raise ValueError("Cannot perform rms_filter without residuals or a model.")
        data = add_model(data=data, model=model, nsamples_strategy=nsamples_strategy)

    w = get_weights_from_strategy(data, nsamples_strategy)[0]

    rms = np.sqrt(averaging.weighted_mean(data=data.residuals**2, weights=w)[0])

    return GSFlag(
        flags=(rms > threshold),
        axes=(
            "load",
            "pol",
            "time",
        ),
    )


@gsregister("filter")
@gsdata_filter()
@deprecation.deprecated(
    deprecated_in="8.1.0",
    removed_in="9.0.0",
    current_version=__version__,
    details="Use the rms_filter function instead",
)
def rmsf_filter(
    *,
    data: GSData,
    threshold: float = 200,
    freq_range: tuple[float, float] = (60, 80),
) -> GSFlag:
    """
    Filter data based on rms calculated between 60 and 80 MHz.

    Note that this function is deprecated in favour of the more general
    :func:`rms_filter`.
    """
    warnings.warn(
        "rmsf_filter is deprecated, please use rms_filter instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    if data.data_unit not in ("uncalibrated_temp", "temperature"):
        raise ValueError(
            "Unsupported data_unit for rmsf_filter. "
            "Need temperature or uncalibrated_temp"
        )

    return rms_filter.func(
        data=data,
        threshold=threshold,
        freq_range=freq_range,
        nsamples_strategy=NsamplesStrategy.FLAGGED_NSAMPLES,
        model=mdl.LinLog(n_terms=1, beta=-2.5),
    )


@gsdata_filter()
def rms_rfi_filter(
    data: GSData,
    threshold: float = 3.0,
    nsamples_strategy: NsamplesStrategy = NsamplesStrategy.FLAGGED_NSAMPLES,
) -> GSFlag:
    """Flag specific channel-integrations via their outlier-ness compared to RMS."""
    w = get_weights_from_strategy(data, nsamples_strategy)[0]

    rms = np.sqrt(np.average(np.square(data.residuals), weights=w, axis=-1))[..., None]
    return GSFlag(
        flags=data.residuals > rms * threshold, axes=("load", "pol", "time", "freq")
    )


def _utc_day_number(t: Time) -> np.ndarray:
    """Get the Julian Day Number of the UTC calendar day containing each time.

    A UTC calendar day runs from JD ``N - 0.5`` (midnight) to ``N + 0.5``, where ``N``
    is its Julian Day Number (the JD at noon of that day).
    """
    return np.floor(t.utc.jd + 0.5).astype(int)


@gsregister("filter")
@gsdata_filter()
def explicit_day_filter(
    data: GSData,
    flag_days: Sequence[tuple[int, int] | tuple[int, int, int] | int | Time],
) -> GSFlag:
    """Filter out any data coming from specific (UTC calendar) days.

    Each day is a UTC calendar day, i.e. it runs from 00:00 to 24:00 UTC.

    Parameters
    ----------
    flag_days
        A list of days to flag. Each entry can be a 2-tuple, 3-tuple, astropy.Time or an
        integer. If a 2-tuple, it is interpreted as ``(year, day_of_year)``. If a
        3-tuple, it is interpreted as ``(year, month, day)``. If an astropy Time, the
        day(s) containing the time(s) are flagged. If an integer (including numpy
        integer types), it is interpreted as a Julian Day Number, i.e. the day whose
        noon (12:00 UTC) has a JD equal to that integer. This list is not modified.
    """
    day_numbers = []
    for day in flag_days:
        if isinstance(day, Time):
            day_numbers.extend(np.atleast_1d(_utc_day_number(day)).tolist())
        elif isinstance(day, numbers.Integral):
            day_numbers.append(int(day))
        elif isinstance(day, tuple | list) and len(day) in (2, 3):
            if len(day) == 2:
                t = Time(
                    f"{day[0]:04}:{day[1]:03}:00:00:00.000", format="yday", scale="utc"
                )
            else:
                t = Time(
                    f"{day[0]:04}-{day[1]:02}-{day[2]:02} 00:00:00.000",
                    format="iso",
                    scale="utc",
                )
            day_numbers.append(int(_utc_day_number(t)))
        else:
            raise ValueError(
                "Each entry in flag_days must be a 2-tuple, 3-tuple, astropy Time or "
                f"an integer, got {day!r}."
            )

    return GSFlag(
        flags=np.any(np.isin(_utc_day_number(data.times), day_numbers), axis=-1),
        axes=("time",),
    )


@gsregister("reduce")
def prune_flagged_integrations(data: GSData, **kwargs) -> GSData:
    """Remove integrations that are flagged for all freq-pol-loads."""
    flg = np.all(data.complete_flags, axis=(0, 1, 3))
    return _mask_times(data, ~flg)
