"""Functions for working with data from a thermistor."""

import numbers
from collections.abc import Sequence
from typing import Self

import attrs
import numpy as np
from astropy import units as un
from astropy.table import QTable
from astropy.time import TIME_DELTA_SCALES, Time, TimeDelta

from .. import types as tp
from ..io.serialization import hickleable
from ..io.thermistor import read_thermistor_csv

IgnoreTimesType = int | float | un.Quantity[un.percent] | un.Quantity[un.s]


def ignore_ntimes(times: Time, ignore_times: IgnoreTimesType) -> int:
    """Number of time integrations to ignore from the start of the observation.

    Parameters
    ----------
    times
        The times of each integration.
    ignore_times
        What to ignore. If a number (int, numpy integer, or a float with an integer
        value), it is the number of integrations to ignore. If a Quantity with units
        of time, all integrations up to (and including) that time after the first
        integration are ignored (all integrations, if it is longer than the
        observation). If a Quantity in percent, that percentage of the integrations
        is ignored.

    Returns
    -------
    int
        The number of integrations to ignore.
    """
    if isinstance(ignore_times, un.Quantity):
        if ignore_times.unit.is_equivalent(un.second):
            time_since_start = (times - times[0]).to("second")
            keep = np.where(time_since_start > ignore_times)[0]
            return int(keep[0]) if len(keep) else len(times)
        if ignore_times.unit.is_equivalent(un.percent):
            return int(len(times) * ignore_times.to(un.dimensionless_unscaled))
        raise TypeError("ignore_times is not a valid type!")

    if isinstance(ignore_times, numbers.Integral):
        return int(ignore_times)

    if isinstance(ignore_times, numbers.Real):
        if not float(ignore_times).is_integer():
            raise ValueError(
                "ignore_times given as a number must be a whole number of "
                f"integrations, got {ignore_times}"
            )
        return int(ignore_times)

    raise TypeError("ignore_times is not a valid type!")


def get_temperature_thermistor(
    resistance: tp.OhmType,
    coeffs: str | Sequence[float] = "oven_industries_TR136_170",
) -> tp.TemperatureType:
    """
    Convert resistance of a thermistor to temperature.

    Uses a pre-defined set of standard coefficients.

    Parameters
    ----------
    resistance
    coeffs : str or len-3 iterable of floats, optional
        If str, should be an identifier of a standard set of coefficients, otherwise,
        should specify the coefficients.

    Returns
    -------
    temperature
        The temperature for each `resistance` given.
    """
    if isinstance(coeffs, str):
        # Steinhart-Hart coefficients
        coeffs = {
            "oven_industries_TR136_170": [1.03514e-3, 2.33825e-4, 7.92467e-8],
            "omega_ON_930_44006": [
                0.001296751267466723,
                0.00019737361897609893,
                3.0403175473012516e-7,
            ],
        }[coeffs]

    assert len(coeffs) == 3

    # TK in Kelvin
    return (
        1
        * un.K
        / (
            coeffs[0]
            + coeffs[1] * np.log(resistance.to_value("ohm"))
            + coeffs[2] * (np.log(resistance.to_value("ohm"))) ** 3
        )
    )


def _is_well_ordered(times: Time, min_spacing: un.Quantity = 1 * un.us) -> bool:
    """Whether ``times`` are non-decreasing with no near-coincident distinct entries.

    Consecutive entries must either be identical (same internal representation) or
    increase by more than ``min_spacing``. This guarantees that comparisons of the
    form ``t - times[i] <= dt`` are monotonic in ``i`` despite the (sub-picosecond)
    rounding of astropy's two-double time arithmetic.
    """
    if len(times) < 2:
        return True
    identical = (times.jd1[1:] == times.jd1[:-1]) & (times.jd2[1:] == times.jd2[:-1])
    return bool(np.all(identical | ((times[1:] - times[:-1]) > min_spacing)))


def _first_true(predicate, n_items: int, n_queries: int) -> np.ndarray:
    """Vectorised binary search for the first index at which a predicate is true.

    ``predicate(idx)`` takes an integer array of length ``n_queries`` (one candidate
    index per query) and returns a boolean array of the same length. For each query
    the predicate must be monotonic in the index (False ... False True ... True).
    Returns, per query, the first index in ``[0, n_items)`` where it is True, or
    ``n_items`` if it is never True.
    """
    lo = np.zeros(n_queries, dtype=int)
    hi = np.full(n_queries, n_items, dtype=int)
    while np.any(lo < hi):
        active = lo < hi
        mid = (lo + hi) // 2
        ptrue = predicate(np.minimum(mid, n_items - 1))
        lo = np.where(active & ~ptrue, mid + 1, lo)
        hi = np.where(active & ptrue, mid, hi)
    return lo


def _thermistor_indices_sorted(
    spec: Time, therm: Time, deltat: TimeDelta
) -> np.ndarray:
    """Thermistor matching for well-ordered readings, by binary search.

    Equivalent to :func:`_thermistor_indices_scan` (and to the original sequential
    scan) when ``therm`` satisfies :func:`_is_well_ordered`. Every comparison is the
    same astropy ``TimeDelta`` comparison as in the scan, so results are identical.

    For each spectrum time ``d`` the readings split into three contiguous runs:
    more than ``deltat`` before ``d`` (indices ``< a``), within ``[0, deltat]``
    before ``d`` (``a <= i < c``) and after ``d`` (``>= c``). The scan's running
    start index after processing ``d`` is the running maximum of ``min(a, c)``,
    and ``d`` is matched to ``max(start, a)`` if that is below ``c``.
    """
    zero = TimeDelta(0 * un.s)
    n_therm = len(therm)
    n_spec = len(spec)

    first_recent = _first_true(lambda i: (spec - therm[i]) <= deltat, n_therm, n_spec)
    first_future = _first_true(lambda i: ~((spec - therm[i]) >= zero), n_therm, n_spec)

    passed = np.minimum(first_recent, first_future)
    start = np.concatenate(([0], np.maximum.accumulate(passed)[:-1]))
    candidate = np.maximum(start, first_recent)
    return np.where(candidate < first_future, candidate, -1)


def _thermistor_indices_scan(spec: Time, therm: Time, deltat: TimeDelta) -> np.ndarray:
    """Thermistor matching by a sequential scan, vectorised over the readings.

    This reproduces the original algorithm exactly for arbitrary (even unsorted)
    reading times: for each spectrum time ``d`` in turn, scan the readings from a
    running start index and take the first with ``0 <= d - t <= deltat``; the start
    index is advanced by one for each scanned reading with ``d - t > deltat``.
    Returns ``-1`` where there is no match.
    """
    zero = TimeDelta(0 * un.s)
    n_therm = len(therm)
    out = np.full(len(spec), -1, dtype=int)
    indx = 0
    for k in range(len(spec)):
        if indx >= n_therm:
            continue
        diff = spec[k] - therm[indx:]
        ge0 = diff >= zero
        within = diff <= deltat
        too_old = ge0 & ~within
        hits = np.flatnonzero(ge0 & within)
        if hits.size:
            out[k] = indx + hits[0]
            indx += int(np.count_nonzero(too_old[: hits[0]]))
        else:
            indx += int(np.count_nonzero(too_old))
    return out


def voltage_to_resistance(
    voltage: tp.VoltageType,
    load_resistance: tp.OhmType,
) -> tp.OhmType:
    """
    Convert thermistor voltage reading to resistance.

    Parameters
    ----------
    voltage
        The voltage reading from the thermistor circuit.
    load_resistance
        The resistance of the load resistor in series with the thermistor.

    Returns
    -------
    resistance
        The resistance of the thermistor.
    """
    return load_resistance * voltage / (5 * un.V - voltage)


@hickleable
@attrs.define
class ThermistorReadings:
    """
    Object containing thermistor readings.

    Parameters
    ----------
    data
        The data array containing the readings.
    """

    data: QTable = attrs.field()

    @data.validator
    def _data_vld(self, att, val):
        if "times" not in val.colnames:
            raise ValueError("'times' must be in the data for ThermistorReadings")
        if "load_resistance" not in val.colnames:
            raise ValueError("'load_resistance' must be in data for THermistorReadings")

    def ignore_times(self, ignore_times: IgnoreTimesType):
        """Number of time integrations to ignore from the start of the observation."""
        n = ignore_ntimes(self.data["times"], ignore_times=ignore_times)
        return attrs.evolve(self, data=self.data[n:])

    @classmethod
    def from_csv(
        cls,
        path: tp.PathLike,
        ignore_times: IgnoreTimesType = 0,
    ) -> Self:
        """Generate the object from an io.Resistance object."""
        return ThermistorReadings(data=read_thermistor_csv(path)).ignore_times(
            ignore_times
        )

    def get_physical_temperature(self) -> tp.TemperatureType:
        """The associated thermistor temperature in K."""
        return get_temperature_thermistor(self.data["load_resistance"])

    def get_thermistor_indices(self, timestamps: Time) -> list[int | None]:
        """Get the index of the closest therm measurement for each spectrum.

        For each timestamp (in the order given), this returns the index of the first
        thermistor reading, at or after a running start index, that is at most one
        reading interval (``times[1] - times[0]``) *before* the timestamp (or at the
        same time). The running start index only moves forward, past readings that
        were more than one interval before some earlier timestamp. Timestamps with no
        such reading get ``np.nan``.

        Parameters
        ----------
        timestamps
            The times (e.g. of spectra) for which to find a thermistor reading.

        Returns
        -------
        list
            For each timestamp, the integer index of the matched reading, or
            ``np.nan`` if there is none.
        """
        thermistor_timestamps = self.data["times"]

        # Raises IndexError for fewer than two readings, as it always has.
        deltat = thermistor_timestamps[1] - thermistor_timestamps[0]

        if len(timestamps) == 0:
            return []

        # Do the time subtractions in the scale astropy itself uses for Time - Time
        # (e.g. TAI for UTC times), converting each array only once.
        scale = timestamps.scale if timestamps.scale in TIME_DELTA_SCALES else "tai"
        spec = getattr(timestamps, scale)
        therm = getattr(thermistor_timestamps, scale)

        if _is_well_ordered(therm):
            closest = _thermistor_indices_sorted(spec, therm, deltat)
        else:
            closest = _thermistor_indices_scan(spec, therm, deltat)

        return [np.nan if c < 0 else int(c) for c in closest]
