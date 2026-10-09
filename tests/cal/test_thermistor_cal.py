"""Tests of the thermistor calibration functions."""

import astropy.units as un
import numpy as np
import pytest
from astropy.table import QTable
from astropy.time import TIME_DELTA_SCALES, Time, TimeDelta

from edges.cal.thermistor import (
    ThermistorReadings,
    _is_well_ordered,
    _thermistor_indices_scan,
    ignore_ntimes,
    voltage_to_resistance,
)
from edges.types import OhmType, VoltageType


class TestVoltageToResistance:
    """Tests of the voltage_to_resistance function."""

    def test_basic(self):
        """Test basic functionality."""
        voltage: VoltageType = 2.5 * un.V
        load_resistance: OhmType = 10000 * un.ohm

        resistance = voltage_to_resistance(voltage, load_resistance)
        expected_resistance: OhmType = 10000 * un.ohm
        assert resistance == expected_resistance


class TestIgnoreNtimes:
    """Tests of the number of integrations to ignore at the start."""

    times = Time("2023-01-01T00:00:00") + np.arange(10) * 30 * un.s

    @pytest.mark.parametrize("n", [3, np.int64(3), np.int32(3), 3.0, np.float64(3.0)])
    def test_integer_like(self, n):
        assert ignore_ntimes(self.times, n) == 3

    def test_fractional_float(self):
        with pytest.raises(ValueError, match="whole number"):
            ignore_ntimes(self.times, 2.5)

    def test_percent(self):
        assert ignore_ntimes(self.times, 20 * un.percent) == 2

    def test_seconds(self):
        # Integrations strictly after 45 s from the start are kept.
        assert ignore_ntimes(self.times, 45 * un.s) == 2
        assert ignore_ntimes(self.times, 1 * un.min) == 3

    def test_seconds_longer_than_observation(self):
        assert ignore_ntimes(self.times, 1 * un.hour) == len(self.times)

    def test_bad_unit(self):
        with pytest.raises(TypeError, match="ignore_times is not a valid type"):
            ignore_ntimes(self.times, 3 * un.m)


def _reference_thermistor_indices(thermistor_timestamps, timestamps):
    """The original (sequential, scalar astropy) thermistor matching algorithm."""
    closest = []
    indx = 0
    deltat = thermistor_timestamps[1] - thermistor_timestamps[0]

    for d in timestamps:
        if indx >= len(thermistor_timestamps):
            closest.append(np.nan)
            continue

        for i, td in enumerate(thermistor_timestamps[indx:], start=indx):
            if d - td >= TimeDelta(0 * un.s):
                if d - td <= deltat:
                    closest.append(i)
                    break
                indx += 1
        else:
            closest.append(np.nan)

    return closest


def _readings(times: Time) -> ThermistorReadings:
    return ThermistorReadings(
        QTable({"times": times, "load_resistance": np.ones(len(times)) * 1e4 * un.ohm})
    )


def _assert_same_indices(
    therm_times: Time, spec_times: Time, well_ordered: bool = True
):
    new = _readings(therm_times).get_thermistor_indices(spec_times)
    ref = _reference_thermistor_indices(therm_times, spec_times)
    assert len(new) == len(ref)

    # Check which implementation was used, and that the general scan agrees with
    # the reference whichever it was.
    scale = spec_times.scale if spec_times.scale in TIME_DELTA_SCALES else "tai"
    spec = getattr(spec_times, scale)
    therm = getattr(therm_times, scale)
    assert _is_well_ordered(therm) == well_ordered
    scan = _thermistor_indices_scan(spec, therm, therm_times[1] - therm_times[0])
    np.testing.assert_array_equal(scan, np.nan_to_num(ref, nan=-1).astype(int))

    for n, r in zip(new, ref, strict=True):
        if isinstance(r, float):
            assert isinstance(n, float)
            assert np.isnan(n)
        else:
            assert type(n) is int
            assert n == r


class TestGetThermistorIndices:
    """The vectorised matching must reproduce the original sequential scan exactly."""

    t0 = Time("2023-01-01T00:00:00", scale="utc")
    cadence = 15.0

    def therm(self, n=60):
        return self.t0 + np.arange(n) * self.cadence * un.s

    @pytest.mark.parametrize("seed", [0, 1, 2])
    def test_random_sorted_spectra(self, seed):
        rng = np.random.default_rng(seed)
        therm = self.therm()
        # Spectra from before the log starts until after it ends.
        offsets = np.sort(rng.uniform(-200, 60 * self.cadence + 200, 70))
        _assert_same_indices(therm, self.t0 + offsets * un.s)

    @pytest.mark.parametrize("seed", [0, 1])
    def test_random_unsorted_spectra(self, seed):
        rng = np.random.default_rng(seed)
        therm = self.therm()
        offsets = rng.uniform(-100, 60 * self.cadence + 100, 50)
        _assert_same_indices(therm, self.t0 + offsets * un.s)

    def test_exactly_on_readings_and_interval_edges(self):
        therm = self.therm(20)
        # Exactly on readings, exactly one interval after, and just either side.
        base = np.arange(20) * self.cadence
        offsets = np.concatenate([base, base + self.cadence, base + 1e-3, base - 1e-3])
        _assert_same_indices(therm, self.t0 + np.sort(offsets) * un.s)
        _assert_same_indices(therm, therm)
        _assert_same_indices(therm, therm + (therm[1] - therm[0]))

    def test_spectra_before_and_after_log(self):
        therm = self.therm(20)
        before = self.t0 - np.arange(10, 0, -1) * 37 * un.s
        after = therm[-1] + np.arange(1, 10) * 37 * un.s
        _assert_same_indices(therm, before)
        _assert_same_indices(therm, after)
        _assert_same_indices(therm, after[::-1])

    def test_gaps_and_jitter(self):
        rng = np.random.default_rng(42)
        steps = self.cadence + rng.normal(0, 2, 59)
        steps[[10, 30, 31]] = [200.0, 500.0, 80.0]  # gaps in the log
        therm = self.t0 + np.concatenate([[0], np.cumsum(steps)]) * un.s
        offsets = np.sort(rng.uniform(-50, np.sum(steps) + 50, 80))
        _assert_same_indices(therm, self.t0 + offsets * un.s)

    def test_duplicate_readings(self):
        therm = self.t0 + np.array([0, 15, 15, 30, 45, 45, 45, 60]) * un.s
        offsets = np.array([0, 10, 15, 20, 30, 44, 45, 50, 60, 75, 76])
        _assert_same_indices(therm, self.t0 + offsets * un.s)
        # Duplicate first reading: zero cadence.
        therm = self.t0 + np.array([0, 0, 15, 30]) * un.s
        _assert_same_indices(therm, self.t0 + offsets * un.s)

    def test_unsorted_readings(self):
        # Exercises the general (non-monotonic) path.
        rng = np.random.default_rng(3)
        therm = self.t0 + rng.uniform(0, 600, 40) * un.s
        offsets = np.sort(rng.uniform(-50, 650, 40))
        _assert_same_indices(therm, self.t0 + offsets * un.s, well_ordered=False)
        # Decreasing log (negative cadence).
        _assert_same_indices(
            self.therm(30)[::-1], self.t0 + offsets * un.s, well_ordered=False
        )

    def test_near_coincident_readings(self):
        # Distinct readings closer than the well-ordered spacing use the general path.
        therm = self.t0 + np.array([0, 15, 15 + 1e-8, 30, 45]) * un.s
        offsets = np.array([0, 15, 15 + 5e-9, 20, 30, 60])
        _assert_same_indices(therm, self.t0 + offsets * un.s, well_ordered=False)

    def test_across_leap_second(self):
        t0 = Time("2016-12-31T23:59:00", scale="utc")
        therm = t0 + np.arange(12) * 10 * un.s
        spec = Time(
            [
                "2016-12-31T23:59:50",
                "2016-12-31T23:59:59",
                "2016-12-31T23:59:60",
                "2017-01-01T00:00:00",
                "2017-01-01T00:00:05",
                "2017-01-01T00:00:59",
            ],
            scale="utc",
        )
        _assert_same_indices(therm, spec)

    def test_different_scales(self):
        therm = self.therm(30)
        spec = self.t0 + np.linspace(-20, 480, 25) * un.s
        _assert_same_indices(therm, spec.tt)
        _assert_same_indices(therm.tt, spec)

    def test_empty_timestamps(self):
        assert _readings(self.therm(5)).get_thermistor_indices(self.therm(3)[:0]) == []

    def test_fewer_than_two_readings(self):
        with pytest.raises(IndexError):
            _readings(self.therm(1)).get_thermistor_indices(self.therm(3))
