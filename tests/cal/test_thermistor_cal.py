"""Tests of the thermistor calibration functions."""

import astropy.units as un
import numpy as np
import pytest
from astropy.time import Time

from edges.cal.thermistor import ignore_ntimes, voltage_to_resistance
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
