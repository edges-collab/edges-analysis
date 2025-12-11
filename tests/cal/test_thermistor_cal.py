"""Tests of the thermistor calibration functions."""

import astropy.units as un

from edges.cal.thermistor import voltage_to_resistance
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
