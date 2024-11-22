"""Tests of the lstbin module."""
import numpy as np
from edges_analysis.averaging.lstbin import average_over_times
from pygsdata import GSData


def test_averaging_over_times(gsd_ones: GSData):
    """Testr that averaging over times doesn't error out."""
    new = average_over_times(gsd_ones)
    assert np.all(new.data == 1.0)
