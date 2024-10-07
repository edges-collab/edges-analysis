"""Test the freqbin module."""

import numpy as np

from edges_analysis.averaging import freqbin
from edges_analysis.gsdata import GSFlag


def test_freq_bin_direct(gsd_ones):
    new = freqbin.freq_bin(gsd_ones, resolution=2, debias=False)
    assert len(new.freq_array) == len(gsd_ones.freq_array) // 2


def test_freqbin_gaussian(gsd_ones):
    data = gsd_ones

    new = freqbin.gauss_smooth(data, size=2)
    assert len(new.freq_array) == len(data.freq_array) // 2
    assert np.all(new.data == 1)

    new = freqbin.gauss_smooth(data, size=2, decimate=False)
    assert len(new.freq_array) == len(data.freq_array)
    assert np.all(new.data == 1)

    flags = np.zeros(data.freq_array.shape, dtype=bool)
    flags[0] = True

    data2 = data.update(flags={"f0": GSFlag(flags, axes=("freq",))})
    new = freqbin.gauss_smooth(data2, size=2, maintain_flags=True, decimate=False)

    assert np.all(new.complete_flags[:, :, :, 0] == 0)
