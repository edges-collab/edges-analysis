"""Test the freqbin module."""

import numpy as np
from pygsdata import GSFlag

from edges.averaging import freqbin


def test_freq_bin_direct(gsd_ones):
    new = freqbin.freq_bin(gsd_ones, bins=2, debias=False)
    assert len(new.freqs) == len(gsd_ones.freqs) // 2


def test_freqbin_gaussian(gsd_ones):
    data = gsd_ones

    new = freqbin.gauss_smooth(data, size=2)
    assert len(new.freqs) == len(data.freqs) // 2
    assert np.all(new.data == 1)

    new = freqbin.gauss_smooth(data, size=2, decimate=False)
    assert len(new.freqs) == len(data.freqs)
    assert np.all(new.data == 1)

    flags = np.zeros(data.freqs.shape, dtype=bool)
    flags[0] = True

    data2 = data.update(flags={"f0": GSFlag(flags, axes=("freq",))})
    new = freqbin.gauss_smooth(data2, size=2, maintain_flags=True, decimate=False)

    assert np.all(new.complete_flags[:, :, :, 0] == 0)
