"""Test frequency range classes."""

from astropy import units as u

from edges.frequencies import edges_raw_freqs


def test_edges_freq():
    freq = edges_raw_freqs()
    assert freq.min() == 0.0 * u.MHz
    assert freq.max() < 200.0 * u.MHz
    assert freq.size == 32768


def test_edges_freq_limited():
    freq = edges_raw_freqs(f_low=50.0 * u.MHz, f_high=100.0 * u.MHz)
    assert freq.size == 8193
    assert freq.min() == 50.0 * u.MHz
    assert freq.max() == 100.0 * u.MHz
