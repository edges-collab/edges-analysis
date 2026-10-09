"""Test the freqbin module."""

import numpy as np
import pytest
from astropy import units as un
from pygsdata import GSData, GSFlag

from edges.averaging import freqbin
from edges.testing import create_mock_edges_data


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


def test_freq_bin_size_one_is_identity(gsd_ones: GSData):
    rng = np.random.default_rng(3)
    data = gsd_ones.update(
        data=rng.normal(size=gsd_ones.data.shape),
        nsamples=rng.uniform(1, 3, size=gsd_ones.data.shape),
    )
    new = freqbin.freq_bin(data, bins=1, debias=False)
    np.testing.assert_allclose(new.freqs, data.freqs, rtol=1e-12)
    np.testing.assert_allclose(new.data, data.data, rtol=1e-12)
    np.testing.assert_allclose(new.nsamples, data.nsamples, rtol=1e-12)


@pytest.mark.parametrize("last_edge", [101, 100], ids=["past-last", "on-last"])
def test_freq_bin_conserves_nsamples(gsd_ones: GSData, last_edge: float):
    # Channels are at 50, 52, ..., 100 MHz; interior edges lie exactly on channels.
    # A channel on the final edge is included in the last bin.
    bins = np.array([49, 60, 70, 80, 90, last_edge]) * un.MHz
    new = freqbin.freq_bin(gsd_ones, bins=bins, debias=False)
    np.testing.assert_allclose(
        new.nsamples.sum(axis=-1), gsd_ones.nsamples.sum(axis=-1), rtol=1e-12
    )


def test_freq_bin_edge_channel_in_upper_bin(gsd_ones: GSData):
    # The channel at 60 MHz lies on an interior edge: it belongs to the upper bin.
    bins = np.array([49, 60, 101]) * un.MHz
    new = freqbin.freq_bin(gsd_ones, bins=bins, debias=False)
    nlow = np.sum(gsd_ones.freqs < 60 * un.MHz)
    np.testing.assert_allclose(new.nsamples[..., 0], nlow, rtol=1e-12)
    np.testing.assert_allclose(
        new.freqs[0].to_value("MHz"),
        np.mean(gsd_ones.freqs[:nlow].to_value("MHz")),
        rtol=1e-12,
    )


def test_freq_bin_int_bins_unchanged(gsd_ones: GSData):
    # Integer bins have edges between channels, so each bin has exactly `bins`
    # channels.
    new = freqbin.freq_bin(gsd_ones, bins=2, debias=False)
    np.testing.assert_allclose(new.nsamples, 2, rtol=1e-12)


@pytest.mark.parametrize("use_nsamples", [False, True])
def test_gauss_smooth_flag_threshold_nsamples_scale_invariant(
    gsd_ones: GSData, use_nsamples: bool
):
    out1 = freqbin.gauss_smooth(
        gsd_ones, size=2, flag_threshold=0.25, use_nsamples=use_nsamples
    )
    out10 = freqbin.gauss_smooth(
        gsd_ones.update(nsamples=10 * gsd_ones.nsamples),
        size=2,
        flag_threshold=0.25,
        use_nsamples=use_nsamples,
    )
    assert np.any(out1.nsamples > 0)
    np.testing.assert_array_equal(out10.nsamples == 0, out1.nsamples == 0)


def test_gauss_smooth_flag_threshold_flags_sparse_windows(gsd_ones: GSData):
    # Flag all but every 8th channel: each smoothed window then contains only a small
    # fraction of the unflagged weight, so a high threshold flags everything, while a
    # zero threshold flags nothing.
    nsamples = np.zeros_like(gsd_ones.nsamples)
    nsamples[..., ::8] = 3.0
    data = gsd_ones.update(nsamples=nsamples)
    out = freqbin.gauss_smooth(data, size=4, decimate=False, flag_threshold=0.9)
    assert np.all(out.nsamples[..., 8:-8] == 0)
    out = freqbin.gauss_smooth(data, size=4, decimate=False, flag_threshold=0)
    assert np.all(out.nsamples > 0)


@pytest.mark.xfail(
    strict=True,
    reason=(
        "gauss_smooth reports nsamples as sum(k*n) rather than the effective "
        "inverse variance (sum k)^2/sum(k^2/n), which is ~sqrt(2) larger; fix pending "
        "(result-changing)"
    ),
)
def test_gauss_smooth_nsamples_is_inverse_variance():
    template = create_mock_edges_data(ntime=100, flow=50 * un.MHz, fhigh=52 * un.MHz)
    rng = np.random.default_rng(5)
    # Unit-variance white noise with nsamples=1, so 1/var(output) is the true nsamples.
    data = template.update(
        data=1 + rng.normal(size=template.data.shape),
        nsamples=np.ones(template.data.shape),
        auxiliary_measurements=None,
    )
    out = freqbin.gauss_smooth(data, size=8)
    inner = slice(5, -5)  # avoid the band edges, where the kernel is truncated
    reported = np.mean(out.nsamples[0, 0, :, inner])
    empirical = 1 / np.var(out.data[0, 0, :, inner])
    assert reported == pytest.approx(empirical, rel=0.1)


def test_gauss_smooth_maintains_nan_flags_with_any_nsamples(gsd_ones: GSData):
    """NaN channels are flagged whatever their nsamples (operator precedence bug)."""
    data = gsd_ones.data.copy()
    data[..., 10] = np.nan
    new = gsd_ones.update(data=data, nsamples=2 * np.ones_like(data))

    out = freqbin.gauss_smooth(new, size=1, decimate=False, maintain_flags=1)
    assert np.all(out.nsamples[..., 10] == 0)
    assert np.all(out.nsamples[..., 9] > 0)
    assert np.all(out.nsamples[..., 11] > 0)
