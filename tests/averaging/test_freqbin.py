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


@pytest.mark.xfail(
    strict=True,
    reason=(
        "freq_bin includes both bin edges, so a channel lying exactly on an "
        "edge is counted in two bins and total nsamples is not conserved; fix pending "
        "(result-changing)"
    ),
)
def test_freq_bin_conserves_nsamples(gsd_ones: GSData):
    # Channels are at 50, 52, ..., 100 MHz; interior edges lie exactly on channels.
    bins = np.array([49, 60, 70, 80, 90, 101]) * un.MHz
    new = freqbin.freq_bin(gsd_ones, bins=bins, debias=False)
    np.testing.assert_allclose(
        new.nsamples.sum(axis=-1), gsd_ones.nsamples.sum(axis=-1), rtol=1e-12
    )


@pytest.mark.xfail(
    strict=True,
    reason=(
        "gauss_smooth compares nsamples/max(nsamples) against "
        "size*flag_threshold, mixing units, so scaling nsamples changes which "
        "channels are flagged (nsamples=10 flags everything); fix pending "
        "(result-changing)"
    ),
)
def test_gauss_smooth_flag_threshold_nsamples_scale_invariant(gsd_ones: GSData):
    out1 = freqbin.gauss_smooth(gsd_ones, size=2, flag_threshold=0.25)
    out10 = freqbin.gauss_smooth(
        gsd_ones.update(nsamples=10 * gsd_ones.nsamples), size=2, flag_threshold=0.25
    )
    assert np.any(out1.nsamples > 0)
    np.testing.assert_array_equal(out10.nsamples == 0, out1.nsamples == 0)


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
