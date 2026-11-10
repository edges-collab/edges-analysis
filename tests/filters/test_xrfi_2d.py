"""Tests of the 2D (LST-frequency) sliding-window xRFI."""

import numpy as np
import pytest

from edges import modeling as mdl
from edges.filters import xrfi
from edges.filters.runners import run_xrfi

NFREQ = 400
NLST = 8


@pytest.fixture(scope="module")
def freqs():
    return np.linspace(50, 100, NFREQ)


def _lst_sky(freqs, n_lst=NLST, noise=0.01, seed=0):
    """A power-law sky whose amplitude changes quadratically with LST, plus noise."""
    rng = np.random.default_rng(seed)
    x = np.linspace(0, 1, n_lst)
    amplitude = 1750 * (1 + 0.3 * x - 0.2 * x**2)
    sky = amplitude[:, None] * (freqs / 75.0) ** -2.5
    return sky + rng.normal(scale=noise, size=sky.shape)


def _inject(sky, n_rfi=12, seed=1, strength=(0.5, 5.0)):
    rng = np.random.default_rng(seed)
    rows = rng.integers(0, sky.shape[0], size=n_rfi)
    cols = rng.choice(np.arange(20, sky.shape[1] - 20), size=n_rfi, replace=False)
    out = sky.copy()
    out[rows, cols] += rng.uniform(*strength, size=n_rfi)
    truth = np.zeros(sky.shape, dtype=bool)
    truth[rows, cols] = True
    return out, truth


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_single_lst_bin_matches_1d(freqs, seed):
    """With one LST bin and no time extent, the 2D flagger is the 1D one."""
    rng = np.random.default_rng(seed)
    data = 1750 * (freqs / 75.0) ** -2.5 + rng.normal(scale=0.01, size=freqs.size)
    spikes = rng.choice(np.arange(10, NFREQ - 10), size=15, replace=False)
    data[spikes] += rng.uniform(0.2, 3, size=spikes.size)
    kwargs = {
        "freqs": freqs,
        "model": mdl.EdgesPoly(n_terms=6),
        "window_frac": 16,
        "threshold": 3.0,
        "max_iter": 20,
        "reflag_thresh": 1.0,
        "watershed": {1.0: 1, 10.0: 2},
        "fit_kwargs": {"method": "alan-qrd"},
    }
    flags_1d, info_1d = xrfi.xrfi_iterative_sliding_window(data.copy(), **kwargs)
    flags_2d, info_2d = xrfi.xrfi_iterative_sliding_window_2d(
        data[None].copy(), lst_terms=1, time_half_width=0, **kwargs
    )
    np.testing.assert_array_equal(flags_2d[0], flags_1d)
    assert info_2d.n_iters == info_1d.n_iters


def test_one_term_per_lst_bin_matches_1d_per_bin(freqs):
    """With lst_terms=None and no time extent, each LST bin is flagged as in 1D."""
    data, _ = _inject(_lst_sky(freqs), strength=(0.2, 3.0))
    kwargs = {
        "freqs": freqs,
        "model": mdl.EdgesPoly(n_terms=6),
        "window_frac": 16,
        "threshold": 3.0,
        "max_iter": 20,
        "reflag_thresh": 1.0,
        "watershed": {1.0: 1, 10.0: 2},
        "fit_kwargs": {"method": "alan-qrd"},
    }
    flags_1d = np.array([
        xrfi.xrfi_iterative_sliding_window(row.copy(), **kwargs)[0] for row in data
    ])
    flags_2d, _ = xrfi.xrfi_iterative_sliding_window_2d(
        data.copy(), lst_terms=None, time_half_width=0, **kwargs
    )
    assert flags_1d.any()
    np.testing.assert_array_equal(flags_2d, flags_1d)


def test_whitening_keeps_lightly_filled_lst_bins(freqs):
    """A bin with few samples is noisier; whitening stops it being flagged wholesale."""
    nsamples = np.full((NLST, NFREQ), 75.0)
    nsamples[-1] = 4.0
    rng = np.random.default_rng(3)
    data = _lst_sky(freqs, noise=0.0) + rng.normal(size=nsamples.shape) * 0.1 / np.sqrt(
        nsamples
    )
    data, truth = _inject(data, strength=(0.3, 3.0))
    kwargs = {
        "freqs": freqs,
        "model": mdl.EdgesPoly(n_terms=6),
        "weights": nsamples,
        "lst_terms": None,
        "threshold": 3.5,
        "max_iter": 20,
        "reflag_thresh": 1.0,
    }
    whitened, _ = xrfi.xrfi_iterative_sliding_window_2d(data.copy(), **kwargs)
    raw, _ = xrfi.xrfi_iterative_sliding_window_2d(data.copy(), whiten=False, **kwargs)
    assert np.all(whitened[truth])
    assert np.mean(whitened[~truth]) < 0.01
    assert np.mean(raw[-1]) > 0.2


def test_finds_rfi_in_lst_frequency(freqs):
    """The joint model follows a sky that changes with LST, so only RFI is flagged."""
    data, truth = _inject(_lst_sky(freqs))
    flags, info = xrfi.xrfi_iterative_sliding_window_2d(
        data,
        freqs=freqs,
        model=mdl.EdgesPoly(n_terms=6),
        lst_terms=3,
        threshold=3.5,
        max_iter=20,
        reflag_thresh=1.0,
    )
    assert flags.shape == data.shape
    np.testing.assert_array_equal(flags, truth)
    assert info.n_flags_changed[-1] <= 0


def test_empty_lst_bins_are_left_alone(freqs):
    """LST bins with no weight (and NaN data) are skipped and stay flagged."""
    data, truth = _inject(_lst_sky(freqs))
    weights = np.ones_like(data)
    empty = [0, 5]
    data[empty] = np.nan
    weights[empty] = 0
    truth[empty] = True
    flags, _ = xrfi.xrfi_iterative_sliding_window_2d(
        data,
        freqs=freqs,
        model=mdl.EdgesPoly(n_terms=6),
        weights=weights,
        lst_terms=3,
        threshold=3.5,
        max_iter=20,
        reflag_thresh=1.0,
    )
    np.testing.assert_array_equal(flags, truth)


def test_watershed_is_along_frequency_only(freqs):
    data = _lst_sky(freqs)
    data[3, 200] += 2.0
    flags, _ = xrfi.xrfi_iterative_sliding_window_2d(
        data,
        freqs=freqs,
        model=mdl.EdgesPoly(n_terms=6),
        lst_terms=3,
        threshold=3.5,
        max_iter=20,
        reflag_thresh=1.0,
        watershed={1.0: 2},
    )
    assert flags[3, 198:203].all()
    assert not flags[np.arange(NLST) != 3].any()


def test_run_xrfi_flags_each_day_separately(freqs):
    days = np.stack([_inject(_lst_sky(freqs, seed=s), seed=s + 10)[0] for s in (0, 1)])
    kwargs = {
        "model": mdl.EdgesPoly(n_terms=6),
        "lst_terms": 3,
        "threshold": 3.5,
        "max_iter": 20,
        "reflag_thresh": 1.0,
    }
    flags = run_xrfi(
        method="iterative_sliding_window_2d",
        spectrum=days,
        freqs=freqs,
        n_threads=1,
        **kwargs,
    )
    for day, day_flags in zip(days, flags, strict=True):
        expected, _ = xrfi.xrfi_iterative_sliding_window_2d(day, freqs=freqs, **kwargs)
        np.testing.assert_array_equal(day_flags, expected)


def test_rejects_1d_input(freqs):
    with pytest.raises(ValueError, match="2D"):
        xrfi.xrfi_iterative_sliding_window_2d(
            np.ones(NFREQ), freqs=freqs, model=mdl.EdgesPoly(n_terms=3)
        )
