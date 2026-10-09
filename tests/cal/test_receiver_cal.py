import inspect
from types import SimpleNamespace

import numpy as np
import pytest
from astropy import units as un

from edges.cal import Calibrator, receiver_cal
from edges.cal.receiver_cal import perform_term_sweep


def test_term_sweep(calobs):
    calobs_opt = perform_term_sweep(
        calobs,
        max_cterms=6,
        max_wterms=8,
    )

    assert isinstance(calobs_opt, Calibrator)


def test_term_sweep_return_annotation():
    assert inspect.signature(perform_term_sweep).return_annotation is Calibrator


def _mock_sweep(monkeypatch, rms_grid: dict[tuple[int, int], float]) -> list:
    """Make perform_term_sweep fast: each 'calibrator' is just its (c, w) pair.

    The RMS of the (c, w) calibrator is ``rms_grid[(c, w)]``. A very large number
    of frequencies is used so that the degrees-of-freedom normalisation has a
    negligible effect on the ordering of the RMS values.
    """
    calls = []

    def fake_calibration(calobs, cterms, wterms, **kwargs):
        calls.append((cterms, wterms))
        return (cterms, wterms)

    monkeypatch.setattr(
        receiver_cal, "get_noise_wave_calibration_iterative", fake_calibration
    )
    calobs = SimpleNamespace(
        freqs=np.zeros(10**6),
        loads={"ambient": None},
        get_rms=lambda cal: {"ambient": rms_grid[cal] * un.K},
    )
    return calobs, calls


def test_term_sweep_all_nan_rms(monkeypatch):
    """If no RMS is finite, a clear error is raised (not UnboundLocalError)."""
    grid = {(c, w): np.nan for c in range(4, 8) for w in range(4, 8)}
    calobs, _ = _mock_sweep(monkeypatch, grid)
    with pytest.raises(RuntimeError, match="No finite RMS"):
        perform_term_sweep(calobs, max_cterms=7, max_wterms=7)


def test_term_sweep_tries_max_terms(monkeypatch):
    # RMS strictly decreasing in both c and w: the sweep must go all the way.
    grid = {(c, w): 100.0 - c - w for c in range(4, 8) for w in range(4, 8)}
    calobs, calls = _mock_sweep(monkeypatch, grid)

    best = perform_term_sweep(calobs, max_cterms=6, max_wterms=6)
    assert (6, 6) in calls
    assert best == (6, 6)


def test_term_sweep_returns_argmin(monkeypatch):
    # For each c, the RMS decreases with w up to w=6 and is worse at w=7. The best
    # RMS per c decreases up to c=6 and is worse at c=7. The w=4 column is flat,
    # so comparing across c at a fixed w (rather than at each c's best w) is wrong.
    grid = {}
    for c, row in {
        4: [50, 30, 20, 25],
        5: [50, 25, 15, 18],
        6: [50, 10, 5, 8],
        7: [50, 40, 30, 35],
    }.items():
        for w, val in zip(range(4, 8), row, strict=True):
            grid[c, w] = float(val)

    calobs, _ = _mock_sweep(monkeypatch, grid)
    best = perform_term_sweep(calobs, max_cterms=7, max_wterms=7)
    assert best == min(grid, key=grid.get)


def test_term_sweep_metric_counts_all_parameters(monkeypatch):
    """The metric is sqrt(chi^2 / dof) with dof = N*Nsrc - 2c - 3w.

    With few frequencies, the number of parameters matters: here (5, 4) has a
    slightly lower raw RMS than (4, 4), but not by enough to pay for its two extra
    scale/offset parameters, so (4, 4) wins.
    """
    nfreq = 20
    grid = {(4, 4): 1.0, (4, 5): 2.0, (5, 4): 0.99, (5, 5): 2.0}

    def fake_calibration(calobs, cterms, wterms, **kwargs):
        return (cterms, wterms)

    monkeypatch.setattr(
        receiver_cal, "get_noise_wave_calibration_iterative", fake_calibration
    )
    calobs = SimpleNamespace(
        freqs=np.zeros(nfreq),
        loads={"ambient": None, "hot_load": None},
        get_rms=lambda cal: {
            "ambient": grid[cal] * un.K,
            "hot_load": grid[cal] * un.K,
        },
    )
    best = perform_term_sweep(calobs, max_cterms=5, max_wterms=5)

    def metric(c, w):
        return np.sqrt(2 * nfreq * grid[c, w] ** 2 / (2 * nfreq - 2 * c - 3 * w))

    assert metric(4, 4) < metric(5, 4)
    assert best == (4, 4)
    assert receiver_cal._sweep_metric(
        calobs.get_rms((5, 4)), nfreq, 5, 4
    ) == pytest.approx(metric(5, 4), rel=1e-12)
