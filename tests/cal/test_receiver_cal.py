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


@pytest.mark.xfail(
    strict=True,
    reason=(
        "perform_term_sweep uses range(min, max), so max_cterms/max_wterms are never "
        "tried; fix pending (result-changing)"
    ),
)
def test_term_sweep_tries_max_terms(monkeypatch):
    # RMS strictly decreasing in both c and w: the sweep must go all the way.
    grid = {(c, w): 100.0 - c - w for c in range(4, 8) for w in range(4, 8)}
    calobs, calls = _mock_sweep(monkeypatch, grid)

    best = perform_term_sweep(calobs, max_cterms=6, max_wterms=6)
    assert (6, 6) in calls
    assert best == (6, 6)


@pytest.mark.xfail(
    strict=True,
    reason=(
        "perform_term_sweep leaves winner[i]=0 when the inner wterms loop never "
        "breaks, so the cterms loop stops early and misses the minimum RMS; fix "
        "pending (result-changing)"
    ),
)
def test_term_sweep_returns_argmin(monkeypatch):
    # For each c, the RMS decreases with w up to w=6 and is worse at w=7. The best
    # RMS per c decreases up to c=6 and is worse at c=7. The w=4 column is flat,
    # which is where the current implementation (wrongly) compares across c.
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
