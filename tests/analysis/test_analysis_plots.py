"""Test the plots module."""

import numpy as np
import pytest
from pygsdata import GSData

from edges.analysis import plots
from edges.analysis.datamodel import add_model
from edges.modeling import LinLog


def test_plot_time_average_bad_attribute(mock):
    with pytest.raises(ValueError, match="Cannot use attribute"):
        plots.plot_time_average(mock, attribute="flags")


@pytest.mark.parametrize(
    "step",
    [
        "mock",
        "mock_power",
        "mock_with_model",
    ],
)
def test_plot_time_average(step, request):
    """Test that plotting a time average doesn't crash."""
    step = request.getfixturevalue(step)
    plots.plot_time_average(step, lst_min=6.0, lst_max=18.0)


@pytest.mark.parametrize(
    "step",
    [
        "mock_season",
        "mock_season_dicke",
        "mock_season_modelled",
    ],
)
def test_plot_daily_residuals(step, request):
    """Test that plotting daily residuals doesn't crash."""
    step: GSData = request.getfixturevalue(step)
    if step[0].residuals is None:
        with pytest.raises(ValueError, match="If data has no model, must provide one!"):
            plots.plot_daily_residuals(step)
        plots.plot_daily_residuals(step, model=LinLog(n_terms=5))
    else:
        print(step[0].data)
        plots.plot_daily_residuals(step)


def test_plot_daily_residuals_forwards_load_and_pol(mock_season, monkeypatch):
    """Regression test for ANA-13: load/pol were not passed to plot_time_average."""
    # Three-load (power) data, so that a non-zero load index can be chosen.
    modelled = [add_model(m, model=LinLog(n_terms=2)) for m in mock_season]
    calls = []
    original = plots.plot_time_average

    def spy(*args, **kwargs):
        calls.append(kwargs)
        return original(*args, **kwargs)

    monkeypatch.setattr(plots, "plot_time_average", spy)
    ax = plots.plot_daily_residuals(modelled, load=1, pol=0)

    assert len(calls) == len(modelled)
    assert all(c["load"] == 1 and c["pol"] == 0 for c in calls)

    # The plotted line is the time-averaged residual of the requested load.
    _, avg = original(modelled[0], attribute="residuals", load=1)
    np.testing.assert_allclose(
        ax.lines[0].get_ydata(), avg.residuals[1, 0, 0], rtol=1e-12, atol=0
    )
