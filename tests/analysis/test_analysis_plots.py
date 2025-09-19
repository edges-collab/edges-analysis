"""Test the plots module."""

import pytest
from pygsdata import GSData

from edges.analysis import plots
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
