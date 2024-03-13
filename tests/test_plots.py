"""Test the plots module."""

import pytest
from edges_analysis import plots
from edges_cal.modelling import LinLog
from pygsdata import GSData


@pytest.mark.parametrize(
    "step",
    ["mock", "mock_power", "mock_with_model", "mock_lstbinned"],
)
@pytest.mark.parametrize("title", [None, "a title"])
def test_plot_waterfall(step, title, request):
    """Test that plotting a waterfall doesn't crash."""
    astep = request.getfixturevalue(step)
    plots.plot_waterfall(
        astep,
        attribute="residuals" if step == "mock_with_model" else "data",
        title=title,
    )


def test_plot_waterfall_bad_attribute(mock):
    with pytest.raises(ValueError, match="Cannot use attribute"):
        plots.plot_waterfall(mock, attribute="flags")


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
        plots.plot_daily_residuals(step)
