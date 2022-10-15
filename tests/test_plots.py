import pytest

from edges_cal.modelling import LinLog

from edges_analysis import plots
from edges_analysis.gsdata import GSData


@pytest.mark.parametrize(
    "step",
    [
        "raw_step",
        "cal_step",
        "model_step",
        "lstbin_step",
        "lstavg_step",
        "lstbin24_step",
        "final_step",
    ],
)
def test_plot_waterfall(step, request):
    """Test that plotting a waterfall doesn't crash."""
    step = request.getfixturevalue(step)
    plots.plot_waterfall(step[0])


@pytest.mark.parametrize(
    "step",
    [
        "raw_step",
        "cal_step",
        "model_step",
        "lstbin_step",
        "lstavg_step",
        "lstbin24_step",
        "final_step",
    ],
)
def test_plot_time_average(step, request):
    """Test that plotting a time average doesn't crash."""
    step = request.getfixturevalue(step)
    plots.plot_time_average(step[0])


@pytest.mark.parametrize("step", ["raw_step", "cal_step", "model_step", "lstbin_step"])
def test_plot_daily_residuals(step, request):
    """Test that plotting daily residuals doesn't crash."""
    step: GSData = request.getfixturevalue(step)
    if step[0].data_model is None:
        with pytest.raises(ValueError, match="If data has no model, must provide one!"):
            plots.plot_daily_residuals(step)
        plots.plot_daily_residuals(step, model=LinLog(n_terms=5))
    else:
        plots.plot_daily_residuals(step)
