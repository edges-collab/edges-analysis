from edges_analysis.analysis import (
    CalibratedData,
    CombinedData,
    DayAveragedData,
    BinnedData,
    FilteredData,
    ModelData,
)
from typing import List
import dill as pickle


def test_calibrate_step(cal_step: List[CalibratedData]):
    assert cal_step[0].raw_frequencies.shape == (8192,)
    assert cal_step[1].raw_frequencies.shape == (8192,)

    # Ensure it's pickleable
    pickle.dumps(cal_step[0])


def test_filter_step(filter_step: List[FilteredData]):
    assert filter_step[0].raw_frequencies.shape == (8192,)
    assert filter_step[1].raw_frequencies.shape == (8192,)

    # Ensure it's pickleable
    pickle.dumps(filter_step)


def test_model_step(model_step: List[ModelData]):
    assert model_step[0].raw_frequencies.shape == (8192,)
    assert model_step[1].raw_frequencies.shape == (8192,)

    # Ensure it's pickleable
    pickle.dumps(model_step)


def test_combine_step(combo_step: CombinedData):
    assert combo_step.resids.shape[-1] == len(combo_step.raw_frequencies)
    assert combo_step.spectrum.shape == combo_step.resids.shape

    # Ensure it's pickleable
    pickle.dumps(combo_step)

    # just run some plotting methods to make sure they don't error...
    combo_step.plot_daily_residuals(freq_resolution=1.0, gha_max=18, gha_min=6)


def test_day_step(day_step: DayAveragedData):
    assert day_step.resids.shape[-1] == len(day_step.raw_frequencies)
    assert day_step.spectrum.shape == day_step.resids.shape
    # Ensure it's pickleable
    pickle.dumps(day_step)


def test_bin_step(gha_step: BinnedData):
    print(gha_step.resids.shape, gha_step.raw_frequencies.shape)
    assert gha_step.resids.shape[-1] == len(gha_step.raw_frequencies)
    assert gha_step.resids.shape == gha_step.spectrum.shape

    # Ensure it's pickleable
    pickle.dumps(gha_step)
