from edges_analysis.analysis import (
    CalibratedData,
    CombinedData,
    CombinedBinnedData,
    DayAveragedData,
    BinnedData,
    ModelData,
    RawData,
)
from typing import List, Tuple
import dill as pickle
import numpy as np


def test_raw_step(raw_step: Tuple[RawData, RawData]):
    assert raw_step[0].raw_frequencies.shape == (26214,)
    assert raw_step[1].raw_frequencies.shape == (26214,)
    assert np.min(raw_step[0].freq.freq) >= 40
    # Ensure it's pickleable
    pickle.dumps(raw_step[0])

    # ensure plotting functions don't error
    raw_step[0].plot_waterfalls()

    assert (
        len(raw_step[0].lst)
        == len(raw_step[0].gha)
        == len(raw_step[0].raw_time_data)
        == len(raw_step[0].datetimes)
    )


def test_calibrate_step(cal_step: List[CalibratedData]):
    assert cal_step[0].raw_frequencies.shape == (8193,)
    assert cal_step[1].raw_frequencies.shape == (8193,)

    # Ensure it's pickleable
    pickle.dumps(cal_step[0])

    # ensure plotting functions don't error
    cal_step[0].plot_waterfall()
    cal_step[0].plot_time_averaged_spectrum()
    cal_step[0].plot_s11()

    assert (
        len(cal_step[0].lst)
        == len(cal_step[0].gha)
        == len(cal_step[0].raw_time_data)
        == len(cal_step[0].datetimes)
    )


def test_calibrate_step2(cal_step2: List[CalibratedData]):
    assert cal_step2[0].raw_frequencies.shape == (8193,)
    assert cal_step2[1].raw_frequencies.shape == (8193,)

    # Ensure it's pickleable
    pickle.dumps(cal_step2[0])

    # ensure plotting functions don't error
    cal_step2[0].plot_waterfall()
    cal_step2[0].plot_time_averaged_spectrum()
    cal_step2[0].plot_s11()

    assert (
        len(cal_step2[0].lst)
        == len(cal_step2[0].gha)
        == len(cal_step2[0].raw_time_data)
        == len(cal_step2[0].datetimes)
    )


def test_filtering(cal_step: CalibratedData):
    assert cal_step[0].raw_frequencies.shape == (8193,)
    assert cal_step[1].raw_frequencies.shape == (8193,)
    assert not np.all(cal_step[0].weights == cal_step[0].raw_weights)
    assert len(cal_step[0].filters_applied) == 1
    assert "rfi_model_filter" in cal_step[0].filters_applied


def test_model_step(model_step: List[ModelData]):
    assert model_step[0].raw_frequencies.shape == (8193,)
    assert model_step[1].raw_frequencies.shape == (8193,)

    # Ensure it's pickleable
    pickle.dumps(model_step)

    m = model_step[0]
    assert m.model_nterms == 5


def test_combine_step(combo_step: CombinedData, model_step: List[ModelData]):
    assert combo_step.resids.shape[-1] == len(combo_step.raw_frequencies)
    assert combo_step.spectrum.shape == combo_step.resids.shape

    # Ensure it's pickleable
    pickle.dumps(combo_step)

    assert combo_step.gha_edges.max() >= max(
        model_step[0].gha.max(), model_step[1].gha.max()
    )
    assert combo_step.gha_edges.min() <= min(
        model_step[0].gha.min(), model_step[1].gha.min()
    )

    # just run some plotting methods to make sure they don't error...
    combo_step.plot_daily_residuals(freq_resolution=1.0, gha_max=18, gha_min=6)
    combo_step.plot_waterfall(day=292)


def test_bin_aftercombine_step(combo_bin_step: CombinedBinnedData):
    assert combo_bin_step.resids.shape[-1] == len(combo_bin_step.raw_frequencies)
    assert combo_bin_step.spectrum.shape == combo_bin_step.resids.shape

    # Ensure it's pickleable
    pickle.dumps(combo_bin_step)

    # just run some plotting methods to make sure they don't error...
    combo_bin_step.plot_daily_residuals(freq_resolution=1.0, gha_max=18, gha_min=6)
    combo_bin_step.plot_waterfall(day=292)


def test_bin_after_combine2(combo_bin2_step: CombinedBinnedData):
    assert combo_bin2_step.resids.shape[-1] == len(combo_bin2_step.raw_frequencies)
    assert combo_bin2_step.spectrum.shape == combo_bin2_step.resids.shape


def test_day_step(day_step: DayAveragedData):
    assert day_step.resids.shape[-1] == len(day_step.raw_frequencies)
    assert day_step.spectrum.shape == day_step.resids.shape
    # Ensure it's pickleable
    pickle.dumps(day_step)

    f, s, w = day_step.fully_averaged_spectrum()
    assert len(f) == len(s) == len(w) == len(day_step.raw_frequencies)

    day_step.plot_resids()


def test_bin_step(gha_step: BinnedData):
    print(gha_step.resids.shape, gha_step.raw_frequencies.shape)
    assert gha_step.resids.shape[-1] == len(gha_step.raw_frequencies)
    assert gha_step.resids.shape == gha_step.spectrum.shape

    # Ensure it's pickleable
    pickle.dumps(gha_step)

    gha_step.plot_resids()
