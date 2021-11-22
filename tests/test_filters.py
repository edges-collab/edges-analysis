from datetime import datetime
import numpy as np
from edges_analysis.analysis import filters
from edges_cal import modelling as mdl
import pytest


def test_aux_filter():
    gha = np.linspace(0, 24, 48, endpoint=False)

    sun_el = np.zeros(48)
    sun_el[0] = 60

    moon_el = np.zeros(48)
    moon_el[1] = 60

    humidity = np.zeros(48)
    humidity[2] = 200

    receiver_temp = np.ones(48)
    receiver_temp[3] = 0
    receiver_temp[4] = 200

    out_flags = filters.time_filter_auxiliary(
        gha,
        sun_el,
        moon_el,
        humidity,
        receiver_temp,
        gha_range=(0, 12),
        sun_el_max=50,
        moon_el_max=50,
        amb_hum_max=100,
        max_receiver_temp=100,
        adcmax=0.4,
    )

    assert np.all(out_flags[:4])
    assert np.all(gha[~out_flags] <= 12)


def test_tp_filter(cal_step):
    out_flags = filters.total_power_filter(
        data=cal_step,
        in_place=False,
        metric_model=mdl.FourierDay(n_terms=3),
        std_model=mdl.FourierDay(n_terms=3),
    )
    assert len(out_flags) == 2


def test_rms_filter(cal_step):
    out_flags = filters.rms_filter(
        data=cal_step,
        in_place=False,
        metric_model=mdl.FourierDay(n_terms=3),
        std_model=mdl.FourierDay(n_terms=3),
    )
    assert len(out_flags) == 2


def test_negpower_filter(cal_step):
    out_flags = filters.negative_power_filter(data=cal_step)
    assert len(out_flags) == 2


def test_peak_power_filter(cal_step):
    out_flags = filters.peak_power_filter(data=cal_step)
    assert len(out_flags) == 2
    assert out_flags[0].shape == cal_step[0].spectrum.shape

    with pytest.raises(ValueError):
        filters.peak_power_filter(data=cal_step, peak_freq_range=(60, 60))

    with pytest.raises(ValueError):
        filters.peak_power_filter(data=cal_step, mean_freq_range=(61, 60))


def test_peak_orbcomm_filter(cal_step):
    out_flags = filters.peak_orbcomm_filter(data=cal_step)
    assert len(out_flags) == 2
    assert out_flags[0].shape == cal_step[0].spectrum.shape


def test_150mhz_filter(cal_step):
    out_flags = filters.filter_150mhz(data=cal_step, threshold=1)
    assert len(out_flags) == 2


def test_rmsf(cal_step):
    out_flags = filters.rmsf_filter(data=cal_step, threshold=200)
    assert len(out_flags) == 2


def test_max_fm_filter(cal_step):
    out_flags = filters.maxfm_filter(data=cal_step, threshold=200)
    assert len(out_flags) == 2


def test_percent_power_filter(raw_step):
    out_flags = filters.power_percent_filter(data=raw_step)
    assert len(out_flags) == 2


def test_rfi_filter(raw_step):
    out_flags = filters.rfi_model_filter(data=raw_step, freq_range=(40, 100))
    assert len(out_flags) == 2


def test_day_filter(combo_step):
    out_flags = filters.day_filter(data=[combo_step], dates=[(2016, 292)])
    assert out_flags[0].ndim == 3
    assert np.all(out_flags[0][0])

    out_flags2 = filters.day_filter(data=[combo_step], dates=[(2016, 292, 18)])
    np.testing.assert_equal(out_flags, out_flags2)

    out_flags3 = filters.day_filter(data=[combo_step], dates=[datetime(2016, 10, 18)])
    np.testing.assert_equal(out_flags, out_flags3)

    with pytest.raises(ValueError):
        filters.day_filter(data=[combo_step], dates=["heythere"])
