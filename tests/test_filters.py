from __future__ import annotations

import pytest

import numpy as np
from astropy import units as un
from edges_cal import modelling as mdl

from edges_analysis import GSData
from edges_analysis.filters import filters, lst_model


def test_aux_filter(raw_step):
    run_filter_check(
        raw_step[0],
        filters.aux_filter,
        maxima={"ambient_hum": 100, "receiver_temp": 100},
    )


def test_sun_filter(raw_step):
    run_filter_check(
        raw_step[0],
        filters.sun_filter,
        elevation_range=(-np.inf, 40),
    )


def test_moon_filter(raw_step):
    run_filter_check(
        raw_step[0],
        filters.moon_filter,
        elevation_range=(-np.inf, 40),
    )


def test_tp_filter(cal_step):
    run_filter_check(
        cal_step,
        lst_model.total_power_filter,
        write=False,
        metric_model=mdl.FourierDay(n_terms=3),
        std_model=mdl.FourierDay(n_terms=3),
    )


def test_rms_filter(cal_step):
    run_filter_check(
        cal_step,
        lst_model.rms_filter,
        write=False,
        metric_model=mdl.FourierDay(n_terms=3),
        std_model=mdl.FourierDay(n_terms=3),
    )


def test_peak_power_filter(cal_step):
    run_filter_check(cal_step[0], filters.peak_power_filter)

    with pytest.raises(ValueError):
        filters.peak_power_filter(data=cal_step[0], peak_freq_range=(60, 60))

    with pytest.raises(ValueError):
        filters.peak_power_filter(data=cal_step[0], mean_freq_range=(61, 60))


def run_filter_check(data: GSData, fnc: callable, **kwargs):
    new_data = fnc(data, **kwargs)
    if isinstance(new_data, GSData):
        assert new_data.data.shape == data.data.shape
        assert fnc.__name__ in new_data.flags
        assert len(new_data.flags) - len(data.flags) == 1
    else:
        for nd, d in zip(new_data, data):
            assert nd.data.shape == d.data.shape
            assert fnc.__name__ in nd.flags
            assert len(nd.flags) - len(d.flags) == 1


def test_peak_orbcomm_filter(cal_step):
    run_filter_check(cal_step[0], filters.peak_orbcomm_filter)


def test_150mhz_filter(cal_step):
    run_filter_check(cal_step[0], filters.filter_150mhz, threshold=100)


def test_rmsf(cal_step):
    run_filter_check(cal_step[0], filters.rmsf_filter, threshold=100)


def test_max_fm_filter(cal_step):
    run_filter_check(cal_step[0], filters.maxfm_filter, threshold=200)


def test_percent_power_filter(raw_step):
    run_filter_check(raw_step[0], filters.power_percent_filter)


def test_rfi_filter(raw_step):
    run_filter_check(raw_step[0], filters.rfi_model_filter, freq_range=(40, 100))


def test_rmsf_filter(gsd_ones: GSData):
    with pytest.raises(ValueError, match="Unsupported data_unit for rmsf_filter."):
        filters.rmsf_filter(gsd_ones.update(data_unit="power"), threshold=100)

    # Let data be perfect:
    pl = np.outer(
        1 + np.random.normal(scale=0.1, size=gsd_ones.ntimes * gsd_ones.npols),
        (gsd_ones.freq_array / (75 * un.MHz)) ** -2.5,
    )
    pl.shape = gsd_ones.data.shape

    data = gsd_ones.update(data=pl, data_unit="uncalibrated_temp")

    print(gsd_ones.data.shape, data.data.shape)
    new = filters.rmsf_filter(data, threshold=1)

    data = gsd_ones.update(data=(pl - 300) / 1000, data_unit="uncalibrated")
    new2 = filters.rmsf_filter(data, threshold=1, tcal=300, tload=1000)

    assert np.all(new.complete_flags == new2.complete_flags)
