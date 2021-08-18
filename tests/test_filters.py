import numpy as np
from edges_analysis.analysis import filters
from edges_cal import modelling as mdl


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
