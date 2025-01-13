"""Test the lstmodel module."""

import numpy as np
from edges_cal import modelling as mdl
from pygsdata import GSData

from edges_analysis.filters import lst_model


def run_filter_check(data: GSData, fnc: callable, **kwargs):
    new_data = fnc(data, **kwargs)
    if isinstance(new_data, GSData):
        assert new_data.data.shape == data.data.shape
        assert fnc.__name__ in new_data.flags
        assert len(new_data.flags) - len(data.flags) == 1
    else:
        print(len(new_data))
        print(len(data))
        for nd, d in zip(new_data, data, strict=False):
            assert nd.data.shape == d.data.shape
            assert fnc.__name__ in nd.flags
            assert len(nd.flags) - len(d.flags) == 1


def test_tp_filter(mock_with_model):
    run_filter_check(
        [mock_with_model],
        lst_model.total_power_filter,
        write=False,
        metric_model=mdl.FourierDay(n_terms=3),
        std_model=mdl.FourierDay(n_terms=3),
        init_flag_threshold=np.inf,
    )


def test_basic(mock):
    run_filter_check(
        [mock],
        lst_model.rms_filter,
        metric_model=mdl.FourierDay(n_terms=3),
        std_model=mdl.FourierDay(n_terms=3),
    )
