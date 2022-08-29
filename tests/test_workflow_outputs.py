import pytest

import dill as pickle
import numpy as np
from astropy import units as u
from typing import List, Tuple

from edges_analysis.gsdata import GSData


@pytest.mark.parametrize(
    "steps, nfreqs",
    [
        ("raw_step", 26214),
        ("cal_step", 8193),
        ("cal_step_nobeam", 8193),
        ("cal_step_s11format", 8193),
        ("model_step", 8193),
        ("lstbin_step", 8193),
        ("lstavg_step", 8193),
        ("lstbin24_step", 8193),
        ("final_step", 1024),
    ],
)
def test_step_basic(request, steps: Tuple[GSData, GSData], nfreqs: int):
    steps = request.getfixturevalue(steps)
    for step in steps:
        assert step.freq_array.shape == (nfreqs,)
        assert np.min(step.freq_array) >= 40 * u.MHz

        # Ensure it's pickleable
        pickle.dumps(step)


def test_model_step(model_step: List[GSData]):
    m = model_step[0]
    assert m.data_model.nparams == 5


def test_lstavg_step(
    lstavg_step: tuple[GSData, GSData], lstbin_step: tuple[GSData, GSData]
):
    avg = lstavg_step[0]
    assert avg.data_model.nparams == 5
    assert avg.in_lst

    binned = lstbin_step[0]
    assert avg.ntimes == binned.ntimes
