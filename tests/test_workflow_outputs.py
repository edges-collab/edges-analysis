from typing import List, Tuple

import dill as pickle
import numpy as np
from astropy import units as u
import pytest
from edges_analysis.gsdata import GSData

@pytest.mark.parametrize(
    'steps, nfreqs', [
        ('raw_step', 26214),
        ('cal_step', 8193),
        ('cal_step_nobeam', 8193),
        ('cal_step_s11format', 8193),
        ('model_step', 8193),
        ('lstbin_step', 8193),
        ('lstavg_step', 8193),
        ('lstbin24_step', 8193),
        ('final_step', 1025),
    ]
)
def test_step_basic(request, steps: Tuple[GSData, GSData], nfreqs: int):
    steps = request.getfixture(steps)
    for step in steps:
        assert steps.freq_array.shape == (nfreqs,)
        assert np.min(step.freq_array) >= 40 * u.MHz

        # Ensure it's pickleable
        pickle.dumps(step)


def test_model_step(model_step: List[GSData]):
    m = model_step[0]
    assert m.data_model.nterms == 5


def test_lstavg_step(steps: tuple[GSData, GSData], lstbin_steps: tuple[GSData, GSData]):
    avg = steps[0]
    assert avg.data_model.nterms == 5
    assert avg.in_lst

    binned = lstbin_steps[0]
    assert avg.ntimes == binned.ntimes
