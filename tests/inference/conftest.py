import numpy as np
import pytest

from edges.inference.foregrounds import DampedOscillations, LogPoly
from edges.modeling import LinLog


@pytest.fixture
def fiducial_fg():
    return LinLog(n_terms=5, parameters=[2000, 10, -10, 5, -5])


@pytest.fixture
def fiducial_fg_logpoly():
    return LogPoly(
        freqs=np.linspace(50, 100, 100),
        poly_order=2,
        params={
            "p0": {"fiducial": 2, "min": -5, "max": 5},
            "p1": {"fiducial": -2.5, "min": -3, "max": -2},
            "p2": {"fiducial": 50, "min": -100, "max": 100},
        },
    )


@pytest.fixture
def fiducial_dampedoscillations():
    return DampedOscillations(
        freqs=np.linspace(50, 100, 100),
        params={
            "amp_sin": {"fiducial": 0.3, "min": -5, "max": 5},
            "amp_cos": {"fiducial": -0.2, "min": -5, "max": 5},
            "P": {"fiducial": 15, "min": 1, "max": 100},
            "b": {"fiducial": 1, "min": -10, "max": 10},
        },
    )
