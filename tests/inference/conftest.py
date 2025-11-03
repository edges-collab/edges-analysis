import numpy as np
import pytest
from scipy import stats

from edges.inference.eor_models import FlattenedGaussian
from edges.inference.foregrounds import DampedOscillations, LogPoly
from edges.modeling import LinLog


@pytest.fixture(scope="session")
def fiducial_eor(calobs):
    return FlattenedGaussian(
        freqs=calobs.freq.freq,
        params={
            "A": {
                "fiducial": 0.5,
                "min": 0,
                "max": 1.5,
                "ref": stats.norm(0.5, scale=0.01),
            },
            "w": {
                "fiducial": 15,
                "min": 5,
                "max": 25,
                "ref": stats.norm(15, scale=0.1),
            },
            "tau": {
                "fiducial": 5,
                "min": 0,
                "max": 20,
                "ref": stats.norm(5, scale=0.1),
            },
            "nu0": {
                "fiducial": 78,
                "min": 60,
                "max": 90,
                "ref": stats.norm(78, scale=0.1),
            },
        },
    )


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
            "amp_sin": {"fiducial": 0, "min": -5, "max": 5},
            "amp_cos": {"fiducial": 0, "min": -3, "max": -2},
            "P": {"fiducial": 15, "min": -100, "max": 100},
            "b": {"fiducial": 1, "min": -100, "max": 100},
        },
    )
