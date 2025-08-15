import numpy as np
import pytest
from yabf import chi2, run_map

from edges import inference as inf


def create_mock_data(fiducial_fg_logpoly):
    spec = fiducial_fg_logpoly()
    assert len(spec["LogPoly_spectrum"]) == 100
    return spec["LogPoly_spectrum"]


def test_retrieve_params(fiducial_fg_logpoly):
    spec = create_mock_data(fiducial_fg_logpoly)
    lk = chi2.MultiComponentChi2(
        kind="spectrum", components=[fiducial_fg_logpoly], data=spec
    )
    a = run_map(lk)
    assert a.success
    assert np.allclose(a.x, [2, -2.5, 50])
    assert len(a.x) == 3


def test_damped_oscillations(fiducial_dampedoscillations):
    spec = fiducial_dampedoscillations()
    print(spec)
    assert np.allclose(spec["DampedOscillations_spectrum"], 0)
    assert len(spec["DampedOscillations_spectrum"]) == 100


@pytest.mark.parametrize(
    "model",
    [
        inf.PhysicalHills,
        inf.PhysicalSmallIonDepth,
        inf.PhysicalLin,
        inf.IonContrib,
        inf.LinLog,
        inf.LinPoly,
        inf.LogPoly,
    ],
)
def test_positivity_of_foregrounds(model):
    model = model(freqs=np.linspace(50, 100, 100))
    fgspec = next(iter(model().values()))

    assert np.all(fgspec >= 0)
