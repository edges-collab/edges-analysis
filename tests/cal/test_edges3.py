"""Tests of creating EDGES-3 style calibration."""

import pytest
from astropy import units as un

from edges.cal import CalibrationObservation
from edges.io import TEST_DATA_PATH, calobsdef3


@pytest.fixture(scope="module")
def smallcal() -> calobsdef3.CalObsDefEDGES3:
    return calobsdef3.CalObsDefEDGES3.from_standard_layout(
        rootdir=TEST_DATA_PATH / "edges3-mock-root",
        year=2023,
        day=70,
    )


@pytest.fixture(scope="module")
def calobs(smallcal: calobsdef3.CalObsDefEDGES3) -> CalibrationObservation:
    with pytest.warns(UserWarning, match="has no value"):
        return CalibrationObservation.from_edges3_caldef(
            smallcal,
            f_low=50 * un.MHz,
            f_high=100 * un.MHz,
            spectrum_kwargs={
                "default": {"allow_closest_time": True, "temperature": 300.0 * un.K}
            },
        )


def test_calobs_creation(calobs):
    assert calobs is not None
