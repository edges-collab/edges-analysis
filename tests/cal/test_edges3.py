"""Tests of creating EDGES-3 style calibration."""

import attrs
import numpy as np
import pytest
from astropy import units as un

from edges.cal import CalibrationObservation, LoadSpectrum
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


@pytest.mark.filterwarnings("ignore:.*has no value:UserWarning")
def test_load_spectrum_from_split_templogs(smallcal, tmp_path):
    """Temperatures from several overlapping templogs match those from the full log."""
    full = smallcal.ambient.templog
    text = full.read_text(encoding="latin-1")
    # Split at an entry boundary, with the two halves overlapping.
    starts = [i for i in range(len(text)) if text.startswith("2023_", i)]
    part0, part1 = tmp_path / "part0.log", tmp_path / "part1.log"
    part0.write_text(text[: starts[500]], encoding="latin-1")
    part1.write_text(text[starts[400] :], encoding="latin-1")

    split = attrs.evolve(smallcal.ambient, templog=[part0, part1])

    kw = {"f_low": 50 * un.MHz, "f_high": 100 * un.MHz}
    with pytest.warns(UserWarning, match="malformed lines"):
        ref = LoadSpectrum.from_loaddef(smallcal.ambient, **kw)
    with pytest.warns(UserWarning, match="malformed lines"):
        new = LoadSpectrum.from_loaddef(split, **kw)

    assert new.temp_ave.unit == un.K
    assert np.isfinite(ref.temp_ave)
    np.testing.assert_allclose(
        new.temp_ave.value, ref.temp_ave.value, rtol=0, atol=1e-10
    )
