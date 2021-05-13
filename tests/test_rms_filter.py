from edges_analysis.analysis.filters import get_rms_info, RMSInfo
import numpy as np
import pytest
from edges_analysis import FilteredData
from pathlib import Path


@pytest.mark.parametrize("resolution", [0, 0.0488, 5])
def test_get_rms(mock_calibrated_data, resolution):
    rms_info = get_rms_info(
        [mock_calibrated_data],
        models={
            "linlog": {
                "model": "linlog",
                "bands": ["full", "low", "high"],
                "params": {"n_terms": 2},
                "resolution": resolution,
            }
        },
    )

    assert isinstance(rms_info, RMSInfo)
    assert "linlog" in rms_info.model_params
    assert (50.0, 100.0) in rms_info.model_eval["linlog"]
    assert (50.0, 75.0) in rms_info.model_eval["linlog"]
    assert (75.0, 100.0) in rms_info.model_eval["linlog"]
    assert len(rms_info.model_eval["linlog"][(50.0, 100.0)]) == 50
    assert np.sum(rms_info.flags["linlog"][(50.0, 100.0)]) == 0


@pytest.mark.parametrize("resolution", [0, 0.0488, 5])
def test_rms_filter(mock_calibrated_data, resolution, tmpdir: Path):
    rms_info = get_rms_info(
        [mock_calibrated_data],
        models={
            "linlog": {
                "model": "linlog",
                "bands": ["full", "low", "high"],
                "params": {"n_terms": 2},
                "resolution": resolution,
            }
        },
    )
    rms_info.write(tmpdir / "rms_info.h5")

    fd = FilteredData.promote(
        mock_calibrated_data, rms_filter_file=tmpdir / "rms_info.h5", n_sigma_rms=4
    )
    assert fd.weights.size == mock_calibrated_data.spectrum.size == 5000

    # We only consider flags that are not near the edges of GHA, since
    # these are hard to fit with the model.
    assert np.isclose(np.mean(fd.weights[:, 5:45]), 1.0, rtol=0.07)
