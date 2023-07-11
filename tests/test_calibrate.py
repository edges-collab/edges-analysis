import pytest

import numpy as np
from pathlib import Path

from edges_analysis.calibration import calibrate
from edges_analysis.gsdata import GSData


def test_approximate_temperature(gsd_ones: GSData):
    # mock gsdata
    data = gsd_ones.update(data_unit="uncalibrated")

    new = calibrate.approximate_temperature(data, tload=300, tns=1000)
    assert new.data_unit == "uncalibrated_temp"
    assert not np.any(new.data == data.data)

    new = data.update(data_unit="temperature")
    with pytest.raises(ValueError, match="data_unit must be 'uncalibrated'"):
        calibrate.approximate_temperature(new, tload=300, tns=1000)


class TestApplyNoiseWaveCalibration:
    def test_bad_inputs(
        self, gsd_ones: GSData, gsd_ones_power: GSData, calpath: Path, s11path: Path
    ):
        data = gsd_ones.update(data_unit="temperature")
        with pytest.raises(
            ValueError, match="Data must be uncalibrated to apply calibration!"
        ):
            calibrate.apply_noise_wave_calibration(
                data, calobs=calpath, band="low", s11_path=s11path
            )

        data = gsd_ones.update(data_unit="uncalibrated_temp")
        with pytest.raises(
            ValueError,
            match="You need to supply tload and tns if data_unit is uncalibrated_temp",
        ):
            calibrate.apply_noise_wave_calibration(
                data, calobs=calpath, band="low", s11_path=s11path
            )

        data = gsd_ones_power.update(data_unit="uncalibrated")
        with pytest.raises(
            ValueError,
            match="Can only apply noise-wave calibration to single load data!",
        ):
            calibrate.apply_noise_wave_calibration(
                data, calobs=calpath, band="low", s11_path=s11path
            )

    def test_equality_uncal_vs_uncaltemp(
        self, gsd_ones: GSData, calpath: Path, s11path: Path
    ):
        data = gsd_ones.update(data_unit="uncalibrated")
        new = calibrate.apply_noise_wave_calibration(
            data, calobs=calpath, band="low", s11_path=s11path
        )

        data_temp = calibrate.approximate_temperature(data, tload=300, tns=1000)
        new2 = calibrate.apply_noise_wave_calibration(
            data_temp, calobs=calpath, band="low", s11_path=s11path, tload=300, tns=1000
        )

        assert np.all(new.data == new2.data)
