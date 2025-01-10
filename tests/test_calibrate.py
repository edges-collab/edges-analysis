"""Test the calibration module."""

from pathlib import Path

import numpy as np
import pytest
from astropy import units as un
from edges_analysis.calibration import calibrate
from pygsdata import GSData


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

    @pytest.mark.filterwarnings("ignore:loader 'b'!edges_cal")
    def test_equality_uncal_vs_uncaltemp(
        self, gsd_ones: GSData, calpath: Path, s11path: Path
    ):
        data = gsd_ones.update(data_unit="uncalibrated")
        new = calibrate.apply_noise_wave_calibration(
            data, calobs=calpath, band="low", ant_s11_object=s11path
        )

        data_temp = calibrate.approximate_temperature(data, tload=300, tns=1000)
        new2 = calibrate.apply_noise_wave_calibration(
            data_temp,
            calobs=calpath,
            band="low",
            ant_s11_object=s11path,
            tload=300,
            tns=1000,
        )

        assert np.all(new.data == new2.data)


class TestApplyLossCorrection:
    def test_explicit_unity_loss(self, mock: GSData):
        loss = np.ones(mock.nfreqs)
        data = calibrate.apply_loss_correction(mock, ambient_temp=300 * un.K, loss=loss)
        np.testing.assert_array_equal(data.data, mock.data)

    def test_functional_unity_loss(self, mock: GSData):
        def loss(freqs):
            return np.ones(len(freqs))

        data = calibrate.apply_loss_correction(
            mock, ambient_temp=300 * un.K, loss_function=loss
        )
        np.testing.assert_array_equal(data.data, mock.data)

    def test_zero_temp(self, mock: GSData):
        loss = np.ones(mock.nfreqs) * 2
        data = calibrate.apply_loss_correction(mock, ambient_temp=0 * un.K, loss=loss)
        np.testing.assert_array_equal(data.data, mock.data / 2)

    def test_bad_data_unit(self, mock_power: GSData):
        loss = np.ones(mock_power.nfreqs)
        with pytest.raises(ValueError, match="Data must be temperature"):
            calibrate.apply_loss_correction(
                mock_power, ambient_temp=300 * un.K, loss=loss
            )
