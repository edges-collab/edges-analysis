"""Test the calibration module."""

import numpy as np
import pytest
from astropy import units as un
from pygsdata import GSData

from edges import modeling as mdl
from edges.analysis import calibrate
from edges.cal import Calibrator, apply
from edges.cal.s11.base import CalibratedS11
from edges.sim.antenna_beam_factor import BeamFactor


def get_ideal_s11model(freqs):
    return CalibratedS11(s11=np.zeros(freqs.size, dtype=complex), freqs=freqs)


class TestApplyNoiseWaveCalibration:
    def test_bad_inputs(
        self,
        gsd_ones: GSData,
        gsd_ones_power: GSData,
        calibrator: Calibrator,
    ):
        data = gsd_ones.update(data_unit="temperature")
        s11 = get_ideal_s11model(data.freqs)
        with pytest.raises(
            ValueError, match="Data must be uncalibrated to apply calibration!"
        ):
            calibrate.apply_noise_wave_calibration(
                data, calibrator=calibrator, antenna_s11=s11
            )

        data = gsd_ones.update(data_unit="uncalibrated_temp")
        with pytest.raises(
            ValueError,
            match="You need to supply tload and tns if data_unit is uncalibrated_temp",
        ):
            calibrate.apply_noise_wave_calibration(
                data, calibrator=calibrator, antenna_s11=s11
            )

        data = gsd_ones_power.update(data_unit="uncalibrated")
        with pytest.raises(
            ValueError,
            match="Can only apply noise-wave calibration to single load data!",
        ):
            calibrate.apply_noise_wave_calibration(
                data, calibrator=calibrator, antenna_s11=s11
            )

    @pytest.mark.filterwarnings("ignore:loader 'b'!edges.cal")
    def test_equality_uncal_vs_uncaltemp(
        self,
        gsd_ones: GSData,
        calibrator: Calibrator,
    ):
        s11 = get_ideal_s11model(freqs=gsd_ones.freqs)

        data = gsd_ones.update(data_unit="uncalibrated")
        new = calibrate.apply_noise_wave_calibration(
            data, calibrator=calibrator, antenna_s11=s11
        )

        data_temp = apply.approximate_temperature(
            data, tload=300 * un.K, tns=1000 * un.K
        )
        new2 = calibrate.apply_noise_wave_calibration(
            data_temp,
            calibrator=calibrator,
            antenna_s11=s11,
            tload=300 * un.K,
            tns=1000 * un.K,
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


class TestApplyBeamFactorDirectly:
    def _setup(self, tmp_path, gsd: GSData):
        bfac = (gsd.freqs / (75.0 * un.MHz)) ** 0.2
        x = np.ones_like(bfac)

        out = tmp_path / "beamfac.txt"
        np.savetxt(out, np.array([x, x, x, bfac]).T)
        return out

    def test_happy_path(self, tmp_path, mock):
        pth = self._setup(tmp_path, mock)
        out = calibrate.apply_beam_factor_directly(mock, pth)
        assert not np.allclose(out.data, mock.data)

    def test_bad_data(self, tmp_path, mock_power):
        pth = self._setup(tmp_path, mock_power)
        with pytest.raises(
            NotImplementedError,
            match="Can only apply beam correction to data with a single load",
        ):
            calibrate.apply_beam_factor_directly(mock_power, pth)

    def test_data_with_resids(self, tmp_path, mock_with_model):
        pth = self._setup(tmp_path, mock_with_model)
        out = calibrate.apply_beam_factor_directly(mock_with_model, pth)
        assert not np.allclose(out.residuals, mock_with_model.residuals)


class TestApplyBeamCorrection:
    def _setup(self) -> BeamFactor:
        frequencies = np.linspace(50, 100, 501)
        lsts = np.arange(0, 24, 0.01)

        return BeamFactor(
            frequencies=frequencies,
            lsts=lsts,
            reference_frequency=75.0,
            antenna_temp=np.ones((len(lsts), len(frequencies))),
            antenna_temp_ref=np.ones(len(lsts)),
        )

    @pytest.mark.parametrize("integrate_before_ratio", [True, False])
    @pytest.mark.parametrize("resample_beam_lsts", [True, False])
    @pytest.mark.parametrize("oversample_factor", [1, 5])
    def test_with_unity_beam_factor(
        self,
        integrate_before_ratio,
        resample_beam_lsts,
        oversample_factor,
        mock_lstbinned,
    ):
        bf = self._setup()

        out = calibrate.apply_beam_correction(
            mock_lstbinned,
            beam=bf,
            integrate_before_ratio=integrate_before_ratio,
            resample_beam_lsts=resample_beam_lsts,
            oversample_factor=oversample_factor,
            freq_model=mdl.Polynomial(n_terms=5),
        )
        np.testing.assert_allclose(out.data, mock_lstbinned.data)

    def test_with_bad_loads(self, mock_power):
        bf = self._setup()
        with pytest.raises(
            NotImplementedError,
            match="Can only apply beam correction to data with a single load",
        ):
            calibrate.apply_beam_correction(
                mock_power,
                beam=bf,
                freq_model=mdl.Polynomial(n_terms=5),
            )

    def test_single_lst_with_interp(self, mock):
        bf = self._setup()
        bf = bf.at_lsts(bf.lsts[:1])

        with pytest.raises(ValueError, match="Your beam has a single LST"):
            calibrate.apply_beam_correction(
                mock,
                beam=bf,
                resample_beam_lsts=True,
                freq_model=mdl.Polynomial(n_terms=5),
            )
