"""Test the calibration module."""

import numpy as np
import pytest
from astropy import units as un
from pygsdata import GSData

from edges import modeling as mdl
from edges.analysis import calibrate
from edges.cal import Calibrator, apply
from edges.cal.sparams import ReflectionCoefficient
from edges.sim.antenna_beam_factor import BeamFactor


def get_ideal_s11model(freqs):
    return ReflectionCoefficient(
        reflection_coefficient=np.zeros(freqs.size, dtype=complex), freqs=freqs
    )


def get_random_calibrator(freqs, seed: int = 1234) -> Calibrator:
    """A calibrator with random (but physically-sized) noise-wave coefficients."""
    rng = np.random.default_rng(seed)
    n = freqs.size
    return Calibrator(
        freqs=freqs,
        Tsca=rng.uniform(800, 1200, n),
        Toff=rng.uniform(250, 350, n),
        Tunc=rng.uniform(-50, 50, n),
        Tcos=rng.uniform(-50, 50, n),
        Tsin=rng.uniform(-50, 50, n),
        receiver_s11=rng.uniform(0, 0.1, n) * np.exp(2j * np.pi * rng.uniform(size=n)),
    )


def get_random_s11(freqs, seed: int = 4321) -> ReflectionCoefficient:
    rng = np.random.default_rng(seed)
    n = freqs.size
    return ReflectionCoefficient(
        reflection_coefficient=rng.uniform(0.05, 0.5, n)
        * np.exp(2j * np.pi * rng.uniform(size=n)),
        freqs=freqs,
    )


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

    def test_round_trip_random_calibrator(self, gsd_ones: GSData):
        """Calibrating de-calibrated temperatures recovers them, for non-zero S11s."""
        calibrator = get_random_calibrator(gsd_ones.freqs)
        ant_s11 = get_random_s11(gsd_ones.freqs)

        rng = np.random.default_rng(0)
        temp = rng.uniform(1000, 5000, gsd_ones.data.shape) * un.K
        q = calibrator.decalibrate(
            temp, ant_s11=ant_s11.reflection_coefficient, freqs=gsd_ones.freqs
        )
        data = gsd_ones.update(data=q.to_value(""), data_unit="uncalibrated")

        out = calibrate.apply_noise_wave_calibration(
            data, calibrator=calibrator, antenna_s11=ant_s11
        )
        assert out.data_unit == "temperature"
        np.testing.assert_allclose(out.data, temp.to_value("K"), rtol=1e-12, atol=0)

    def _decalibrated(self, gsd: GSData, calibrator: Calibrator, ant_s11, seed=0):
        rng = np.random.default_rng(seed)
        temp = rng.uniform(1000, 5000, gsd.data.shape) * un.K
        temp_model = temp * (1 + rng.uniform(-0.01, 0.01, gsd.data.shape))
        kw = {"ant_s11": ant_s11.reflection_coefficient, "freqs": gsd.freqs}
        q = calibrator.decalibrate(temp, **kw).to_value("")
        qmodel = calibrator.decalibrate(temp_model, **kw).to_value("")
        return temp, temp_model, q, qmodel

    def test_residuals_are_calibrated(self, gsd_ones: GSData):
        """Regression test for ANA-3: data with a model used to crash."""
        calibrator = get_random_calibrator(gsd_ones.freqs)
        ant_s11 = get_random_s11(gsd_ones.freqs)
        temp, temp_model, q, qmodel = self._decalibrated(gsd_ones, calibrator, ant_s11)
        data = gsd_ones.update(data=q, residuals=q - qmodel, data_unit="uncalibrated")

        out = calibrate.apply_noise_wave_calibration(
            data, calibrator=calibrator, antenna_s11=ant_s11
        )
        np.testing.assert_allclose(out.data, temp.to_value("K"), rtol=1e-12, atol=0)
        np.testing.assert_allclose(
            out.residuals, (temp - temp_model).to_value("K"), rtol=0, atol=1e-8
        )

    def test_residuals_uncalibrated_temp(self, gsd_ones: GSData):
        """Residuals of approximate-temperature data are calibrated too (ANA-3)."""
        calibrator = get_random_calibrator(gsd_ones.freqs)
        ant_s11 = get_random_s11(gsd_ones.freqs)
        temp, temp_model, q, qmodel = self._decalibrated(gsd_ones, calibrator, ant_s11)
        tload, tns = 300.0, 1000.0
        data = gsd_ones.update(
            data=q * tns + tload,
            residuals=(q - qmodel) * tns,
            data_unit="uncalibrated_temp",
        )

        out = calibrate.apply_noise_wave_calibration(
            data, calibrator=calibrator, antenna_s11=ant_s11, tload=tload, tns=tns
        )
        np.testing.assert_allclose(out.data, temp.to_value("K"), rtol=1e-12, atol=0)
        np.testing.assert_allclose(
            out.residuals, (temp - temp_model).to_value("K"), rtol=0, atol=1e-8
        )

    def test_calibrator_from_path(self, gsd_ones: GSData, tmp_path):
        """A path to a calibrator file can be given instead of the object (ANA-3)."""
        calibrator = get_random_calibrator(gsd_ones.freqs)
        ant_s11 = get_random_s11(gsd_ones.freqs)
        temp, _, q, _ = self._decalibrated(gsd_ones, calibrator, ant_s11)
        data = gsd_ones.update(data=q, data_unit="uncalibrated")

        calibrator.write(tmp_path / "calibrator.h5")

        for pth in (tmp_path / "calibrator.h5", str(tmp_path / "calibrator.h5")):
            out = calibrate.apply_noise_wave_calibration(
                data, calibrator=pth, antenna_s11=ant_s11
            )
            np.testing.assert_allclose(out.data, temp.to_value("K"), rtol=1e-12, atol=0)

    def test_antenna_s11_interpolated_to_data_freqs(self, gsd_ones: GSData):
        """An antenna S11 on a different frequency grid is modelled (ANA-3)."""
        calibrator = get_random_calibrator(gsd_ones.freqs)
        gamma = 0.2 * np.exp(0.3j)
        exact = ReflectionCoefficient(
            reflection_coefficient=np.full(gsd_ones.nfreqs, gamma), freqs=gsd_ones.freqs
        )
        fine_freqs = np.linspace(40, 110, 301) * un.MHz
        fine = ReflectionCoefficient(
            reflection_coefficient=np.full(fine_freqs.size, gamma), freqs=fine_freqs
        )
        temp, _, q, _ = self._decalibrated(gsd_ones, calibrator, exact)
        data = gsd_ones.update(data=q, data_unit="uncalibrated")

        out = calibrate.apply_noise_wave_calibration(
            data, calibrator=calibrator, antenna_s11=fine
        )
        np.testing.assert_allclose(out.data, temp.to_value("K"), rtol=1e-9, atol=0)


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

    @pytest.mark.parametrize("loss_value", [1e-3, 0.3, 0.9, 1.0])
    def test_ambient_is_invariant(self, mock: GSData, loss_value: float):
        """A measurement at the ambient temperature is unchanged by any loss."""
        tamb = 295.0 * un.K
        data = mock.update(data=np.full_like(mock.data, tamb.to_value("K")))
        loss = np.full(mock.nfreqs, loss_value)
        out = calibrate.apply_loss_correction(data, ambient_temp=tamb, loss=loss)
        np.testing.assert_allclose(out.data, tamb.to_value("K"), rtol=1e-12, atol=0)

    def test_inverts_forward_model(self, mock: GSData):
        """Correcting T_meas = L T + (1 - L) T_amb gives back T."""
        rng = np.random.default_rng(1)
        tamb = 290.0 * un.K
        loss = rng.uniform(0.5, 1.0, mock.nfreqs)
        measured = loss * mock.data + (1 - loss) * tamb.to_value("K")

        out = calibrate.apply_loss_correction(
            mock.update(data=measured), ambient_temp=tamb, loss=loss
        )
        np.testing.assert_allclose(out.data, mock.data, rtol=1e-12, atol=0)

    def test_per_time_ambient_temp(self, mock: GSData):
        """A different ambient temperature can be given for each time."""
        rng = np.random.default_rng(2)
        tamb = rng.uniform(280, 310, mock.ntimes) * un.K
        loss = rng.uniform(0.5, 1.0, mock.nfreqs)
        measured = loss * mock.data + (1 - loss) * tamb.to_value("K")[:, None]

        out = calibrate.apply_loss_correction(
            mock.update(data=measured), ambient_temp=tamb, loss=loss
        )
        np.testing.assert_allclose(out.data, mock.data, rtol=1e-12, atol=0)

    def test_ambient_temp_in_celsius(self, mock: GSData):
        """Regression test for ANA-12: thermlog temperatures in deg C used to crash."""
        loss = np.linspace(0.9, 0.99, mock.nfreqs)
        in_kelvin = calibrate.apply_loss_correction(
            mock, ambient_temp=298.15 * un.K, loss=loss
        )
        in_celsius = calibrate.apply_loss_correction(
            mock, ambient_temp=25.0 * un.deg_C, loss=loss
        )
        np.testing.assert_allclose(in_celsius.data, in_kelvin.data, rtol=1e-12, atol=0)

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
