"""Test the calibration module."""

from datetime import datetime
from pathlib import Path

import numpy as np
import pytest
from astropy import units as un
from astropy.time import Time
from edges_cal import modelling as mdl
from pygsdata import GSData

from edges_analysis.beams import BeamFactor
from edges_analysis.calibration import calibrate


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


class TestGetClosestS11Time:
    """Tests of the _get_closest_s11_time method."""

    def _setup_dir(self, tmp_path, year: int = 2015):
        d = tmp_path / "s11"
        if not d.exists():
            d.mkdir()

        # Create a few empty files
        find_these_files = [
            (d / f"{year}_312_00_0.s1p"),
            (d / f"{year}_312_00_1.s1p"),
            (d / f"{year}_312_00_2.s1p"),
            (d / f"{year}_312_00_3.s1p"),
        ]
        for fl in find_these_files:
            fl.touch()

        return d, find_these_files

    def test_happy_path(self, tmp_path):
        d, find_these_files = self._setup_dir(tmp_path)

        # Find these files
        out = calibrate._get_closest_s11_time(
            d,
            datetime(year=2015, month=1, day=1),
            s11_file_pattern="{y}_{jd}_{h}_{input}.s1p",
        )
        assert out == sorted(find_these_files)

        # Add some more files
        self._setup_dir(tmp_path, year=2011)
        out = calibrate._get_closest_s11_time(
            d,
            datetime(year=2015, month=1, day=1),
            s11_file_pattern="{y}_{jd}_{h}_{input}.s1p",
        )
        assert out == sorted(find_these_files)

        # If the time object is a Time object...
        out = calibrate._get_closest_s11_time(
            d,
            Time(datetime(year=2015, month=1, day=1)),
            s11_file_pattern="{y}_{jd}_{h}_{input}.s1p",
        )
        assert out == sorted(find_these_files)

    def test_value_errors(self, tmp_path):
        d, _find_these_files = self._setup_dir(tmp_path)

        with pytest.raises(ValueError, match="s11_file_pattern must contain"):
            calibrate._get_closest_s11_time(
                d,
                datetime(year=2015, month=1, day=1),
                s11_file_pattern="{y}_{h}{input}.s1p",
            )

        with pytest.raises(ValueError, match="s11_file_pattern must not contain both"):
            calibrate._get_closest_s11_time(
                d,
                datetime(year=2015, month=1, day=1),
                s11_file_pattern="{y}_{jd}{d}_{h}_{input}.s1p",
            )

    def test_no_files_exist(self, tmp_path):
        tmp_path / "s11"
        with pytest.raises(FileNotFoundError, match="No files found"):
            calibrate._get_closest_s11_time(
                tmp_path,
                datetime(year=2015, month=1, day=1),
                s11_file_pattern="{y}_{jd}_{h}_{input}.s1p",
            )

    def test_wrong_number_of_files(self, tmp_path):
        d, find_these_files = self._setup_dir(tmp_path)
        find_these_files[-1].unlink()

        with pytest.raises(
            FileNotFoundError, match="There need to be four input S1P files"
        ):
            calibrate._get_closest_s11_time(
                d,
                datetime(year=2015, month=1, day=1),
                s11_file_pattern="{y}_{jd}_{h}_{input}.s1p",
            )

    def test_include_files_that_dont_match(self, tmp_path):
        d, find_these_files = self._setup_dir(tmp_path)
        (d / "unmatching.s1p").touch()

        out = calibrate._get_closest_s11_time(
            d,
            datetime(year=2015, month=1, day=1),
            s11_file_pattern="{y}_{jd}_{h}_{input}.s1p",
        )
        assert out == sorted(find_these_files)

    def test_ignore_file(self, tmp_path):
        d, find_these_files = self._setup_dir(tmp_path)
        (d / "2015_001_00_0.s1p").touch()

        out = calibrate._get_closest_s11_time(
            d,
            datetime(year=2015, month=1, day=1),
            s11_file_pattern="{y}_{jd}_{h}_{input}.s1p",
            ignore_files="2015_001",
        )
        assert out == sorted(find_these_files)

    def test_use_month_day(self, tmp_path):
        d = tmp_path / "s11"
        d.mkdir()

        # Create a few empty files
        find_these_files = [
            (d / "2015_01_01_00_0.s1p"),
            (d / "2015_01_01_00_1.s1p"),
            (d / "2015_01_01_00_2.s1p"),
            (d / "2015_01_01_00_3.s1p"),
        ]
        for fl in find_these_files:
            fl.touch()

        out = calibrate._get_closest_s11_time(
            d,
            datetime(year=2015, month=1, day=1),
            s11_file_pattern="{y}_{m}_{d}_{h}_{input}.s1p",
        )
        assert out == sorted(find_these_files)


class TestGetS11Paths:
    """Tests of the get_s11_paths method."""

    def test_with_sequence(self, tmp_path):
        d = tmp_path / "s11"
        d.mkdir()

        paths = []
        for i in range(4):
            paths.append(d / f"{i}.s1p")
            paths[-1].touch()

        out = calibrate.get_s11_paths(paths)
        assert out == paths

        with pytest.raises(ValueError, match="length must be 4"):
            calibrate.get_s11_paths(paths[:-1])

    def test_single_file(self, tmp_path):
        fl = tmp_path / "file.csv"
        fl.touch()
        out = calibrate.get_s11_paths(fl)
        assert out == [fl]

    def test_get_closest_time(self, tmp_path):
        d = tmp_path / "s11"
        if not d.exists():
            d.mkdir()

        # Create a few empty files
        find_these_files = [d / f"2015_312_00_{i}.s1p" for i in range(4)]
        for fl in find_these_files:
            fl.touch()

        # Find these files
        out = calibrate.get_s11_paths(
            d,
            begin_time=datetime(year=2015, month=1, day=1),
            s11_file_pattern="{y}_{jd}_{h}_{input}.s1p",
        )
        assert out == sorted(find_these_files)

    def test_find_directly(self, tmp_path):
        d = tmp_path / "s11"
        if not d.exists():
            d.mkdir()

        # Create a few empty files
        find_these_files = [d / f"2015_312_00_{i}.s1p" for i in range(4)]
        for fl in find_these_files:
            fl.touch()

        # Find these files
        out = calibrate.get_s11_paths(
            f"{d}/2015_312_00_{{load}}.s1p",
            begin_time=datetime(year=2015, month=1, day=1),
        )
        assert out == sorted(find_these_files)

        find_these_files[-1].unlink()
        with pytest.raises(FileNotFoundError, match="There are not exactly four"):
            calibrate.get_s11_paths(
                f"{d}/2015_312_00_{{load}}.s1p",
                begin_time=datetime(year=2015, month=1, day=1),
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
