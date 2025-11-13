"""Test spectrum reading."""

from types import SimpleNamespace

import numpy as np
import pytest
from astropy import units as un
from astropy.table import QTable
from astropy.time import Time
from pygsdata import GSData
from pygsdata.select import select_freqs, select_times

from edges import io
from edges.cal import LoadSpectrum, spectra
from edges.cal.spectra import read_temperature_log
from edges.cal.thermistor import ThermistorReadings
from edges.io.calobsdef import CalObsDefEDGES2


class TestFlagDataOutsideTemperatureRange:
    def setup_class(self):
        rng = np.random.default_rng()
        self.spec_times = Time(np.linspace(2459867, 2459868, 100), format="jd")
        self.therm = ThermistorReadings(
            QTable({
                "times": self.spec_times,
                "load_resistance": rng.normal(loc=18000, scale=100, size=100) * un.Ohm,
            })
        )

    def test_happy_path(self):
        good = spectra.flag_data_outside_temperature_range(
            temperature_range=5 * un.K,
            spec_times=self.spec_times,
            thermistor=self.therm,
        )

        assert np.all(good)

    def test_large_scatter(self):
        good = spectra.flag_data_outside_temperature_range(
            temperature_range=0.1 * un.K,
            spec_times=self.spec_times,
            thermistor=self.therm,
        )

        assert np.sum(~good) > 0

    def test_all_flagged(self):
        with pytest.raises(
            RuntimeError, match="The temperature range has masked all spectra"
        ):
            spectra.flag_data_outside_temperature_range(
                temperature_range=(0 * un.K, 10 * un.K),
                spec_times=self.spec_times,
                thermistor=self.therm,
            )


class TestLoadSpectrum:
    def test_read(self, caldef: CalObsDefEDGES2):
        for load in ["ambient", "hot_load", "open", "short"]:
            spec = LoadSpectrum.from_loaddef(
                getattr(caldef, load), f_high=100 * un.MHz, f_low=50 * un.MHz
            )

            print(spec.averaged_q.shape)
            print(np.where(np.isnan(spec.averaged_q)))

            mask = ~np.isnan(spec.averaged_q)
            assert np.sum(~mask) < 100
            assert spec.averaged_q.ndim == 1

            assert np.all(~np.isnan(spec.variance_q[mask]))
            assert ~np.isinf(spec.temp_ave)
            assert ~np.isnan(spec.temp_ave)

    def test_gauss_smooth(self, caldef: CalObsDefEDGES2):
        spec = LoadSpectrum.from_loaddef(
            caldef.ambient,
            f_high=100 * un.MHz,
            f_low=50 * un.MHz,
            freq_bin_size=8,
            frequency_smoothing="gauss",
        )
        spec2 = LoadSpectrum.from_loaddef(
            caldef.ambient,
            f_high=100 * un.MHz,
            f_low=50 * un.MHz,
            freq_bin_size=8,
            frequency_smoothing="bin",
        )

        # gauss smooth has one more value because the way it bins is like Alan's code,
        # which starts at index 0, rather than in the middle of the bin.
        assert len(spec.averaged_q) - 1 == len(spec2.averaged_q)
        mask = (~np.isnan(spec.averaged_q[:-1])) & (~np.isnan(spec2.averaged_q))
        np.testing.assert_allclose(
            spec.averaged_q[:-1][mask], spec2.averaged_q[mask], atol=1e-2, rtol=0.1
        )

    def test_equality(self, caldef: CalObsDefEDGES2):
        spec1 = LoadSpectrum.from_loaddef(
            caldef.ambient, f_high=100 * un.MHz, f_low=50 * un.MHz
        )
        spec2 = LoadSpectrum.from_loaddef(
            caldef.ambient, f_high=100 * un.MHz, f_low=50 * un.MHz
        )

        assert spec1 == spec2

        spec3 = LoadSpectrum.from_loaddef(
            caldef.ambient, f_high=100 * un.MHz, f_low=60 * un.MHz
        )
        assert spec3 != spec2

    def test_temperature_range(self, caldef: CalObsDefEDGES2):
        with pytest.raises(RuntimeError, match="The temperature range has masked"):
            # Fails only because our test data is awful. spectra and thermistor
            # measurements don't overlap.
            LoadSpectrum.from_loaddef(caldef.ambient, temperature_range=0.5 * un.K)

        with pytest.raises(RuntimeError, match="The temperature range has masked"):
            LoadSpectrum.from_loaddef(caldef.ambient, temperature_range=(20, 40))

    def test_bad_init(self, gsd_ones: GSData):
        with pytest.raises(TypeError, match="q must be a GSData object"):
            LoadSpectrum(q=1, variance=2, temp_ave=3 * un.K)

        with pytest.raises(ValueError, match="q must have a single time"):
            LoadSpectrum(q=gsd_ones, temp_ave=3 * un.K)

        gsd = select_times(gsd_ones, indx=[0])
        gsd2 = select_freqs(gsd, indx=range(10))
        with pytest.raises(ValueError, match="variance must be the same shape as q"):
            LoadSpectrum(q=gsd, variance=gsd2, temp_ave=3 * un.K)

    def test_cache(self, tmp_path, caldef):
        spec = LoadSpectrum.from_loaddef(
            caldef.ambient, f_high=100 * un.MHz, f_low=50 * un.MHz, cache_dir=tmp_path
        )

        spec2 = LoadSpectrum.from_loaddef(
            caldef.ambient, f_high=100 * un.MHz, f_low=50 * un.MHz, cache_dir=tmp_path
        )

        all_cached_files = list(tmp_path.glob("*.gsh5"))
        assert any(fl.name.startswith("Ambient_") for fl in all_cached_files)

        assert spec == spec2

    def test_bad_loaddef(self, monkeypatch):
        with pytest.raises(
            ValueError, match="Either loaddef or specfiles AND load_name"
        ):
            LoadSpectrum.from_loaddef(f_high=100 * un.MHz, f_low=50 * un.MHz)

    def test_missing_templog(self, monkeypatch):
        def _mock_ave_var_spec(*args, **kwargs):
            return None, None

        def _mock_read_spectra(*args, **kwargs):
            return None

        monkeypatch.setattr(
            "edges.cal.spectra.get_ave_and_var_spec", _mock_ave_var_spec
        )
        monkeypatch.setattr("edges.cal.spectra.read_spectra", _mock_read_spectra)

        with pytest.raises(ValueError, match=r"templog doesn't exist"):
            LoadSpectrum.from_loaddef(
                specfiles=[],
                load_name="ambient",
                f_high=100 * un.MHz,
                f_low=50 * un.MHz,
            )

    @pytest.mark.filterwarnings("ignore::UserWarning")
    def test_with_templog(self, gsd_averaged, monkeypatch):
        tlog = io.TEST_DATA_PATH / "edges3-mock-root/temperature_logger/temperature.log"
        table = read_temperature_log(tlog)

        start = table["time"].min()
        end = table["time"].max()

        def _mock_read_spectra(*args, **kwargs):
            times = Time(np.linspace(start.jd, end.jd, 10), format="jd")
            return SimpleNamespace(times=times)

        def _mock_ave_var_spec(*args, **kwargs):
            return gsd_averaged, gsd_averaged

        monkeypatch.setattr("edges.cal.spectra.read_spectra", _mock_read_spectra)
        monkeypatch.setattr(
            "edges.cal.spectra.get_ave_and_var_spec", _mock_ave_var_spec
        )

        LoadSpectrum.from_loaddef(
            specfiles=[],
            load_name="ambient",
            templog=tlog,
            f_high=100 * un.MHz,
            f_low=50 * un.MHz,
        )
