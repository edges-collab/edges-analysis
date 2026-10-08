"""Test spectrum reading."""

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from astropy import units as un
from astropy.table import QTable
from astropy.time import Time
from pygsdata import GSData
from pygsdata.select import select_freqs, select_times

from edges import const, io
from edges.cal import LoadSpectrum, spectra
from edges.cal.spectra import read_temperature_log
from edges.cal.thermistor import ThermistorReadings, get_temperature_thermistor
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


def _write_synthetic_load(tmp_path: Path) -> dict:
    """Write synthetic power spectra and an overlapping thermistor CSV.

    The first half of the integrations has Q = 0.2 and a cool thermistor reading;
    the second half has Q = 0.8 and a warm one.
    """
    ntime, nfreq = 20, 32
    nhalf = ntime // 2
    t0 = Time("2023-03-01T12:00:00", scale="utc")
    cadence = 39.0  # seconds per three-position cycle
    t_ant = t0 + np.arange(ntime) * cadence * un.s
    times = Time(
        np.stack([(t_ant + i * 13 * un.s).jd for i in range(3)], axis=1), format="jd"
    )

    p_load, p_ns = 1.0, 2.0
    q_true = np.where(np.arange(ntime) < nhalf, 0.2, 0.8)
    data = np.zeros((3, 1, ntime, nfreq))
    data[0, 0] = (p_load + q_true * (p_ns - p_load))[:, None]
    data[1] = p_load
    data[2] = p_ns

    gsd = GSData(
        data=data,
        freqs=np.linspace(50, 100, nfreq) * un.MHz,
        times=times,
        telescope=const.KNOWN_TELESCOPES["edges-low"],
        loads=("ant", "internal_load", "internal_load_plus_noise_source"),
        data_unit="power",
    )
    specfile = tmp_path / "synthetic.gsh5"
    gsd.write_gsh5(specfile)

    # Thermistor readings 2 s before each spectrum, at the same cadence, so that
    # each spectrum unambiguously matches one reading.
    r_cool, r_warm = 17000.0, 12000.0
    resistance = np.where(np.arange(ntime) < nhalf, r_cool, r_warm)
    therm_times = t_ant - 2 * un.s
    lines = [
        "Date,Time,LNA Voltage,LNA Thermistor (Ohm),LNA (C),SP4T Voltage,"
        "SP4T Thermistor (Ohm),SP4T (C),Load Voltage,Load-thermistor (Ohm),"
        "Load (C),Room_Temp(C)"
    ]
    for t, r in zip(therm_times, resistance, strict=True):
        dt = t.to_datetime()
        lines.append(
            f"{dt:%m/%d/%Y},{dt.hour}:{dt.minute}:{dt.second},"
            f"0.0,50.0,25.0,2.5,10000.0,25.0,3.0,{r},25.0,25.0"
        )
    thermfile = tmp_path / "thermistor.csv"
    thermfile.write_text("\n".join(lines) + "\n")

    t_cool, t_warm = get_temperature_thermistor(np.array([r_cool, r_warm]) * un.Ohm).to(
        "K"
    )
    return {
        "specfile": specfile,
        "thermfile": thermfile,
        "t_cool": t_cool,
        "t_warm": t_warm,
    }


class TestTemperatureRangeSelection:
    """Unmocked tests of selecting integrations by thermistor temperature."""

    @pytest.mark.parametrize("which", ["cool", "warm"])
    def test_from_loaddef_temperature_range(self, tmp_path, which):
        syn = _write_synthetic_load(tmp_path)
        tsel = syn[f"t_{which}"]
        spec = LoadSpectrum.from_loaddef(
            specfiles=[syn["specfile"]],
            thermistor_pth=syn["thermfile"],
            load_name="ambient",
            f_low=50 * un.MHz,
            f_high=100 * un.MHz,
            ignore_times=0,
            temperature_range=(tsel - 0.5 * un.K, tsel + 0.5 * un.K),
        )
        expected = 0.2 if which == "cool" else 0.8
        np.testing.assert_allclose(spec.averaged_q, expected, rtol=1e-12)
        np.testing.assert_allclose(spec.variance_q, 0, atol=1e-20)

    def test_from_loaddef_without_temperature_range(self, tmp_path):
        syn = _write_synthetic_load(tmp_path)
        spec = LoadSpectrum.from_loaddef(
            specfiles=[syn["specfile"]],
            thermistor_pth=syn["thermfile"],
            load_name="ambient",
            f_low=50 * un.MHz,
            f_high=100 * un.MHz,
            ignore_times=0,
        )
        np.testing.assert_allclose(spec.averaged_q, 0.5, rtol=1e-12)

    def test_get_ave_and_var_spec_temperature_range(self, tmp_path):
        syn = _write_synthetic_load(tmp_path)
        data = GSData.from_file(syn["specfile"])
        therm = ThermistorReadings.from_csv(syn["thermfile"])
        mean, var = spectra.get_ave_and_var_spec(
            data,
            thermistor=therm,
            temperature_range=(syn["t_cool"] - 0.5 * un.K, syn["t_cool"] + 0.5 * un.K),
        )
        np.testing.assert_allclose(mean.data, 0.2, rtol=1e-12)
        np.testing.assert_allclose(var.data, 0, atol=1e-20)

    def test_tuple_time_coordinate_swpos(self, tmp_path):
        """A (base, load) tuple for time_coordinate_swpos is unpacked."""
        syn = _write_synthetic_load(tmp_path)
        data = GSData.from_file(syn["specfile"])
        mean, _ = spectra.get_ave_and_var_spec(
            data, ignore_times=10, time_coordinate_swpos=(0, 0)
        )
        np.testing.assert_allclose(mean.data, 0.8, rtol=1e-12)


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

    def test_cache_invalidated_when_input_file_changes(self, tmp_path, caldef):
        """Rewriting an input file must not silently reuse a stale cache."""
        import os

        cache = tmp_path / "cache"
        kw = {"f_high": 100 * un.MHz, "f_low": 50 * un.MHz, "cache_dir": cache}
        LoadSpectrum.from_loaddef(caldef.ambient, **kw)
        assert len(list(cache.glob("*.gsh5"))) == 1

        # Same inputs: cache is reused.
        LoadSpectrum.from_loaddef(caldef.ambient, **kw)
        assert len(list(cache.glob("*.gsh5"))) == 1

        # Bump the modification time of the thermistor file: new cache entry.
        therm = caldef.ambient.thermistor
        st = therm.stat()
        try:
            os.utime(therm, ns=(st.st_atime_ns, st.st_mtime_ns + 10**9))
            LoadSpectrum.from_loaddef(caldef.ambient, **kw)
        finally:
            os.utime(therm, ns=(st.st_atime_ns, st.st_mtime_ns))
        assert len(list(cache.glob("*.gsh5"))) == 2

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
