"""Tests of adding aux data from weather/thermlog files to GSData objects."""

from pathlib import Path

import pytest
from astropy import units as un
from astropy.time import Time

from edges.analysis.aux_data import WeatherError, add_thermlog_data, add_weather_data
from edges.config import config
from edges.io import TEST_DATA_PATH
from edges.testing import create_mock_edges_data


def test_add_thermlog_data():
    t0 = Time("2020:104:14:00")

    gsd = create_mock_edges_data(
        flow=50 * un.MHz,
        fhigh=100 * un.MHz,
        ntime=100,
        time0=t0.jd,
    )

    out = add_thermlog_data(
        gsd, thermlog_file=TEST_DATA_PATH / "thermlog_test_data.txt"
    )

    assert "receiver_temp" in out.auxiliary_measurements.columns


def test_add_thermlog_data_out_of_bounds():
    t0 = Time("2020:115:14:00")

    gsd = create_mock_edges_data(
        flow=50 * un.MHz,
        fhigh=100 * un.MHz,
        ntime=100,
        time0=t0.jd,
    )

    with pytest.raises(WeatherError):
        add_thermlog_data(gsd, thermlog_file=TEST_DATA_PATH / "thermlog_test_data.txt")


def test_add_weather_data():
    t0 = Time("2020:107:14:00")

    gsd = create_mock_edges_data(
        flow=50 * un.MHz,
        fhigh=100 * un.MHz,
        ntime=100,
        time0=t0.jd,
    )

    out = add_weather_data(gsd, weather_file=TEST_DATA_PATH / "weather_test_data.txt")

    assert "ambient_temp" in out.auxiliary_measurements.columns


def test_add_weather_data_out_of_bounds():
    t0 = Time("2020:115:14:00")

    gsd = create_mock_edges_data(
        flow=50 * un.MHz,
        fhigh=100 * un.MHz,
        ntime=100,
        time0=t0.jd,
    )

    with pytest.raises(WeatherError):
        add_weather_data(gsd, weather_file=TEST_DATA_PATH / "weather_test_data.txt")


def _mock(time0: str, ntime: int = 10):
    return create_mock_edges_data(
        flow=50 * un.MHz, fhigh=100 * un.MHz, ntime=ntime, time0=Time(time0).jd
    )


def _copy_aux_file(src: Path, dest: Path, old_day: str = "", new_day: str = ""):
    """Copy an aux file, optionally re-labelling its year:day of each line."""
    text = src.read_text()
    if old_day:
        text = text.replace(old_day, new_day)
    dest.write_text(text)
    return dest


class TestAuxFileResolution:
    """Regression tests: default and relative aux paths were unreachable."""

    def test_weather_relative_to_raw_field_data(self):
        gsd = _mock("2020:107:14:00")
        with config.use(raw_field_data=TEST_DATA_PATH):
            for fname in ("weather_test_data.txt", Path("weather_test_data.txt")):
                out = add_weather_data(gsd, weather_file=fname)
                assert "ambient_temp" in out.auxiliary_measurements.columns

    def test_weather_absolute_str_path(self):
        gsd = _mock("2020:107:14:00")
        out = add_weather_data(
            gsd, weather_file=str(TEST_DATA_PATH / "weather_test_data.txt")
        )
        assert "ambient_temp" in out.auxiliary_measurements.columns

    def test_weather_default_file_new(self, tmp_path: Path):
        _copy_aux_file(
            TEST_DATA_PATH / "weather_test_data.txt", tmp_path / "weather2.txt"
        )
        gsd = _mock("2020:107:14:00")
        with config.use(raw_field_data=tmp_path):
            out = add_weather_data(gsd)
        assert "ambient_temp" in out.auxiliary_measurements.columns

    @pytest.mark.parametrize("day", ["2017:100", "2017:329"])
    def test_weather_default_file_old(self, tmp_path: Path, day: str):
        # Only the pre-2017-11-25 file exists, so the old file must be chosen.
        _copy_aux_file(
            TEST_DATA_PATH / "weather_test_data.txt",
            tmp_path / "weather_upto_20171125.txt",
            old_day="2020:107",
            new_day=day,
        )
        gsd = _mock(f"{day}:14:00")
        with config.use(raw_field_data=tmp_path):
            out = add_weather_data(gsd)
        assert "ambient_temp" in out.auxiliary_measurements.columns

    def test_weather_no_file_no_config(self):
        gsd = _mock("2020:107:14:00")
        with (
            config.use(raw_field_data=None),
            pytest.raises(ValueError, match="raw_field_data"),
        ):
            add_weather_data(gsd)

    def test_weather_missing_file(self, tmp_path: Path):
        gsd = _mock("2020:107:14:00")
        with (
            config.use(raw_field_data=tmp_path),
            pytest.raises(FileNotFoundError, match=r"nonexistent\.txt"),
        ):
            add_weather_data(gsd, weather_file="nonexistent.txt")

    def test_thermlog_relative_to_raw_field_data(self):
        gsd = _mock("2020:104:14:00")
        with config.use(raw_field_data=TEST_DATA_PATH):
            out = add_thermlog_data(gsd, thermlog_file="thermlog_test_data.txt")
        assert "receiver_temp" in out.auxiliary_measurements.columns

    def test_thermlog_default_file(self, tmp_path: Path):
        _copy_aux_file(
            TEST_DATA_PATH / "thermlog_test_data.txt", tmp_path / "thermlog_low.txt"
        )
        gsd = _mock("2020:104:14:00")
        with config.use(raw_field_data=tmp_path):
            out = add_thermlog_data(gsd, band="low")
        assert "receiver_temp" in out.auxiliary_measurements.columns

    def test_thermlog_default_needs_band(self, tmp_path: Path):
        gsd = _mock("2020:104:14:00")
        with (
            config.use(raw_field_data=tmp_path),
            pytest.raises(ValueError, match="band"),
        ):
            add_thermlog_data(gsd)
