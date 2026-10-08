from pathlib import Path

import numpy as np
import pytest
from astropy import units as un
from astropy.time import Time

from edges.io.auxiliary import read_thermlog_file, read_weather_file


def test_weather_reading_default(datadir: Path):
    data_file = datadir / "weather_test_data.txt"
    out = read_weather_file(data_file, Time("2020:108:04:00"))

    # Default is to read the rest of the day
    assert int(out["time"][-1].yday.split(":")[1]) == 108
    assert np.all(out["time"].ymdhms["hour"] >= 4)


def test_weather_reading_1hr(datadir: Path):
    data_file = datadir / "weather_test_data.txt"
    out = read_weather_file(data_file, Time("2020:108:04:00"), n_hours=1)

    assert (
        out["time"].ymdhms["hour"].max() == 4
    )  # It's exclusive, so it stays in the fourth hour.


def test_thermlog_reading_end_time(datadir: Path):
    data_file = datadir / "thermlog_test_data.txt"
    end_time = Time.strptime("2020/109:05:30", "%Y/%j:%H:%M")
    out = read_thermlog_file(data_file, start=Time("2020:108:04:00"), end=end_time)

    # Default is to read the rest of the day
    assert out["time"].max() < end_time


_OLD = (
    "{t}  rack_temp  306.64 Kelvin, ambient_temp  299.16 Kelvin, "
    "ambient_hum   26.19 percent\n"
)
_NEW = (
    "{t}  rack_temp  306.44 Kelvin, ambient_temp  299.05 Kelvin, "
    "ambient_hum   26.35 percent, frontend  298.95 Kelvin, rcv3_lna  273.20 Kelvin\n"
)


@pytest.mark.parametrize("old_first", [True, False])
def test_weather_mixed_formats(tmp_path: Path, old_first: bool):
    first, second = (_OLD, _NEW) if old_first else (_NEW, _OLD)
    fl = tmp_path / "weather.txt"
    fl.write_text(
        first.format(t="2020:107:11:00:01")
        + second.format(t="2020:107:11:05:01")
        + first.format(t="2020:107:11:10:01")
    )
    out = read_weather_file(fl, Time("2020:107:11:00"))

    assert len(out) == 3
    assert out.colnames == [
        "rack_temp",
        "ambient_temp",
        "ambient_hum",
        "frontend_temp",
        "lna_temp",
        "time",
    ]
    new = np.array([not old_first, old_first, not old_first])
    np.testing.assert_allclose(
        out["rack_temp"].to_value(un.K),
        np.where(new, 306.44, 306.64),
        rtol=0,
        atol=1e-9,
    )
    assert out["frontend_temp"].unit == un.K
    np.testing.assert_allclose(
        out["frontend_temp"][new].to_value(un.K), 298.95, rtol=0, atol=1e-9
    )
    np.testing.assert_allclose(
        out["lna_temp"][new].to_value(un.K), 273.20, rtol=0, atol=1e-9
    )
    assert np.all(np.isnan(out["frontend_temp"][~new]))
    assert np.all(np.isnan(out["lna_temp"][~new]))


def test_weather_malformed_lines(tmp_path: Path):
    fl = tmp_path / "weather.txt"
    fl.write_text(
        _NEW.format(t="2020:107:11:00:01")
        + "\n"
        + "garbage that is not a time\n"
        + "2020:107:11:05:01  rack_temp  garbage\n"
        + _NEW.format(t="2020:107:11:10:01")
    )
    with pytest.warns(UserWarning, match="Skipped 3 blank or malformed lines"):
        out = read_weather_file(fl, Time("2020:107:11:00"))
    assert len(out) == 2
    assert [t.isot for t in out["time"]] == [
        "2020-04-16T11:00:01.000",
        "2020-04-16T11:10:01.000",
    ]


def test_weather_only_malformed_lines(tmp_path: Path):
    fl = tmp_path / "weather.txt"
    fl.write_text("2020:107:11:05:01  rack_temp  garbage\n")
    with pytest.raises(ValueError, match="No lines of the aux file"):
        read_weather_file(fl, Time("2020:107:11:00"))


def test_weather_no_data_in_range(datadir: Path):
    out = read_weather_file(
        datadir / "weather_test_data.txt", Time("2019:001:00:00"), n_hours=1
    )
    assert len(out) == 0


def test_thermlog_values(datadir: Path):
    out = read_thermlog_file(
        datadir / "thermlog_test_data.txt", start=Time("2020:104:01:30"), n_hours=1
    )
    assert out.colnames == ["temp_set", "receiver_temp", "power_percent", "time"]
    assert out["temp_set"].unit == un.deg_C
    assert out["receiver_temp"][0].to_value(un.deg_C) == pytest.approx(25.01, abs=1e-9)
    assert out["power_percent"][0].to_value(un.percent) == pytest.approx(
        -32.94, abs=1e-9
    )
