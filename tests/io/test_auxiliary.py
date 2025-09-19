from pathlib import Path

import numpy as np
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
