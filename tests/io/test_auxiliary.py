from pathlib import Path

import numpy as np

from edges.io.auxiliary import read_thermlog_file, read_weather_file


def test_weather_reading_default(datadir: Path):
    data_file = datadir / "weather_test_data.txt"
    out = read_weather_file(data_file, year=2020, day=108, hour=4)

    # Default is to read the rest of the day
    assert out["day"].max() == 108
    assert np.all(out["hour"] >= 4)


def test_weather_reading_1hr(datadir: Path):
    data_file = datadir / "weather_test_data.txt"
    out = read_weather_file(data_file, year=2020, day=108, hour=4, n_hours=1)

    assert out["hour"].max() == 4  # It's exclusive, so it stays in the fourth hour.


def test_thermlog_reading_end_time(datadir: Path):
    data_file = datadir / "thermlog_test_data.txt"
    end_time = (2020, 109, 5, 30)
    out = read_thermlog_file(data_file, year=2020, day=108, hour=4, end_time=end_time)

    # Default is to read the rest of the day
    for d in out:
        assert (d["year"], d["day"], d["hour"], d["minute"]) < end_time
