import warnings
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pytest
from astropy import units as un
from astropy.table import QTable
from astropy.time import Time

from edges.io.auxiliary import (
    _PATTERNS,
    _UNITS,
    _get_end_time,
    _parse_aux_line,
    _read_aux_file,
    read_thermlog_file,
    read_weather_file,
)


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


def test_end_is_exclusive(tmp_path: Path):
    fl = tmp_path / "weather.txt"
    fl.write_text(
        "".join(
            _NEW.format(t=t)
            for t in ("2020:107:11:00:01", "2020:107:11:05:01", "2020:107:11:10:01")
        )
    )
    start = Time("2020:107:11:00:01")
    out = read_weather_file(fl, start, end=Time("2020:107:11:05:01"))
    assert len(out) == 1
    assert out["time"][0] == start  # start is inclusive

    out = read_weather_file(fl, start, end=Time("2020:107:11:05:02"))
    assert len(out) == 2


def _reference_read_aux_file(aux_file: Path, start: Time, end: Time):
    """Straightforward line-by-line reader, as a reference for ``_read_aux_file``.

    Returns the table and the number of skipped lines.
    """
    auxdata = []
    n_bad = 0
    with aux_file.open("r") as fl:
        for line in fl:
            if not line.strip():
                n_bad += 1
                continue

            try:
                t = Time.strptime(line[:17], "%Y:%j:%H:%M:%S")
            except ValueError:
                n_bad += 1
                continue

            if t < start:
                continue
            if t >= end:
                break

            values = _parse_aux_line(line[17:])
            if values is None:
                n_bad += 1
                continue
            auxdata.append(values | {"time": t})

    if n_bad and not auxdata:
        raise ValueError("No lines of the aux file")

    if not auxdata:
        return QTable(), n_bad

    names = [
        name
        for pattern in _PATTERNS
        for name in pattern.groupindex
        if any(name in row for row in auxdata)
    ]
    out = QTable()
    for name in dict.fromkeys(names):
        out[name] = [row.get(name, np.nan) for row in auxdata] * _UNITS[name]
    out["time"] = Time([row["time"] for row in auxdata])
    return out, n_bad


_THERM = "{t}  temp_set   25.00 deg_C tmp   25.01 deg_C pwr  -32.94 percent\n"


def _random_aux_file(path: Path, rng: np.random.Generator, n: int) -> list[Time]:
    """Write a random aux file mixing formats and malformed lines.

    Returns the times of the lines with a valid stamp, to use as range boundaries.
    """
    # Start just before a leap day, and cross into a new year.
    t = datetime(2020, 12, 30, 23, 0, 0)
    lines = []
    times = []
    for _ in range(n):
        t += timedelta(seconds=int(rng.integers(1, 3600 * 3)))
        stamp = t.strftime("%Y:%j:%H:%M:%S")
        kind = rng.choice(
            ["old", "new", "therm", "blank", "garbage", "bad_stamp", "bad_values"],
            p=[0.3, 0.3, 0.2, 0.05, 0.05, 0.05, 0.05],
        )
        rack, ambient, hum, frontend, lna = rng.uniform(100, 400, 5)
        old = (
            f"{stamp}  rack_temp  {rack:6.2f} Kelvin, ambient_temp  "
            f"{ambient:6.2f} Kelvin, ambient_hum  {hum / 10:6.2f} percent"
        )
        if kind == "old":
            line = old + "\n"
        elif kind == "new":
            line = (
                f"{old}, frontend  {frontend:6.2f} Kelvin, "
                f"rcv3_lna  {lna:6.2f} Kelvin\n"
            )
        elif kind == "therm":
            line = _THERM.format(t=stamp)
        elif kind == "blank":
            line = "   \n"
        elif kind == "garbage":
            line = "not a time stamp at all\n"
        elif kind == "bad_stamp":
            line = stamp.replace(":", "-", 1) + "  rack_temp  306.64 Kelvin\n"
        else:
            line = f"{stamp}  rack_temp  garbage\n"

        lines.append(line)
        if kind not in ("blank", "garbage", "bad_stamp"):
            times.append(Time.strptime(stamp, "%Y:%j:%H:%M:%S"))
    path.write_text("".join(lines))
    return times


def _assert_same_read(fl: Path, start: Time, end: Time):
    """Assert that _read_aux_file gives exactly the reference result."""
    try:
        expected, n_bad = _reference_read_aux_file(fl, start, end)
    except ValueError:
        with pytest.raises(ValueError, match="No lines of the aux file"):
            _read_aux_file(fl, start, end=end)
        return

    with warnings.catch_warnings(record=True) as record:
        warnings.simplefilter("always")
        out = _read_aux_file(fl, start, end=end)

    messages = [str(w.message) for w in record]
    if n_bad:
        assert messages == [
            f"Skipped {n_bad} blank or malformed lines in the aux file {fl}."
        ]
    else:
        assert messages == []

    assert out.colnames == expected.colnames
    for name in expected.colnames:
        if name == "time":
            assert out[name].scale == expected[name].scale
            assert out[name].format == expected[name].format
            np.testing.assert_array_equal(out[name].jd1, expected[name].jd1)
            np.testing.assert_array_equal(out[name].jd2, expected[name].jd2)
        else:
            assert out[name].unit == expected[name].unit
            np.testing.assert_array_equal(out[name].value, expected[name].value)


@pytest.mark.parametrize("seed", range(4))
def test_read_aux_file_matches_reference(tmp_path: Path, seed: int):
    rng = np.random.default_rng(seed)
    fl = tmp_path / "aux.txt"
    times = _random_aux_file(fl, rng, n=300)

    ranges = [
        (times[0], times[-1]),  # the whole file but the last line
        (times[0] - 1 * un.day, times[-1] + 1 * un.day),  # the whole file
        (times[-1] + 1 * un.s, times[-1] + 1 * un.day),  # after the file
        (times[0] - 3 * un.day, times[0]),  # before the file
    ]
    for _ in range(8):
        i, j = np.sort(rng.integers(0, len(times), 2))
        ranges.extend([
            (times[i], times[j]),  # exactly on lines
            # Sub-second, and in another time scale.
            (times[i] + 0.5 * un.s, (times[j] - 0.5 * un.s).tt),
            # The default end: the rest of the day.
            (times[i], Time(_get_end_time(times[i]))),
        ])

    for start, end in ranges:
        _assert_same_read(fl, start, end)


def test_read_aux_file_non_canonical_stamps(tmp_path: Path):
    """Unusual stamps that strptime accepts are read as strptime reads them."""
    fl = tmp_path / "aux.txt"
    fl.write_text(
        _NEW.format(t="2016:366:23:55:00")  # leap day
        + _NEW.format(t="2016:366:23:59:60")  # leap second
        + _NEW.format(t="2017:001:00:05:00")
        + _NEW.format(t="٢٠١٧:001:00:10:00")  # non-ASCII digits
        + _NEW.format(t="2017:001:00:15:0x")  # malformed
        + _NEW.format(t="2017:366:00:05:00")  # day 366 of a non-leap year
    )
    for start, end in [
        (Time("2016:365:00:00:00"), Time("2018:002:00:00:00")),
        (Time("2016:366:23:59:60"), Time("2017:001:00:10:00")),
        (Time("2017:001:00:00:00"), Time("2017:001:01:00:00")),
        (Time("2017:365:00:00:00"), Time("2018:001:01:00:00")),
    ]:
        _assert_same_read(fl, start, end)


def test_read_aux_file_out_of_order(tmp_path: Path):
    """Lines out of order are read as they would be line by line."""
    fl = tmp_path / "aux.txt"
    fl.write_text(
        _NEW.format(t="2020:107:11:00:01")
        + _NEW.format(t="2020:105:11:05:01")  # before the start: skipped
        + _NEW.format(t="2020:107:11:10:01")
        + "\n"
        + _NEW.format(t="2020:107:13:00:00")  # past the end: stops reading
        + "\n"
        + _NEW.format(t="2020:107:11:20:01")
    )
    _assert_same_read(fl, Time("2020:107:11:00:00"), Time("2020:107:12:00:00"))
    _assert_same_read(fl, Time("2020:107:11:00:00"), Time("2020:109:00:00:00"))
    # An end before the start: lines are skipped until one is after the start.
    _assert_same_read(fl, Time("2020:107:11:15:00"), Time("2020:103:00:00:00"))
