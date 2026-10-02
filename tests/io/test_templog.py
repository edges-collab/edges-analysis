import warnings
from collections import Counter

import numpy as np
import pytest
from astropy import units as un
from astropy.table import QTable
from astropy.time import Time

from edges.io import CalObsDefEDGES3, templogs


@pytest.fixture(scope="module")
def templog_table(smallcaldef_edges3: CalObsDefEDGES3) -> QTable:
    with pytest.warns(UserWarning, match="3 malformed lines"):
        # There are two corrupted entries, and this should not make the reader break.
        return templogs.read_temperature_log(smallcaldef_edges3.ambient.templog)


def _entry(header: str, timestamp: str, codes: dict[int, float | str]) -> str:
    lines = [header, timestamp] + [f"{k} {v}" for k, v in codes.items()]
    return "\n".join(lines) + "\n"


GOOD_CODES = {
    0: "+3.500000e+01",
    100: "+3.505190e+01",
    101: "+3.418057e+01",
    102: "+1.228620e+02",
    103: "+3.494265e+01",
    106: "+0.000000e+00",
    150: "+13.11",
    152: "+0.593",
}


def test_temperature_read(templog_table: QTable):
    assert "time" in templog_table.columns
    assert "hot_load_temperature" in templog_table.columns
    assert "amb_load_temperature" in templog_table.columns
    assert "front_end_temperature" in templog_table.columns
    assert "inner_box_temperature" in templog_table.columns
    assert "thermal_control" in templog_table.columns
    assert "battery_voltage" in templog_table.columns
    assert "pr59_current" in templog_table.columns

    assert all(
        templog_table[col].unit == un.K
        for col in templog_table.columns
        if col.endswith("temperature")
    )
    assert templog_table["battery_voltage"].unit == un.V
    assert templog_table["pr59_current"].unit == un.A
    assert templog_table["thermal_control"].unit == un.percent


def test_corrupted_entries_are_partial(templog_table: QTable):
    # 827 clean entries, plus two garbled ones that keep their readable fields.
    assert len(templog_table) == 829
    assert np.all(np.diff(templog_table["time"].unix) > 0)
    assert not np.any(np.isnan(templog_table["front_end_temperature"]))
    assert np.sum(np.isnan(templog_table["inner_box_temperature"])) == 2


def test_get_mean_temperature(templog_table: QTable):
    mean_temp = templogs.get_mean_temperature(templog_table, load="amb")
    assert mean_temp.unit == un.K
    assert mean_temp.value == pytest.approx(300, abs=15)

    mean_temp0 = templogs.get_mean_temperature(
        templog_table, load="amb", start_time=templog_table["time"].min()
    )
    assert mean_temp0 == mean_temp

    mean_temp0 = templogs.get_mean_temperature(
        templog_table, load="open", end_time=templog_table["time"].max()
    )
    assert mean_temp0 == mean_temp

    mean_temp0 = templogs.get_mean_temperature(templog_table, load="hot")
    assert mean_temp0 > mean_temp

    with pytest.raises(ValueError, match="No data found"):
        templogs.get_mean_temperature(
            templog_table, start_time=Time("2000-01-01"), end_time=Time("2000-01-02")
        )

    with pytest.raises(ValueError, match="Unknown load fake"):
        templogs.get_mean_temperature(templog_table, load="fake")


def test_entry_parsed_by_code():
    # Reversed line order: values must still go to the right fields.
    codes = dict(reversed(GOOD_CODES.items()))
    entry = templogs.read_temperature_log_entry(
        _entry("2023_069_00", "Fri Mar 10 00:05:07 UTC 2023", codes).splitlines()
    )
    assert entry["time"] == Time("2023-03-10T00:05:07", scale="utc")
    assert entry["front_end_temperature"].to_value(un.deg_C) == pytest.approx(35.0519)
    assert entry["hot_load_temperature"].to_value(un.deg_C) == pytest.approx(122.862)
    assert entry["battery_voltage"].to_value(un.V) == pytest.approx(13.11)
    assert entry["pr59_current"].to_value(un.A) == pytest.approx(0.593)


def test_entry_missing_and_unknown_codes():
    codes = {k: v for k, v in GOOD_CODES.items() if k != 106} | {999: "+1.0"}
    issues = Counter()
    entry = templogs.read_temperature_log_entry(
        _entry("2023_069_00", "Fri Mar 10 00:05:07 UTC 2023", codes).splitlines(),
        issues=issues,
    )
    assert np.isnan(entry["thermal_control"])
    assert entry["inner_box_temperature"].to_value(un.deg_C) == pytest.approx(34.94265)
    assert issues == {"lines with unknown code 999": 1}


def test_entry_conflicting_duplicate():
    lines = _entry(
        "2023_069_00", "Fri Mar 10 00:05:07 UTC 2023", GOOD_CODES
    ).splitlines()
    issues = Counter()
    entry = templogs.read_temperature_log_entry(
        # the third occurrence of 101 must not overwrite the conflict
        [*lines, "101 +2.0e+01", "101 +3.418057e+01", "102 +1.228620e+02"],
        issues=issues,
    )
    assert np.isnan(entry["amb_load_temperature"])
    # an identical duplicate is fine
    assert entry["hot_load_temperature"].to_value(un.deg_C) == pytest.approx(122.862)
    assert issues == {"conflicting duplicate values": 1}


def test_entry_padded_day():
    # `date` pads single-digit days with a space.
    entry = templogs.read_temperature_log_entry(
        _entry("2023_062_00", "Fri Mar  3 00:05:07 UTC 2023", GOOD_CODES).splitlines()
    )
    assert entry["time"] == Time("2023-03-03T00:05:07", scale="utc")


@pytest.mark.parametrize(
    ("header", "timestamp"),
    [
        ("2023_069_01", "Fri Mar 10 00:05:07 UTC 2023"),  # hour mismatch
        ("2023_070_00", "Fri Mar 10 00:05:07 UTC 2023"),  # day mismatch
        ("2023_069_00", "sent"),  # garbled timestamp
        ("2023-069-00", "Fri Mar 10 00:05:07 UTC 2023"),  # garbled header
    ],
)
def test_entry_bad_header(header, timestamp):
    with pytest.raises(ValueError, match=r"(?i)temperature log"):
        templogs.read_temperature_log_entry(
            _entry(header, timestamp, GOOD_CODES).splitlines()
        )


def test_read_log_skips_bad_entries(tmp_path):
    log = tmp_path / "temperature.log"
    log.write_text(
        "leading junk\n"
        + _entry("2023_069_00", "Fri Mar 10 00:05:07 UTC 2023", GOOD_CODES)
        + _entry("2023_069_01", "Fri Mar 10 00:10:19 UTC 2023", GOOD_CODES)
        + _entry("2023_069_00", "Fri Mar 10 00:15:32 UTC 2023", {0: "+3.5e+01", 1: "x"})
        + _entry("2023_069_00", "Fri Mar 10 00:20:44 UTC 2023", GOOD_CODES)
        + "\n"  # blank lines are ignored
        + "2023_069_00\n"  # truncated entry at the end of the file
    )
    with pytest.warns(UserWarning, match="Skipped bad data") as record:
        table = templogs.read_temperature_log(log)
    assert len(record) == 1
    msg = str(record[0].message)
    assert "2 entries with a bad header" in msg
    assert "1 entries with no valid data" in msg
    assert "2 malformed lines" in msg
    assert len(table) == 2


def test_read_multiple_logs(tmp_path):
    t1 = "Fri Mar 10 00:05:07 UTC 2023"
    t2 = "Fri Mar 10 00:10:19 UTC 2023"
    t3 = "Fri Mar 10 00:15:32 UTC 2023"
    first = tmp_path / "a.log"
    # Not time-ordered.
    first.write_text(
        _entry("2023_069_00", t2, GOOD_CODES) + _entry("2023_069_00", t1, GOOD_CODES)
    )
    # Cumulative copy, overlapping the first, with one partial entry that is filled in
    # from the other file.
    partial = {k: v for k, v in GOOD_CODES.items() if k != 150}
    second = tmp_path / "b.log"
    second.write_text(
        _entry("2023_069_00", t1, partial)
        + _entry("2023_069_00", t2, GOOD_CODES)
        + _entry("2023_069_00", t3, GOOD_CODES)
    )

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        table = templogs.read_temperature_log([first, str(second)])
    assert len(table) == 3
    assert np.all(np.diff(table["time"].unix) > 0)
    assert not np.any(np.isnan(table["battery_voltage"]))


def test_read_multiple_logs_conflict(tmp_path):
    t1 = "Fri Mar 10 00:05:07 UTC 2023"
    first = tmp_path / "a.log"
    first.write_text(_entry("2023_069_00", t1, GOOD_CODES))
    second = tmp_path / "b.log"
    second.write_text(_entry("2023_069_00", t1, GOOD_CODES | {101: "+1.0"}))
    third = tmp_path / "c.log"
    third.write_text(_entry("2023_069_00", t1, GOOD_CODES))

    with pytest.warns(UserWarning, match="1 \\(time, code\\) pairs had conflicting"):
        table = templogs.read_temperature_log([first, second, third])
    assert len(table) == 1
    assert np.isnan(table["amb_load_temperature"][0])
    assert table["hot_load_temperature"][0].to_value(un.K) == pytest.approx(
        122.862 + 273.15
    )


def test_read_log_no_entries(tmp_path):
    log = tmp_path / "empty.log"
    log.write_text("")
    with pytest.raises(ValueError, match="No valid temperature log entries"):
        templogs.read_temperature_log(log)


def test_read_tmp_file_fixture(testdata_path):
    path = testdata_path / "alanmode/edges3-2023-210-raw/2023_210_03.tmp"
    out = templogs.read_tmp_file(path)
    assert out["time"] == Time("2023-07-29T03:00:00", scale="utc")
    assert out["hot_load_temperature"].unit == un.K
    assert out["hot_load_temperature"].to_value(un.K) == pytest.approx(
        118.5631 + 273.15
    )
    assert out["pr59_current"].to_value(un.A) == pytest.approx(0.818)
    assert np.isnan(out["thermal_control"])
    assert np.isnan(out["battery_voltage"])


def test_read_tmp_file_garbage(tmp_path):
    path = tmp_path / "snapshot.tmp"
    path.write_bytes(
        b"100 +3.005550e+01\r\n101 +2.726269e+01\r\n$$R102101\r\n"
        b"103 +2.934121e+01\r\n103 +2.934121e+01\r\n152 +0.818\r\n"
    )
    with pytest.warns(UserWarning, match="1 malformed lines"):
        out = templogs.read_tmp_file(path)
    assert out["time"] is None
    assert np.isnan(out["hot_load_temperature"])
    assert out["inner_box_temperature"].to_value(un.K) == pytest.approx(
        29.34121 + 273.15
    )
    assert out["pr59_current"].to_value(un.A) == pytest.approx(0.818)


def test_read_log_binary_junk(tmp_path):
    # Non-UTF-8 bytes must not crash the reader, or a multi-log read.
    bad = tmp_path / "bad.log"
    bad.write_bytes(
        _entry("2023_069_00", "Fri Mar 10 00:05:07 UTC 2023", GOOD_CODES).encode()
        + b"\xff\x9c\x00garbage\n"
        + _entry("2023_069_00", "Fri Mar 10 00:10:19 UTC 2023", GOOD_CODES).encode()
    )
    good = tmp_path / "good.log"
    good.write_text(_entry("2023_069_00", "Fri Mar 10 00:15:32 UTC 2023", GOOD_CODES))

    with pytest.warns(UserWarning, match="1 malformed lines"):
        table = templogs.read_temperature_log([bad, good])
    assert len(table) == 3
    assert not np.any(np.isnan(table["front_end_temperature"]))


@pytest.mark.parametrize(
    "line",
    [
        "100 1101",  # bare integer: two lines of output run together
        "100 +3.505190e+01101",  # exponent run into the next line's code
        "100 +1.0e400",  # overflows to inf
    ],
)
def test_entry_rejects_run_together_values(line):
    codes = {k: v for k, v in GOOD_CODES.items() if k != 100}
    issues = Counter()
    entry = templogs.read_temperature_log_entry(
        [
            *_entry("2023_069_00", "Fri Mar 10 00:05:07 UTC 2023", codes).splitlines(),
            line,
        ],
        issues=issues,
    )
    assert np.isnan(entry["front_end_temperature"])
    assert issues == {"malformed lines": 1}


def test_read_log_truncated_last_line(tmp_path):
    log = tmp_path / "temperature.log"
    text = _entry("2023_069_00", "Fri Mar 10 00:05:07 UTC 2023", GOOD_CODES) + _entry(
        "2023_069_00", "Fri Mar 10 00:10:19 UTC 2023", GOOD_CODES
    )
    # Copied while being written: ends part-way through a line with no newline.
    log.write_text(text[: text.rindex("103 ") + len("103 +3")])

    with pytest.warns(UserWarning, match="1 truncated last lines"):
        table = templogs.read_temperature_log(log)
    assert len(table) == 2
    assert np.isnan(table["inner_box_temperature"][1])
    assert table["hot_load_temperature"][1].to_value(un.K) == pytest.approx(
        122.862 + 273.15
    )


@pytest.mark.parametrize(
    "suffix", ["", "_ant", "_amb", "_hot", "_open", "_short", "_L"]
)
def test_read_tmp_file_load_suffix(tmp_path, suffix):
    path = tmp_path / f"2024_345_09{suffix}.tmp"
    path.write_text("100 +3.005550e+01\n152 +0.818")  # no final newline is fine
    out = templogs.read_tmp_file(path)
    assert out["time"] == Time("2024-12-10T09:00:00", scale="utc")
    assert out["pr59_current"].to_value(un.A) == pytest.approx(0.818)
