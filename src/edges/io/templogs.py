"""Functions for reading EDGES-3 temperature logs (``.log`` and ``.tmp`` files).

Both formats consist of lines of the form ``<code> <value>``, where the numeric code
identifies the quantity (see :data:`CODE_NAMES`). The ``.log`` files group these lines
into entries, each preceded by two header lines::

    2023_069_00
    Fri Mar 10 00:05:07 UTC 2023
    0 +3.500000e+01
    100 +3.505190e+01
    ...

while ``.tmp`` files are a single snapshot of ``<code> <value>`` lines with no header
(the time is taken from the filename, e.g. ``2023_210_03.tmp``).

The files written in the field are not always clean: lines can be garbled or missing,
entries can be merged, and log files are sometimes cumulative copies of each other. The
readers here parse each line by its code (so a missing or reordered line never shifts
values into the wrong field), skip anything malformed, and emit a single summary
warning per file describing what was skipped. Specifically:

- Files are decoded as latin-1, so junk bytes never raise; they just produce
  malformed lines.
- A line is accepted only if it is ``<code> <value>`` with a decimal point in the
  value. Lines with unknown codes are skipped.
- An entry missing some codes is kept, with NaN for the missing values.
- If the same code appears more than once for the same time (within an entry, or
  across overlapping log files) with *different* values, the value is set to NaN,
  since there is no way to tell which is right. Identical repeats are merged.
- A ``.log`` file that does not end with a newline has its last line dropped, as it
  may have been truncated while being written.

References
----------
EDGES memo 300, Table 1:
https://www.haystack.mit.edu/wp-content/uploads/2020/07/memo_EDGES_300.pdf
"""

import re
import warnings
from collections import Counter
from collections.abc import Sequence
from datetime import UTC, datetime, timedelta
from itertools import pairwise
from pathlib import Path
from typing import Any, Literal

import numpy as np
from astropy import units as un
from astropy.table import QTable
from astropy.time import Time

#: Mapping of the numeric codes in temperature logs to column names. The codes are the
#: PR59 thermal controller registers listed in Table 1 of EDGES memo 300: 100 front-end
#: box temperature, 101 ambient load temperature, 102 hot load temperature, 103
#: temperature of the PR59 in the inner box, 106 thermal control output (%), 150
#: battery voltage and 152 PR59 current. The thermal control output is signed (the
#: MRO logs range from about -70% to +80%).
CODE_NAMES: dict[int, str] = {
    100: "front_end_temperature",
    101: "amb_load_temperature",
    102: "hot_load_temperature",
    103: "inner_box_temperature",
    106: "thermal_control",
    150: "battery_voltage",
    152: "pr59_current",
}

#: Units of the values recorded under each code. Memo 300 gives the quantity but not
#: the scale for 150 and 152; volts and amps match the recorded values (~13 and ~0-2).
CODE_UNITS: dict[int, un.UnitBase] = {
    100: un.deg_C,
    101: un.deg_C,
    102: un.deg_C,
    103: un.deg_C,
    106: un.percent,
    150: un.V,
    152: un.A,
}

# Temperatures are converted to Kelvin in tables; everything else keeps its unit.
_OUTPUT_UNITS = {code: un.K for code, unit in CODE_UNITS.items() if unit == un.deg_C}

#: Codes that appear in the logs but are deliberately not read. Code 0 is not listed
#: in memo 300; it is constant within a file (e.g. 25.0 or 35.0) and is probably a
#: temperature set-point.
IGNORED_CODES: frozenset[int] = frozenset({0})

_HEADER = re.compile(r"^(\d{4})_(\d{3})_(\d{2})$")
# .tmp files are named YYYY_DDD_HH.tmp, usually with a load suffix (e.g. _ant, _hot).
_TMP_NAME = re.compile(r"^(\d{4})_(\d{3})_(\d{2})(?:_\w+)?$")
# Every genuine value has a decimal point (e.g. +3.505190e+01, +12.88), and printf
# exponents have two or three digits. Being strict here rejects lines that were run
# together, such as "100 1101" or "100 +3.505190e+01101".
_CODE_LINE = re.compile(r"^(\d+)\s+([+-]?(?:\d+\.\d*|\.\d+)(?:[eE][+-]?\d{2,3})?)$")


def _read_lines(path: Path, issues: Counter, *, drop_unterminated: bool) -> list[str]:
    """Read the lines of a log file, tolerating non-UTF-8 junk.

    Bytes are decoded as latin-1, which never fails, so junk bytes simply produce
    lines that fail the line patterns. Lines are split on newlines only (not on the
    other characters :meth:`str.splitlines` treats as line breaks).

    If ``drop_unterminated`` is True and the file does not end with a newline, its
    last line is dropped, since the file was probably copied while being written and
    the line may be truncated (e.g. ``103 +3`` instead of ``103 +3.124716e+01``).
    """
    lines = path.read_bytes().decode("latin-1").replace("\r\n", "\n").split("\n")
    last = lines.pop()  # empty if the file ends with a newline
    if last.strip():
        if drop_unterminated:
            issues["truncated last lines"] += 1
        else:
            lines.append(last)
    return lines


def _parse_code_lines(lines: Sequence[str], issues: Counter) -> dict[int, float]:
    """Parse ``<code> <value>`` lines into a code->value dict.

    Malformed lines and unknown codes are skipped. A code that appears more than once
    with different values is set to NaN, since there is no way to know which is right.
    Problems are tallied in ``issues``.
    """
    values: dict[int, float] = {}
    for line in lines:
        line = line.strip()
        if not line:
            continue

        match = _CODE_LINE.match(line)
        if match is None:
            issues["malformed lines"] += 1
            continue

        code, value = int(match[1]), float(match[2])
        if not np.isfinite(value):
            issues["malformed lines"] += 1
            continue
        if code in IGNORED_CODES:
            continue
        if code not in CODE_NAMES:
            issues[f"lines with unknown code {code}"] += 1
            continue

        old = values.get(code)
        if old is not None and old != value:  # NaN (already conflicted) stays NaN
            if not np.isnan(old):
                issues["conflicting duplicate values"] += 1
            value = np.nan
        values[code] = value
    return values


def _to_record(values: dict[int, float]) -> dict[str, Any]:
    """Convert a code->value dict to named fields, with NaN for missing codes."""
    record = {}
    for code, name in CODE_NAMES.items():
        value = values.get(code, np.nan)
        record[name] = value * CODE_UNITS[code]
    return record


def _to_kelvin(record: dict[str, Any]) -> dict[str, Any]:
    return {
        k: v.to(un.K, equivalencies=un.temperature())
        if isinstance(v, un.Quantity) and v.unit == un.deg_C
        else v
        for k, v in record.items()
    }


def _warn_issues(path: Path, issues: Counter, stacklevel: int = 3):
    if issues := +issues:  # drop zero counts
        summary = ", ".join(f"{n} {what}" for what, n in sorted(issues.items()))
        warnings.warn(
            f"Skipped bad data in temperature log {path}: {summary}",
            stacklevel=stacklevel,
        )


def _parse_entry_time(lines: Sequence[str]) -> datetime:
    """Parse and cross-check the two header lines of a ``.log`` entry."""
    if len(lines) < 2:
        raise ValueError("Temperature log entry has no timestamp line")

    header = _HEADER.match(lines[0].strip())
    if header is None:
        raise ValueError(f"Bad temperature log entry header: {lines[0]!r}")
    year, day, hour = (int(x) for x in header.groups())

    try:
        # `date` pads single-digit days with a space, so normalise whitespace.
        timestamp = datetime.strptime(
            " ".join(lines[1].split()), "%a %b %d %H:%M:%S %Z %Y"
        ).replace(tzinfo=UTC)
    except ValueError as e:
        raise ValueError(f"Bad temperature log timestamp: {lines[1]!r}") from e

    expected = datetime(year, 1, 1, hour, tzinfo=UTC) + timedelta(days=day - 1)
    if (timestamp.date(), timestamp.hour) != (expected.date(), expected.hour):
        raise ValueError(
            f"Temperature log header {lines[0].strip()!r} does not match timestamp "
            f"{lines[1].strip()!r}"
        )
    return timestamp


def read_temperature_log_entry(
    lines: list[str], issues: Counter | None = None
) -> dict[str, Any]:
    """Read a single entry from a temperature ``.log`` file.

    Parameters
    ----------
    lines
        The lines of the entry: the ``YYYY_DDD_HH`` header, the ``date``-style
        timestamp, and then any number of ``<code> <value>`` lines.
    issues
        If given, problems with individual lines (malformed lines, unknown codes,
        conflicting duplicates) are tallied here instead of being silently dropped.

    Returns
    -------
    dict
        The ``time`` of the entry and one field per code in :data:`CODE_NAMES`,
        with the units in :data:`CODE_UNITS` (so temperatures are in degrees
        Celsius). Fields whose code is missing from the entry are NaN.

    Raises
    ------
    ValueError
        If the header or timestamp lines are malformed or inconsistent.
    """
    timestamp = _parse_entry_time(lines)
    values = _parse_code_lines(lines[2:], issues if issues is not None else Counter())
    return {"time": Time(timestamp), **_to_record(values)}


def _read_log_values(
    path: Path, issues: Counter
) -> list[tuple[datetime, dict[int, float]]]:
    lines = _read_lines(path, issues, drop_unterminated=True)

    starts = [i for i, line in enumerate(lines) if _HEADER.match(line.strip())]
    if n_junk := sum(bool(x.strip()) for x in lines[: starts[0] if starts else None]):
        issues["malformed lines"] += n_junk
    starts.append(len(lines))

    entries = []
    for start, end in pairwise(starts):
        block = lines[start:end]
        try:
            timestamp = _parse_entry_time(block)
        except ValueError:
            issues["entries with a bad header"] += 1
            continue
        values = _parse_code_lines(block[2:], issues)
        if not values:
            issues["entries with no valid data"] += 1
            continue
        entries.append((timestamp, values))
    return entries


def read_temperature_log(
    logfile: Path | str | Sequence[Path | str],
) -> QTable:
    """Read one or more temperature ``.log`` files into a single table.

    Parameters
    ----------
    logfile
        A path, or a sequence of paths, to temperature log files. Log files can overlap
        in time (e.g. cumulative copies of the same log) and need not be time-ordered.

    Returns
    -------
    QTable
        A table sorted by ``time`` with one row per unique timestamp and one column per
        code in :data:`CODE_NAMES`, with the units in :data:`CODE_UNITS` except that
        temperatures are converted to Kelvin. Values that are
        missing from an entry are NaN. Where the same (time, code) appears more than
        once (e.g. in overlapping logs), identical values are merged, and conflicting
        values are set to NaN with a warning.

    Raises
    ------
    ValueError
        If no valid entries are found in any of the files.
    """
    paths = [logfile] if isinstance(logfile, str | Path) else list(logfile)

    merged: dict[datetime, dict[int, float]] = {}
    n_conflicts = 0
    for path in map(Path, paths):
        issues = Counter()
        for timestamp, values in _read_log_values(path, issues):
            existing = merged.setdefault(timestamp, {})
            for code, value in values.items():
                old = existing.get(code)
                if old is not None and old != value:
                    if not np.isnan(old):
                        n_conflicts += 1
                    value = np.nan
                existing[code] = value
        _warn_issues(path, issues)

    if n_conflicts:
        warnings.warn(
            f"{n_conflicts} (time, code) pairs had conflicting values across the "
            "temperature logs; these are set to NaN.",
            stacklevel=2,
        )

    if not merged:
        raise ValueError(f"No valid temperature log entries found in {paths}")

    times = sorted(merged)
    out = QTable({"time": Time(times)})
    for code, name in CODE_NAMES.items():
        col = np.array([merged[t].get(code, np.nan) for t in times])
        out[name] = (col * CODE_UNITS[code]).to(
            _OUTPUT_UNITS.get(code, CODE_UNITS[code]),
            equivalencies=un.temperature(),
        )
    return out


def read_tmp_file(path: Path | str) -> dict[str, Any]:
    """Read a ``.tmp`` temperature snapshot file.

    These files contain ``<code> <value>`` lines with no header. The set of codes
    varies between files (e.g. some lack 106 and 150); missing codes are NaN.

    Parameters
    ----------
    path
        The path to the ``.tmp`` file.

    Returns
    -------
    dict
        One field per code in :data:`CODE_NAMES` (temperatures in Kelvin, for
        consistency with :func:`read_temperature_log`), plus ``time``. The time is
        read from a ``YYYY_DDD_HH`` or ``YYYY_DDD_HH_<load>`` filename (so only has
        hour resolution) and is None if the filename is not of that form.
    """
    path = Path(path)
    issues = Counter()
    lines = _read_lines(path, issues, drop_unterminated=False)
    record = _to_kelvin(_to_record(_parse_code_lines(lines, issues)))
    _warn_issues(path, issues, stacklevel=2)

    time = None
    if match := _TMP_NAME.match(path.stem):
        year, day, hour = (int(x) for x in match.groups())
        time = Time(datetime(year, 1, 1, hour, tzinfo=UTC) + timedelta(days=day - 1))
    return {"time": time, **record}


def get_mean_temperature(
    temperature_table: QTable,
    start_time: Time | None = None,
    end_time: Time | None = None,
    load: Literal["box", "amb", "hot", "open", "short"] = "box",
):
    """Get the mean temperature for a particular load from the temperature table.

    NaN values (missing or corrupted readings) are ignored.
    """
    if start_time is not None or end_time is not None:
        if start_time is None:
            start_time = temperature_table["time"].min()
        if end_time is None:
            end_time = temperature_table["time"].max()

        mask = (temperature_table["time"] >= start_time) & (
            temperature_table["time"] <= end_time
        )

        if not np.any(mask):
            raise ValueError(
                f"No data found between {start_time} and {end_time} in temperature "
                f"table"
            )

        temperature_table = temperature_table[mask]

    if load in ("hot", "hot_load"):
        return np.nanmean(temperature_table["hot_load_temperature"])
    if load in ("amb", "ambient", "open", "short", "box"):
        return np.nanmean(temperature_table["amb_load_temperature"])
    raise ValueError(f"Unknown load {load}")
