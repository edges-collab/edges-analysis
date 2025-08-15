"""Functions for reading temperature log files (EDGES-3 format .tmp and .log)."""

import re
import warnings
from datetime import UTC, datetime, timedelta
from itertools import pairwise
from pathlib import Path
from typing import Any, Literal

import numpy as np
from astropy import units as un
from astropy.table import QTable
from astropy.time import Time


def read_temperature_log_entry(lines: list[str]) -> dict[str, Any]:
    """Read a single entry from a thermistor log file."""
    if len(lines) != 10:
        raise ValueError("Expected 10 lines for a temperature log entry")

    # first line is the date
    year, day, hour = (int(x) for x in lines[0].split("_"))

    # second line is the time
    _, _, _, timestamp, _, year2 = lines[1].split(" ")

    hh, mm, ss = (int(x) for x in timestamp.split(":"))

    if year != int(year2):
        raise ValueError(
            f"Year mismatch in temperature log entry. Got {year} and {year2}"
        )
    if hour != hh:
        raise ValueError(f"Hour mismatch in temperature log entry. Got {hour} and {hh}")

    # don't care about the third line
    date_object = datetime(year, 1, 1, hh, mm, ss, tzinfo=UTC) + timedelta(days=day - 1)
    try:
        return {
            "time": Time(date_object),
            "front_end_temperature": float(lines[3].split(" ")[1]) * un.deg_C,  # 100
            "amb_load_temperature": float(lines[4].split(" ")[1]) * un.deg_C,  # 101
            "hot_load_temperature": float(lines[5].split(" ")[1]) * un.deg_C,  # 102
            "inner_box_temperature": float(lines[6].split(" ")[1]) * un.deg_C,  # 103
            "thermal_control": float(lines[7].split(" ")[1]),  # 106
            "battery_voltage": float(lines[8].split(" ")[1]),  # 150
            "pr59_current": float(lines[9].split(" ")[1]),  # 152
        }
    except (ValueError, IndexError) as e:
        linestr = "".join(lines)
        raise ValueError(f"Error parsing temperature log entry:\n{linestr}") from e


def read_temperature_log(logfile: Path | str) -> QTable:
    """Read a full temperature log file."""
    pattern = re.compile(r"(\d{4})_(\d{3})_(\d{2})")
    with Path(logfile).open("r") as fl:
        all_data = _extracted_from_read_temperature_log_(fl, pattern)
    out = QTable(all_data)

    # Convert temperatures to Kelvin
    for col in out.columns:
        if col.endswith("temperature"):
            out[col] = out[col].to(un.K, equivalencies=un.temperature())

    return out


# TODO Rename this here and in `read_temperature_log`
def _extracted_from_read_temperature_log_(fl, pattern):
    lines = fl.readlines()

    chunk_indices = [
        i for i, line in enumerate(lines) if re.match(pattern, line) is not None
    ]
    chunk_indices.append(len(lines))

    # Only take chunks that are length 10
    chunk_indices = list(pairwise(chunk_indices))
    chunk_indices = [c for c in chunk_indices if c[1] - c[0] == 10]

    result = []
    for c in chunk_indices:
        try:
            result.append(read_temperature_log_entry(lines[c[0] : c[1]]))
        except ValueError as e:
            warnings.warn(str(e), stacklevel=2)

    return result


def get_mean_temperature(
    temperature_table: QTable,
    start_time: Time | None = None,
    end_time: Time | None = None,
    load: Literal["box", "amb", "hot", "open", "short"] = "box",
):
    """Get the mean temperature for a particular load from the temperature table."""
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
        return temperature_table["hot_load_temperature"].mean()
    if load in ("amb", "ambient", "open", "short", "box"):
        return temperature_table["amb_load_temperature"].mean()
    raise ValueError(f"Unknown load {load}")
