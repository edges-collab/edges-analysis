"""Functions for reading thermistor files."""

import contextlib
from io import StringIO
from pathlib import Path

import numpy as np
from astropy import units
from astropy.table import QTable
from astropy.time import Time

from .. import types as tp


def read_new_style_csv(path: tp.PathLike) -> QTable:
    """Read a new-style CSV file as a thermistor."""
    data = np.genfromtxt(
        path,
        skip_header=1,
        delimiter=",",
        dtype=np.dtype([
            ("date", "S10"),
            ("time", "S8"),
            ("lna_voltage", float),
            ("lna_resistance", float),
            ("lna_temp", float),
            ("sp4t_voltage", float),
            ("sp4t_resistance", float),
            ("sp4t_temp", float),
            ("load_voltage", float),
            ("load_resistance", float),
            ("load_temp", float),
            ("room_temp", float),
        ]),
    )

    times = data["time"]
    dates = data["date"]
    datetimes = Time(
        [
            f"{d.decode('UTF-8')}:{t.decode('UTF-8')}"
            for d, t in zip(dates, times, strict=False)
        ],
        format="iso_custom",
    )

    # Add units
    qtable = {
        "times": datetimes,
    }
    units_dict = {
        "voltage": units.volt,
        "resistance": units.ohm,
        "temp": units.deg_C,
    }

    for name in data.dtype.names:
        if name in ("date", "time"):
            continue
        qtable[name] = data[name] * units_dict[name.split("_")[-1]]
    return QTable(qtable)


def _read_old_style_csv_header(path: tp.PathLike) -> tuple[dict, int]:
    with Path(path).open("r", errors="ignore") as fl:
        if not fl.readline().startswith("FLUKE"):
            return {}, 0

        done = False
        out = {}
        nheader_lines = 0
        while not done:
            line = fl.readline()

            if line.startswith(("Start Time,", "Max Time,")):
                names = line.split(",")

                next_line = fl.readline()
                nheader_lines += 1
                values = next_line.split(",")

                out |= dict(zip(names, values, strict=False))

            if line.startswith("1,") or line == "":
                done = True

            nheader_lines += 1

    return out, nheader_lines


def read_old_style_csv(path: tp.PathLike) -> QTable:
    """Read an old=style thermistor CSV."""
    # Weirdly, some old-style files use KOhm, and some just use Ohm.

    # These files have bad encoding, which we can ignore. This means we have to
    # read in the whole thing as text first (while ignoring errors) and construct
    # a StringIO object to pass to genfromtxt.
    header, nheader_lines = _read_old_style_csv_header(path)
    nlines = int(header["Total readings"])

    with Path(path).open("r", errors="ignore") as fl:
        # Get past the header.
        for _ in range(nheader_lines):
            next(fl)

        s = StringIO("".join([next(fl) for i in range(nlines - 1)]))

        # Determine whether the file is in KOhm

        def float_from_kohm(x: bytes | str):
            with contextlib.suppress(AttributeError):
                x = x.decode("utf-8")

            kohm = "KOhm" in x
            y = float(x.split(" ")[0])
            return y * 1000 if kohm else y

        data = np.genfromtxt(
            s,
            delimiter=",",
            dtype=np.dtype([
                ("reading_num", int),
                ("sample_resistance", float),
                ("start_time", "S20"),
                ("duration", "S9"),
                ("max_time", "S20"),
                ("max_resistance", float),
                ("load_resistance", float),
                ("min_time", "S20"),
                ("min_resistance", float),
                ("description", "S20"),
                ("end_time", "S22"),
            ]),
            converters={
                1: float_from_kohm,
                5: float_from_kohm,
                6: float_from_kohm,
                8: float_from_kohm,
            },
        )
    table = QTable(data)
    for name in table.columns:
        if name.endswith("_resistance"):
            table[name] <<= units.ohm

    table["times"] = Time(table["start_time"], format="iso_custom")

    return table


def read_thermistor_csv(path: tp.PathLike) -> QTable:
    """Read a CSV as a thermistor object."""
    with Path(path).open("r", errors="ignore") as fl:
        if fl.readline().startswith("FLUKE"):
            return read_old_style_csv(path)
        return read_new_style_csv(path)
