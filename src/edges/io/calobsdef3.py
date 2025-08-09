"""Methods for dealing with EDGES-3 files and structures."""

from __future__ import annotations

import contextlib
from pathlib import Path
from typing import Literal, Self, get_args

import attrs
from bidict import bidict

from .. import types as tp
from .calobsdef import Calkit, LoadS11, ReceiverS11, _list_of_path, _vld_path_exists

S11LoadName = Literal["amb", "ant", "hot", "open", "short", "lna"]
SpecLoadName = Literal["amb", "ant", "hot", "open", "short"]
CalLoadName = Literal["amb", "hot", "open", "short"]

LOADMAP = bidict({
    "amb": "ambient",
    "hot": "hot_load",
    "open": "open",
    "short": "short",
    "lna": "lna",
})


def _get_single_s1p_file(
    root: Path,
    year: int,
    day: int,
    label: str,
    hour: int | str = "first",
    allow_closest: int = 30,
) -> Path:
    if isinstance(hour, int):
        glob = f"{year:04d}_{day:03d}_{hour:02d}_{label}.s1p"
    else:
        glob = f"{year:04d}_{day:03d}_??_{label}.s1p"

    file_temp = sorted(root.glob(glob))

    if len(file_temp) == 0:
        if allow_closest:
            for dday in range(1, allow_closest):
                with contextlib.suppress(FileNotFoundError):
                    return _get_single_s1p_file(
                        root,
                        year,
                        day + dday,
                        label=label,
                        hour=hour,
                        allow_closest=False,
                    )
                with contextlib.suppress(FileNotFoundError):
                    return _get_single_s1p_file(
                        root,
                        year,
                        day - dday,
                        label=label,
                        hour=hour,
                        allow_closest=False,
                    )
        raise FileNotFoundError(f"No s1p files found in {root} with glob {glob}")
    if len(file_temp) > 1:
        if hour == "first":
            return file_temp[0]
        if hour == "last":
            return file_temp[-1]
        raise OSError(f"More than one file found for {year}, {day}, {label}")

    return file_temp[0]


def get_s1p_files(
    load: S11LoadName,
    year: int,
    day: int,
    root_dir: Path | str,
    hour: int | str = "first",
    allow_closest_s11_within: int = 5,
) -> dict[str, Path]:
    """Take the load and return a list of .s1p files for that load."""
    root_dir = Path(root_dir)

    files = {
        "input": _get_single_s1p_file(
            root_dir, year, day, load, hour, allow_closest_s11_within
        )
    }

    for name, label in {"open": "O", "short": "S", "match": "L"}.items():
        if load == "lna":
            label = f"{load}_{label}"
        files[name] = _get_single_s1p_file(
            root_dir, year, day, label, hour, allow_closest_s11_within
        )

    return files


class CalkitEdges3(Calkit):
    @classmethod
    def from_standard_layout(
        cls,
        root_dir: Path | str,
        load: S11LoadName,
        year: int,
        day: int,
        hour: int | str = "first",
        allow_closest_s11_within: int = 5,
    ) -> Self:
        files = get_s1p_files(
            load=load,
            year=year,
            day=day,
            root_dir=root_dir,
            hour=hour,
            allow_closest_s11_within=allow_closest_s11_within,
        )
        del files["input"]
        return cls(**files)


def get_spectrum_files(
    load: Literal["amb", "hot", "short", "open"],
    root: Path,
    year: int,
    day: int,
    fmt: Literal["acq", "gsh5"] = "acq",
) -> Path:
    """Get the ACQ files for a particular load."""
    d = root / "mro" / load / str(year)
    glob = f"{year}_{day:03d}_??_??_??_{load}.{fmt}"
    files = sorted(d.glob(glob))

    if not files:
        raise FileNotFoundError(f"No files found in {d} for {glob}")
    if len(files) > 1:
        raise OSError(f"More than one file found in {d} for {glob}")

    return files[0]


@attrs.define(frozen=True, kw_only=True)
class LoadDefEDGES3:
    name: str = attrs.field()
    
    s11: LoadS11 = attrs.field()
    spectra: list[Path] = attrs.field(converter=_list_of_path)
    templog: Path | None = attrs.field(
        converter=attrs.converters.optional(Path), 
        validator=attrs.validators.optional(_vld_path_exists)
    )

    @classmethod
    def from_standard_layout(
        cls,
        root: Path,
        loadname: CalLoadName,
        year: int,
        day: int,
        s11_year: int | None = None,
        s11_day: int | None = None,
        s11_hour: int | str = "first",
        allow_closest_s11_within: int = 5,
        specfmt: Literal["acq", "gsh5"] = "acq",
    ) -> Self:
        if s11_year is None:
            s11_year = year
        if s11_day is None:
            s11_day = day

        # First Get the S11s
        calkit = CalkitEdges3.from_standard_layout(
            root_dir=root,
            load=loadname,
            year=s11_year,
            day=s11_day,
            hour=s11_hour,
            allow_closest_s11_within=allow_closest_s11_within,
        )
        external = calkit.open.parent / calkit.open.name.replace("_O", f"_{loadname}")
        s11 = LoadS11(calkit=calkit, external=external)

        # Now get spectra
        fl = get_spectrum_files(loadname, root=root, year=year, day=day, fmt=specfmt)

        templog=root / "temperature_logger/temperature.log"
        if not templog.exists():
            # It's OK, sometimes you just want to pass you own temperatures.
            templog = None
            
        return cls(
            s11=s11, spectra=[fl], templog=templog, name=loadname
        )


@attrs.define(frozen=True, kw_only=True)
class CalObsDefEDGES3:
    open: LoadDefEDGES3 = attrs.field()
    short: LoadDefEDGES3 = attrs.field()
    ambient: LoadDefEDGES3 = attrs.field()
    hot_load: LoadDefEDGES3 = attrs.field()

    receiver_s11: ReceiverS11 = attrs.field()

    @property
    def loads(self) -> dict[str, LoadDefEDGES3]:
        return {
            "open": self.open,
            "short": self.short,
            "ambient": self.ambient,
            "hot_load": self.hot_load,
        }

    @classmethod
    def from_standard_layout(
        cls,
        rootdir: tp.PathLike,
        year: int,
        day: int,
        s11_year: int | None = None,
        s11_day: int | None = None,
        s11_hour: int | str = "first",
        allow_closest_s11_within: int = 5,
        specfmt: Literal["acq", "gsh5"] = "acq",
    ) -> Self:
        if s11_year is None:
            s11_year = year
        if s11_day is None:
            s11_day = day

        rootdir = Path(rootdir)
        assert rootdir.exists()

        # Get the ReceiverS11
        rcv_calkit = CalkitEdges3.from_standard_layout(
            rootdir,
            load="lna",
            year=s11_year,
            day=s11_day,
            hour=s11_hour,
            allow_closest_s11_within=allow_closest_s11_within,
        )

        rcv = ReceiverS11(
            calkit=rcv_calkit,
            device=rcv_calkit.open.parent / rcv_calkit.open.name.replace("_O", ""),
        )

        # Get the actual S11 date from the rcv
        fname = rcv.calkit.open.stem
        s11_year, s11_day, s11_hour = map(int, fname.split("_")[:3])

        # Now, get the Loads
        kw = {
            "root": rootdir,
            "s11_year": s11_year,
            "s11_day": s11_day,
            "s11_hour": s11_hour,
            "allow_closest_s11_within": 0,
            "year": year,
            "day": day,
            "specfmt": specfmt,
        }

        loads = {
            LOADMAP[load]: LoadDefEDGES3.from_standard_layout(loadname=load, **kw)
            for load in get_args(CalLoadName)
        }
        return cls(receiver_s11=rcv, **loads)
