"""Methods for dealing with EDGES-3 files and structures."""

import contextlib
from pathlib import Path
from typing import Literal, Self, get_args

import attrs
from bidict import bidict

from .. import types as tp
from .calobsdef import (
    CalkitFileSpec,
    LoadS11,
    ReceiverS11,
    _list_of_path,
    _vld_path_exists,
)

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


def from_edges3_layout(
    cls,
    root_dir: Path | str,
    load: S11LoadName,
    year: int,
    day: int,
    hour: int | str = "first",
    allow_closest_s11_within: int = 5,
) -> Self:
    """
    Create a CalkitEdges3 object from a standard directory layout.

    Parameters
    ----------
    root_dir
        The root directory to search in
    load
        The name of the load for which the calkit observations were taken.
    year
        The year of observation.
    day
        The day of observation
    hour
        The hour of observation, or "first" for automaic search.
    allow_closest_s11_within
        The number of surrounding days to search for S11.
    """
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


CalkitFileSpec.from_edges3_layout = classmethod(from_edges3_layout)


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
    """File-specification for an EDGES-3 Calibration Load.

    Parameters
    ----------
    name
        The name of the load.
    s11
        The S11 measurement files.
    spectra
        The spectrum measurement files.
    templog
        The temperature logger file.
    """

    name: str = attrs.field()

    s11: LoadS11 = attrs.field()
    spectra: list[Path] = attrs.field(converter=_list_of_path)
    templog: Path | None = attrs.field(
        converter=attrs.converters.optional(Path),
        validator=attrs.validators.optional(_vld_path_exists),
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
        """
        Create a LoadDefEDGES3 instance from a standard directory layout.

        This method locates and loads all required calibration and observation files
        for the specified date and configuration.

        Parameters
        ----------
        root : PathLike
            The root directory containing the data.
        loadname
            The name of the load to gather files for.
        year : int
            The year of the observation.
        day : int
            The day of the year for the observation.
        s11_year : int or None, optional
            The year to use for S11 calibration files. Defaults to `year` if not
            provided.
        s11_day : int or None, optional
            The day to use for S11 calibration files. Defaults to `day` if not provided.
        s11_hour : int or str, optional
            The hour to use for S11 calibration files, or "first"/"last" for automatic
            selection.
        allow_closest_s11_within : int, optional
            Maximum number of days to search for the closest S11 file.
        specfmt : {'acq', 'gsh5'}, optional
            The file format for spectrum files.

        Returns
        -------
        LoadDefEDGES3
            A LoadDefEDGES3 instance with all loads and receiver S11 populated.

        Raises
        ------
        AssertionError
            If the root directory does not exist.
        FileNotFoundError
            If required files are not found.
        OSError
            If multiple files are found where only one is expected.
        """
        if s11_year is None:
            s11_year = year
        if s11_day is None:
            s11_day = day

        # First Get the S11s
        calkit = CalkitFileSpec.from_edges3_layout(
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

        templog = root / "temperature_logger/temperature.log"
        if not templog.exists():
            # It's OK, sometimes you just want to pass you own temperatures.
            templog = None

        return cls(s11=s11, spectra=[fl], templog=templog, name=loadname)


@attrs.define(frozen=True, kw_only=True)
class CalObsDefEDGES3:
    """A class for holding calibration observation definitions for EDGES3.

    This class holds specifications for where all the data required for the EDGES-3
    receiver calibration is located on disk. That is, it holds only paths to files,
    rather than actual data.

    It is convenient in that it also has methods for finding this files in standard
    situations.

    Parameters
    ----------
    open
        The loading definition for the open load.
    short
        The loading definition for the short load.
    ambient
        The loading definition for the ambient load.
    hot_load
        The loading definition for the hot load.
    receiver
        The definition for the Receiver S11.
    """

    open: LoadDefEDGES3 = attrs.field()
    short: LoadDefEDGES3 = attrs.field()
    ambient: LoadDefEDGES3 = attrs.field()
    hot_load: LoadDefEDGES3 = attrs.field()

    receiver_s11: ReceiverS11 = attrs.field()

    @property
    def loads(self) -> dict[str, LoadDefEDGES3]:
        """A dictionary of all loads."""
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
        """
        Create a CalObsDefEDGES3 instance from the standard directory layout.

        This method locates and loads all required calibration and observation files
        for the specified date and configuration.

        Parameters
        ----------
        rootdir
            The root directory containing the data.
        year
            The year of the observation.
        day : int
            The day of the year for the observation.
        s11_year : int or None, optional
            The year to use for S11 calibration files. Defaults to `year` if not
            provided.
        s11_day : int or None, optional
            The day to use for S11 calibration files. Defaults to `day` if not provided.
        s11_hour : int or str, optional
            The hour to use for S11 calibration files, or "first"/"last" for automatic
            selection.
        allow_closest_s11_within : int, optional
            Maximum number of days to search for the closest S11 file.
        specfmt : {'acq', 'gsh5'}, optional
            The file format for spectrum files.

        Returns
        -------
        CalObsDefEDGES3
            A CalObsDefEDGES3 instance with all loads and receiver S11 populated.

        Raises
        ------
        AssertionError
            If the root directory does not exist.
        FileNotFoundError
            If required files are not found.
        OSError
            If multiple files are found where only one is expected.
        """
        if s11_year is None:
            s11_year = year
        if s11_day is None:
            s11_day = day

        rootdir = Path(rootdir)
        assert rootdir.exists()

        # Get the ReceiverS11
        rcv_calkit = CalkitFileSpec.from_edges3_layout(
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
