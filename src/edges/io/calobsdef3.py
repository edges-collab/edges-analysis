"""Methods for dealing with EDGES-3 files and structures."""

import re
import warnings
from collections.abc import Mapping, Sequence
from datetime import date, timedelta
from pathlib import Path
from typing import ClassVar, Literal, Self, get_args

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

#: Default directory (relative to the root) holding the ``.s1p`` files.
DEFAULT_S11_DIR = "."
#: Default directory template (relative to the root) holding the spectrum files.
#: It is formatted with ``load`` and ``year``.
DEFAULT_SPECTRUM_DIR = "mro/{load}/{year}"
#: Default temperature log (relative to the root).
DEFAULT_TEMPLOG = "temperature_logger/temperature.log"
#: Default metadata of the receiver S11 measurement.
DEFAULT_RECEIVER_METADATA = {"calkit_match_resistance": 49.8, "calkit": "AGILENT_ALAN"}

# S11 files are named YYYY_DDD_HH_<label>.s1p, e.g. 2023_070_11_lna_O.s1p.
_S11_NAME = re.compile(r"^(\d{4})_(\d{3})_(\d{2})_(\w+)$")


def _shift_day(year: int, day: int, ddays: int) -> tuple[int, int]:
    """Shift a (year, day-of-year) pair by a number of days, wrapping across years."""
    shifted = date(year, 1, 1) + timedelta(days=day - 1 + ddays)
    return shifted.year, shifted.timetuple().tm_yday


def _resolve_dir(root: Path, template: tp.PathLike, **kwargs) -> Path:
    """Format a directory template and resolve it relative to ``root``.

    An absolute template is used as-is (after formatting).
    """
    return root / str(template).format(**kwargs)


def _get_single_s1p_file(
    root: Path,
    year: int,
    day: int,
    label: str,
    hour: int | str = "first",
    allow_closest: int = 30,
    s11_dir: tp.PathLike = DEFAULT_S11_DIR,
) -> Path:
    # Search the given day first, then day+1, day-1, day+2, ... out to allow_closest.
    offsets = [0]
    for dday in range(1, allow_closest + 1):
        offsets += [dday, -dday]

    for offset in offsets:
        this_year, this_day = _shift_day(year, day, offset)
        direc = _resolve_dir(root, s11_dir, year=this_year)
        hourglob = f"{hour:02d}" if isinstance(hour, int) else "??"
        glob = f"{this_year:04d}_{this_day:03d}_{hourglob}_{label}.s1p"

        files = sorted(direc.glob(glob))
        if not files:
            continue
        if len(files) == 1 or hour == "first":
            return files[0]
        if hour == "last":
            return files[-1]
        raise OSError(f"More than one file found for {this_year}, {this_day}, {label}")

    raise FileNotFoundError(
        f"No s1p files found for label '{label}' within {allow_closest} days of "
        f"{year:04d}_{day:03d} (searched with s11_dir='{s11_dir}' under {root})"
    )


def get_s1p_files(
    load: S11LoadName,
    year: int,
    day: int,
    root_dir: Path | str,
    hour: int | str = "first",
    allow_closest_s11_within: int = 5,
    s11_dir: tp.PathLike = DEFAULT_S11_DIR,
) -> dict[str, Path]:
    """Take the load and return a dict of .s1p files for that load.

    Parameters
    ----------
    load
        The name of the load.
    year
        The year of the S11 measurement.
    day
        The day of year of the S11 measurement.
    root_dir
        The root directory of the data.
    hour
        The hour of the S11 measurement, or "first"/"last" to take the first/last
        measurement of the day.
    allow_closest_s11_within
        If no file exists for the given day, search up to this many days either
        side of it (across year boundaries) for the closest day that has one. Days
        are tried in the order day+1, day-1, day+2, ..., so the later day wins a
        tie.
    s11_dir
        Directory holding the ``.s1p`` files, relative to ``root_dir`` (or absolute).
        It can contain a ``{year}`` placeholder.

    Returns
    -------
    dict
        Paths keyed by "input", "open", "short" and "match".
    """
    root_dir = Path(root_dir)

    def _get(label: str) -> Path:
        return _get_single_s1p_file(
            root_dir, year, day, label, hour, allow_closest_s11_within, s11_dir=s11_dir
        )

    files = {"input": _get(load)}
    for name, label in {"open": "O", "short": "S", "match": "L"}.items():
        files[name] = _get(f"lna_{label}" if load == "lna" else label)

    return files


def from_edges3_layout(
    cls,
    root_dir: Path | str,
    load: S11LoadName,
    year: int,
    day: int,
    hour: int | str = "first",
    allow_closest_s11_within: int = 5,
    s11_dir: tp.PathLike = DEFAULT_S11_DIR,
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
        The number of surrounding days (either side) to search for S11. Days are
        tried in the order day+1, day-1, day+2, ..., so the later day wins a tie.
    s11_dir
        Directory holding the ``.s1p`` files, relative to ``root_dir`` (or absolute).
        It can contain a ``{year}`` placeholder.
    """
    files = get_s1p_files(
        load=load,
        year=year,
        day=day,
        root_dir=root_dir,
        hour=hour,
        allow_closest_s11_within=allow_closest_s11_within,
        s11_dir=s11_dir,
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
    spectrum_dir: tp.PathLike = DEFAULT_SPECTRUM_DIR,
    allow_multiple: bool = False,
) -> list[Path]:
    """Get the spectrum files for a particular load on a particular day.

    Parameters
    ----------
    load
        The name of the load.
    root
        The root directory of the data.
    year
        The year of the observation.
    day
        The day of year of the observation.
    fmt
        The file format of the spectra.
    spectrum_dir
        Directory holding the spectrum files, relative to ``root`` (or absolute). It
        can contain ``{load}`` and ``{year}`` placeholders.
    allow_multiple
        Whether to allow more than one file for the load on the day. If False, an
        error is raised when more than one file is found.

    Returns
    -------
    list of Path
        The files, sorted by name (i.e. by time).

    Raises
    ------
    FileNotFoundError
        If no files are found.
    OSError
        If more than one file is found and ``allow_multiple`` is False.

    Warns
    -----
    UserWarning
        If there are files for the load that start at midnight (``00_00``) on the
        next day. These are probably the continuation of the last file of the day
        (files are split at UTC midnight), but are not included. To include them,
        use :meth:`CalObsDefEDGES3.from_files`.
    """
    d = _resolve_dir(Path(root), spectrum_dir, load=load, year=year)
    glob = f"{year}_{day:03d}_??_??_??_{load}.{fmt}"
    files = sorted(d.glob(glob))

    if not files:
        raise FileNotFoundError(f"No files found in {d} for {glob}")
    if len(files) > 1 and not allow_multiple:
        raise OSError(
            f"More than one file found in {d} for {glob}. Set allow_multiple=True to "
            "use all of them."
        )

    next_year, next_day = _shift_day(year, day, 1)
    next_dir = _resolve_dir(Path(root), spectrum_dir, load=load, year=next_year)
    if continued := sorted(
        next_dir.glob(f"{next_year}_{next_day:03d}_00_00_??_{load}.{fmt}")
    ):
        warnings.warn(
            f"Spectrum file(s) {[fl.name for fl in continued]} start at midnight "
            f"after {files[-1].name}, so are probably its continuation, but are not "
            "included. To include them, use CalObsDefEDGES3.from_files.",
            stacklevel=2,
        )

    return files


def _templog_converter(
    x: tp.PathLike | Sequence[tp.PathLike] | None,
) -> Path | list[Path] | None:
    if x is None:
        return None
    if isinstance(x, str | Path):
        return Path(x)
    return [Path(xx) for xx in x] or None


def _vld_templog(inst, att, val):
    for pth in [val] if isinstance(val, Path) else val or []:
        _vld_path_exists(inst, att, pth)


def _resolve_templog(
    root: Path, templog: tp.PathLike | Sequence[tp.PathLike] | None
) -> Path | list[Path] | None:
    """Resolve the templog argument of the ``from_standard_layout`` methods."""
    if templog is None:
        # It's OK if this doesn't exist: sometimes you want to pass your own
        # temperatures.
        default = root / DEFAULT_TEMPLOG
        return default if default.exists() else None
    if isinstance(templog, str | Path):
        return root / templog
    return [root / pth for pth in templog] or None


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
        The spectrum measurement files. They are concatenated in time.
    templog
        The temperature logger file, or a list of files. Several files are merged
        (overlapping entries are de-duplicated), see
        :func:`edges.io.templogs.read_temperature_log`.

    Attributes
    ----------
    telescope
        The telescope used when reading the spectra. None means it is inferred from
        the header of the spectrum files.
    """

    telescope: ClassVar[str | None] = None

    name: str = attrs.field()

    s11: LoadS11 = attrs.field()
    spectra: list[Path] = attrs.field(converter=_list_of_path)
    templog: Path | list[Path] | None = attrs.field(
        converter=_templog_converter, validator=_vld_templog
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
        s11_dir: tp.PathLike = DEFAULT_S11_DIR,
        spectrum_dir: tp.PathLike = DEFAULT_SPECTRUM_DIR,
        templog: tp.PathLike | Sequence[tp.PathLike] | None = None,
        allow_multiple_spectra: bool = False,
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
            Maximum number of days (either side) to search for the closest S11 file.
            The search wraps across year boundaries. Days are tried in the order
            day+1, day-1, day+2, ..., so the later day wins a tie.
        specfmt : {'acq', 'gsh5'}, optional
            The file format for spectrum files.
        s11_dir
            Directory holding the ``.s1p`` files, relative to ``root`` (or absolute).
            It can contain a ``{year}`` placeholder.
        spectrum_dir
            Directory holding the spectrum files, relative to ``root`` (or absolute).
            It can contain ``{load}`` and ``{year}`` placeholders.
        templog
            The temperature log file(s), relative to ``root`` (or absolute). Several
            files are merged. By default, ``temperature_logger/temperature.log`` is
            used if it exists. Pass an empty list to use no temperature log.
        allow_multiple_spectra
            Whether to allow (and concatenate) more than one spectrum file for the
            load on the day.

        Returns
        -------
        LoadDefEDGES3
            A LoadDefEDGES3 instance with all loads and receiver S11 populated.

        Raises
        ------
        FileNotFoundError
            If required files are not found.
        OSError
            If multiple files are found where only one is expected.
        """
        root = Path(root)
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
            s11_dir=s11_dir,
        )
        external = calkit.open.parent / calkit.open.name.replace("_O", f"_{loadname}")
        s11 = LoadS11(calkit=calkit, external=external)

        # Now get spectra
        spectra = get_spectrum_files(
            loadname,
            root=root,
            year=year,
            day=day,
            fmt=specfmt,
            spectrum_dir=spectrum_dir,
            allow_multiple=allow_multiple_spectra,
        )

        return cls(
            s11=s11,
            spectra=spectra,
            templog=_resolve_templog(root, templog),
            name=loadname,
        )


def _receiver_metadata(metadata: Mapping | None) -> dict:
    """Merge user-given receiver S11 metadata with the defaults."""
    return {**DEFAULT_RECEIVER_METADATA, **(metadata or {})}


def _label_s11_files(files: Sequence[tp.PathLike]) -> dict[str, Path]:
    """Map the label of each S11 file (e.g. "amb", "lna_O") to its path."""
    out = {}
    for fl in map(Path, files):
        match = _S11_NAME.match(fl.stem)
        if fl.suffix != ".s1p" or match is None:
            raise ValueError(f"S11 file {fl} is not named like YYYY_DDD_HH_<label>.s1p")
        label = match[4]
        if label in out:
            raise ValueError(f"More than one S11 file with label '{label}' given.")
        out[label] = fl
    return out


@attrs.define(frozen=True, kw_only=True)
class CalObsDefEDGES3:
    """A class for holding calibration observation definitions for EDGES3.

    This class holds specifications for where all the data required for the EDGES-3
    receiver calibration is located on disk. That is, it holds only paths to files,
    rather than actual data.

    It is convenient in that it also has methods for finding this files in standard
    situations (:meth:`from_standard_layout`) or creating it from explicit lists of
    files (:meth:`from_files`).

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
    def from_files(
        cls,
        s11_files: Sequence[tp.PathLike],
        spectra: Mapping[str, tp.PathLike | Sequence[tp.PathLike]],
        templog: tp.PathLike | Sequence[tp.PathLike] | None = None,
        receiver_metadata: dict | None = None,
    ) -> Self:
        """
        Create a CalObsDefEDGES3 instance from explicit lists of files.

        No directory layout is assumed, so this is the constructor to use when the
        files to use have been selected elsewhere (e.g. by a file catalog).

        Parameters
        ----------
        s11_files
            The ``.s1p`` files. Each is identified by its label, i.e. the part of its
            name after the ``YYYY_DDD_HH_`` timestamp. The labels ``O``, ``S``,
            ``L`` (the calkit for the loads), ``lna_O``, ``lna_S``, ``lna_L`` (the
            calkit for the receiver), ``lna``, ``amb``, ``hot``, ``open`` and
            ``short`` must each be given exactly once. Files with other labels
            (e.g. ``ant``) are ignored. The files need not share a timestamp.
        spectra
            The spectrum files for each load, keyed by load name (either short names
            ``amb``, ``hot``, ``open``, ``short`` or long names ``ambient``,
            ``hot_load``, ``open``, ``short``). Each value is a file or a list of
            files, which are sorted by name (i.e. time) and concatenated.
        templog
            The temperature log file(s), merged if more than one.
        receiver_metadata
            Metadata of the receiver S11 measurement (see
            :class:`edges.io.calobsdef.ReceiverS11`). Keys not given are taken
            from :data:`DEFAULT_RECEIVER_METADATA`.

        Returns
        -------
        CalObsDefEDGES3
            A CalObsDefEDGES3 instance with all loads and receiver S11 populated.

        Raises
        ------
        FileNotFoundError
            If any required S11 file or load spectra are not given.
        ValueError
            If an S11 file is not named as expected, or more than one S11 file has
            the same label.
        """
        s11 = _label_s11_files(s11_files)

        loads = get_args(CalLoadName)
        required = ["O", "S", "L", "lna_O", "lna_S", "lna_L", "lna", *loads]
        if missing := [label for label in required if label not in s11]:
            raise FileNotFoundError(f"No S11 files given for labels {missing}")

        spectra = {LOADMAP.inverse.get(k, k): v for k, v in spectra.items()}
        if missing := [load for load in loads if load not in spectra]:
            raise FileNotFoundError(f"No spectrum files given for loads {missing}")

        def _calkit(prefix: str = "") -> CalkitFileSpec:
            return CalkitFileSpec(
                open=s11[f"{prefix}O"],
                short=s11[f"{prefix}S"],
                match=s11[f"{prefix}L"],
            )

        rcv = ReceiverS11(
            calkit=_calkit("lna_"),
            device=s11["lna"],
            metadata=_receiver_metadata(receiver_metadata),
        )

        load_calkit = _calkit()
        loaddefs = {}
        for load in loads:
            files = spectra[load]
            files = [files] if isinstance(files, str | Path) else files
            loaddefs[LOADMAP[load]] = LoadDefEDGES3(
                name=load,
                s11=LoadS11(calkit=load_calkit, external=s11[load]),
                spectra=sorted(map(Path, files)),
                templog=templog,
            )
        return cls(receiver_s11=rcv, **loaddefs)

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
        receiver_metadata: dict | None = None,
        s11_dir: tp.PathLike = DEFAULT_S11_DIR,
        spectrum_dir: tp.PathLike = DEFAULT_SPECTRUM_DIR,
        templog: tp.PathLike | Sequence[tp.PathLike] | None = None,
        allow_multiple_spectra: bool = False,
    ) -> Self:
        """
        Create a CalObsDefEDGES3 instance from the standard directory layout.

        This method locates and loads all required calibration and observation files
        for the specified date and configuration. The directories searched can be
        changed with ``s11_dir``, ``spectrum_dir`` and ``templog``; to specify the
        files directly, use :meth:`from_files`.

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
            Maximum number of days (either side) to search for the closest S11 file.
            The search wraps across year boundaries. Days are tried in the order
            day+1, day-1, day+2, ..., so the later day wins a tie.
        specfmt : {'acq', 'gsh5'}, optional
            The file format for spectrum files.
        receiver_metadata
            Metadata of the receiver S11 measurement (see
            :class:`edges.io.calobsdef.ReceiverS11`). Keys not given are taken
            from :data:`DEFAULT_RECEIVER_METADATA`.
        s11_dir
            Directory holding the ``.s1p`` files, relative to ``rootdir`` (or
            absolute). It can contain a ``{year}`` placeholder.
        spectrum_dir
            Directory holding the spectrum files, relative to ``rootdir`` (or
            absolute). It can contain ``{load}`` and ``{year}`` placeholders.
        templog
            The temperature log file(s), relative to ``rootdir`` (or absolute).
            Several files are merged, de-duplicating overlapping entries. By default,
            ``temperature_logger/temperature.log`` is used if it exists. Pass an empty
            list to use no temperature log.
        allow_multiple_spectra
            Whether to allow (and concatenate) more than one spectrum file per load
            on the day.

        Returns
        -------
        CalObsDefEDGES3
            A CalObsDefEDGES3 instance with all loads and receiver S11 populated.

        Raises
        ------
        FileNotFoundError
            If the root directory does not exist, or required files are not found.
        OSError
            If multiple files are found where only one is expected.
        """
        if s11_year is None:
            s11_year = year
        if s11_day is None:
            s11_day = day

        rootdir = Path(rootdir)
        if not rootdir.is_dir():
            raise FileNotFoundError(f"rootdir {rootdir} does not exist")

        # Get the ReceiverS11
        rcv_calkit = CalkitFileSpec.from_edges3_layout(
            rootdir,
            load="lna",
            year=s11_year,
            day=s11_day,
            hour=s11_hour,
            allow_closest_s11_within=allow_closest_s11_within,
            s11_dir=s11_dir,
        )

        rcv = ReceiverS11(
            calkit=rcv_calkit,
            device=rcv_calkit.open.parent / rcv_calkit.open.name.replace("_O", ""),
            metadata=_receiver_metadata(receiver_metadata),
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
            "s11_dir": s11_dir,
            "spectrum_dir": spectrum_dir,
            "templog": templog,
            "allow_multiple_spectra": allow_multiple_spectra,
        }

        loads = {
            LOADMAP[load]: LoadDefEDGES3.from_standard_layout(loadname=load, **kw)
            for load in get_args(CalLoadName)
        }
        return cls(receiver_s11=rcv, **loads)
