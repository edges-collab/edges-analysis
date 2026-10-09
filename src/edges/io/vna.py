"""IO routines for VNA readings (S11, S12, S22 etc)."""

import warnings
from pathlib import Path
from typing import Self

import attrs
import numpy as np
from astropy import units as un
from astropy.table import QTable

from .. import types as tp
from ..frequencies import get_mask

_FREQ_UNITS = {"HZ": un.Hz, "KHZ": un.kHz, "MHZ": un.MHz, "GHZ": un.GHz}
_FORMATS = ("DB", "MA", "RI")
_DEFAULT_Z0 = 50.0


def _strip_comment(line: str) -> str:
    """Remove a trailing '!' comment and surrounding whitespace from a line."""
    return line.split("!", 1)[0].strip()


def _parse_option_line(
    line: str, path: Path
) -> tuple[un.Unit | None, str | None, float, list[str]]:
    """Parse a Touchstone option line, e.g. ``# MHz S RI R 50``.

    Options are case-insensitive and may come in any order.

    Returns
    -------
    unit
        The frequency unit, or None if not specified.
    flag
        The data format (one of DB, MA, RI), or None if not specified.
    z0
        The reference impedance in Ohms (default 50).
    unknown
        Any tokens that were not recognized.
    """
    tokens = line.lstrip("#").upper().split()
    unit = None
    flag = None
    z0 = _DEFAULT_Z0
    unknown = []

    i = 0
    while i < len(tokens):
        tok = tokens[i]
        if tok in _FREQ_UNITS:
            unit = _FREQ_UNITS[tok]
        elif tok in _FORMATS:
            flag = tok
        elif tok == "R" and i + 1 < len(tokens):
            z0 = float(tokens[i + 1])
            i += 1
        elif tok in ("Y", "Z", "H", "G"):
            raise ValueError(
                f"Only S-parameter Touchstone files are supported, but {path} "
                f"contains '{tok}' parameters."
            )
        elif tok != "S":
            unknown.append(tok)
        i += 1

    return unit, flag, z0, unknown


def _get_s1p_kind(path: Path) -> tuple[np.ndarray, str, un.Unit, float]:
    """Read the data and format options of a Touchstone (or similar) VNA file.

    Two layouts are supported:

    1. Standard Touchstone v1: optional ``!`` comment lines, then an option line
       (``# <freq-unit> S <DB|MA|RI> R <z0>``), then data lines. Inline ``!``
       comments and blank lines are ignored.
    2. A ``BEGIN`` line, followed by a line containing only the format flag
       (``DB``, ``MA`` or ``RI``), then data lines (with frequency in Hz), then an
       ``END`` line.

    Returns
    -------
    data
        A 2D array of shape (Nfreq, Ncolumns).
    flag
        The data format: one of DB, MA or RI.
    unit
        The unit of the frequency column.
    z0
        The reference impedance, in Ohms.
    """
    with path.open("r") as fl:
        lines = [_strip_comment(line) for line in fl]

    flag = None
    unit = None
    z0 = _DEFAULT_Z0
    data_lines = []

    nonempty = [line for line in lines if line]
    if (
        len(nonempty) > 1
        and nonempty[0].upper().startswith("BEGIN")
        and nonempty[1].upper() in _FORMATS
    ):
        # A format with a BEGIN line, then a FLAG line, then data then END.
        flag = nonempty[1].upper()
        unit = un.Hz
        candidates = nonempty[2:]
    else:
        candidates = []
        for line in lines:
            if not line:
                continue
            if line.startswith("#"):
                if flag is not None:
                    # Only the first option line is used (as per the standard).
                    continue

                opt_unit, opt_flag, opt_z0, unknown = _parse_option_line(line, path)
                if opt_flag is None:
                    # Not a valid option line (some VNAs write lines like
                    # "#48062!..." before the real option line): ignore it.
                    continue

                flag, z0 = opt_flag, opt_z0
                if unknown:
                    warnings.warn(
                        f"Unrecognized tokens {unknown} in Touchstone option line "
                        f"of {path}.",
                        stacklevel=3,
                    )
                if opt_unit is None:
                    warnings.warn(
                        f"No frequency unit in the option line of {path}. Assuming Hz.",
                        stacklevel=3,
                    )
                    opt_unit = un.Hz
                unit = opt_unit
                continue
            if flag is None:
                if not line.upper().startswith("BEGIN"):
                    warnings.warn(
                        f"Non standard line in S11 file {path}: '{line}'\n"
                        "...Treating as a comment line.",
                        stacklevel=3,
                    )
                continue
            candidates.append(line)

    if flag is None:
        raise OSError(f"The file {path} has incorrect format.")

    for line in candidates:
        if line.upper().startswith(("BEGIN", "END")):
            continue
        data_lines.append([float(x) for x in line.replace(",", " ").split()])

    if not data_lines:
        raise ValueError(f"The file {path} contains no data.")

    ncols = {len(row) for row in data_lines}
    if len(ncols) != 1:
        raise ValueError(
            f"The file {path} has data rows with differing numbers of columns: "
            f"{sorted(ncols)}."
        )

    return np.array(data_lines, dtype=float), flag, unit, z0


def _to_complex(a: np.ndarray, b: np.ndarray, flag: str) -> np.ndarray:
    """Convert a pair of data columns in the given format to complex numbers."""
    if flag == "RI":
        return a + 1j * b

    mag = 10 ** (a / 20) if flag == "DB" else a
    return mag * (np.cos((np.pi / 180) * b) + 1j * np.sin((np.pi / 180) * b))


def read_s1p(
    path: str | Path,
    f_low: tp.FreqType = 0 * un.MHz,
    f_high: tp.FreqType = np.inf * un.MHz,
) -> QTable:
    """Read a file in either s1p or s2p format, recorded by a VNA.

    The frequency unit, data format (RI, MA or DB) and reference impedance are read
    (case-insensitively) from the Touchstone option line. For two-port files, the
    columns are read in the Touchstone v1 order: f, S11, S21, S12, S22.

    Parameters
    ----------
    path
        The path to the file.
    f_low
        Minimum frequency to keep
    f_high
        Maximum frequency to keep

    Returns
    -------
    table
        A table with a ``frequency`` column (in Hz) and an ``s11`` column, and also
        ``s21``, ``s12`` and ``s22`` columns for two-port files. The reference
        impedance of the S-parameters is stored in
        ``table.meta["reference_impedance"]``.
    """
    path = Path(path)
    d, flag, unit, z0 = _get_s1p_kind(path)

    if d.shape[1] not in (3, 9):
        raise ValueError(
            f"The file {path} has {d.shape[1]} columns, but only one-port (3 columns)"
            " and two-port (9 columns) files are supported."
        )

    if z0 != _DEFAULT_Z0:
        warnings.warn(
            f"The file {path} has a reference impedance of {z0:g} Ohm. Note that "
            "the S-parameters are not renormalized to 50 Ohm.",
            stacklevel=2,
        )

    f = (d[:, 0] * unit).to(un.Hz)

    # Restrict to frequency range.
    mask = get_mask(f, f_low, f_high)
    d = d[mask]

    table = QTable({"frequency": f[mask]}, meta={"reference_impedance": z0 * un.ohm})

    names = ["s11"] if d.shape[1] == 3 else ["s11", "s21", "s12", "s22"]
    for i, name in enumerate(names):
        table[name] = _to_complex(d[:, 2 * i + 1], d[:, 2 * i + 2], flag)

    return table


@attrs.define
class SParams:
    """A class for holding S-parameters.

    All parameters are optional other than freq. The class is simply a flexible
    container for S-parameter measurements.

    Parameters
    ----------
    freq
        The frequency vector.
    s11
        The S11 parameter, same length as freq.
    s12
        The S12 parameter, same length as freq.
    s21
        The S21 parameter, same length as freq.
    s22
        The S22 parameter, same length as freq.
    """

    freq: tp.FreqType = attrs.field()
    s11: np.ndarray | None = attrs.field(default=None)
    s12: np.ndarray | None = attrs.field(default=None)
    s21: np.ndarray | None = attrs.field(default=None)
    s22: np.ndarray | None = attrs.field(default=None)

    @s11.validator
    @s12.validator
    @s21.validator
    @s22.validator
    def _check_dims(self, attribute, value):
        if value is None:
            return

        if value.shape != self.freq.shape:
            raise ValueError(
                f"Shape of {attribute.name} does not match shape of frequency"
            )

        if not np.iscomplexobj(value):
            raise ValueError("s-parameters must be complex")

    @freq.validator
    def _check_freq(self, attribute, value):
        if not value.unit.is_equivalent(un.Hz):
            raise ValueError(f"Frequency must be in units of Hz, got {value.unit}")

        if value.ndim != 1:
            raise ValueError("Frequency must be 1D")

    @classmethod
    def from_table(cls, table: QTable):
        """Create an SParams object from a table."""
        # We slice each entry here so that we copy values, so we don't use a weakref
        return cls(
            freq=table["frequency"][:],
            s11=table["s11"][:] if "s11" in table.columns else None,
            s12=table["s12"][:] if "s12" in table.columns else None,
            s21=table["s21"][:] if "s21" in table.columns else None,
            s22=table["s22"][:] if "s22" in table.columns else None,
        )

    def to_table(self) -> QTable:
        """Convert to a table."""
        return QTable(self.to_dict())

    def to_dict(self) -> dict[str, np.ndarray]:
        """Convert to a dictionary."""
        return {k: v for k, v in attrs.asdict(self).items() if v is not None}

    @classmethod
    def from_s1p_file(
        cls,
        path: str | Path,
        f_low: tp.FreqType = 0 * un.MHz,
        f_high: tp.FreqType = np.inf * un.MHz,
    ) -> Self:
        """Read an S1P file."""
        table = read_s1p(path, f_low, f_high)
        return cls.from_table(table)
