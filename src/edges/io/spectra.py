"""Functions for reading lists of spectrum files."""

from collections.abc import Sequence
from pathlib import Path

from pygsdata import GSData
from read_acq.gsdata import read_acq_to_gsdata

from ..const import KNOWN_TELESCOPES


def read_spectra(files: Sequence[Path]) -> GSData:
    """Read common spectrum file formats."""
    fmt = files[0].suffix

    if fmt in (".h5", ".gsh5"):
        return GSData.from_file(files, concat_axis="time")
    if fmt == ".acq":
        return read_acq_to_gsdata(files, telescope=KNOWN_TELESCOPES["edges-low"])
    raise ValueError(f"File format '{fmt}' not supported.")
