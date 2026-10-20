"""Data for use in edges-analysis.

Importing this sub-package is cheap: it only defines the paths to the data shipped
with the package. The functions that fetch external datasets (which need ``pooch``)
and the ``edges data`` CLI commands are only imported when first accessed (e.g.
``edges.data.fetch_b18cal_full`` or ``from edges.data import fetch_b18cal_full``).
"""

import importlib
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from . import cli
    from ._pooch import (
        fetch_b18_cal_outputs,
        fetch_b18cal_ants11s,
        fetch_b18cal_calibrated_s11s,
        fetch_b18cal_full,
        fetch_b18cal_resistances,
        fetch_b18cal_s11s,
        fetch_b18cal_spectra,
    )

__all__ = [
    "BEAM_PATH",
    "DATA_PATH",
    "LOSS_PATH",
    "cli",
    "fetch_b18_cal_outputs",
    "fetch_b18cal_ants11s",
    "fetch_b18cal_calibrated_s11s",
    "fetch_b18cal_full",
    "fetch_b18cal_resistances",
    "fetch_b18cal_s11s",
    "fetch_b18cal_spectra",
    "get_data_path",
]

DATA_PATH = Path(__file__).absolute().parent
BEAM_PATH = DATA_PATH / "beams"
LOSS_PATH = DATA_PATH / "loss"


def get_data_path(pth: str | Path) -> Path:
    """Impute the global data path to a given input in place of a colon.

    Parameters
    ----------
    pth
        A path. If it is a string starting with a colon (e.g. ``":beams/x.txt"``),
        the colon is replaced by :data:`DATA_PATH`.

    Returns
    -------
    Path
        The resolved path.
    """
    if isinstance(pth, str):
        return DATA_PATH / pth[1:] if pth.startswith(":") else Path(pth)
    return pth


_POOCH_FUNCTIONS = frozenset({
    "fetch_b18_cal_outputs",
    "fetch_b18cal_ants11s",
    "fetch_b18cal_calibrated_s11s",
    "fetch_b18cal_full",
    "fetch_b18cal_resistances",
    "fetch_b18cal_s11s",
    "fetch_b18cal_spectra",
})
_LAZY_SUBMODULES = frozenset({"cli", "_pooch"})


def __getattr__(name: str):
    """Lazily import the dataset-fetching functions and sub-modules (PEP 562)."""
    if name in _POOCH_FUNCTIONS:
        value = getattr(importlib.import_module(f"{__name__}._pooch"), name)
    elif name in _LAZY_SUBMODULES:
        value = importlib.import_module(f"{__name__}.{name}")
    else:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    """List the public attributes, including lazily-imported ones."""
    return sorted(set(globals()) | _POOCH_FUNCTIONS | _LAZY_SUBMODULES)
