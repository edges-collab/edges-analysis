"""Data for use in edges-analysis."""

from pathlib import Path

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
