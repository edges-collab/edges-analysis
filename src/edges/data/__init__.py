"""Data for use in edges-analysis."""

from pathlib import Path

DATA_PATH = Path(__file__).absolute().parent
BEAM_PATH = DATA_PATH / "beams"
LOSS_PATH = DATA_PATH / "loss"

from ._pooch import (
    fetch_b18_cal_outputs,
    fetch_b18cal_calibrated_s11s,
    fetch_b18cal_full,
    fetch_b18cal_resistances,
    fetch_b18cal_s11s,
    fetch_b18cal_spectra,
)
