"""Calibration of EDGES data."""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("edges-cal")
except PackageNotFoundError:  # pragma: no cover
    # package is not installed
    __version__ = "unknown"
finally:
    del version, PackageNotFoundError

from pathlib import Path

DATA_PATH = Path(__file__).parent / "data"


def get_data_path(pth: str | Path) -> Path:
    """Impute the global data path to a given input in place of a colon."""
    if isinstance(pth, str):
        return DATA_PATH / pth[1:] if pth.startswith(":") else Path(pth)
    return pth


from .load_data import Load
from .calobs import CalibrationObservation
from .calibrator import Calibrator 
from .spectra import LoadSpectrum
from .noise_waves import NoiseWaves, NoiseWaveLinearModel
from .receiver_cal import get_calcoeffs_iterative
from . import plots

from .s11 import InternalSwitch, S11Model, StandardsReadings

del Path
