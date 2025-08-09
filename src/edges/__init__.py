from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

try:
    __version__ = version("edges-analysis")
except PackageNotFoundError:
    # package is not installed
    __version__ = "unknown"
finally:
    del version, PackageNotFoundError

__all__ = ["__version__"]

DATA_PATH = Path(__file__).parent / "data"

def get_data_path(pth: str | Path) -> Path:
    """Impute the global data path to a given input in place of a colon."""
    if isinstance(pth, str):
        return DATA_PATH / pth[1:] if pth.startswith(":") else Path(pth)
    return pth


from . import types
from . import tools
from .frequencies import edges_raw_freqs
from . import io
from . import modelling

from . import filters
from . import averaging

from . import sim
from . import cal
from . import analysis
from . import alanmode
from .io import serialization

