from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

try:
    __version__ = version("edges-analysis")
except PackageNotFoundError:
    # package is not installed
    __version__ = "unknown"
finally:
    del version, PackageNotFoundError

DATA_PATH = Path(__file__).parent / "data"

__all__ = ["__version__"]

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
