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
