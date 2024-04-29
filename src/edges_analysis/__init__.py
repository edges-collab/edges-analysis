"""A package for analysing EDGES field data."""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("edges-analysis")
except PackageNotFoundError:
    # package is not installed
    __version__ = "unknown"
finally:
    del version, PackageNotFoundError

__all__ = [
    "averaging",
    "beams",
    "calibration",
    "coordinates",
    "filters",
    "sky_models",
    "tools",
    "cfg",
    "groupdays",
]

from . import (
    averaging,
    beams,
    calibration,
    coordinates,
    filters,
    groupdays,
    sky_models,
    tools,
)
from .config import config as cfg
