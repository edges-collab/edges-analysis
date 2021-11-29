"""A package for analysing EDGES field data."""
from pkg_resources import DistributionNotFound, get_distribution

try:
    __version__ = get_distribution(__name__).version
except DistributionNotFound:  # pragma: no cover
    __version__ = "unknown"
finally:
    del get_distribution, DistributionNotFound

from . import analysis, simulation
from .analysis import (
    BinnedData,
    CalibratedData,
    CombinedData,
    DayAveragedData,
    ModelData,
    beams,
    coordinates,
    filters,
    levels,
    loss,
    read_step,
    s11,
    sky_models,
    tools,
)
from .config import config as cfg
