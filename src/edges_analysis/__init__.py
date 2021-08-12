"""A package for analysing EDGES field data."""
from pkg_resources import get_distribution, DistributionNotFound

try:
    __version__ = get_distribution(__name__).version
except DistributionNotFound:
    __version__ = "unknown"
finally:
    del get_distribution, DistributionNotFound

from . import analysis
from . import simulation
from .config import config as cfg

from .analysis import (
    tools,
    levels,
    s11,
    beams,
    filters,
    sky_models,
    coordinates,
    loss,
    CalibratedData,
    CombinedData,
    ModelData,
    DayAveragedData,
    BinnedData,
    read_step,
)
