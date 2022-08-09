"""A package for analysing EDGES field data."""
from pkg_resources import DistributionNotFound, get_distribution

try:
    __version__ = get_distribution(__name__).version
except DistributionNotFound:
    __version__ = "unknown"
finally:
    del get_distribution, DistributionNotFound

from . import averaging, beams, calibration, coordinates, filters, sky_models, tools
from .config import config as cfg
