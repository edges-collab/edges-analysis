"""A package for analysing EDGES field data."""
from pkg_resources import get_distribution, DistributionNotFound

try:
    __version__ = get_distribution(__name__).version
except DistributionNotFound:
    __version__ = "unknown"
finally:
    del get_distribution, DistributionNotFound

from . import averaging, beams, coordinates, sky_models, tools
from .config import config as cfg
from .calibration import loss, s11
from .filters import filters
