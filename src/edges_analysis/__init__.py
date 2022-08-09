"""A package for analysing EDGES field data."""
from pkg_resources import get_distribution, DistributionNotFound

try:
    __version__ = get_distribution(__name__).version
except DistributionNotFound:
    __version__ = "unknown"
finally:
    del get_distribution, DistributionNotFound

from . import filters
from . import beams, coordinates, sky_models, tools
from .config import config as cfg
from . import calibration
from . import averaging
