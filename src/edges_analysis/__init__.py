# -*- coding: utf-8 -*-
from pkg_resources import get_distribution, DistributionNotFound

try:
    __version__ = get_distribution(__name__).version
except DistributionNotFound:
    __version__ = "unknown"
finally:
    del get_distribution, DistributionNotFound

from . import analysis
from . import estimation
from . import simulation

from .analysis import (
    tools,
    levels,
    s11,
    beams,
    filters,
    sky_models,
    coordinates,
    loss,
    Level1,
    Level2,
    Level3,
    Level4,
)
