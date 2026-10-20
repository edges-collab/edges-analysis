"""The edges analysis package.

Sub-packages that register processing functions with :mod:`pygsdata` (via
``gsregister``, so that they are available by name in
``pygsdata.GSDATA_PROCESSORS``) are imported eagerly, so that ``import edges`` keeps
registering all of them. The remaining sub-packages (``alanmode`` and ``inference``)
are imported lazily on first attribute access (PEP 562), e.g. ``edges.inference`` or
``from edges import alanmode``.
"""

import importlib
from typing import TYPE_CHECKING

__all__ = [
    "DATA_PATH",
    "__version__",
    "alanmode",
    "analysis",
    "averaging",
    "cal",
    "edges_raw_freqs",
    "filters",
    "get_data_path",
    "inference",
    "io",
    "modeling",
    "serialization",
    "sim",
    "tools",
    "types",
]

from ._version import __version__
from .data import DATA_PATH, get_data_path

# These sub-packages register GSData processors with pygsdata on import, and so must
# be imported eagerly.
from . import (  # isort: skip
    analysis,
    averaging,
    cal,
    filters,
)
from . import io, modeling, serialization, sim, tools, types
from .cal import apply as _cal_apply  # registers approximate_temperature
from .frequencies import edges_raw_freqs

if TYPE_CHECKING:
    from . import alanmode, inference

_LAZY_SUBPACKAGES = frozenset({"alanmode", "inference"})


def __getattr__(name: str):
    """Import the heavier, non-registering sub-packages on first access."""
    if name in _LAZY_SUBPACKAGES:
        # Importing the sub-module also sets it as an attribute of this package.
        return importlib.import_module(f"{__name__}.{name}")
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    """List the public attributes, including lazily-imported sub-packages."""
    return sorted(set(globals()) | _LAZY_SUBPACKAGES)
