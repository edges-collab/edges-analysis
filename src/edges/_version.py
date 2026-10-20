"""The version of the installed ``edges-analysis`` distribution.

This lives in its own dependency-free module so that sub-packages can access the
version without importing the top-level :mod:`edges` package (which would create an
import cycle).
"""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("edges-analysis")
except PackageNotFoundError:  # pragma: no cover
    # package is not installed
    __version__ = "unknown"
