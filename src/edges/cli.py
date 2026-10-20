"""A CLI for edges-analysis."""

from cyclopts import App

app = App()

# Register the sub-commands defined in other modules. These modules import ``app``
# from here (which is already defined at this point), so the imports must come last.
# Previously these were only registered as a side effect of ``import edges``.
from .data import cli as _data_cli  # noqa: E402, F401
