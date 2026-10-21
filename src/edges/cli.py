"""A CLI for edges-analysis."""

from cyclopts import App

app = App()

# The ``alanmode`` sub-command is registered lazily by import path, so that importing
# this module and running the other sub-commands (e.g. ``edges data ...``) do not pay
# for importing alanmode. (The top-level ``edges --help`` does import it, to show
# its help text.) Its help text is defined on the App in ``edges.alanmode.cli``.
app.command("edges.alanmode.cli:amode", name="alanmode")

# Register the sub-commands defined in other modules. These modules import ``app``
# from here (which is already defined at this point), so the imports must come last.
# Previously these were only registered as a side effect of ``import edges``.
from .data import cli as _data_cli  # ruff: ignore[module-import-not-at-top-of-file, unused-import]
