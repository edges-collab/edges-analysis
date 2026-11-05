"""Tests of the top-level ``edges`` CLI."""

import os
import subprocess
import sys

# Runs the same ``app`` as the ``edges`` console script, in a fresh interpreter.
_RUN_CLI = "import sys; from edges.cli import app; app()"


def _edges(*args: str) -> str:
    proc = subprocess.run(
        [sys.executable, "-c", _RUN_CLI, *args],
        capture_output=True,
        text=True,
        check=False,
        env=os.environ | {"COLUMNS": "200"},
    )
    assert proc.returncode == 0, proc.stderr
    return proc.stdout


def test_help_lists_subcommands():
    out = _edges("--help")
    assert "alanmode" in out
    assert "Run scripts similar to legacy C code" in out
    assert "data" in out


def test_alanmode_help_lists_commands():
    out = _edges("alanmode", "--help")
    assert "cal-edges2" in out
    assert "cal-edges3" in out
