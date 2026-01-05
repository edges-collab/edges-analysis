"""App commands for data fetching and caching."""

from typing import Literal

from cyclopts import App
from rich.console import Console

from ..cli import app
from ._pooch import (
    fetch_b18cal_calibrated_s11s,
    fetch_b18cal_full,
    fetch_b18cal_resistances,
    fetch_b18cal_s11s,
)

cns = Console()

app.command(data := App(name="data", help="Fetch and cache EDGES data files"))


@data.command()
def fetch_b18(
    dataset: Literal["testing", "full", "none"] = "testing",
):
    """Fetch B18CAL data files and cache them locally.

    Parameters
    ----------
    testing
        If True, fetch only the files required for testing.
    all
        If True, fetch all B18CAL data files.
    """
    if dataset == "full":
        out = fetch_b18cal_full()
        cns.print(f"Fetched all B18 calibration data to {out}")
    elif dataset == "testing":
        fls = [
            fetch_b18cal_calibrated_s11s(),
            fetch_b18cal_resistances(),
            fetch_b18cal_s11s(),
        ]
        cns.print("Fetched B18 calibration testing data files:")
        for fl in fls:
            cns.print(f" - {fl}")
    else:
        cns.print("No files fetched.")
