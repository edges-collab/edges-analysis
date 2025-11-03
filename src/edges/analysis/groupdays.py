"""Functions for grouping GSData objects together."""

from collections.abc import Sequence

from pygsdata import GSData, gsregister
from pygsdata.concat import concat


@gsregister("gather")
def group_days(*data: Sequence[GSData]) -> list[GSData]:
    """Group multiple GSData objects together by days."""
    sublists = {}
    for d in data:
        yd = d.get_initial_yearday()
        if yd not in sublists:
            sublists[yd] = [d]
        else:
            sublists[yd].append(d)

    # Ensure they're sorted
    sublists = {k: sorted(v, key=lambda x: x.times.min()) for k, v in sublists.items()}

    return [concat(x, axis="time") for x in sublists.values()]
