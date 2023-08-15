"""Functions for combining multiple GSData files/objects."""
from __future__ import annotations

import numpy as np

from ..gsdata import GSData, gsregister


@gsregister("gather")
def concatenate_gsdata(*objs) -> GSData:
    """Concatenate a set of GSData objects over the time axis.

    Note that to do this, each object must be in time-mode, not LST mode.
    """
    if any(obj.in_lst for obj in objs):
        raise ValueError(
            "One or more of the input GSData objects is in LST-mode. Can't concatenate."
        )

    return sum(objs)


@gsregister("gather")
def lst_average(*objs) -> GSData:
    """Average multiple objects together using their flagged weights."""
    if any(not obj.in_lst for obj in objs):
        raise ValueError(
            "One or more of the input objects is not in LST-mode. Can't LST-average."
        )

    if any(not np.allclose(obj.time_array, objs[0].time_array) for obj in objs[1:]):
        raise ValueError("All objects must have the same LST array to average them.")

    out = objs[0]
    for obj in objs[1:]:
        print(out.data)
        out += obj
    return out


def lst_average_files(*files) -> GSData:
    """Average multiple files together using their flagged weights.

    This has better memory management than lst_average, as it only ever reads two
    files at once.
    """
    obj = GSData.from_file(files[0])

    if not obj.in_lst:
        raise ValueError("First input file is not in LST-mode. Cannot LST-average.")

    for i, fl in enumerate(files, start=1):
        new = GSData.from_file(fl)
        try:
            obj = lst_average(obj, new)
        except ValueError as e:
            raise ValueError(f"{str(e)}: File {i}") from e

    return obj
