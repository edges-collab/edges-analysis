"""Functions for combining multiple GSData files/objects."""
from ..gsdata import GSData, register_gsgather
import numpy as np

@register_gsgather
def concatenate_gsdata(*objs) -> GSData:
    """Concatenate a set of GSData objects over the time axis.
    
    Note that to do this, each object must be in time-mode, not LST mode.
    """
    if any(obj.in_lst for obj in objs):
        raise ValueError("One or more of the input GSData objects is in LST-mode. Cannot concatenate.")

    return sum(objs)

@register_gsgather
def lst_average(*objs) -> GSData:
    """Average multiple objects together using their flagged weights."""
    if any(not obj.in_lst for obj in objs):
        raise ValueError("One or more of the input GSData objects is not in LST-mode. Cannot LST-average.")

    if any(~np.allclose(obj.time_array, objs[0].time_array) for obj in objs[1:]):
        raise ValueError("All objects must have the same LST array to average them.")

    return sum(objs)


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