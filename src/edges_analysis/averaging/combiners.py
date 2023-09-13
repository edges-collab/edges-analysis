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
def lst_average(
    *objs,
    use_nsamples: bool = True,
    use_flags: bool = True,
    use_resids: bool | None = None,
) -> GSData:
    """Average multiple objects together using their flagged weights."""
    if any(not obj.in_lst for obj in objs):
        raise ValueError(
            "One or more of the input objects is not in LST-mode. Can't LST-average."
        )

    if any(
        not np.allclose(obj.time_array.hour, objs[0].time_array.hour)
        for obj in objs[1:]
    ):
        raise ValueError("All objects must have the same LST array to average them.")

    if any(obj.data.shape != objs[0].data.shape for obj in objs[1:]):
        raise ValueError("All objects must have the same shape to average them.")

    if use_nsamples:
        nsamples = [obj.nsamples for obj in objs]
    elif use_flags:
        nsamples = [(~(obj.nsamples == 0)).astype(float) for obj in objs]
    else:
        nsamples = [1] * len(objs)

    tot_nsamples = sum(nsamples)

    if use_resids is None:
        use_resids = all(obj.residuals is not None for obj in objs)

    if use_resids and any(obj.residuals is None for obj in objs):
        raise ValueError("One or more of the input objects has no residuals.")

    if use_resids:
        residuals = sum(obj.residuals * n for obj, n in zip(objs, nsamples))
        residuals[tot_nsamples > 0] /= tot_nsamples[tot_nsamples > 0]
        tot_model = sum(obj.model for obj in objs)
        tot_model[tot_nsamples>0] /= tot_nsamples[tot_nsamples>0]
        final_data = tot_model + residuals
    else:
        final_data = sum(obj.data * n for obj, n in zip(objs, nsamples))
        final_data[tot_nsamples > 0] /= tot_nsamples[tot_nsamples > 0]
        residuals = None

    return objs[0].update(
        data=final_data,
        residuals=residuals,
        nsamples=tot_nsamples,
    )


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
