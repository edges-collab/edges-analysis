"""Functions for combining multiple GSData files/objects."""

from __future__ import annotations

import logging
import warnings

import numpy as np
from pygsdata import GSData, gsregister

from .lstbin import NsamplesStrategy, get_weights_from_strategy

logger = logging.getLogger(__name__)


@gsregister("gather")
def average_multiple_objects(
    *objs: tuple[GSData],
    nsamples_strategy: NsamplesStrategy = NsamplesStrategy.FLAGGED_NSAMPLES,
    use_resids: bool | None = None,
) -> GSData:
    """Average multiple GSData objects together.

    In this function, each GSData object is expected to have the same data shape, so
    that each GSData object's data array can be directly summed together. This is most
    useful when each file represents a single night's worth of data, and the time axis
    represents different LSTs. However, the function is agnostic to the details
    of what each object represents.
    """
    if any(obj.data.shape != objs[0].data.shape for obj in objs[1:]):
        raise ValueError("All objects must have the same shape to average them.")

    if use_resids is None:
        use_resids = all(obj.residuals is not None for obj in objs)

    if use_resids and any(obj.residuals is None for obj in objs):
        raise ValueError("One or more of the input objects has no residuals.")

    weights, nsamples = [], []
    for obj in objs:
        w, n = get_weights_from_strategy(obj, nsamples_strategy)
        weights.append(w)
        nsamples.append(n)

    wtot = np.sum(weights, axis=0)
    ntot = np.sum(nsamples, axis=0)

    if use_resids:
        residuals = np.nansum(
            [obj.residuals * w for obj, w in zip(objs, weights)], axis=0
        )

        residuals[wtot > 0] /= wtot[wtot > 0]
        tot_model = np.nansum([obj.model for obj in objs], axis=0)
        nobj = len(objs) - sum(np.all(np.isnan(obj.model), axis=3) for obj in objs)

        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            tot_model /= nobj

        final_data = tot_model + residuals
    else:
        final_data = np.nansum([obj.data * w for obj, w in zip(objs, weights)], axis=0)
        final_data[wtot > 0] /= wtot[wtot > 0]
        residuals = None

    return objs[0].update(
        data=final_data,
        residuals=residuals,
        nsamples=ntot,
        flags={},
    )


def average_files_pairwise(*files, **kwargs) -> GSData:
    """Average multiple files together using their flagged weights.

    This has better memory management than lst_average, as it only ever reads two
    files at once.
    """
    obj = GSData.from_file(files[0])

    for i, fl in enumerate(files, start=1):
        new = GSData.from_file(fl)
        try:
            obj = average_multiple_objects(obj, new, **kwargs)
        except ValueError as e:
            raise ValueError(f"{e!s}: File {i}") from e

    return obj
