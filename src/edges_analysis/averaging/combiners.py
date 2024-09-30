"""Functions for combining multiple GSData files/objects."""

from __future__ import annotations

import logging
import warnings
from typing import Literal

import numpy as np
from astropy import units as un
from astropy.coordinates import Longitude
from pygsdata import GSData, gsregister

logger = logging.getLogger(__name__)


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


@gsregister("reduce")
def average_over_times(
    data: GSData,
    nsamples_strategy: Literal[
        "flagged-nsamples",
        "flags-only",
        "flagged-nsamples-uniform",
        "nsamples-only",
    ] = "flagged-nsamples",
    use_resids: bool | None = None,
):
    """Average a GSData object over the time axis.

    Parameters
    ----------
    data : GSData object
        The data over which to average.
    nsamples_strategy : str, optional
        The strategy to use when defining the weights of each sample. Defaults to
        'flagged-nsamples'. The choices are:
        - 'flagged-nsamples': Use the flagged nsamples (i.e. set nsamples at flagged
            data to zero, otherwise use nsamples)
        - 'flags-only': Use the flags only (i.e. set nsamples at flagged data to
            zero, otherwise use 1)
        - 'flagged-nsamples-uniform': Use the flagged nsamples (i.e. set nsamples at
            flagged data to zero, and keep zero-samples as zero, otherwise use 1)
        - 'nsamples-only': Use the nsamples only (don't set nsamples at flagged
            data to zero)
    use_resids : bool, optional
        Whether to average the residuals and add them back to the mean model, or simply
        average the data directly.
    """
    if use_resids is None:
        use_resids = data.residuals is not None

    if use_resids and data.residuals is None:
        raise ValueError("Cannot average residuals without a model.")

    if nsamples_strategy == "flagged-nsamples":
        w = data.flagged_nsamples
    elif nsamples_strategy == "flags-only":
        w = (~data.complete_flags).astype(float)
    elif nsamples_strategy == "flagged-nsamples-uniform":
        w = (data.flagged_nsamples > 0).astype(float)
    elif nsamples_strategy == "nsamples-only":
        w = np.nsamples
    else:
        raise ValueError(
            f"Invalid nsamples_strategy: {nsamples_strategy}. Must be one of "
            "'flagged-nsamples', 'flags-only', 'flagged-nsamples-uniform' or "
            "'nsamples-only'"
        )

    ntot = np.sum(w, axis=-2)
    if use_resids:
        sum_resids = np.sum(data.residuals * w, axis=-2)
        mean_resids = sum_resids / ntot
        mean_model = np.mean(data.model, axis=-2)
        new_data = mean_model + mean_resids
    else:
        sum_data = np.sum(data.data * w, axis=-2)
        new_data = sum_data / ntot

    return data.update(
        data=new_data[:, :, None, :],
        residuals=mean_resids[:, :, None, :] if use_resids else None,
        times=np.atleast_2d(np.mean(data.times)),
        time_ranges=np.array([[[data.time_ranges.min(), data.time_ranges.max()]]]),
        lst_ranges=Longitude(
            np.array([[[data.lst_ranges.hour.min(), data.lst_ranges.hour.max()]]])
            * un.hour
        ),
        nsamples=ntot,
        flags={},
    )


@gsregister("gather")
def lst_average(
    *objs,
    use_nsamples: bool = True,
    use_flags: bool = True,
    use_resids: bool | None = None,
) -> GSData:
    """Average multiple objects together using their flagged weights."""
    if any(not np.allclose(obj.lsts.hour, objs[0].lsts.hour) for obj in objs[1:]):
        raise ValueError("All objects must have the same LST array to average them.")

    if any(obj.data.shape != objs[0].data.shape for obj in objs[1:]):
        raise ValueError("All objects must have the same shape to average them.")

    if use_nsamples:
        nsamples = [obj.nsamples for obj in objs]
    elif use_flags:
        nsamples = [(~(obj.nsamples == 0)).astype(float) for obj in objs]
    else:
        nsamples = [1] * len(objs)

    tot_nsamples = np.nansum(nsamples, axis=0)

    if use_resids is None:
        use_resids = all(obj.residuals is not None for obj in objs)

    if use_resids and any(obj.residuals is None for obj in objs):
        raise ValueError("One or more of the input objects has no residuals.")

    if use_resids:
        residuals = np.nansum(
            [obj.residuals * n for obj, n in zip(objs, nsamples)], axis=0
        )
        residuals[tot_nsamples > 0] /= tot_nsamples[tot_nsamples > 0]
        tot_model = np.nansum([obj.model for obj in objs], axis=0)
        tot_obj = len(objs) - sum([np.all(np.isnan(obj.model), axis=3) for obj in objs])
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            tot_model /= (tot_obj)[..., None]
        logger.debug(f"After combining sum(residuals): {np.nansum(residuals)}")
        final_data = tot_model + residuals
    else:
        final_data = np.nansum([obj.data * n for obj, n in zip(objs, nsamples)], axis=0)
        final_data[tot_nsamples > 0] /= tot_nsamples[tot_nsamples > 0]
        residuals = None

    return objs[0].update(
        data=final_data,
        residuals=residuals,
        nsamples=tot_nsamples,
        flags={},
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
            raise ValueError(f"{e!s}: File {i}") from e

    return obj
