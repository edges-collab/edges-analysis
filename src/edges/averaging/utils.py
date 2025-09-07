"""Utility functions and classes for averaging operations."""

from enum import Enum

import numpy as np
from pygsdata import GSData


class NsamplesStrategy(Enum):
    """An enumeration of strategies for computing Nsamples when combining data.

    Note that generally the strategy can influence *two* components of the calculation:
    firstly it influences how the data is _weighted_ when it is being averaged, and
    secondly, it influences what the final number of samples are (i.e. the effective
    variance of the data). Generally, these align, but some options specifically choose
    different conventions for each of these choices.

    Options
    -------
    FLAGGED_NSAMPLES
        Combine the underlying Nsamples, setting any flagged data to have zero samples.
    FLAGS_ONLY
        Count each datum as one sample, but only if it is not flagged.
    FLAGGED_NSAMPLES_UNIFORM
        Give each data that is both unflagged and has at least one sample a weight of
        unity in an average/model, but propagate nsamples using the full flagged
        nsamples.
    NSAMPLES_ONLY
        Only consider the Nsamples of the underlying data, not any flags.
    """

    FLAGGED_NSAMPLES = 0
    FLAGS_ONLY = 1
    FLAGGED_NSAMPLES_UNIFORM = 2
    NSAMPLES_ONLY = 3
    UNFLAGGED_UNIFORM = 4


def get_weights_from_strategy(
    data: GSData, strategy: NsamplesStrategy
) -> tuple[np.ndarray, np.ndarray]:
    """Compute weights and nsamples used for a particular strategy."""
    nans = np.isnan(data.data)

    # n is the nsamples that is propagated through to compute the variance.
    # for now, we always propagate the true "flagged_nsamples" for simplicity.
    # In the future, it may be better to adjust it for each strategy such that
    # the resulting summed nsamples is indicative of the variance of the average,
    # taking into account which weights are being used. This is a little complicated,
    # because it depends on assumptions about the distribution of the data.
    n = data.flagged_nsamples

    if strategy == NsamplesStrategy.FLAGGED_NSAMPLES:
        w = data.flagged_nsamples
    elif strategy == NsamplesStrategy.FLAGS_ONLY:
        w = (~data.complete_flags).astype(float)
    elif strategy == NsamplesStrategy.FLAGGED_NSAMPLES_UNIFORM:
        w = (data.flagged_nsamples > 0).astype(float)
    elif strategy == NsamplesStrategy.NSAMPLES_ONLY:
        w = data.nsamples
    elif strategy == NsamplesStrategy.UNFLAGGED_UNIFORM:
        w = np.ones_like(data.data)
    else:
        raise ValueError(
            f"Invalid nsamples_strategy: {strategy}. "
            f"Must be a member of {NsamplesStrategy}"
        )
    return np.where(nans, 0, w), np.where(nans, 0, n)
