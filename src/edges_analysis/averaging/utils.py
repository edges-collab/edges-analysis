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


def get_weights_from_strategy(
    data: GSData, strategy: NsamplesStrategy
) -> tuple[np.ndarray, np.ndarray]:
    """Compute weights and nsamples used for a particular strategy."""
    if strategy == NsamplesStrategy.FLAGGED_NSAMPLES:
        w = data.flagged_nsamples
        n = w
    elif strategy == NsamplesStrategy.FLAGS_ONLY:
        w = (~data.complete_flags).astype(float)
        n = data.flagged_nsamples
    elif strategy == NsamplesStrategy.FLAGGED_NSAMPLES_UNIFORM:
        w = (data.flagged_nsamples > 0).astype(float)
        n = data.flagged_nsamples
    elif strategy == NsamplesStrategy.NSAMPLES_ONLY:
        w = data.nsamples
        n = w
    else:
        raise ValueError(
            f"Invalid nsamples_strategy: {strategy}. "
            f"Must be a member of {NsamplesStrategy}"
        )
    return w, n
