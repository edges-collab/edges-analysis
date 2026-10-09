"""Functions for combining multiple GSData files/objects."""

import logging
import warnings
from collections.abc import Sequence
from pathlib import Path

import attrs
import numpy as np
from pygsdata import GSData, gsregister

from .utils import NsamplesStrategy, get_weights_from_strategy

logger = logging.getLogger(__name__)


@gsregister("gather")
def average_multiple_objects(
    *objs: Sequence[GSData],
    nsamples_strategy: NsamplesStrategy = NsamplesStrategy.FLAGGED_NSAMPLES,
    use_resids: bool | None = None,
) -> GSData:
    """Average multiple GSData objects together.

    In this function, each GSData object is expected to have the same data shape, so
    that each GSData object's data array can be directly summed together. This is most
    useful when each file represents a single night's worth of data, and the time axis
    represents different LSTs. However, the function is agnostic to the details
    of what each object represents.

    Parameters
    ----------
    objs : GSData
        The GSData objects to average together.
    nsamples_strategy : NsamplesStrategy
        The strategy to use when considering the weights with which to combine the
        objects. See the documentation for
        :class:`~edges.analysis.averaging.lstbin.NsamplesStrategy` for more information.
    use_resids : bool, optional
        If True, the residuals will be averaged and added to the average model. If
        False, the residuals will be ignored. If None, the residuals will be used if
        they are present in all objects, and otherwise ignored.

    Returns
    -------
    GSData
        The averaged GSData object.

    Raises
    ------
    ValueError
        If the objects do not all have the same shape, or if use_resids is True and one
        or more of the objects has no residuals.
    """
    if any(obj.data.shape != objs[0].data.shape for obj in objs[1:]):
        raise ValueError("All objects must have the same shape to average them.")

    if use_resids is None:
        use_resids = all(obj.residuals is not None for obj in objs)

    if use_resids and any(obj.residuals is None for obj in objs):
        raise ValueError("One or more of the input objects has no residuals.")

    acc = _RunningAverage(nsamples_strategy=nsamples_strategy, use_resids=use_resids)
    for obj in objs:
        acc.add(obj)

    return acc.result(objs[0])


@attrs.define
class _RunningAverage:
    """Accumulate the sums required to average GSData objects one at a time.

    The final result is identical (up to floating-point round-off) to averaging all
    the objects at once, for every weighting strategy, since only additive quantities
    (sums of weights, nsamples, weighted data/residuals, models and model counts)
    are accumulated.

    Parameters
    ----------
    nsamples_strategy
        The strategy used to weight each object.
    use_resids
        Whether to average residuals (and models separately), or the data directly.
    """

    nsamples_strategy: NsamplesStrategy
    use_resids: bool
    wtot: np.ndarray | None = None
    ntot: np.ndarray | None = None
    sum_wdata: np.ndarray | None = None
    sum_model: np.ndarray | None = None
    nmodel: np.ndarray | None = None
    nobj: int = 0

    def add(self, obj: GSData) -> None:
        """Add a GSData object to the running sums."""
        if self.wtot is not None and obj.data.shape != self.wtot.shape:
            raise ValueError("All objects must have the same shape to average them.")
        if self.use_resids and obj.residuals is None:
            raise ValueError("One or more of the input objects has no residuals.")

        w, n = get_weights_from_strategy(obj, self.nsamples_strategy)
        d = obj.residuals if self.use_resids else obj.data
        wd = np.where(np.isnan(d * w), 0, d * w)

        if self.wtot is None:
            self.wtot = np.zeros(obj.data.shape)
            self.ntot = np.zeros(obj.data.shape)
            self.sum_wdata = np.zeros(obj.data.shape, dtype=wd.dtype)

        self.wtot += w
        self.ntot += n
        self.sum_wdata += wd
        self.nobj += 1

        if self.use_resids:
            model = obj.model
            if self.sum_model is None:
                self.sum_model = np.zeros(obj.data.shape, dtype=model.dtype)
                self.nmodel = np.zeros(obj.data.shape[:-1])
            self.sum_model += np.where(np.isnan(model), 0, model)
            self.nmodel += ~np.all(np.isnan(model), axis=-1)

    def result(self, template: GSData) -> GSData:
        """Compute the average, with metadata taken from ``template``."""
        if self.nobj == 0:
            raise ValueError("No objects were added to the average.")

        final = self.sum_wdata.copy()
        mask = self.wtot > 0
        final[mask] /= self.wtot[mask]

        if self.use_resids:
            residuals = final
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore")
                tot_model = self.sum_model / self.nmodel[..., None]
            final = tot_model + residuals
        else:
            residuals = None

        return template.update(
            data=final,
            residuals=residuals,
            nsamples=self.ntot,
            flags={},
        )


def average_files_pairwise(
    *files: str | Path,
    nsamples_strategy: NsamplesStrategy = NsamplesStrategy.FLAGGED_NSAMPLES,
    use_resids: bool | None = None,
) -> GSData:
    """Average multiple files together, reading at most two at a time.

    This has better memory management than :func:`average_multiple_objects`, as it
    only ever holds one file in memory (plus running sums). The result is the same
    (up to floating-point round-off) as reading all the files and calling
    :func:`average_multiple_objects` on them, for every ``nsamples_strategy``.

    Parameters
    ----------
    files : str or Path
        The files to average together.
    nsamples_strategy : NsamplesStrategy
        The strategy to use when considering the weights with which to combine the
        files. See :func:`average_multiple_objects`.
    use_resids : bool, optional
        If True, the residuals will be averaged and added to the average model. If
        False, the residuals will be ignored. If None, the residuals will be used if
        they are present in all files, and otherwise ignored.

    Returns
    -------
    GSData
        The averaged GSData object (with metadata from the first file).

    Raises
    ------
    ValueError
        If the files do not all have the same shape, or if use_resids is True and one
        or more of the files has no residuals. The message names the offending file.

    See Also
    --------
    average_multiple_objects : The equivalent function for in-memory objects.
    """
    if not files:
        raise ValueError("At least one file is required.")

    first = GSData.from_file(files[0])

    # When use_resids is None, we only know whether to use the residuals once we have
    # seen every file, so we accumulate sums for both cases while any are possible.
    accs = {}
    if use_resids is not False and first.residuals is not None:
        accs[True] = _RunningAverage(
            nsamples_strategy=nsamples_strategy, use_resids=True
        )
    if not use_resids:
        accs[False] = _RunningAverage(
            nsamples_strategy=nsamples_strategy, use_resids=False
        )

    for i, fl in enumerate(files):
        obj = first if i == 0 else GSData.from_file(fl)
        if use_resids is None and obj.residuals is None:
            accs.pop(True, None)

        try:
            if not accs:
                raise ValueError("One or more of the input objects has no residuals.")
            for acc in accs.values():
                acc.add(obj)
        except ValueError as e:
            raise ValueError(f"{e!s}: File {i} ({fl})") from e

    acc = accs[True] if True in accs else accs[False]
    return acc.result(first)
