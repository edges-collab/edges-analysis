"""Functions for combining multiple GSData files/objects."""

import logging
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
    fill_value: float = np.nan,
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
    fill_value : float
        The data of channels to which no object contributes any weight (as in
        :func:`~edges.averaging.lstbin.average_over_times`). Defaults to NaN. Their
        nsamples are zero, and the model (if any) is kept there.

    Returns
    -------
    GSData
        The averaged GSData object.

    Raises
    ------
    ValueError
        If the objects do not all have the same shape, or if use_resids is True and one
        or more of the objects has no residuals.

    Notes
    -----
    With ``use_resids``, the average is the unweighted mean of the models plus the
    weighted mean of the residuals (B18; EDGES memo 183). The mean model is over the
    same objects as the residuals, i.e. only those that have weight in a channel:
    including the model of an object whose data are flagged there would bias the mean
    wherever only some of the objects have data. Objects with a non-finite residual
    in a channel have zero weight (and zero nsamples) there; so do objects whose data
    are NaN there (their model is lost with the residual).
    """
    if any(obj.data.shape != objs[0].data.shape for obj in objs[1:]):
        raise ValueError("All objects must have the same shape to average them.")

    if use_resids is None:
        use_resids = all(obj.residuals is not None for obj in objs)

    if use_resids and any(obj.residuals is None for obj in objs):
        raise ValueError("One or more of the input objects has no residuals.")

    acc = _RunningAverage(
        nsamples_strategy=nsamples_strategy,
        use_resids=use_resids,
        fill_value=fill_value,
    )
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
    fill_value
        The data of channels to which no object contributes any weight.
    """

    nsamples_strategy: NsamplesStrategy
    use_resids: bool
    fill_value: float = np.nan
    wtot: np.ndarray | None = None
    ntot: np.ndarray | None = None
    sum_wdata: np.ndarray | None = None
    sum_model: np.ndarray | None = None
    nmodel: np.ndarray | None = None
    sum_model_w: np.ndarray | None = None
    nmodel_w: np.ndarray | None = None
    nobj: int = 0

    def add(self, obj: GSData) -> None:
        """Add a GSData object to the running sums."""
        if self.wtot is not None and obj.data.shape != self.wtot.shape:
            raise ValueError("All objects must have the same shape to average them.")
        if self.use_resids and obj.residuals is None:
            raise ValueError("One or more of the input objects has no residuals.")

        w, n = get_weights_from_strategy(obj, self.nsamples_strategy)
        d = obj.residuals if self.use_resids else obj.data
        if self.use_resids:
            # A non-finite residual carries no information: zero weight and nsamples.
            w = np.where(np.isfinite(d), w, 0.0)
            n = np.where(np.isfinite(d), n, 0.0)
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
                self.nmodel = np.zeros(obj.data.shape)
                self.sum_model_w = np.zeros(obj.data.shape, dtype=model.dtype)
                self.nmodel_w = np.zeros(obj.data.shape)
            finite = np.isfinite(model)
            # Sums over all objects (used only to keep the model in empty channels)...
            self.sum_model += np.where(finite, model, 0)
            self.nmodel += finite
            # ...and over the objects that have weight (the model of the estimator).
            contributing = (w > 0) & finite
            self.sum_model_w += np.where(contributing, model, 0)
            self.nmodel_w += contributing

    def result(self, template: GSData) -> GSData:
        """Compute the average, with metadata taken from ``template``."""
        if self.nobj == 0:
            raise ValueError("No objects were added to the average.")

        # Channels to which no object contributes weight are empty: their data are
        # fill_value (not the zero of an empty sum).
        mask = self.wtot > 0
        final = np.full_like(self.sum_wdata, self.fill_value)
        final[mask] = self.sum_wdata[mask] / self.wtot[mask]

        if self.use_resids:
            # Mean model over the objects that have weight in each channel. Channels
            # that have none are empty, where the model is kept as the mean over all
            # objects whose model is known there (NaN where none is).
            tot_model = np.full_like(self.sum_model, np.nan)
            known = self.nmodel > 0
            tot_model[known] = self.sum_model[known] / self.nmodel[known]
            used = self.nmodel_w > 0
            tot_model[used] = self.sum_model_w[used] / self.nmodel_w[used]
            final = np.where(mask, tot_model + final, self.fill_value)
        else:
            tot_model = None

        return template.update(
            data=final,
            model=tot_model,
            nsamples=self.ntot,
            flags={},
        )


def average_files_pairwise(
    *files: str | Path,
    nsamples_strategy: NsamplesStrategy = NsamplesStrategy.FLAGGED_NSAMPLES,
    use_resids: bool | None = None,
    fill_value: float = np.nan,
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
    fill_value : float
        The data of channels to which no file contributes any weight. Defaults to NaN.

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
            nsamples_strategy=nsamples_strategy, use_resids=True, fill_value=fill_value
        )
    if not use_resids:
        accs[False] = _RunningAverage(
            nsamples_strategy=nsamples_strategy, use_resids=False, fill_value=fill_value
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
