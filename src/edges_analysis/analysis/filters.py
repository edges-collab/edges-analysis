from __future__ import annotations

import logging
from dataclasses import dataclass
from multiprocessing import cpu_count
from pathlib import Path
from sys import stderr
from typing import Tuple, List, Sequence, Dict, Union, Optional, Type, Callable
from .levels import FilteredData, CalibratedData, FrequencyRange, read_step
from . import types as tp
import dill as pickle
import attr
import h5py
import numpy as np
import yaml
from multiprocess.pool import Pool
from edges_cal.modelling import Model, Polynomial
from edges_cal.xrfi import _get_mad, flagged_filter, robust_divide
from cached_property import cached_property
import abc

from matplotlib import pyplot as plt

logger = logging.getLogger(__name__)


@attr.s(frozen=True)
class GHAModelFilter:
    """A filter that is able to act upon data that has been aggregated along frequency.

    Parameters
    ----------
    metric_model
        A linear model that provides a good guess of the typical aggregated metric as it
        evolves over GHA.
    std_model
        A linear model that provides a good guess of the typical standard deviation of
        the metric as it evolves over GHA. This may be gotten by fitting a model to the
        absolute residuals, for example.
    meta
        A dictionary holding meta-information about how the models were obtained.
    n_sigma
        The number of sigma to threshold at.
    """

    metric_model: Model = attr.ib()
    std_model: Model = attr.ib()
    meta: dict = attr.ib(factory=dict)

    n_sigma: float = attr.ib(default=3.0)

    def apply_filter(self, gha: np.ndarray, metric: np.ndarray) -> np.ndarray:
        """Apply the filter to a set of metric data.

        Parameters
        ----------
        gha
            The GHAs at which the data to be filtered was measured (1D array)
        metric
            The metric to filter on (eg. RMS or Total Power) as a function of GHA
            (1D array)

        Returns
        -------
        flags
            Boolean array specifying which data points should be flagged.
        """
        resids = metric - self.metric_model(x=gha)
        return np.abs(resids) > (self.n_sigma * self.std_model(x=gha))

    @classmethod
    def from_data(
        cls,
        info: GHAModelFilterInfo,
        metric_model: Model,
        std_model: Model,
        n_sigma: float = 3.0,
        meta=None,
    ):
        """Create the object from input metric data and models."""
        flags = np.where(np.isnan(info.metric) | np.isinf(info.metric), True, info.flags)

        mmodel = metric_model.fit(xdata=info.gha, ydata=info.metric, weights=(~flags).astype(float))

        flags = np.where(np.isnan(info.std) | np.isinf(info.std), True, info.flags)
        smodel = std_model.fit(xdata=info.gha, ydata=info.std, weights=(~flags).astype(float))

        return cls(metric_model=mmodel.fit, std_model=smodel.fit, n_sigma=n_sigma, meta=meta or {})

    @classmethod
    def from_file(cls, fname: tp.PathLike) -> GHAModelFilter:
        """Create the class from a h5 file."""
        with h5py.File(fname, "r") as fl:
            metric_model_kwargs = dict(fl["metric"].attrs)
            metric_model_type = Model.get_mdl(metric_model_kwargs.pop("model_type"))
            metric_model_params = fl["metric"]["params"]

            std_model_kwargs = dict(fl["std"].attrs)
            std_model_type = Model.get_mdl(std_model_kwargs.pop("model_type"))
            std_model_params = fl["std"]["params"]

            gha = fl["gha"][...]

            metric_model = Model.get_mdl(metric_model_type)(
                default_x=gha, parameters=metric_model_params, **metric_model_kwargs
            )
            std_model = Model.get_mdl(std_model_type)(
                default_x=gha, parameters=std_model_params, **std_model_kwargs
            )

            n_sigma = fl.attrs["n_sigma"]
            meta = dict(fl["meta"].attrs)

            return cls(metric_model=metric_model, std_model=std_model, n_sigma=n_sigma, meta=meta)

    def write(self, fname: tp.PathLike):
        """Write a h5 file."""
        with h5py.File(fname, "w") as fl:
            metric = fl.create_group("metric")
            metric.attrs["n_terms"] = self.std_model.n_terms
            metric.attrs["model_type"] = self.std_model.__class__.__name__
            metric["params"] = self.std_model.parameters

            std = fl.create_group("std")
            std.attrs["n_terms"] = self.std_model.n_terms
            std.attrs["model_type"] = self.std_model.__class__.__name__
            std["params"] = self.std_model.parameters

            fl["gha"] = self.metric_model["default_x"]
            fl.attrs["n_sigma"] = self.n_sigma
            meta = fl.create_group("meta")
            for k, v in self.meta:
                meta[k] = v


@attr.s(frozen=True, auto_attribs=True, kw_only=True)
class GHAModelFilterInfo:
    """An object containing the data going into creating a :class:`GHAModelFIlter`.

    This is useful for saving the full information to disk, or as an intermediate object
    used when determining the model information.
    """

    gha: np.ndarray
    metric: np.ndarray
    std: np.ndarray
    flags: np.ndarray

    def __attrs_post_init__(self):
        """Run post-init scripts.

        This just runs validation on all of the inputs after they are set on the class.
        """
        self._validate_inputs()

    def _validate_inputs(self) -> bool:
        assert self.gha.shape == self.metric.shape == self.flags.shape == self.std.shape
        assert self.gha.ndim == 1
        assert self.gha.dtype == float
        assert self.metric.dtype == float
        assert self.std.dtype == float
        assert self.flags.dtype == bool

    def get_metric_residuals(
        self, model_filter: GHAModelFilter, params: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """Get the residuals of the metric to a smooth model fit over all GHA."""
        return self.metric - model_filter.metric_model(parameters=params)


@attr.s(frozen=True)
class FrequencyAggregator(metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def aggregate_file(self, data: Union[FilteredData, CalibratedData]) -> np.ndarray:
        """Actually aggregate over frequency."""
        raise NotImplementedError

    def aggregate(
        self, data: Sequence[Union[tp.PathLike, FilteredData, CalibratedData]]
    ) -> Union[np.ndarray, np.ndarray]:
        """Aggregate a set of data files over frequency.

        Parameters
        ----------
        data
            A sequence of data files to use to obtain the metric over which to filter.

        Returns
        -------
        gha
            The GHA's at which the aggregated metric is measured. Sorted in ascending
            order.
        metric
            The aggregated metric at each GHA.
        """
        # Convert the data to actual Step objects
        data = [read_step(d) for d in data]

        # get the total number of GHAs in all the files
        n_gha_total = sum(len(d.gha) for d in data)

        # Put all the GHA's together
        gha = np.concatenate(tuple(d.gha for d in data))

        # Get the aggregated metrics for all input files.
        metric = np.empty(n_gha_total)
        count = 0
        for datafile in data:
            n_gha = len(datafile.gha)
            metric[count : count + n_gha] = self.aggregate_file(datafile)
            count += n_gha

        # Now, sort the output
        sortidx = np.argsort(gha)

        return gha[sortidx], metric[sortidx]


def get_gha_model_filter(
    data: Sequence[Union[tp.PathLike, FilteredData, CalibratedData]],
    aggregator: FrequencyAggregator,
    metric_model_type: tp.Modelable,
    std_model_type: tp.Modelable,
    metric_model_kwargs: Optional[Dict] = None,
    std_model_kwargs: Optional[Dict] = None,
    detrend_metric_model: Optional[Model] = None,
    detrend_std_model: Optional[Model] = None,
    detrend_gha_chunk_size: float = 24.0,
    n_sigma: float = 3.0,
    std_estimator: str = "absres",
    medfilt_width: int = 100,
) -> Tuple[GHAModelFilter, GHAModelFilterInfo]:
    """Obtain a filtering object from a given set of representative data.

    The algorithm here is to first intrinsically flag the input data, then fit a model
    to it over GHA, which can be used to flag further files. The initial intrinsice
    flagging is *by default* done with the same model that will be applied to other data,
    but it can be done at a more fine-grained level, fitting to small chunks of GHA
    at a time with a lower-order model.

    Parameters
    ----------
    data
        A sequence of data files to use to obtain the metric over which to filter.
    aggregator
        A specialized function that takes a file and some parameters and returns
        an aggregated metric.
    metric_model_type
        A linear model to be fit to the aggregated metric data.
    std_model_type
        A linear model to be fit to the absolute residuals of the metric data.
    metric_model_kwargs
        Parameters for the linear model (such as ``n_terms``) to fit to the metric.
    std_model_kwargs
        Parameters for the linear model (such as ``n_terms``) to fit to the std.
    detrend_metric_model
        A model to fit to the data to detrend it in order to determine flags. This is
        a model *instance*, but it should not have the ``default_x`` set on it. By
        default, the same as ``metric_model_type`` (with the same parameters).
    detrend_std_model
        A model to fit to the residuals of the data in order to determine flags. This is
        a model *instance*, but it should not have the ``default_x`` set on it. By
        default, the same as ``std_model_type`` (with the same parameters).
    detrend_gha_chunk_size
        The chunk size (in GHA) to use for model-fitting, in order to detrend and find
        flags in the input data. By default, use the whole data set to find flags.
    n_sigma
        The number of sigma the data can be away from the model before being flagged.

    Returns
    -------
    filter
        An object that can be used to filter other files based on the same aggregation.
    info
        An object containing information about the fit itself -- useful for inspection.

    """
    # Aggregate the data for each file along the frequency axis.
    gha, metric = aggregator.aggregate(data)

    # Remove nan/inf values from the data.
    mask = np.isnan(metric) | np.isinf(metric)
    gha = gha[~mask]
    metric = metric[~mask]

    # Now get the metric model.
    # Here, we do things like iteratively flagging on the data so that the metric is
    # well defined.
    metric_model = Model.get_mdl(metric_model_type)(**metric_model_kwargs)
    std_model = Model.get_mdl(std_model_type)(**std_model_kwargs)

    if detrend_metric_model is None:
        detrend_metric_model = metric_model
    if detrend_std_model is None:
        detrend_std_model = std_model

    flags, resid, std = chunked_iterative_model_filter(
        x=gha,
        data=metric,
        model=detrend_metric_model,
        std_model=detrend_std_model,
        chunk_size=detrend_gha_chunk_size,
        n_sigma=n_sigma,
        std_estimator=std_estimator,
        medfilt_width=medfilt_width,
    )

    info = GHAModelFilterInfo(gha=gha, metric=metric, std=std, flags=flags)
    return (
        GHAModelFilter.from_data(
            info,
            metric_model,
            std_model,
            n_sigma=n_sigma,
            meta={
                "infiles": [getattr(d, "filename", str(d)) for d in data],
                "gha_chunk_size": detrend_gha_chunk_size,
                "detrend_model_type": detrend_metric_model.__class__.__name__,
                "detrend_std_model_type": detrend_std_model.__class__.__name__,
                "detrend_model_nterms": detrend_metric_model.n_terms,
                "detrend_std_nterms": detrend_std_model.n_terms,
                "std_estimator": std_estimator,
                "medfilt_width": medfilt_width,
            },
        ),
        info,
    )


def iterative_model_filter(
    x: np.ndarray,
    data: np.ndarray,
    model: Model,
    std_model: Model,
    n_sigma: float = 3.0,
    flags: np.ndarray = None,
    init_flags: np.ndarray = None,
    std_estimator: str = "absres",
    medfilt_width: int = 100,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Iteratively determine a model fit to a given array of data.

    This function takes a set of input data as a function of a variable ``x`` and
    iteratively fits a particular model to it, fits a model to the absolute residuals,
    and then flags out data that has residuals greater than ``n_sigma`` times the
    absolute residual model. The newly-flagged data is then re-modelled, and so on until
    no new flags are obtained between iterations.

    The model of the absolute residuals is a good approximation of the local standard
    deviation if there is enough data to hand.

    Parameters
    ----------
    x
        The coordinates of the data
    data
        The data to which to fit a model.
    model
        The linear model to fit to the data
    std_model
        The linear model to fit to the absolute residuals.
    n_sigma
        The threshold number of standard deviations the data may be away from the model
        before being flagged.
    flags
        Any pre-known flags of the data (i.e. data that is known to be bad and should
        always be flagged).
    init_flags
        This is an initial guess of the flags. This is useful to do a rough initial
        flagging of VERY bad values that would otherwise distort the initial modelling.
        These flags do no persist to the second iteration, but of course very bad values
        will still be flagged.
    std_estimator
        The estimator to use to get the standard deviation each data point. One of
        'medfilt' (to use a median filtered RMS), 'absres' (to use a model fit to the
        absolute residuals), 'std' (to use a simple standard deviation within the window)
        or 'mad' (to use a simple median absolute deviation within the window).

    Returns
    -------
    flags
        A boolean array specifying which data is bad. All input ``flags`` are still
        flagged in this array. The function is careful to NOT overwrite the input flags
        internally, however.
    resid
        Residuals to the model
    std
        Estimates of the standard deviation of the data at each data point.
    """
    if flags is None:
        flags = np.zeros(len(x), dtype=bool)
    if init_flags is None:
        init_flags = np.zeros(len(x), dtype=bool)

    this_flg = flags | init_flags
    old_flg = np.ones(len(this_flg), dtype=bool)

    while np.any(this_flg != old_flg):
        old_flg = this_flg.copy()

        wght = (~this_flg).astype(float)

        # TODO: this is not as efficient as possible, since we keep specifying
        # the basis functions.
        resid = model.fit(ydata=data, xdata=x, weights=wght).residual

        # Here we estimate the standard deviation of the data at each data point.
        # There are lots of ways to do this.
        if std_estimator == "medfilt":
            std = np.sqrt(
                flagged_filter(
                    resid ** 2, size=2 * (medfilt_width // 2) + 1, kind="median", flags=this_flg
                )
                / 0.456
            )
        elif std_estimator == "absres":
            std = std_model.fit(ydata=np.abs(resid), xdata=x, weights=wght).evaluate()
        elif std_estimator == "std":
            std = np.std(resid) * np.ones_like(x)
        elif std_estimator == "mad":
            std = _get_mad(resid) * np.ones_like(x)
        else:
            raise ValueError("std_estimator must be one of 'medfilt', 'absres','std' or 'mad'.")

        std = std_model.fit(ydata=np.abs(resid), xdata=x, weights=wght).evaluate()

        this_flg = flags | (np.abs(resid) > n_sigma * std)

    return this_flg, resid, std


def chunked_iterative_model_filter(
    *,
    x: np.ndarray,
    data: np.ndarray,
    flags: Optional[np.ndarray] = None,
    init_flags: Optional[np.ndarray] = None,
    chunk_size: float = np.inf,
    **kwargs,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Perform a chunk-wise iterative model filter.

    This breaks the given data into smaller chunks and then calls
    :func:`iterative_model_filter` on each chunk, returning the full 1D array of flags
    after all the chunks have been processed.

    Parameters
    ----------
    chunk_size
        The size of the chunks to process, in units of the input coordinates, ``x``.
    **kwargs
        Everything else is passed to :func:`iterative_model_filter`.

    Returns
    -------
    flags
        The 1D array of flags corresponding to the data. Note that input flags are not
        modified in the course of this function, but the output does already contain
        those flags.
    resid
        Residuals to the model
    std
        Estimates of the standard deviation of the data at each data point.
    """

    if flags is None:
        flags = np.zeros(len(x), dtype=bool)
    if init_flags is None:
        init_flags = np.zeros(len(x), dtype=bool)

    out_flags = flags | np.isnan(data)
    resids = np.zeros_like(data)
    std = np.zeros_like(data)

    xmin = x.min()
    while xmin < x.max():
        mask = (x >= xmin) & (x < xmin + chunk_size)

        out_flags[mask], resids[mask], std[mask] = iterative_model_filter(
            x=x[mask], data=data[mask], flags=out_flags[mask], init_flags=init_flags[mask], **kwargs
        )
        xmin += chunk_size

    return out_flags, resids, std


@attr.s(frozen=True)
class RMSAggregator(FrequencyAggregator):
    """An aggregator that fits a model and yields the RMS over a given freq range."""

    model_type: tp.Modelable = attr.ib(default="LinLog")
    band: Tuple[float, float] = attr.ib(default=(0, np.inf))
    model_kwargs: Optional[dict] = attr.ib(factory=dict)

    def aggregate_file(self, data: Union[FilteredData, CalibratedData]) -> np.ndarray:
        freq_mask = (data.raw_frequencies >= self.band[0]) & (data.raw_frequencies < self.band[1])

        model = Model.get_mdl(self.model_type)(
            default_x=data.raw_frequencies[freq_mask], **self.model_kwargs
        )

        rms = np.empty(len(data.gha))
        for i, spectrum in enumerate(data.spectrum):
            fit = model.fit(ydata=spectrum[freq_mask], weights=data.weights[i, freq_mask])
            rms[i] = np.sqrt(np.mean(fit.residual ** 2))

        return rms


@attr.s(frozen=True)
class TotalPowerAggregator(FrequencyAggregator):
    """An aggregator that fits a model and yields the RMS over a given freq range."""

    band: Tuple[float, float] = attr.ib(default=(0, np.inf))

    def aggregate_file(self, data: Union[FilteredData, CalibratedData]) -> np.ndarray:
        freq_mask = (data.raw_frequencies >= self.band[0]) & (data.raw_frequencies < self.band[1])
        return np.sum(data.spectrum[:, freq_mask], axis=1)


def apply_gha_model_filter(
    data: Sequence[Union[tp.PathLike, FilteredData, CalibratedData]],
    aggregator: FrequencyAggregator,
    filt: Optional[GHAModelFilter] = None,
    n_files: int = 0,
    **kwargs,
):
    """Apply a GHA-based model filter to a set of data files."""
    if n_files == 0:
        n_files = len(data)

    if filt is None:
        filt, info = get_gha_model_filter(data=data[:n_files], aggregator=aggregator, **kwargs)


def mean_power_model(gha, nu_min, nu_max, beta=-2.5):
    """A really rough model of expected mean power between two frequencies"""
    t75 = 1750 * np.cos(np.pi * gha / 12) + 3250  # approximate model based on haslam
    return (t75 / ((beta + 1) * 75.0 ** beta) * (nu_max ** (beta + 1) - nu_min ** (beta + 1))) / (
        nu_max - nu_min
    )


def explicit_filter(times, bad, ret_times=False):
    """
    Explicitly filter out certain times.

    Parameters
    ----------
    times : array-like
        The input times. This can be either a recarray, a list of tuples, a list
        of ints, or a 2D array of ints. The columns of the recarray (or the entries
        of the tuples) should correspond to `year`, 'day` and `hour`. The last two
        are not required, eg. 2-tuples will be interpreted as ``(year, hour)``, and a
        list of ints will be interpreted as just years.
    bad : str or array-like
        Like `times`, but specifying the bad entries. Need not have the same columns
        as `times`. If any bad exists within a given time frame, it will be considered
        bad. Likewise, if bad has higher scope than times, then it will also be bad.
        Eg.: ``times = [2018], bad=[(2018, 125)]``, times will be considered bad.
        Also, ``times=[(2018, 125)], bad=[2018]``, times will be considered bad.
        If a str, reads the bad times from a properly configured YAML file.
    ret_times : bool, optional
        If True, return the good times as well as the indices of such in original array.

    Returns
    -------
    keep :
        indices marking which times are not bad if inplace=False.
    times :
        Only if `ret_times=True`. An array of the times that are good.
    """
    if isinstance(bad, str):
        with open(bad, "r") as fl:
            bad = yaml.load(fl, Loader=yaml.FullLoader)["bad_days"]

    try:
        nt = len(times[0])
    except AttributeError:
        nt = 1

    try:
        nb = len(bad[0])
    except AttributeError:
        nb = 1

    assert nt in {1, 2, 3}, "times must be an array of 1,2 or 3-tuples"
    assert nb in {1, 2, 3}, "bad must be an array of 1,2 or 3-tuples"

    if nt < nb:
        bad = {b[:nt] for b in bad}
        nb = nt

    keep = [t[:nb] not in bad for t in times]

    if ret_times:
        return keep, times[keep]
    else:
        return keep


def time_filter_auxiliary(
    gha: np.ndarray,
    sun_el: np.ndarray,
    moon_el: np.ndarray,
    humidity: np.ndarray,
    receiver_temp: np.ndarray,
    gha_range: Tuple[float, float] = (0, 24),
    sun_el_max: float = 90,
    moon_el_max: float = 90,
    amb_hum_max: float = 200,
    min_receiver_temp: float = 0,
    max_receiver_temp: float = 100,
    flags: Optional[np.ndarray] = None,
) -> np.ndarray:
    loc_flgs = np.zeros(len(gha), dtype=bool)

    def filter(condition, message, loc_flgs):
        nflags = np.sum(loc_flgs)

        loc_flgs |= condition
        nnew = np.sum(loc_flgs) - nflags
        if nnew:
            logger.debug(f"{nnew}/{len(loc_flgs) - nflags} times flagged due to {message}")

    filter((gha < gha_range[0]) | (gha >= gha_range[1]), "GHA range", loc_flgs)
    filter(sun_el > sun_el_max, "sun position", loc_flgs)
    filter(moon_el > moon_el_max, "moon position", loc_flgs)
    filter(humidity > amb_hum_max, "humidity", loc_flgs)
    filter(
        (receiver_temp >= max_receiver_temp) | (receiver_temp <= min_receiver_temp),
        "receiver temp",
        loc_flgs,
    )

    if flags is not None:
        assert flags.shape[-1] == len(
            gha
        ), f"flags must be an array with last axis len(gha). Got {flags.shape}"
        flags |= loc_flgs
    else:
        flags = loc_flgs

    return flags


def filter_explicit_gha(gha, first_day, last_day):
    # TODO: this should not be used -- it's arbitrary!
    if gha in [0, 2, 3, 4, 5, 16, 17, 18, 19, 20, 21, 22]:
        good_days = np.arange(140, 300, 1)
    else:
        good_day_dct = {
            1: np.concatenate((np.arange(148, 160), np.arange(161, 220))),
            10: np.concatenate(
                (
                    np.arange(148, 168),
                    np.arange(177, 194),
                    np.arange(197, 202),
                    np.arange(205, 216),
                )
            ),
            11: np.arange(187, 202),
            12: np.arange(147, 150),
            13: np.array([147, 149, 157, 159]),
            14: np.arange(148, 183),
            15: np.concatenate((np.arange(140, 183), np.arange(187, 206), np.arange(210, 300))),
            23: np.arange(148, 300),
            6: np.arange(147, 300),
            7: np.concatenate(
                (
                    np.arange(147, 153),
                    np.arange(160, 168),
                    np.arange(174, 202),
                    np.arange(210, 300),
                )
            ),
            8: np.concatenate((np.arange(147, 151), np.arange(160, 168), np.arange(174, 300))),
            9: np.concatenate(
                (
                    np.arange(147, 153),
                    np.arange(160, 168),
                    np.arange(174, 194),
                    np.arange(197, 202),
                    np.arange(210, 300),
                )
            ),
        }
        good_days = good_day_dct[gha]

    return good_days[(good_days >= first_day) & (good_days <= last_day)]
