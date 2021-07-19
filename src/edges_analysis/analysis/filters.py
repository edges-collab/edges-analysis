"""Functions that identify and flag bad data in various ways."""
from __future__ import annotations

import logging
from multiprocessing import cpu_count
from typing import Sequence, Callable, Literal, Any
from .levels import FilteredData, CalibratedData, read_step, _ReductionStep
from . import types as tp
import attr
import h5py
import numpy as np
import yaml
from edges_cal.modelling import Model
from edges_cal.xrfi import _get_mad, flagged_filter
import abc
import functools
from .data import DATA_PATH
from . import tools
from edges_cal import xrfi as rfi
import inspect
import p_tqdm

logger = logging.getLogger(__name__)

STEP_FILTERS = {}


def get_step_filter(filt: str) -> Callable:
    """Obtain a registered step filter function from a string name."""
    if filt not in STEP_FILTERS:
        raise KeyError(f"'{filt}' does not ")
    return STEP_FILTERS[filt]


def step_filter(
    axis: Literal["time", "freq", "both"],
    multi_data: bool = False,
    data_type: type[_ReductionStep] | tuple[_ReductionStep] = _ReductionStep,
):
    """A decorator to register a filtering function as a potential filter.

    Any function that is wrapped by :func:`step_filter` must implement the following
    signature::

        def fnc(
            data: Union[_ReductionStep, List[_ReductionStep]]),
            flags: Union[np.ndarray, List[np.ndarray]],
            **kwargs
        ) -> np.ndarray

    Where the ``data`` is either a single reduction step object, or sequence of such
    objects. The ``flags`` are an input array of boolean flags on the existing data,
    the same shape as ``data.spectrum``.

    The returned array should be a 1D or 2D boolean array of flags that may or may
    not include the input flags. The function itself should *not* modify the input
    flags. These will be combined with the input flags (if any) and returned.

    Parameters
    ----------
    axis
        The axis over which the filter works. If either 'time' or 'freq', the returned
        array from the wrapped function should be 1D, corresponding to either the
        time or frequency axis of the data. If 'both', the return should be 2D. In the
        case of 'time' or 'freq' filters, the flags will be broadcast across the other
        dimension.
    multi_data
        Whether the filter accepts multiple objects at the same time to filter. This
        is *usually* so as to enable more accurate filtering when comparing different
        days for instance, rather than just performing a loop over the days and flagging
        each independently.
    data_type
        Types against which the input data will be checked. This is useful to restrict
        functionality to certain steps (example restricting total power filtering to
        the calibration or modelling step, where raw integrations still exist).
    """

    def inner(
        fnc: Callable[
            [_ReductionStep | list[_ReductionStep], np.ndarray | list[np.ndarray]],
            np.ndarray,
        ]
    ):
        STEP_FILTERS[fnc.__name__] = fnc

        uses_flags = "flags" in inspect.signature(fnc).parameters

        @functools.wraps(fnc)
        def wrapper(
            *,
            data: Sequence[tp.PathLike | _ReductionStep],
            flags: Sequence[np.ndarray] | None = None,
            in_place: bool = False,
            n_threads: int = 1,
            **kwargs,
        ) -> np.ndarray:
            logger.info(f"Running {fnc.__name__} filter.")

            # Read all the data, in case they haven't been turned into objects yet.
            # And check that everything is the right type.
            data = [read_step(d) for d in data]
            assert all(isinstance(d, data_type) for d in data)

            # Specify initial flags (can either be list or single array)
            flg = (
                [d.get_flags() for d in data]
                if flags is None
                else [f.copy() for f in flags]
            )

            if np.all(flg):
                return flags

            pre_flag_n = [np.sum(f) for f in flg]

            if multi_data:
                if uses_flags:
                    kwargs.update(flags=flg)
                this_flag = fnc(data=data, **kwargs)
            else:

                def fnc_(data, flags, **kwargs):
                    if uses_flags:
                        return fnc(data=data, flags=flags, **kwargs)
                    else:
                        return fnc(data=data, **kwargs)

                this_flag = list(
                    p_tqdm.p_umap(fnc_, data, flags, unit="files", num_cpus=n_threads)
                )

            if flags is not None and any(
                np.any(f0 != f1) for f0, f1 in zip(flg, flags)
            ):
                logger.warning(
                    f"Filter {fnc.__name__} modified input flags in-place. "
                    "This should not happen!"
                )

            for n, d, f, this_f in zip(pre_flag_n, data, flg, this_flag):
                f |= this_f if axis in ("both", "freq") else np.atleast_2d(this_f).T

                if np.all(f):
                    logger.warning(
                        f"{d.filename.name} was fully flagged during {fnc.__name__} "
                        "filter"
                    )
                else:
                    logger.info(
                        f"'{d.filename.name}': {100 * n / f.size:.2f} → "
                        f"{100 * np.sum(f) / f.size:.2f}% [red]<+"
                        f"{100 * (np.sum(f) - n) / f.size:.2f}%>[/] flagged after "
                        f"'{fnc}' filter"
                    )

            if in_place:
                for d, f in zip(data, flg):
                    with d.open() as fl:
                        _write_filter_to_file(
                            fl, flags=f, fnc_name=fnc.__name__, **kwargs
                        )
                        # this resets weights so that it will use the new flags.
                        del d.weights

            return flg

        return wrapper

    return inner


def _write_filter_to_file(fl: h5py.File, flags: np.ndarray, fnc_name, **kwargs):
    if "flags" not in fl:
        flg_grp = fl.create_group("flags")
        flg_grp.create_dataset(
            "flags", data=flags, shape=(1,) + flags.size, maxshape=(None,) + flags.shape
        )
    else:
        flg_grp = fl["flags"]
        flg_data = fl["flags"]["flags"]
        flg_data.resize(flg_data.shape[0] + 1, axis=0)
        flg_data[-1] = flags

    n_filters = len(dict(flg_grp.attrs))

    while fnc_name in flg_grp.attrs:
        fnc_name += "+"

    flg_grp.attrs[fnc_name] = n_filters
    meta_grp = flg_grp.create_group(fnc_name)

    for k, v in kwargs.items():
        meta_grp.attrs[k] = v


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
        flags = np.where(
            np.isnan(info.metric) | np.isinf(info.metric), True, info.flags
        )

        mmodel = metric_model.fit(
            xdata=info.gha, ydata=info.metric, weights=(~flags).astype(float)
        )

        flags = np.where(np.isnan(info.std) | np.isinf(info.std), True, info.flags)
        smodel = std_model.fit(
            xdata=info.gha, ydata=info.std, weights=(~flags).astype(float)
        )

        return cls(
            metric_model=mmodel.fit,
            std_model=smodel.fit,
            n_sigma=n_sigma,
            meta=meta or {},
        )

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

            return cls(
                metric_model=metric_model,
                std_model=std_model,
                n_sigma=n_sigma,
                meta=meta,
            )

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
    files: list[tp.PathLike | FilteredData | CalibratedData]
    indx_map: list[np.ndarray]

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
        self, model_filter: GHAModelFilter, params: np.ndarray | None = None
    ) -> np.ndarray:
        """Get the residuals of the metric to a smooth model fit over all GHA."""
        return self.metric - model_filter.metric_model(parameters=params)


@attr.s(frozen=True)
class FrequencyAggregator(metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def aggregate_file(self, data: FilteredData | CalibratedData) -> np.ndarray:
        """Actually aggregate over frequency."""
        raise NotImplementedError

    def aggregate(
        self, data: Sequence[tp.PathLike | FilteredData | CalibratedData]
    ) -> np.ndarray | np.ndarray | list[np.ndarray]:
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
        indx_map
            A list of integer arrays that give the index of each file's GHAs back to
            the sorted array.
        """
        # Convert the data to actual Step objects
        data = [read_step(d) for d in data]

        # get the total number of GHAs in all the files
        n_ghas = [len(d.gha) for d in data]
        n_gha_total = sum(n_ghas)

        # Put all the GHA's together
        gha = np.concatenate(tuple(d.gha for d in data))

        # Get the aggregated metrics for all input files.
        metric = np.empty(n_gha_total)
        count = 0
        for n_gha, datafile in zip(n_ghas, data):
            metric[count : count + n_gha] = self.aggregate_file(datafile)
            count += n_gha

        # Now, sort the output
        sortidx = np.argsort(gha)

        # Build a dictionary mapping indices in the sorted array to file/gha index in
        # the files
        indx_map = []
        counter = 0
        for n in n_ghas:
            indx_map.append(sortidx[counter : counter + n])
            counter += n

        return gha[sortidx], metric[sortidx], indx_map


def get_gha_model_filter(
    data: Sequence[tp.PathLike | FilteredData | CalibratedData],
    aggregator: FrequencyAggregator,
    metric_model_type: tp.Modelable,
    std_model_type: tp.Modelable,
    metric_model_kwargs: dict | None = None,
    std_model_kwargs: dict | None = None,
    detrend_metric_model: Model | None = None,
    detrend_std_model: Model | None = None,
    detrend_gha_chunk_size: float = 24.0,
    n_sigma: float = 3.0,
    std_estimator: str = "absres",
    medfilt_width: int = 100,
) -> tuple[GHAModelFilter, GHAModelFilterInfo]:
    """Obtain a filtering object from a given set of representative data.

    The algorithm here is to first intrinsically flag the input data, then fit a model
    to it over GHA, which can be used to flag further files. The initial intrinsice
    flagging is *by default* done with the same model that will be applied to other
    data, but it can be done at a more fine-grained level, fitting to small chunks of
    GHA at a time with a lower-order model.

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
    gha, metric, indx_map = aggregator.aggregate(data)

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

    info = GHAModelFilterInfo(
        gha=gha, metric=metric, std=std, flags=flags, files=data, indx_map=indx_map
    )
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
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
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
        absolute residuals), 'std' (to use a simple standard deviation within the
        window) or 'mad' (to use a simple median absolute deviation within the window).

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
                    resid ** 2,
                    size=2 * (medfilt_width // 2) + 1,
                    kind="median",
                    flags=this_flg,
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
            raise ValueError(
                "std_estimator must be one of 'medfilt', 'absres','std' or 'mad'."
            )

        std = std_model.fit(ydata=np.abs(resid), xdata=x, weights=wght).evaluate()

        this_flg = flags | (np.abs(resid) > n_sigma * std)

    return this_flg, resid, std


def chunked_iterative_model_filter(
    *,
    x: np.ndarray,
    data: np.ndarray,
    flags: np.ndarray | None = None,
    init_flags: np.ndarray | None = None,
    chunk_size: float = np.inf,
    **kwargs,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
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
            x=x[mask],
            data=data[mask],
            flags=out_flags[mask],
            init_flags=init_flags[mask],
            **kwargs,
        )
        xmin += chunk_size

    return out_flags, resids, std


@attr.s(frozen=True)
class RMSAggregator(FrequencyAggregator):
    """An aggregator that fits a model and yields the RMS over a given freq range."""

    model_type: tp.Modelable = attr.ib(default="LinLog")
    band: tuple[float, float] = attr.ib(default=(0, np.inf))
    model_kwargs: dict | None = attr.ib(factory=dict)

    def aggregate_file(self, data: FilteredData | CalibratedData) -> np.ndarray:
        """Compute the RMS over frequency for each integration in a file."""
        return data.get_model_rms(
            freq_range=self.band, model=self.model_type, **self.model_kwargs
        )


@attr.s(frozen=True)
class TotalPowerAggregator(FrequencyAggregator):
    """An aggregator that fits a model and yields the RMS over a given freq range."""

    band: tuple[float, float] = attr.ib(default=(0, np.inf))

    def aggregate_file(self, data: FilteredData | CalibratedData) -> np.ndarray:
        """Compute the total power over frequency for each integration in a file."""
        freq_mask = (data.raw_frequencies >= self.band[0]) & (
            data.raw_frequencies < self.band[1]
        )
        weights = np.sum(data.weights[:, freq_mask], axis=1)
        return np.where(
            weights > 0,
            (
                np.sum(data.spectrum[:, freq_mask] * data.weights[:, freq_mask], axis=1)
                / weights
            ),
            np.nan,
        )


def apply_gha_model_filter(
    data: Sequence[tp.PathLike | FilteredData | CalibratedData],
    aggregator: FrequencyAggregator,
    filt: GHAModelFilter | None = None,
    n_files: int = 0,
    use_intrinsic_flags: bool = True,
    out_file: tp.PathLike | None = None,
    **kwargs,
) -> list[np.ndarray[bool]]:
    """Apply a GHA-based model filter to a set of data files.

    Parameters
    ----------
    data

    """
    if n_files == 0:
        n_files = len(data)

    info = None
    if filt is None:
        filt, info = get_gha_model_filter(
            data=data[:n_files], aggregator=aggregator, **kwargs
        )

        # If desired, go ahead and just flag the objects themselves.
        # Can only do this if we just computed the filter.
        if use_intrinsic_flags:
            for i, d in enumerate(data[:n_files]):
                dd = read_step(d)
                dd.weights[info.indx_map[i]] = info.flags

        if out_file:
            filt.write(out_file)

    flags = []
    for d in data[n_files:]:
        dd = read_step(d)
        metric = aggregator.aggregate_file(dd)
        flags.append(filt.apply_filter(gha=dd.gha, metric=metric))

    return flags


def mean_power_model(gha, nu_min, nu_max, beta=-2.5):
    """A really rough model of expected mean power between two frequencies."""
    t75 = 1750 * np.cos(np.pi * gha / 12) + 3250  # approximate model based on haslam
    return (
        t75
        / ((beta + 1) * 75.0 ** beta)
        * (nu_max ** (beta + 1) - nu_min ** (beta + 1))
    ) / (nu_max - nu_min)


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
        with open(bad) as fl:
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
    gha_range: tuple[float, float] = (0, 24),
    sun_el_max: float = 90,
    moon_el_max: float = 90,
    amb_hum_max: float = 200,
    min_receiver_temp: float = 0,
    max_receiver_temp: float = 100,
) -> np.ndarray:
    """Flag on auxiliary data."""
    flags = np.zeros(len(gha), dtype=bool)

    def filt(condition, message, flags):
        nflags = np.sum(flags)

        flags |= condition
        nnew = np.sum(flags) - nflags
        if nnew:
            logger.debug(f"{nnew}/{len(flags) - nflags} times flagged due to {message}")

    filt((gha < gha_range[0]) | (gha >= gha_range[1]), "GHA range", flags)
    filt(sun_el > sun_el_max, "sun position", flags)
    filt(moon_el > moon_el_max, "moon position", flags)
    filt(humidity > amb_hum_max, "humidity", flags)
    filt(
        (receiver_temp >= max_receiver_temp) | (receiver_temp <= min_receiver_temp),
        "receiver temp",
        flags,
    )

    return flags


@step_filter(axis="time", data_type=CalibratedData)
def aux_filter(
    *,
    data: CalibratedData,
    sun_el_max: float = 90,
    moon_el_max: float = 90,
    ambient_humidity_max: float = 40,
    min_receiver_temp: float = 0,
    max_receiver_temp: float = 100,
) -> np.ndarray:
    """
    Perform an auxiliary filter on the object.

    Parameters
    ----------
    sun_el_max
        Maximum elevation of the sun to keep.
    moon_el_max
        Maximum elevation of the moon to keep.
    ambient_humidity_max
        Maximum ambient humidity to keep.
    min_receiver_temp
        Minimum receiver temperature to keep.
    max_receiver_temp
        Maximum receiver temp to keep.

    Returns
    -------
    flags
        Boolean array giving which entries are bad.
    """
    return time_filter_auxiliary(
        gha=data.ancillary["gha"],
        sun_el=data.ancillary["sun_el"],
        moon_el=data.ancillary["moon_el"],
        humidity=data.ancillary["ambient_hum"],
        receiver_temp=data.ancillary["receiver_temp"],
        sun_el_max=sun_el_max,
        moon_el_max=moon_el_max,
        amb_hum_max=ambient_humidity_max,
        min_receiver_temp=min_receiver_temp,
        max_receiver_temp=max_receiver_temp,
    )


@step_filter(axis="both")
def rfi_filter(
    *,
    data: _ReductionStep,
    flags: np.ndarray[bool],
    xrfi_pipe: dict,
    n_threads: int = cpu_count(),
) -> np.ndarray:
    """
    Perform xRFI for a data file.

    Parameters
    ----------
    xrfi_pipe
        A dictionary with keys specifying RFI function names, and values being
        dictionaries of parameters to pass to the function.

    Returns
    -------
    flags
        The boolean flag array, specifying which freqs/times are bad.
    """
    if "explicit" in xrfi_pipe:
        kwargs = xrfi_pipe.pop("explicit")

        if kwargs["file"] is None:
            known_rfi_file = DATA_PATH / "known_rfi_channels.yaml"
        else:
            known_rfi_file = kwargs["file"]

        flags |= rfi.xrfi_explicit(
            data.raw_frequencies,
            rfi_file=known_rfi_file,
        )

        if np.all(flags):
            return flags

    return tools.run_xrfi_pipe(
        spectrum=data.spectrum,
        freq=data.raw_frequencies,
        flags=flags,
        xrfi_pipe=xrfi_pipe,
        n_threads=n_threads,
        fl_id=data.datestring,  # TODO: this is not useful for combined files.
    )


@step_filter(axis="time", multi_data=True, data_type=CalibratedData)
def rms_filter(
    *,
    data: Sequence[CalibratedData],
    flags: np.ndarray[bool],
    band: tuple[float, float] = (0, np.inf),
    model_type: tp.Modelable = "linlog",
    model_kwargs: dict[str, Any] | None = None,
    **kwargs,
) -> np.ndarray[bool]:
    """Perform a filter on the RMS of an integration.

    Parameters
    ----------
    band
        A frequency range in which to fit the model and take the RMS.
    model_type
        The type of :class:`~edges_cal.modelling.Model` to fit to each spectrum.
    model_kwargs
        Other arguments to the model.
    **kwargs
        Other arguments to :func:`apply_gha_model_filter`.
    """
    return apply_gha_model_filter(
        data=data,
        flags=flags,
        aggregator=RMSAggregator(
            band=band, model_type=model_type, model_kwargs=model_kwargs
        ),
        **kwargs,
    )


@step_filter(axis="time", multi_data=True, data_type=CalibratedData)
def total_power_filter(
    *,
    data: Sequence[CalibratedData],
    flags: np.ndarray[bool],
    band: tuple[float, float] = (0, np.inf),
    **kwargs,
):
    """Perform a filter on the total power of an integration.

    Parameters
    ----------
    band
        A frequency range in which to fit the model and take the RMS.
    **kwargs
        Other arguments to :func:`apply_gha_model_filter`.
    """
    return apply_gha_model_filter(
        data=data, flags=flags, aggregator=TotalPowerAggregator(band=band), **kwargs
    )


@step_filter(axis="time", data_type=CalibratedData)
def negative_power_filter(*, data: CalibratedData):
    """Filter out integrations that have *any* negative/zero power.

    These integrations obviously have some weird stuff going on.
    """
    return np.array([np.any(spec <= 0) for spec in data.spectrum])
