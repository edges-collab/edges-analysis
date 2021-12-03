"""Functions that identify and flag bad data in various ways."""
from __future__ import annotations

import logging
from multiprocessing import cpu_count
from typing import Sequence, Callable, Literal
from .levels import CalibratedData, read_step, _ReductionStep, RawData, CombinedData
from . import types as tp
import attr
import h5py
import numpy as np
import yaml
from edges_cal.modelling import Model, LinLog, FourierDay
from edges_cal.xrfi import (
    ModelFilterInfoContainer,
    model_filter,
)
import abc
import functools
from .data import DATA_PATH
from . import tools
from edges_cal import xrfi as rfi
import inspect
import p_tqdm
from pathlib import Path
from .averaging import weighted_mean
import datetime
from edges_io.utils import ymd_to_jd

logger = logging.getLogger(__name__)

STEP_FILTERS = {}


def get_step_filter(filt: str) -> Callable:
    """Obtain a registered step filter function from a string name."""
    if filt not in STEP_FILTERS:
        raise KeyError(f"'{filt}' does not exist as a filter.")
    return STEP_FILTERS[filt]


def step_filter(
    axis: Literal["time", "freq", "day", "all"],
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
        uses_flags = "flags" in inspect.signature(fnc).parameters

        @functools.wraps(fnc)
        def wrapper(
            *,
            data: Sequence[tp.PathLike | _ReductionStep],
            flags: Sequence[np.ndarray] | None | list[None] = None,
            in_place: bool = False,
            n_threads: int = 1,
            **kwargs,
        ) -> list[np.ndarray]:
            logger.info(f"Running {fnc.__name__} filter.")

            # Read all the data, in case they haven't been turned into objects yet.
            # And check that everything is the right type.
            if not hasattr(data, "__len__"):
                data = [data]
            data = [read_step(d) for d in data]
            assert all(isinstance(d, data_type) for d in data)

            def per_file_processing(data, init_flags, input_flags, out_flags):
                # This function does all the processing of the output flags
                # for a particular file.
                if input_flags is not None and np.any(init_flags != input_flags):
                    logger.warning(
                        f"Filter {fnc.__name__} modified input flags in-place. "
                        "This should not happen!"
                    )

                n = np.sum(init_flags)

                if axis in ("all", "freq"):
                    init_flags |= out_flags
                elif axis == "day":
                    init_flags[out_flags] = True
                elif axis == "time":
                    init_flags[..., out_flags, :] = True

                if np.all(init_flags):
                    logger.warning(
                        f"{data.filename.name} was fully flagged during {fnc.__name__} "
                        "filter"
                    )
                else:
                    logger.info(
                        f"'{data.filename.name}': {100 * n / init_flags.size:.2f} → "
                        f"{100 * np.sum(init_flags) / init_flags.size:.2f}% [red]<+"
                        f"{100 * (np.sum(init_flags) - n) / init_flags.size:.2f}%>[/] "
                        f"flagged after '{fnc.__name__}' filter"
                    )

                if in_place:
                    with data.open("r+") as fl:
                        _write_filter_to_file(
                            fl, flags=init_flags, fnc_name=fnc.__name__, **kwargs
                        )

                        try:
                            del data.weights
                        except AttributeError:
                            pass

                    data.clear()

                    return None
                else:
                    return init_flags

            if multi_data:
                flg = (
                    [d.get_flags() for d in data]
                    if flags is None
                    else [f.copy() for f in flags]
                )

                if flags is None:
                    flags = [None] * len(data)

                if uses_flags:
                    this_flag = fnc(data=data, flags=flg, **kwargs)
                else:
                    this_flag = fnc(data=data, **kwargs)

                for d, init_flg, inp_flg, out_flg in zip(data, flg, flags, this_flag):
                    per_file_processing(d, init_flg, inp_flg, out_flg)

            else:
                if flags is None:
                    flags = [None] * len(data)

                def fnc_(data, input_flags):
                    init_flags = (
                        data.get_flags() if input_flags is None else input_flags.copy()
                    )

                    if uses_flags:
                        out = fnc(data=data, flags=init_flags, **kwargs)
                    else:
                        out = fnc(data=data, **kwargs)

                    return per_file_processing(data, init_flags, input_flags, out)

                if n_threads > 1:
                    flg = list(
                        p_tqdm.p_map(
                            fnc_, data, flags, unit="files", num_cpus=n_threads
                        )
                    )
                else:
                    flg = list(p_tqdm.t_map(fnc_, data, flags, unit="files"))

            return flg

        STEP_FILTERS[fnc.__name__] = wrapper
        return wrapper

    return inner


def _write_filter_to_file(fl: h5py.File, flags: np.ndarray, fnc_name, **kwargs):
    flg_grp = fl.create_group("flags") if "flags" not in fl else fl["flags"]

    if "flags" not in flg_grp:
        flg_grp.create_dataset(
            "flags",
            data=flags,
            shape=(1,) + flags.shape,
            maxshape=(None,) + flags.shape,
        )
    else:
        flg_data = fl["flags"]["flags"]
        flg_data.resize(flg_data.shape[0] + 1, axis=0)
        flg_data[-1] = flags

    n_filters = len(dict(flg_grp.attrs))

    while fnc_name in flg_grp.attrs:
        fnc_name += "+"

    flg_grp.attrs[fnc_name] = n_filters
    try:
        meta_grp = flg_grp.create_group(fnc_name)
    except ValueError:
        logger.debug(f"Trying to create group '{fnc_name}' that already exists.")

    for k, v in kwargs.items():
        try:
            if isinstance(v, dict):
                for kk, vv in v.items():
                    meta_grp.attrs[k + "_" + kk] = vv
            elif isinstance(v, Model):
                meta_grp.attrs[k] = yaml.dump(v)
            else:
                meta_grp.attrs[k] = v
        except TypeError:
            logger.warning(
                f"Metadata key '{k}' has value of type '{type(v)}' which can't be saved"
                " in HDF5. Continuing without writing it."
            )


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
            metric_model=mmodel.fit.model,
            std_model=smodel.fit.model,
            n_sigma=n_sigma,
            meta=meta or {},
        )

    @classmethod
    def from_file(cls, fname: tp.PathLike) -> GHAModelFilter:
        """Create the class from a h5 file."""
        with h5py.File(fname, "r") as fl:
            metric_model = yaml.load(fl.attrs["metric_model"])
            std_model = yaml.load(fl.attrs["std_model"])

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
            fl.attrs["metric_model"] = yaml.dump(self.metric_model)
            fl.attrs["std_model"] = yaml.dump(self.std_model)

            fl.attrs["n_sigma"] = self.n_sigma
            meta = fl.create_group("meta")
            for k, v in self.meta.items():
                try:
                    meta[k] = v
                except TypeError:
                    if isinstance(v, Model):
                        meta[k] = yaml.dump(v)
                    else:
                        logger.warning(f"Key '{k}' was unable to be written.")


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
    files: list[tp.PathLike | CalibratedData]
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

    def write(self, fname: tp.PathLike):
        """Write the object to H5 file."""
        with h5py.File(fname, "w") as fl:
            fl["gha"] = self.gha
            fl["metric"] = self.metric
            fl["std"] = self.std
            fl["flags"] = self.flags

            for i, idx in enumerate(self.indx_map):
                fl[f"idx_map_{i}"] = idx

            fl.attrs["files"] = ":".join(str(f) for f in self.files)

    @classmethod
    def from_file(cls, fname: tp.PathLike):
        """Create an object from a file."""
        with h5py.File(fname, "r") as fl:
            gha = fl["gha"][...]
            metric = fl["metric"][...]
            std = fl["std"][...]
            flags = fl["flags"][...]

            idx_map = [
                fl[key][...] for key in sorted(fl.keys()) if key.startswith("idx_map")
            ]

            files = fl.attrs["files"].split(":")

        return GHAModelFilterInfo(
            gha=gha, metric=metric, std=std, flags=flags, indx_map=idx_map, files=files
        )


@attr.s(frozen=True)
class FrequencyAggregator(metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def aggregate_file(self, data: CalibratedData) -> np.ndarray:
        """Actually aggregate over frequency."""
        raise NotImplementedError

    def get_init_flags(self, gha, metric):
        """Base function to define some inital set of flags."""
        return np.zeros(len(gha), dtype=bool)

    def aggregate(
        self, data: Sequence[tp.PathLike | CalibratedData]
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
        invsortidx = np.argsort(sortidx)

        # Build a dictionary mapping indices in the sorted array to file/gha index in
        # the files
        indx_map = []
        counter = 0
        for n in n_ghas:
            indx_map.append(invsortidx[counter : counter + n])
            counter += n

        return gha[sortidx], metric[sortidx], indx_map


def get_gha_model_filter(
    data: Sequence[tp.PathLike | CalibratedData],
    aggregator: FrequencyAggregator,
    metric_model: Model,
    std_model: Model,
    detrend_metric_model: Model | None = None,
    detrend_std_model: Model | None = None,
    detrend_gha_chunk_size: float = 24.0,
    n_resid: int = 1,
    **kwargs,
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
    metric_model
        A linear model to be fit to the aggregated metric data.
    std_model
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
    **kwargs
        All other arguments passed to :func:`edges_cal.xrfi.model_filter`

    Returns
    -------
    filter
        An object that can be used to filter other files based on the same aggregation.
    info
        An object containing information about the fit itself -- useful for inspection.

    """
    # Aggregate the data for each file along the frequency axis.
    gha, metric, indx_map = aggregator.aggregate(data)
    init_flags = aggregator.get_init_flags(gha, metric)

    if detrend_metric_model is None:
        detrend_metric_model = metric_model
    if detrend_std_model is None:
        detrend_std_model = std_model

    flags, resid, std, flag_info = chunked_iterative_model_filter(
        x=gha,
        data=metric,
        init_flags=init_flags,
        flags=np.isnan(metric) | np.isinf(metric),
        model=detrend_metric_model,
        resid_model=detrend_std_model,
        chunk_size=detrend_gha_chunk_size,
        n_resid=n_resid,
        **kwargs,
    )

    info = GHAModelFilterInfo(
        gha=gha,
        metric=metric,
        std=std,
        flags=flags,
        files=[d.filename for d in data],
        indx_map=indx_map,
    )
    return (
        GHAModelFilter.from_data(
            info,
            metric_model,
            std_model,
            n_sigma=flag_info.thresholds[-1],
            meta={
                **{
                    "infiles": ":".join(
                        str(getattr(d, "filename", str(d))) for d in data
                    ),
                    "gha_chunk_size": detrend_gha_chunk_size,
                    "detrend_model": detrend_metric_model,
                    "detrend_std_model": detrend_std_model,
                },
                **kwargs,
            },
        ),
        info,
        flag_info,
    )


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
    :func:`edges_cal.xrfi.model_filter` on each chunk, returning the full 1D array of
    flags after all the chunks have been processed.

    Parameters
    ----------
    chunk_size
        The size of the chunks to process, in units of the input coordinates, ``x``.
    **kwargs
        Everything else is passed to :func:`edges_cal.xrfi.model_filter`.

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
    infos = ModelFilterInfoContainer()
    while xmin < x.max():
        mask = (x >= xmin) & (x < xmin + chunk_size)

        out_flags[mask], info = model_filter(
            x=x[mask],
            data=data[mask],
            flags=out_flags[mask],
            init_flags=init_flags[mask],
            **kwargs,
        )
        resids[mask] = info.get_residual()
        std[mask] = info.stds[-1]
        infos = infos.append(info)
        xmin += chunk_size

    return out_flags, resids, std, infos


@attr.s(frozen=True)
class RMSAggregator(FrequencyAggregator):
    """An aggregator that fits a model and yields the RMS over a given freq range."""

    model: Model = attr.ib(default=LinLog(n_terms=5))
    band: tuple[float, float] = attr.ib(default=(0, np.inf))

    def aggregate_file(self, data: CalibratedData) -> np.ndarray:
        """Compute the RMS over frequency for each integration in a file."""
        return data.get_model_rms(freq_range=self.band, model=self.model)


@attr.s(frozen=True)
class TotalPowerAggregator(FrequencyAggregator):
    """An aggregator that fits a model and yields the mean over a given freq range."""

    band: tuple[float, float] = attr.ib(default=(0, np.inf))
    model: Model = attr.ib(default=FourierDay(n_terms=40))
    init_threshold: float = attr.ib(default=1.0)

    def get_init_flags(self, gha, metric):
        """Compute the inital flags based on the power in a simulated spectra."""
        fiducial_data = np.load(
            DATA_PATH / "Lowband_30mx30m_Haslam_2p5_20minlst_50_100.npy",
            allow_pickle=True,
        )
        band_mask = (fiducial_data[1] >= self.band[0]) & (
            fiducial_data[1] <= self.band[1]
        )
        fiducial_spectrum = fiducial_data[0][:, band_mask]

        standard_model = (
            self.model.at(x=fiducial_data[2])
            .fit(ydata=np.nanmean(fiducial_spectrum, axis=1))
            .fit
        )
        return (
            np.abs(standard_model(gha) - metric) / standard_model(gha)
        ) > self.init_threshold

    def aggregate_file(self, data: CalibratedData) -> np.ndarray:
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
    data: Sequence[tp.PathLike | CalibratedData],
    aggregator: FrequencyAggregator,
    filt: GHAModelFilter | tp.PathLike | None = None,
    n_files: int = 0,
    use_intrinsic_flags: bool = True,
    out_file: tp.PathLike | None = None,
    write_info: bool = True,
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
    flags = []
    if filt is None:
        filt, info, flag_info = get_gha_model_filter(
            data=data[:n_files], aggregator=aggregator, **kwargs
        )

        # If desired, go ahead and just flag the objects themselves.
        # Can only do this if we just computed the filter.
        if use_intrinsic_flags:
            for i in range(n_files):
                flags.append(info.flags[info.indx_map[i]])

        if write_info:
            if out_file is None:
                hsh = hash(
                    "".join(f"{k}:{repr(v)}" for k, v in kwargs.items())
                    + ":".join(str(d) for d in data[:n_files])
                )
                out_file = Path("GHAModel_" + str(aggregator) + str(hsh))
            else:
                out_file = Path(out_file)

            filt.write(out_file.with_suffix(".filter"))
            info.write(out_file.with_suffix(".info"))
            flag_info.write(out_file.with_suffix(".flag_info"))

    elif not isinstance(filt, GHAModelFilter):
        filt = GHAModelFilter.from_file(filt)

    for i, d in enumerate(data[len(flags) :]):
        dd = read_step(d)
        metric = aggregator.aggregate_file(dd)
        flags.append(filt.apply_filter(gha=dd.gha, metric=metric))

    return flags


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
    adcmax: np.ndarray,
    gha_range: tuple[float, float] = (0, 24),
    sun_el_max: float = 90,
    moon_el_max: float = 90,
    amb_hum_max: float = 200,
    min_receiver_temp: float = 0,
    max_receiver_temp: float = 100,
    adcmax_max: float = 0.4,
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
    filt(adcmax > adcmax_max, "adc max level", flags)

    return flags


@step_filter(axis="time", data_type=(RawData, CalibratedData))
def aux_filter(
    *,
    data: CalibratedData,
    sun_el_max: float = 90,
    moon_el_max: float = 90,
    ambient_humidity_max: float = 40,
    min_receiver_temp: float = 0,
    max_receiver_temp: float = 100,
    adcmax_max: float = 0.4,
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
    adcmax_max
        Maximum adcmax level to keep.

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
        adcmax=data.ancillary["adcmax"][:, 0],
        sun_el_max=sun_el_max,
        moon_el_max=moon_el_max,
        amb_hum_max=ambient_humidity_max,
        min_receiver_temp=min_receiver_temp,
        max_receiver_temp=max_receiver_temp,
        adcmax_max=adcmax_max,
    )


def _rfi_filter_factory(method: str):
    def fnc(
        *,
        data: _ReductionStep,
        flags: np.ndarray[bool],
        n_threads: int = cpu_count(),
        freq_range: tuple[float, float] = (40, 200),
        **kwargs,
    ) -> np.ndarray:

        mask = (data.raw_frequencies >= freq_range[0]) & (
            data.raw_frequencies <= freq_range[1]
        )

        out_flags = tools.run_xrfi(
            method=method,
            spectrum=data.spectrum[..., mask],
            freq=data.raw_frequencies[mask],
            flags=flags[..., mask],
            weights=data.weights[..., mask],
            n_threads=n_threads,
            **kwargs,
        )

        out = np.zeros_like(flags)
        out[..., mask] = out_flags

        return out

    fnc.__name__ = f"rfi_{method}_filter"

    return step_filter(axis="all")(fnc)


rfi_model_filter = _rfi_filter_factory("model")
rfi_model_sweep_filter = _rfi_filter_factory("model_sweep")
rfi_watershed_filter = _rfi_filter_factory("watershed")


@step_filter(axis="freq")
def rfi_explicit_filter(*, data: _ReductionStep, file: tp.PathLike | None = None):
    """A filter of explicit channels of RFI."""
    if file is None:
        file = DATA_PATH / "known_rfi_channels.yaml"

    return rfi.xrfi_explicit(
        data.raw_frequencies,
        rfi_file=file,
    )


@step_filter(axis="time", multi_data=True, data_type=CalibratedData)
def rms_filter(
    *,
    data: Sequence[CalibratedData],
    bands: Sequence[tuple[float, float]] = ((0, np.inf),),
    model: Model = LinLog(n_terms=5),
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
    flags = [
        apply_gha_model_filter(
            data=data,
            aggregator=RMSAggregator(band=band, model=model),
            **kwargs,
        )
        for band in bands
    ]
    return np.any(flags, axis=0)


@step_filter(axis="time", multi_data=True, data_type=CalibratedData)
def total_power_filter(
    *,
    data: Sequence[CalibratedData],
    bands: Sequence[tuple[float, float]] = ((0, np.inf),),
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
    flags = [
        apply_gha_model_filter(
            data=data, aggregator=TotalPowerAggregator(band=band), **kwargs
        )
        for band in bands
    ]

    return [np.any([f[iday] for f in flags], axis=0) for iday in range(len(flags[0]))]


@step_filter(axis="time", data_type=(RawData, CalibratedData))
def negative_power_filter(*, data: CalibratedData):
    """Filter out integrations that have *any* negative/zero power.

    These integrations obviously have some weird stuff going on.
    """
    return np.array([np.any(spec <= 0) for spec in data.spectrum])


def _peak_power_filter(
    *,
    data: RawData | CalibratedData,
    threshold: float = 40.0,
    peak_freq_range: tuple[float, float] = (80, 200),
    mean_freq_range: tuple[float, float] | None = None,
):
    """
    Filters out whole integrations that have high power > 80 MHz.

    Parameters
    ----------
    threshold
        This is the threshold beyond which the peak power causes the integration to be
        flagged. The units of the threhsold are 10*log10(peak_power / mean), where the
        mean is the mean power of spectrum in the same frequency range (omitting
        power spikes > peak_power/10)
    peak_freq_range
        The range of frequencies over which to search for the peak.
    mean_freq_range
        The range of frequencies over which to take a mean to compare to the peak.
        By default, the same as the ``peak_freq_range``.
    """
    if peak_freq_range[0] >= peak_freq_range[1]:
        raise ValueError(
            f"The frequency range of the peak must be non-zero, got {peak_freq_range}"
        )

    if mean_freq_range is not None and mean_freq_range[0] >= mean_freq_range[1]:
        raise ValueError(
            f"The frequency range of the peak must be non-zero, got {peak_freq_range}"
        )

    mask = (data.raw_frequencies > peak_freq_range[0]) & (
        data.raw_frequencies <= peak_freq_range[1]
    )

    if not np.any(mask):
        return np.zeros(len(data.spectrum), dtype=bool)

    spec = data.spectrum[:, mask]
    peak_power = spec.max(axis=1)

    if mean_freq_range is not None:
        mask = (data.raw_frequencies > mean_freq_range[0]) & (
            data.raw_frequencies <= mean_freq_range[1]
        )
        if not np.any(mask):
            return np.zeros(len(data.spectrum), dtype=bool)

        spec = data.spectrum[:, mask]

    mean, _ = weighted_mean(
        spec,
        weights=((spec > 0) & ((spec.T < peak_power / 10).T)).astype(float),
        axis=1,
    )
    peak_power = 10 * np.log10(peak_power / mean)
    return peak_power > threshold


@step_filter(axis="time", data_type=(RawData, CalibratedData))
def peak_power_filter(
    *,
    data: RawData | CalibratedData,
    threshold: float = 40.0,
    peak_freq_range: tuple[float, float] = (80, 200),
    mean_freq_range: tuple[float, float] | None = None,
):
    """
    Filters out whole integrations that have high power > 80 MHz.

    Parameters
    ----------
    threshold
        This is the threshold beyond which the peak power causes the integration to be
        flagged. The units of the threhsold are 10*log10(peak_power / mean), where the
        mean is the mean power of spectrum in the same frequency range (omitting
        power spikes > peak_power/10)
    peak_freq_range
        The range of frequencies over which to search for the peak.
    mean_freq_range
        The range of frequencies over which to take a mean to compare to the peak.
        By default, the same as the ``peak_freq_range``.
    """
    return _peak_power_filter(
        data=data,
        threshold=threshold,
        peak_freq_range=peak_freq_range,
        mean_freq_range=mean_freq_range,
    )


@step_filter(axis="time", data_type=(RawData, CalibratedData))
def peak_orbcomm_filter(
    *,
    data: RawData | CalibratedData,
    threshold: float = 40.0,
    mean_freq_range: tuple[float, float] | None = (80, 200),
):
    """
    Filters out whole integrations that have high power between (137, 138) MHz.

    Parameters
    ----------
    threshold
        This is the threshold beyond which the peak power causes the integration to be
        flagged. The units of the threhsold are 10*log10(peak_power / mean), where the
        mean is the mean power of spectrum in the ``mean_freq_range`` (omitting
        power spikes > peak_power/10)
    mean_freq_range
        The range of frequencies over which to take a mean to compare to the peak.
        By default, the same as the ``peak_freq_range``.
    """
    return _peak_power_filter(
        data=data,
        threshold=threshold,
        peak_freq_range=(137.0, 138.0),
        mean_freq_range=mean_freq_range,
    )


@step_filter(axis="time", data_type=(RawData, CalibratedData))
def maxfm_filter(*, data: CalibratedData, threshold: float = 200):
    """Max FM power filter.

    This takes power of the spectrum between 80 MHz and 120 MHz(the fm range).
    In that range, it checks each frequency bin to the estimated values..
    using the mean from the side bins.
    And then takes the max of all the all values that exceeded its expected..
    value (from mean).
    Compares the max exceeded power with the threshold and if it is greater
    than the threshold given, the integration will be flagged.
    """
    fm_freq = (data.raw_frequencies >= 88) & (data.raw_frequencies <= 120)
    # freq mask between 80 and 120 MHz for the FM range

    if not np.any(fm_freq):
        return np.zeros(len(data.spectrum), dtype=bool)

    fm_power = data.spectrum[:, fm_freq]

    avg = (fm_power[:, 2:] + fm_power[:, :-2]) / 2
    fm_deviation_power = np.abs(fm_power[:, 1:-1] - avg)
    maxfm = np.max(fm_deviation_power, axis=1)

    return maxfm > threshold


@step_filter(axis="time", data_type=(RawData, CalibratedData))
def rmsf_filter(
    *,
    data: CalibratedData | RawData,
    threshold: float = 200,
    freq_range: tuple[float, float] = (60, 80),
    tload: float = 1000,
    tcal: float = 300,
):
    """
    Rmsf filter - filters out based on rms calculated between 60 and 80 MHz.

    An initial powerlaw model is calculated using the normalized frequency range.
    Data between the freq_range is clipped.
    A standard deviation is calculated using the data and the init_model.
    Then rms is calculated from the mean that is eatimated
    using the standard deviation times initmodel.
    """
    freq_mask = (data.raw_frequencies >= freq_range[0]) & (
        data.raw_frequencies <= freq_range[1]
    )

    if not np.any(freq_mask):
        return np.zeros(len(data.spectrum), dtype=bool)

    semi_calibrated_data = (data.spectrum * tload) + tcal
    freq = data.raw_frequencies[freq_mask]
    init_model = (freq / 75.0) ** -2.5

    T75 = np.sum(init_model * semi_calibrated_data[:, freq_mask], axis=1) / np.sum(
        init_model ** 2
    )

    rms = np.sqrt(
        np.mean(
            (semi_calibrated_data[:, freq_mask] - np.outer(T75, init_model)) ** 2,
            axis=1,
        )
    )

    return rms > threshold


@step_filter(axis="time", data_type=(RawData, CalibratedData))
def filter_150mhz(*, data: RawData | CalibratedData, threshold: float):
    """Filter based on power around 150 MHz.

    This takes the RMS of the power around 153.5 MHz (in a 1.5 MHz bin), after
    subtracting the mean, then compares this to the mean power of a 1.5 MHz bin around
    157 MHz (which is expected to be cleaner). If this ratio (RMS to mean) is greater
    than 200 times the threshold given, the integration will be flagged.
    """
    if data.freq.max < 157:
        return np.zeros(len(data.spectrum), dtype=bool)

    freq_mask = (data.raw_frequencies >= 152.75) & (data.raw_frequencies <= 154.25)
    mean = np.mean(data.spectrum[:, freq_mask], axis=1)
    rms = np.sqrt(np.mean((data.spectrum[:, freq_mask] - mean) ** 2))

    freq_mask2 = (data.raw_frequencies >= 156.25) & (data.raw_frequencies <= 157.75)
    av = np.mean(data.spectrum[:, freq_mask2], axis=1)
    d = 200.0 * np.sqrt(rms) / av

    return d > threshold


@step_filter(axis="time", data_type=(RawData,))
def power_percent_filter(
    *,
    data: RawData,
    freq_range: tuple[float, float] = (100, 200),
    min_threshold: float = -0.7,
    max_threshold: float = 3,
):
    """Filter for the power above 100 MHz seen in swpos 0.

    Calculates the percentage of power between 100 and 200 MHz
    & when the switch is in position 0.
    And flags integrations if the percentage is above or below the given threshold.
    """
    p0 = data.spectra["switch_powers"][0]

    mask = (data.raw_frequencies > freq_range[0]) & (
        data.raw_frequencies <= freq_range[1]
    )

    if not np.any(mask):
        return np.zeros(len(data.spectrum), dtype=bool)

    ppercent = 100 * np.sum(p0[:, mask], axis=1) / np.sum(p0, axis=1)
    return (ppercent < min_threshold) | (ppercent > max_threshold)


@step_filter(axis="day", data_type=(CombinedData,))
def day_filter(
    *, data: CombinedData, dates: Sequence[datetime.datetime | tuple[int, int]]
):
    """Filter out specific days."""
    filter_dates = []
    for date in dates:
        if isinstance(date, datetime.date):
            date = (date.year, ymd_to_jd(date.year, date.month, date.day))
            filter_dates.append(date)
        elif len(date) == 3:
            filter_dates.append(tuple(date)[:2])
        elif len(date) == 2:
            filter_dates.append(tuple(date))
        else:
            raise ValueError(f"date '{date}' cannot be parsed as a date.")

    return np.array([date[:2] in filter_dates for date in data.dates])
