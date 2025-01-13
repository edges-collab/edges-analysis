"""Support for filters that compress multiple files along the LST axis."""

from __future__ import annotations

import abc
import logging
from collections.abc import Sequence
from pathlib import Path

import h5py
import numpy as np
import yaml
from astropy import units as un
from attrs import define, field
from edges_cal.modelling import FourierDay, LinLog, Model
from edges_io import types as tp
from pygsdata import GSData, GSFlag, gsregister

from ..data import DATA_PATH
from ..datamodel import add_model
from .filters import chunked_iterative_model_filter, gsdata_filter

logger = logging.getLogger(__name__)


@define(frozen=True)
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

    metric_model: Model = field()
    std_model: Model = field()
    meta: dict = field(factory=dict)

    n_sigma: float = field(default=3.0)

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
                except TypeError:  # noqa: PERF203
                    if isinstance(v, Model):
                        meta[k] = yaml.dump(v)
                    else:
                        logger.warning(f"Key '{k}' was unable to be written.")


@define(frozen=True, kw_only=True)
class GHAModelFilterInfo:
    """An object containing the data going into creating a :class:`GHAModelFIlter`.

    This is useful for saving the full information to disk, or as an intermediate object
    used when determining the model information.
    """

    gha: np.ndarray
    metric: np.ndarray
    std: np.ndarray
    flags: np.ndarray
    files: list[tp.PathLike | GSData]
    indx_map: list[np.ndarray]

    def __attrs_post_init__(self):
        """Run post-init scripts.

        This just runs validation on all of the inputs after they are set on the class.
        """
        self._validate_inputs()

    def _validate_inputs(self) -> bool:
        assert self.gha.shape == self.metric.shape
        assert self.gha.shape == self.flags.shape
        assert self.gha.shape == self.std.shape
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


@define(frozen=True)
class FrequencyAggregator(metaclass=abc.ABCMeta):
    """An aggregator over frequency."""

    @abc.abstractmethod
    def aggregate_file(self, data: GSData) -> np.ndarray:
        """Actually aggregate over frequency."""
        raise NotImplementedError

    def get_init_flags(self, gha, metric):
        """Define some inital set of flags."""
        return np.zeros(len(gha), dtype=bool)

    def aggregate(
        self, data: Sequence[tp.PathLike | GSData]
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
        data = [d if isinstance(d, GSData) else GSData.from_file(d) for d in data]

        # get the total number of GHAs in all the files
        n_ghas = [len(d.gha) for d in data]
        n_gha_total = sum(n_ghas)

        # Put all the GHA's together
        gha = np.concatenate(tuple(d.gha[:, 0] for d in data))

        # Get the aggregated metrics for all input files.
        metric = np.empty(n_gha_total)
        count = 0
        for n_gha, datafile in zip(n_ghas, data, strict=False):
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
    data: Sequence[tp.PathLike | GSData],
    aggregator: FrequencyAggregator,
    metric_model: Model,
    std_model: Model,
    detrend_metric_model: Model | None = None,
    detrend_std_model: Model | None = None,
    detrend_gha_chunk_size: un.Quantity[un.hourangle] = 24.0 * un.hourangle,
    n_resid: int = 1,
    **kwargs,
) -> tuple[GHAModelFilter, GHAModelFilterInfo]:
    """Obtain a filtering object from a given set of representative data.

    The algorithm here is to first intrinsically flag the input data, then fit a model
    to it over GHA, which can be used to flag further files. The initial intrinsic
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

    flags, _resid, std, flag_info = chunked_iterative_model_filter(
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
    model_filter = GHAModelFilter.from_data(
        info,
        metric_model,
        std_model,
        n_sigma=flag_info.thresholds[-1],
        meta={
            "infiles": ":".join(str(getattr(d, "filename", str(d))) for d in data),
            "gha_chunk_size": detrend_gha_chunk_size,
            "detrend_model": detrend_metric_model,
            "detrend_std_model": detrend_std_model,
            **kwargs,
        },
    )
    return model_filter, info, flag_info


@define(frozen=True)
class RMSAggregator(FrequencyAggregator):
    """An aggregator that fits a model and yields the RMS over a given freq range."""

    model: Model = field(default=LinLog(n_terms=5))
    band: tuple[float, float] = field(default=(0, np.inf))

    def aggregate_file(self, data: GSData) -> np.ndarray:
        """Compute the RMS over frequency for each integration in a file."""
        if data.data_unit not in ("temperature", "model_residuals"):
            raise ValueError("Can only run total power aggregator on temperature data.")

        data = add_model(data, model=self.model)

        mask = data.nsamples[0, 0] > 0
        r = data.residuals[0, 0].copy()
        r = np.where(mask, r, np.nan)
        return np.sqrt(np.nanmean(r**2, axis=-1))


@define(frozen=True)
class TotalPowerAggregator(FrequencyAggregator):
    """An aggregator that fits a model and yields the mean over a given freq range."""

    band: tuple[float, float] = field(default=(0, np.inf))
    model: Model = field(default=FourierDay(n_terms=40))
    init_threshold: float = field(default=1.0)

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

    def aggregate_file(self, data: GSData) -> np.ndarray:
        """Compute the total power over frequency for each integration in a file."""
        if data.data_unit not in ("temperature", "model_residuals"):
            raise ValueError("Can only run total power aggregator on temperature data.")

        freqs = data.freqs.to_value("MHz")
        freq_mask = (freqs >= self.band[0]) & (freqs < self.band[1])

        weights = np.sum(data.nsamples[0, 0, :, freq_mask].T, axis=-1)
        return np.where(
            weights > 0,
            (
                np.sum(
                    data.data[0, 0, :, freq_mask].T
                    * data.nsamples[0, 0, :, freq_mask].T,
                    axis=-1,
                )
                / weights
            ),
            np.nan,
        )


def apply_gha_model_filter(
    data: Sequence[tp.PathLike | GSData],
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
            flags.extend(info.flags[info.indx_map[i]] for i in range(n_files))

        if write_info:
            if out_file is None:
                hsh = hash(
                    "".join(f"{k}:{v!r}" for k, v in kwargs.items())
                    + ":".join(str(d) for d in data[:n_files])
                )
                out_file = Path(f"GHAModel_{hsh}")
            else:
                out_file = Path(out_file)

            filt.write(out_file.with_suffix(".filter"))
            info.write(out_file.with_suffix(".info"))
            flag_info.write(out_file.with_suffix(".flag_info"))

    elif not isinstance(filt, GHAModelFilter):
        filt = GHAModelFilter.from_file(filt)

    for d in data[len(flags) :]:
        dd = d if isinstance(d, GSData) else GSData.from_file(d)
        metric = aggregator.aggregate_file(dd)
        flags.append(filt.apply_filter(gha=dd.gha, metric=metric))

    return flags


@gsregister("filter")
@gsdata_filter(multi_data=True)
def rms_filter(
    *,
    data: Sequence[GSData],
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
    flags = np.any(flags, axis=0)
    return [GSFlag(flg, axes=("time",)) for flg in flags]


@gsregister("filter")
@gsdata_filter(multi_data=True)
def total_power_filter(
    *,
    data: Sequence[GSData],
    bands: Sequence[tuple[float, float]] = ((0, np.inf),),
    init_flag_threshold: float = 1.0,
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
            data=data,
            aggregator=TotalPowerAggregator(
                band=band, init_threshold=init_flag_threshold
            ),
            **kwargs,
        )
        for band in bands
    ]

    return [
        GSFlag(flags=np.any([f[iday] for f in flags], axis=0), axes=("time",))
        for iday in range(len(flags[0]))
    ]
