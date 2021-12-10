"""Module defining discrete processing levels for EDGES field data."""
from __future__ import annotations

import glob
import inspect
import json
import logging
import os
import re
import sys
import time
import copy
import warnings
from copy import deepcopy
from datetime import datetime
from multiprocessing import cpu_count
from pathlib import Path
from typing import Sequence, Callable, Any

import attr
import edges_io as io
import h5py
import matplotlib.pyplot as plt
import numpy as np
import tqdm
from cached_property import cached_property
from edges_cal import (
    FrequencyRange,
    modelling as mdl,
    Calibration,
)
from edges_io.auxiliary import auxiliary_data
from edges_io.h5 import HDF5Object, HDF5RawSpectrum

from . import averaging
from . import loss, beams, coordinates
from .coordinates import get_jd, dt_from_jd
from .. import __version__
from .. import const
from ..config import config
from .calibrate import LabCalibration
from . import types as tp
from . import coordinates as coords
import psutil

logger = logging.getLogger(__name__)


def float_array_ndim(n: int) -> Callable:
    """Define a function that validates array type and dimension."""
    return lambda x: x.ndim == n and x.dtype.name.startswith("float")


def is_array(kind: str, n: int) -> Callable:
    """Define a function that validates array type and dimension."""
    return lambda x: x.ndim == n and x.dtype.name.startswith(kind)


class FullyFlaggedError(ValueError):
    pass


class WeatherError(ValueError):
    pass


class _CombinedFileMixin:
    @property
    def days(self) -> np.ndarray:
        return np.array(self.ancillary["days"])

    @property
    def gha_edges(self) -> np.ndarray:
        """The edges of the GHA bins."""
        return np.array(self.ancillary["gha_edges"])

    def get_day(self, day: int) -> ModelData:
        """Get the :class:`ModelData` class corresponding to day."""
        if day not in self.days:
            raise ValueError(
                f"That day ({day}) does not exist. Existing days: {self.days}"
            )

        for obj in self.model_step:
            if obj.day == day:
                return obj

    @cached_property
    def gha_centres(self) -> np.ndarray:
        return (self.gha_edges[1:] + self.gha_edges[:-1]) / 2


def add_structure(cls):
    """Add basic data structure information to a given class."""
    # Determine the spectra keys
    spectra = deepcopy(cls._spectra_structure)

    spectra["weights"] = float_array_ndim(cls._spec_dim)

    if issubclass(cls, _ModelMixin):
        spectra["resids"] = float_array_ndim(cls._spec_dim)
    else:
        spectra["spectrum"] = float_array_ndim(cls._spec_dim)

    # Get ancillary
    ancillary = deepcopy(cls._ancillary)

    if issubclass(cls, _ModelMixin):
        ancillary["model_params"] = lambda x: x.dtype.name.startswith("float")

    if issubclass(cls, _CombinedFileMixin):
        ancillary["gha_edges"] = lambda x: x.ndim == 1 and x.dtype.name.startswith(
            "float"
        )

    meta = deepcopy(cls._meta) or {}

    # Add meta keys that are in every level
    meta["write_time"] = None
    meta["edges_io_version"] = None
    meta["object_name"] = None
    meta["edges_analysis_version"] = None
    meta["message"] = lambda x: isinstance(x, str)

    # Automatically add keys from the signature of _promote
    meta["parent_files"] = None

    sig = inspect.signature(cls._promote)

    for k, v in sig.parameters.items():
        if k != "prev_step" and k not in meta:
            meta[k] = None

    structure = {
        "frequency": lambda x: x.ndim == 1 and x.dtype.name.startswith("float"),
        "spectra": spectra,
        "ancillary": ancillary,
        "meta": meta,
    }

    cls._structure = structure
    cls.promote.__func__.__doc__ = cls._promote.__doc__

    return cls


@attr.s
class _ReductionStep(HDF5Object):
    """Base object for formal data reduction steps in edges-analysis."""

    _structure = {}
    _spectra_structure = {}
    _spec_dim = 2
    _possible_parents = ()
    _self_parent = False
    default_root = config["paths"]["field_products"]
    _multi_input = False
    _ancillary = {}
    _meta = {}

    def _get_parent_of_kind(self, kind: type[_ReductionStep]):
        def _get(c):
            if c.__class__ == kind:
                return c
            elif c.__class__ == RawData:
                raise AttributeError(
                    f"This object has no parent of kind {kind.__class__.__name__}"
                )
            elif hasattr(c, "__len__"):
                return [_get(cc) for cc in c]
            else:
                return _get(c.parent)

        return _get(self)

    @cached_property
    def calibration_step(self):
        return self._get_parent_of_kind(CalibratedData)

    @cached_property
    def raw_data(self):
        return self._get_parent_of_kind(RawData)

    @cached_property
    def model_step(self):
        return self._get_parent_of_kind(ModelData)

    @cached_property
    def combination_step(self):
        return self._get_parent_of_kind(CombinedData)

    @cached_property
    def day_average_step(self):
        return self._get_parent_of_kind(DayAveragedData)

    @cached_property
    def calibration(self):
        """The calibration object defining the calibration of this data."""
        return self.calibration_step.calibration

    @staticmethod
    def _get_object_class(fname: tp.PathLike):
        with h5py.File(fname, "r") as fl:
            obj_name = fl.attrs.get("object_name", None)

        if not obj_name:
            raise AttributeError(f"File {fname} is not a valid HDF5Object")

        try:
            return getattr(sys.modules[__name__], obj_name)
        except AttributeError:
            raise AttributeError(
                f"File {fname} has object type {obj_name} which is not a valid "
                "ReductionStep."
            )

    @classmethod
    def _promote(
        cls, prev_step: _ReductionStep | list[_ReductionStep] | io.Spectrum, **kwargs
    ) -> tuple[np.ndarray, dict, dict, dict]:
        pass

    @classmethod
    def promote(
        cls,
        prev_step: (
            _ReductionStep
            | list[_ReductionStep | str | Path]
            | io.Spectrum
            | tp.PathLike
        ),
        filename: tp.PathLike | None = None,
        clobber: bool = False,
        **kwargs,
    ):
        """
        Promote a :class:`_ReductionStep` to this level.

        Notes
        -----
        .. note::
            This docstring will be overwritten by the :func:`add_structure` class
            decorator on each subclass.
        """

        def _validate_obj(obj) -> _ReductionStep | HDF5RawSpectrum:
            # Either convert str/path to the proper object, return the object, or raise.
            if isinstance(obj, (str, Path)):
                with warnings.catch_warnings():
                    warnings.filterwarnings(action="ignore", category=UserWarning)
                    return read_step(obj)
            elif isinstance(obj, cls._possible_parents) or (
                cls._self_parent and obj.__class__ == cls
            ):
                return obj
            else:
                raise ValueError(f"{obj} is not a valid data set.")

        if cls._multi_input:
            # Validate each file, and sort them by date.
            # Sorting is important because we need to be able to know which index
            # corresponds to which file later on.
            prev_step = sorted(
                (_validate_obj(obj) for obj in prev_step),
                key=lambda x: (x.year, x.day, x.hour),
            )
        else:
            prev_step = _validate_obj(prev_step)

        freq, data, ancillary, meta = cls._promote(prev_step, **kwargs)
        meta["parent_files"] = (
            ":".join(str(p.filename) for p in prev_step)
            if isinstance(prev_step, list)
            else str(prev_step.filename)
        )

        filename = Path(filename).absolute() if filename else None

        if clobber and Path(filename).exists():
            os.remove(filename)

        out = cls.from_data(
            {"frequency": freq, "spectra": data, "ancillary": ancillary, "meta": meta},
            validate=False,
        )
        if filename:
            out.write(filename)

        out._parent = prev_step
        return out

    @cached_property
    def parent(
        self,
    ) -> _ReductionStep | io.Spectrum | list[_ReductionStep | io.Spectrum]:
        try:
            return self._parent
        except AttributeError:
            filenames = self.meta["parent_files"].split(":")

            if len(filenames) == 1:
                return read_step(filenames[0])
            else:
                return [read_step(fname) for fname in filenames]

    @property
    def meta(self):
        """Dictionary of meta information for this object."""
        return self["meta"]

    @property
    def raw_frequencies(self):
        return self["frequency"]

    @property
    def freq(self):
        return FrequencyRange(self.raw_frequencies)

    @property
    def ancillary(self):
        return self["ancillary"]

    @property
    def spectra(self):
        return self["spectra"]

    @property
    def raw_weights(self) -> np.ndarray[float]:
        """The raw in-file weights, before any flagging is applied."""
        return self.spectra["weights"]

    @cached_property
    def weights(self) -> np.ndarray[float]:
        """The weights of the data after taking account of current flags."""
        return self.get_weights()

    def get_weights(self, filt: str | int | None = None) -> np.ndarray[float]:
        """Get the weights of the data after taking account of some flags."""
        if filt in {"old", "initial"}:
            return self.raw_weights

        return np.where(self.get_flags(filt), 0, self.raw_weights)

    @property
    def spectrum(self):
        """Residuals of all spectra after being fit by the fiducial model."""
        return self.spectra["spectrum"]

    @classmethod
    def _get_meta(cls, params: dict) -> dict:
        sig = inspect.signature(cls._promote)
        out = {k: params[k] for k in sig.parameters if k != "prev_step"}

        for k, v_ in out.items():
            # Some rules for serialising to HDF5
            if isinstance(out[k], dict):
                out[k] = json.dumps(out[k])
            elif v_ is None:
                out[k] = ""
            elif isinstance(out[k], Path):
                out[k] = str(out[k])

        out.update(cls._extra_meta(params))
        out.update(cls._get_extra_meta())

        return out

    @classmethod
    def _extra_meta(cls, kwargs):
        return {}

    @classmethod
    def _get_extra_meta(cls):
        out = HDF5Object._get_extra_meta()
        out["edges_analysis_version"] = __version__
        out["message"] = ""

        # Need to forcibly overwrite the object name here, because it will be
        # `HDF5Object`. Problematically, can just call super()._get_extra_meta, because
        # this function is called in __init_subclass__, and so doesn't know about this
        # class definition when it is called.
        out["object_name"] = cls.__name__

        return out

    @property
    def filters_applied(self) -> dict[str, int]:
        """A map of filters that have been applied to their respective index order."""
        with self.open() as fl:
            filter_map = dict(fl["flags"].attrs) if "flags" in fl else {}
        return filter_map

    def _get_filter_index(self, filt: int | str | None) -> int:
        if not self.filters_applied:
            raise OSError("There are no existing flags to retrieve!")

        if isinstance(filt, str):
            filt = self.filters_applied[filt]
        elif filt is None:
            filt = len(self.filters_applied) - 1

        return int(filt)

    def get_flags(self, filt: int | str | None = None) -> np.ndarray:
        """Get the flags associated with a given filter.

        Paramaters
        ----------
        filt
            If a string, should be the name of an existing filter (i.e. it should exist
            in :attr:`filters_applied`). If an int, retrieve the flags corresponding to
            that index. By default, retrieve the flags after the latest round of
            filtering.

        Returns
        -------
        flags
            The array of flags.
        """
        if filt in {"initial", "old"} or not self.filters_applied:
            return self.raw_weights == 0

        filt = self._get_filter_index(filt)
        with self.open() as fl:
            flags = fl["flags"]["flags"][filt, ...]

        return flags

    def get_filter_meta(self, filt: str) -> dict[str, Any]:
        """Get the metadata associated with a given flagging filter.

        Paramaters
        ----------
        filt
            The filter to retrieve metadata for. Must exist in :attr:`filters_applied`.

        Returns
        -------
        meta
            A dictionary of metadata.
        """
        if filt not in self.filters_applied:
            raise ValueError(
                f"The filter {filt} does not exist in this data! Existing filters: "
                f"{self.filters_applied}"
            )

        with self.open() as fl:
            meta = dict(fl["flags"][filt].attrs)

        return meta

    @property
    def is_fully_flagged(self) -> bool:
        """Whether this object has been fully flagged."""
        return np.all(self.get_flags())


def read_step(
    fname: tp.PathLike | _ReductionStep | io.HDF5RawSpectrum,
    validate: bool = True,
) -> _ReductionStep | io.HDF5RawSpectrum:
    """Read a filename as a processing reduction step.

    The function is idempotent, so calling it on a step object just returns the object.
    """
    if isinstance(fname, (_ReductionStep, io.HDF5RawSpectrum)):
        return fname

    fname = Path(fname)
    if fname.suffix == ".acq":
        return io.FieldSpectrum(fname).data
    else:
        return get_step_type(fname)(filename=fname, validate=validate)


def get_step_type(
    fname: tp.PathLike | _ReductionStep | io.HDF5RawSpectrum,
) -> type[_ReductionStep] | type[io.HDF5RawSpectrum]:
    """Read a filename as a processing reduction step.

    The function is idempotent, so calling it on a step object just returns the object.
    """
    fname = Path(fname)
    if fname.suffix == ".acq":
        return io.HDF5RawSpectrum
    else:
        return _ReductionStep._get_object_class(fname)


class _ModelMixin:
    @cached_property
    def _model(self) -> mdl.Model:
        """The Model class that is used to fit foregrounds."""
        if isinstance(self.model_step, (list, tuple)):
            meta = self.model_step[0].meta
        else:
            meta = self.model_step.meta

        return meta["model"]

    @cached_property
    def model(self) -> mdl.FixedLinearModel:
        """The abstract linear model that is fit to each integration.

        Note that the parameters are not set on this model, but the basis vectors are
        set.
        """
        return self._model.at(x=self.freq.freq)

    @property
    def model_nterms(self):
        """Number of terms in the foreground model."""
        return self._model.n_terms

    @property
    def model_params(self):
        return self.ancillary["model_params"]

    def get_model(
        self,
        indx: int | list[int],
        p: np.ndarray | None = None,
        freq: np.ndarray | None = None,
    ) -> np.ndarray:
        """Obtain the fiducial fitted model spectrum for integration/gha at indx."""
        model = self.model if freq is None else self.model.at_x(freq)
        if p is None:
            p = self.model_params

        if not hasattr(indx, "__len__"):
            indx = [indx]

        for i in indx:
            p = p[i]

        return model(parameters=p)

    def get_spectrum(
        self, resids: np.ndarray, params: np.ndarray, freq: np.ndarray | None = None
    ) -> np.ndarray:
        """The processed spectra at this level."""
        indx = np.indices(resids.shape[:-1]).reshape((resids.ndim - 1, -1)).T
        out = np.zeros_like(resids)
        for i in indx:
            ix = tuple(np.atleast_2d(i).T.tolist())
            out[ix] = self.get_model(i, params, freq=freq) + resids[ix]

        return out

    @cached_property
    def spectrum(self):
        return self.get_spectrum(self.resids, self.model_params)

    @property
    def resids(self):
        """Residuals of all spectra after being fit by the fiducial model."""
        return self.spectra["resids"]


class _SingleDayMixin:
    @property
    def day(self):
        return self.raw_data.meta["day"]

    @property
    def year(self):
        return self.raw_data.meta["year"]

    @property
    def hour(self):
        return self.raw_data.meta["hour"]

    @property
    def minute(self):
        return self.raw_data.meta["minute"]

    @property
    def gha(self):
        return self.raw_data.ancillary["gha"]

    @property
    def lst(self):
        return self.raw_data.ancillary["lst"]

    @property
    def datestring(self):
        """The date this observation was started, as a string."""
        return f"{self.year:04}-{self.day:03}-{self.hour:02}-{self.minute:02}"

    @property
    def raw_time_data(self):
        """Raw string times at which the spectra were taken."""
        return self.raw_data.ancillary["times"]

    @cached_property
    def datetimes(self):
        """List of python datetimes at which the spectra were taken."""
        return self.get_datetimes(self.raw_time_data)

    @classmethod
    def get_datetimes(cls, times):
        return [datetime.strptime(d, "%Y:%j:%H:%M:%S") for d in times.astype(str)]

    def bin_in_frequency(
        self,
        indx: int | None = None,
        resolution: float = 0.0488,
        weights: np.ndarray | None = None,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Perform a frequency-average over the spectrum.

        Parameters
        ----------
        indx
            The (time) index at which to compute the frequency-averaged spectrum.
            If not given, returns a 2D array, with time on the first axis.
        resolution : float, optional
            The frequency resolution of the output.
        weights
            The weights to use when binning.

        Returns
        -------
        f
            The new frequency bin-centres.
        t
            The weighted-average of the spectrum in each bin
        w
            The total weight in each bin
        """
        if weights is None:
            weights = self.weights

        if indx is not None:
            s, w = self.spectrum[indx], weights[indx]
        else:
            s, w = self.spectrum, weights

        new_f, new_s, new_w = averaging.bin_array_unbiased_irregular(
            data=s, coords=self.raw_frequencies, weights=w, bins=resolution, axis=-1
        )

        return new_f, new_s, new_w

    def get_model_parameters(
        self,
        model: mdl.Model = mdl.LinLog(n_terms=5),
        resolution: int | float | None = 0.0488,
        weights: np.ndarray | None = None,
        indices: list[int] | None = None,
        freq_range: tuple[float, float] = (0, np.inf),
    ) -> tuple[mdl.Model, np.ndarray]:
        """
        Determine a callable model of the spectrum at a given time.

        Optionally computed over averaged original data.

        Parameters
        ----------
        model
            The kind of model to fit.
        resolution
            The resolution at which to bin the spectra before fitting models. If float,
            resolution in MHz, if int the number of samples per bin.
        weights
            Weights with which to fit the spectra.
        indices
            Which integrations to generate models for.
        freq_range
            The frequency range over which to fit the model.

        Other Parameters
        ----------------
        Passed through to the Model.

        Returns
        -------
        callable :
            Function of frequency (in units of self.raw_frequency) that will return
            the model.
        """
        if weights is None:
            weights = self.weights

        if indices is None:
            indices = range(len(self.spectrum))

        if resolution:
            f, s, w = self.bin_in_frequency(resolution=resolution, weights=weights)
            model = model.at(x=f[0])
        else:
            f, s, w = self.raw_frequencies, self.spectrum, weights
            mask = (f >= freq_range[0]) & (f < freq_range[1])
            f = f[mask]
            s = s[:, mask]
            w = w[:, mask]
            model = model.at(x=f)

        def get_params(indx, model):
            ss = s[indx]
            ww = w[indx]

            if resolution:
                ff = f[indx]
                mask = (ff >= freq_range[0]) & (ff < freq_range[1])
                ff = ff[mask]
                ss = ss[mask]
                ww = ww[mask]
                modelx = model.at_x(ff)
            else:
                modelx = model

            if np.sum(ww > 0) <= 2 * model.n_terms:
                # Only try to fit if we have enough non-flagged data points.
                return np.nan * np.ones(model.n_terms)

            return modelx.fit(ydata=ss, weights=ww).model_parameters

        params = np.array([get_params(indx, model) for indx in indices])
        return model, params

    def get_model_residuals(
        self,
        params: np.ndarray | None = None,
        indices: list[int] | None = None,
        freq_range: tuple[float, float] = (0, np.inf),
        **kwargs,
    ) -> np.ndarray:
        freq_mask = (self.raw_frequencies >= freq_range[0]) & (
            self.raw_frequencies < freq_range[1]
        )
        if params is None:
            model, params = self.get_model_parameters(
                resolution=0, freq_range=freq_range, **kwargs
            )
        else:
            try:
                f = self.raw_frequencies[freq_mask]
                model = kwargs["model"].at(x=f)
            except KeyError:
                raise KeyError("You must supply 'model' if supplying params.")

        if indices is None:
            indices = range(len(self.spectrum))
        # Index by indices so they have the same length as 'params'
        spec = self.spectrum[indices]
        return np.array(
            [s[freq_mask] - model(parameters=p) for s, p in zip(spec, params)]
        )

    def get_model_rms(
        self,
        weights: np.ndarray | None = None,
        freq_range: tuple[float, float] = (-np.inf, np.inf),
        indices: list[int] | None = None,
        **model_kwargs,
    ) -> np.ndarray:
        """Obtain the RMS of the residual of a model-fit to a particular integration.

        Parameters
        ----------
        weights
            The weights of the spectrum to use in the fitting. Must be the same shape
            as :attr:`~spectrum`. Default is to use the weights intrinsic to the object.
        freq_range
            The frequency range over which to fit the model (min, max in MHz)
        indices
            The integration indices for which to return the RMS.

        Other Parameters
        ----------------
        All other parameters are passed to :meth:`~get_model_parameters`.

        Returns
        -------
        rms
            A dictionary where keys are tuples specifying the input bands, and the
            values are arrays of rms values, as a function of time.

        Notes
        -----
        The averaging into frequency bins is *only* done for the fit itself. The final
        residuals are computed on the un-averaged spectrum. The binning is done fully
        self-consistently with the weights  -- it uses :func:`~tools.bin_array` to do
        the binning, returning non-equi-spaced frequencies.
        """
        resids = self.get_model_residuals(
            indices=indices, freq_range=freq_range, **model_kwargs
        )

        if indices is None:
            indices = range(len(self.spectrum))

        if weights is None:
            weights = self.weights[indices]

        freq_mask = (self.raw_frequencies >= freq_range[0]) & (
            self.raw_frequencies < freq_range[1]
        )

        # access the default basis
        def _get_rms(indx):
            mask = weights[indx, freq_mask] > 0
            return np.sqrt(np.nanmean(resids[indx, mask] ** 2))

        return np.array([_get_rms(i) for i in range(len(indices))])

    def plot_waterfall(
        self,
        quantity: str = "spectrum",
        filt: str | int | None = None,
        ax: plt.Axes | None = None,
        cbar=True,
        xlab=True,
        ylab=True,
        title=True,
        **imshow_kwargs,
    ):
        if quantity in {"p0", "p1", "p2"}:
            q = self.raw_data.spectra["switch_powers"][int(quantity[-1])]
        elif quantity == "Q":
            q = self.raw_data.spectrum
        else:
            q = getattr(self, quantity)

        q = np.where(self.get_flags(filt), np.nan, q)

        if ax is None:
            ax = plt.subplots(1, 1)[1]

        if quantity == "resids":
            cmap = imshow_kwargs.pop("cmap", "coolwarm")
        else:
            cmap = imshow_kwargs.pop("cmap", "magma")

        sec = self.raw_data.ancillary["seconds"]

        if quantity == "spectrum":
            vmax = imshow_kwargs.pop("vmax", 13000)
        elif quantity == "Q":
            vmax = imshow_kwargs.pop("vmax", 7)
        elif quantity == "p0":
            vmax = imshow_kwargs.pop("vmax", 5e-8)
        elif quantity == "resids":
            vmin = imshow_kwargs.pop("vmin", -np.nanmax(np.abs(q)))
            vmax = imshow_kwargs.pop("vmax", -vmin)
        else:
            vmax = imshow_kwargs.pop("vmax", None)

        img = ax.imshow(
            q,
            origin="lower",
            extent=(
                self.raw_frequencies.min(),
                self.raw_frequencies.max(),
                sec.min() / 60 / 60,
                sec.max() / 60 / 60,
            ),
            cmap=cmap,
            aspect="auto",
            vmax=vmax,
            interpolation="none",
            **imshow_kwargs,
        )

        if xlab:
            ax.set_xlabel("Frequency [MHz]")
        if ylab:
            ax.set_ylabel("Hours into Observation")
        if title:
            ax.set_title(f"{self.datestring}. GHA0={self.gha[0]:.2f}")

        if cbar:
            cb = plt.colorbar(img, ax=ax)
            cb.set_label(quantity)

        return ax

    def plot_waterfalls(self, quanties="all", **imshow_kwargs):
        if quanties == "all":
            quanties = ["spectrum", "Q", "weights", "p0", "p1", "p2"]

        fig, ax = plt.subplots(
            len(quanties),
            1,
            sharex=True,
            sharey=True,
            figsize=(10, 10),
            gridspec_kw={"hspace": 0.05, "wspace": 0.05},
        )

        for i, (q, axx) in enumerate(zip(quanties, ax)):
            self.plot_waterfall(
                q,
                ax=axx,
                **imshow_kwargs.get(q, {}),
                xlab=i == (len(quanties) - 1),
                title=i == 0,
            )

        return fig, ax

    def bin_gha(
        self, gha_bins=(0, 24), **model_kwargs
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        model, params = self.get_model_parameters(resolution=0, **model_kwargs)
        resids = self.get_model_residuals(params=params, model=model.model)

        p, r, w = averaging.bin_gha_unbiased_regular(
            params=params,
            resids=resids,
            weights=self.weights,
            gha=self.gha,
            bins=gha_bins,
        )
        spec = [model(parameters=pp) + rr for pp, rr in zip(p, r)]
        return np.squeeze(np.array(spec)), np.squeeze(r), np.squeeze(w)

    def plot_time_averaged_spectrum(
        self, ax: plt.Axes | None = None, logy=True, gha_bins=(0, 24), **model_kwargs
    ):
        if ax is not None:
            fig = ax.figure
        else:
            fig, ax = plt.subplots(1, 1)

        if self.__class__.__name__ == "RawData" and "model" not in model_kwargs:
            model_kwargs["model"] = mdl.Polynomial(
                n_terms=8, transform=mdl.UnitTransform()
            )

        spec, r, w = self.bin_gha(gha_bins, **model_kwargs)

        ax.plot(self.raw_frequencies, spec)
        ax.set_xlabel("Frequency [MHz]")
        ax.set_ylabel("Average Spectrum [K]")

        if logy:
            ax.set_yscale("log")

        return ax

    def plot_s11(self, ax=None):
        if ax is None:
            fig, ax = plt.subplots(2, 2, figsize=(8, 8), sharex=True)
        ax[0, 0].plot(self.raw_frequencies, 20 * np.log10(np.abs(self.antenna_s11)))
        ax[0, 0].set_title("Magnitude of Antenna S11")
        ax[0, 0].set_xlabel("Frequency [MHz]")
        ax[0, 0].set_ylabel("$|S_{11}|$ [dB]")

        ax[0, 1].plot(
            self.raw_frequencies, (180 / np.pi) * np.unwrap(np.angle(self.antenna_s11))
        )
        ax[0, 1].set_title("Phase of Antenna S11")
        ax[0, 1].set_xlabel("Frequency [MHz]")
        ax[0, 1].set_ylabel(r"$\angle S_{11}$ [${}^\circ$]")

        ax[1, 0].plot(
            self.raw_antenna_s11_freq,
            np.real(self.raw_antenna_s11)
            - np.real(self.antenna_s11_model(self.raw_antenna_s11_freq)),
        )
        ax[1, 0].set_title("Residual (Real Part)")
        ax[1, 0].set_xlabel("Frequency [MHz]")
        ax[1, 0].set_ylabel(r"Data - Model")

        ax[1, 1].plot(
            self.raw_antenna_s11_freq,
            np.imag(self.raw_antenna_s11)
            - np.imag(self.antenna_s11_model(self.raw_antenna_s11_freq)),
        )
        ax[1, 1].set_title("Residual (Imag Part)")
        ax[1, 1].set_xlabel("Frequency [MHz]")
        ax[1, 1].set_ylabel(r"Data - Model")

        return ax


@add_structure
class RawData(_ReductionStep, _SingleDayMixin):
    """Object representing raw input data and matched auxiliary data.

    See :class:`_ReductionStep` for documentation about the various datasets within this
    class instance. Note that you can always check which data is inside each group by
    checking its ``.keys()``.

    Create the class either directly from a level-1 file (via normal instantiation), or
    by calling :meth:`from_acq` on a raw ACQ file.
    """

    _spectra_structure = {
        "switch_powers": float_array_ndim(3),
    }

    _ancillary = {
        "adcmax": float_array_ndim(2),
        "adcmin": float_array_ndim(2),
        "ambient_hum": float_array_ndim(1),
        "ambient_temp": float_array_ndim(1),
        "frontend_temp": float_array_ndim(1),
        "gha": float_array_ndim(1),
        "lna_temp": float_array_ndim(1),
        "lst": float_array_ndim(1),
        "moon_az": float_array_ndim(1),
        "moon_el": float_array_ndim(1),
        "power_percent": float_array_ndim(1),
        "rack_temp": float_array_ndim(1),
        "receiver_temp": float_array_ndim(1),
        "seconds": is_array("int", 1),
        "sun_az": float_array_ndim(1),
        "sun_el": float_array_ndim(1),
        "temp_set": float_array_ndim(1),
        "times": None,
    }

    _meta = {
        "day": lambda x: isinstance(x, (int, np.int64)),
        "year": lambda x: isinstance(x, (int, np.int64)),
        "hour": lambda x: isinstance(x, (int, np.int64)),
        "freq_max": lambda x: isinstance(x, (float, np.float32)),
        "freq_min": lambda x: isinstance(x, (float, np.float32)),
        "freq_res": lambda x: isinstance(x, (float, np.float32)),
        "n_file_lines": lambda x: isinstance(x, (int, np.int64)),
        "nblk": lambda x: isinstance(x, (int, np.int64)),
        "nfreq": lambda x: isinstance(x, (int, np.int64)),
        "resolution": None,
        "temperature": None,
        "band": lambda x: isinstance(x, str),
        "thermlog_file": None,
        "weather_file": None,
    }

    @classmethod
    def _promote(
        cls,
        prev_step: io.HDF5RawSpectrum,
        band: str,
        weather_file: str | Path | None = None,
        thermlog_file: str | Path | None = None,
        f_low: float = 0.0,
    ) -> tuple[np.ndarray, dict, dict, dict]:
        """
        Create the object directly from calibrated data.

        Parameters
        ----------
        prev_step
            The filename of the ACQ file to read.
        band
            Defines the instrument that took the data (mid, low, high).
        weather_file
            A weather file to use in order to capture that information (may find the
            default weather file automatically).
        thermlog_file
            A thermlog file to use in order to capture that information (may find the
            default thermlog file automatically).
        """
        pr = psutil.Process()

        logger.debug(
            f"Memory at start of RawData promote: {pr.memory_info().rss / 1024**2}"
        )

        t = time.time()
        freq_rng = FrequencyRange(
            prev_step["freq_ancillary"]["frequencies"], f_low=f_low
        )
        freq = freq_rng.freq
        q = prev_step["spectra"]["Q"][freq_rng.mask]
        p = [
            prev_step["spectra"]["p0"][freq_rng.mask],
            prev_step["spectra"]["p1"][freq_rng.mask],
            prev_step["spectra"]["p2"][freq_rng.mask],
        ]

        logger.debug(
            f"Memory after loading prev_step: {pr.memory_info().rss / 1024**2}"
        )

        ancillary = prev_step["time_ancillary"]

        logger.info(f"Time for reading: {time.time() - t:.2f} sec.")

        logger.info("Converting time strings to datetimes...")
        t = time.time()
        times = cls.get_datetimes(ancillary["times"])
        logger.info(f"...  finished in {time.time() - t:.2f} sec.")

        logger.debug(
            f"Memory after getting Datetimes: {pr.memory_info().rss / 1024**2}"
        )

        meta = {
            "year": times[0].year,
            "day": get_jd(times[0]),
            "hour": times[0].hour,
            "minute": times[0].minute,
            **prev_step["meta"],
        }

        time_based_anc = ancillary

        logger.info("Getting ancillary weather data...")
        t = time.time()
        new_anc, new_meta = cls._get_weather_thermlog(
            band, times, weather_file, thermlog_file
        )
        meta = {**meta, **new_meta}

        logger.debug(
            f"Memory fter getting weather data: {pr.memory_info().rss / 1024**2}"
        )

        new_anc = {k: new_anc[k] for k in new_anc.dtype.names}
        time_based_anc = {**time_based_anc, **new_anc}
        time_based_anc = {**time_based_anc, **cls.get_ancillary_coords(times)}

        logger.info(f"... finished in {time.time() - t:.2f} sec.")

        logger.debug(f"Memory after ancillary:{pr.memory_info().rss / 1024**2}")

        data = {
            "switch_powers": np.array([pp.T for pp in p]),
            "weights": np.ones_like(q.T),
            "spectrum": q.T,
        }

        meta = {**meta, **cls._get_meta(locals())}
        return freq, data, time_based_anc, meta

    @classmethod
    def _get_weather_thermlog(
        cls,
        band: str,
        times: list[datetime],
        weather_file: None | tp.PathLike = None,
        thermlog_file: None | tp.PathLike = None,
    ):
        """
        Read the appropriate weather and thermlog file, returning their contents.

        Parameters
        ----------
        band
            The band/telescope of the data (mid, low2, low3, high).
        times
            List of datetime objects giving the date-times of the (beginning of)
            observations.
        weather_file
            Path to a weather file from which to read the weather data. Must be
            formatted appropriately. By default, will choose an appropriate file from
            the configured `raw_field_data` directory. If provided, will search in
            the current directory and the `raw_field_data` directory for the given
            file (if not an absolute path).
        thermlog_file
            Path to a thermlog file from which to read the thermlog data. Must be
            formatted appropriately. By default, will choose an appropriate file from
            the configured `raw_field_data` directory. If provided, will search in
            the current directory and the `raw_field_data` directory for the given
            file (if not an absolute path).

        Returns
        -------
        auxiliary
            Containing:

            * "ambient_temp": Ambient temperature as a function of time
            * "ambient_humidity": Ambient humidity as a function of time
            * "receiver1_temp": Receiver1 temperature as a function of time
            * "receiver2_temp": Receiver2 temperature as a function of time
            * "lst": LST for each observation in the spectrum.
            * "gha": GHA for each observation in the spectrum.
            * "sun_moon_azel": Coordinates of the sun and moon as function of time.

        meta : dict
            Containing:

            * "thermlog_file": absolute path to the thermlog information used (filled in
              with the default if necessary).
            * "weather_file": absolute path to the weather information used (filled in
              with the default if necessary).

        """
        start = min(times)
        end = max(times)

        pth = Path(config["paths"]["raw_field_data"])
        if weather_file is not None:
            weather_file = Path(weather_file)
            if not (weather_file.exists() or weather_file.is_absolute()):
                weather_file = pth / weather_file
        else:
            if (start.year, start.day) <= (2017, 329):
                weather_file = pth / "weather_upto_20171125.txt"
            else:
                weather_file = pth / "weather2.txt"

        if thermlog_file is not None:
            thermlog_file = Path(thermlog_file)
            if not (thermlog_file.exists() or thermlog_file.is_absolute()):
                thermlog_file = pth / thermlog_file
        else:
            thermlog_file = pth / f"thermlog_{band}.txt"

        # Get all aux data covering our times, up to the next minute (so we have some
        # overlap).
        weather, thermlog = auxiliary_data(
            weather_file,
            thermlog_file,
            year=start.year,
            day=get_jd(start),
            hour=start.hour,
            end_time=(end.year, get_jd(end), end.hour, end.minute + 1),
        )

        if len(weather) == 0:
            raise WeatherError(
                f"Weather file '{weather_file}' has no dates between "
                f"{start.strftime('%Y/%m/%d')} "
                f"and {end.strftime('%Y/%m/%d')}."
            )

        if len(thermlog) == 0:
            raise WeatherError(
                f"Thermlog file '{thermlog_file}' has no dates between "
                f"{start.strftime('%Y/%m/%d')} "
                f"and {end.strftime('%Y/%m/%d')}."
            )

        logger.info("Setting up arrays...")

        t = time.time()
        # Get the seconds since obs start for the data (not the auxiliary).
        seconds = np.array([(t - times[0]).total_seconds() for t in times])

        time_based_anc = np.zeros(
            len(seconds),
            dtype=[("seconds", int)]
            + [
                (name, float)
                for name, (kind, off) in weather.dtype.fields.items()
                if kind.kind == "f"
            ]
            + [
                (name, float)
                for name, (kind, off) in thermlog.dtype.fields.items()
                if kind.kind == "f"
            ],
        )
        time_based_anc["seconds"] = seconds
        logger.info(f".... took {time.time() - t} sec.")

        t = time.time()
        # Interpolate weather

        for i, thing in enumerate([weather, thermlog]):
            thing_seconds = [
                (
                    dt_from_jd(
                        x["year"], int(x["day"]), x["hour"], x["minute"], x["second"]
                    )
                    - times[0]
                ).total_seconds()
                for x in thing
            ]

            for name, (kind, _) in thing.dtype.fields.items():
                if kind.kind == "i":
                    continue

                time_based_anc[name] = np.interp(seconds, thing_seconds, thing[name])

                # Convert to celsius
                if name.endswith("_temp") and np.any(time_based_anc[name] > 273.15):
                    time_based_anc[name] -= 273.15

        logger.info(f"Took {time.time() - t} sec to interpolate auxiliary data.")

        meta = {
            "thermlog_file": str(thermlog_file.absolute()),
            "weather_file": str(weather_file.absolute()),
        }
        return time_based_anc, meta

    @classmethod
    def get_ancillary_coords(cls, times: list[datetime]) -> dict[str, np.ndarray]:
        """
        Obtain a dictionary of ancillary co-ordinates based on a list of times.

        Parameters
        ----------
        times
            Nx6 array of floats or integers, where each row is of the form
            [yyyy, mm, dd,HH, MM, SS]. It can also be a 6-element 1D array.
        """
        out = {}

        # LST
        t = time.time()
        out["lst"] = coordinates.utc2lst(times, const.edges_lon_deg)
        out["gha"] = coordinates.lst2gha(out["lst"])
        logger.info(f"Took {time.time() - t} sec to get lst/gha")

        # Sun/Moon coordinates
        t = time.time()
        sun, moon = coordinates.sun_moon_azel(
            const.edges_lat_deg, const.edges_lon_deg, times
        )
        logger.info(f"Took {time.time() - t} sec to get sun/moon coords.")

        out["sun_az"] = sun[:, 0]
        out["sun_el"] = sun[:, 1]
        out["moon_az"] = moon[:, 0]
        out["moon_el"] = moon[:, 1]

        return out


@add_structure
class CalibratedData(_ReductionStep, _SingleDayMixin):
    """Object representing calibrated data.

    This object essentially represents a Calibrated spectrum.

    See :class:`_ReductionStep` for documentation about the various datasets within this
    class instance. Note that you can always check which data is inside each group by
    checking its ``.keys()``.

    The data at this level have (in this order):

    * Calibration applied from existing calibration solutions
    * Collected associated weather and thermal auxiliary data (not used at this level,
      just collected)
    * Potential xRFI applied to the raw switch powers individually.
    """

    _ancillary = copy.copy(RawData._ancillary)

    _meta = {
        **RawData._meta,
        **{
            "calobs_path": None,
            "cterms": lambda x: isinstance(x, (int, np.int64)),
            "wterms": lambda x: isinstance(x, (int, np.int64)),
            "s11_files": None,
        },
    }

    @classmethod
    def _promote(
        cls,
        prev_step: RawData,
        calfile: str | Path,
        s11_path: str | Path,
        s11_file_pattern: str = r"{y}_{jd}_{h}_*_input{input}.s1p",
        ignore_s11_files: list[str] | None = None,
        antenna_s11_n_terms: int = 15,
        antenna_correction: str | Path | None = ":",
        balun_correction: str | Path | None = ":",
        ground_correction: str | Path | None | float = ":",
        beam_file=None,
    ) -> tuple[np.ndarray, dict, dict, dict]:
        """
        Create the object directly from calibrated data.

        Parameters
        ----------
        prev_step
            The filename of the RawData file to read.
        calfile
            A file containing the output of
            :meth:`edges_cal.CalibrationObservation.write` -- i.e. all the information
            required to calibrate the raw data. Determination of calibration parameters
            occurs externally and saved to this file.
        s11_path
            Path to the receiver S11 information relevant to this observation.
        s11_file_pattern
            A format string defining the naming pattern of S11 files at ``s11_path``.
            This is used to automatically find the S11 file closest in time to the
            observation, if the ``s11_path`` is not explicit (i.e. it is a directory).
        ignore_s11_files
            A list of paths to ignore when attempting to find the S11 file closest to
            the observation (perhaps they are known to be bad).

        Other Parameters
        ----------------
        All other parameters are passed to :meth:`_calibrate` -- see its documentation
        for details.

        Returns
        -------
        data
            An instantiated :class:`CalibratedData` object.
        """
        prev_step = read_step(prev_step)

        s11_files = cls.get_s11_paths(
            s11_path,
            prev_step.meta["band"],
            prev_step.datetimes[0],
            s11_file_pattern,
            ignore_files=ignore_s11_files,
        )

        calobs = Calibration(Path(calfile).expanduser())
        labcal = LabCalibration(
            calobs=calobs,
            s11_files=s11_files,
            ant_s11_model=mdl.Polynomial(
                n_terms=antenna_s11_n_terms,
                transform=mdl.UnitTransform(range=(calobs.freq.min, calobs.freq.max)),
            ),
        )

        logger.info("Calibrating data ...")
        t = time.time()
        calspec, freq, new_meta = cls.calibrate(
            q=prev_step.spectrum,
            freq=prev_step.raw_frequencies,
            band=prev_step.meta["band"],
            labcal=labcal,
            ambient_temp=prev_step.ancillary["ambient_temp"],
            lst=prev_step.ancillary["lst"],
            configuration="",
            antenna_correction=antenna_correction,
            balun_correction=balun_correction,
            ground_correction=ground_correction,
            beam_file=beam_file,
            f_low=labcal.calobs.freq.min,
            f_high=labcal.calobs.freq.max,
        )
        logger.info(f"... finished in {time.time() - t:.2f} sec.")

        meta = {**prev_step.meta, **new_meta}
        meta = {**meta, **cls._get_meta(locals())}
        meta["s11_files"] = ":".join(str(fl) for fl in s11_files)

        freq_mask = (prev_step.raw_frequencies >= labcal.calobs.freq.min) & (
            prev_step.raw_frequencies <= labcal.calobs.freq.max
        )
        data = {
            "spectrum": calspec,
            "weights": prev_step.weights[:, freq_mask],
        }

        return freq.freq, data, {k: v for k, v in prev_step.ancillary.items()}, meta

    @property
    def s11_files(self):
        """The antenna S11 files used by the calibration."""
        return self.meta["s11_files"].split(":")

    def get_subset(self, integrations=100):
        """Write a subset of the data to a new mock :class:`CalibratedData` file."""
        freq = self.raw_frequencies
        spectra = {k: self.spectra[k] for k in self.spectra.keys()}
        ancillary = self.ancillary
        meta = self.meta

        spectra = {k: s[:integrations] for k, s in spectra.items()}
        ancillary = ancillary[:integrations]

        return self.from_data(
            {
                "frequency": freq,
                "spectra": spectra,
                "ancillary": ancillary,
                "meta": meta,
            }
        )

    @classmethod
    def default_s11_directory(cls, band: str) -> Path:
        """Get the default S11 directory for this observation."""
        return Path(config["paths"]["raw_field_data"]) / "mro" / band / "s11"

    @classmethod
    def _get_closest_s11_time(
        cls,
        s11_dir: Path,
        time: datetime,
        s11_file_pattern: str = "{y}_{jd}_{h}_*_input{input}.s1p",
        ignore_files=None,
    ):
        """From a given filename pattern, within a directory, find file closest to time.

        Parameters
        ----------
        s11_dir : Path
            The directory in which to search for S11 files.
        time : datetime
            The time to find the closest match to.
        s11_file_pattern : str
            A pattern that matches files in the directory. A few tags are available:
            {input}: tags the input number (should be 1-4)
            {y}: year (four digit number)
            {m}: month (two-digit number)
            {d}: day of month (two-digit number)
            {jd}: day of year (three-digit number)
            {h}: hour of day (observation start) (two digit number)
        ignore_files : list, optional
            A list of file patterns to ignore. They need only partially match
            the actual filenames. So for example, you could specify
            ``ignore_files=['2020_076']`` and it will ignore the file
            ``/home/user/data/2020_076_01_02_input1.s1p``. Full regex can be used.
        """
        # Replace the suffix dot with a literal dot for regex
        s11_file_pattern = s11_file_pattern.replace(".", r"\.")

        # Replace any glob-style asterisks with non-greedy regex version
        s11_file_pattern = s11_file_pattern.replace("*", r".*?")

        # First, we need to build a regex pattern out of the s11_file_pattern
        dct = {
            "input": r"(?P<input>\d)",
            "y": r"(?P<year>\d\d\d\d)",
            "m": r"(?P<month>\d\d)",
            "d": r"(?P<day>\d\d)",
            "jd": r"(?P<jd>\d\d\d)",
            "h": r"(?P<hour>\d\d)",
        }
        dct = {d: v for d, v in dct.items() if "{%s}" % d in s11_file_pattern}

        if "d" not in dct and "jd" not in dct:
            raise ValueError("s11_file_pattern must contain a tag {d} or {jd}.")
        if "d" in dct and "jd" in dct:
            raise ValueError("s11_file_pattern must not contain both {d} and {jd}.")

        p = re.compile(s11_file_pattern.format(**dct))

        ignore = [re.compile(ign) for ign in (ignore_files or [])]

        files = list(s11_dir.glob("*"))

        s11_times = []
        indx = []
        for i, fl in enumerate(files):
            match = p.match(str(fl.name))

            # Ignore files that don't match the pattern
            if not match:
                continue
            if any(ign.match(str(fl.name)) for ign in ignore):
                continue

            d = match.groupdict()

            indx.append(i)

            # Different time constructor for Day of year vs Day of month
            if "jd" in d:
                dt = coords.dt_from_jd(
                    int(d.get("year", time.year)),
                    int(d.get("jd")),
                    int(d.get("hour", 0)),
                )
            else:
                dt = datetime(
                    int(d.get("year", time.year)),
                    int(d.get("month", time.month)),
                    int(d.get("day")),
                    int(d.get("hour", 0)),
                )
            s11_times.append(dt)

        if not len(s11_times):
            raise FileNotFoundError(
                f"No files found matching the input pattern. Available files: "
                f"{[fl.name for fl in files]}. Regex pattern: {p.pattern}. "
            )

        files = [fl for i, fl in enumerate(files) if i in indx]
        time_diffs = np.array([abs((time - t).total_seconds()) for t in s11_times])
        indx = np.where(time_diffs == time_diffs.min())[0]

        # Gets a representative closest time file
        closest = [fl for i, fl in enumerate(files) if i in indx]

        assert (
            len(closest) == 4
        ), f"There need to be four input S1P files of the same time, got {closest}."
        return sorted(closest)

    @classmethod
    def get_s11_paths(
        cls,
        s11_path: str | Path | tuple | list,
        band: str,
        begin_time: datetime,
        s11_file_pattern: str,
        ignore_files: list[str] | None = None,
    ):
        """Given an s11_path, return list of paths for each of the inputs."""
        # If we get four files, make sure they exist and pass them back
        if isinstance(s11_path, (tuple, list)):
            if len(s11_path) != 4:
                raise ValueError(
                    "If passing explicit paths to S11 inputs, length must be 4."
                )

            fls = []
            for pth in s11_path:
                p = Path(pth).expanduser().absolute()
                assert p.exists()
                fls.append(p)

            return fls
        if s11_path.endswith("csv"):
            return [s11_path]
        # Otherwise it must be a path.
        s11_path = Path(s11_path).expanduser()

        if str(s11_path).startswith(":"):
            s11_path = cls.default_s11_directory(band) / str(s11_path)[1:]

        if s11_path.is_dir():
            # Get closest measurement
            return cls._get_closest_s11_time(
                s11_path, begin_time, s11_file_pattern, ignore_files=ignore_files
            )
        # The path *must* have an {input} tag in it which we can search on
        fls = glob.glob(str(s11_path).format(input="?"))
        assert (
            len(fls) == 4
        ), f"There are not exactly four files matching {s11_path}. Found: {fls}."

        return sorted(Path(fl) for fl in fls)

    @cached_property
    def lab_calibrator(self) -> LabCalibration:
        """Object that performs lab-based calibration on the data."""
        return LabCalibration(
            calobs=self.calibration,
            s11_files=self.s11_files,
            ant_s11_model=mdl.Polynomial(
                n_terms=self.meta["antenna_s11_n_terms"],
                transform=mdl.UnitTransform(
                    range=(self.calibration.freq.min, self.calibration.freq.max)
                ),
            ),
        )

    def antenna_s11_model(self, freq) -> Callable[[np.ndarray], np.ndarray]:
        """The antennas S11 model, evaluated at ``freq``."""
        return self.lab_calibrator.antenna_s11_model(freq)

    @property
    def antenna_s11(self) -> np.ndarray:
        """The anatenna S11 array after calibration."""
        return self.antenna_s11_model(self.raw_frequencies)

    @property
    def raw_antenna_s11(self) -> np.ndarray:
        """The raw antenna S11 (i.e. directly from file, before calibration)."""
        return self.lab_calibrator.raw_antenna_s11

    @property
    def raw_antenna_s11_freq(self) -> np.ndarray:
        """The frequencies of the raw antenna S11."""
        return self.lab_calibrator.raw_antenna_s11_freq

    @cached_property
    def calibration(self):
        """The Calibration object used to calibrate this observation."""
        return Calibration(self.meta["calfile"])

    @classmethod
    def labcal(
        cls, *, q: np.ndarray, freq: FrequencyRange, labcal: LabCalibration
    ) -> np.ndarray:
        """Perform lab calibration on given three-position switch ratio data."""
        # Cut the frequency range
        q = q[:, freq.mask]
        return labcal.calibrate_q(q)

    @classmethod
    def loss_correct(
        cls,
        *,
        spec: np.ndarray,
        freq: FrequencyRange,
        band: str,
        antenna_correction: tp.PathLike | None = ":",
        ambient_temp: np.ndarray,
        configuration="",
        balun_correction: tp.PathLike | None = ":",
        ground_correction: tp.PathLike | None | float = ":",
        s11_ant: np.ndarray | None = None,
    ) -> np.ndarray:
        """Correct a spectrum for losses."""
        # Antenna Loss (interface between panels and balun)
        gain = np.ones_like(freq.freq)
        if antenna_correction:
            gain *= loss.antenna_loss(
                antenna_correction, freq.freq, band=band, configuration=configuration
            )

        # Balun+Connector Loss
        if balun_correction:
            balun_gain, connector_gain = loss.balun_and_connector_loss(
                band, freq.freq, s11_ant
            )
            gain *= balun_gain * connector_gain

        # Ground Loss
        if isinstance(ground_correction, (str, Path)):
            gain *= loss.ground_loss(
                ground_correction, freq.freq, band=band, configuration=configuration
            )
        elif isinstance(ground_correction, float):
            gain *= ground_correction

        a = (
            ambient_temp + const.absolute_zero
            if ambient_temp[0] < 200
            else ambient_temp
        )
        spec = (spec - np.outer(a, (1 - gain))) / gain

        return spec

    @classmethod
    def beam_correct(
        cls,
        *,
        spec: np.ndarray,
        freq: FrequencyRange,
        lst: np.ndarray,
        band: str | None = None,
        beam_file: tp.PathLike | None = ":",
    ):
        """Correct a spectrum for beam chromaticity."""
        # Beam factor
        if beam_file:
            beam_fac = beams.InterpolatedBeamFactor.from_beam_factor(
                beam_file, band=band, f_new=freq.freq
            )
            bf = beam_fac.evaluate(lst)

        else:
            bf = np.ones_like(freq.freq)

        return spec / bf

    @classmethod
    def calibrate(
        cls,
        *,
        q: np.ndarray,
        freq: np.ndarray,
        labcal: LabCalibration,
        band: str | None = None,
        ambient_temp: np.ndarray,
        antenna_correction: tp.PathLike | None = ":",
        configuration: str = "",
        balun_correction: tp.PathLike | None = ":",
        ground_correction: tp.PathLike | None | float = ":",
        beam_file=None,
        lst: np.ndarray | None = None,
        f_low: float = 50,
        f_high: float = 150.0,
    ) -> tuple[np.ndarray, FrequencyRange, dict]:
        """
        Calibrate data.

        This method performs the following operations on the data:

        * Restricts frequency range
        * Applies a lab-based calibration solution
        * Corrects for ground/balun/antenna losses
        * Corrects for beam factor

        Parameters
        ----------
        q
            The raw Q-ratio of the spectra.
        freq
            An array of frequencies over which to calibrate.
        calfile
            The lab-based calibration observation to use to calibrate the data.
        band
            The band of the instrument. Required if any of the loss/beam files are
            prepended with ':'.
        s11_files
            List of four files defining the S11 measurements of the antenna.
        ambient_temp
            The ambient temperature at which observations were taken.
        switch_state_dir
            The directory in which the switch state measurements for the calibration
            live. Typically leave this as None, in which case it will be read from the
            calibration file. However, if you are working on a different computer than
            the one the calibration was performed on, you may need to set this.
        antenna_s11_n_terms
            Number of terms used in fitting the S11 model.
        antenna_correction
            Whether to perform the antenna correction
        configuration
            Specification of the antenna -- orientation etc. Should be a predefined
            format, eg '45deg'.
        balun_correction
            Path to a balun correction file.
        ground_correction
            Path to a ground correction file.
        beam_file
            Filename (not absolute) of a beam model to use for correcting for the beam
            factor. Not used if not provided.
        f_low : float
            Minimum frequency to use.
        f_high : float
            Maximum frequency to use.
        switch_state_repeat_num
            The repeat num of the switch state observations to use.
        lst
            The LST at which to apply beam correction.

        Returns
        -------
        spec
            The calibrated spectrum.
        freq
            The :class:`FrequencyRange` object associated with the solutions.
        meta
            Contains all input parameters.
        """
        freq = FrequencyRange(freq, f_low=f_low, f_high=f_high)

        calibrated_temp = cls.labcal(q=q, freq=freq, labcal=labcal)

        calibrated_temp = cls.loss_correct(
            spec=calibrated_temp,
            freq=freq,
            band=band,
            antenna_correction=antenna_correction,
            ambient_temp=ambient_temp,
            configuration=configuration,
            balun_correction=balun_correction,
            ground_correction=ground_correction,
            s11_ant=labcal.antenna_s11,
        )

        calibrated_temp = cls.beam_correct(
            freq=freq, spec=calibrated_temp, band=band, lst=lst, beam_file=beam_file
        )

        meta = {
            "beam_file": str(Path(beam_file).absolute())
            if beam_file is not None
            else "",
            "s11_files": ":".join(str(f) for f in labcal.s11_files),
            "wterms": labcal.calobs.wterms,
            "cterms": labcal.calobs.cterms,
            "calfile": str(labcal.calobs.calfile),
            "calobs_path": str(labcal.calobs.calobs_path),
        }

        return calibrated_temp, freq, meta


@add_structure
class ModelData(_ModelMixin, _ReductionStep, _SingleDayMixin):
    _ancillary = {"gha": float_array_ndim(1)}
    _possible_parents = (CalibratedData,)

    @classmethod
    def _promote(
        cls,
        prev_step: CalibratedData,
        model: mdl.LinLog(n_terms=5),
        model_resolution: int | float = 8,
    ) -> tuple[np.ndarray, dict, dict, dict]:
        """
        Fit fiducial linear models to each individual spectrum.

        Parameters
        ----------
        prev_step
            The previous step. Can act on calibrated data, or filtered data.
        model_nterms
            The number of terms to use when fitting smooth models to each spectrum.
        model_basis
            The model basis -- a string representing a model from
            :mod:`edges_cal.modelling`
        model_resolution
            If integer, the number of frequency samples binned together before fitting
            the fiducial model. If float, the resolution of the bins in MHz. Set to zero
            to not bin. Residuals of the model are still evaluated at full frequency
            resolution -- this just affects the modeling itself.

        Returns
        -------
        data
            A :class:`ModelData` instance.
        """
        logger.info(f"Determining '{model}' models for each integration...")

        # Exit out early if the whole file is flagged.
        try:
            if prev_step.is_fully_flagged:
                raise FullyFlaggedError()
        except AttributeError:
            pass

        modelx, params = prev_step.get_model_parameters(
            model,
            resolution=model_resolution,
        )

        if model_resolution:
            modelx = modelx.at_x(x=prev_step.freq.freq)

        resids = np.array(
            [
                prev_step.spectrum[j] - modelx(parameters=pp)
                for j, pp in enumerate(params)
            ]
        )

        ancillary = {"model_params": params, "gha": prev_step.ancillary["gha"]}
        data = {"weights": prev_step.weights, "resids": resids}

        return prev_step.raw_frequencies, data, ancillary, cls._get_meta(locals())


@add_structure
class CombinedData(_ModelMixin, _ReductionStep, _CombinedFileMixin):
    """
    Object representing many observed days combined.

    Given a sequence of :class:`ModelData` objects, this class combines them into one
    file, aligning them in (ideally small) bins in GHA/LST.

    See :class:`_ReductionStep` for documentation about the various datasets within this
    class instance. Note that you can always check which data is inside each group by
    checking its ``.keys()``.

    See :meth:`CombinedData.promote` for detailed information about the processes
    involved in creating this data from :class:`Level1` objects.
    """

    _possible_parents = (ModelData,)
    _multi_input = True
    _spec_dim = 3

    _ancillary = {
        "years": is_array("int", 1),
        "days": is_array("int", 1),
        "hours": is_array("int", 1),
    }

    _meta = {
        "n_files": lambda x: isinstance(x, (int, np.int, np.int64)) and x > 0,
    }

    @classmethod
    def _promote(
        cls,
        prev_step: Sequence[ModelData],
        gha_min: float | None = None,
        gha_max: float | None = None,
        gha_bin_size: float = 0.1,
    ):
        """
        Convert a list of :class:`ModelData` objects into a combined object.

        Each file is binned in the same regular grid of GHA so all the files can
        be aligned. The final residuals/spectra have shape ``(Nfiles, Ngha, Nfreq)``,
        where each file essentially describes a day/night. This binning is de-biased
        by using the models from the previous step to "in-paint" filtered gaps.

        Parameters
        ----------
        prev_step
            The list of Level1 files.
        gha_min
            The minimum of the regular GHA grid.
        gha_max
            The maximum of the regular GHA grid.
        gha_bin_size
            The bin size of the regular GHA grid.
        xrfi_pipe
            A pipeline to apply xrfi to resulting gridded spectra.

        Returns
        -------
        data
            The combined data.
        """
        if gha_min is None:
            gha_min = np.floor(min(p.ancillary["gha"].min() for p in prev_step))
        if gha_max is None:
            gha_max = np.ceil(max(p.ancillary["gha"].max() for p in prev_step))

        if gha_min < 0 or gha_min > 24 or gha_min >= gha_max:
            raise ValueError("gha_min must be between 0 and 24")

        if gha_max < 0 or gha_max > 24:
            raise ValueError("gha_max must be between 0 and 24")

        if gha_bin_size > (gha_max - gha_min):
            raise ValueError(
                f"gha_bin_size must be smaller than the gha range, got {gha_bin_size}"
            )

        model_params = [p.ancillary["model_params"] for p in prev_step]
        model_resids = [p.resids for p in prev_step]
        flags = [~p.weights.astype("bool") for p in prev_step]

        # Bin in GHA using the models and residuals
        params, resids, weights, gha_edges = cls.bin_gha(
            prev_step,
            model_params,
            model_resids,
            gha_min,
            gha_max,
            gha_bin_size,
            flags=flags,
        )

        data = {"weights": weights, "resids": resids}

        ancillary = {
            "years": [p.year for p in prev_step],
            "days": [p.day for p in prev_step],
            "hours": [p.hour for p in prev_step],
            "gha_edges": gha_edges,
            "model_params": params,
        }

        return prev_step[0].raw_frequencies, data, ancillary, cls._get_meta(locals())

    @classmethod
    def _extra_meta(cls, kwargs):
        return {
            "n_files": len(kwargs["prev_step"]),
        }

    @cached_property
    def dates(self) -> list[tuple[int, int, int]]:
        """All the dates that went into this object."""
        return [
            (y, d, h)
            for y, d, h in zip(
                self.ancillary["years"], self.ancillary["days"], self.ancillary["hours"]
            )
        ]

    @classmethod
    def bin_gha(
        cls,
        model_objs,
        model_params,
        model_resids,
        gha_min,
        gha_max,
        gha_bin_size,
        flags=None,
        use_pbar=True,
    ):
        """Bin a list of files into small aligning bins of GHA."""
        gha_edges = np.arange(gha_min, gha_max, gha_bin_size)
        if np.isclose(gha_max, gha_edges.max() + gha_bin_size):
            gha_edges = np.concatenate((gha_edges, [gha_edges.max() + gha_bin_size]))

        # Averaging data within GHA bins
        weights = np.zeros((len(model_objs), len(gha_edges) - 1, model_objs[0].freq.n))
        resids = np.zeros((len(model_objs), len(gha_edges) - 1, model_objs[0].freq.n))
        params = np.zeros(
            (len(model_objs), len(gha_edges) - 1, model_params[0].shape[-1])
        )

        pbar = tqdm.tqdm(
            enumerate(model_objs),
            unit="files",
            total=len(model_objs),
            disable=not use_pbar,
        )
        for i, l1 in pbar:
            pbar.set_description(f"GHA Binning for {l1.filename.name}")

            gha = l1.ancillary["gha"]

            l1_weights = l1.weights.copy()
            if flags is not None:
                l1_weights[flags[i]] = 0

            params[i], resids[i], weights[i] = averaging.bin_gha_unbiased_regular(
                model_params[i], model_resids[i], l1_weights, gha, gha_edges
            )

        return params, resids, weights, gha_edges

    def plot_daily_residuals(
        self,
        separation: float = 20,
        ax: plt.Axes | None = None,
        gha_min: float = 0,
        gha_max: float = 24,
        freq_resolution: float | None = None,
        days: list[int] | None = None,
        weights: np.ndarray | str | int | None = None,
    ) -> plt.Axes:
        """
        Make a single plot of residuals for each day in the dataset.

        Parameters
        ----------
        separation
            The separation between residuals in K (on the plot).
        ax
            An optional axis on which to plot.
        gha_min
            A minimum GHA to include in the averaged residuals.
        gha_max
            A maximum GHA to include in the averaged residuals.
        freq_resolution
            The frequency resolution to bin the spectra into for the plot. In same
            units as the instance frequencies.
        days
            The integer day numbers to include in the plot. Default is to include
            all days in the dataset.
        weights
            The weights to use for flagging. By default, use the weights of the object.
            If 'old' is given, use the pre-filter weights. Otherwise, must be an array
            the same size as the spectrum/resids.

        Returns
        -------
        ax
            The matplotlib Axes on which the plot is made.
        """
        if not isinstance(weights, np.ndarray):
            weights = self.get_weights(weights)

        gha = (self.ancillary["gha_edges"][1:] + self.ancillary["gha_edges"][:-1]) / 2

        if ax is None:
            fig, ax = plt.subplots(1, 1, figsize=(7, 12))

        mask = (gha > gha_min) & (gha < gha_max)

        for ix, (param, resid, weight, day) in enumerate(
            zip(self.model_params, self.resids, weights, self.days)
        ):
            if np.sum(weight[mask]) == 0:
                continue

            # skip days not explicitly requested.
            if days and day not in days:
                continue

            mean_p, mean_r, mean_w = averaging.bin_gha_unbiased_regular(
                params=param[mask],
                resids=resid[mask],
                weights=weight[mask],
                gha=gha[mask],
                bins=np.array([gha_min, gha_max]),
            )

            if freq_resolution:
                f, mean_r, mean_w = averaging.bin_freq_unbiased_irregular(
                    mean_r, self.freq.freq, mean_w, resolution=freq_resolution
                )
                f = f[0]
            else:
                f = self.freq.freq

            ax.plot(f, mean_r[0] - ix * separation)
            rms = np.sqrt(
                averaging.weighted_mean(data=mean_r[0] ** 2, weights=mean_w[0])[0]
            )
            ax.text(
                self.freq.max + 5,
                -ix * separation,
                f"{day} RMS={rms:.2f}",
            )

        return ax

    def plot_waterfall(
        self,
        day: int | None = None,
        indx: int | None = None,
        quantity: str = "spectrum",
        cmap: str | None = None,
        vmin: float | None = None,
        vmax: float | None = None,
        filt: str | int | None = None,
        ax: plt.Axes | None = None,
        cbar: bool = True,
        xlab: bool = True,
        ylab: bool = True,
        title: bool = True,
        **imshow_kwargs,
    ):
        """
        Make a single waterfall plot of any 2D quantity (weights, spectrum, resids).

        Parameters
        ----------
        day
            The calendar day to plot (eg. 237). Must exist in the dataset
        indx
            The index representing the day to plot. Can be passed instead of `day`.
        quantity
            The quantity to plot -- must exist as an attribute and have the same shape
            as spectrum/resids/weights.
        cmap
            The colormap to use. Default is to use 'coolwarm' for residuals (since it
            is diverging) and 'magma' for spectra.
        vmin
            The minimum colorbar value to use. Auto-set to encompass the range of the
            data symmetrically if plotting residuals (so that zeros are in the middle).
        vmax
            Same as vmin but the max.

        Other Parameters
        ----------------
        Other parameters are passed through to ``plt.imshow``.
        """
        if day is not None:
            indx = self.day_index(day)

        if indx is None:
            raise ValueError("Must either supply 'day' or 'indx'")

        extent = (
            self.freq.min,
            self.freq.max,
            self.ancillary["gha_edges"].min(),
            self.ancillary["gha_edges"].max(),
        )

        q = getattr(self, quantity)[indx]
        assert q.shape == self.resids[indx].shape

        flags = self.get_flags(filt)[indx]
        q = np.where(flags, np.nan, q)

        if quantity == "resids":
            cmap = cmap or "coolwarm"

            if vmin is None:
                vmin = -np.nanmax(np.abs(q))
                vmax = -vmin
        else:
            cmap = cmap or "magma"

        if ax:
            plt.sca(ax)

        plt.imshow(
            q,
            origin="lower",
            aspect="auto",
            extent=extent,
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
            interpolation=imshow_kwargs.pop("interpolation", "none"),
            **imshow_kwargs,
        )

        if xlab:
            plt.xlabel("Frequency [MHz]")
        if ylab:
            plt.ylabel("GHA (hours)")
        if title:
            plt.title(f"{self.dates[indx][0]}-{self.dates[indx][1]}")

        if cbar:
            plt.colorbar()

    def day_index(self, day: int) -> int:
        """
        Get the index corresponding to the given day in the dataset.

        Parameters
        ----------
        day
            The day to find the index for.
        unflagged
            Whether the index should be that of the unflagged data (i.e. the index
            to pass to ``.spectrum``) or for the full list of input files (i.e. that
            to pass to ``previous_level``).

        Returns
        -------
        indx
            The index in appropriate lists of that day's data.
        """
        return self.days.tolist().index(day)


@add_structure
class CombinedBinnedData(_ModelMixin, _ReductionStep, _CombinedFileMixin):
    """
    Object representing gha binned data after observed days are combined.

    See :class:`_ReductionStep` for documentation about the various datasets within this
    class instance. Note that you can always check which data is inside each group by
    checking its ``.keys()``.

    """

    _possible_parents = (CombinedData,)
    _multi_input = False
    _spec_dim = 3
    _self_parent = True

    _ancillary = {
        "years": is_array("int", 1),
        "days": is_array("int", 1),
        "hours": is_array("int", 1),
    }

    _meta = {
        "n_files": lambda x: isinstance(x, (int, np.int, np.int64)) and x > 0,
    }

    @classmethod
    def _promote(
        cls,
        prev_step: CombinedData,
        gha_bin_size: float = 0.1,
        gha_min: None = None,
        gha_max: None = None,
    ):
        """
        Bins the 3D spectrum along time axis for the :class:`CombinedData` object.

        Each day in the 3D spectrum is binned in the same regular grid of GHA.
        The final residuals/spectra have shape ``(Ndays, Ngha, Nfreq)``,
        where each file essentially describes a day/night. This binning is de-biased
        by using the models from the previous step to "in-paint" filtered gaps.

        Parameters
        ----------
        prev_step
            The 3d spectrum from combined data.
        gha_bin_size
            The bin size of the regular GHA grid.

        Returns
        -------
        data
            The combined binned data.
        """
        if gha_bin_size < 0 or gha_bin_size > 24:
            raise ValueError(
                f"gha_bin_size must be non zero or smaller than 24 hours \
                , got {gha_bin_size}"
            )

        flags = prev_step.get_flags()

        # Bin in GHA using the models and residuals
        params, resids, weights, gha_edges = cls.bin_gha(
            prev_step,
            gha_bin_size,
            gha_min,
            gha_max,
            flags=flags,
        )

        data = {"weights": weights, "resids": resids}

        ancillary = {
            "years": prev_step.ancillary["years"],
            "days": prev_step.ancillary["days"],
            "hours": prev_step.ancillary["hours"],
            "gha_edges": gha_edges,
            "model_params": params,
        }

        return prev_step.raw_frequencies, data, ancillary, cls._get_meta(locals())

    @classmethod
    def _extra_meta(cls, kwargs):
        return {
            "n_files": kwargs["prev_step"].meta["n_files"],
        }

    @cached_property
    def dates(self) -> list[tuple[int, int, int]]:
        """All the dates that went into this object."""
        return [
            (y, d, h)
            for y, d, h in zip(
                self.ancillary["years"], self.ancillary["days"], self.ancillary["hours"]
            )
        ]

    @classmethod
    def bin_gha(
        cls,
        combined_obj: CombinedData,
        gha_bin_size: float,
        gha_min: float | None,
        gha_max: float | None,
        flags=None,
    ):
        """Bin a list of files into small aligning bins of GHA."""
        if gha_min is None and gha_max is None:
            gha_edges = np.arange(
                combined_obj.gha_edges.min(), combined_obj.gha_edges.max(), gha_bin_size
            )
            if np.isclose(combined_obj.gha_edges.max(), gha_edges.max() + gha_bin_size):
                gha_edges = np.concatenate(
                    (gha_edges, [gha_edges.max() + gha_bin_size])
                )
        else:
            gha_edges = np.arange(
                gha_min, gha_max + gha_bin_size / 10, gha_bin_size, dtype=float
            )
        # Averaging data within GHA bins
        weights = np.zeros(
            (len(combined_obj.resids), len(gha_edges) - 1, combined_obj.freq.n)
        )
        resids = np.zeros(
            (len(combined_obj.resids), len(gha_edges) - 1, combined_obj.freq.n)
        )
        params = np.zeros(
            (
                len(combined_obj.resids),
                len(gha_edges) - 1,
                combined_obj.model_params.shape[-1],
            )
        )

        gha = combined_obj.gha_centres

        for i, (p, r, w) in enumerate(
            zip(combined_obj.model_params, combined_obj.resids, combined_obj.weights)
        ):

            params[i], resids[i], weights[i] = averaging.bin_gha_unbiased_regular(
                p, r, w, gha, gha_edges
            )

        return params, resids, weights, gha_edges

    def plot_daily_residuals(
        self,
        separation: float = 20,
        ax: plt.Axes | None = None,
        gha_min: float = 0,
        gha_max: float = 24,
        freq_resolution: float | None = None,
        days: list[int] | None = None,
        weights: np.ndarray | str | int | None = None,
    ) -> plt.Axes:
        """
        Make a single plot of residuals for each day in the dataset.

        Parameters
        ----------
        separation
            The separation between residuals in K (on the plot).
        ax
            An optional axis on which to plot.
        gha_min
            A minimum GHA to include in the averaged residuals.
        gha_max
            A maximum GHA to include in the averaged residuals.
        freq_resolution
            The frequency resolution to bin the spectra into for the plot. In same
            units as the instance frequencies.
        days
            The integer day numbers to include in the plot. Default is to include
            all days in the dataset.
        weights
            The weights to use for flagging. By default, use the weights of the object.
            If 'old' is given, use the pre-filter weights. Otherwise, must be an array
            the same size as the spectrum/resids.

        Returns
        -------
        ax
            The matplotlib Axes on which the plot is made.
        """
        if not isinstance(weights, np.ndarray):
            weights = self.get_weights(weights)

        gha = (self.ancillary["gha_edges"][1:] + self.ancillary["gha_edges"][:-1]) / 2

        if ax is None:
            fig, ax = plt.subplots(1, 1, figsize=(7, 12))

        mask = (gha > gha_min) & (gha < gha_max)

        for ix, (param, resid, weight, day) in enumerate(
            zip(self.model_params, self.resids, weights, self.days)
        ):
            if np.sum(weight[mask]) == 0:
                continue

            # skip days not explicitly requested.
            if days and day not in days:
                continue

            mean_p, mean_r, mean_w = averaging.bin_gha_unbiased_regular(
                params=param[mask],
                resids=resid[mask],
                weights=weight[mask],
                gha=gha[mask],
                bins=np.array([gha_min, gha_max]),
            )

            if freq_resolution:
                f, mean_r, mean_w = averaging.bin_freq_unbiased_irregular(
                    mean_r, self.freq.freq, mean_w, resolution=freq_resolution
                )
                f = f[0]
            else:
                f = self.freq.freq

            ax.plot(f, mean_r[0] - ix * separation)
            rms = np.sqrt(
                averaging.weighted_mean(data=mean_r[0] ** 2, weights=mean_w[0])[0]
            )
            ax.text(
                self.freq.max + 5,
                -ix * separation,
                f"{day} RMS={rms:.2f}",
            )

        return ax

    def plot_waterfall(
        self,
        day: int | None = None,
        indx: int | None = None,
        quantity: str = "spectrum",
        cmap: str | None = None,
        vmin: float | None = None,
        vmax: float | None = None,
        filt: str | int | None = None,
        ax: plt.Axes | None = None,
        cbar: bool = True,
        xlab: bool = True,
        ylab: bool = True,
        title: bool = True,
        **imshow_kwargs,
    ):
        """
        Make a single waterfall plot of any 2D quantity (weights, spectrum, resids).

        Parameters
        ----------
        day
            The calendar day to plot (eg. 237). Must exist in the dataset
        indx
            The index representing the day to plot. Can be passed instead of `day`.
        quantity
            The quantity to plot -- must exist as an attribute and have the same shape
            as spectrum/resids/weights.
        cmap
            The colormap to use. Default is to use 'coolwarm' for residuals (since it
            is diverging) and 'magma' for spectra.
        vmin
            The minimum colorbar value to use. Auto-set to encompass the range of the
            data symmetrically if plotting residuals (so that zeros are in the middle).
        vmax
            Same as vmin but the max.

        Other Parameters
        ----------------
        Other parameters are passed through to ``plt.imshow``.
        """
        if day is not None:
            indx = self.day_index(day)

        if indx is None:
            raise ValueError("Must either supply 'day' or 'indx'")

        extent = (
            self.freq.min,
            self.freq.max,
            self.ancillary["gha_edges"].min(),
            self.ancillary["gha_edges"].max(),
        )

        q = getattr(self, quantity)[indx]
        assert q.shape == self.resids[indx].shape

        flags = self.get_flags(filt)[indx]
        q = np.where(flags, np.nan, q)

        if quantity == "resids":
            cmap = cmap or "coolwarm"

            if vmin is None:
                vmin = -np.nanmax(np.abs(q))
                vmax = -vmin
        else:
            cmap = cmap or "magma"

        if ax:
            plt.sca(ax)

        plt.imshow(
            q,
            origin="lower",
            aspect="auto",
            extent=extent,
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
            interpolation=imshow_kwargs.pop("interpolation", "none"),
            **imshow_kwargs,
        )

        if xlab:
            plt.xlabel("Frequency [MHz]")
        if ylab:
            plt.ylabel("GHA (hours)")
        if title:
            plt.title(f"{self.dates[indx][0]}-{self.dates[indx][1]}")

        if cbar:
            plt.colorbar()

    def day_index(self, day: int) -> int:
        """
        Get the index corresponding to the given day in the dataset.

        Parameters
        ----------
        day
            The day to find the index for.
        unflagged
            Whether the index should be that of the unflagged data (i.e. the index
            to pass to ``.spectrum``) or for the full list of input files (i.e. that
            to pass to ``previous_level``).

        Returns
        -------
        indx
            The index in appropriate lists of that day's data.
        """
        return self.days.tolist().index(day)


@add_structure
class DayAveragedData(_ModelMixin, _ReductionStep, _CombinedFileMixin):
    """
    Object representing a dataset that is averaged over days.

    See :class:`_ReductionStep` for documentation about the various datasets within this
    class instance. Note that you can always check which data is inside each group by
    checking its ``.keys()``.
    """

    _possible_parents = (CombinedData, CombinedBinnedData)
    _ancillary = {
        "years": is_array("int", 1),
        "days": is_array("int", 1),
        "hours": is_array("int", 1),
    }

    _meta = {}

    @classmethod
    def _promote(
        cls,
        prev_step: CombinedData,
        day_range: tuple[int, int] | None = None,
        ignore_days: Sequence[int] | None = None,
        gha_filter_file: None | tp.PathLike = None,
        n_threads: int = cpu_count(),
    ):
        """
        Convert a :class:`CombinedData` object into a :class:`DayAveragedData` object.

        This step integrates over days to form a spectrum as a function of GHA and
        frequency. It also applies an optional frequency averaging.

        Parameters
        ----------
        prev_step
            The :class:`CombinedData` object to convert.
        day_range
            Min and max days to include (from a given year).
        ignore_days
            A sequence of days to ignore in the integration.
        xrfi_pipe
            A dictionary specifying further RFI flagging methods. See
            :meth:`Level2.from_previous_level` for details.
        xrfi_on_resids
            Whether to do xRFI on the residuals of the data, or the averaged spectrum
            itself.
        n_threads
            The number of threads to use for the xRFI.
        """
        # Compute the residuals
        days = prev_step.days

        if day_range is None:
            day_range = (days.min(), days.max())

        if ignore_days is None:
            ignore_days = []

        if ignore_days:
            logger.info("Masking days provided in [blue]ignore_days...")
            day_mask = np.array([day not in ignore_days for day in days])
            resid = prev_step.resids[day_mask]
            wght = prev_step.weights[day_mask]
        else:
            resid = prev_step.resids
            wght = prev_step.weights
            day_mask = [True] * len(resid)

        if gha_filter_file:
            raise NotImplementedError("Using a GHA filter file is not yet implemented")

        logger.info("Integrating over nights...")
        # Take mean over nights.
        params = np.nanmean(prev_step.ancillary["model_params"], axis=0)
        resid, wght = averaging.weighted_mean(resid, wght, axis=0)

        data = {
            "resids": resid,
            "weights": wght,
        }

        ancillary = {
            "years": np.unique(prev_step.ancillary["years"]),
            "days": np.unique(prev_step.ancillary["days"]),
            "hours": np.unique(prev_step.ancillary["hours"]),
            "gha_edges": prev_step.ancillary["gha_edges"],
            "model_params": params,
        }

        return prev_step.raw_frequencies, data, ancillary, cls._get_meta(locals())

    def fully_averaged_spectrum(
        self,
        gha_min: float | None = None,
        gha_max: float | None = None,
        weights: np.ndarray | str | int | None = None,
        freq_resolution: int | float = 0,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Get a single fully averaged spectrum at a given frequency resolution.

        Parameters
        ----------
        gha_min
            Minimum GHA to use.
        gha_max
            Maximum GHA to use.
        weights
            The weights to use in the averaging. By default, the weights of the
            instance.

        Returns
        -------
        f
            The frequencies (irregularly spaced).
        spec
            The averaged spectrum (1D)
        weights
            The final weights (1D)
        """
        p, r, w, _ = self.bin_gha(
            gha_min=gha_min, gha_max=gha_max, gha_bin_size=24, weights=weights
        )

        if freq_resolution:
            f, s, w = averaging.bin_freq_unbiased_irregular(
                r[0], self.freq.freq, w[0], resolution=freq_resolution
            )
        else:
            f = self.freq.freq
            s = self.model(parameters=p[0]) + r[0]
            w = w[0]

        return f, s, w

    def bin_gha(
        self,
        gha_min: float | None = None,
        gha_max: float | None = None,
        gha_bin_size: float = 1.0,
        weights: np.ndarray | str | int | None = None,
    ):
        """Bin the data in GHA bins."""
        if gha_min is None:
            gha_min = self.gha_edges.min()
        if gha_max is None:
            gha_max = self.gha_edges.max()

        gha_edges = np.arange(gha_min, gha_max, gha_bin_size)
        if (
            np.isclose(gha_max, gha_edges.max() + gha_bin_size)
            or np.isscalar(gha_max)
            or len(gha_max) == 1
        ):
            gha_edges = np.concatenate((gha_edges, [gha_edges.max() + gha_bin_size]))

        if not isinstance(weights, np.ndarray):
            weights = self.get_weights(weights)

        params, resids, weights = averaging.bin_gha_unbiased_regular(
            self.model_params, self.resids, weights, self.gha_centres, gha_edges
        )
        return params, resids, weights, gha_edges

    def plot_waterfall(
        self, quantity="resids", filt: str | int | None = None, **kwargs
    ):
        """Plot a simple waterfall plot of time vs. frequency."""
        extent = (
            self.freq.min,
            self.freq.max,
            self.gha_edges.min(),
            self.gha_edges.max(),
        )

        flags = self.get_flags(filt)

        q = getattr(self, quantity)
        q = np.where(flags, np.nan, q)

        if quantity == "resids":
            cmap = kwargs.get("cmap", "coolwarm")
        else:
            cmap = kwargs.get("cmap", "magma")

        plt.imshow(q, origin="lower", extent=extent, aspect="auto", cmap=cmap, **kwargs)

        plt.xlabel("Frequency")
        plt.ylabel("GHA")

    def plot_resids(
        self,
        gha_min: float | None = None,
        gha_max: float | None = None,
        weights: np.ndarray | str | int | None = None,
        gha_bin_size: float = 1.0,
        ax: plt.Axes = None,
        freq_resolution=0,
        separation=10,
    ):
        """Plot the residuals."""
        params, resids, weights, gha_edges = self.bin_gha(
            gha_min, gha_max, gha_bin_size=gha_bin_size, weights=weights
        )

        if ax is None:
            fig, ax = plt.subplots(1, 1, figsize=(7, 3 + 2 * len(params)))

        for ix, (rr, ww) in enumerate(zip(resids, weights)):
            if np.sum(ww) == 0:
                continue

            if freq_resolution:
                f, rr, ww = averaging.bin_freq_unbiased_irregular(
                    rr, self.freq.freq, ww, resolution=freq_resolution
                )
            else:
                f = self.freq.freq

            rr = np.where(ww > 0, rr, np.nan)

            ax.plot(f, rr - ix * separation)
            ax.text(
                self.freq.max + 5,
                -ix * separation,
                f"GHA={gha_edges[ix]:.2f} RMS="
                f"{np.sqrt(averaging.weighted_mean(data=rr ** 2, weights=ww)[0]):.2f}",
            )

        return ax


@add_structure
class BinnedData(_ModelMixin, _ReductionStep, _CombinedFileMixin):
    """
    Data that is binned in GHA and/or frequency.

    This step performs a final average over GHA to yield a GHA vs frequency dataset
    that is as averaged as one wants.
    """

    _possible_parents = (DayAveragedData,)
    _self_parent = True
    _ancillary = {
        "years": is_array("int", 1),
        "days": is_array("int", 1),
        "hours": is_array("int", 1),
    }
    _meta = None

    @classmethod
    def _promote(
        cls,
        prev_step: DayAveragedData | BinnedData,
        f_low: float | None = None,
        f_high: float | None = None,
        ignore_freq_ranges: Sequence[tuple[float, float]] | None = None,
        freq_resolution: float | None = None,
        gha_min: float = 0,
        gha_max: float = 24,
        gha_bin_size: float | None = None,
    ):
        """
        Average over GHA and Frequency.

        This step primarily averages further over GHA (potentially over all GHA) and
        potentially over some frequency bins.

        Parameters
        ----------
        prev_step
            The :class:`Level3` objects to average.
        f_low
            The lowest frequency to keep.
        f_high
            The highest frequency to keep.
        ignore_freq_ranges
            Set the weights between these frequency ranges to zero, so they are
            completely ignored in any following fits.
        freq_resolution
            The frequency resolution to average down to.
        gha_min
            The minimum GHA to keep.
        gha_max
            The maximum GHA to keep.
        gha_bin_size
            The GHA bin size after averaging.
        xrfi_pipe
            A final run of xRFI -- see :meth:`Level2.from_previous_level` for details.

        Returns
        -------
        level4
            A :class:`Level4` object.
        """
        freq = FrequencyRange(prev_step.raw_frequencies, f_low=f_low, f_high=f_high)

        resid = prev_step.resids[:, freq.mask]
        wght = prev_step.weights[:, freq.mask]

        if ignore_freq_ranges:
            for (low, high) in ignore_freq_ranges:
                wght[:, (freq.freq >= low) & (freq.freq <= high)] = 0

        if freq_resolution:
            logger.info("Averaging in frequency bins...")
            f, wght, spec, resid, params = averaging.bin_freq_unbiased_regular(
                model=prev_step._model,
                params=prev_step.model_params,
                freq=freq.freq,
                resids=resid,
                weights=wght,
                resolution=freq_resolution,
            )
            logger.info(f".... produced {len(f)} frequency bins.")
        else:
            f = freq.freq
            params = prev_step.model_params

        if gha_bin_size is None:
            gha_bin_size = gha_max - gha_min

        if gha_min > gha_max:
            gha_edges = np.arange(
                gha_min - 24, gha_max + gha_bin_size / 10, gha_bin_size, dtype=float
            )

        else:
            gha_edges = np.arange(
                gha_min, gha_max + gha_bin_size / 10, gha_bin_size, dtype=float
            )

        logger.info(f"Averaging into {len(gha_edges) - 1} GHA bins.")
        params, resid, wght = averaging.bin_gha_unbiased_regular(
            params=params,
            resids=resid,
            weights=wght,
            gha=prev_step.gha_centres,
            bins=gha_edges,
        )

        data = {"resids": resid, "weights": wght}

        ancillary = {
            "years": np.unique(prev_step.ancillary["years"]),
            "days": np.unique(prev_step.ancillary["days"]),
            "hours": np.unique(prev_step.ancillary["hours"]),
            "gha_edges": gha_edges,
            "model_params": params,
        }

        return f, data, ancillary, cls._get_meta(locals())

    def rebin(
        self,
        gha_min=None,
        gha_max=None,
        gha_bin_size=None,
        f_low=None,
        f_high=None,
        resolution=0,
    ):
        """Rebin the data in GHA and frequency."""
        gha_edges = np.arange(
            gha_min, gha_max + gha_bin_size / 10, gha_bin_size, dtype=float
        )
        avg_p, avg_r, avg_w = averaging.bin_gha_unbiased_regular(
            self.model_params,
            self.resids,
            self.weights,
            self.gha_centres,
            bins=gha_edges,
        )

        f, w, s, new_r, new_p = averaging.bin_freq_unbiased_regular(
            model=self._model,
            params=avg_p,
            freq=self.raw_frequencies,
            resids=avg_r,
            weights=avg_w,
            new_freq_edges=np.linspace(f_low, f_high, resolution),
        )

        return BinnedData.from_data(
            {
                "frequency": f,
                "spectra": {"weights": w, "resids": new_r},
                "ancillary": {"gha_edges": gha_edges, "model_params": new_p},
                "meta": {},
            }
        )

    def plot_resids(
        self,
        weights: np.ndarray | int | str | None = None,
        freq_resolution=0,
        separation=10,
        refit_model: mdl.Model | None = None,
        ax=None,
        labels=True,
        f_range: tuple[float | None, float | None] = (0, np.inf),
        plot_full_avg: bool = False,
    ):
        """Plot residuals."""
        if ax is None:
            fig, ax = plt.subplots(1, 1, figsize=(7, 12))

        if not isinstance(weights, np.ndarray):
            weights = self.get_weights(weights)

        for ix, (rr, ww) in enumerate(zip(self.resids, weights)):
            if np.sum(ww) == 0:
                continue

            ww = np.where(
                (self.freq.freq > f_range[0]) & (self.freq.freq < f_range[1]), ww, 0
            )
            if refit_model is not None:
                rr = refit_model.fit(ydata=self.spectrum[ix], weights=ww).residual

            if freq_resolution:
                f, rr, ww = averaging.bin_freq_unbiased_irregular(
                    rr, self.freq.freq, ww, resolution=freq_resolution
                )
            else:
                f = self.freq.freq

            rr = np.where(ww > 0, rr, np.nan)

            ax.plot(f, rr - ix * separation)
            if labels:
                rms = np.sqrt(averaging.weighted_mean(data=rr ** 2, weights=ww)[0])
                ax.text(
                    self.freq.max + 5,
                    -ix * separation,
                    f"GHA={self.gha_edges[ix]:.2f} RMS={rms:.2f}",
                )

        if plot_full_avg:
            # Now average EVERYTHING
            avg_p, avg_r, avg_w = averaging.bin_gha_unbiased_regular(
                self.model_params,
                self.resids,
                weights,
                self.gha_centres,
                bins=np.array([self.gha_edges.min(), self.gha_edges.max()]),
            )

            if refit_model is not None:
                avg_r = refit_model.fit(
                    ydata=avg_r[0] + self.model(parameters=avg_p[0]), weights=avg_w[0]
                ).residual

            ax.plot(f, avg_r[0] - (ix + 2) * separation)

            if labels:
                rms = np.sqrt(
                    averaging.weighted_mean(data=avg_r[0] ** 2, weights=avg_w[0])[0]
                )
                ax.text(
                    self.freq.max + 5,
                    -(ix + 2) * separation,
                    f"Full Avg. RMS={rms:.2f}",
                )

        return ax
