from __future__ import annotations

import glob
import inspect
import json
import logging
import os
import re
import sys
import time
import warnings
from copy import deepcopy
from datetime import datetime
from multiprocessing import cpu_count
from os.path import dirname
from pathlib import Path
from typing import Tuple, Optional, Sequence, List, Union, Dict, Callable, Type

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
    xrfi as rfi,
    Calibration,
)
from edges_io.auxiliary import auxiliary_data
from edges_io.h5 import HDF5Object

from . import s11 as s11m, loss, beams, tools, filters, coordinates
from .coordinates import get_jd, dt_from_jd
from .. import __version__
from .. import const
from ..config import config

logger = logging.getLogger(__name__)


def float_array_ndim(n: int):
    return lambda x: x.ndim == n and x.dtype.name.startswith("float")


def is_array(kind: str, n: int):
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
            raise ValueError(f"That day ({day}) does not exist. Existing days: {self.days}")

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
        ancillary["gha_edges"] = lambda x: x.ndim == 1 and x.dtype.name.startswith("float")

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
    """Base object for formal data reduction steps in edges-analysis.

    The structure is such that four groups will always be available:
    * frequency: the frequencies at which all else is measured.
    * spectra : containing frequency-based data. Arrays here include ``weights`` and
      possibly ``spectrum`` or ``resids`` (depending on the level).
      Each of these will always have the frequency as their _last_ axis.
    * ancillary : containing non-defining data that is not frequency based (usually
      time based). May contain arrays such as ``time``, ``lst``, ``ambient_temp`` etc.
    * meta : parameters defining the data (eg. input parameters) or other scalars
      that describe the data.
    """

    _structure = {}
    _spectra_structure = {}
    _spec_dim = 2
    _possible_parents = tuple()
    default_root = config["paths"]["field_products"]
    _multi_input = False
    _ancillary = {}
    _meta = {}

    def _get_parent_of_kind(self, kind: Type[_ReductionStep]):
        def _get(c):
            if c.__class__ == kind:
                return c
            elif c.__class__ == CalibratedData:
                raise AttributeError(f"This object has no parent of kind {kind.__class__.__name__}")
            elif hasattr(c, "__len__"):
                return [_get(cc) for cc in c]
            else:
                return _get(c.parent)

        return _get(self)

    @cached_property
    def calibration_step(self):
        return self._get_parent_of_kind(CalibratedData)

    @cached_property
    def model_step(self):
        return self._get_parent_of_kind(ModelData)

    @cached_property
    def filter_step(self):
        return self._get_parent_of_kind(FilteredData)

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
    def _get_object_class(fname: [str, Path]):
        with h5py.File(fname, "r") as fl:
            obj_name = fl.attrs.get("object_name", None)

        if obj_name:
            try:
                return getattr(sys.modules[__name__], obj_name)
            except AttributeError:
                raise AttributeError(
                    f"File {fname} has object type {obj_name} which is not a valid ReductionStep."
                )
        else:
            raise AttributeError(f"File {fname} is not a valid HDF5Object")

    @classmethod
    def _promote(
        cls, prev_step: [_ReductionStep, List[_ReductionStep], io.Spectrum], **kwargs
    ) -> Tuple[np.ndarray, dict, dict, dict]:
        pass

    @classmethod
    def promote(
        cls,
        prev_step: [_ReductionStep, List[Union[_ReductionStep, str, Path]], io.Spectrum, str, Path],
        filename: [str, Path, None] = None,
        clobber: bool = False,
        **kwargs,
    ):
        """
        Promote a :class:`_ReductionStep` to this level.

        .. notes::
            This docstring will be overwritten by the :func:`add_structure` class
            decorator on each subclass.
        """

        def _validate_obj(obj):
            # Either convert str/path to the proper object, return the object, or raise.
            if isinstance(obj, (str, Path)):
                with warnings.catch_warnings():
                    warnings.filterwarnings(action="ignore", category=UserWarning)
                    return read_step(obj)
            elif isinstance(obj, cls._possible_parents) or (
                "self" in cls._possible_parents and obj.__class__ == cls
            ):
                return obj
            else:
                raise ValueError(f"{obj} is not a valid data set.")

        if cls._multi_input:
            # Validate each file, and sort them by date.
            # Sorting is important because we need to be able to know which index
            # corresponds to which file later on.
            prev_step = sorted(
                (_validate_obj(obj) for obj in prev_step), key=lambda x: (x.year, x.day, x.hour)
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

        return out

    @cached_property
    def parent(self) -> [_ReductionStep, io.Spectrum, List[Union[_ReductionStep, io.Spectrum]]]:
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
    def weights(self):
        return self.spectra["weights"]

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

        # Need to forcibly overwrite the object name here, because it will be `HDF5Object`
        # Problematically, can just call super()._get_extra_meta, because this function
        # is called in __init_subclass__, and so doesn't know about this class definition
        # when it is called.
        out["object_name"] = cls.__name__

        return out


def read_step(fname: [str, Path]) -> [_ReductionStep, io.HDF5RawSpectrum]:
    fname = Path(fname)
    if fname.suffix == ".acq":
        return io.FieldSpectrum(fname).data
    else:
        return _ReductionStep._get_object_class(fname)(fname)


class _ModelMixin:
    @cached_property
    def model(self) -> mdl.Model:
        """The abstract linear model that is fit to each integration.

        Note that the parameters are not set on this model, but the basis vectors are
        set.
        """
        if isinstance(self.model_step, (list, tuple)):
            meta = self.model_step[0].meta
        else:
            meta = self.model_step.meta

        return mdl.Model.get_mdl(meta["model_basis"])(
            default_x=self.freq.freq, n_terms=meta["model_nterms"]
        )

    @property
    def model_params(self):
        return self.ancillary["model_params"]

    def get_model(self, indx: [int, List[int]], p: Optional[np.ndarray] = None) -> np.ndarray:
        """Obtain the fiducial fitted model spectrum for integration/gha at indx."""
        model = self.model

        if p is None:
            p = self.model_params

        if not hasattr(indx, "__len__"):
            indx = [indx]

        for i in indx:
            p = p[i]

        return model(parameters=p)

    def get_spectrum(
        self, resids: np.ndarray, params: np.ndarray, freq: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """The processed spectra at this level."""
        if freq is not None:
            try:
                del self.model.default_basis
            except AttributeError:
                pass
            self.model.default_x = freq

        indx = np.indices(resids.shape[:-1]).reshape((resids.ndim - 1, -1)).T
        out = np.zeros_like(resids)
        for i in indx:
            ix = tuple(np.atleast_2d(i).T.tolist())
            out[ix] = self.get_model(i, params) + resids[ix]

        if freq is not None:
            del self.__dict__["model"]

        return out

    @cached_property
    def spectrum(self):
        return self.get_spectrum(self.resids, self.model_params)

    @property
    def resids(self):
        """Residuals of all spectra after being fit by the fiducial model."""
        return self.spectra["resids"]


@attr.s
class _Filters:
    _caldata: [CalibratedData, FilteredData] = attr.ib()

    @property
    def ancillary(self):
        return self._caldata.ancillary

    @property
    def weights(self):
        return self._caldata.weights

    @property
    def spectrum(self):
        return self._caldata.spectrum

    @property
    def freq(self):
        return self._caldata.freq

    @property
    def raw_frequencies(self):
        return self._caldata.raw_frequencies

    def aux_filter(
        self,
        sun_el_max: float = 90,
        moon_el_max: float = 90,
        ambient_humidity_max: float = 40,
        min_receiver_temp: float = 0,
        max_receiver_temp: float = 100,
        flags: [None, np.ndarray] = None,
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
        flags
            If given, do filtering in-place.

        Returns
        -------
        flags
            Boolean array giving which entries are bad.
        """

        return filters.time_filter_auxiliary(
            gha=self.ancillary["gha"],
            sun_el=self.ancillary["sun_el"],
            moon_el=self.ancillary["moon_el"],
            humidity=self.ancillary["ambient_hum"],
            receiver_temp=self.ancillary["receiver_temp"],
            sun_el_max=sun_el_max,
            moon_el_max=moon_el_max,
            amb_hum_max=ambient_humidity_max,
            min_receiver_temp=min_receiver_temp,
            max_receiver_temp=max_receiver_temp,
            flags=flags,
        )

    aux_filter.axis = "time"

    def rfi_filter(
        self, xrfi_pipe: dict, flags: [None, np.ndarray] = None, n_threads: int = cpu_count()
    ) -> np.ndarray:
        """
        Perform filtering on auxiliary data and RFI for a level 1 file.

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
        if flags is None:
            flags = np.zeros(self.weights.shape, dtype=bool)

        if "explicit" in xrfi_pipe:
            kwargs = xrfi_pipe.pop("explicit")

            if kwargs["file"] is None:
                known_rfi_file = Path(dirname(__file__)) / "data" / "known_rfi_channels.yaml"
            else:
                known_rfi_file = kwargs["file"]

            flags |= rfi.xrfi_explicit(
                self.raw_frequencies,
                rfi_file=known_rfi_file,
            )

            if np.all(flags):
                return flags

        return tools.run_xrfi_pipe(
            spectrum=self.spectrum,
            freq=self.raw_frequencies,
            flags=flags,
            xrfi_pipe=xrfi_pipe,
            n_threads=n_threads,
            fl_id=self.datestring,
        )

    rfi_filter.axis = "both"

    @property
    def datestring(self):
        """The date this observation was started, as a string."""
        return self._caldata.datestring

    def rms_filter(
        self,
        rms_info: [filters.RMSInfo, str, Path],
        n_sigma_rms: int = 3,
        flags: [np.ndarray, None] = None,
    ):
        weights = self.weights.T if flags is None else np.where(flags, 0, self.weights.T)
        if not isinstance(rms_info, filters.RMSInfo):
            rms_info = filters.RMSInfo.from_file(rms_info)

        rms = {
            mdl_name: self._caldata.get_model_rms(
                freq_ranges=bands, weights=weights.T, **rms_info.model_params[mdl_name]
            )
            for mdl_name, bands in rms_info.bands.items()
        }

        if flags is None:
            flags = np.zeros(self.weights.T.shape, dtype=bool)

        flags |= filters.rms_filter(
            rms_info, self.ancillary["gha"], rms, n_sigma_rms, fl_id=self.datestring
        )

        return flags

    rms_filter.axis = "time"

    def total_power_filter(
        self,
        flags=None,
        n_poly: int = 3,
        n_sigma: float = 3.0,
        bands: [None, List[Tuple[float, float]]] = None,
        std_thresholds=None,
    ):
        if flags is None:
            flags = np.zeros(self.weights.T.shape, dtype=bool)

        flags |= filters.total_power_filter(
            self.ancillary["gha"],
            self.spectrum,
            self.freq.freq,
            flags=flags.T,
            n_poly=n_poly,
            n_sigma=n_sigma,
            bands=bands,
            std_thresholds=std_thresholds,
        )
        return flags

    total_power_filter.axis = "time"

    def negative_power_filter(self, flags: np.ndarray):
        """Filter out integrations that have *any* negative/zero power.

        These integrations obviously have some weird stuff going on.
        """
        return np.array([np.any(spec <= 0) for spec in self.spectrum])

    negative_power_filter.axis = "time"


class _SingleDayMixin:
    @property
    def day(self):
        return self.calibration_step.meta["day"]

    @property
    def year(self):
        return self.calibration_step.meta["year"]

    @property
    def hour(self):
        return self.calibration_step.meta["hour"]

    @property
    def gha(self):
        return self.calibration_step.ancillary["gha"]

    @property
    def lst(self):
        return self.calibration_step.ancillary["lst"]

    @property
    def datestring(self):
        """The date this observation was started, as a string."""
        return f"{self.year:04}-{self.day:03}-{self.hour:02}"

    @property
    def raw_time_data(self):
        """Raw string times at which the spectra were taken."""
        return self.calibration_step.ancillary["times"]

    @cached_property
    def datetimes(self):
        """List of python datetimes at which the spectra were taken."""
        return self.get_datetimes(self.raw_time_data)

    @classmethod
    def get_datetimes(cls, times):
        return [datetime.strptime(d, "%Y:%j:%H:%M:%S") for d in times.astype(str)]

    def bin_in_frequency(
        self,
        indx: Optional[int] = None,
        resolution: float = 0.0488,
        weights: Optional[np.ndarray] = None,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Perform a frequency-average over the spectrum.

        Parameters
        ----------
        indx
            The (time) index at which to compute the frequency-averaged spectrum.
            If not given, returns a 2D array, with time on the first axis.
        resolution : float, optional
            The frequency resolution of the output.
        weights :

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

        new_f, new_s, new_w = tools.bin_array(
            data=s, coords=self.raw_frequencies, weights=w, bins=resolution, axis=-1
        )

        return new_f, new_s, new_w

    def get_model_parameters(
        self,
        model: [str, mdl.Model] = "linlog",
        resolution: [int, float, None] = 0.0488,
        weights: Optional[np.ndarray] = None,
        indices: Optional[List[int]] = None,
        **kwargs,
    ) -> Tuple[mdl.Model, np.ndarray]:
        """
        Determine a callable model of the spectrum at a given time, optionally
        computed over averaged original data.

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

        indices = indices or range(len(self.spectrum))

        if isinstance(model, str):
            model = mdl.Model.get_mdl(model)

        if resolution:
            f, s, w = self.bin_in_frequency(resolution=resolution, weights=weights)
            model = model(**kwargs)
        else:
            f, s, w = self.raw_frequencies, self.spectrum, weights
            model = model(default_x=f, **kwargs)

        def get_params(indx):
            ss = s[indx]
            ww = w[indx]

            if np.sum(ww > 0) <= 2 * model.n_terms:
                # Only try to fit if we have enough non-flagged data points.
                return np.nan * np.ones(model.n_terms)

            if resolution:
                try:
                    del model.default_basis
                except AttributeError:
                    pass
                model.default_x = f[indx]

            return model.fit(ydata=ss, weights=ww).model_parameters

        params = np.array([get_params(indx) for indx in indices])

        return model, params

    def get_model_rms(
        self,
        weights: Optional[np.ndarray] = None,
        freq_ranges: List[Tuple[float, float]] = [(-np.inf, np.inf)],
        indices: Optional[List[int]] = None,
        **model_kwargs,
    ) -> Dict[Tuple[float, float], np.ndarray]:
        """Obtain the RMS of the residual of a model-fit to a particular integration.

        This method is cached, so that calling it again for the same arguments is
        fast.

        Parameters
        ----------
        weights
            The weights of the spectrum to use in the fitting. Must be the same shape
            as :attr:`~spectrum`. Default is to use the weights intrinsic to the object.
        freq_ranges
            While the model given is fit to the entire spectrum, the RMS values can be
            taken over sub-portions of the spectrum. Each frequency range should be
            given as a tuple of (low, high) values, in a list.
        indices
            The integration indices for which to return the RMS.

        Other Parameters
        ----------------
        All other parameters are passed to :method:`~get_model_parameters`.

        Returns
        -------
        rms
            A dictionary where keys are tuples specifying the input bands, and the values
            are arrays of rms values, as a function of time.

        Notes
        -----
        The averaging into frequency bins is *only* done for the fit itself. The final
        residuals are computed on the un-averaged spectrum. The binning is done fully
        self-consistently with the weights  -- it uses :func:`~tools.bin_array` to do
        the binning, returning non-equi-spaced frequencies.
        """
        model, params = self.get_model_parameters(weights=weights, indices=indices, **model_kwargs)

        if indices is None:
            indices = range(len(self.spectrum))
        if weights is None:
            weights = np.ones(self.spectrum.shape)

        # access the default basis
        def _get_rms(indx):

            m = model(x=self.raw_frequencies, parameters=params[indx])

            out = {}
            for band in freq_ranges:
                freq_mask = (self.raw_frequencies >= band[0]) & (self.raw_frequencies < band[1])
                resid = self.spectrum[indx, freq_mask] - m[freq_mask]
                mask = weights[indx, freq_mask] > 0
                out[band] = np.sqrt(np.nanmean(resid[mask] ** 2))

            return out

        res = [_get_rms(i) for i in range(len(indices))]

        # Return dict of arrays (from list of dicts).
        return {band: np.array([r[band] for r in res]) for band in res[0]}

    def plot_waterfall(
        self, quantity: str = "spectrum", ax: [None, plt.Axes] = None, cbar=True, **imshow_kwargs
    ):
        if quantity in ["p0", "p1", "p2"]:
            q = self.calibration_data.spectra["switch_powers"][int(quantity[-1])]
        else:
            q = getattr(self, quantity)

        if ax is not None:
            fig = ax.figure
        else:
            fig, ax = plt.subplots(1, 1)

        if quantity == "resids":
            cmap = imshow_kwargs.pop("cmap", "coolwarm")
        else:
            cmap = imshow_kwargs.pop("cmap", "magma")

        sec = self.calibration_step.ancillary["seconds"]

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
            **imshow_kwargs,
        )

        ax.set_xlabel("Frequency [MHz]")
        ax.set_ylabel("Hours into Observation")
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
            self.plot_waterfall(q, ax=axx, **imshow_kwargs)

        return fig, ax

    def plot_time_averaged_spectrum(
        self,
        quantity="spectrum",
        integrator="mean",
        ax: [None, plt.Axes] = None,
        logy=True,
    ):
        if ax is not None:
            fig = ax.figure
        else:
            fig, ax = plt.subplots(1, 1)

        q, w = self.integrate_over_time(quantity=quantity, integrator=integrator)

        unit = "[K]"
        if quantity == "Q":
            unit = ""

        ax.plot(self.raw_frequencies, q)
        ax.set_xlabel("Frequency [MHz]")
        ax.set_ylabel(f"{quantity} {unit}")

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

        ax[0, 1].plot(self.raw_frequencies, (180 / np.pi) * np.unwrap(np.angle(self.antenna_s11)))
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
class CalibratedData(_ReductionStep, _SingleDayMixin):
    """Object representing the level-1 stage of processing.

    This object essentially represents a Calibrated spectrum.

    See :class:`_ReductionStep` for documentation about the various datasets within this class
    instance. Note that you can always check which data is inside each group by checking
    its ``.keys()``.

    Create the class either directly from a level-1 file (via normal instantiation), or
    by calling :meth:`from_acq` on a raw ACQ file (this does the calibration).

    The data at this level have (in this order):

    * Calibration applied from existing calibration solutions
    * Collected associated weather and thermal auxiliary data (not used at this level,
      just collected)
    * Potential xRFI applied to the raw switch powers individually.
    """

    _spectra_structure = {
        "Q": float_array_ndim(2),
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
        "calobs_path": None,
        "cterms": lambda x: isinstance(x, (int, np.int64)),
        "wterms": lambda x: isinstance(x, (int, np.int64)),
        "freq_max": lambda x: isinstance(x, (float, np.float32)),
        "freq_min": lambda x: isinstance(x, (float, np.float32)),
        "freq_res": lambda x: isinstance(x, (float, np.float32)),
        "n_file_lines": lambda x: isinstance(x, (int, np.int64)),
        "nblk": lambda x: isinstance(x, (int, np.int64)),
        "nfreq": lambda x: isinstance(x, (int, np.int64)),
        "resolution": None,
        "s11_files": None,
        "temperature": None,
    }

    @cached_property
    def filters(self):
        return _Filters(caldata=self)

    @classmethod
    def _promote(
        cls,
        prev_step: io.HDF5RawSpectrum,
        band: str,
        calfile: [str, Path],
        s11_path: [str, Path],
        weather_file: Optional[Union[str, Path]] = None,
        thermlog_file: Optional[Union[str, Path]] = None,
        out_file: Optional[Union[str, Path]] = None,
        progress: bool = True,
        leave_progress: bool = True,
        xrfi_pipe: [None, dict] = None,
        s11_file_pattern: str = r"{y}_{jd}_{h}_*_input{input}.s1p",
        ignore_s11_files: [None, List[str]] = None,
        switch_state_dir: Optional[Union[str, Path]] = None,
        antenna_s11_n_terms: int = 15,
        antenna_correction: [str, Path, None] = ":",
        configuration: str = "",
        balun_correction: [str, Path, None] = ":",
        ground_correction: [str, Path, None, float] = ":",
        beam_file=None,
        f_low: float = 50.0,
        f_high: float = 150.0,
        switch_state_repeat_num: [int, None] = None,
    ) -> Tuple[np.ndarray, dict, dict, dict]:
        """
        Create the object directly from calibrated data.

        Parameters
        ----------
        prev_step
            The filename of the ACQ file to read.
        band
            Defines the instrument that took the data (mid, low, high).
        calfile
            A file containing the output of :method:`edges_cal.CalibrationObservation.write` --
            i.e. all the information required to calibrate the raw data. Determination of
            calibration parameters occurs externally and saved to this file.
        s11_path
            Path to the receiver S11 information relevant to this observation.
        weather_file
            A weather file to use in order to capture that information (may find the
            default weather file automatically).
        thermlog_file
            A thermlog file to use in order to capture that information (may find the
            default thermlog file automatically).
        out_file
            Specify the name of a file to output this particular data to. By default,
            save it in the level cache with the same name as the input file.
        progress
            Whether to show a progress bar.
        leave_progress
            Whether to leave the progress bar on the screen at the end.
        xrfi_pipe
            A dictionary in which keys specify xrfi method names (see :module:`edges_cal.xrfi`)
            and values are dictionaries which specify the parameters to be passed to those
            methods (not requiring the spectrum/weights arguments).
        s11_file_pattern
            A format string defining the naming pattern of S11 files at ``s11_path``.
            This is used to automatically find the S11 file closest in time to the
            observation, if the ``s11_path`` is not explicit (i.e. it is a directory).
        ignore_s11_files
            A list of paths to ignore when attempting to find the S11 file closest to
            the observation (perhaps they are known to be bad).

        Other Parameters
        ----------------
        All other parameters are passed to :method:`_calibrate` -- see its documentation
        for details.

        Returns
        -------
        data
            An instantiated :class:`CalibratedData` object.
        """
        t = time.time()
        q = prev_step["spectra"]["Q"]
        p = [
            prev_step["spectra"]["p0"],
            prev_step["spectra"]["p1"],
            prev_step["spectra"]["p2"],
        ]
        ancillary = prev_step["time_ancillary"]

        logger.info(f"Time for reading: {time.time() - t:.2f} sec.")

        logger.info("Converting time strings to datetimes...")
        t = time.time()
        times = cls.get_datetimes(ancillary["times"])
        logger.info(f"...  finished in {time.time() - t:.2f} sec.")

        meta = {
            "year": times[0].year,
            "day": get_jd(times[0]),
            "hour": times[0].hour,
            **prev_step["meta"],
        }

        time_based_anc = ancillary

        logger.info("Getting ancillary weather data...")
        t = time.time()
        new_anc, new_meta = cls._get_weather_thermlog(band, times, weather_file, thermlog_file)
        meta = {**meta, **new_meta}

        new_anc = {k: new_anc[k] for k in new_anc.dtype.names}
        time_based_anc = {**time_based_anc, **new_anc}
        time_based_anc = {**time_based_anc, **cls.get_ancillary_coords(times)}

        # tools.join_struct_arrays((time_based_anc, new_anc))
        logger.info(f"... finished in {time.time() - t:.2f} sec.")

        s11_files = cls.get_s11_paths(
            s11_path, band, times[0], s11_file_pattern, ignore_files=ignore_s11_files
        )

        logger.info("Calibrating data ...")
        t = time.time()
        calspec, freq, new_meta = cls.calibrate(
            q=q,
            freq=prev_step["freq_ancillary"]["frequencies"],
            band=band,
            calfile=Path(calfile).expanduser(),
            ambient_temp=time_based_anc["ambient_temp"],
            lst=time_based_anc["lst"],
            s11_files=s11_files,
            configuration="",
            switch_state_dir=switch_state_dir,
            antenna_s11_n_terms=antenna_s11_n_terms,
            antenna_correction=antenna_correction,
            balun_correction=balun_correction,
            ground_correction=ground_correction,
            beam_file=beam_file,
            f_low=f_low,
            f_high=f_high,
            switch_state_repeat_num=switch_state_repeat_num,
        )
        logger.info(f"... finished in {time.time() - t:.2f} sec.")

        # RFI cleaning.
        # We need to do any rfi cleaning desired on the raw powers right here, as in
        # future levels they are not stored.
        if xrfi_pipe:
            logger.info("Running xRFI...")
            t = time.time()
            for pspec in p:
                tools.run_xrfi_pipe(spectrum=pspec, freq=freq.freq, xrfi_pipe=xrfi_pipe)
            logger.info(f"... finished in {time.time() - t:.2f} sec.")

        meta = {**meta, **new_meta}
        meta = {**meta, **cls._get_meta(locals())}
        data = {
            "spectrum": calspec,
            "switch_powers": np.array([pp[freq.mask] for pp in p]),
            "weights": np.ones_like(calspec),
            "Q": q[freq.mask],
        }

        return freq.freq, data, time_based_anc, meta

    def get_subset(self, integrations=100):
        """Write a subset of the data to a new mock :class:`CalibratedData` file."""
        freq = self.raw_frequencies
        spectra = {k: self.spectra[k] for k in self.spectra.keys()}
        ancillary = self.ancillary
        meta = self.meta

        spectra = {k: s[:integrations] for k, s in spectra.items()}
        ancillary = ancillary[:integrations]

        return self.from_data(
            {"frequency": freq, "spectra": spectra, "ancillary": ancillary, "meta": meta}
        )

    @classmethod
    def default_s11_directory(cls, band):
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
            the actual filenames. So for example, you could specify ``ignore_files=['2020_076']``
            and it will ignore the file ``/home/user/data/2020_076_01_02_input1.s1p``.
            Full regex can be used.
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
                dt = tools.dt_from_year_day(
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
        s11_path: [str, Path, Tuple, List],
        band: str,
        begin_time: datetime,
        s11_file_pattern: str,
        ignore_files: [None, List[str]] = None,
    ):
        """Given an s11_path, return list of paths for each of the inputs"""

        # If we get four files, make sure they exist and pass them back
        if isinstance(s11_path, (tuple, list)):
            if len(s11_path) != 4:
                raise ValueError("If passing explicit paths to S11 inputs, length must be 4.")

            fls = []
            for pth in s11_path:
                p = Path(pth).expanduser().absolute()
                assert p.exists()
                fls.append(p)

            return fls

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
        assert len(fls) == 4, f"There are not exactly four files matching {s11_path}. Found: {fls}."
        return sorted([Path(fl) for fl in fls])

    @classmethod
    def _get_weather_thermlog(
        cls,
        band: str,
        times: List[datetime],
        weather_file: [None, Path, str] = None,
        thermlog_file: [None, Path, str] = None,
    ):
        """
        Read the appropriate weather and thermlog file, returning their contents.

        Parameters
        ----------
        band
            The band/telescope of the data (mid, low2, low3, high).
        times
            List of datetime objects giving the date-times of the (beginning of) observations.
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
        auxiliary : numpy structured array
            Containing
            * "ambient_temp": Ambient temperature as a function of time
            * "ambient_humidity": Ambient humidity as a function of time
            * "receiver1_temp": Receiver1 temperature as a function of time
            * "receiver2_temp": Receiver2 temperature as a function of time
            * "lst": LST for each observation in the spectrum.
            * "gha": GHA for each observation in the spectrum.
            * "sun_moon_azel": Coordinates of the sun and moon as function of time.
        meta : dict
            Containing
            * "thermlog_file": absolute path to the thermlog information used (filled in with
              the default if necessary).
            * "weather_file": absolute path to the weather information used (filled in with
              the default if necessary).
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
                f"Weather file '{weather_file}' has no dates between {start.strftime('%Y/%m/%d')} "
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
                    dt_from_jd(x["year"], int(x["day"]), x["hour"], x["minute"], x["second"])
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
    def get_ancillary_coords(cls, times: np.ndarray) -> Dict[str, np.ndarray]:
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
        sun, moon = coordinates.sun_moon_azel(const.edges_lat_deg, const.edges_lon_deg, times)
        logger.info(f"Took {time.time() - t} sec to get sun/moon coords.")

        out["sun_az"] = sun[:, 0]
        out["sun_el"] = sun[:, 1]
        out["moon_az"] = moon[:, 0]
        out["moon_el"] = moon[:, 1]

        return out

    @classmethod
    def _get_antenna_s11(
        cls,
        s11_files: Sequence[Union[str, Path]],
        freq: FrequencyRange,
        switch_state_dir: [str, Path],
        n_terms: int,
        switch_state_repeat_num: [int, None],
    ):
        # Get files
        model, raw, raw_freq = s11m.antenna_s11_remove_delay(
            s11_files,
            f_low=freq.min,
            f_high=freq.max,
            switch_state_dir=switch_state_dir,
            delay_0=0.17,
            n_fit=n_terms,
            switch_state_repeat_num=switch_state_repeat_num,
        )
        return model, raw, raw_freq

    @cached_property
    def _antenna_s11(self) -> Tuple[Callable[[np.ndarray], np.ndarray], np.ndarray, np.ndarray]:
        s11_files = self.meta["s11_files"].split(":")
        freq = self.raw_frequencies
        switch_state_dir = self.meta["switch_state_dir"]
        switch_state_repeat_num = self.meta["switch_state_repeat_num"]
        n_terms = self.meta["antenna_s11_n_terms"]

        return self._get_antenna_s11(
            s11_files, freq, switch_state_dir, n_terms, switch_state_repeat_num
        )

    @property
    def antenna_s11_model(self) -> Callable[[np.ndarray], np.ndarray]:
        return self._antenna_s11[0]

    @property
    def antenna_s11(self) -> np.ndarray:
        return self.antenna_s11_model(self.raw_frequencies)

    @property
    def raw_antenna_s11(self) -> np.ndarray:
        return self._antenna_s11[1]

    @property
    def raw_antenna_s11_freq(self) -> np.ndarray:
        return self._antenna_s11[2]

    @cached_property
    def calibration(self):
        """The Calibration object used to calibrate this observation."""
        return Calibration(self.meta["calfile"])

    @classmethod
    def _get_ant_s11_from_calfile(
        cls,
        calfile: Calibration,
        s11_files: Sequence[Union[str, Path]],
        freq: FrequencyRange,
        switch_state_dir: Optional[Union[str, Path]] = None,
        antenna_s11_n_terms: int = 15,
        switch_state_repeat_num=None,
    ):
        if switch_state_dir is not None:
            if calfile.internal_switch is not None:
                warnings.warn(
                    "You should use the switch state that is inherently in the calibration object."
                )
            switch_state_dir = str(Path(switch_state_dir).absolute())
        else:
            if calfile.internal_switch is None:
                raise ValueError(
                    "Internal switch of calfile not found, and no switch_state_dir given!"
                )
            else:
                switch_state_dir = calfile.internal_switch.path

        if switch_state_repeat_num is not None:
            warnings.warn(
                "You should use the switch state repeat_num that is inherently in the "
                "calibration object."
            )
            switch_state_repeat_num = switch_state_repeat_num
        else:
            if calfile.internal_switch is None:
                switch_state_repeat_num = 1
            else:
                switch_state_repeat_num = calfile.internal_switch.repeat_num

        return cls._get_antenna_s11(
            s11_files,
            freq,
            switch_state_dir,
            antenna_s11_n_terms,
            switch_state_repeat_num,
        )[0](freq.freq)

    @classmethod
    def labcal(
        cls, *, q: np.ndarray, freq: FrequencyRange, calfile: Calibration, s11_ant: np.ndarray
    ):

        # Cut the frequency range
        q = q.T[:, freq.mask]

        # Calibrated antenna temperature with losses and beam chromaticity
        calibrated_temp = calfile.calibrate_Q(freq.freq, q, s11_ant)

        return calibrated_temp

    @classmethod
    def loss_correct(
        cls,
        *,
        spec: np.ndarray,
        freq: FrequencyRange,
        band: str,
        antenna_correction: [str, Path, None] = ":",
        ambient_temp: np.ndarray,
        configuration="",
        balun_correction: [str, Path, None] = ":",
        ground_correction: [str, Path, None, float] = ":",
        s11_ant: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        # Antenna Loss (interface between panels and balun)
        gain = np.ones_like(freq.freq)
        if antenna_correction:
            gain *= loss.antenna_loss(
                antenna_correction, freq.freq, band=band, configuration=configuration
            )

        # Balun+Connector Loss
        if balun_correction:
            balun_gain, connector_gain = loss.balun_and_connector_loss(band, freq.freq, s11_ant)
            gain *= balun_gain * connector_gain

        # Ground Loss
        if isinstance(ground_correction, (str, Path)):
            gain *= loss.ground_loss(
                ground_correction, freq.freq, band=band, configuration=configuration
            )
        elif isinstance(ground_correction, float):
            gain *= ground_correction

        a = ambient_temp + const.absolute_zero if ambient_temp[0] < 200 else ambient_temp
        spec = (spec - np.outer(a, (1 - gain))) / gain

        return spec

    @classmethod
    def beam_correct(
        cls,
        *,
        spec: np.ndarray,
        freq: FrequencyRange,
        lst: np.ndarray,
        band: Optional[str] = None,
        beam_file: Optional[Union[str, Path]] = ":",
    ):
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
        calfile: [str, Calibration],
        band: Optional[str] = None,
        s11_files: Sequence[Union[str, Path]],
        ambient_temp: np.ndarray,
        switch_state_dir: Optional[Union[str, Path]] = None,
        antenna_s11_n_terms: int = 15,
        antenna_correction: [str, Path, None] = ":",
        configuration: str = "",
        balun_correction: [str, Path, None] = ":",
        ground_correction: [str, Path, None, float] = ":",
        beam_file=None,
        f_low: float = 50.0,
        f_high: float = 150.0,
        switch_state_repeat_num: [int, None] = None,
        lst: Optional[np.ndarray] = None,
    ) -> Tuple[np.ndarray, FrequencyRange, dict]:
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

        if not isinstance(calfile, Calibration):
            calfile = Calibration(calfile)

        s11_ant = cls._get_ant_s11_from_calfile(
            calfile=calfile,
            s11_files=s11_files,
            freq=freq,
            switch_state_dir=switch_state_dir,
            antenna_s11_n_terms=antenna_s11_n_terms,
            switch_state_repeat_num=switch_state_repeat_num,
        )

        calibrated_temp = cls.labcal(q=q, freq=freq, calfile=calfile, s11_ant=s11_ant)

        calibrated_temp = cls.loss_correct(
            spec=calibrated_temp,
            freq=freq,
            band=band,
            antenna_correction=antenna_correction,
            ambient_temp=ambient_temp,
            configuration=configuration,
            balun_correction=balun_correction,
            ground_correction=ground_correction,
            s11_ant=s11_ant,
        )

        calibrated_temp = cls.beam_correct(
            freq=freq, spec=calibrated_temp, band=band, lst=lst, beam_file=beam_file
        )

        meta = {
            "beam_file": str(Path(beam_file).absolute()) if beam_file is not None else "",
            "s11_files": ":".join(str(f) for f in s11_files),
            "wterms": calfile.wterms,
            "cterms": calfile.cterms,
            "calfile": str(calfile.calfile),
            "calobs_path": str(calfile.calobs_path),
        }

        return calibrated_temp, freq, meta


@add_structure
class FilteredData(_ReductionStep, _SingleDayMixin):
    _meta = {"flagged": lambda x: isinstance(x, (bool, np.bool_))}
    _possible_parents = (CalibratedData, "self")

    _ancillary = {
        "aux_flags": is_array("bool", 2),
        "rms_flags": is_array("bool", 2),
        "tp_flags": is_array("bool", 2),
        "rfi_flags": is_array("bool", 2),
        "neg_flags": is_array("bool", 2),
        "gha": float_array_ndim(1),
    }

    _spectra_structure = {}

    @classmethod
    def run_filter(
        cls,
        fnc,
        cal_data: CalibratedData,
        flags: Optional[List[np.ndarray]] = None,
        **kwargs,
    ):

        logger.info(f"Running {fnc} filter.")

        method = getattr(cal_data.filters, f"{fnc}_filter")
        axis = method.axis

        if flags is None:
            flg = ~cal_data.weights.astype("bool")
        elif np.all(flags):
            return flags
        else:
            flg = flags.copy()

        pre_flag_n = np.sum(flg)

        this_flag = method(flags=flg.T if axis == "time" else flg, **kwargs)

        flg |= this_flag if axis in ("both", "freq") else np.atleast_2d(this_flag).T

        if np.all(flg):
            logger.warning(f"{cal_data.filename.name} was fully flagged during {fnc} filter")
        else:
            logger.info(
                f"'{cal_data.filename.name}': {100 * pre_flag_n / flg.size:.2f} → "
                f"{100 * np.sum(flg) / flg.size:.2f}% [red]<+"
                f"{100 * (np.sum(flg) - pre_flag_n) / flg.size:.2f}%>[/] flagged after "
                f"'{fnc}' filter"
            )

        return flg

    @classmethod
    def _promote(
        cls,
        prev_step: [CalibratedData, FilteredData],
        *,
        sun_el_max: float = 90,
        moon_el_max: float = 90,
        ambient_humidity_max: float = 40,
        min_receiver_temp: float = 0,
        max_receiver_temp: float = 100,
        rms_filter_file: [None, Path, str] = None,
        do_total_power_filter: bool = True,
        xrfi_pipe: [None, dict] = None,
        n_poly_tp_filter: int = 3,
        n_sigma_tp_filter: float = 3.0,
        bands_tp_filter: [None, List[Tuple[float, float]]] = None,
        std_thresholds_tp_filter: [None, List[float]] = None,
        n_sigma_rms: float = 3,
        n_threads: int = cpu_count(),
        model_nterms: int = 5,
        model_basis: str = "linlog",
        model_resolution: [int, float] = 8,
        negative_power_filter: [bool] = True,
    ):
        """
        Convert a list of :class:`Level1` objects into a combined :class:`Level2` object.

        Steps taken tofilter the files are (in order):

        1. Filter entire times from each file based on auxiliary data:
           * Sun/moon position
           * Humidity
           * Receiver Temperature
        2. xRFI (arbitrary flagging routines) on each file. See :module:`edges_cal.xrfi`
           for details.
        3. Filter entire times from each file based on the total calibrated power in
           in each spectrum compared to a gold standard. See
           :method:`~Level1.total_power_filter`
           for details.
        4. Filter entire times from each file based on the RMS of various models fit
           to the spectra or some fraction thereof, and compared to a pre-prepared
           set of fiducial "good" RMS values. See :method:`~Level2._run_rms_filter` for
           details.

        Parameters
        ----------
        prev_step
            The calibrated data (or previously filtered data).
        sun_el_max
            The maximum elevation of the sun with which to still use the data.
        moon_el_max
            The maximum elevation of the moon with which to still use the data.
        ambient_humidity_max
            THe maximum ambient humidity which which to still use the data.
        min_receiver_temp
            Filter data where receiver was below this temperature
        max_receiver_temp
            Filter data where receiver was above this temperature.
        rms_filter_file
            A file output by :func:`~edges_analysis.analysis.filters.get_rms_info`.
            If not given, but ``do_rms_filter=True``, then this file will be created
            on the fly. Other arguments control how it is produced.
        do_total_power_filter
            Whether to use the total power filter.
        xrfi_pipe
            A dictionary where keys are method names in :module:`edges_cal.xrfi`, and
            values are further dictionaries where entries are parameter-value pairs to
            pass to each method.
        n_poly_tp_filter
            See :method:`Level1.total_power_filter` for details.
        n_sigma_tp_filter
            See :method:`Level1.total_power_filter` for details.
        bands_tp_filter
            See :method:`Level1.total_power_filter` for details.
        std_thresholds_tp_filter
            See :method:`Level1.total_power_filter` for details.
        n_sigma_rms
            Number of sigma at which to filter the spectrum.
        n_threads
            Number of threads to use when performing filters (each thread is used for a
            file).
        model_nterms
            The number of terms to use when fitting smooth models to each spectrum.
        model_basis
            The model basis -- a string representing a model from :module:`edges_cal.modelling`
        model_resolution
            If integer, the number of frequency samples binned together before fitting
            the fiducial model. If float, the resolution of the bins in MHz. Set to zero
            to not bin. Residuals of the model are still evaluated at full frequency
            resolution -- this just affects the modeling itself.
        negative_power_filter
            Whether to filter out entire integrations that have any zero/negative power.

        Returns
        -------
        data
            A :class:`FilteredData` object.
        """
        xrfi_pipe = xrfi_pipe or {}

        cal_step = prev_step.calibration_step
        aux_flags = cls.run_filter(
            "aux",
            cal_step,
            sun_el_max=sun_el_max,
            moon_el_max=moon_el_max,
            ambient_humidity_max=ambient_humidity_max,
            min_receiver_temp=min_receiver_temp,
            max_receiver_temp=max_receiver_temp,
        )

        if negative_power_filter:
            neg_flags = cls.run_filter(
                "negative_power",
                cal_step,
                flags=aux_flags,
            )
        else:
            neg_flags = aux_flags

        if xrfi_pipe:
            rfi_flags = cls.run_filter("rfi", cal_step, flags=neg_flags, xrfi_pipe=xrfi_pipe)
        else:
            rfi_flags = neg_flags

        if do_total_power_filter:
            tp_flags = cls.run_filter(
                "total_power",
                cal_step,
                flags=rfi_flags,
                n_poly=n_poly_tp_filter,
                n_sigma=n_sigma_tp_filter,
                std_thresholds=std_thresholds_tp_filter,
                bands=bands_tp_filter,
            )
        else:
            tp_flags = rfi_flags

        if rms_filter_file:
            rms_flags = cls.run_filter(
                "rms",
                cal_step,
                flags=tp_flags,
                rms_info=rms_filter_file,
                n_sigma_rms=n_sigma_rms,
            )
        else:
            rms_flags = tp_flags

        data = {
            "spectrum": prev_step.spectrum,
            "weights": np.where(rms_flags, 0, prev_step.weights),
        }

        ancillary = {
            "aux_flags": aux_flags,
            "rfi_flags": rfi_flags,
            "tp_flags": tp_flags,
            "rms_flags": rms_flags,
            "neg_flags": neg_flags,
            "gha": cal_step.ancillary["gha"],
        }

        return prev_step.raw_frequencies, data, ancillary, cls._get_meta(locals())

    @classmethod
    def _extra_meta(cls, kwargs):
        return {
            "flagged": np.all(kwargs["rms_flags"]),
        }

    @property
    def is_fully_flagged(self):
        return self.meta["flagged"]


@add_structure
class ModelData(_ModelMixin, _ReductionStep, _SingleDayMixin):
    _ancillary = {"gha": float_array_ndim(1)}
    _possible_parents = (CalibratedData, FilteredData)

    @classmethod
    def _promote(
        cls,
        prev_step: [CalibratedData, FilteredData],
        model_nterms: int = 5,
        model_basis: str = "linlog",
        model_resolution: [int, float] = 8,
    ) -> Tuple[np.ndarray, dict, dict, dict]:
        """
        Fit fiducial linear models to each individual spectrum.

        Parameters
        ----------
        prev_step
            The previous step. Can act on calibrated data, or filtered data.
        model_nterms
            The number of terms to use when fitting smooth models to each spectrum.
        model_basis
            The model basis -- a string representing a model from :module:`edges_cal.modelling`
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
        logger.info(
            f"Determining {model_nterms}-term '{model_basis}' models for each integration..."
        )

        # Exit out early if the whole file is flagged.
        try:
            if prev_step.is_fully_flagged:
                raise FullyFlaggedError()
        except AttributeError:
            pass

        model, params = prev_step.get_model_parameters(
            model_basis,
            resolution=model_resolution,
            n_terms=model_nterms,
        )
        x = prev_step.freq.freq if model_resolution else None

        resids = np.array(
            [prev_step.spectrum[j] - model(parameters=pp, x=x) for j, pp in enumerate(params)]
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

    See :class:`_ReductionStep` for documentation about the various datasets within this class
    instance. Note that you can always check which data is inside each group by checking
    its ``.keys()``.

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
        "pre_filter_weights": float_array_ndim(3),
    }

    _meta = {
        "n_files": lambda x: isinstance(x, (int, np.int, np.int64)) and x > 0,
    }

    @classmethod
    def _promote(
        cls,
        prev_step: Sequence[ModelData],
        gha_min: Optional[float] = None,
        gha_max: Optional[float] = None,
        gha_bin_size: float = 0.1,
        xrfi_pipe: Optional[dict] = None,
        xrfi_on_resids: bool = True,
        n_threads: int = cpu_count(),
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
            gha_max = np.ceil(min(p.ancillary["gha"].max() for p in prev_step))

        if gha_min < 0 or gha_min > 24 or gha_min >= gha_max:
            raise ValueError("gha_min must be between 0 and 24")

        if gha_max < 0 or gha_max > 24:
            raise ValueError("gha_max must be between 0 and 24")

        if gha_bin_size > (gha_max - gha_min):
            raise ValueError(f"gha_bin_size must be smaller than the gha range, got {gha_bin_size}")

        model_params = [p.ancillary["model_params"] for p in prev_step]
        model_resids = [p.resids for p in prev_step]
        flags = [~p.weights.astype("bool") for p in prev_step]

        # Bin in GHA using the models and residuals
        params, resids, weights, gha_edges = cls.bin_gha(
            prev_step, model_params, model_resids, gha_min, gha_max, gha_bin_size, flags=flags
        )

        if xrfi_pipe:
            logger.info("Running xRFI...")
            t = time.time()
            flags = tools.run_xrfi_pipe(
                spectrum=resids if xrfi_on_resids else prev_step[0].get_spectrum(resids, params),
                weights=weights,
                freq=prev_step[0].raw_frequencies,
                xrfi_pipe=xrfi_pipe,
                n_threads=n_threads,
            )
            new_weights = np.where(flags, 0, weights)
            logger.info(f"... took {time.time() - t:.2f} sec.")
        else:
            new_weights = weights

        data = {"weights": new_weights, "resids": resids}

        ancillary = {
            "years": [p.year for p in prev_step],
            "days": [p.day for p in prev_step],
            "hours": [p.hour for p in prev_step],
            "gha_edges": gha_edges,
            "model_params": params,
            "pre_filter_weights": weights,
        }

        return prev_step[0].raw_frequencies, data, ancillary, cls._get_meta(locals())

    @classmethod
    def _extra_meta(cls, kwargs):
        return {
            "n_files": len(kwargs["prev_step"]),
        }

    @cached_property
    def dates(self) -> List[Tuple[int, int, int]]:
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
        params = np.zeros((len(model_objs), len(gha_edges) - 1, model_params[0].shape[-1]))

        pbar = tqdm.tqdm(
            enumerate(model_objs), unit="files", total=len(model_objs), disable=not use_pbar
        )
        for i, l1 in pbar:
            pbar.set_description(f"GHA Binning for {l1.filename.name}")

            gha = l1.ancillary["gha"]

            l1_weights = l1.weights.copy()
            if flags is not None:
                l1_weights[flags[i]] = 0

            params[i], resids[i], weights[i] = tools.model_bin_gha(
                model_params[i], model_resids[i], l1_weights, gha, gha_edges
            )

        return params, resids, weights, gha_edges

    def plot_daily_residuals(
        self,
        separation: float = 20,
        ax: [None, plt.Axes] = None,
        gha_min: float = 0,
        gha_max: float = 24,
        freq_resolution: Optional[float] = None,
        days: Optional[List[int]] = None,
        weights: Optional[Union[np.ndarray, str]] = None,
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

        Returns
        -------
        ax
            The matplotlib Axes on which the plot is made.
        """
        if weights is None:
            weights = self.weights
        elif weights == "old":
            weights = self.ancillary["pre_filter_weights"]

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

            mean_p, mean_r, mean_w = tools.model_bin_gha(
                params=param[mask],
                resids=resid[mask],
                weights=weight[mask],
                gha=gha[mask],
                bins=np.array([gha_min, gha_max]),
            )

            if freq_resolution:
                f, mean_r, mean_w, s = tools.average_in_frequency(
                    mean_r, self.freq.freq, mean_w, resolution=freq_resolution
                )
            else:
                f = self.freq.freq

            ax.plot(f, mean_r[0] - ix * separation)
            ax.text(
                self.freq.max + 5,
                -ix * separation,
                f"{day} RMS="
                f"{np.sqrt(tools.weighted_mean(data=mean_r[0] ** 2, weights=mean_w[0])[0]):.2f}",
            )

        return ax

    def plot_waterfall(
        self,
        day: Optional[int] = None,
        indx: Optional[int] = None,
        flagged: bool = False,
        quantity: str = "spectrum",
        cmap: Optional[str] = None,
        vmin: Optional[float] = None,
        vmax: Optional[float] = None,
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
        flagged
            Whether to render pixels that are flagged as NaN.
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

        q = getattr(self, quantity)
        assert q.shape == self.resids.shape

        q = np.where(self.weights[indx] > 0, q[indx], np.nan) if flagged else q[indx]
        if quantity == "resids":
            cmap = cmap or "coolwarm"

            if vmin is None:
                vmin = -np.max(np.abs(q))
                vmax = -vmin
        else:
            cmap = cmap or "magma"

        plt.imshow(
            q,
            origin="lower",
            aspect="auto",
            extent=extent,
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
            **imshow_kwargs,
        )

        plt.xlabel("Frequency [MHz]")
        plt.ylabel("GHA (hours)")
        plt.title(f"Level2 {self.dates[indx][0]}-{self.dates[indx][1]}")

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

    See :class:`_ReductionStep` for documentation about the various datasets within this class
    instance. Note that you can always check which data is inside each group by checking
    its ``.keys()``.
    """

    _possible_parents = (CombinedData,)
    _ancillary = {
        "years": is_array("int", 1),
        "days": is_array("int", 1),
        "hours": is_array("int", 1),
        "pre_filter_weights": float_array_ndim(2),
    }

    _meta = {}

    @classmethod
    def _promote(
        cls,
        prev_step: CombinedData,
        day_range: Optional[Tuple[int, int]] = None,
        ignore_days: Optional[Sequence[int]] = None,
        gha_filter_file: [None, str, Path] = None,
        xrfi_pipe: [None, dict] = None,
        xrfi_on_resids: bool = True,
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
            :method:`Level2.from_previous_level` for details.
        xrfi_on_resids
            Whether to do xRFI on the residuals of the data, or the averaged spectrum
            itself.
        n_threads
            The number of threads to use for the xRFI.
        """
        xrfi_pipe = xrfi_pipe or {}

        # Compute the residuals
        days = prev_step.days
        freq = prev_step.freq

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
        resid, wght = tools.weighted_mean(resid, wght, axis=0)

        # Perform xRFI on GHA-averaged spectra.
        if xrfi_pipe:
            logger.info("Running xRFI...")
            t = time.time()
            flags = tools.run_xrfi_pipe(
                spectrum=resid if xrfi_on_resids else prev_step.get_spectrum(resid, params),
                weights=wght,
                freq=prev_step.raw_frequencies,
                xrfi_pipe=xrfi_pipe,
                n_threads=n_threads,
            )
            new_wght = np.where(flags, 0, wght)
            logger.info(f"... took {time.time() - t:.2f} sec.")
        else:
            new_wght = wght

        data = {
            "resids": resid,
            "weights": new_wght,
        }

        ancillary = {
            "years": np.unique(prev_step.ancillary["years"]),
            "days": np.unique(prev_step.ancillary["days"]),
            "hours": np.unique(prev_step.ancillary["hours"]),
            "gha_edges": prev_step.ancillary["gha_edges"],
            "model_params": params,
            "pre_filter_weights": wght,
        }

        return prev_step.raw_frequencies, data, ancillary, cls._get_meta(locals())

    def bin_gha(
        self,
        gha_min: float,
        gha_max: float,
        gha_bin_size: float = 1.0,
        weights: Optional[Union[np.ndarray, str]] = None,
    ):
        gha_edges = np.arange(gha_min, gha_max, gha_bin_size)
        if np.isclose(gha_max, gha_edges.max() + gha_bin_size):
            gha_edges = np.concatenate((gha_edges, [gha_edges.max() + gha_bin_size]))

        if weights is None:
            weights = self.weights
        elif weights == "old":
            weights = self.ancillary["pre_filter_weights"]

        params, resids, weights = tools.model_bin_gha(
            self.model_params, self.resids, weights, self.gha_centres, gha_edges
        )
        return params, resids, weights, gha_edges

    def plot_waterfall(self, quantity="resids", flagged=True, weights=None, **kwargs):
        """Plot a simple waterfall plot of time vs. frequency."""
        extent = (self.freq.min, self.freq.max, self.gha_edges.min(), self.gha_edges.max())

        if weights is None:
            weights = self.weights
        elif weights == "old":
            weights = self.ancillary["pre_filter_weights"]

        q = getattr(self, quantity)
        if flagged:
            q = np.where(weights > 0, q, np.nan)

        if quantity == "resids":
            cmap = kwargs.get("cmap", "coolwarm")
        else:
            cmap = kwargs.get("cmap", "magma")

        plt.imshow(q, origin="lower", extent=extent, aspect="auto", cmap=cmap, **kwargs)

        plt.xlabel("Frequency")
        plt.ylabel("GHA")

    def plot_resids(
        self,
        gha_min: float,
        gha_max: float,
        weights: Optional[Union[np.ndarray, str]] = None,
        gha_bin_size: float = 1.0,
        ax: plt.Axes = None,
        freq_resolution=0,
        separation=10,
    ):
        params, resids, weights, gha_edges = self.bin_gha(
            gha_min, gha_max, gha_bin_size=gha_bin_size, weights=weights
        )

        if ax is None:
            fig, ax = plt.subplots(1, 1, figsize=(7, 12))

        for ix, (rr, ww) in enumerate(zip(resids, weights)):
            if np.sum(ww) == 0:
                continue

            if freq_resolution:
                f, rr, ww, s = tools.average_in_frequency(
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
                f"{np.sqrt(tools.weighted_mean(data=rr ** 2, weights=ww)[0]):.2f}",
            )

        return ax


@add_structure
class BinnedData(_ModelMixin, _ReductionStep, _CombinedFileMixin):
    """
    A Level-4 Calibrated Spectrum.

    This step performs a final average over GHA to yield a GHA vs frequency dataset
    that is as averaged as one wants.
    """

    _possible_parents = (DayAveragedData, "self")
    _ancillary = {
        "years": is_array("int", 1),
        "days": is_array("int", 1),
        "hours": is_array("int", 1),
        "pre_filter_weights": float_array_ndim(2),
    }
    _meta = None

    @classmethod
    def _promote(
        cls,
        prev_step: [DayAveragedData, BinnedData],
        f_low: Optional[float] = None,
        f_high: Optional[float] = None,
        ignore_freq_ranges: Optional[Sequence[Tuple[float, float]]] = None,
        freq_resolution: Optional[float] = None,
        gha_min: float = 0,
        gha_max: float = 24,
        gha_bin_size: [None, float] = None,
        xrfi_pipe: [None, dict] = None,
        xrfi_on_resids: bool = True,
        n_threads: int = cpu_count(),
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
            Set the weights between these frequency ranges to zero, so they are completely
            ignored in any following fits.
        freq_resolution
            The frequency resolution to average down to.
        gha_min
            The minimum GHA to keep.
        gha_max
            The maximum GHA to keep.
        gha_bin_size
            The GHA bin size after averaging.
        xrfi_pipe
            A final run of xRFI -- see :method:`Level2.from_previous_level` for details.

        Returns
        -------
        level4
            A :class:`Level4` object.
        """
        xrfi_pipe = xrfi_pipe or {}

        freq = FrequencyRange(prev_step.raw_frequencies, f_low=f_low, f_high=f_high)

        resid = prev_step.resids[:, freq.mask]
        wght = prev_step.weights[:, freq.mask]

        if ignore_freq_ranges:
            for (low, high) in ignore_freq_ranges:
                wght[:, (freq.freq >= low) & (freq.freq <= high)] = 0

        if freq_resolution:
            logger.info("Averaging in frequency bins...")
            f, resid, wght, s = tools.average_in_frequency(
                resid, freq.freq, wght, resolution=freq_resolution
            )
            logger.info(f".... produced {len(f)} frequency bins.")
        else:
            f = freq.freq

        if gha_bin_size is None:
            gha_bin_size = gha_max - gha_min

        gha_edges = np.arange(gha_min, gha_max + gha_bin_size / 10, gha_bin_size, dtype=float)

        logger.info(f"Averaging into {len(gha_edges) - 1} GHA bins.")
        params, resid, wght = tools.model_bin_gha(
            prev_step.ancillary["model_params"],
            resid,
            wght,
            prev_step.gha_centres,
            gha_edges,
        )

        # Perform xRFI on GHA-averaged spectra.
        if xrfi_pipe:
            logger.info("Running xRFI...")
            flags = tools.run_xrfi_pipe(
                spectrum=resid if xrfi_on_resids else prev_step.get_spectrum(resid, params, freq=f),
                weights=wght,
                freq=f,
                xrfi_pipe=xrfi_pipe,
                n_threads=n_threads,
            )
            new_wght = np.where(flags, 0, wght)
        else:
            new_wght = wght

        data = {"resids": resid, "weights": new_wght}
        ancillary = {
            "years": np.unique(prev_step.ancillary["years"]),
            "days": np.unique(prev_step.ancillary["days"]),
            "hours": np.unique(prev_step.ancillary["hours"]),
            "gha_edges": gha_edges,
            "model_params": params,
            "pre_filter_weights": wght,
        }

        return f, data, ancillary, cls._get_meta(locals())

    def rebin(
        self, gha_min=None, gha_max=None, gha_bin_size=None, f_low=None, f_high=None, resolution=0
    ):
        gha_edges = np.arange(gha_min, gha_max + gha_bin_size / 10, gha_bin_size, dtype=float)
        avg_p, avg_r, avg_w = tools.model_bin_gha(
            self.model_params,
            self.resids,
            self.weights,
            self.gha_centres,
            bins=gha_edges,
        )

        f, w, s, new_r, new_p = tools.unbiased_freq_bin(
            model_type=self.model.__class__,
            params=avg_p,
            freq=self.raw_frequencies,
            resids=avg_r,
            weights=avg_w,
            new_freq_edges=np.linspace(f_low, f_high, resolution),
            n_terms=self.model_step.meta["model_nterms"],
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
        weights=None,
        freq_resolution=0,
        separation=10,
        refit_model: Optional[mdl.Model] = None,
        ax=None,
        labels=True,
        f_range: Tuple[Optional[float], Optional[float]] = (0, np.inf),
        plot_full_avg: bool = False,
    ):

        if ax is None:
            fig, ax = plt.subplots(1, 1, figsize=(7, 12))

        if weights is None:
            weights = self.weights
        elif weights == "old":
            weights = self.ancillary["pre_filter_weights"]

        for ix, (rr, ww) in enumerate(zip(self.resids, weights)):
            if np.sum(ww) == 0:
                continue

            ww = np.where((self.freq.freq > f_range[0]) & (self.freq.freq < f_range[1]), ww, 0)
            if refit_model is not None:
                rr = refit_model.fit(ydata=self.spectrum[ix], weights=ww).residual

            if freq_resolution:
                f, rr, ww, s = tools.average_in_frequency(
                    rr, self.freq.freq, ww, resolution=freq_resolution
                )
            else:
                f = self.freq.freq

            rr = np.where(ww > 0, rr, np.nan)

            ax.plot(f, rr - ix * separation)
            if labels:
                ax.text(
                    self.freq.max + 5,
                    -ix * separation,
                    f"GHA={self.gha_edges[ix]:.2f} RMS="
                    f"{np.sqrt(tools.weighted_mean(data=rr ** 2, weights=ww)[0]):.2f}",
                )

        if plot_full_avg:
            # Now average EVERYTHING
            avg_p, avg_r, avg_w = tools.model_bin_gha(
                self.model_params,
                self.resids,
                weights,
                self.gha_centres,
                bins=[self.gha_edges.min(), self.gha_edges.max()],
            )

            if refit_model is not None:
                avg_r = refit_model.fit(
                    ydata=avg_r[0] + self.model(parameters=avg_p[0]), weights=avg_w[0]
                ).residual

            ax.plot(f, avg_r[0] - (ix + 2) * separation)

            if labels:
                ax.text(
                    self.freq.max + 5,
                    -(ix + 2) * separation,
                    f"Full Avg. RMS={np.sqrt(tools.weighted_mean(data=avg_r[0] ** 2, weights=avg_w[0])[0]):.2f}",
                )

        return ax


def run_complete_process():
    pass
