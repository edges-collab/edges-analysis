"""
A module containing the class GSData, a variant of UVData specific to single antennas.

The GSData object simplifies handling of radio astronomy data taken from a single
antenna, adding self-consistent metadata along with the data itself, and providing
key methods for data selection, I/O, and analysis.
"""
from __future__ import annotations

import astropy
import astropy.units as un
import datetime
import edges_cal
import edges_cal.modelling as mdl
import edges_io
import functools
import h5py
import logging
import numpy as np
import read_acq
import warnings
import yaml
from astropy.coordinates import EarthLocation, Longitude
from astropy.time import Time
from attrs import asdict, cmp_using
from attrs import converters as cnv
from attrs import define, evolve, field
from attrs import validators as vld
from functools import cached_property, partial
from pathlib import Path
from typing import Callable, Iterable, Literal

from . import __version__
from . import coordinates as crd

logger = logging.getLogger(__name__)

npfield = partial(field, eq=cmp_using(np.array_equal))


class _Register:
    def __init__(self, func: Callable, kind: str) -> None:
        self.func = func
        self.kind = kind
        functools.update_wrapper(self, func, updated=())

    def __call__(self, *args, message: str = "", **kw) -> GSData | list[GSData]:
        newdata = self.func(*args, **kw)

        if isinstance(newdata, GSData):
            return newdata.update(
                history={
                    "message": message,
                    "function": self.func.__name__,
                    "parameters": kw,
                }
            )
        else:
            try:
                return [
                    nd.update(
                        history={
                            "message": message,
                            "function": self.func.__name__,
                            "parameters": kw,
                        }
                    )
                    for nd in newdata
                ]
            except Exception as e:
                raise TypeError(
                    f"{self.func.__name__} returned {type(newdata)} "
                    f"instead of GSData or list thereof."
                ) from e


GSDATA_PROCESSORS = {}


@define
class gsregister:  # noqa: N801
    kind: Literal["gather", "calibrate", "filter", "reduce", "supplement"]

    def __call__(self, func: Callable) -> Callable:
        """Register a function as a processor for GSData objects."""
        out = _Register(func, self.kind)
        GSDATA_PROCESSORS[func.__name__] = out
        return out


@define(frozen=True)
class Stamp:
    """Class representing a historical record of a process applying to an object.

    Parameters
    ----------
    message
        A message describing the process. Optional -- either this or the function
        must be defined.
    function
        The name of the function that was applied. Optional -- either this or the
        message must be defined.
    parameter(s)
        The parameters passed to the function. Optional -- if ``function`` is defined,
        this should be specified.
    versions
        A dictionary of the versions of the software used to perform the process.
        Created by default when the History is created.
    timestamp
        A datetime object corresponding to the time the process was performed.
        By default, this is set to the time that the Stamp object is created.
    """

    message: str = field(default="")
    function: str = field(default="")
    parameters: dict = field(factory=dict)
    versions: dict = field()
    timestamp: datetime.datetime = field(factory=datetime.datetime.now)

    @function.validator
    def _function_vld(self, _, value):
        if not value and not self.message:
            raise ValueError("History record must have a message or a function")

    @versions.default
    def _versions_default(self):
        return {
            "edges-analysis": __version__,
            "edges-cal": edges_cal.__version__,
            "read_acq": read_acq.__version__,
            "edges-io": edges_io.__version__,
            "numpy": np.__version__,
            "astropy": astropy.__version__,
        }

    def _to_yaml_dict(self):
        dct = asdict(self)
        dct["timestamp"] = dct["timestamp"].isoformat()
        return dct

    def __repr__(self):
        """Technical representation of the history record."""
        return yaml.dump(self._to_yaml_dict())

    def __str__(self):
        """Human-readable representation of the history record."""
        pstring = "        ".join(f"{k}: {v}" for k, v in self.parameters.items())
        vstring = " | ".join(f"{k} ({v})" for k, v in self.versions.items())

        return f"""{self.timestamp.isoformat()}
    function: {self.function}
    message : {self.message}
    parameters:
        {pstring}
    versions: {vstring}
        """

    def pretty(self):
        """Return a rich-compatible string representation of the history record."""
        pstring = "        ".join(
            f"[green]{k}[/]: [dim]{v}[/]" for k, v in self.parameters.items()
        )
        vstring = " | ".join(f"{k} ([blue]{v}[/])" for k, v in self.versions.items())

        return f"""[bold underline blue]{self.timestamp.isoformat()}[/]
    [bold green]function[/]  : {self.function}
    [bold green]message [/]  : {self.message}
    [bold green]parameters[/]:
        {pstring}
    [bold green]versions[/]  : {vstring}
        """

    @classmethod
    def from_repr(cls, repr_string: str):
        """Create a Stamp object from a string representation."""
        dct = yaml.load(repr_string, Loader=yaml.FullLoader)
        return cls.from_yaml_dict(dct)

    @classmethod
    def from_yaml_dict(cls, d: dict) -> Stamp:
        """Create a Stamp object from a dictionary representing a history record."""
        d["timestamp"] = datetime.datetime.strptime(
            d["timestamp"], "%Y-%m-%dT%H:%M:%S.%f"
        )
        return cls(**d)


@define(slots=False)
class History:
    """A collection of Stamp objects defining the history."""

    stamps: tuple[Stamp] = field(
        factory=tuple,
        converter=tuple,
        validator=vld.deep_iterable(vld.instance_of(Stamp), vld.instance_of(tuple)),
    )

    def __attrs_post_init__(self):
        """Define the timestamps as keys."""
        self._keysdates = tuple(stamp.timestamp for stamp in self.stamps)
        self._keystring = tuple(stamp.timestamp.isoformat() for stamp in self.stamps)

    def __repr__(self):
        """Technical representation of the history."""
        out = tuple(s._to_yaml_dict() for s in self.stamps)
        return yaml.dump(out)

    def __str__(self):
        """Human-readable representation of the history."""
        return "\n\n".join(str(s) for s in self.stamps)

    def pretty(self):
        """Return a rich-compatible string representation of the history."""
        return "\n\n".join(s.pretty() for s in self.stamps)

    def __getitem__(self, key):
        """Return the Stamp object corresponding to the given key."""
        if isinstance(key, int):
            return self.stamps[key]
        elif isinstance(key, str):
            if key not in self._keystring:
                raise KeyError(
                    f"{key} not in history. Make sure the key is in ISO format."
                )
            return self.stamps[self._keystring.index(key)]
        elif isinstance(key, datetime.datetime):
            if key not in self._keysdates:
                raise KeyError(f"{key} not in history")
            return self.stamps[self._keysdates.index(key)]
        else:
            raise KeyError(
                f"{key} not a valid key. Must be int, ISO date string, or datetime."
            )

    @classmethod
    def from_repr(cls, repr_string: str):
        """Create a History object from a string representation."""
        d = yaml.load(repr_string, Loader=yaml.FullLoader)
        return cls(stamps=[Stamp.from_yaml_dict(s) for s in d])

    def add(self, stamp: Stamp | dict):
        """Add a stamp to the history."""
        if isinstance(stamp, dict):
            stamp = Stamp(**stamp)

        return evolve(self, stamps=self.stamps + (stamp,))

    def __len__(self):
        """Returns the number of stamps."""
        return len(self.stamps)


@define(slots=False)
class GSData:
    """A generic container for Global-Signal data.

    Parameters
    ----------
    data
        The data array (i.e. what the telescope measures). This must be a 4D array whose
        dimensions are (load, polarization, time, frequency). The data can be raw
        powers, calibrated temperatures, or even model residuals to such. Their type is
        specified by the ``data_unit`` attribute.
    freq_array
        The frequency array. This must be a 1D array of frequencies specified as an
        astropy Quantity.
    time_array
        The time array. This must be a 2D array of shape (times, loads). It can be in
        one of two formats: either an astropy Time object, specifying the absolute time,
        or an astropy Longitude object, specying the LSTs. In "lst" mode, there are
        many methods that become unavailable.
    telescope_location
        The telescope location. This must be an astropy EarthLocation object.
    loads
        The names of the loads. Usually there is a single load ("ant"), but arbitrary
        loads may be specified.
    nsamples
        An array with the same shape as the data array, specifying the number of samples
        that go into each data point. This is unitless, and can be used with the
        ``effective_integration_time`` attribute to compute the total effective
        integration time going into any measurement.
    effective_integration_time
        An astropy Quantity that specifies the amount of time going into a single
        "sample" of the data.
    flags
        A dictionary mapping filter names to boolean arrays. Each boolean array has the
        same shape as the data array, and is True where the data is flagged.
    history
        A tuple of dictionaries, each of which is a record of a previous processing
        step.
    telescope_name
        The name of the telescope.
    data_model
        A :class:`GSDataModel` object describing a model of the data per-integration.
        This is not required. If given, the data array is expected to be in the form
        of residuals to the model.
    data_unit
        The type of the data. This must be one of "power", "temperature", "uncalibrated"
        or "model_residuals".
    auxiliary_measurements
        A dictionary mapping measurement names to arrays. Each array must have its
        leading axis be the same length as the time array.
    filename
        The filename from which the data was read (if any). Used for writing additional
        data if more is added (eg. flags, data model).
    """

    data: np.ndarray = npfield()
    freq_array: un.Quantity[un.MHz] = npfield()
    time_array: Time | Longitude = npfield()
    telescope_location: EarthLocation = field(
        validator=vld.instance_of(EarthLocation),
        converter=lambda x: EarthLocation(*x)
        if not isinstance(x, EarthLocation)
        else x,
    )

    loads: tuple[str] = field(converter=tuple)
    nsamples: np.ndarray = npfield()

    effective_integration_time: un.Quantity[un.s] = field(default=1 * un.s)
    flags: dict[str, np.ndarray] = npfield(factory=dict)

    history: History = field(
        factory=History, validator=vld.instance_of(History), eq=False
    )
    telescope_name: str = field(default="unknown")
    data_model: GSDataModel | None = field(default=None)
    data_unit: Literal[
        "power", "temperature", "uncalibrated", "model_residuals"
    ] = field(default="power")
    auxiliary_measurements: dict = field(factory=dict)
    filename: Path | None = field(default=None, converter=cnv.optional(Path))

    @data.validator
    def _data_validator(self, attribute, value):
        if not isinstance(value, np.ndarray):
            raise TypeError("data must be a numpy array")

        if value.ndim != 4:
            raise ValueError(
                "data must be a 4D array: (Nload, Npol, Ntime, Nfreq). "
                f"Got {value.shape}"
            )

        if np.iscomplex(value).any():
            raise ValueError("data must be real")

    @nsamples.validator
    def _nsamples_validator(self, attribute, value):
        if value.shape != self.data.shape:
            raise ValueError("nsamples must have the same shape as data")

        if np.iscomplex(value).any():
            raise ValueError("nsamples must be real")

    @nsamples.default
    def _nsamples_default(self) -> np.ndarray:
        return np.ones_like(self.data)

    @flags.validator
    def _flags_validator(self, attribute, value):
        if not isinstance(value, dict):
            raise TypeError("flags must be a dict")

        for key, flag in value.items():
            if not isinstance(flag, np.ndarray):
                raise TypeError("flags values must be numpy arrays")

            if flag.shape != self.data.shape:
                raise ValueError("flags must have the same shape as the data")

            if flag.dtype != bool:
                raise ValueError("flags must be boolean")

            if not isinstance(key, str):
                raise ValueError("flags keys must be strings")

    @freq_array.validator
    def _freq_array_validator(self, attribute, value):
        if not isinstance(value, un.Quantity):
            raise TypeError("freq_array must be a Quantity")

        if not value.unit.is_equivalent("Hz"):
            raise ValueError("freq_array must have frequency units")

        if value.shape != (self.nfreqs,):
            raise ValueError(
                "freq_array must have the size nfreqs. "
                f"Got {value.shape} instead of {self.nfreqs}"
            )

    @time_array.validator
    def _time_array_validator(self, attribute, value):
        if not isinstance(value, (Time, Longitude)):
            raise TypeError(
                "time_array must either be an astropy Time object or a Longitude object"
            )

        if value.shape != (self.ntimes, self.nloads):
            raise ValueError(
                f"time_array must have the size (ntimes, nloads), got {value.shape} "
                f"instead of {(self.ntimes, self.nloads)}"
            )

    @loads.default
    def _loads_default(self) -> tuple[str]:
        if self.data.shape[0] == 1:
            return ("ant",)
        elif self.data.shape[0] == 3:
            return ("ant", "internal_load", "internal_load_plus_noise_source")
        else:
            raise ValueError(
                "If data has more than one source, loads must be specified"
            )

    @loads.validator
    def _loads_validator(self, attribute, value):
        if len(value) != self.data.shape[0]:
            raise ValueError(
                "loads must have the same length as the number of loads in data"
            )

        if not all(isinstance(x, str) for x in value):
            raise ValueError("loads must be a tuple of strings")

    @effective_integration_time.validator
    def _effective_integration_time_validator(self, attribute, value):
        if not isinstance(value, un.Quantity):
            raise TypeError("effective_integration_time must be a Quantity")

        if not value.unit.is_equivalent("s"):
            raise ValueError("effective_integration_time must be in seconds")

    @auxiliary_measurements.validator
    def _aux_meas_vld(self, attribute, value):
        if not isinstance(value, dict):
            raise TypeError("auxiliary_measurements must be a dictionary")

        if isinstance(self.time_array, Longitude) and value:
            raise ValueError(
                "If times are LSTs, auxiliary_measurements cannot be specified"
            )

        for key, val in value.items():
            if not isinstance(key, str):
                raise TypeError("auxiliary_measurements keys must be strings")
            if not isinstance(val, np.ndarray):
                raise TypeError("auxiliary_measurements values must be arrays")
            if val.shape[0] != self.ntimes:
                raise ValueError(
                    "auxiliary_measurements values must have the size ntimes "
                    f"({self.ntimes}), but for {key} got shape {val.shape}"
                )

    @data_unit.validator
    def _data_unit_validator(self, attribute, value):
        if value not in ("power", "temperature", "uncalibrated", "model_residuals"):
            raise ValueError(
                'data_unit must be one of "power", "temperature", "uncalibrated",'
                '"model_residuals"'
            )

        if value == "model_residuals" and self.data_model is None:
            raise ValueError(
                'data_unit cannot be "model_residuals" if data_model is None'
            )

    @data_model.validator
    def _data_model_validator(self, att, val):
        if val is None:
            return

        if not isinstance(val, GSDataModel):
            raise TypeError("data_model must be a GSDataModel")

        if val.parameters.shape[:3] != self.data.shape[:3]:
            raise ValueError(
                f"data_model parameters shape mismatch: {val.parameters.shape[:3]} "
                f"vs. {self.data.shape[:3]}"
            )

    @cached_property
    def spectra(self) -> np.ndarray:
        """The measured spectra.

        In the case that the data is not in units of "model_residuals", this is simply
        the data. Otherwise, it is the models + data.
        """
        if self.data_unit == "model_residuals":
            return self.data_model.get_spectra(self)
        else:
            return self.data

    @cached_property
    def resids(self) -> np.ndarray:
        """The model residuals.

        In the case that the data is in units of "model_residuals", this is simply
        the data. Otherwise, it is the data - model.
        """
        if self.data_unit != "model_residuals":
            return self.data_model.get_residuals(self)
        else:
            return self.data

    @property
    def nfreqs(self) -> int:
        """The number of frequency channels."""
        return self.data.shape[-1]

    @property
    def nloads(self) -> int:
        """The number of loads."""
        return self.data.shape[0]

    @property
    def ntimes(self) -> int:
        """The number of times."""
        return self.data.shape[-2]

    @property
    def npols(self) -> int:
        """The number of polarizations."""
        return self.data.shape[1]

    @classmethod
    def read_acq(cls, filename: str | Path, **kw) -> GSData:
        """Read an ACQ file."""
        try:
            from read_acq import read_acq
        except ImportError as e:
            raise ImportError(
                "read_acq is not installed -- install it to read ACQ files"
            ) from e

        _, (pant, pload, plns), anc = read_acq.decode_file(filename, meta=True)

        times = Time(anc.data.pop("times"), format="yday", scale="utc")

        return cls(
            data=np.array([pant.T, pload.T, plns.T])[:, np.newaxis],
            time_array=times,
            freq_array=anc.frequencies * un.MHz,
            data_unit="power",
            loads=("ant", "internal_load", "internal_load_plus_noise_source"),
            auxiliary_measurements={name: anc.data[name] for name in anc.data},
            filename=filename,
            **kw,
        )

    @classmethod
    def from_file(cls, filename: str | Path, **kw) -> GSData:
        """Create a GSData instance from a file.

        This method attempts to auto-detect the file type and read it.
        """
        filename = Path(filename)

        if filename.suffix == ".acq":
            return cls.read_acq(filename, **kw)
        elif filename.suffix == ".gsh5":
            return cls.read_gsh5(filename)
        else:
            raise ValueError("Unrecognized file type")

    @classmethod
    def read_gsh5(cls, filename: str) -> GSData:
        """Reads a GSH5 file and stores the data in the GSData object."""
        with h5py.File(filename, "r") as fl:
            data = fl["data"][:]
            lat, lon, alt = fl["telescope_location"][:]
            telescope_location = EarthLocation(
                lat=lat * un.deg, lon=lon * un.deg, height=alt * un.m
            )
            times = fl["time_array"][:]

            if np.all(times < 24.0):
                time_array = Longitude(times * un.hour)
            else:
                time_array = Time(times, format="jd", location=telescope_location)
            freq_array = fl["freq_array"][:] * un.MHz
            data_unit = fl.attrs["data_unit"]
            loads = fl.attrs["loads"].split("|")
            auxiliary_measurements = {
                name: fl["auxiliary_measurements"][name][:]
                for name in fl["auxiliary_measurements"].keys()
            }
            nsamples = fl["nsamples"][:]

            flg_grp = fl["flags"]
            if "names" in flg_grp.attrs:
                flag_keys = flg_grp.attrs["names"]
                flags = flg_grp["values"][:]
                flags = {name: flags[i] for i, name in enumerate(flag_keys)}
            else:
                flags = {}

            filename = filename

            history = History.from_repr(fl.attrs["history"])

            if "data_model" in fl:
                data_model = GSDataModel.from_h5(fl["data_model"])
            else:
                data_model = None

        return cls(
            data=data,
            time_array=time_array,
            freq_array=freq_array,
            data_unit=data_unit,
            loads=loads,
            auxiliary_measurements=auxiliary_measurements,
            filename=filename,
            nsamples=nsamples,
            flags=flags,
            history=history,
            data_model=data_model,
            telescope_location=telescope_location,
        )

    def write_gsh5(self, filename: str) -> GSData:
        """Writes the data in the GSData object to a GSH5 file."""
        with h5py.File(filename, "w") as fl:
            fl["data"] = self.data
            fl["freq_array"] = self.freq_array.to_value("MHz")
            if self.in_lst:
                fl["time_array"] = self.time_array.hour
            else:
                fl["time_array"] = self.time_array.jd

            fl["telescope_location"] = np.array(
                [
                    self.telescope_location.lat.deg,
                    self.telescope_location.lon.deg,
                    self.telescope_location.height.to_value("m"),
                ]
            )

            fl.attrs["loads"] = "|".join(self.loads)
            fl["nsamples"] = self.nsamples
            fl.attrs[
                "effective_integration_time"
            ] = self.effective_integration_time.to_value("s")

            flg_grp = fl.create_group("flags")
            if self.flags:
                flg_grp.attrs["names"] = tuple(self.flags.keys())
                flg_grp.create_dataset(
                    "values",
                    data=np.array(list(self.flags.values())),
                    maxshape=(None,) + self.data.shape,
                    chunks=True,
                )

            fl.attrs["telescope_name"] = self.telescope_name
            fl.attrs["data_unit"] = self.data_unit

            # Now history
            fl.attrs["history"] = repr(self.history)

            # Data model
            if self.data_model is not None:
                self.data_model.write(fl, "data_model")

            # Now aux measurements
            aux_grp = fl.create_group("auxiliary_measurements")
            for name, meas in self.auxiliary_measurements.items():
                aux_grp[name] = meas

        return self.update(filename=filename)

    def update(self, **kwargs):
        """Returns a new GSData object with updated attributes."""
        # If the user passes a single dictionary as history, append it.
        # Otherwise raise an error, unless it's not passed at all.
        history = kwargs.pop("history", None)
        if isinstance(history, Stamp):
            history = self.history.add(history)
        elif isinstance(history, dict):
            history = self.history.add(Stamp(**history))
        elif history is not None:
            raise ValueError("History must be a Stamp object or dictionary")
        else:
            history = self.history

        return evolve(self, history=history, **kwargs)

    def select_freqs(
        self,
        range: tuple[un.Quantity[un.MHz], un.Quantity[un.MHz]] | None = None,
        indx: np.ndarray | None = None,
    ) -> GSData:
        """Selects a subset of the frequency channels."""
        return select_freqs(self, range=range, indx=indx)

    def select_times(
        self,
        range: tuple[Time | Longitude, Time | Longitude] | None = None,
        fmt: str = "jd",
        indx: np.ndarray | None = None,
    ) -> GSData:
        """Selects a subset of the times."""
        return select_times(self, range=range, fmt=fmt, indx=indx)

    def select_lsts(
        self,
        range: tuple[Time | Longitude, Time | Longitude] | None = None,
        indx: np.ndarray | None = None,
    ) -> GSData:
        """Selects a subset of the times."""
        return select_lsts(self, range=range, indx=indx)

    def select(
        self,
        freq_range: tuple[Time | Longitude, Time | Longitude] | None = None,
        freq_indx: np.ndarray | None = None,
        time_range: tuple[Time | Longitude, Time | Longitude] | None = None,
        time_indx: np.ndarray | None = None,
        lst_range: tuple[Time | Longitude, Time | Longitude] | None = None,
    ) -> GSData:
        """Selects a subset of the data."""
        return (
            self.select_freqs(freq_range, freq_indx)
            .select_times(time_range, time_indx)
            .select_lsts(lst_range)
        )

    def __add__(self, other: GSData) -> GSData:
        """Adds two GSData objects."""
        if not isinstance(other, GSData):
            raise TypeError("can only add GSData objects")

        if np.any(self.freq_array != other.freq_array) and np.any(
            self.time_array != other.time_array
        ):
            raise ValueError(
                "Cannot add GSData objects with different frequency and time arrays"
            )

        if not np.all(self.time_array == other.time_array):
            # concatenate over time axis
            data = np.concatenate((self.data, other.data), axis=2)
            aux = {
                k: np.concatenate(
                    (self.auxiliary_measurements[k], other.auxiliary_measurements[k])
                )
                for k in self.auxiliary_measurements
            }
            nsamples = np.concatenate((self.nsamples, other.nsamples), axis=2)
            if all(k in other.flags for k in self.flags):
                flags = {
                    k: np.concatenate((self.flags[k], other.flags[k]), axis=2)
                    for k in self.flags
                }
            else:
                # Can only use "complete flags"
                flags = {
                    "complete": np.concatenate(
                        (self.complete_flags, other.complete_flags), axis=2
                    )
                }

            if getattr(self.data_model, "model", 1) != getattr(
                other.data_model, "model", 2
            ):
                warnings.warn(
                    "data models for two objects are different. "
                    "Result will have no data model."
                )
                data_model = None
            else:
                data_model = self.data_model.update(
                    parameters=np.concatenate(
                        (self.data_model.parameters, other.data_model.parameters),
                        axis=2,
                    )
                )

            if self.in_lst:
                time_array = np.concatenate((self.time_array, other.time_array), axis=0)
            else:
                time_array = Time(
                    np.concatenate((self.time_array.jd, other.time_array.jd), axis=0),
                    format="jd",
                )

            return self.update(
                data=data,
                auxiliary_measurements=aux,
                nsamples=nsamples,
                flags=flags,
                time_array=time_array,
                data_model=data_model,
            )

        if not np.all(self.freq_array == other.freq_array):
            if self.data_model is not None:
                warnings.warn(
                    "Cannot concatenate existing data_models over frequency axis."
                )

            if all(k in other.flags for k in self.flags):
                flags = {
                    k: np.concatenate((self.flags[k], other.flags[k]), axis=3)
                    for k in self.flags
                }
            else:
                # Can only use "complete flags"
                flags = {
                    "complete": np.concatenate(
                        (self.complete_flags, other.complete_flags), axis=3
                    )
                }
            # concatenate over frequency axis
            return self.update(
                data=np.concatenate((self.data, other.data), axis=3),
                nsamples=np.concatenate((self.nsamples, other.nsamples), axis=3),
                freq_array=np.concatenate((self.freq_array, other.freq_array)),
                data_model=None,
                flags=flags,
            )

        # If non of the above, then we have two GSData objects at the same times and
        # frequencies. Adding them should just be a weighted sum.
        if self.auxiliary_measurements or other.auxiliary_measurements:
            raise ValueError("Cannot add GSData objects with auxiliary measurements")

        nsamples = self.flagged_nsamples + other.flagged_nsamples
        mean = (
            self.flagged_nsamples * self.data + other.flagged_nsamples * other.data
        ) / nsamples

        if getattr(self.data_model, "model", 1) != getattr(
            other.data_model, "model", 2
        ):
            warnings.warn(
                "data models for two objects are different. "
                "Result will have no data model."
            )
            data_model = None
        else:
            data_model = self.data_model.update(
                parameters=self.data_model.parameters + other.data_model.parameters
            )
        return self.update(data=mean, nsamples=nsamples, data_model=data_model)

    @cached_property
    def lst_array(self) -> Longitude:
        """The local sidereal time array."""
        if self.in_lst:
            return self.time_array
        else:
            return self.time_array.sidereal_time("apparent", self.telescope_location)

    @cached_property
    def gha(self) -> np.ndarray:
        """The GHA's of the observations."""
        return crd.lst2gha(self.lst_array.hour)

    def get_moon_azel(self) -> tuple[np.ndarray, np.ndarray]:
        """Get the Moon's azimuth and elevation for each time in deg."""
        if self.in_lst:
            raise ValueError(
                "Cannot compute Moon positions when time array is not a Time object"
            )

        return crd.moon_azel(
            self.time_array[:, self.loads.index("ant")], self.telescope_location
        )

    def get_sun_azel(self) -> tuple[np.ndarray, np.ndarray]:
        """Get the Sun's azimuth and elevation for each time in deg."""
        if self.in_lst:
            raise ValueError(
                "Cannot compute Sun positions when time array is not a Time object"
            )

        return crd.sun_azel(
            self.time_array[:, self.loads.index("ant")], self.telescope_location
        )

    def to_lsts(self) -> GSData:
        """
        Converts the time array to LST.

        Warning: this is an irreversible operation. You cannot go back to UTC after
        doing this. Furthermore, the auxiliary measurements will be lost.
        """
        if self.in_lst:
            return self

        return self.update(time_array=self.lst_array, auxiliary_measurements={})

    @property
    def in_lst(self) -> bool:
        """Returns True if the time array is in LST."""
        return isinstance(self.time_array, Longitude)

    @property
    def nflagging_ops(self) -> int:
        """Returns the number of flagging operations."""
        return len(self.flags)

    def get_cumulative_flags(
        self, which_flags: tuple[str] | None = None, ignore_flags: tuple[str] = ()
    ) -> np.ndarray:
        """Returns accumulated flags."""
        if which_flags is None:
            which_flags = self.flags.keys()
        elif not which_flags or not self.flags:
            return np.zeros(self.data.shape, dtype=bool)

        which_flags = tuple(s for s in which_flags if s not in ignore_flags)

        if len(which_flags) == self.nflagging_ops:
            return self.complete_flags
        else:
            return np.any(tuple(self.flags[k] for k in which_flags), axis=0)

    @cached_property
    def complete_flags(self) -> np.ndarray:
        """Returns the complete flag array."""
        return (
            np.any(tuple(self.flags.values()), axis=0)
            if self.flags
            else np.zeros(self.data.shape, dtype=bool)
        )

    def get_flagged_nsamples(
        self, which_flags: tuple[str] | None = None, ignore_flags: tuple[str] = ()
    ) -> np.ndarray:
        """Get the nsamples of the data after accounting for flags."""
        cumflags = self.get_cumulative_flags(which_flags, ignore_flags)
        return self.nsamples * (~cumflags).astype(int)

    @cached_property
    def flagged_nsamples(self) -> np.ndarray:
        """Weights accounting for all flags."""
        return self.get_flagged_nsamples()

    def get_initial_yearday(self, hours: bool = False, minutes: bool = False) -> str:
        """Returns the year-day representation of the first time-sample in the data."""
        if minutes and not hours:
            raise ValueError("Cannot return minutes without hours")

        if hours:
            subfmt = "date_hm"
        else:
            subfmt = "date"

        if self.in_lst:
            raise ValueError(
                "Cannot represent times as year-days, as the object is in LST mode"
            )

        out = self.time_array[0, self.loads.index("ant")].to_value("yday", subfmt)

        if hours and not minutes:
            out = ":".join(out.split(":")[:-1])

        return out

    def add_flags(
        self, filt: str, flags: np.ndarray, append_to_file: bool | None = None
    ):
        """Append a set of flags to the object and optionally append them to file.

        You can always write out a *new* file, but appending flags is non-destructive,
        and so we allow it to be appended, in order to save disk space and I/O.
        """
        if flags.shape != self.data.shape:
            raise ValueError("Shape mismatch between flags and data")

        if filt in self.flags:
            raise ValueError(f"Flags for filter '{filt}' already exist")

        new = self.update(flags={**self.flags, **{filt: flags}})

        if append_to_file is None:
            append_to_file = new.filename is not None

        if append_to_file and new.filename is None:
            raise ValueError(
                "Cannot append to file without a filename specified on the object!"
            )

        if append_to_file:
            with h5py.File(new.filename, "a") as fl:
                flg_grp = fl["flags"]

                if "values" not in flg_grp:
                    flg_grp.create_dataset(
                        "values",
                        data=flags[np.newaxis],
                        maxshape=(None,) + new.data.shape,
                        chunks=True,
                    )
                else:
                    v = flg_grp["values"]

                    if v.shape[0] < len(self.flags):

                        # The file is inconsistent with the object, so we need to
                        # delete it.
                        del flg_grp["values"]
                        flg_grp.create_dataset(
                            "values",
                            data=np.array(list(new.flags.values())),
                            maxshape=(None,) + new.data.shape,
                            chunks=True,
                        )
                    else:
                        v.resize(v.shape[0] + 1, axis=0)
                        v[-1] = flags

                flg_grp.attrs["names"] = tuple(new.flags.keys())

        return new

    def remove_flags(self, filt: str) -> GSData:
        """Remove flags for a given filter."""
        if filt not in self.flags:
            raise ValueError(f"No flags for filter '{filt}'")

        return self.update(flags={k: v for k, v in self.flags.items() if k != filt})

    def time_iter(self) -> Iterable[tuple[slice, slice, slice]]:
        """Returns an iterator over the time axis of data-shape arrays."""
        for i in range(self.ntimes):
            yield (slice(None), slice(None), i, slice(None))

    def load_iter(self) -> Iterable[tuple[int]]:
        """Returns an iterator over the load axis of data-shape arrays."""
        for i in range(self.nloads):
            yield (i,)

    def freq_iter(self) -> Iterable[tuple[slice, slice, slice]]:
        """Returns an iterator over the frequency axis of data-shape arrays."""
        for i in range(self.nfreqs):
            yield (slice(None), slice(None), slice(None), i)


@define
class GSDataModel:
    """A model of a GSData object."""

    model: mdl.Model = field()
    parameters: np.ndarray = npfield()

    @parameters.validator
    def _params_vld(self, att, val):
        if not isinstance(val, np.ndarray):
            raise TypeError("parameters must be a numpy array")

        if val.ndim != 4:
            raise ValueError(
                "parameters must have 4 dimensions (Nloads, Npol, Ntimes, Nparams)"
            )

        if val.shape[-1] != self.model.n_terms:
            raise ValueError(
                f"parameters array has {val.shape[-1]} parameters, "
                f"but model has {self.model.n_terms}"
            )

    @property
    def nloads(self) -> int:
        """Number of loads in the model."""
        return self.parameters.shape[0]

    @property
    def npols(self) -> int:
        """Number of polarisations in the model."""
        return self.parameters.shape[1]

    @property
    def ntimes(self) -> int:
        """Number of times in the model."""
        return self.parameters.shape[2]

    @property
    def nparams(self) -> int:
        """Number of parameters in the model."""
        return self.parameters.shape[3]

    def get_residuals(self, gsdata: GSData) -> np.ndarray:
        """Calculates the residuals of the model given the input GSData object."""
        d = gsdata.spectra.reshape((-1, gsdata.nfreqs))
        p = self.parameters.reshape((-1, gsdata.data_model.nparams))

        model = self.model.at(x=gsdata.freq_array.to_value("MHz"))

        resids = np.zeros_like(d)
        for i, (dd, pp) in enumerate(zip(d, p)):
            resids[i] = dd - model(parameters=pp)

        resids.shape = gsdata.data.shape
        return resids

    def get_spectra(self, gsdata: GSData) -> np.ndarray:
        """Calculates the data spectra given the input GSData object."""
        d = gsdata.spectra.reshape((-1, gsdata.nfreqs))
        p = self.parameters.reshape((-1, self.nparams))

        model = self.model.at(x=gsdata.freq_array.to_value("MHz"))

        spectra = np.zeros_like(d)
        for i, (dd, pp) in enumerate(zip(d, p)):
            spectra[i] = dd + model(parameters=pp)

        spectra.shape = gsdata.data.shape
        return spectra

    @classmethod
    def from_gsdata(cls, model: mdl.Model, gsdata: GSData) -> GSDataModel:
        """Creates a GSDataModel from a GSData object."""
        d = gsdata.spectra.reshape((-1, gsdata.nfreqs))
        w = gsdata.flagged_nsamples.reshape((-1, gsdata.nfreqs))

        xmodel = model.at(x=gsdata.freq_array.to_value("MHz"))

        params = np.zeros((gsdata.nloads * gsdata.npols * gsdata.ntimes, model.n_terms))

        for i, (dd, ww) in enumerate(zip(d, w)):
            try:
                params[i] = xmodel.fit(ydata=dd, weights=ww).model_parameters
            except np.linalg.LinAlgError as e:
                raise ValueError(
                    f"Linear algebra error: {e}.\nIndex={i}\ndata={dd}\nweights={ww}"
                ) from e

        params.shape = (gsdata.nloads, gsdata.npols, gsdata.ntimes, model.n_terms)
        return cls(model=model, parameters=params)

    def update(self, **kw) -> GSDataModel:
        """Return a new GSDataModel instance with updated attributes."""
        return evolve(self, **kw)

    def write(self, fl: h5py.File | h5py.Group, path: str = ""):
        """Write the object to an HDF5 file, potentially to a particular path."""
        grp = fl.create_group(path) if path else fl
        grp.attrs["model"] = yaml.dump(self.model)
        grp.create_dataset("parameters", data=self.parameters)

    @classmethod
    def from_h5(cls, fl: h5py.File | h5py.Group, path: str = "") -> GSDataModel:
        """Read the object from an HDF5 file, potentially from a particular path."""
        grp = fl[path] if path else fl
        model = yaml.load(grp.attrs["model"], Loader=yaml.FullLoader)
        params = grp["parameters"][Ellipsis]
        return cls(model=model, parameters=params)


@gsregister("supplement")
def add_model(data: GSData, *, model: mdl.Model, append_to_file: bool | None = None):
    """Return a new GSData instance which contains a data model."""
    new = data.update(data_model=GSDataModel.from_gsdata(model, data))

    if append_to_file is None:
        append_to_file = new.filename is not None

    if append_to_file and new.filename is None:
        raise ValueError(
            "Cannot append to file without a filename specified on the object!"
        )

    if append_to_file:
        with h5py.File(new.filename, "a") as fl:
            if "data_model" in fl.keys():
                logger.warning(
                    f"Data model already exists in {new.filename}, not overwriting."
                )

            else:
                new.data_model.write(fl, "data_model")

    return new


@gsregister("reduce")
def select_freqs(
    data: GSData,
    *,
    range: tuple[un.Quantity[un.MHz], un.Quantity[un.MHz]] | None = None,
    indx: np.ndarray | None = None,
) -> GSData:
    """Selects a subset of the frequency channels."""
    mask = None
    if range is not None:
        if not isinstance(range[0], un.Quantity):
            logger.warning("frequency range given without units, assuming MHz.")
            range = (range[0] * un.MHz, range[1] * un.MHz)

        mask = (data.freq_array >= range[0]) & (data.freq_array <= range[1])

    if indx is not None:
        if mask is None:
            mask = np.zeros(len(data.freq_array), dtype=bool)
        mask[indx] = True

    if mask is None:
        return data

    return data.update(
        data=data.data[..., mask],
        freq_array=data.freq_array[mask],
        nsamples=data.nsamples[..., mask],
        flags={k: v[..., mask] for k, v in data.flags.items()},
    )


def _mask_times(data: GSData, mask: np.ndarray) -> GSData:
    if mask is None:
        return data

    return data.update(
        data=data.data[:, :, mask],
        time_array=data.time_array[mask],
        auxiliary_measurements={
            k: v[mask] for k, v in data.auxiliary_measurements.items()
        },
        nsamples=data.nsamples[:, :, mask],
        flags={k: v[:, :, mask] for k, v in data.flags.items()},
        data_model=data.data_model.update(
            parameters=data.data_model.parameters[:, :, mask]
        )
        if data.data_model is not None
        else None,
    )


@gsregister("reduce")
def select_times(
    data: GSData,
    *,
    range: tuple[Time | float, Time | float] | None = None,
    fmt: str = "jd",
    indx: np.ndarray | None = None,
    load: int | str = "ant",
) -> GSData:
    """Selects a subset of the times."""
    if data.in_lst:
        raise ValueError("LST-binned data cannot be selected on times.")

    if isinstance(load, str):
        load = data.loads.index(load)

    mask = None
    if range is not None:
        if len(range) != 2:
            raise ValueError("range must be a length-2 tuple")

        if not isinstance(range[0], Time):
            range = (Time(range[0], format=fmt), Time(range[1], format=fmt))

            t = data.time_array[:, load]
            mask = (t >= range[0]) & (t <= range[1])

    if indx is not None:
        if mask is None:
            mask = np.zeros((len(data.time_array),), dtype=bool)
        mask[indx] = True

    return _mask_times(data, mask)


@gsregister("reduce")
def select_lsts(
    data: GSData,
    *,
    range: tuple[Longitude | float, Longitude | float] | None = None,
    indx: np.ndarray | None = None,
    load: int | str = "ant",
) -> GSData:
    """Selects a subset of the times."""
    if isinstance(load, str):
        load = data.loads.index(load)

    mask = None
    if range is not None:
        if len(range) != 2:
            raise ValueError("range must be a length-2 tuple")

        if not isinstance(range[0], Longitude):
            range = (range[0] % 24 * un.hourangle, range[1] % 24 * un.hourangle)

        t = data.lst_array[:, load]
        if range[0] > range[1]:
            mask = (t >= range[1]) & (t <= range[0])
        else:
            mask = (t >= range[0]) & (t <= range[1])

    if indx is not None:
        if mask is None:
            mask = np.zeros((len(data.time_array),), dtype=bool)
        mask[indx] = True

    return _mask_times(data, mask)
