"""
This module contains the class GSData, a variant of UVData specific to single antennas.

The GSData object simplifies handling of radio astronomy data taken from a single 
antenna, adding self-consistent metadata along with the data itself, and providing
key methods for data selection, I/O, and analysis.
"""
from __future__ import annotations
from attrs import define, field, cmp_using, validators as vld, evolve, asdict
from attrs import converters as cnv
import numpy as np
from astropy.coordinates import EarthLocation, Longitude
from typing import Literal, Callable
import astropy.units as un
from astropy.time import Time
from functools import partial
from . import coordinates as crd
from functools import cached_property
from pathlib import Path
import edges_cal.modelling as mdl
import functools
import datetime
import astropy
import read_acq
import edges_cal
import edges_io
from . import __version__
from edges_io.h5 import hickleable
import hickle

npfield = partial(field, eq=cmp_using(np.array_equal))

@hickleable()
@define
class GSData:
    """
    This class is a variant of UVData specific to single antennas.
    """
    _data: np.ndarray = npfield()
    freq_array: un.Quantity['frequency'] = npfield()
    time_array: Time | Longitude = npfield()
    telescope_location: EarthLocation = field(validator=vld.instance_of(EarthLocation))

    loads: tuple[str] = field(converter=tuple)
    nsamples: un.Quantity = npfield()
    

    effective_integration_time: un.Quantity['time'] = field(default=1 * un.s)
    flags: dict[str, np.ndarray] = npfield(factory = dict)
    
    history: tuple[dict] = field(factory=tuple)
    telescope_name: str = field(default='unknown')
    data_model: GSDataModel | None = field(default=None)
    data_unit: Literal['power', 'temperature', 'uncalibrated', 'model_residuals'] = field(default='power')
    auxiliary_measurements: dict = field(factory=dict)
    filename: Path | None = field(default=None, converter=cnv.optional(Path))

    @_data.validator
    def _data_validator(self, attribute, value):
        if not isinstance(value, np.ndarray):
            raise TypeError('data must be a numpy array')

        if value.ndim != 4:
            raise ValueError('data must be a 4D array: (Nload, Npol, Ntime, Nfreq)')
        
        if np.iscomplex(value).any():
            raise ValueError('data must be real')

    @nsamples.validator
    def _nsamples_validator(self, attribute, value):
        if value.shape != self.data.shape:
            raise ValueError('nsamples must have the same shape as data')

        if np.np.iscomplex(value).any():
            raise ValueError('nsamples must be real')

    @nsamples.default
    def _nsamples_default(self) -> np.ndarray:
        return np.ones_like(self.data)

    @flags.validator
    def _flags_validator(self, attribute, value):
        if not isinstance(value, dict):
            raise TypeError('flags must be a dict')

        for key, flag in value:
            if not isinstance(flag, np.ndarray):
                raise TypeError('flags must be a tuple of numpy arrays')

            if flag.shape != self.data.shape:
                raise ValueError('flags must have the same shape as the data')

            if flag.dtype != bool:
                raise ValueError('flags must be boolean')

            if not isinstance(key, str):
                raise ValueError("flag keys must be strings")

    @freq_array.validator
    def _freq_array_validator(self, attribute, value):
        if not isinstance(value, un.Quantity):
            raise TypeError('freq_array must be a Quantity')

        if not value.unit.is_equivalent('Hz'):
            raise ValueError('freq_array must be have frequency units')

        if value.shape != (self.nfreqs,):
            raise ValueError('freq_array must have the size nfreqs')

    @time_array.validator
    def _time_array_validator(self, attribute, value):
        if not isinstance(value, (Time, Longitude)):
            raise TypeError('time_array must either be an astropy Time object or a Longitude object')

        if value.shape != (self.ntimes,):
            raise ValueError('time_array must have the size ntimes')

    @loads.default
    def _loads_default(self) -> tuple[str]:
        if self.data.shape[0] == 1:
            return ('ant',)
        elif self.data.shape[0] == 3:
            return ('ant', 'internal_load', 'internal_load_plus_noise_source')
        else:
            raise ValueError('If data has more than one source, loads must be specified')

    @loads.validator
    def _loads_validator(self, attribute, value):
        if len(value) != self.data.shape[0]:
            raise ValueError('loads must have the same length as the number of loads in data')

        if not all(isinstance(x, str) for x in value):
            raise ValueError('loads must be a tuple of strings')

    @effective_integration_time.validator
    def _effective_integration_time_validator(self, attribute, value):
        if not isinstance(value, un.Quantity):
            raise TypeError('effective_integration_time must be a Quantity')

        if not value.unit.is_equivalent('s'):
            raise ValueError('effective_integration_time must be in seconds')

    @auxiliary_measurements.validator
    def _aux_meas_vld(self, attribute, value):
        if not isinstance(value, dict):
            raise TypeError('auxiliary_measurements must be a dictionary')

        if isinstance(self.time_array, Longitude) and value:
            raise ValueError("If times are LSTs, auxiliary_measurements cannot be specified")

        for key, val in value.items():
            if not isinstance(key, str):
                raise TypeError('auxiliary_measurements keys must be strings')
            if not isinstance(val, np.ndarray):
                raise TypeError('auxiliary_measurements values must be arrays')
            if val.shape != (self.ntimes, ):
                raise ValueError('auxiliary_measurements values must have the size ntimes')

    @history.validator
    def _history_validator(self, attribute, value):
        if not isinstance(value, tuple):
            raise TypeError('history must be a tuple')

        if not all(isinstance(x, dict) for x in value):
            raise TypeError('history must be a tuple of dictionaries')

        if any('timestamp' not in x for x in value):
            raise ValueError('history dictionaries must contain a timestamp')

        if any('message' not in x for x in value):
            raise ValueError('history dictionaries must contain a message')

        if any('function' not in x for x in value):
            raise ValueError('history dictionaries must contain the function name')

        if any('parameters' not in x for x in value):
            raise ValueError('history dictionaries must contain parameters')

        if any('versions' not in x for x in value):
            raise ValueError('history dictionaries must contain versions')

    @data_unit.validator
    def _data_unit_validator(self, attribute, value):
        if value not in ('power', 'temperature', 'uncalibrated', 'model_residuals'):
            raise ValueError('data_unit must be one of "power", "temperature", "uncalibrated", "model_residuals"')

        if value == 'model_residuals' and self.data_model is None:
            raise ValueError('data_unit cannot be "model_residuals" if data_model is None')

    @cached_property
    def spectra(self) -> np.ndarray:
        """The measured spectra. 
        
        In the case that the data is not in units of "model_residuals", this is simply
        the data. Otherwise, it is the models + data.
        """
        if self.data_unit == "model_residuals":
            return self.data_model.get_spectra(self.data)
        else:
            return self.data

    @cached_property
    def resids(self) -> np.ndarray:
        """The model residuals. 
        
        In the case that the data is in units of "model_residuals", this is simply
        the data. Otherwise, it is the data - model.
        """
        if self.data_unit == "model_residuals":
            return self.data_model.get_resids(self.data)
        else:
            return self.data

    @property
    def nfreqs(self) -> int:
        """
        The number of frequency channels.
        """
        return self.data.shape[-1]

    @property
    def nloads(self) -> int:
        """
        The number of loads.
        """
        return self.data.shape[0]

    @property
    def ntimes(self) -> int:
        """
        The number of times.
        """
        return self.data.shape[-2]

    @property
    def npols(self) -> int:
        """
        The number of polarizations.
        """
        return self.data.shape[1]

    @classmethod
    def read_acq(cls, filename: str | Path, **kw) -> 'GSData':
        """
        Read an ACQ file.
        """
        try:
            from read_acq import read_acq
        except ImportError as e:
            raise ImportError('read_acq is not installed -- install it to read ACQ files') from e

        _, (pant, pload, plns), anc = read_acq.decode_file(filename)
        
        times = Time(anc.data.pop("times"), format="%Y:%j:%H:%M:%S", scale='utc')


        return cls(
            data=np.array([pant, pload, plns])[np.newaxis],
            time_array=times,
            freq_array=anc.frequencies * un.MHz,
            data_unit = 'power',
            loads=('ant', 'internal_load', 'internal_load_plus_noise_source'),
            auxiliary_measurements={name: anc.data[name] for name in anc.data},
            **kw
        )

    @classmethod
    def from_file(cls, filename: str | Path, **kw) -> 'GSData':
        """Create a GSData instance from a file.
        
        This method attempts to auto-detect the file type and read it.
        """
        filename = Path(filename)

        if filename.suffix == '.acq':
            return cls.read_acq(filename, **kw)
        elif filename.suffix == '.gsh5':
            return cls.read_gsh5(filename)
        else:
            raise ValueError('Unrecognized file type')

    @classmethod
    def read_gsh5(cls, filename: str) -> 'GSData':
        """
        Reads a GSH5 file and stores the data in the GSData object.
        """
        return hickle.load(filename)

    def write_gsh5(self, filename: str):
        """
        Writes the data in the GSData object to a GSH5 file.
        """
        hickle.dump(self, filename)

    def update(self, **kwargs):
        """
        Returns a new GSData object with updated attributes.
        """
        # If the user passes a single dictionary as history, append it.
        # Otherwise raise an error, unless it's not passed at all.
        history = kwargs.pop('history', None)
        if isinstance(history, dict):
            history = self.history + (history, )
        elif history is not None:
            raise ValueError('History must be a dictionary, which is appended to the tuple')
        else:
            history = self.history

        return evolve(self, history=history, **kwargs)

    def select_freqs(
        self, 
        range: tuple[un.Quantity['frequency'], un.Quantity['frequency']] | None,
        indx: np.ndarray | None
    ) -> 'GSData':
        """
        Selects a subset of the frequency channels.
        """
        if indx is not None:
            data = self.data[..., indx]
            freq = self.freq_array[indx]
        else:
            data = self.data
            freq = self.freq_array

        if range is not None:
            indx = np.where((freq >= range[0]) & (freq <= range[1]))[0]
            data = data[:, :, indx, :]
            freq = freq[indx]

        return self.update(data=data, freq_array=freq)

    def select_times(
        self, 
        range: tuple[Time | Longitude, Time | Longitude] | None,
        indx: np.ndarray | None
    ) -> 'GSData':
        """
        Selects a subset of the times.
        """
        if indx is not None:
            data = self.data[:, :, indx, :]
            time = self.time_array[indx]
            aux = {k: v[indx] for k, v in self.auxiliary_measurements.items()}
        else:
            data = self.data
            time = self.time_array
            aux = self.auxiliary_measurements

        if range is not None:
            if self.in_lst or all(isinstance(r, Time) for r in range):
                indx = np.where((time >= range[0]) & (time <= range[1]))[0]
            else:
                lsts = self.lst_array[indx] if indx is not None else self.lst_array
                indx = np.where((lsts >= range[0]) & (lsts <= range[1]))[0]

            data = data[:, :, indx, :]
            time = time[indx]
            aux = {k: v[indx] for k, v in aux.items()}

        return self.update(data=data, time_array=time, auxiliary_measurements=aux)

    def select(
        self, 
        freq_range: tuple[Time | Longitude, Time | Longitude] | None,
        freq_indx: np.ndarray | None,
        time_range: tuple[Time | Longitude, Time | Longitude] | None,
        time_indx: np.ndarray | None
    ) -> 'GSData':
        """
        Selects a subset of the data.
        """
        return self.select_freqs(freq_range, freq_indx).select_times(time_range, time_indx)
        
    def __add__(self, other: 'GSData') -> 'GSData':
        """
        Adds two GSData objects.
        """
        if not isinstance(other, GSData):
            raise TypeError('can only add GSData objects')

        if np.any(self.freq_array != other.freq_array) and np.any(self.time_array != other.time_array):
            raise ValueError('Cannot add GSData objects with different frequency and time arrays')

        if not np.all(self.time_array == other.time_array):
            # concatenate over time axis
            data = np.concatenate((self.data, other.data), axis=1)
            aux = {k: np.concatenate((self.auxiliary_measurements[k], other.auxiliary_measurements[k])) for k in self.auxiliary_measurements}
            nsamples = np.concatenate((self.nsamples, other.nsamples), axis=1)
            if all(k in other.flags for k in self.flags):
                flags = {k: np.concatenate((self.flags[k], other.flags[k]), axis=1) for k in self.flags}
            else:
                # Can only use "complete flags"
                flags = {'complete': np.concatenate((self.complete_flags, other.complete_flags), axis=1)}

            return self.update(
                data = data, 
                auxiliary_measurements=aux, 
                nsamples=nsamples, 
                flags=flags, 
                time_array=np.concatenate((self.time_array, other.time_array))
            )

        if not np.all(self.freq_array == other.freq_array):
            # concatenate over frequency axis
            return self.update(
                data = np.concatenate((self.data, other.data), axis=2),
                freq_array = np.concatenate((self.freq_array, other.freq_array))
            )

        # If non of the above, then we have two GSData objects at the same times and
        # frequencies. Adding them should just be a weighted sum.
        if self.auxiliary_measurements or other.auxiliary_measurements:
            raise ValueError('Cannot add GSData objects with auxiliary measurements')

        nsamples = (self.flagged_nsamples + other.flagged_nsamples)
        mean = (self.flagged_nsamples * self.data + other.flagged_nsamples * other.data) / nsamples
        
        return self.update(data = mean, nsamples = nsamples)

    def __mul__(self, other: np.typing.ArrayLike) -> 'GSData':
        """
        Multiplies the data by a scalar.
        """
        return self.update(data=self.data * other)

    @cached_property
    def lst_array(self) -> Longitude:
        """
        The local sidereal time array.
        """
        if self.in_lst:
            return self.time_array
        else:
            return self.time_array.sidereal_time('apparent', self.telescope_location)

    @cached_property
    def gha(self) -> np.ndarray:
        """The GHA's of the observations."""
        return crd.lst2gha(self.lst_array)

    def get_moon_azel(self) -> tuple[np.ndarray, np.ndarray]:
        """
        Returns the azimuth and elevation of the moon for each time in the observation.
        """
        if self.in_lst:
            raise ValueError("Cannot compute Moon positions when time array is not a Time object")

        return crd.moon_azel(self.time_array, self.telescope_location)

    def get_sun_azel(self) -> tuple[np.ndarray, np.ndarray]:
        """
        Returns the azimuth and elevation of the moon for each time in the observation.
        """
        if self.in_lst:
            raise ValueError("Cannot compute Moon positions when time array is not a Time object")

        return crd.sun_azel(self.time_array, self.telescope_location)
    
    def to_lsts(self) -> 'GSData':
        """
        Converts the time array to LST.

        Warning: this is an irreversible operation. You cannot go back to UTC after
        doing this. Furthermore, the auxiliary measurements will be lost.
        """
        if not isinstance(self.time_array, Time):
            raise ValueError("Cannot convert time array to LST when time array is not a Time object")

        return self.update(time_array=self.lst_array, auxiliary_measurements={})

    @property
    def in_lst(self) -> bool:
        """
        Returns True if the time array is in LST.
        """
        return isinstance(self.time_array, Longitude)

    @property
    def nflagging_ops(self) -> int:
        """
        Returns the number of flagging operations.
        """
        return len(self.flags)

    def get_cumulative_flags(self, which_flags: tuple[str] | None = None, ignore_flags: tuple[str] = ()) -> np.ndarray:
        """Returns accumulated flags."""
        if which_flags is None:
            which_flags = self.flags.keys()

        which_flags = tuple(s for s in which_flags if s not in ignore_flags)

        if len(which_flags) == len(self.flags):
            complete = self.complete_flags
        else:
            complete = np.any(tuple(self.flags[k] for k in which_flags), axis=0)

    @cached_property
    def complete_flags(self) -> np.ndarray:
        """
        Returns the complete flag array.
        """
        return np.any(self.flags.values(), axis=0) if self.flags else np.zeros(self.data.shape, dtype=bool)

    def get_flagged_nsamples(self, which_flags: tuple[str] | None = None, ignore_flags: tuple[str] = ()) -> np.ndarray:
        """
        Get the nsamples of the data after accounting for flags.
        """
        cumflags = self.get_cumulative_flags(which_flags, ignore_flags)
        return self.nsamples * (~cumflags).astype(int)

    @cached_property
    def flagged_nsamples(self) -> np.ndarray:
        """Weights accounting for all flags."""
        return self.get_flagged_nsamples()

    def get_initial_yearday(self) -> str:
        """
        Returns the year-day representation of the first recorded time-sample in the data.
        """
        if self.in_lst:
            raise ValueError("Cannot represent times as year-days, as the object is in LST mode")

        return self.time_array[0].to_value("yday")

    def add_model(self, model: mdl.Model):
        """Return a new GSData instance which contains a data model."""
        return self.update(data_model=GSDataModel.from_gsdata(model, self))

    def add_flags(self, filt: str, flags: np.ndarray, append_to_file: bool | None = None):
        """Append a set of flags to the object and optionally append them to file on disk.
                
        You can always write out a *new* file, but appending flags is non-destructive,
        and so we allow it to be appended, in order to save disk space and I/O.
        """
        if flags.shape != self.data.shape:
            raise ValueError("Shape mismatch between flags and data")

        new = self.update(flags={**self.flags, **{filt: flags}})
        
        if append_to_file is None:
            append_to_file = new.filename is not None

        if append_to_file and new.filename is None:
            raise ValueError("Cannot append to file without a filename specified on the object!")

        if append_to_file:        
            raise NotImplementedError

        return new

@hickleable
@define
class GSDataModel:
    """
    A model of a GSData object.
    """
    model: mdl.Model = field()
    parameters: np.ndarray = npfield()

    @parameters.validator
    def _params_vld(self, att, val):
        if not isinstance(val, np.ndarray):
            raise(" TypeError: parameters must be a numpy array")

        if val.ndim != 4:
            raise ValueError("parameters must have 4 dimensions (Nloads, Npol, Ntimes, Nparams")

    @property
    def nloads(self) -> int:
        return self.parameters.shape[0]

    @property
    def npol(self) -> int:
        return self.parameters.shape[1]

    @property
    def ntimes(self) -> int:
        return self.parameters.shape[2]

    @property
    def nparams(self) -> int:
        return self.parameters.shape[3]
        
    def get_residuals(self, gsdata: GSData) -> np.ndarray:
        """Calculates the residuals of the model given the input GSData object."""
        if gsdata.data_unit == "model_residuals":
            raise ValueError("Cannot compute model residuals on data that is already residuals!")

        d = gsdata.data.reshape((-1, gsdata.nfreqs))
        p = self.parameters.reshape((-1, gsdata.nparams))

        model = self.model.at(x=gsdata.freq_array.to_value("MHz"))

        resids = np.zeros_like(d)
        for i, (dd, pp) in enumerate(zip(d, p)):
            resids[i] = dd - model(parameters=pp)

        resids.shape = gsdata.data.shape
        return resids

    def get_spectra(self, gsdata: GSData) -> np.ndarray:
        """Calculates the data spectra given the input GSData object."""
        if gsdata.data_unit != "model_residuals":
            raise ValueError("Cannot compute model spectra on data that aren't residuals!")

        d = gsdata.data.reshape((-1, gsdata.nfreqs))
        p = self.parameters.reshape((-1, gsdata.nparams))

        model = self.model.at(x=gsdata.freq_array.to_value("MHz"))

        spectra = np.zeros_like(d)
        for i, (dd, pp) in enumerate(zip(d, p)):
            spectra[i] = dd + model(parameters=pp)

        spectra.shape = gsdata.data.shape
        return spectra

    @classmethod
    def from_gsdata(cls, model: mdl.Model, gsdata: GSData) -> 'GSDataModel':
        """Creates a GSDataModel from a GSData object."""
        if gsdata.data_unit == "model_residuals":
            raise ValueError("Cannot compute model on data that is already residuals!")

        d = gsdata.data.reshape((-1, gsdata.nfreqs))
        w = gsdata.nsamples.reshape((-1, gsdata.nfreqs))

        xmodel = model.at(x=gsdata.freq_array.to_value("MHz"))

        params = np.zeros((gsdata.nloads* gsdata.npols* gsdata.ntimes, model.n_terms))

        for i, (dd, ww) in enumerate(zip(d, w)):
            params[i] = xmodel.fit(ydata=dd, weights=ww).model_parameters

        return cls(model=model, parameters=params)

    def update(self, **kw) -> 'GSDataModel':
        return clone(self, **kw)


GSDATA_PROCESSORS = {}
def register_gsprocess(fnc: Callable[..., GSData]) -> Callable[..., GSData]:
    GSDATA_PROCESSORS[fnc.__name__] = fnc

    @functools.wraps(fnc)
    def inner(data: GSData, **kw) -> GSData:
        newdata = fnc(data, **kw)
        newdata.update(history={
            'timestamp': datetime.datetime.now(),
            'message': '',
            'function': fnc.__name__,
            'parameters': kw,
            'versions': {
                'edges-analysis': __version__,
                'edges-cal': edges_cal.__version__,
                'read_acq': read_acq.__version__,
                'edges_io': edges_io.__version__,
                'numpy': np.__version__,
                'astropy': astropy.__version__,
            }
        })
        return newdata

    inner.gatherer = False

    return inner

GSDATA_PROCESSORS = {}
def register_gsgather(fnc: Callable[..., GSData]) -> Callable[..., GSData]:
    GSDATA_PROCESSORS[fnc.__name__] = fnc

    @functools.wraps(fnc)
    def inner(*data: GSData, **kw) -> GSData:
        newdata = fnc(*data, **kw)
        newdata.update(history={
            'timestamp': datetime.datetime.now(),
            'message': f"Combined from {', '.join(d.get_initial_yearday() for d in data)}",
            'function': fnc.__name__,
            'parameters': kw,
            'versions': {
                'edges-analysis': __version__,
                'edges-cal': edges_cal.__version__,
                'read_acq': read_acq.__version__,
                'edges_io': edges_io.__version__,
                'numpy': np.__version__,
                'astropy': astropy.__version__,
            }
        })
        return newdata

    inner.gatherer = True

    return inner




