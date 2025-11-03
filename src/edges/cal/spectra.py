"""Module dealing with calibration spectra and thermistor measurements."""

import contextlib
import inspect
import logging
from pathlib import Path

import attrs
import numpy as np
from astropy import units as un
from astropy.time import Time
from numpy.typing import NDArray
from pygsdata import GSData
from pygsdata.attrs import npfield
from pygsdata.select import select_freqs, select_times

from .. import __version__
from .. import types as tp
from ..averaging import freqbin, lstbin
from ..io import (
    LoadDefEDGES2,
    LoadDefEDGES3,
    get_mean_temperature,
    read_temperature_log,
)
from ..io.serialization import hickleable
from ..io.spectra import read_spectra
from ..tools import stable_hash
from .dicke import dicke_calibration
from .thermistor import IgnoreTimesType, ThermistorReadings, ignore_ntimes

logger = logging.getLogger(__name__)


def flag_data_outside_temperature_range(
    temperature_range: tp.TemperatureType
    | tuple[tp.TemperatureType, tp.TemperatureType],
    spec_times: Time,
    thermistor: ThermistorReadings,
) -> NDArray[np.bool]:
    """Get a mask that flags data outside a temperature range."""
    thermistor_temp = thermistor.get_physical_temperature()
    thermistor_times = thermistor.data["times"]

    # Cut on temperature.
    if isinstance(temperature_range, un.Quantity):
        median = np.median(thermistor_temp)
        temp_range = (
            median - temperature_range / 2,
            median + temperature_range / 2,
        )
    else:
        temp_range = temperature_range

    temp_mask = np.zeros(len(spec_times), dtype=bool)
    for i, c in enumerate(thermistor.get_thermistor_indices(spec_times)):
        if c is None or np.isnan(c):
            temp_mask[i] = False
        else:
            temp_mask[i] = (thermistor_temp[c] >= temp_range[0]) & (
                thermistor_temp[c] < temp_range[1]
            )

    if not np.any(temp_mask):
        raise RuntimeError(
            "The temperature range has masked all spectra!"
            f"Temperature Range Desired: {temp_range}.\n"
            "Temperature Range of Data: "
            f"{(thermistor_temp.min(), thermistor_temp.max())}\n"
            f"Time Range of Spectra: "
            f"{(spec_times[0], spec_times[-1])}\n"
            f"Time Range of Thermistor: "
            f"{(thermistor_times[0], thermistor_times[-1])}"
        )

    return temp_mask


def get_ave_and_var_spec(
    data: GSData,
    thermistor: ThermistorReadings | None = None,
    frequency_smoothing: str = "gauss",
    f_low: tp.FreqType = 0 * un.MHz,
    f_high: tp.FreqType = np.inf * un.MHz,
    ignore_times: IgnoreTimesType = 0,
    freq_bin_size: int = 1,
    temperature_range: tp.TemperatureType
    | tuple[tp.TemperatureType, tp.TemperatureType]
    | None = None,
    time_coordinate_swpos: int | tuple[int, int] = 0,
) -> tuple[GSData, GSData]:
    """Get the mean and variance of the spectra.

    Parameters
    ----------
    frequency_smoothing
        How to average frequency bins together. Default is to merely bin them
        directly. Other options are 'gauss' to do Gaussian filtering (this is the
        same as Alan's C pipeline).
    """
    data = select_freqs(data, freq_range=(f_low, f_high))

    spec_timestamps = data.times[:, time_coordinate_swpos]  # jd

    with contextlib.suppress(Exception):
        _base_time, time_coordinate_swpos = time_coordinate_swpos

    ignore_ninteg = ignore_ntimes(spec_timestamps, ignore_times)

    data = select_times(
        data,
        indx=slice(ignore_ninteg, None),
        load=data.loads[time_coordinate_swpos],
    )
    spec_timestamps = spec_timestamps[ignore_ninteg:]

    if temperature_range is not None:
        temp_mask = flag_data_outside_temperature_range(
            temperature_range, spec_timestamps, thermistor
        )
        data = select_times(data, idx=temp_mask)

    q = dicke_calibration(data)
    if freq_bin_size > 1:
        if frequency_smoothing == "bin":
            q = freqbin.freq_bin(q, bins=freq_bin_size)
        elif frequency_smoothing == "gauss":
            # We only really allow Gaussian smoothing so that we can match Alan's
            # pipeline. In that case, the frequencies actually kept start from the
            # 0th index, instead of taking the centre of each new bin. Thus we
            # set decimate_at = 0.
            q = freqbin.gauss_smooth(q, size=freq_bin_size, decimate_at=0)
        else:
            raise ValueError("frequency_smoothing must be one of ('bin', 'gauss').")

    mean = lstbin.average_over_times(q)
    variance = np.nanvar(q.data, axis=2)[:, :, None]

    variance = mean.update(data=variance)
    return mean, variance


@hickleable
@attrs.define(kw_only=True, frozen=True)
class LoadSpectrum:
    """A class representing a measured spectrum from some Load averaged over time.

    Parameters
    ----------
    q
        The measured power-ratios of the three-position switch averaged over time.
    variance
        The variance of *a single* time-integration as a function of frequency.
    temp_ave
        The average measured physical temperature of the load while taking spectra.
    """

    q: GSData = attrs.field()
    variance: GSData | None = attrs.field(default=None)
    temp_ave: tp.TemperatureType = npfield(
        possible_ndims=(
            0,
            1,
        ),
        unit=un.K,
    )

    @q.validator
    def _q_vld(self, att, val):
        if not isinstance(val, GSData):
            raise TypeError("q must be a GSData object")

        if val.ntimes != 1:
            raise ValueError("q must have a single time (averaged over times)")

    @variance.validator
    def _var_vld(self, att, val):
        if val is None:
            return

        if val.data.shape != self.q.data.shape:
            raise ValueError("variance must be the same shape as q")

    @property
    def freqs(self) -> tp.FreqType:
        """The frequencies at which the spectrum is measured."""
        return self.q.freqs

    @classmethod
    def from_loaddef(
        cls,
        loaddef: LoadDefEDGES2 | LoadDefEDGES3,
        # load_name: str,
        f_low=40.0 * un.MHz,
        f_high=np.inf * un.MHz,
        f_range_keep: tuple[tp.FreqType, tp.FreqType] | None = None,
        freq_bin_size=1,
        ignore_times: IgnoreTimesType = 5.0 * un.percent,
        temperature_range: tp.TemperatureType
        | tuple[tp.TemperatureType, tp.TemperatureType]
        | None = None,
        frequency_smoothing: str = "bin",
        time_coordinate_swpos: int = 0,
        invalidate_cache: bool = False,
        cache_dir: Path | None = None,
        temperature: tp.TemperatureType | None = None,
        allow_closest_time: bool = True,
    ):
        """Instantiate the class from a given load name and directory.

        Parameters
        ----------
        direc : str or Path
            The top-level calibration observation directory.
        run_num : int
            The run number to use for the spectra.
        filetype : str
            The filetype to look for (acq or h5).
        freqeuncy_smoothing
            How to average frequency bins together. Default is to merely bin them
            directly. Other options are 'gauss' to do Gaussian filtering (this is the
            same as Alan's C pipeline).
        ignore_times_percent
            The fraction of readings to ignore at the start of the observation. If
            greater than 100, will be interpreted as being a number of seconds to
            ignore.
        kwargs :
            All other arguments to :class:`LoadSpectrum`.

        Returns
        -------
        :class:`LoadSpectrum`.
        """
        if cache_dir is not None:
            cache_dir = Path(cache_dir)

            sig = inspect.signature(cls.from_loaddef)
            lc = locals()
            defining_dict = {
                p: lc[p] for p in sig.parameters if p not in ["cls", "caldef"]
            }
            defining_dict["files"] = loaddef

            hsh = stable_hash((
                *tuple(defining_dict.values()),
                __version__.split(".")[0],
            ))

            fname = cache_dir / f"{loaddef.name}_{hsh}.gsh5"

        if not invalidate_cache and cache_dir is not None and fname.exists():
            logger.info(
                f"Reading in previously-created integrated {loaddef.name} spectra..."
            )
            return cls.from_file(fname)

        data: GSData = read_spectra(loaddef.spectra)

        if hasattr(loaddef, "thermistor"):
            thermistor = ThermistorReadings.from_csv(
                loaddef.thermistor, ignore_times=ignore_times
            )
        else:
            thermistor = None
            temperature_range = None

        meanq, varq = get_ave_and_var_spec(
            data=data,
            f_low=f_low,
            f_high=f_high,
            ignore_times=ignore_times,
            freq_bin_size=freq_bin_size,
            temperature_range=temperature_range,
            thermistor=thermistor,
            frequency_smoothing=frequency_smoothing,
            time_coordinate_swpos=time_coordinate_swpos,
        )

        if temperature is None:
            if thermistor is not None:
                temperature = np.nanmean(thermistor.get_physical_temperature())
            elif loaddef.templog is None:
                raise ValueError(
                    f"templog doesn't exist, and no source temperature passed for"
                    f"{loaddef.name}"
                )
            else:
                start = data.times.min()
                end = data.times.max()
                table = read_temperature_log(loaddef.templog)

                if (
                    not np.any((table["time"] >= start) & (table["time"] <= end))
                    and allow_closest_time
                ):
                    start = table["time"][np.argmin(np.abs(table["time"] - start))]
                    end = table["time"][np.argmin(np.abs(table["time"] - end))]

                temperature = get_mean_temperature(
                    table, start_time=start, end_time=end, load=loaddef.name
                ).to("K")

        out = cls(
            q=meanq,
            variance=varq,
            temp_ave=temperature,
        )

        if f_range_keep is not None:
            out = out.between_freqs(*f_range_keep)

        if cache_dir is not None:
            if not cache_dir.exists():
                cache_dir.mkdir(parents=True)
            out.write(fname)

        return out

    def between_freqs(self, f_low: tp.FreqType, f_high: tp.FreqType = np.inf * un.MHz):
        """Return a new LoadSpectrum that is masked between new frequencies."""
        return attrs.evolve(
            self,
            q=select_freqs(self.q, freq_range=(f_low, f_high)),
            variance=select_freqs(self.variance, freq_range=(f_low, f_high))
            if self.variance is not None
            else None,
        )

    @property
    def averaged_q(self) -> np.ndarray:
        """Ratio of powers averaged over time.

        Notes
        -----
        The formula is

        .. math:: Q = (P_source - P_load)/(P_noise - P_load)
        """
        return self.q.data[0, 0, 0]

    @property
    def variance_q(self) -> np.ndarray:
        """Variance of Q across time (see averaged_q)."""
        return self.variance.data[0, 0, 0]
