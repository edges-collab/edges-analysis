from os import makedirs, listdir
from os.path import exists, dirname
from typing import Tuple, Optional, Sequence, List
from functools import lru_cache

import h5py
import numpy as np
from edges_cal import (
    FrequencyRange,
    receiver_calibration_func as rcf,
    modelling as mdl,
    xrfi as rfi,
    CalibrationObservation,
)
from edges_io.io import Spectrum
import attr
from pathlib import Path
from cached_property import cached_property

# import src.edges_analysis
import src.edges_analysis.analysis.filters
from . import io, s11 as s11m, loss, beams, tools, filters, coordinates
from ..config import config
from .. import const
from datetime import datetime


@attr.s
class _Level(io.HDF5Object):
    _structure = {"spectra": None, "ancillary": None, "meta": None}
    _prev_level = None

    @classmethod
    def from_previous_level(cls, prev_level, filename=None, **kwargs):
        if not cls._prev_level:
            raise AttributeError(
                f"from_previous_level is not defined for {cls.__name__}"
            )

        if not isinstance(prev_level, cls._prev_level):
            if hasattr(prev_level, "__len__"):
                prev_level = [
                    p if isinstance(p, cls._prev_level) else cls._prev_level(p)
                    for p in prev_level
                ]
            else:
                prev_level = cls._prev_level(prev_level)

        data = cls._from_prev_level(prev_level, **kwargs)
        out = cls.from_data(data, filename=filename)

        if filename:
            out.write()
        return out

    @classmethod
    def _from_prev_level(
        cls, prev_level: [_prev_level, Sequence[_prev_level]], **kwargs
    ):
        pass

    @property
    def meta(self):
        return self["meta"]

    @property
    def spectrum(self):
        return self["spectra"]["antenna_temp"]

    @property
    def raw_frequencies(self):
        return self["spectra"]["frequencies"]

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
        try:
            return self.spectra["weights"]
        except KeyError:
            return np.ones_like(self.raw_frequencies)


@attr.s
class Level1(_Level):
    """Object representing the level-1 stage of processing.

    I think this should actually just be an edges_io.Spectrum.
    # TODO: add actual description.
    """

    @property
    def raw_time_data(self):
        return self.ancillary["times"]

    @cached_property
    def datetimes(self):
        return [datetime(*d) for d in self.raw_time_data]

    @cached_property
    def datetimes_np(self):
        return np.array(self.datetimes, dtype="datetime64[s]")


@attr.s
class Level2(_Level):
    @classmethod
    def _from_prev_level(cls, level1, **kwargs):
        data, ancillary, meta = level1_to_level2(level1, **kwargs)
        return {"spectra": data, "ancillary": ancillary, "meta": meta}


def level1_to_level2(
    level1: [Level1, Path, str],
    weather_file: [Path, str],
    thermlog_file: [Path, str],
    band: str,
    year: int,
    day: int,
    hour: int = 0,
):
    """
    Convert a Level1 file or object to Level2.

    The process of converting one to the other does not modify the spectrum in any
    way -- there is no averaging/integration done in this step. Rather, auxiliary
    information is *added* to the object, containing weather and thermal characteristics
    at the site during observation, interpolated to the observing times.

    Parameters
    ----------
    level1 : :class:`Level1` instance, Path or str
        The level1 object to update to level2.
    weather_file : Path or str
        The file containing weather information relevant to the dates contained in the
        level1 object.
    thermlog_file : Path or str
        The file containing thermistor log information relevant to the dates
        contained in the level1 object.
    band : str
        A string specifying which band this observation is for (either `low2, `low3`,
        `mid`, or `high`).
    year : int
        The year of observation.
    day : int
        The day (of the year) of observation.
    hour : int
        The hour of the day at which observations began.

    Returns
    -------
    data : dict
        Dict containing "frequencies" and "antenna_temp", which are just the same
        as level1.
    auxiliary : dict
        Containing
        * "ambient_temp": Ambient temperature as a function of time
        * "ambient_humidity": Ambient humidity as a function of time
        * "receiver1_temp": Receiver1 temperature as a function of time
        * "recevier2_temp": Receiver2 temperature as a function of time
        * "lst": LST for each observation in the spectrum.
        * "gha": GHA for each observation in the spectrum.
        * "sun_moon_azel": Coordinates of the sun and moon as function of time.
    meta : dict
        Containing
        * "year": year of observation
        * "day": day of observation (within year)
        * "hour": hour of observation (within day)
        * "band": the band of observation (a string)
        * "thermlog_file": path to the thermlog information used
        * "weather_file": path to the weather information used
    """
    weather_file = Path(weather_file)
    if not weather_file.is_absolute():
        weather_file = (Path(config["MRO_folder"]) / weather_file).absolute()

    thermlog_file = Path(thermlog_file)
    if not thermlog_file.is_absolute():
        thermlog_file = (Path(config["MRO_folder"]) / thermlog_file).absolute()

    if not isinstance(level1, Level1):
        level1 = Level1(level1)

    # LST
    lst = coordinates.utc2lst(level1.datetimes, const.edges_lon_deg)
    gha = lst - const.galactic_centre_lst
    gha[gha < -12.0] += 24

    # Sun/Moon coordinates
    sun, moon = coordinates.sun_moon_azel(
        const.edges_lat_deg, const.edges_lon_deg, level1.datetimes
    )

    # TODO: Check if band/year/day can be gotten directly from level1
    aux1, aux2 = io.auxiliary_data(weather_file, thermlog_file, band, year, day)
    seconds = (level1.datetimes_np - level1.datetimes_np[0]).astype(float)
    amb_temp_interp = np.interp(seconds, aux1[:, 0], aux1[:, 1]) - 273.15
    amb_hum_interp = np.interp(seconds, aux1[:, 0], aux1[:, 2])
    rec1_temp_interp = np.interp(seconds, aux1[:, 0], aux1[:, 3]) - 273.15

    if len(aux2) == 1:
        rec2_temp_interp = 25 * np.ones(len(seconds))
    else:
        rec2_temp_interp = np.interp(seconds, aux2[:, 0], aux2[:, 1])

    # Meta
    meta = {
        "band": band,
        "year": year,
        "day": day,
        "hour": hour,
        "thermlog_file": str(thermlog_file),
        "weather_file": str(weather_file),
    }

    ancillary = {
        "ambient_temp": amb_temp_interp,
        "ambient_humidity": amb_hum_interp,
        "receiver1_temp": rec1_temp_interp,
        "receiver2_temp": rec2_temp_interp,
        "lst": lst,
        "gha": gha,
        "sun_azel": sun,
        "moon_azel": moon,
    }

    # TODO: consider filtering on auxilary HERE rather than in Level4.

    data = {"frequency": level1.freq.freq, "antenna_temp": level1.spectrum}

    return data, ancillary, meta


class Level3(_Level):
    def _from_prev_level(cls, level2: [Level2], **kwargs):
        data, ancillary, meta = level2_to_level3(level2, **kwargs)
        return {"spectra": data, "ancillary": ancillary, "meta": meta}

    @lru_cache()
    def frequency_average_spectrum(self, indx=None, resolution=0.0488):
        """
        Perform a frequency-average over the spectrum.

        Parameters
        ----------
        indx : int, optional
            The (time) index at which to compute the frequency-averaged spectrum.
            If not given, returns a 2D array, with time on the first axis.
        resolution : float, optional
            The frequency resolution of the output.

        Returns
        -------
        f : array-like
            The mean frequency in each output bin
        t : array-like
            The weighted-average of the spectrum in each bin
        w : array-like
            The total weight in each bin
        std : array-like
            The standard deviation about the mean in each bin.
        """
        if indx is None:
            out = [
                self.frequency_average_spectrum(i, resolution)
                for i in range(len(self.spectrum))
            ]
            return tuple(np.array(x) for x in out)
        else:
            # Fitting foreground model to binned version of spectra
            return tools.average_in_frequency(
                self.spectrum[indx],
                freq=self.raw_frequencies,
                weights=self.weights[indx],
                resolution=resolution,
            )

    @lru_cache()
    def model(self, indx, model="LINLOG", n_terms=5, resolution=0.0488):
        """
        Determine a callable model of the spectrum at a given time, optionally
        computed over averaged original data.

        Parameters
        ----------
        indx : int
            The (time) index to compute the model for.
        model : str, optional
            The kind of model to fit.
        n_terms : int, optional
            The number of terms to use in the fit.

        Returns
        -------
        callable :
            Function of frequency (in units of self.raw_frequency) that will return
            the model.
        """
        f, s, w = self.frequency_average_spectrum(indx, resolution)[:3]

        freq = FrequencyRange(f)
        params = mdl.fit_polynomial_fourier(
            model, freq.freq_recentred, s, n_terms, Weights=w
        )[0]

        return lambda nu: mdl.model_evaluate(model, params, freq.normalize(nu))

    @lru_cache()
    def get_model_rms(
        self,
        indx=None,
        model="LINLOG",
        n_terms=5,
        resolution=0.0488,
        freq_range=(-np.inf, np.inf),
    ):
        if indx is None:
            return np.array(
                [
                    self.get_model_rms(i, model, n_terms, resolution, freq_range)
                    for i in range(len(self.spectrum))
                ]
            )
        else:
            mask = (
                self.raw_frequencies
                >= freq_range[0] & self.raw_frequencies
                <= freq_range[1]
            )

            model = self.model(indx, model, n_terms, resolution)(
                self.raw_frequencies[mask]
            )
            resid = self.spectrum[indx, mask] - model
            return np.sqrt(
                np.sum((resid[self.weights > 0]) ** 2) / np.sum(self.weights > 0)
            )


def level2_to_level3(
    level2: [str, Path, Level2],
    calobs: [str, CalibrationObservation],
    s11_path="antenna_s11_2018_147_17_04_33.txt",
    antenna_s11_n_terms=15,
    antenna_correction=True,
    balun_correction=True,
    ground_correction=True,
    beam_file=None,
    f_low: float = 50,
    f_high: float = 150,
    n_fg=7,
) -> Tuple[dict, dict, dict]:
    """
    Convert a Level2 file or object to Level3.

    This algorithm performs the following operations on the data:

    * Restricts frequency range
    * Applies a lab-based calibration solution
    * Corrects for ground/balun/antenna losses
    * Corrects for beam factor
    * Flags RFI using an explicit list of channels
    * Flags RFI using a moving window polynomial filter

    Parameters
    ----------
    level2 : :class:`Level2` instance or path
        The level2 instance to process.
    calobs : :class:`CalibrationObservation` instance or path
        The lab-based calibration observation to use to calibrate the data.
    s11_path : path, optional
        Path to the S11 measurement of the antenna.
    antenna_s11_n_terms : int, optional
        Number of terms used in fitting the S11 model.
    antenna_correction : bool, optional
        Whether to perform the antenna correction
    balun_correction : bool, optional
        Whether to perform the balun correction
    ground_correction : bool, optional
        Whether to perform the ground correction
    beam_file : path, optional
        Filename (not absolute) of a beam model to use for correcting for the beam
        factor. Not used if not provided.
    f_low : float
        Minimum frequency to use.
    f_high : float
        Maximum frequency to use.
    n_fg : int, optional
        Number of foreground terms to use in obtaining the model and residual.

    Returns
    -------
    data : dict
        Same keys as for :func:`level1_to_level2` but adding `weights`.
    ancillary : dict
        The same ancillary data as contained in `level2`.
    meta : dict
        Contains all input parameters, as well as level2 meta.
    """
    if not isinstance(level2, Level2):
        level2 = Level2(level2)

    meta = {
        "s11_path": Path(s11_path).absolute(),
        "antenna_s11_n_terms": antenna_s11_n_terms,
        "antenna_correction": antenna_correction,
        "balun_correction": balun_correction,
        "ground_correction": ground_correction,
        "beam_file": beam_file,
        "f_low": f_low,
        "f_high": f_high,
        "n_poly_xrfi": n_fg,
    }

    meta = {**meta, **level2["meta"]}

    if np.all(level2["spectra"] == 0):
        raise Exception("The level2 file given has no non-zero spectra!")

    # Cut the frequency range
    freq = FrequencyRange(level2.raw_frequencies, f_low=f_low, f_high=f_high)
    # fin = fin_X[(fin_X >= f_low) & (fin_X <= f_high)]
    spectra = level2.spectrum[:, freq.mask]
    weights = level2.weights[:, freq.mask]

    if not isinstance(calobs, CalibrationObservation):
        calobs = CalibrationObservation.from_file(calobs)

    # Antenna S11
    s11_ant = s11m.antenna_s11_remove_delay(
        s11_path, freq.freq, delay_0=0.17, n_fit=antenna_s11_n_terms
    )

    # Calibrated antenna temperature with losses and beam chromaticity
    calibrated_temp = rcf.calibrated_antenna_temperature(
        spectra,
        s11_ant,
        calobs.lna.s11_model(freq.freq),
        calobs.C1(freq.freq),
        calobs.C2(freq.freq),
        calobs.Tunc(freq.freq),
        calobs.Tcos(freq.freq),
        calobs.Tsin(freq.freq),
    )

    # Antenna Loss (interface between panels and balun)
    G = np.ones_like(freq.freq)
    if antenna_correction:
        G *= loss.antenna_loss(level2.meta["band"], freq.freq)

    # Balun+Connector Loss
    if balun_correction:
        Gb, Gc = loss.balun_and_connector_loss(level2.meta["band"], freq.freq, s11_ant)
        G *= Gb * Gc

    # Ground Loss
    if ground_correction:
        G *= loss.ground_loss(level2.meta["band"], freq.freq)

    # Remove loss
    # TODO: it is *not* clear which temperature to use here...
    ambient_temp = 273.15 + level2.ancillary["ambient_temp"]
    calibrated_temp = (calibrated_temp - ambient_temp * (1 - G)) / G

    # Beam factor
    if beam_file:
        if not Path(beam_file).exists():
            beam_file = f"{config['edges_folder']}{level2.meta['band']}/calibration/beam_factors/table/{beam_file}"

        f_table, lst_table, bf_table = beams.beam_factor_table_read(beam_file)
        bf = beams.beam_factor_table_evaluate(
            f_table, lst_table, bf_table, level2["ancillary"]["lst"]
        )[:, ((f_table >= f_low) & (f_table <= f_high))]

        # Remove beam chromaticity
        calibrated_temp /= bf

    # RFI cleaning
    flags = rfi.excision_raw_frequency(
        freq.freq, rfi_file=dirname(__file__) + "data/known_rfi_channels.yaml"
    )

    calibrated_temp[flags] = 0
    weights[flags] = 0

    # RFI cleaning
    for i, (temp_cal, wi) in enumerate(calibrated_temp, weights):
        flags = rfi.xrfi_poly(
            temp_cal,
            wi,
            f_ratio=freq.max / freq.min,
            n_signal=n_fg,
            n_resid=5,
            n_abs_resid_threshold=3.5,
        )
        # TODO: this should probably be nan, not zero
        temp_cal[flags] = 0
        wi[flags] = 0

    ancillary = level2.ancillary

    data = {
        "antenna_temp": calibrated_temp,
        "frequency": freq.freq,
        "weights": weights,
    }

    return data, ancillary, meta


def level3_to_level4(
    level3: Sequence[Level3, Path, str],
    gha_edges: Sequence[Tuple[float, float]],
    sun_el_max,
    moon_el_max,
    ambient_humidity_max=40,
    min_receiver_temp=0,
    max_receiver_temp=100,
    n_sigma_rms=3,
    rfi_window_size=3,  # MHz
    n_poly_rfi=2,
    n_bootstrap_rfi=20,
    n_sigma_rfi=3.5,
):
    """
    Combine and Convert level3 objects to a single level4 object.

    Given a sequence of :class:`Level3` objects, this function combines them together,
    filters out some times (see below), and integrates over time into given bins in
    LST/GHA.

    Times are filtered based on sun/moon position, humidity, receiver temperature,
    the total RMS of two halves of the spectrum after subtraction of a 5-term polynomial
    (fitted to frequency-binned data), the total RMS of the full spectrum after
    subtractino of a 3-term polynomial (fitted to frequency-binned data), and the total
    summed temperature over the spectrum (for each half, and the whole).

    Times are then averaged within provided bins of GHA.

    Further frequency filtering is performed based on a moving window polynomial filter
    of the mean data within each GHA bin.

    Parameters
    ----------
    level3 : list of :class:`Level3` instances or paths
        A bunch of level3 objects to combine and integrate into the level4 data.
    gha_edges : list of tuple
        A list of 2-tuples, each containing a min and max GHA.
    sun_el_max : float
        The maximum elevation of the sun before the time is filtered.
    moon_el_max : float
        The maximum elevation of the moon before the time is filtered.
    ambient_humidity_max : float, optional
        The maximum humidity allowed before the time is filtered.
    min_receiver_temp : float, optional
        The minimum temperature of the receiver before the observation is filtered.
    max_receiver_temp : float, optional
        The maximum temperature of the receiver before the observation is filtered.
    n_sigma_rms : int, optional
        The number of sigma at which to filter a time due to its RMS after subtracting
        a smooth model.
    rfi_window_size : float, optional
        The size of the moving polynomial window used to locate frequency-based RFI.
        In MHz.
    n_poly_rfi : int, optional
        The order of the polynomial used to fit the sliding window for RFI.
    n_bootstrap_rfi : int, optional
        The number of bootstrap samples to take to initialize the RMS for the first
        sliding window.
    n_sigma_rfi : float, optional
        Number of sigma to use for clipping RFI.

    Returns
    -------
    data : dict
        The same keys as in :func:`level2_to_level3`. However, the axes for the weights
        and spectra in this case are ``(N_FILES, N_GHA, N_FREQ)``, where ``N_FILES``
        is the original number of Level3 objects input.
    ancillary : dict
        Containing:
        * "n_total_times_per_file": The number of observations per input file.
        * "used_times": A 2D boolean array specifying which times (second axis)
          are used for which files (first axis). The array is padded out to the maximum
          number of times in any of the files, and the `n_total_times_per_file` should
          be used to index the relevant arrays.
        * 'years': The year for each file.
        * 'days': The day for each file
        * 'hours': The starting hour for each file.
    meta : dict
        Containing all input parameters, and the number of files used.
    """
    meta = {
        "n_files": len(level3),
        "gha_edges": gha_edges,
        "sun_el_max": sun_el_max,
        "moon_el_max": moon_el_max,
        "ambient_humidity_max": ambient_humidity_max,
        "min_receiver_temp": min_receiver_temp,
        "max_receiver_temp": max_receiver_temp,
        "n_sigma_rms": n_sigma_rms,
        "rfi_window_size": rfi_window_size,
        "n_poly_rfi": n_poly_rfi,
        "n_bootstrap_rfi": n_bootstrap_rfi,
        "n_sigma_rfi": n_sigma_rfi,
    }

    # Sort the inputs in ascending date.
    level3 = sorted(
        level3, key=lambda x: f"{x.meta['year']-x.meta['day']-x.meta['hour']}"
    )

    years = [x.meta["year"] for x in level3]
    days = [x.meta["day"] for x in level3]
    hours = [x.meta["hour"] for x in level3]

    weights = np.zeros((len(level3), len(gha_edges), level3[0].freq.n))
    spectra = np.zeros((len(level3), len(gha_edges), level3[0].freq.n))

    # Create a master array of indices of good-quality spectra (over the time axis)
    # used in the final averages
    ntimes = np.array([len(l3.spectrum) for l3 in level3])
    master_index = np.zeros((len(level3), len(gha_edges), ntimes.max()), dtype=bool)

    for i, l3 in enumerate(level3):
        # Filter based on aux data.
        good = filters.time_filter_auxiliary(
            gha=l3.ancillary["gha"],
            sun_el=l3.ancillary["sun_azel"][1],
            moon_el=l3.ancillary["sun_azel"][1],
            humidity=l3.ancillary["ambient_humidity"],
            receiver_temp=l3.ancillary["receiver1_temp"],
            sun_el_max=sun_el_max,
            moon_el_max=moon_el_max,
            amb_hum_max=ambient_humidity_max,
            min_receiver_temp=min_receiver_temp,
            max_receiver_temp=max_receiver_temp,
        )

        # Get RMS
        rms_lower = l3.get_model_rms(freq_range=(-np.inf, l3.freq.center))
        rms_upper = l3.get_model_rms(freq_range=(l3.freq.center, np.inf))
        rms_3term = l3.get_model_rms(n_terms=3)

        # px = py[IX, :]
        # rx = ry[IX, :]
        # wx = wy[IX, :]

        #        rmsx = rmsy[IX, :]
        #         tpx = tpy[IX, :]
        #         mx = my[IX, :]
        #         daily_index2 = daily_index1[IX]
        # master_index[i, IX] = 1

        # Finding index of clean data
        gha = l3.ancillary["gha"]

        good &= filters.rms_filter(
            l3.meta["band"],
            gha,
            np.array([rms_lower, rms_upper, rms_3term]).T,
            n_sigma_rms,
        )

        # Applying total-power filter
        good &= filters.total_power_filter(
            gha,
            np.array(
                [
                    np.sum(
                        l3.spectrum[:, l3.raw_frequencies <= l3.freq.center], axis=1
                    ),
                    np.sum(
                        l3.spectrum[:, l3.raw_frequencies >= l3.freq.center], axis=1
                    ),
                    np.sum(l3.spectrum, axis=1),
                ]
            ),
        )

        # Selecting good data
        # p = px[index_good, :]
        # r = rx[index_good, :]
        # w = wx[index_good, :]
        # rms = rmsx[index_good, :]
        # m = mx[index_good, :]
        # daily_index3 = daily_index2[index_good]
        #
        # AT = np.vstack((gx, rmsx.T))
        # BT = np.vstack((GHA, rms.T))
        #
        # A = AT.T
        # B = BT.T
        #
        # if flag == 0:
        #
        #     grx_all = np.copy(A)
        #     gr_all = np.copy(B)
        #
        # if flag > 0:
        #     grx_all = np.vstack((grx_all, A))
        #     gr_all = np.vstack((gr_all, B))

        # Averaging data within each GHA bin
        for j, (gha_low, gha_high) in enumerate(gha_edges):

            mask = (gha >= gha_low) & (gha < gha_high)
            spec = l3.spectrum[mask]
            wght = l3.weights[mask]
            # r1 = r[(GHA >= GHA_LOW) & (GHA < GHA_HIGH), :]
            # w1 = w[(GHA >= GHA_LOW) & (GHA < GHA_HIGH), :]
            # m1 = m[(GHA >= GHA_LOW) & (GHA < GHA_HIGH), :]
            # daily_index4 = daily_index3[(GHA >= GHA_LOW) & (GHA < GHA_HIGH)]

            these_indx = good & mask

            if np.any(these_indx):
                spec_mean, wght_mean = tools.weighted_mean(spec, weights=wght, axis=0)

                # RFI cleaning of average spectra
                flags = rfi.xrfi_poly_filter(
                    spec_mean,
                    wght_mean,
                    window_width=int(rfi_window_size / l3.freq.df),
                    n_poly=n_poly_rfi,
                    n_bootstrap=n_bootstrap_rfi,
                    n_sigma=n_sigma_rfi,
                )
                spec_mean[flags] = 0
                wght_mean[flags] = 0

                # Store this iteration
                spectra[i, j] = spec_mean
                weights[i, j] = wght_mean
                master_index[i, j, these_indx] = True

    data = {
        "antenna_temp": spectra,
        "weights": weights,
        "frequency": level3[0].raw_frequencies,
    }

    ancillary = {
        "n_total_times_per_file": ntimes,
        "used_times": master_index,
        "years": years,
        "days": days,
        "hours": hours,
    }

    return data, ancillary, meta
