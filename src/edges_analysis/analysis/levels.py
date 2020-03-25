from os.path import dirname, basename
from typing import Tuple, Optional, Sequence, List, Union
from functools import lru_cache
import sys

import numpy as np
from edges_cal import (
    FrequencyRange,
    receiver_calibration_func as rcf,
    modelling as mdl,
    xrfi as rfi,
    Calibration,
)
from edges_io.auxiliary import auxiliary_data
import attr
from pathlib import Path
from cached_property import cached_property
from read_acq import decode_file

# import src.edges_analysis
from . import io, s11 as s11m, loss, beams, tools, filters, coordinates
from ..config import config
from .. import const
from datetime import datetime


@attr.s
class _Level(io.HDF5Object):
    """Base object for formal data reduction levels in edges-analysis.

    The structure is such that three groups will always be available:
    * spectra : containing frequency-based data. Arrays here include ``frequency``,
      ``antenna_temp`` and possibly ``weights``. Each of these will always have
      the frequency as their _last_ axis.
    * ancillary : containing non-defining data that is not frequency based (usually
      time based). May contain arrays such as ``time``, ``lst``, ``ambient_temp`` etc.
    * meta : parameters defining the data (eg. input parameters) or other scalars
      that describe the data.
    """

    default_root = config["paths"]["field_products"]

    _structure = {
        "spectra": {"frequency": None, "spectrum": None},
        "ancillary": {},
        "meta": None,
    }

    @classmethod
    def from_previous_level(cls, prev_level, filename=None, **kwargs):
        _prev_level = int(cls.__name__[-1]) - 1
        if _prev_level:
            _prev_level = getattr(sys.modules[__name__], f"Level{_prev_level}")
        else:
            raise AttributeError(
                f"from_previous_level is not defined for {cls.__name__}"
            )

        if not isinstance(prev_level, _prev_level):
            if hasattr(prev_level, "__len__"):
                prev_level = [
                    p if isinstance(p, _prev_level) else _prev_level(p)
                    for p in prev_level
                ]
            else:
                prev_level = _prev_level(prev_level)

        data = cls._from_prev_level(prev_level, **kwargs)
        out = cls.from_data(data, filename=filename)

        if filename:
            out.write()
        return out

    @classmethod
    def _from_prev_level(cls, prev_level, **kwargs):
        pass

    @property
    def meta(self):
        return self["meta"]

    @property
    def spectrum(self):
        return self["spectra"]["spectrum"]

    @property
    def raw_frequencies(self):
        return self["spectra"]["frequency"]

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
            return np.ones_like(self.spectrum)


@attr.s
class Level1(_Level):
    """Object representing the level-1 stage of processing.

    This is essentially just a wrapper object for raw output from the correlator.
    It has a :func:`from_acq` classmethod for defining the object from an ACQ file,
    if HDF5 is not directly available.

    Attributes
    ----------
    spectra : dict
        Containing
        * ``frequency``: the raw frequencies of the spectrum.
        * ``antenna_temp``: the nominal un-calibrated temperature measured by the antenna.
        * ``switch_powers``: (3, NFREQ) array containing the 3-switch position power
          measurements from the antenna.
        * ``Q``: (Q = (p0 - p1)/(p0 - p2))
    ancillary : dict
        Containing:
        * ``times``: string representations of the time of each switch-0 reading.
        * ``adcmin``: Minimum analog-to-digital converter value (size NTIMES)
        * ``adcmax``: Maximum analog-to-digital converter value (size NTIMES)
    meta : dict
        Containing:
        * ``year``: integer year that the data was taken (first measurement in file)
        * ``day``: integer day that the data was taken (first measurement in file)
        * ``hour``: integer hour that the data was taken (first measurement in file)
        * ``temperature``: not sure what temperature this is ?? # TODO
        * ``nblk``: also not sure what this is # TODO

    """

    @classmethod
    def from_acq(cls, filename, out_file=None):
        Q, p, ancillary = decode_file(filename, write_formats=[], meta=True)

        # TODO: weights from data drops?
        data = {
            "frequency": ancillary.frequencies,
            "spectrum": Q,
            "switch_powers": p,
        }

        meta = {
            "year": int(ancillary.data["time"].astype(str)[0].split(":")[0]),
            "day": int(ancillary.data["time"].astype(str)[0].split(":")[1]),
            "hour": int(ancillary.data["time"].astype(str)[0].split(":")[2]),
            **ancillary.meta,
        }
        ancillary = {
            "times": ancillary.data["time"],
            "adcmax": ancillary.data["adcmax"],
            "adcmin": ancillary.data["adcmin"],
        }

        if out_file is None:
            out_file = (
                Path(config["paths"]["field_products"])
                / "level1"
                / basename(filename).replace(".acq", ".h5")
            )

        return cls.from_data(
            {"spectra": data, "ancillary": ancillary, "meta": meta},
            filename=str(out_file),
        )

    @property
    def raw_time_data(self):
        """Raw string times at which the spectra were taken."""
        return self.ancillary["times"]

    @cached_property
    def datetimes(self):
        """List of python datetimes at which the spectra were taken."""
        return [
            datetime.strptime(d, "%Y:%j:%H:%M:%S")
            for d in self.raw_time_data.astype(str)
        ]

    @cached_property
    def datetimes_np(self):
        """Numpy datetime64 array of times when spectra were taken."""
        return np.array(self.datetimes, dtype="datetime64[s]")


@attr.s
class Level2(_Level):
    @classmethod
    def _from_prev_level(cls, level1, **kwargs):
        data, ancillary, meta = level1_to_level2(level1, **kwargs)
        return {"spectra": data, "ancillary": ancillary, "meta": meta}


def level1_to_level2(
    level1: [Level1, Path, str],
    band: str,
    weather_file: [None, Path, str] = None,
    thermlog_file: [None, Path, str] = None,
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
        * "receiver2_temp": Receiver2 temperature as a function of time
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
    if not isinstance(level1, Level1):
        level1 = Level1(level1)

    year = level1.meta["year"]
    day = level1.meta["day"]

    pth = Path(config["paths"]["raw_field_data"])
    if weather_file is not None:
        weather_file = Path(weather_file)
        if not (weather_file.exists() or weather_file.is_absolute()):
            weather_file = pth
    else:
        if (year, day) <= (2017, 329):
            weather_file = pth / "weather_upto_20171125.txt"
        else:
            weather_file = pth / "weather2.txt"

    if thermlog_file is not None:
        thermlog_file = Path(thermlog_file)
        if not (thermlog_file.exists() or thermlog_file.is_absolute()):
            thermlog_file = pth / thermlog_file
    else:
        thermlog_file = pth / f"thermlog_{band}.txt"

    # TODO: Check if band can be gotten directly from level1
    weather, thermlog = auxiliary_data(weather_file, thermlog_file, year, day)
    seconds = (level1.datetimes_np - level1.datetimes_np[0]).astype(float)

    time_based_anc = np.zeros(
        len(seconds),
        dtype=[(name, float) for name in weather.dtype.names]
        + [(name, float) for name in thermlog.dtype.names if name != "seconds"]
        + [
            ("lst", float),
            ("gha", float),
            ("sun_az", float),
            ("sun_el", float),
            ("moon_az", float),
            ("moon_el", float),
        ],
    )
    time_based_anc["seconds"] = seconds

    # Interpolate weather
    for name in weather.dtype.names:
        if name == "seconds":
            continue

        time_based_anc[name] = np.interp(seconds, weather["seconds"], weather[name])

        # Convert to celsius
        if name.endswith("_temp"):
            time_based_anc[name] -= 273.15

    for name in thermlog.dtype.names:
        if name == "seconds":
            continue

        time_based_anc[name] = np.interp(seconds, thermlog["seconds"], thermlog[name])

    # LST
    time_based_anc["lst"] = coordinates.utc2lst(level1.datetimes, const.edges_lon_deg)
    time_based_anc["gha"] = coordinates.lst2gha(time_based_anc["lst"])

    # Sun/Moon coordinates
    sun, moon = coordinates.sun_moon_azel(
        const.edges_lat_deg, const.edges_lon_deg, level1.datetimes
    )

    time_based_anc["sun_az"] = sun[:, 0]
    time_based_anc["sun_el"] = sun[:, 1]
    time_based_anc["moon_az"] = moon[:, 0]
    time_based_anc["moon_el"] = moon[:, 1]

    # Meta
    meta = {"band": band, **level1.meta}

    ancillary = {
        "time_ancillary": time_based_anc,
    }

    # TODO: consider filtering on auxiliary HERE rather than in Level4.

    data = {"frequency": level1.raw_frequencies, "spectrum": level1.spectrum}

    return data, ancillary, meta


class Level3(_Level):
    @classmethod
    def _from_prev_level(cls, prev_level, **kwargs):
        data, ancillary, meta = level2_to_level3(prev_level, **kwargs)
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
    calfile: [str, Calibration],
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

    if not isinstance(calfile, Calibration):
        calfile = Calibration(calfile)

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
        "wterms": calfile.wterms,
        "cterms": calfile.cterms,
        "calfile": calfile.path,
    }

    meta = {**meta, **level2["meta"]}

    if np.all(level2.spectrum == 0):
        raise Exception("The level2 file given has no non-zero spectra!")

    # Cut the frequency range
    freq = FrequencyRange(level2.raw_frequencies, f_low=f_low, f_high=f_high)
    Q = level2.spectrum[:, freq.mask]
    weights = level2.weights[:, freq.mask]

    # Antenna S11
    s11_ant = s11m.antenna_s11_remove_delay(
        s11_path, freq.freq, delay_0=0.17, n_fit=antenna_s11_n_terms
    )

    # Calibrated antenna temperature with losses and beam chromaticity
    calibrated_temp = calfile.calibrate_Q(freq.freq, Q, s11_ant)

    # Antenna Loss (interface between panels and balun)
    G = np.ones_like(freq.freq)
    if antenna_correction:
        G *= loss.antenna_loss(
            "default_antenna_loss.txt", freq.freq, band=level2.meta["band"]
        )

    # Balun+Connector Loss
    if balun_correction:
        Gb, Gc = loss.balun_and_connector_loss(level2.meta["band"], freq.freq, s11_ant)
        G *= Gb * Gc

    # Ground Loss
    if ground_correction:
        G *= loss.ground_loss(
            "default_ground_loss.txt", freq.freq, band=level2.meta["band"]
        )

    # Remove loss
    # TODO: it is *not* clear which temperature to use here...
    ambient_temp = 273.15 + level2.ancillary["time_ancillary"]["ambient_temp"]
    calibrated_temp = (calibrated_temp - np.outer(ambient_temp, (1 - G))) / G

    # Beam factor
    if beam_file:
        if not Path(beam_file).exists():
            beam_file = (
                Path(config["paths"]["beams"])
                / level2.meta["band"]
                / "beam_factors"
                / beam_file
            )

        beam_fac = beams.InterpolatedBeamFactor(beam_file)
        bf = beam_fac.evaluate(level2.ancillary["time_ancillary"]["lst"])
        bf = bf[:, (beam_fac["frequency"] >= f_low) & (beam_fac["frequency"] <= f_high)]

        # Remove beam chromaticity
        calibrated_temp /= bf

    # RFI cleaning
    flags = rfi.xrfi_explicit(
        freq.freq, rfi_file=Path(dirname(__file__)) / "data" / "known_rfi_channels.yaml"
    )

    weights[:, flags] = 0

    # RFI cleaning
    for i, (temp_cal, wi) in enumerate(zip(calibrated_temp, weights)):
        flags = rfi.xrfi_poly(
            temp_cal,
            wi,
            f_ratio=freq.max / freq.min,
            n_signal=n_fg,
            n_resid=5,
            n_abs_resid_threshold=3.5,
            max_iter=50,
        )
        wi[flags] = 0

    ancillary = level2.ancillary

    data = {
        "spectrum": calibrated_temp,
        "frequency": freq.freq,
        "weights": weights,
    }

    return data, ancillary, meta


class Level4(_Level):
    @classmethod
    def _from_prev_level(cls, level3: [Level3], **kwargs):
        data, ancillary, meta = level3_to_level4(level3, **kwargs)
        return {"spectra": data, "ancillary": ancillary, "meta": meta}


def level3_to_level4(
    level3: Sequence[Union[Level3, Path, str]],
    gha_edges: Sequence[Tuple[float, float]],
    sun_el_max,
    moon_el_max,
    ambient_humidity_max=40,
    min_receiver_temp=0,
    max_receiver_temp=100,
    n_sigma_rms=3,
    rfi_window_size=3,
    n_poly_rfi=2,
    n_bootstrap_rfi=20,
    n_sigma_rfi=3.5,
    rms_filter_file=None,
):
    """
    Combine and convert level3 objects to a single level4 object.

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
            moon_el=l3.ancillary["moon_azel"][1],
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

        # Finding index of clean data
        gha = l3.ancillary["gha"]

        if rms_filter_file:
            good &= filters.rms_filter(
                rms_filter_file,
                gha,
                np.array([rms_lower, rms_upper, rms_3term]).T,
                n_sigma_rms,
            )

        # Applying total-power filter
        # TODO: this filter should be removed/reworked -- it uses arbitrary numbers.
        # TODO: it at *least* should be done on the mean, not the sum.
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

        # Averaging data within each GHA bin
        for j, (gha_low, gha_high) in enumerate(gha_edges):

            mask = (gha >= gha_low) & (gha < gha_high)
            spec = l3.spectrum[mask]
            wght = l3.weights[mask]

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
        "spectrum": spectra,
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


class Level5(_Level):
    @classmethod
    def _from_prev_level(cls, level4: [Level4], **kwargs):
        data, ancillary, meta = level4_to_level5(level4, **kwargs)
        return {"spectra": data, "ancillary": ancillary, "meta": meta}


def level4_to_level5(
    level4,
    day_range: Optional[Tuple[int, int]] = None,
    ignore_days: Optional[Sequence[int]] = None,
    f_low: Optional[float] = None,
    f_high: Optional[float] = None,
    freq_resolution: Optional[float] = None,
):
    """
    Convert a level4 to a level5.

    This step integrates over days to form a spectrum as a function of GHA and
    frequency. It also applies an optional further frequency averaging.

    Parameters
    ----------
    level4 : :class:`Level4` instance
        The level4 object to convert.
    day_range : 2-tuple
        Min and max days to include (from a given year).
    ignore_days : sequence
        A sequence of days to ignore in the integration.
    f_low : float, optional
        A minimum frequency to use.
    f_high : float, optional
        A maximum frequency to use.
    freq_resolution : float, optional
        A frequency resolution to average down to.

    Returns
    -------
    data : dict
        Consisting of ``spectrum``, ``weights`` and ``frequency``. Both spectrum and
        weights are 2D, with the first axis the length of the GHA bins in the Level4
        object.
    ancillary : dict
        Consisting of
        * ``std_dev``: The standard deviation in each bin of (GHA, frequency)
        * ``years``: An array of years in which all observations in the dataset were
          taken.
    meta : dict
        Consisting of
        * ``day_range``: The user-specified allowed range of days going into the
          integration.
        * ``ignore_days``: The user-specified list of explicit days ignored in the
          integration.
    """
    meta = {"day_range": day_range, "ignore_days": ignore_days, **level4.meta}

    # Compute the residuals
    spec, wght = [], []
    days = level4.ancillary["days"]
    freq = FrequencyRange(level4.raw_frequencies, f_low=f_low, f_high=f_high)

    if day_range is None:
        day_range = (days.min(), days.max())

    for i, (low, high) in enumerate(level4.meta["gha_edges"]):
        gha_range = range(int(np.floor(low)), int(np.floor(high)))

        good_days = [filters.filter_explicit_gha(gha, *day_range) for gha in gha_range]
        for j, day in enumerate(days):
            if all(day in g for g in good_days) and day not in ignore_days:
                spec.append(level4.spectrum[j, i, freq.mask])
                wght.append(level4.weights[j, i, freq.mask])

    spec, wght = tools.weighted_mean(spec, wght, axis=0)

    out_spec, out_wght, out_std = [], [], []
    if freq_resolution:
        f, p, w, s = tools.average_in_frequency(
            spec, freq.freq, weights=wght, resolution=freq_resolution
        )
        out_spec.append(p)
        out_wght.append(w)
        out_std.append(s)

    data = {
        "spectrum": out_spec,
        "weights": out_wght,
        "frequency": f,
    }

    ancillary = {"std_dev": out_std, "years": np.unique(level4.ancillary["years"])}

    return data, ancillary, meta


class Level6(_Level):
    @classmethod
    def _from_prev_level(cls, level5: [Level5], **kwargs):
        data, ancillary, meta = level5_to_level6(level5, **kwargs)
        return {"spectra": data, "ancillary": ancillary, "meta": meta}


def level5_to_level6(
    level5,
    f_low: Optional[float] = None,
    f_high: Optional[float] = None,
    ignore_freq_ranges: Optional[Sequence[Tuple[float, float]]] = None,
    freq_resolution: Optional[float] = None,
):
    """
    Convert a Level5 to a Level6 object.

    This step performs a final average over GHA to yield a simple spectrum of
    frequency. It also allows a final averaging in frequency after the GHA average.

    Parameters
    ----------
    level5 : :class:`Level5` instance
        The Level5 object to convert.
    f_low : float, optional
        The min frequency to keep in the final average.
    f_high : float, optional
        The max frequency to keep in the final average.
    ignore_freq_ranges : list of tuple, optional
        An optional list of 2-tuples specifying frequency ranges to omit in the final
        average (they are weighted to zero).
    freq_resolution : float, optional
        An optional frequency resolution down to which to average.

    Returns
    -------
    data : dict
        Consisting of ``spectrum``, ``weights`` and ``frequency``. In this step,
        all of these are 1D and the same shape.
    ancillary : dict
        Consisting of
        * ``std_dev``: The standard deviation of the spectrum in each bin.
    meta : dict
        Consisting of all level 5 meta info, plus
        * ``ignore_freq_ranges``: the list of ignored frequency ranges in the object.
    """
    meta = {"ignore_freq_ranges": ignore_freq_ranges, **level5.meta}

    freq = FrequencyRange(level5.raw_frequencies, f_low=f_low, f_high=f_high)

    spec = level5.spectrum[freq.mask]
    wght = level5.weights[freq.mask]

    # Another round of XRFI
    flags = rfi.xrfi_poly_filter(
        spec,
        wght,
        window_width=int(3 / level5.freq.df),
        n_poly=2,
        n_bootstrap=20,
        n_sigma=3,
    )
    wght = np.where(flags, 0, level5.weights)

    if ignore_freq_ranges:
        for (low, high) in ignore_freq_ranges:
            wght[(freq.freq >= low) & (freq.freq <= high)] = 0

    if freq_resolution:
        f, spec, wght, s = tools.average_in_frequency(
            spec, freq.freq, wght, resolution=freq_resolution
        )
    else:
        f = freq.freq
        s = level5.ancillary["std_dev"]

    spec, wght = tools.weighted_mean(spec, weights=wght, axis=0)

    data = {"frequency": f, "spectrum": spec, "weights": wght}

    ancillary = {"std_dev": s}

    return data, ancillary, meta
