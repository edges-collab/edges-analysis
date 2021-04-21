from __future__ import annotations

import logging
from dataclasses import dataclass
from multiprocessing import cpu_count
from pathlib import Path
from typing import Tuple, List, Sequence, Dict, Union, Optional
from .levels import FilteredData, CalibratedData, FrequencyRange
import dill as pickle

import h5py
import numpy as np
import yaml
from multiprocess.pool import Pool

from matplotlib import pyplot as plt

logger = logging.getLogger(__name__)


@dataclass
class RMSInfo:
    gha: np.ndarray
    model_params: dict
    meta: dict
    rms: dict
    flags: dict
    model_eval: dict
    abs_resids: dict
    model_stds: dict
    poly_params: dict
    poly_params_std: dict

    @classmethod
    def from_file(cls, fname: [str, Path]) -> RMSInfo:
        fname = Path(fname)

        def get_key(fl, key):
            return {
                model: {
                    (float(band[1:].split(",")[0]), float(band[:-1].split(",")[1])): v[...]
                    for band, v in model_dct.items()
                }
                for model, model_dct in fl[key].items()
            }

        with h5py.File(fname, "r") as fl:
            data = {
                key: get_key(fl, key)
                for key in [
                    "rms",
                    "flags",
                    "model_eval",
                    "abs_resids",
                    "model_stds",
                    "poly_params",
                    "poly_params_std",
                ]
            }

            model_params = {}
            for mdl_name, mdl_data in fl["model_params"].items():
                model_params[mdl_name] = dict(mdl_data.attrs)

            out = cls(
                gha=fl["gha"][...],
                meta={k: v for k, v in fl.attrs.items() if not k.startswith("model_")},
                model_params=model_params,
                **data,
            )
        return out

    def write(self, fname: [str, Path]):
        with h5py.File(fname, "w") as fl:
            fl["gha"] = self.gha

            for key in [
                "rms",
                "flags",
                "model_eval",
                "abs_resids",
                "model_stds",
                "poly_params",
                "poly_params_std",
            ]:
                grp = fl.create_group(key)

                for model in self.model_names:
                    mdl_grp = grp.create_group(model)

                    for band in self.bands[model]:
                        mdl_grp[str(band)] = getattr(self, key)[model][band]

            for key, val in self.meta.items():
                fl.attrs[key] = val

            grp = fl.create_group("model_params")
            for model in self.model_names:
                mdl_grp = grp.create_group(model)

                for k, v in self.model_params[model].items():
                    mdl_grp.attrs[k] = v

    @property
    def model_names(self):
        """The names of the models used to generate RMS"""
        return tuple(self.model_eval.keys())

    @property
    def bands(self) -> Dict[str, List[Tuple[float, float]]]:
        """The bands."""
        return {m: list(self.model_eval[m].keys()) for m in self.model_names}

    def plot(
        self,
        n_sigma: int = 3,
        data_gha: Optional[np.ndarray] = None,
        data_rms: Optional[np.ndarray] = None,
    ):
        isort = np.argsort(self.gha)
        gha = self.gha[isort]

        i = 0
        for name, data in self.rms.items():
            for band, rms in data.items():
                model = self.model_eval[name][band][isort]
                std = self.model_stds[name][band][isort]

                plt.plot(gha, rms[isort], label=f"{name}: {band}", color=f"C{i}")
                plt.plot(gha, model, ls="--", color=f"C{i}")
                plt.fill_between(
                    gha, model - n_sigma * std, model + n_sigma * std, color=f"C{i}", alpha=0.4
                )
                if data_gha is not None:
                    plt.scatter(data_gha, data_rms[name][band], color=f"C{i}")
                i += 1

        plt.legend()


def get_rms_info(
    steps: Sequence[Union[FilteredData, CalibratedData]],
    models: Dict[str, Dict],
    n_poly: int = 3,
    n_sigma: float = 3,
    n_terms: int = 16,
    n_std: int = 6,
    n_threads: int = cpu_count(),
) -> RMSInfo:
    """Computation of RMS info on a subset of files.

    This function computes the RMS, and polynomial models of the RMS over GHA. It
    also computes an estimated model of the standard deviation of the model.

    Parameters
    ----------
    steps
        A list of :class:`~FilteredData` or :class:`~CalibratedData` objects on which
        to compute the RMS statistics.
    models
        Dictionary of models to fit in order to obtain the RMS. Each should be a dictionary,
        with keys "model", "params", "resolution", and "bands". The "model" should be a
        string matching an edges-cal model, the "params" should be a dictionary of
        parameters to pass to this :class:`Model`. The "resolution" should be an int or
        float specifying the frequency bin size (in no. of channels or bandwidth in MHz).
        The "bands" should be a list of tuples of frequency ranges (in MHz), or the
        strings "full", "low" or "high" (corresponding to the full band, lower half,
        or upper half). These are used to compute the RMS (the model itself is fit over
        the full range every time).
    n_poly
        Number of polynomial terms with which to fit the the RMS as a function of GHA,
        purely for flagging bad RMS values.
    n_sigma
        Number of sigma at which to threshold individual RMS values from the model
        fit over GHA.
    n_terms
        The number of polynomial terms to fit to the final model of RMS vs GHA.
    n_std
        The number of polynomial terms to fit to the absolute residual of RMS
        as a function of GHA.
    n_threads
        The number of threads to use to compute the RMS.

    Returns
    -------
    rms_info
        A special data structure representing the RMS calculation.

    See Also
    --------
    rms_filter
        The output of this function is supposed to be used in ``rms_filter`` to actually
        filter files (those files may not be the same as went into this function).
    """
    # Make a big vector of all GHA's in all files.
    gha = np.hstack([l1.ancillary["gha"] for l1 in steps])

    n_threads = min(n_threads, len(steps))
    if n_threads > 1:
        pool = Pool(n_threads)
        m = pool.map
    else:
        m = map

    bands = _get_band_specs(models, steps[0].freq)

    out = {name: {b: {} for b in bands} for name, bands in bands.items()}
    params = {}
    for name, model in models.items():
        # Get the RMS values for each of the files, for each of the bands.
        mdl = model.get("model")
        prms = model.get("params", {})

        # for k, v in steps[0].__memcache__["ancillary"].__dict__.items():
        #     print(k)
        #     pickle.dumps(v)

        # Put all the RMS values for all files into one long vector.
        rms = list(
            m(
                lambda i: steps[i].get_model_rms(
                    freq_ranges=list(bands[name]),
                    model=mdl,
                    resolution=model.get("resolution", 0.0488),
                    **prms,
                ),
                list(range(len(steps))),
            )
        )
        params[name] = prms.copy()
        params[name].update(**{k: v for k, v in model.items() if k not in ["params", "bands"]})

        for band in bands[name]:
            this = out[name][band]
            this["rms"] = np.hstack([r[band] for r in rms])
            this.update(get_rms_model_over_time(this["rms"], gha, n_poly, n_sigma, n_terms, n_std))

    # Transform the out dict to have the measurements as top-level keys
    new_out = _bring_measurements_to_top(out)

    return RMSInfo(
        gha=gha,
        meta={"n_poly": n_poly, "n_sigma": n_sigma, "n_terms": n_terms, "n_std": n_std},
        model_params=params,
        **new_out,
    )


def _bring_measurements_to_top(out: Dict) -> Dict[str, Dict]:
    for name, value_ in out.items():
        for band in value_:
            keys = value_[band].keys()

    new_out = {}
    for key in keys:
        new_out[key] = {}
        for name, value in out.items():
            new_out[key][name] = {}
            for band in value:
                new_out[key][name][band] = out[name][band][key]

    return new_out


def _get_band_specs(
    models: Dict[str, Dict], freq: FrequencyRange
) -> Dict[str, List[Tuple[float, float]]]:
    bands = {name: [] for name in models}
    for name, model in models.items():
        bands_ = model.get("bands", ["full"])
        if not bands_:
            raise ValueError("'bands' must be a list of strings/tuples")

        for b in bands_:
            if b == "full":
                b = (np.floor(freq.min), np.ceil(freq.max))
            elif b == "low":
                b = (np.floor(freq.min), np.floor(freq.center))
            elif b == "high":
                b = (np.floor(freq.center), np.ceil(freq.max))
            elif not isinstance(b, tuple):
                raise ValueError(f"'bands' must be a list of strings/tuples, got {b}")

            bands[name].append(b)

    return bands


def get_rms_model_over_time(
    rms: np.ndarray, gha: np.ndarray, n_poly: int, n_sigma: float, n_terms: int, n_std: int
):
    flags = np.isnan(rms)

    for i in range(24):  # Go through each hour
        mask = (gha >= i) & (gha < (i + 1))
        extern_flg = flags[mask]
        this_flg = flags[mask].copy()

        n_flg = np.sum(this_flg) - 1

        while np.sum(this_flg) > n_flg and np.sum(~this_flg) > 2 * n_poly:
            n_flg = np.sum(this_flg)
            res, std = _get_polyfit_res_std(n_poly, gha[mask], rms[mask], flags=this_flg)
            this_flg = extern_flg | (np.abs(res) > n_sigma * std)

        extern_flg[:] = this_flg

    par = np.polyfit(gha[~flags], rms[~flags], n_terms - 1)
    model = np.polyval(par, gha)
    abs_res = np.abs(rms - model)
    par_std = np.polyfit(gha[~flags], abs_res[~flags], n_std - 1)
    model_std = np.polyval(par_std, gha)

    return {
        "flags": flags,
        "model_eval": model,
        "abs_resids": abs_res,
        "model_stds": model_std,
        "poly_params": par,
        "poly_params_std": par_std,
    }


def rms_filter(
    rms_info: [str, Path, RMSInfo],
    gha: np.ndarray,
    rms: Dict[str, Dict[Tuple[float, float], np.ndarray]],
    n_sigma: float = 3,
    fl_id=None,
):
    """Filter RMS data based on a summary set of data.

    Parameters
    ----------
    rms_info
        The output of ``get_rms_info``, or a file containing it.
    gha
        1D array of the GHA's of the current data.
    rms
        The RMS of the current data. Keys are different models (must match those in
        ``rms_info``, and values are dicts with keys that are the bands.
    n_sigma
        The threshold at which individual RMS values are flagged.

    Returns
    -------
    flags
        1D array of len(gha) specifying which integrations to flag.
    """
    if not isinstance(rms_info, RMSInfo):
        rms_info = RMSInfo.from_file(rms_info)

    flags = np.zeros(len(gha), dtype=bool)
    fl_id = f"{fl_id}: " if fl_id else ""

    for model, rms_model in rms.items():
        for band, rms_band in rms_model.items():
            pp = rms_info.poly_params[model][band]
            pstd = rms_info.poly_params_std[model][band]

            m = np.polyval(pp, gha)
            ms = np.polyval(pstd, gha)
            flags |= np.abs(rms_band - m) > n_sigma * ms

            logger.info(
                f"{fl_id}{np.sum(flags)}/{len(flags)} GHA's flagged after RMS filter for model={model}, band={band}."
            )

    return flags


def total_power_filter(
    gha: np.ndarray,
    spectra: np.ndarray,
    frequencies: np.ndarray,
    flags: [None, np.ndarray] = None,
    n_poly: int = 3,
    n_sigma: float = 3.0,
    bands: [None, List[Tuple[float, float]]] = None,
    std_thresholds=None,
):
    """
    Filter on total power.

    The algorithm of the filter is:

    Get the sum of each spectrum (for a night/day) in three different bands (low half,
    upper half, total) (yielding, for each band, a "total power" as a function of GHA).
    For each band, split the total power into smallish chunks of GHA (nominally 1hr bins),
    and for each bin, fit a low-order polynomial to the total power.
    GHA's whose total power is outside some threshold are flagged, and the previous step
    is repeated, until no new GHAs are flagged. This threshold is itself thresholded:
    if the RMS of the residuals of the fit are very large, even 1-sigma outliers are
    flagged. Otherwise, only 3-4-sigma outliers are flagged.


    Parameters
    ----------
    gha : array_like
        1D array of galactic hour angles
    spectra
        2D array where first dimension is length GHA and corresponds to different
        integrations, and second dimension corresponds to frequency.
    frequencies
        The frequencies at which the spectra are defined.
    flags
        Any flagged entries in the spectra (2D array, shape (GHA, freq)).
    n_poly
        The number of polynomial terms to fit to each 1-hour GHA bin.
    n_sigma
        The number of sigma at which to threshold individual integrations.
    bands
        A list of tuples, each with two floats: min and max of a frequency band over
        which to take the total power. By default, use the whole band only.
    std_thresholds
        The absolute threshold for each band in terms of the standard deviation of
        residuals. If very high, then individual integrations will be flagged at
        one sigma deviation rather than ``n_sigma``. Must be the same length as ``bands``
        if given.

    """
    # Set the relevant frequency bands over which to take the total powers.
    if bands is None:
        bands = [(frequencies.min(), frequencies.max())]

    for i, band in bands:
        if band is None:
            bands[i] = (frequencies.min(), frequencies.max())

    assert spectra.shape == (len(gha), len(frequencies)), "total_power has wrong shape"

    if flags is None:
        flags = np.zeros(spectra.shape, dtype=bool)

    # Now sum over the frequency bands.
    total_power = np.zeros((len(bands), len(gha)))
    for i, band in enumerate(bands):
        freq_mask = (frequencies >= band[0]) & (frequencies < band[1])
        total_power[i] = np.nanmean(np.where((~flags) | freq_mask, spectra, np.nan), axis=1)

    if std_thresholds is None:
        std_thresholds = [None] * len(bands)

    flags_1d = np.prod(flags, axis=1).astype(bool)

    for j, (tpi, std_threshold) in enumerate(zip(total_power, std_thresholds)):
        for i in range(24):
            mask = (gha >= i) & (gha < (i + 1))
            this_gha = gha[mask]
            this_extern_flags = flags_1d[mask]

            this_tp = tpi[mask]

            this_intern_flags = this_extern_flags | np.isnan(this_tp)

            # If enough data points available per hour
            lx = np.sum(~this_intern_flags)

            if lx <= 10:
                continue

            nflags = -1

            while nflags < np.sum(this_intern_flags):
                nflags = np.sum(this_intern_flags)

                if (len(this_gha) - nflags) < n_poly:
                    # If we've flagged too much, just get rid of everything here.
                    this_intern_flags[:] = True
                else:
                    res, std = _get_polyfit_res_std(n_poly, this_gha, this_tp, this_intern_flags)

                    if std_threshold is not None and std > std_threshold:
                        this_intern_flags = this_extern_flags | (np.abs(res) > std)
                    else:
                        this_intern_flags = this_extern_flags | (np.abs(res) > n_sigma * std)

            flags_1d[mask] = this_intern_flags

    return flags_1d


def _get_polyfit_res_std(n_poly, gha, data, flags=None):
    if flags is None:
        flags = np.zeros(len(gha), dtype=bool)

    if np.sum(~flags) < n_poly:
        raise np.linalg.LinAlgError("After flagging, too few data points left for n_poly.")

    try:
        par = np.polyfit(gha[~flags], data[~flags], n_poly - 1)
    except np.linalg.LinAlgError:
        logger.error("gha:", gha)
        logger.error("data: ", data)
        logger.error("n_poly: ", n_poly)
        raise
    model = np.polyval(par, gha)
    res = data - model
    std = np.nanstd(res)

    return res, std


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
        with open(bad, "r") as fl:
            bad = yaml.load(fl, Loader=yaml.FullLoader)["bad_days"]

    try:
        nt = len(times[0])
    except AttributeError:
        nt = 1

    try:
        nb = len(bad[0])
    except AttributeError:
        nb = 1

    assert nt in (1, 2, 3), "times must be an array of 1,2 or 3-tuples"
    assert nb in (1, 2, 3), "bad must be an array of 1,2 or 3-tuples"

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
    gha_range: Tuple[float, float] = (0, 24),
    sun_el_max: float = 90,
    moon_el_max: float = 90,
    amb_hum_max: float = 200,
    min_receiver_temp: float = 0,
    max_receiver_temp: float = 100,
    flags: [None, np.ndarray] = None,
) -> np.ndarray:
    loc_flgs = np.zeros(len(gha), dtype=bool)

    def filter(condition, message, loc_flgs):
        nflags = np.sum(loc_flgs)

        loc_flgs |= condition
        nnew = np.sum(loc_flgs) - nflags
        if nnew:
            logger.debug(f"{nnew}/{len(loc_flgs) - nflags} times flagged due to {message}")

    filter((gha < gha_range[0]) | (gha >= gha_range[1]), "GHA range", loc_flgs)
    filter(sun_el > sun_el_max, "sun position", loc_flgs)
    filter(moon_el > moon_el_max, "moon position", loc_flgs)
    filter(humidity > amb_hum_max, "humidity", loc_flgs)
    filter(
        (receiver_temp >= max_receiver_temp) | (receiver_temp <= min_receiver_temp),
        "receiver temp",
        loc_flgs,
    )

    if flags is not None:
        assert flags.shape[-1] == len(
            gha
        ), f"flags must be an array with last axis len(gha). Got {flags.shape}"
        flags |= loc_flgs
    else:
        flags = loc_flgs

    return flags


def filter_explicit_gha(gha, first_day, last_day):
    # TODO: this should not be used -- it's arbitrary!
    if gha in [0, 2, 3, 4, 5, 16, 17, 18, 19, 20, 21, 22]:
        good_days = np.arange(140, 300, 1)
    else:
        good_day_dct = {
            1: np.concatenate((np.arange(148, 160), np.arange(161, 220))),
            10: np.concatenate(
                (
                    np.arange(148, 168),
                    np.arange(177, 194),
                    np.arange(197, 202),
                    np.arange(205, 216),
                )
            ),
            11: np.arange(187, 202),
            12: np.arange(147, 150),
            13: np.array([147, 149, 157, 159]),
            14: np.arange(148, 183),
            15: np.concatenate((np.arange(140, 183), np.arange(187, 206), np.arange(210, 300))),
            23: np.arange(148, 300),
            6: np.arange(147, 300),
            7: np.concatenate(
                (
                    np.arange(147, 153),
                    np.arange(160, 168),
                    np.arange(174, 202),
                    np.arange(210, 300),
                )
            ),
            8: np.concatenate((np.arange(147, 151), np.arange(160, 168), np.arange(174, 300))),
            9: np.concatenate(
                (
                    np.arange(147, 153),
                    np.arange(160, 168),
                    np.arange(174, 194),
                    np.arange(197, 202),
                    np.arange(210, 300),
                )
            ),
        }
        good_days = good_day_dct[gha]

    return good_days[(good_days >= first_day) & (good_days <= last_day)]
