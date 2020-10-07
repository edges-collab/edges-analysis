from __future__ import annotations
from pathlib import Path
import numpy as np
from typing import Tuple, List, Sequence, Dict, Union
from dataclasses import dataclass
import h5py
import yaml
from edges_io.logging import logger
from multiprocess.pool import Pool
from multiprocessing import cpu_count


@dataclass
class RMSInfo:
    gha: np.ndarray
    rms: dict
    flags: dict
    models: dict
    abs_resids: dict
    model_stds: dict
    poly_params: dict
    poly_params_std: dict
    bands: List[Tuple[float, float]]
    model: dict
    meta: dict

    @classmethod
    def from_file(cls, fname: [str, Path]) -> RMSInfo:
        fname = Path(fname)

        with h5py.File(fname, "r") as fl:
            bands = fl["bands"][...]
            out = cls(
                gha=fl["gha"][...],
                rms={k: v[...] for k, v in fl["rms"].items()},
                flags={k: v[...] for k, v in fl["flags"].items()},
                models={k: v[...] for k, v in fl["models"].items()},
                abs_resids={k: v[...] for k, v in fl["abs_resids"].items()},
                model_stds={k: v[...] for k, v in fl["model_stds"].items()},
                poly_params={k: v[...] for k, v in fl["poly_params"].items()},
                poly_params_std={k: v[...] for k, v in fl["poly_params_std"].items()},
                bands=[(bands[2 * i], bands[2 * i + 1]) for i in range(len(bands) // 2)],
                meta={k: v for k, v in fl.attrs.items() if not k.startswith("model_")},
                model={k[6:]: v for k, v in fl.attrs.items() if k.startswith("model_")},
            )
        return out

    def write(self, fname: [str, Path]):
        with h5py.File(fname, "w") as fl:
            fl["gha"] = self.gha

            for key in [
                "rms",
                "flags",
                "models",
                "abs_resids",
                "model_stds",
                "poly_params",
                "poly_params_std",
            ]:
                grp = fl.create_group(key)
                for band in self.bands:
                    grp[str(band)] = getattr(self, key)[band]

            for key, val in self.meta.items():
                fl.attrs[key] = val

            for key, val in self.model.items():
                fl.attrs[f"model_{key}"] = val

            fl["bands"] = [v for k in self.bands for v in k]


def get_rms_info(
    level1: Sequence,
    bands: Sequence[Union[Tuple, str]] = ("full", "low", "high"),
    n_poly: int = 3,
    n_sigma: float = 3,
    n_terms: int = 16,
    n_std: int = 6,
    n_threads: int = cpu_count(),
    **rms_model_kwargs,
) -> RMSInfo:
    """Computation of RMS info on a subset of files.

    This function computes the RMS, and polynomial models of the RMS over GHA. It
    also computes an estimated model of the standard deviation of the model.

    Parameters
    ----------
    level1
        A list of :class:`~Level1` objects on which to compute the RMS statistics.
    bands
        The frequency bands in which to compute the model RMS values. By default,
        compute for the entire frequency band, the lower half, and the upper half.
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

    Returns
    -------
    dict
        A dictionary of output arrays, each a model/residual of the fits.

    See Also
    --------
    rms_filter
        The output of this function is supposed to be used in ``rms_filter`` to actually
        filter files (those files may not be the same as went into this function).
    """

    # Get a tuple representation of the bands.
    bands_ = []
    l1 = level1[0]
    for b in bands:
        if b == "full":
            b = (l1.freq.min, l1.freq.max)
        elif b == "low":
            b = (l1.freq.min, l1.freq.center)
        elif b == "high":
            b = (l1.freq.center, l1.freq.max)
        bands_.append(b)
    bands = bands_

    # Make a big vector of all GHA's in all files.
    gha = np.hstack([l1.ancillary["gha"] for l1 in level1])

    n_threads = min(n_threads, len(level1))
    if n_threads > 1:
        pool = Pool(n_threads)
        m = pool.map
    else:
        m = map
    # Get the RMS values for each of the files, for each of the bands.
    rms = {}
    for j, band in enumerate(bands):
        # Put all the RMS values for all files into one long vector.
        rms[band] = np.hstack(
            m(lambda i: level1[i].get_model_rms(freq_range=band, **rms_model_kwargs))
        )

    flags = {}
    models = {}
    abs_resids = {}
    model_stds = {}
    poly_params = {}
    poly_params_std = {}

    for band in bands:
        good, model, abs_res, model_std, par, par_std = get_rms_model_over_time(
            rms[band], gha, n_poly, n_sigma, n_terms, n_std
        )

        flags[band] = ~good
        models[band] = model
        abs_resids[band] = abs_res
        model_stds[band] = model_std
        poly_params[band] = par
        poly_params_std[band] = par_std

    return RMSInfo(
        gha=gha,
        rms=rms,
        flags=flags,
        models=models,
        abs_resids=abs_resids,
        model_stds=model_stds,
        poly_params=poly_params,
        poly_params_std=poly_params_std,
        bands=bands,
        model=rms_model_kwargs,
        meta={"n_poly": n_poly, "n_sigma": n_sigma, "n_terms": n_terms, "n_std": n_std},
    )


def get_rms_model_over_time(
    rms: np.ndarray, gha: np.ndarray, n_poly: int, n_sigma: float, n_terms: int, n_std: int
):
    good = np.ones(len(rms), dtype=bool)

    for i in range(24):  # Go through each hour
        mask = (gha >= i) & (gha < (i + 1))
        n_orig = np.sum(mask)

        while np.sum(good[mask]) < n_orig:
            n_orig = np.sum(good[mask])
            res, std = _get_polyfit_res_std(n_poly, gha[mask], rms[mask], weights=good[mask])
            good[mask] &= np.abs(res) <= n_sigma * std

    par = np.polyfit(gha[good], rms[good], n_terms - 1)
    model = np.polyval(par, gha)
    abs_res = np.abs(rms - model)
    par_std = np.polyfit(gha[good], abs_res[good], n_std - 1)
    model_std = np.polyval(par_std, gha)

    return good, model, abs_res, model_std, par, par_std


def rms_filter(
    rms_info: [str, Path, RMSInfo], gha: np.ndarray, rms: np.ndarray, n_sigma: float = 3
):
    """Filter RMS data based on a summary set of data.

    Parameters
    ----------
    rms_info
        The output of ``get_rms_info``, or a file containing it.
    gha
        1D array of the GHA's of the current data.
    rms
        The RMS of the current data.
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

    for pp, pstd, rr in zip(rms_info.poly_params.values(), rms_info.poly_params_std.values(), rms):
        m = np.polyval(pp, gha)
        ms = np.polyval(pstd, gha)
        flags |= np.abs(rr - m) > n_sigma * ms

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
            this_flags = flags_1d[mask]

            this_tp = tpi[mask]

            this_flags |= np.isnan(this_tp)

            # If enough data points available per hour
            lx = len(this_flags) - np.sum(this_flags)

            if lx <= 10:
                continue

            nflags = -1

            while nflags < np.sum(this_flags):
                nflags = np.sum(this_flags)

                if (len(this_gha) - nflags) < n_poly:
                    # If we've flagged too much, just get rid of everything here.
                    this_flags[:] = True
                else:
                    res, std = _get_polyfit_res_std(n_poly, this_gha, this_tp, this_flags)

                    if std_threshold is not None and std > std_threshold:
                        this_flags[~this_flags] |= np.abs(res) > std
                    else:
                        this_flags[~this_flags] |= np.abs(res) > n_sigma * std

            flags_1d[mask] = this_flags

    return flags_1d


def _get_polyfit_res_std(n_poly, gha, data, flags=None):
    if flags is not None:
        gha = gha[~flags]
        data = data[~flags]

    if len(gha) < n_poly:
        raise np.linalg.LinAlgError("After flagging, too few data points left for n_poly.")

    try:
        par = np.polyfit(gha, data, n_poly - 1)
    except np.linalg.LinAlgError:
        print("gha:", gha)
        print("data: ", data)
        print("n_poly: ", n_poly)
        raise
    model = np.polyval(par, gha)
    res = data - model
    std = np.std(res)

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
