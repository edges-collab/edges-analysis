from pathlib import Path
import numpy as np
from typing import Tuple, List

import h5py
from ..config import config
import yaml
from edges_io.logging import logger


# TODO: clean up all the rms filtering functions in this module.
def rms_filter_computation(level3, path_out, band, n_files=None):
    """
    Computation of the RMS filter
    """
    path_out = Path(path_out)
    if not path_out.is_absolute():
        path_out = Path(config["paths"]["field_products"]) / band / "rms_filters" / path_out

    # Sort the inputs in ascending date.
    level3 = sorted(level3, key=lambda x: f"{x.meta['year'] - x.meta['day'] - x.meta['hour']}")

    # Load data used to compute filter
    # Only using the first "N_files" to compute the filter
    n_files = n_files or len(level3)

    rms_lower, rms_upper, rms_full, gha = [], [], [], []
    # Filter out high humidity
    amb_hum_max = 40
    for i, l3 in enumerate(level3[:n_files]):
        good = time_filter_auxiliary(
            gha=l3.ancillary["gha"],
            sun_el=l3.ancillary["sun_azel"][1],
            moon_el=l3.ancillary["moon_azel"][1],
            humidity=l3.ancillary["ambient_humidity"],
            receiver_temp=l3.ancillary["receiver1_temp"],
            sun_el_max=90,
            moon_el_max=90,
            amb_hum_max=amb_hum_max,
            min_receiver_temp=0,
            max_receiver_temp=100,
        )

        # Get RMS
        rms_lower.append(l3.get_model_rms(freq_range=(-np.inf, l3.freq.center)))
        rms_upper.append(l3.get_model_rms(freq_range=(l3.freq.center, np.inf)))
        rms_full.append(l3.get_model_rms(n_terms=3))

        # Accumulate data
        gha.append(l3.ancillary["gha"][good])

    rms_lower = np.vstack(rms_lower)
    rms_upper = np.vstack(rms_upper)
    rms_full = np.vstack(rms_full)
    gha = np.vstack(gha)

    with h5py.File(str(path_out), "w") as fl:
        # Number of polynomial terms used to fit each 1-hour bin
        # and number of sigma threshold
        n_poly = 3
        n_sigma = 3
        n_terms = 16
        n_std = 6

        for rms, label in zip([(rms_lower, rms_upper, rms_full), ("lower", "upper", "full")]):

            # Identification of bad data, within 1-hour "bins", across 24 hours
            rms, good, model, abs_res, model_std, par, par_std = perform_rms_filter(
                rms, gha, n_poly, n_sigma, n_terms, n_std
            )

            grp = fl.create_group(label)
            grp["rms"] = rms
            grp["good_indices"] = good
            grp["model"] = model
            grp["abs_resid"] = abs_res
            grp["model_std"] = model_std
            grp["polynomial_params"] = par
            grp["polynomial_params_std"] = par_std


def perform_rms_filter(rms, gha, n_poly, n_sigma, n_terms, n_std):
    good = np.ones(len(rms), dtype=bool)

    for i in range(24):  # Go through each hour
        mask = (gha >= i) & (gha < (i + 1))
        n_orig = np.sum(mask)

        while np.sum(good[mask]) < n_orig:
            n_orig = np.sum(good[mask])
            res, std = _get_model_residual_iter(n_poly, gha[mask], rms[mask], weights=good[mask])
            good[mask] &= np.abs(res) <= n_sigma * std

    par = np.polyfit(gha[good], rms[good], n_terms - 1)
    model = np.polyval(par, gha)
    abs_res = np.abs(rms - model)
    par_std = np.polyfit(gha[good], abs_res[good], n_std - 1)
    model_std = np.polyval(par_std, gha)

    return rms, good, model, abs_res, model_std, par, par_std


def rms_filter(filter_file, gx, rms, n_sigma):
    p, ps = [], []
    with h5py.File(filter_file) as fl:
        for key in ["lower", "upper", "full"]:
            p.append(fl[key]["polynomial_params"][...])
            ps.append(fl[key]["polynomial_params_std"][...])

    flags = np.zeros(rms.shape[-1], dtype=bool)

    for pp, pstd, rr in zip(p, ps, rms):
        m = np.polyval(pp, gx)
        ms = np.polyval(pstd, gx)
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
        flags = np.zeros(len(gha), dtype=bool)

    # Now sum over the frequency bands.
    total_power = np.zeros((len(bands), len(gha)))
    for i, band in enumerate(bands):
        freq_mask = (frequencies >= band[0]) & (frequencies < band[1])
        total_power[i] = np.nanmean(spectra[:, freq_mask], axis=1)

    print("total_power", total_power)

    if std_thresholds is None:
        std_thresholds = [None] * len(bands)

    for j, (tpi, std_threshold) in enumerate(zip(total_power, std_thresholds)):
        for i in range(24):
            mask = (gha >= i) & (gha < (i + 1))
            this_gha = gha[mask]
            this_flags = flags[mask]

            this_tp = tpi[mask]

            # If enough data points available per hour
            lx = len(this_flags) - np.sum(this_flags)

            if lx <= 10:
                continue

            nflags = -1

            while nflags < np.sum(this_flags):
                nflags = np.sum(this_flags)

                res, std = _get_model_residual_iter(n_poly, this_gha, this_tp, this_flags)
                # print("res, std", res, std)
                if std_threshold is not None and std > std_threshold:
                    this_flags |= np.abs(res) > std
                else:
                    this_flags |= np.abs(res) > n_sigma * std

            flags[mask] = this_flags

    return flags


def _get_model_residual_iter(n_poly, gha, tp, flags=None):
    if flags is not None:
        this_gha = gha[~flags]
        this_tp = tp[~flags]

    par = np.polyfit(this_gha, this_tp, n_poly - 1)
    model = np.polyval(par, gha)
    res = tp - model
    std = np.std(res[~flags])

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
