from pathlib import Path
import numpy as np
import warnings

import h5py
from ..config import config
import yaml


# TODO: clean up all the rms filtering functions in this module.
def rms_filter_computation(level3, path_out, band, n_files=None):
    """
    Computation of the RMS filter
    """
    path_out = Path(path_out)
    if not path_out.is_absolute():
        path_out = (
            Path(config["paths"]["field_products"]) / band / "rms_filters" / path_out
        )

    # Sort the inputs in ascending date.
    level3 = sorted(
        level3, key=lambda x: f"{x.meta['year'] - x.meta['day'] - x.meta['hour']}"
    )

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

        for rms, label in zip(
            [(rms_lower, rms_upper, rms_full), ("lower", "upper", "full")]
        ):

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
            res, std = _get_model_residual_iter(
                n_poly, gha[mask], rms[mask], weights=good[mask]
            )
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


def total_power_filter(gha, total_power, flags=None):
    """
    Filter on total power.

    Parameters
    ----------
    gha : array_like
        1D array of galactic hour angles
    total_power : array_like
        2D array where first dimension is length 3 and corresponds to different
        frequency ranges (60-90, 90-120, 60-120), and second dimension corresponds to gha.
    """
    # indices = np.arange(0, len(gha))
    n_poly = 3
    n_sigma = 3

    assert total_power.shape == (3, len(gha))
    if flags is None:
        flags = np.zeros(len(gha), dtype=bool)

    # This only applies to mid-band, and they are gotten empirically from some set
    # of data.
    std_thresholds = [5e4, 2e4, 1e5]

    for j, (tpi, std_threshold) in enumerate(zip(total_power, std_thresholds)):
        for i in range(24):
            mask = (gha >= i) & (gha < (i + 1))
            this_gha = gha[mask]
            this_flags = flags[mask]

            this_tp = tpi[mask]
            # this_indx = indices[mask]

            # If enough data points available per hour
            lx = len(this_flags) - np.sum(this_flags)

            if lx <= 10:
                warnings.warn(
                    f"GHA {i} didn't have enough unflagged data to do a total-power filter."
                )
                continue

            nflags = -1

            while nflags < np.sum(this_flags):
                nflags = np.sum(this_flags)

                res, std = _get_model_residual_iter(
                    n_poly, this_gha, this_tp, weights=(~this_flags).astype("float")
                )

                if std <= std_threshold:
                    flags |= np.abs(res) > n_sigma * std
                else:
                    flags |= np.abs(res) > std

    return flags


def _get_model_residual_iter(n_poly, this_gha, this_tp, weights=None):

    if weights:
        mask = weights > 0
        this_gha = this_gha[mask]
        this_tp = this_tp[mask]

    par = np.polyfit(this_gha, this_tp, n_poly - 1)
    model = np.polyval(par, this_gha)
    res = this_tp - model
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
        bad = set(b[:nt] for b in bad)
        nb = nt

    keep = [t[:nb] not in bad for t in times]

    if ret_times:
        return keep, times[keep]
    else:
        return keep


def time_filter_auxiliary(
    gha,
    sun_el,
    moon_el,
    humidity,
    receiver_temp,
    gha_range=(0, 24),
    sun_el_max=90,
    moon_el_max=90,
    amb_hum_max=200,
    min_receiver_temp=0,
    max_receiver_temp=100,
):
    flags = np.zeros(len(gha), dtype=bool)

    flags |= gha < gha_range[0] | gha >= gha_range[1]

    # Sun elevation, Moon elevation, ambient humidity, and receiver temperature
    flags |= sun_el > sun_el_max
    flags |= moon_el > moon_el_max
    flags |= humidity > amb_hum_max
    flags |= receiver_temp >= min_receiver_temp & receiver_temp <= max_receiver_temp

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
            15: np.concatenate(
                (np.arange(140, 183), np.arange(187, 206), np.arange(210, 300))
            ),
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
            8: np.concatenate(
                (np.arange(147, 151), np.arange(160, 168), np.arange(174, 300))
            ),
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
