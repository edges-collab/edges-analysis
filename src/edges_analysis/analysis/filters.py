import os

import numpy as np

from .io import level3read
from ..config import config
import yaml


# TODO: clean up all the rms filtering functions in this module.
def rms_filter_computation(band, case, save_parameters=False):
    """
    Computation of the RMS filter
    """
    # Listing files available
    path_files_direc = config["edges_folder"] + f"/{band}/spectra/level3/"
    save_direc = config["edges_folder"] + f"/{band}/rms_filters/"

    if band == "low_band3":
        if case == 2:
            pth = "case2/"
    elif band == "mid_band":
        paths = {
            10: "rcv18_sw18_nominal/",
            20: "rcv18_sw19_nominal/",
            2: "calibration_2019_10_no_ground_loss_no_beam_corrections/",
            3: "case_nominal_50-150MHz_no_ground_loss_no_beam_corrections/",
            5: "case_nominal_14_14_terms_55-150MHz_no_ground_loss_no_beam_corrections/",
            501: "case_nominal_50-150MHz_LNA2_a2_h2_o2_s1_sim2_all_lc_yes_bc/",
        }
        if case not in paths:
            raise ValueError(
                "for mid_band, case must be one of {}".format(paths.keys())
            )

        pth = paths[case]
    else:
        raise ValueError("band must be 'mid_band' or 'low_band3'.")
    path_files = path_files_direc + pth
    save_folder = save_direc + pth

    new_list = os.listdir(path_files)
    new_list.sort()

    # Loading data used to compute filter
    # -----------------------------------
    n_files = 8  # Only using the first "N_files" to compute the filter
    rms_all, m_all = [], []
    for i in range(n_files):  #
        print(new_list[i])

        # Loading data
        f, t, p, r, w, rms, tp, m = level3read(path_files + new_list[i])

        # Filter out high humidity
        amb_hum_max = 40
        indices = time_filter_auxiliary(
            m,
            use_gha="GHA",
            time_1=0,
            time_2=24,
            sun_el_max=90,
            moon_el_max=90,
            amb_hum_max=amb_hum_max,
            min_receiver_temp=0,
            max_receiver_temp=100,
        )

        # Accumulate data
        rms_all.append(rms[indices])
        m_all.append(m[indices])

    rms_all = np.vstack(rms_all)
    m_all = np.vstack(m_all)

    # Columns necessary for analysis
    # ------------------------------
    gha = m_all[:, 4]
    gha[gha < 0] += 24

    indices = np.arange(0, len(gha))

    # Number of polynomial terms used to fit each 1-hour bin
    # and number of sigma threshold
    n_poly = 3
    n_sigma = 3
    n_terms = 16
    n_std = 6

    # Identification of bad data, within 1-hour "bins", across 24 hours
    out = [
        perform_rms_filter(rms_all[:, i], indices, gha, n_poly, n_sigma, n_terms, n_std)
        for i in range(3)
    ]

    if save_parameters:
        par = np.array([o[-2] for o in out])
        par_std = np.array([o[-1] for o in out])

        np.savetxt(save_folder + "rms_polynomial_parameters.txt", par)
        np.savetxt(save_folder + "rms_std_polynomial_parameters.txt", par_std)

    out = [gha] + [out[i][j] for i in range(3) for j in range(5)]
    return tuple(out)


def perform_rms_filter(rms, indices, gha, n_poly, n_sigma, n_terms, n_std):
    for i in range(24):
        GHA_x = gha[(gha >= i) & (gha < (i + 1))]
        RMS_x = rms[(gha >= i) & (gha < (i + 1))]
        IN_x = indices[(gha >= i) & (gha < (i + 1))]

        W = np.ones(len(GHA_x))
        bad_old = -1
        bad = 0
        iteration = 0

        while bad > bad_old:
            res, std, iteration = get_model_residual_iter(
                W, i, iteration, n_poly, GHA_x, RMS_x
            )

            IN_x_bad = IN_x[np.abs(res) > n_sigma * std]
            W[np.abs(res) > n_sigma * std] = 0

            bad_old = np.copy(bad)
            bad = len(IN_x_bad)

            print("STD: " + str(np.round(std, 3)) + " K")
            print("Number of bad points excised: " + str(bad))

        bad = np.copy(IN_x_bad) if i == 0 else np.append(bad, IN_x_bad)
    good = np.setdiff1d(indices, bad)

    par = np.polyfit(gha[good], rms[good], n_terms - 1)
    model = np.polyval(par, gha)
    abs_res = np.abs(rms - model)
    par_std = np.polyfit(gha[good], abs_res[good], n_std - 1)
    model_std = np.polyval(par_std, gha)

    return rms, good, model, abs_res, model_std, par, par_std


def rms_filter(band, case, gx, rms, n_sigma):
    prefix = config["edges_folder"] + band + "/rms_filters/"

    if band == "mid_band":
        if 10 <= case <= 19:
            file_path = "rcv18_sw18_nominal/"
        elif 20 <= case <= 29:
            file_path = "rcv18_sw19_nominal/"
        elif case == 2:
            file_path = "calibration_2019_10_no_ground_loss_no_beam_corrections/"
        elif case in [3, 406]:
            file_path = "case_nominal_50-150MHz_no_ground_loss_no_beam_corrections/"
        elif case == 5:
            file_path = (
                "case_nominal_14_14_terms_55-150MHz_no_ground_loss_no_beam_corrections/"
            )
        elif case == 501:
            file_path = "case_nominal_50-150MHz_LNA2_a2_h2_o2_s1_sim2_all_lc_yes_bc/"
        else:
            raise ValueError("case {} does not exist".format(case))
    else:
        raise ValueError("band must be mid_band")

    p = np.genfromtxt(prefix + file_path + "rms_polynomial_parameters.txt")
    ps = np.genfromtxt(prefix + file_path + "rms_std_polynomial_parameters.txt")

    m = [np.polyval(pp, gx) for pp in p]
    ms = [np.polyval(pp, gx) for pp in ps]

    index = np.arange(0, len(rms))

    diff = [np.abs(rr - mm) for rr, mm in zip(rms, m)]
    good_indices = [index[d <= n_sigma * mm] for d, mm in zip(diff, ms)]

    index_good = np.intersect1d(good_indices[0], good_indices[1])
    index_good = np.intersect1d(index_good, good_indices[2])

    return (index_good, *good_indices)


def total_power_filter(gha, tp):
    """
    Filter on total power.

    Parameters
    ----------
    gha : array_like
        1D array of galactic hour angles
    tp : array_like
        2D array where first dimension corresponds to gha, and second dimension is
        length 3 and corresponds to different frequency ranges (60-90, 90-120, 60-120).
    """
    indices = np.arange(0, len(gha))
    n_poly = 3
    n_sigma = 3

    assert tp.shape == (len(gha), 3)

    std_thresholds = [5e4, 2e4, 1e5]
    good_indx = []
    for j, (tpi, std_threshold) in enumerate(zip(tp, std_thresholds)):

        flag = False

        for i in range(24):
            mask = (gha >= i) & (gha < (i + 1))
            this_gha = gha[mask]

            this_tp = tpi[mask]
            this_indx = indices[mask]

            # If enough data points available per hour
            lx = len(this_tp)
            if lx <= 10:
                continue

            W = np.ones(len(this_gha))
            bad_old = -1
            bad = 0
            iteration = 0

            while bad_old < bad < int(lx / 2):
                res, std, iteration = get_model_residual_iter(
                    W, i, iteration, n_poly, this_gha, this_tp
                )

                if std <= std_threshold:
                    mask = np.abs(res) > n_sigma * std
                else:
                    mask = np.abs(res) > std

                bad_indx = this_indx[mask]
                W[mask] = 0

                bad_old = np.copy(bad)
                bad = len(bad_indx)

                if bad >= int(lx / 2):
                    bad_indx = np.copy(this_indx)

                print("STD: " + str(np.round(std, 3)) + " K")
                print("Number of bad points excised: " + str(bad))

            # Indices of bad data points
            if not flag:
                IN_bad = np.copy(bad_indx)
                flag = True
            else:
                IN_bad = np.append(IN_bad, bad_indx)

        bad = np.copy(IN_bad)
        good_indx.append(np.setdiff1d(indices, bad))

    # Combined index of good data points
    good = np.intersect1d(good_indx[0], good_indx[1])
    good = np.intersect1d(good, good_indx[2])

    return (good, *good_indx)


def get_model_residual_iter(W, i, iteration, n_poly, this_gha, this_tp, verbose=False):
    iteration += 1

    if verbose:
        print(" ")
        print("------------")
        print("GHA: " + str(i) + "-" + str(i + 1) + "hr")
        print("Iteration: " + str(iteration))

    par = np.polyfit(this_gha[W > 0], this_tp[W > 0], n_poly - 1)
    model = np.polyval(par, this_gha)
    res = this_tp - model
    std = np.std(res[W > 0])

    return res, std, iteration


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


def daily_rms_filter(band, case, index_GHA, year_day_list, rms_threshold):
    if band == "mid_band" and case == 1:
        d = np.genfromtxt(
            config["edges_folder"]
            + "mid_band/spectra/level4/case1/rms_filters/rms_index"
            + str(index_GHA)
            + ".txt"
        )
        rms_original = d[:, -1]
        good = d[rms_original <= rms_threshold, :]

    return np.array(
        [
            any(year_day[0] == b[0] and year_day[1] == b[1] for b in good)
            for year_day in year_day_list
        ]
    )


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
    good = np.ones(len(gha), dtype=bool)

    good &= gha >= gha_range[0] & gha < gha_range[1]

    # Sun elevation, Moon elevation, ambient humidity, and receiver temperature
    good &= sun_el <= sun_el_max
    good &= moon_el <= moon_el_max
    good &= humidity <= amb_hum_max
    good &= receiver_temp >= min_receiver_temp & receiver_temp <= max_receiver_temp

    return good
