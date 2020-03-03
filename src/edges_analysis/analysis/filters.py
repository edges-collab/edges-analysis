import os

import numpy as np

from .io import data_selection, level3read
from ..config import config


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
        indices = data_selection(
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


def daily_nominal_filter(band, case, index_gha, year_day_list):
    assert index_gha in range(24)

    if band == "mid_band":
        if case == 101:
            if index_gha in [0, 1, 2, 6, 16, 19]:
                bad = np.array([])
            elif index_gha == 10:
                bad = np.array(
                    [[2018, 176], [2018, 196], [2018, 201], [2018, 204], [2018, 218]]
                )
            elif index_gha == 11:
                bad = np.array([[2018, 149], [2018, 204], [2018, 216]])
            elif index_gha == 12:
                bad = np.array([[2018, 176], [2018, 195], [2018, 204]])
            elif index_gha == 13:
                bad = np.array(
                    [[2018, 176], [2018, 185], [2018, 195], [2018, 204], [2018, 208]]
                )
            elif index_gha == 14:
                bad = np.array([[2018, 185], [2018, 208]])
            elif index_gha in [15, 17]:
                bad = np.array([[2018, 185]])
            elif index_gha == 18:
                bad = np.array([[2018, 192]])
            elif index_gha == 20:
                bad = np.array([[2018, 185], [2018, 198], [2018, 216]])
            elif index_gha == 21:
                bad = np.array([[2018, 148], [2018, 160]])
            elif index_gha == 22:
                bad = np.array([[2018, 146]])
            elif index_gha == 23:
                bad = np.array(
                    [
                        [2018, 146],
                        [2018, 170],
                        [2018, 185],
                        [2018, 215],
                        [2018, 216],
                        [2018, 217],
                        [2018, 220],
                    ]
                )
            elif index_gha == 3:
                bad = np.array([[2018, 161]])
            elif index_gha == 4:
                bad = np.array([[2018, 204], [2018, 205]])
            elif index_gha == 5:
                bad = np.array([[2018, 167], [2018, 200]])
            elif index_gha == 7:
                bad = np.array([[2018, 146], [2018, 157], [2018, 209]])
            elif index_gha == 8:
                bad = np.array(
                    [[2018, 146], [2018, 152], [2018, 159], [2018, 162], [2018, 192]]
                )
            elif index_gha == 9:
                bad = np.array([[2018, 159], [2018, 196]])
        elif case == 2:
            if index_gha == 0:
                bad = np.array([[2018, 146], [2018, 220]])
            elif index_gha == 1:
                bad = np.array([[2018, 147], [2018, 180], [2018, 185]])
            elif index_gha == 10:
                bad = np.array([[2018, 159], [2018, 196], [2018, 199]])
            elif index_gha == 11:
                bad = np.array(
                    [[2018, 149], [2018, 152], [2018, 176], [2018, 204], [2018, 209]]
                )
            elif index_gha == 12:
                bad = np.array([[2018, 175], [2018, 195], [2018, 204], [2018, 216]])
            elif index_gha == 13:
                bad = np.array(
                    [[2018, 176], [2018, 185], [2018, 195], [2018, 204], [2018, 208]]
                )
            elif index_gha == 14:
                bad = np.array([[2018, 176], [2018, 185], [2018, 208]])
            elif index_gha in [15, 16]:
                bad = np.array([[2018, 185]])
            elif index_gha == 17:
                bad = np.array([[2018, 216]])
            elif index_gha == 18:
                bad = np.array([[2018, 146], [2018, 192], [2018, 196]])
            elif index_gha in [2, 3, 5, 6, 19, 20, 21, 22]:
                bad = np.array([])
            elif index_gha == 23:
                bad = np.array(
                    [
                        [2018, 147],
                        [2018, 152],
                        [2018, 160],
                        [2018, 182],
                        [2018, 184],
                        [2018, 185],
                        [2018, 186],
                        [2018, 197],
                        [2018, 220],
                    ]
                )
            elif index_gha == 4:
                bad = np.array([[2018, 204]])
            elif index_gha == 7:
                bad = np.array([[2018, 146], [2018, 215]])
            elif index_gha == 8:
                bad = np.array([[2018, 146], [2018, 159], [2018, 211], [2018, 212]])
            elif index_gha == 9:
                bad = np.array(
                    [
                        [2018, 146],
                        [2018, 147],
                        [2018, 159],
                        [2018, 190],
                        [2018, 192],
                        [2018, 193],
                        [2018, 212],
                    ]
                )
        else:
            raise ValueError("case must be 1 or 2")
    else:
        raise ValueError("band must be mid_band")

    return np.array(
        [
            not any(year_day[0] == b[0] and year_day[1] == b[1] for b in bad)
            for year_day in year_day_list
        ]
    )


def daily_strict_filter(band, year_day_list):
    if band == "mid_band":
        good = np.array(
            [
                [2018, 147],
                [2018, 148],
                [2018, 149],
                [2018, 150],
                [2018, 151],
                [2018, 152],
                [2018, 157],
                [2018, 160],
                [2018, 161],
                [2018, 162],
                [2018, 163],
                [2018, 164],
                [2018, 165],
                [2018, 166],
                [2018, 167],
                [2018, 170],
                [2018, 174],
                [2018, 175],
                [2018, 176],
                [2018, 177],
                [2018, 178],
                [2018, 179],
                [2018, 180],
                [2018, 181],
                [2018, 182],
            ]
        )
    else:
        raise ValueError("band must be mid_band")

    return np.array(
        [
            any(year_day[0] == b[0] and year_day[1] == b[1] for b in good)
            for year_day in year_day_list
        ]
    )


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


def one_hour_filter(band, case, year, day, gha):
    # TODO: refactor data out of package.
    if band == "low_band3":
        bad = np.array(
            [
                [2018, 266, 5],
                [2018, 266, 6],
                [2018, 266, 7],
                [2018, 266, 8],
                [2018, 266, 9],
                [2018, 266, 10],
                [2018, 266, 11],
                [2018, 266, 12],
                [2018, 266, 13],
                [2018, 266, 14],
                [2018, 266, 15],
                [2018, 266, 16],
                [2018, 266, 17],
                [2018, 267, 5],
                [2018, 267, 6],
                [2018, 267, 7],
                [2018, 267, 8],
                [2018, 267, 9],
                [2018, 267, 10],
                [2018, 267, 11],
                [2018, 267, 12],
                [2018, 267, 13],
                [2018, 267, 14],
                [2018, 267, 15],
                [2018, 267, 16],
                [2018, 267, 17],
                [2018, 268, 12],
                [2018, 268, 13],
                [2018, 268, 14],
                [2018, 268, 15],
                [2018, 268, 16],
                [2018, 268, 17],
                [2018, 271, 12],
                [2018, 271, 13],
                [2018, 275, 12],
                [2018, 275, 13],
                [2018, 275, 14],
                [2018, 276, 10],
                [2018, 281, 6],
                [2018, 281, 7],
                [2018, 281, 8],
                [2018, 303, 14],
                [2018, 303, 15],
                [2018, 303, 16],
                [2018, 306, 12],
                [2018, 306, 13],
                [2018, 306, 14],
                [2018, 307, 22],
                [2018, 308, 12],
                [2018, 321, 8],
                [2018, 321, 9],
                [2018, 321, 10],
                [2018, 321, 11],
                [2018, 321, 12],
                [2018, 321, 13],
                [2018, 321, 14],
                [2018, 321, 15],
                [2018, 321, 16],
                [2018, 321, 17],
                [2018, 326, 20],
                [2018, 331, 22],
                [2018, 337, 8],
                [2018, 337, 9],
                [2018, 337, 10],
                [2018, 337, 11],
                [2018, 337, 12],
                [2018, 337, 13],
                [2018, 337, 14],
                [2018, 337, 15],
                [2018, 337, 16],
                [2018, 337, 17],
                [2018, 338, 11],
                [2018, 338, 12],
                [2018, 338, 13],
                [2018, 338, 14],
                [2018, 338, 15],
                [2018, 338, 16],
                [2018, 338, 17],
                [2018, 338, 18],
                [2018, 339, 11],
                [2018, 339, 12],
                [2018, 339, 13],
                [2018, 339, 14],
                [2018, 339, 15],
                [2018, 339, 16],
                [2018, 339, 17],
                [2018, 339, 18],
                [2018, 339, 19],
                [2018, 339, 20],
                [2018, 341, 19],
                [2018, 342, 9],
                [2018, 342, 10],
                [2018, 342, 11],
                [2018, 344, 10],
                [2018, 344, 11],
                [2018, 344, 12],
                [2018, 344, 13],
                [2018, 344, 14],
                [2018, 344, 15],
                [2018, 344, 16],
                [2018, 344, 17],
                [2018, 344, 18],
                [2018, 344, 19],
                [2018, 344, 20],
                [2018, 345, 11],
                [2018, 345, 12],
                [2018, 345, 13],
                [2018, 345, 14],
                [2018, 345, 15],
                [2018, 345, 16],
                [2018, 345, 17],
                [2018, 345, 18],
                [2018, 345, 19],
                [2018, 346, 11],
                [2018, 346, 12],
                [2018, 346, 13],
                [2018, 346, 14],
                [2018, 346, 15],
                [2018, 346, 16],
                [2018, 348, 11],
                [2018, 348, 12],
                [2018, 348, 13],
                [2018, 348, 14],
                [2018, 348, 15],
                [2018, 348, 16],
                [2018, 348, 17],
                [2018, 351, 0],
                [2018, 351, 1],
                [2018, 351, 5],
                [2018, 351, 6],
            ]
        )
    elif band == "mid_band":
        if case in [0, 1, 2]:
            bad = np.array(
                [
                    [2018, 146, 6],
                    [2018, 146, 7],
                    [2018, 146, 8],
                    [2018, 146, 9],
                    [2018, 146, 10],
                    [2018, 151, 9],
                    [2018, 157, 7],
                    [2018, 159, 4],
                    [2018, 159, 7],
                    [2018, 159, 8],
                    [2018, 159, 9],
                    [2018, 164, 19],
                    [2018, 170, 16],
                    [2018, 170, 17],
                    [2018, 170, 23],
                    [2018, 184, 11],
                    [2018, 184, 12],
                    [2018, 184, 13],
                    [2018, 184, 14],
                    [2018, 184, 15],
                    [2018, 184, 16],
                    [2018, 184, 17],
                    [2018, 185, 0],
                    [2018, 185, 1],
                    [2018, 185, 2],
                    [2018, 185, 6],
                    [2018, 185, 7],
                    [2018, 185, 8],
                    [2018, 185, 13],
                    [2018, 185, 14],
                    [2018, 185, 15],
                    [2018, 186, 9],
                    [2018, 186, 10],
                    [2018, 186, 11],
                    [2018, 186, 12],
                    [2018, 186, 13],
                    [2018, 186, 14],
                    [2018, 186, 15],
                    [2018, 186, 16],
                    [2018, 186, 17],
                    [2018, 190, 11],
                    [2018, 190, 12],
                    [2018, 190, 13],
                    [2018, 190, 14],
                    [2018, 190, 15],
                    [2018, 190, 16],
                    [2018, 190, 17],
                    [2018, 191, 8],
                    [2018, 191, 9],
                    [2018, 191, 10],
                    [2018, 191, 11],
                    [2018, 191, 12],
                    [2018, 191, 13],
                    [2018, 191, 14],
                    [2018, 191, 15],
                    [2018, 191, 16],
                    [2018, 192, 10],
                    [2018, 192, 11],
                    [2018, 192, 12],
                    [2018, 192, 13],
                    [2018, 192, 14],
                    [2018, 192, 15],
                    [2018, 192, 16],
                    [2018, 192, 17],
                    [2018, 192, 18],
                    [2018, 192, 19],
                    [2018, 195, 6],
                    [2018, 195, 7],
                    [2018, 195, 8],
                    [2018, 195, 9],
                    [2018, 195, 10],
                    [2018, 195, 11],
                    [2018, 195, 12],
                    [2018, 195, 13],
                    [2018, 196, 8],
                    [2018, 196, 9],
                    [2018, 196, 10],
                    [2018, 196, 11],
                    [2018, 196, 12],
                    [2018, 196, 13],
                    [2018, 196, 14],
                    [2018, 196, 15],
                    [2018, 196, 16],
                    [2018, 196, 17],
                    [2018, 199, 14],
                    [2018, 204, 5],
                    [2018, 204, 6],
                    [2018, 204, 7],
                    [2018, 204, 8],
                    [2018, 204, 9],
                    [2018, 204, 10],
                    [2018, 204, 11],
                    [2018, 204, 12],
                    [2018, 208, 13],
                    [2018, 208, 14],
                    [2018, 208, 15],
                    [2018, 209, 12],
                    [2018, 209, 13],
                    [2018, 209, 14],
                    [2018, 209, 15],
                    [2018, 209, 16],
                    [2018, 209, 17],
                    [2018, 209, 18],
                    [2018, 211, 20],
                    [2018, 212, 17],
                    [2018, 213, 0],
                    [2018, 213, 1],
                    [2018, 213, 2],
                    [2018, 213, 3],
                    [2018, 213, 4],
                    [2018, 213, 5],
                    [2018, 213, 6],
                    [2018, 213, 7],
                    [2018, 213, 8],
                    [2018, 213, 9],
                    [2018, 213, 10],
                    [2018, 213, 11],
                    [2018, 213, 12],
                    [2018, 213, 13],
                    [2018, 213, 14],
                    [2018, 213, 15],
                    [2018, 213, 16],
                    [2018, 213, 17],
                    [2018, 213, 18],
                    [2018, 213, 19],
                    [2018, 213, 20],
                    [2018, 213, 21],
                    [2018, 213, 22],
                    [2018, 213, 23],
                    [2018, 214, 13],
                    [2018, 214, 14],
                    [2018, 214, 15],
                    [2018, 214, 16],
                    [2018, 214, 17],
                    [2018, 214, 18],
                    [2018, 215, 0],
                    [2018, 215, 1],
                    [2018, 215, 2],
                    [2018, 215, 3],
                    [2018, 215, 21],
                    [2018, 215, 22],
                    [2018, 215, 23],
                    [2018, 216, 9],
                    [2018, 216, 10],
                    [2018, 216, 11],
                    [2018, 216, 12],
                    [2018, 216, 13],
                    [2018, 216, 14],
                    [2018, 216, 15],
                    [2018, 219, 15],
                    [2018, 219, 16],
                    [2018, 219, 17],
                    [2018, 219, 18],
                    [2018, 219, 19],
                    [2018, 221, 0],
                    [2018, 221, 1],
                    [2018, 221, 2],
                    [2018, 221, 3],
                    [2018, 221, 4],
                    [2018, 221, 5],
                    [2018, 221, 6],
                    [2018, 221, 7],
                    [2018, 221, 8],
                    [2018, 221, 9],
                    [2018, 221, 10],
                    [2018, 221, 11],
                    [2018, 221, 12],
                    [2018, 221, 13],
                    [2018, 221, 14],
                    [2018, 221, 15],
                    [2018, 221, 16],
                    [2018, 221, 17],
                    [2018, 221, 18],
                    [2018, 221, 19],
                    [2018, 221, 20],
                    [2018, 221, 21],
                    [2018, 221, 22],
                    [2018, 221, 23],
                    [2018, 222, 0],
                    [2018, 222, 1],
                    [2018, 222, 2],
                    [2018, 222, 3],
                    [2018, 222, 4],
                    [2018, 222, 5],
                    [2018, 222, 6],
                    [2018, 222, 7],
                    [2018, 222, 8],
                    [2018, 222, 9],
                    [2018, 222, 10],
                    [2018, 222, 11],
                    [2018, 222, 12],
                    [2018, 222, 13],
                    [2018, 222, 14],
                    [2018, 222, 15],
                    [2018, 222, 16],
                    [2018, 222, 17],
                    [2018, 222, 18],
                    [2018, 222, 19],
                    [2018, 222, 20],
                    [2018, 222, 21],
                    [2018, 222, 22],
                    [2018, 222, 23],
                ]
            )

        elif case == 22:
            bad = np.array(
                [
                    [2018, 146, 6],
                    [2018, 146, 7],
                    [2018, 146, 8],
                    [2018, 146, 9],
                    [2018, 146, 10],
                    [2018, 151, 9],
                    [2018, 157, 7],
                    [2018, 159, 8],
                    [2018, 159, 9],
                    [2018, 159, 10],
                    [2018, 160, 23],
                    [2018, 175, 12],
                    [2018, 176, 11],
                    [2018, 176, 12],
                    [2018, 176, 13],
                    [2018, 176, 14],
                    [2018, 176, 23],
                    [2018, 184, 11],
                    [2018, 184, 12],
                    [2018, 184, 13],
                    [2018, 184, 14],
                    [2018, 184, 15],
                    [2018, 184, 16],
                    [2018, 184, 17],
                    [2018, 185, 0],
                    [2018, 185, 1],
                    [2018, 185, 2],
                    [2018, 185, 6],
                    [2018, 185, 7],
                    [2018, 185, 8],
                    [2018, 185, 13],
                    [2018, 185, 14],
                    [2018, 185, 15],
                    [2018, 185, 16],
                    [2018, 185, 23],
                    [2018, 186, 9],
                    [2018, 186, 10],
                    [2018, 186, 11],
                    [2018, 186, 12],
                    [2018, 186, 13],
                    [2018, 186, 14],
                    [2018, 186, 15],
                    [2018, 186, 16],
                    [2018, 186, 17],
                    [2018, 186, 18],
                    [2018, 192, 10],
                    [2018, 192, 11],
                    [2018, 192, 12],
                    [2018, 192, 13],
                    [2018, 192, 14],
                    [2018, 192, 15],
                    [2018, 192, 16],
                    [2018, 192, 17],
                    [2018, 192, 18],
                    [2018, 192, 19],
                    [2018, 195, 6],
                    [2018, 195, 7],
                    [2018, 195, 8],
                    [2018, 195, 9],
                    [2018, 195, 10],
                    [2018, 195, 11],
                    [2018, 195, 12],
                    [2018, 195, 13],
                    [2018, 195, 14],
                    [2018, 196, 0],
                    [2018, 196, 8],
                    [2018, 196, 9],
                    [2018, 196, 10],
                    [2018, 196, 11],
                    [2018, 196, 12],
                    [2018, 196, 13],
                    [2018, 196, 14],
                    [2018, 196, 15],
                    [2018, 196, 16],
                    [2018, 196, 17],
                    [2018, 196, 18],
                    [2018, 199, 14],
                    [2018, 199, 15],
                    [2018, 204, 0],
                    [2018, 204, 1],
                    [2018, 204, 2],
                    [2018, 204, 3],
                    [2018, 204, 4],
                    [2018, 204, 5],
                    [2018, 204, 6],
                    [2018, 204, 7],
                    [2018, 204, 8],
                    [2018, 204, 9],
                    [2018, 204, 10],
                    [2018, 204, 11],
                    [2018, 204, 12],
                    [2018, 204, 13],
                    [2018, 204, 14],
                    [2018, 204, 15],
                    [2018, 208, 0],
                    [2018, 208, 13],
                    [2018, 208, 14],
                    [2018, 209, 10],
                    [2018, 209, 11],
                    [2018, 211, 19],
                    [2018, 211, 20],
                    [2018, 214, 0],
                    [2018, 214, 1],
                    [2018, 214, 11],
                    [2018, 214, 12],
                    [2018, 214, 13],
                    [2018, 214, 14],
                    [2018, 214, 19],
                    [2018, 214, 20],
                    [2018, 214, 21],
                    [2018, 214, 22],
                    [2018, 214, 23],
                    [2018, 215, 7],
                    [2018, 216, 11],
                    [2018, 216, 12],
                    [2018, 216, 13],
                    [2018, 216, 17],
                    [2018, 217, 0],
                    [2018, 217, 1],
                    [2018, 219, 16],
                    [2018, 220, 13],
                ]
            )
        else:
            raise ValueError("wrong case")
    else:
        raise ValueError("band should be 'mid_band' or 'low_band3'")

    return np.array(
        [not any(y == b[0] and d == b[1] for b in bad) for y, d in zip(year, day)]
    )
