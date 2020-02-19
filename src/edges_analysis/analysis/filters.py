import os

import numpy as np

from .io import data_selection, level3read

edges_folder = ""  # TODO: remove this


def rms_filter_computation(band, case, save_parameters=False):
    """

    Last modification:  2019-09-02

    Computation of the RMS filter

    """

    # Listing files available
    # ------------------------
    if band == "mid_band":

        if case == 10:
            path_files = edges_folder + "/mid_band/spectra/level3/rcv18_sw18_nominal/"
            save_folder = edges_folder + "/mid_band/rms_filters/rcv18_sw18_nominal/"

        if case == 20:
            path_files = edges_folder + "/mid_band/spectra/level3/rcv18_sw19_nominal/"
            save_folder = edges_folder + "/mid_band/rms_filters/rcv18_sw19_nominal/"

        if case == 2:
            path_files = (
                edges_folder
                + "/mid_band/spectra/level3/calibration_2019_10_no_ground_loss_no_beam_corrections/"
            )
            save_folder = (
                edges_folder
                + "/mid_band/rms_filters/calibration_2019_10_no_ground_loss_no_beam_corrections/"
            )

        if case == 3:
            path_files = (
                edges_folder + "/mid_band/spectra/level3/case_nominal_50"
                "-150MHz_no_ground_loss_no_beam_corrections/"
            )
            save_folder = (
                edges_folder
                + "/mid_band/rms_filters/case_nominal_50-150MHz_no_ground_loss_no_beam_corrections/"
            )

        if case == 5:
            path_files = (
                edges_folder + "/mid_band/spectra/level3/case_nominal_14_14_terms_55"
                "-150MHz_no_ground_loss_no_beam_corrections/"
            )
            save_folder = (
                edges_folder + "/mid_band/rms_filters/case_nominal_14_14_terms_55"
                "-150MHz_no_ground_loss_no_beam_corrections/"
            )

        if case == 501:
            path_files = (
                edges_folder + "/mid_band/spectra/level3/case_nominal_50"
                "-150MHz_LNA2_a2_h2_o2_s1_sim2_all_lc_yes_bc/"
            )
            save_folder = (
                edges_folder
                + "/mid_band/rms_filters/case_nominal_50-150MHz_LNA2_a2_h2_o2_s1_sim2_all_lc_yes_bc/"
            )

    if band == "low_band3":
        if case == 2:
            path_files = "/media/raul/EXTERNAL_2TB/low_band3/spectra/level3/case2/"
            save_folder = edges_folder + "/low_band3/rms_filters/case2/"

    new_list = os.listdir(path_files)
    new_list.sort()

    # Loading data used to compute filter
    # -----------------------------------
    N_files = 8  # Only using the first "N_files" to compute the filter
    for i in range(N_files):  #
        print(new_list[i])

        # Loading data
        f, t, p, r, w, rms, tp, m = level3read(path_files + new_list[i])

        # Filtering out high humidity
        amb_hum_max = 40
        IX = data_selection(
            m,
            GHA_or_LST="GHA",
            TIME_1=0,
            TIME_2=24,
            sun_el_max=90,
            moon_el_max=90,
            amb_hum_max=amb_hum_max,
            min_receiver_temp=0,
            max_receiver_temp=100,
        )

        px = p[IX, :]
        rx = r[IX, :]
        wx = w[IX, :]
        rmsx = rms[IX, :]
        mx = m[IX, :]

        # Accumulating data
        if i == 0:
            p_all = np.copy(px)
            r_all = np.copy(rx)
            w_all = np.copy(wx)
            rms_all = np.copy(rmsx)
            m_all = np.copy(mx)

        elif i > 0:
            p_all = np.vstack((p_all, px))
            r_all = np.vstack((r_all, rx))
            w_all = np.vstack((w_all, wx))
            rms_all = np.vstack((rms_all, rmsx))
            m_all = np.vstack((m_all, mx))

    # Columns necessary for analysis
    # ------------------------------
    GHA = m_all[:, 4]
    GHA[GHA < 0] = GHA[GHA < 0] + 24

    RMS1 = rms_all[:, 0]
    RMS2 = rms_all[:, 1]
    RMS3 = rms_all[:, 2]

    IN = np.arange(0, len(GHA))

    # Number of polynomial terms used to fit each 1-hour bins
    # and number of sigma threshold
    # -------------------------------------------------------
    Npoly = 3
    Nsigma = 3

    # Analysis for low-frequency half of the spectrum
    # -----------------------------------------------

    # Identification of bad data, within 1-hour "bins", across 24 hours
    # -----------------------------------------------------------------
    for i in range(24):
        GHA_x = GHA[(GHA >= i) & (GHA < (i + 1))]
        RMS_x = RMS1[(GHA >= i) & (GHA < (i + 1))]
        IN_x = IN[(GHA >= i) & (GHA < (i + 1))]

        W = np.ones(len(GHA_x))
        bad_old = -1
        bad = 0
        iteration = 0

        while bad > bad_old:
            iteration = iteration + 1

            print(" ")
            print("------------")
            print("GHA: " + str(i) + "-" + str(i + 1) + "hr")
            print("Iteration: " + str(iteration))

            par = np.polyfit(GHA_x[W > 0], RMS_x[W > 0], Npoly - 1)
            model = np.polyval(par, GHA_x)
            res = RMS_x - model
            std = np.std(res[W > 0])

            IN_x_bad = IN_x[np.abs(res) > Nsigma * std]
            W[np.abs(res) > Nsigma * std] = 0

            bad_old = np.copy(bad)
            bad = len(IN_x_bad)

            print("STD: " + str(np.round(std, 3)) + " K")
            print("Number of bad points excised: " + str(bad))

        # Indices of bad data points
        # --------------------------
        if i == 0:
            IN1_bad = np.copy(IN_x_bad)

        else:
            IN1_bad = np.append(IN1_bad, IN_x_bad)

    # Analysis for high-frequency half of the spectrum
    # ------------------------------------------------

    # Identification of bad data, within 1-hour "bins", across 24 hours
    # -----------------------------------------------------------------
    for i in range(24):
        GHA_x = GHA[(GHA >= i) & (GHA < (i + 1))]
        RMS_x = RMS2[(GHA >= i) & (GHA < (i + 1))]
        IN_x = IN[(GHA >= i) & (GHA < (i + 1))]

        W = np.ones(len(GHA_x))
        bad_old = -1
        bad = 0
        iteration = 0

        while bad > bad_old:
            iteration = iteration + 1

            print(" ")
            print("------------")
            print("GHA: " + str(i) + "-" + str(i + 1) + "hr")
            print("Iteration: " + str(iteration))

            par = np.polyfit(GHA_x[W > 0], RMS_x[W > 0], Npoly - 1)
            model = np.polyval(par, GHA_x)
            res = RMS_x - model
            std = np.std(res[W > 0])

            IN_x_bad = IN_x[np.abs(res) > Nsigma * std]
            W[np.abs(res) > Nsigma * std] = 0

            bad_old = np.copy(bad)
            bad = len(IN_x_bad)

            print("STD: " + str(np.round(std, 3)) + " K")
            print("Number of bad points excised: " + str(bad))

        # Indices of bad data points
        # --------------------------
        if i == 0:
            IN2_bad = np.copy(IN_x_bad)

        else:
            IN2_bad = np.append(IN2_bad, IN_x_bad)

    # Analysis for 3-term residuals
    # ------------------------------------------------

    # Identification of bad data, within 1-hour "bins", across 24 hours
    # -----------------------------------------------------------------
    for i in range(24):
        GHA_x = GHA[(GHA >= i) & (GHA < (i + 1))]
        RMS_x = RMS3[(GHA >= i) & (GHA < (i + 1))]
        IN_x = IN[(GHA >= i) & (GHA < (i + 1))]

        W = np.ones(len(GHA_x))
        bad_old = -1
        bad = 0
        iteration = 0

        while bad > bad_old:
            iteration = iteration + 1

            print(" ")
            print("------------")
            print("GHA: " + str(i) + "-" + str(i + 1) + "hr")
            print("Iteration: " + str(iteration))

            par = np.polyfit(GHA_x[W > 0], RMS_x[W > 0], Npoly - 1)
            model = np.polyval(par, GHA_x)
            res = RMS_x - model
            std = np.std(res[W > 0])

            IN_x_bad = IN_x[np.abs(res) > Nsigma * std]
            W[np.abs(res) > Nsigma * std] = 0

            bad_old = np.copy(bad)
            bad = len(IN_x_bad)

            print("STD: " + str(np.round(std, 3)) + " K")
            print("Number of bad points excised: " + str(bad))

        # Indices of bad data points
        # --------------------------
        if i == 0:
            IN3_bad = np.copy(IN_x_bad)

        else:
            IN3_bad = np.append(IN3_bad, IN_x_bad)

    # All bad/good spectra indices
    # ----------------------------
    # IN_bad = np.union1d(IN1_bad, IN2_bad)
    # IN_good  = np.setdiff1d(IN, IN_bad)

    # Indices of good spectra
    # -----------------------
    IN1_good = np.setdiff1d(IN, IN1_bad)
    IN2_good = np.setdiff1d(IN, IN2_bad)
    IN3_good = np.setdiff1d(IN, IN3_bad)

    # Number of terms for the polynomial fit of the RMS across 24 hours
    # and number of terms for the polynomial fit of the standard deviation across 24 hours
    # ------------------------------------------------------------------------------------
    Nterms = 16
    Nstd = 6

    # Parameters and models from the RMS and STD polynomial fits
    # ----------------------------------------------------------
    par1 = np.polyfit(GHA[IN1_good], RMS1[IN1_good], Nterms - 1)
    model1 = np.polyval(par1, GHA)
    abs_res1 = np.abs(RMS1 - model1)
    par1_std = np.polyfit(GHA[IN1_good], abs_res1[IN1_good], Nstd - 1)
    model1_std = np.polyval(par1_std, GHA)

    par2 = np.polyfit(GHA[IN2_good], RMS2[IN2_good], Nterms - 1)
    model2 = np.polyval(par2, GHA)
    abs_res2 = np.abs(RMS2 - model2)
    par2_std = np.polyfit(GHA[IN2_good], abs_res2[IN2_good], Nstd - 1)
    model2_std = np.polyval(par2_std, GHA)

    par3 = np.polyfit(GHA[IN3_good], RMS3[IN3_good], Nterms - 1)
    model3 = np.polyval(par3, GHA)
    abs_res3 = np.abs(RMS3 - model3)
    par3_std = np.polyfit(GHA[IN3_good], abs_res3[IN3_good], Nstd - 1)
    model3_std = np.polyval(par3_std, GHA)

    par = np.array([par1, par2, par3])
    par_std = np.array([par1_std, par2_std, par3_std])

    # Saving polynomial parameters
    # ----------------------------
    if save_parameters:
        np.savetxt(save_folder + "rms_polynomial_parameters.txt", par)
        np.savetxt(save_folder + "rms_std_polynomial_parameters.txt", par_std)

    return (
        GHA,
        RMS1,
        RMS2,
        RMS3,
        IN1_good,
        IN2_good,
        IN3_good,
        model1,
        model2,
        model3,
        abs_res1,
        abs_res2,
        abs_res3,
        model1_std,
        model2_std,
        model3_std,
    )


def rms_filter(band, case, gx, rms, Nsigma):
    # if (case == 0):
    # file_path = edges_folder + band + '/rms_filters/case_nominal/'

    if band == "mid_band":
        if (case >= 10) and (case <= 19):
            file_path = edges_folder + band + "/rms_filters/rcv18_sw18_nominal/"

        if (case >= 20) and (case <= 29):
            file_path = edges_folder + band + "/rms_filters/rcv18_sw19_nominal/"

        if case == 2:
            file_path = (
                edges_folder
                + band
                + "/rms_filters/calibration_2019_10_no_ground_loss_no_beam_corrections/"
            )

        if case in [3, 406]:
            file_path = (
                edges_folder
                + band
                + "/rms_filters/case_nominal_50-150MHz_no_ground_loss_no_beam_corrections/"
            )

        if case == 5:
            file_path = (
                edges_folder + band + "/rms_filters/case_nominal_14_14_terms_55"
                "-150MHz_no_ground_loss_no_beam_corrections/"
            )
        elif case == 501:
            file_path = (
                edges_folder
                + band
                + "/rms_filters/case_nominal_50-150MHz_LNA2_a2_h2_o2_s1_sim2_all_lc_yes_bc/"
            )
    p = np.genfromtxt(file_path + "rms_polynomial_parameters.txt")
    ps = np.genfromtxt(file_path + "rms_std_polynomial_parameters.txt")

    rms1 = rms[:, 0]
    rms2 = rms[:, 1]
    rms3 = rms[:, 2]

    m1 = np.polyval(p[0, :], gx)
    m2 = np.polyval(p[1, :], gx)
    m3 = np.polyval(p[2, :], gx)

    ms1 = np.polyval(ps[0, :], gx)
    ms2 = np.polyval(ps[1, :], gx)
    ms3 = np.polyval(ps[2, :], gx)

    index = np.arange(0, len(rms1))

    diff1 = np.abs(rms1 - m1)
    diff2 = np.abs(rms2 - m2)
    diff3 = np.abs(rms3 - m3)

    index_good_1 = index[diff1 <= Nsigma * ms1]
    index_good_2 = index[diff2 <= Nsigma * ms2]
    index_good_3 = index[diff3 <= Nsigma * ms3]

    index_good_A = np.intersect1d(index_good_1, index_good_2)
    index_good = np.intersect1d(index_good_A, index_good_3)

    return index_good, index_good_1, index_good_2, index_good_3


def tp_filter(band, GHA, tp):
    IN = np.arange(0, len(GHA))
    Npoly = 3
    Nsigma = 3

    for j in range(3):

        if j == 0:
            std_threshold = 5e4  # 60-90 MHz
        elif j == 1:
            std_threshold = 2e4  # 90-120 MHz
        elif j == 2:
            std_threshold = 1e5  # 60-120 MHz
        flag = 0
        for i in range(24):
            GHA_x = GHA[(GHA >= i) & (GHA < (i + 1))]
            tp_x = tp[
                (GHA >= i) & (GHA < (i + 1)), j
            ]  # one of the three tps (60-90, 90-120, 60-120 MHz)
            IN_x = IN[(GHA >= i) & (GHA < (i + 1))]

            # If enough data points available per hour
            lx = len(tp_x)
            if lx > 10:
                W = np.ones(len(GHA_x))
                bad_old = -1
                bad = 0
                iteration = 0

                while (bad > bad_old) and (bad < int(lx / 2)):

                    iteration += 1
                    print(" ")
                    print("------------")
                    print("GHA: " + str(i) + "-" + str(i + 1) + "hr")
                    print("Iteration: " + str(iteration))

                    par = np.polyfit(GHA_x[W > 0], tp_x[W > 0], Npoly - 1)
                    model = np.polyval(par, GHA_x)
                    res = tp_x - model
                    std = np.std(res[W > 0])

                    if std < std_threshold:
                        IN_x_bad = IN_x[(np.abs(res) > Nsigma * std)]
                        W[np.abs(res) > Nsigma * std] = 0

                    elif std > std_threshold:
                        IN_x_bad = IN_x[(np.abs(res) > 1 * std)]
                        W[np.abs(res) > 1 * std] = 0

                    bad_old = np.copy(bad)
                    bad = len(IN_x_bad)

                    if bad >= int(lx / 2):
                        IN_x_bad = np.copy(IN_x)
                        print(
                            "TANTANTANTANTANTANTANTANTANTANTANTANTANTANTANTANTANTANTANTANTANTANTANTAN"
                        )

                    print("STD: " + str(np.round(std, 3)) + " K")
                    print("Number of bad points excised: " + str(bad))

                # Indices of bad data points
                # --------------------------
                if flag == 0:
                    IN_bad = np.copy(IN_x_bad)
                    flag = 1

                elif flag == 1:
                    IN_bad = np.append(IN_bad, IN_x_bad)

        if j == 0:
            IN_bad1 = np.copy(IN_bad)
            IN_good1 = np.setdiff1d(IN, IN_bad1)

        elif j == 1:
            IN_bad2 = np.copy(IN_bad)
            IN_good2 = np.setdiff1d(IN, IN_bad2)

        elif j == 2:
            IN_bad3 = np.copy(IN_bad)
            IN_good3 = np.setdiff1d(IN, IN_bad3)

    # Combined index of good data points
    # ----------------------------------
    IG_A = np.intersect1d(IN_good1, IN_good2)
    IN_good = np.intersect1d(IG_A, IN_good3)

    return IN_good, IN_good1, IN_good2, IN_good3


def daily_nominal_filter(band, case, index_GHA, year_day_list):
    n_days = len(year_day_list)
    keep_all = np.zeros(n_days)

    # Choosing case
    if band == "mid_band":

        # Calibration from 2018
        if case == 101:
            if index_GHA == 0:
                bad0 = np.array([])

            if index_GHA == 1:
                bad0 = np.array([])

            if index_GHA == 2:
                bad0 = np.array([])

            if index_GHA == 3:
                bad0 = np.array([[2018, 161]])

            if index_GHA == 4:
                bad0 = np.array([[2018, 204], [2018, 205]])

            if index_GHA == 5:
                bad0 = np.array([[2018, 167], [2018, 200]])

            if index_GHA == 6:
                bad0 = np.array([])

            if index_GHA == 7:
                bad0 = np.array([[2018, 146], [2018, 157], [2018, 209]])

            if index_GHA == 8:
                bad0 = np.array(
                    [[2018, 146], [2018, 152], [2018, 159], [2018, 162], [2018, 192]]
                )

            if index_GHA == 9:
                bad0 = np.array([[2018, 159], [2018, 196]])

            if index_GHA == 10:
                bad0 = np.array(
                    [[2018, 176], [2018, 196], [2018, 201], [2018, 204], [2018, 218]]
                )

            if index_GHA == 11:
                bad0 = np.array([[2018, 149], [2018, 204], [2018, 216]])

            if index_GHA == 12:
                bad0 = np.array([[2018, 176], [2018, 195], [2018, 204]])

            if index_GHA == 13:
                bad0 = np.array(
                    [[2018, 176], [2018, 185], [2018, 195], [2018, 204], [2018, 208]]
                )

            if index_GHA == 14:
                bad0 = np.array([[2018, 185], [2018, 208]])

            if index_GHA == 15:
                bad0 = np.array([[2018, 185]])

            if index_GHA == 16:
                bad0 = np.array([])

            if index_GHA == 17:
                bad0 = np.array([[2018, 185]])

            if index_GHA == 18:
                bad0 = np.array([[2018, 192]])

            if index_GHA == 19:
                bad0 = np.array([])

            if index_GHA == 20:
                bad0 = np.array([[2018, 185], [2018, 198], [2018, 216]])

            if index_GHA == 21:
                bad0 = np.array([[2018, 148], [2018, 160]])

            if index_GHA == 22:
                bad0 = np.array([[2018, 146]])

            if index_GHA == 23:
                bad0 = np.array(
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

        # Calibration from 2019

        if case == 2:
            if index_GHA == 0:
                bad0 = np.array([[2018, 146], [2018, 220]])

            if index_GHA == 1:
                bad0 = np.array([[2018, 147], [2018, 180], [2018, 185]])

            if index_GHA == 2:
                bad0 = np.array([])

            if index_GHA == 3:
                bad0 = np.array([])

            if index_GHA == 4:
                bad0 = np.array([[2018, 204]])

            if index_GHA == 5:
                bad0 = np.array([])

            if index_GHA == 6:
                bad0 = np.array([])

            if index_GHA == 7:
                bad0 = np.array([[2018, 146], [2018, 215]])

            if index_GHA == 8:
                bad0 = np.array([[2018, 146], [2018, 159], [2018, 211], [2018, 212]])

            if index_GHA == 9:
                bad0 = np.array(
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

            if index_GHA == 10:
                bad0 = np.array([[2018, 159], [2018, 196], [2018, 199]])

            if index_GHA == 11:
                bad0 = np.array(
                    [[2018, 149], [2018, 152], [2018, 176], [2018, 204], [2018, 209]]
                )

            if index_GHA == 12:
                bad0 = np.array([[2018, 175], [2018, 195], [2018, 204], [2018, 216]])

            if index_GHA == 13:
                bad0 = np.array(
                    [[2018, 176], [2018, 185], [2018, 195], [2018, 204], [2018, 208]]
                )

            if index_GHA == 14:
                bad0 = np.array([[2018, 176], [2018, 185], [2018, 208]])

            if index_GHA == 15:
                bad0 = np.array([[2018, 185]])

            if index_GHA == 16:
                bad0 = np.array([[2018, 185]])

            if index_GHA == 17:
                bad0 = np.array([[2018, 216]])

            if index_GHA == 18:
                bad0 = np.array([[2018, 146], [2018, 192], [2018, 196]])

            if index_GHA == 19:
                bad0 = np.array([])

            if index_GHA == 20:
                bad0 = np.array([])

            if index_GHA == 21:
                bad0 = np.array([])

            if index_GHA == 22:
                bad0 = np.array([])

            if index_GHA == 23:
                bad0 = np.array(
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

    if band == "mid_band":
        bad = bad0.copy()

    for j in range(n_days):
        keep = 1
        for i in range(len(bad)):
            if (year_day_list[j, 0] == bad[i, 0]) and (
                year_day_list[j, 1] == bad[i, 1]
            ):
                keep = 0

        keep_all[j] = keep


def daily_strict_filter(band, year_day_list):
    n_days = len(year_day_list)
    keep_all = np.zeros(n_days)

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

    for j in range(n_days):
        keep = 0
        for i in range(len(good)):
            if (year_day_list[j, 0] == good[i, 0]) and (
                year_day_list[j, 1] == good[i, 1]
            ):
                keep = 1

        keep_all[j] = keep

    return keep_all


def daily_rms_filter(band, case, index_GHA, year_day_list, rms_threshold):
    n_days = len(year_day_list)
    keep_all = np.zeros(n_days)

    if band == "mid_band" and case == 1:
        d = np.genfromtxt(
            edges_folder
            + "mid_band/spectra/level4/case1/rms_filters/rms_index"
            + str(index_GHA)
            + ".txt"
        )
        rms_original = d[:, -1]
        good = d[rms_original <= rms_threshold, :]
        lg = len(good[:, 0])

    for j in range(n_days):
        keep = 0
        for i in range(lg):
            if (year_day_list[j, 0] == good[i, 0]) and (
                year_day_list[j, 1] == good[i, 1]
            ):
                keep = 1

        keep_all[j] = keep

    return keep_all


def one_hour_filter(band, case, year, day, gha):
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
        raise ValueError("should be mid band or lowband3")
    return not any(
        (year == bad[i, 0]) and (day == bad[i, 1]) and (gha == bad[i, 2])
        for i in range(len(bad))
    )
