import numpy as np
import h5py
import matplotlib.pyplot as plt

from edges_cal import modelling as mdl
from . import tools

edges_folder = ""  # TODO: remove this


def auxiliary_data(weather_file, thermlog_file, band, year, day):
    # scp -P 64122 loco@150.101.175.77:/media/DATA/EDGES_data/weather.txt /home/raul/Desktop/
    # scp -P 64122 loco@150.101.175.77:/media/DATA/EDGES_data/thermlog.txt /home/raul/Desktop/

    # OR

    # scp raul@enterprise.sese.asu.edu:/data1/edges/data/2014_February_Boolardy/weather.txt Desktop/
    # scp raul@enterprise.sese.asu.edu:/data1/edges/data/2014_February_Boolardy/thermlog_low.txt
    # Desktop/
    # scp raul@enterprise.sese.asu.edu:/data1/edges/data/2014_February_Boolardy/thermlog.txt
    # Desktop/

    # Gather data from 'weather.txt' file
    f1 = open(weather_file, "r")
    lines_all_1 = f1.readlines()

    array1 = np.zeros((0, 4))

    if year == 2015:
        i1 = 92000  # ~ day 100
    elif year == 2016:
        i1 = 165097  # start of year 2016
    elif (year == 2017) and (day < 330):
        i1 = 261356  # start of year 2017
    elif (year == 2017) and (day > 331):
        i1 = 0  # start of year 2017 in file weather2.txt
    elif year == 2018:
        i1 = 9806  # start of year in file weather2.txt

    line1 = lines_all_1[i1]
    year_iter_1 = int(line1[0:4])
    day_of_year_1 = int(line1[5:8])

    while (day_of_year_1 <= day) and (year_iter_1 <= year):

        if day_of_year_1 == day:
            # print(line1[0:17] + ' ' + line1[59:65] + ' ' + line1[88:93] + ' ' + line1[113:119])

            date_time = line1[0:17]
            ttt = date_time.split(":")
            seconds = 3600 * int(ttt[2]) + 60 * int(ttt[3]) + int(ttt[4])

            try:
                amb_temp = float(line1[59:65])
            except ValueError:
                amb_temp = 0

            try:
                amb_hum = float(line1[87:93])
            except ValueError:
                amb_hum = 0

            try:
                rec_temp = float(line1[113:119])
            except ValueError:
                rec_temp = 0

            array1_temp1 = np.array([seconds, amb_temp, amb_hum, rec_temp])
            array1_temp2 = array1_temp1.reshape((1, -1))
            array1 = np.append(array1, array1_temp2, axis=0)

            print("weather time: " + date_time)

        i1 = i1 + 1
        print(i1)
        if i1 not in [28394, 1768]:
            line1 = lines_all_1[i1]
            year_iter_1 = int(line1[0:4])
            day_of_year_1 = int(line1[5:8])

    # gather data from 'thermlog.txt' file
    f2 = open(thermlog_file, "r")
    lines_all_2 = f2.readlines()

    array2 = np.zeros((0, 2))

    if (band == "high_band") and (year == 2015):
        i2 = 24000  # ~ day 108

    elif (band == "high_band") and (year == 2016):
        i2 = 58702  # beginning of year 2016

    elif (band == "low_band") and (year == 2015):
        i2 = 0

    elif (band == "low_band") and (year == 2016):
        i2 = 14920  # beginning of year 2016

    elif (band == "low_band") and (year == 2017):
        i2 = 59352  # beginning of year 2017

    elif (band == "low_band2") and (year == 2017) and (day < 332):
        return array1, np.array([0])

    elif band == "low_band2" and year == 2017:
        i2 = 0

    elif (band == "low_band2") and (year == 2018):
        i2 = 4768

    elif (band == "low_band3") and (year == 2018):
        i2 = 0  # 33826

    elif (band == "mid_band") and (year == 2018) and (day <= 171):
        i2 = 5624  # beginning of year 2018, file "thermlog_mid.txt"

    elif (band == "mid_band") and (year == 2018) and (day >= 172):
        i2 = 16154

    line2 = lines_all_2[i2]
    year_iter_2 = int(line2[0:4])
    day_of_year_2 = int(line2[5:8])

    while (day_of_year_2 <= day) and (year_iter_2 <= year):

        if day_of_year_2 == day:
            # print(line2[0:17] + ' ' + line2[48:53])

            date_time = line2[0:17]
            ttt = date_time.split(":")
            seconds = 3600 * int(ttt[2]) + 60 * int(ttt[3]) + int(ttt[4])

            try:
                rec_temp = float(line2[48:53])
            except ValueError:
                rec_temp = 0

            array2_temp1 = np.array([seconds, rec_temp])
            array2_temp2 = array2_temp1.reshape((1, -1))
            array2 = np.append(array2, array2_temp2, axis=0)

            print("receiver temperature time: " + date_time)

        i2 = i2 + 1
        if i2 != 26348:
            line2 = lines_all_2[i2]
            year_iter_2 = int(line2[0:4])
            day_of_year_2 = int(line2[5:8])

    return array1, array2


def level2read(path_file, print_key="no"):
    # path_file = home_folder + '/EDGES/spectra/level2/mid_band/file.hdf5'

    # Show keys (array names inside HDF5 file)
    with h5py.File(path_file, "r") as hf:
        if print_key == "yes":
            print([key for key in hf.keys()])

        hf_freq = hf.get("frequency")
        freq = np.array(hf_freq)

        hf_Ta = hf.get("antenna_temperature")
        Ta = np.array(hf_Ta)

        hf_meta = hf.get("metadata")
        meta = np.array(hf_meta)

        hf_weights = hf.get("weights")
        weights = np.array(hf_weights)

    return freq, Ta, meta, weights


def level3read(path_file, print_key="no"):
    # path_file = home_folder + '/EDGES/spectra/level3/mid_band/nominal_60_160MHz/file.hdf5'

    # Show keys (array names inside HDF5 file)
    with h5py.File(path_file, "r") as hf:
        if print_key == "yes":
            print([key for key in hf.keys()])

        hfX = hf.get("frequency")
        f = np.array(hfX)

        hfX = hf.get("antenna_temperature")
        t = np.array(hfX)

        hfX = hf.get("parameters")
        p = np.array(hfX)

        hfX = hf.get("residuals")
        r = np.array(hfX)

        hfX = hf.get("weights")
        w = np.array(hfX)

        hfX = hf.get("rms")
        rms = np.array(hfX)

        hfX = hf.get("total_power")
        tp = np.array(hfX)

        hfX = hf.get("metadata")
        m = np.array(hfX)

    return f, t, p, r, w, rms, tp, m


def level3_single_file_test(
    path_file,
    GHA_1,
    GHA_2,
    FLOW,
    FHIGH,
    plot_residuals_yes_no,
    model_type,
    Nfg,
    save_yes_no,
    save_spectrum_name,
):
    f, t, p, r, w, rms, tp, m = level3read(path_file)

    gha = m[:, 4]
    gha[gha < 0] = gha[gha < 0] + 24

    if GHA_2 > GHA_1:
        avr, avw = tools.spectral_averaging(
            r[(gha >= GHA_1) & (gha <= GHA_2), :], w[(gha >= GHA_1) & (gha <= GHA_2), :]
        )
        avp = np.mean(p[(gha >= GHA_1) & (gha <= GHA_2), :], axis=0)

    if GHA_2 < GHA_1:
        avr, avw = tools.spectral_averaging(
            r[(gha >= GHA_1) | (gha <= GHA_2), :], w[(gha >= GHA_1) | (gha <= GHA_2), :]
        )
        avp = np.mean(p[(gha >= GHA_1) | (gha <= GHA_2), :], axis=0)

    fb, rb, wb, sb = tools.spectral_binning_number_of_samples(f, avr, avw, nsamples=128)

    mb = mdl.model_evaluate("LINLOG", avp, fb / 200)

    tb = mb + rb

    ff = fb[(fb >= FLOW) & (fb <= FHIGH)]
    tt = tb[(fb >= FLOW) & (fb <= FHIGH)]
    ww = wb[(fb >= FLOW) & (fb <= FHIGH)]
    ss = sb[(fb >= FLOW) & (fb <= FHIGH)]

    if plot_residuals_yes_no == "yes":
        if model_type == "LINLOG":
            pp = mdl.fit_polynomial_fourier(
                "LINLOG", ff, tt, Nfg, Weights=1 / (ss ** 2)
            )
            model = pp[1]
        elif model_type == "LOGLOG":
            pp = np.polyfit(np.log(ff), np.log(tt), Nfg - 1)
            log_model = np.polyval(pp, np.log(ff))
            model = np.exp(log_model)

        plt.figure()
        plt.plot(ff[ww > 0], (tt - model)[ww > 0])
        plt.ylim([-1, 1])
        plt.xlim([60, 120])

    if save_yes_no == "yes":
        outT = np.array([ff, tt, ww, ss])
        out = outT.T

        save_path = edges_folder + "mid_band/spectra/level5/one_day_tests/"
        np.savetxt(save_path + save_spectrum_name, out)

    return ff, tt, ww, ss


def data_selection(
    m,
    GHA_or_LST="GHA",
    TIME_1=0,
    TIME_2=24,
    sun_el_max=90,
    moon_el_max=90,
    amb_hum_max=200,
    min_receiver_temp=0,
    max_receiver_temp=100,
):
    # Master index
    index = np.arange(len(m[:, 0]))

    if GHA_or_LST == "GHA":
        GHA = m[:, 4]
        GHA[GHA < 0] = GHA[GHA < 0] + 24

        index_TIME_1 = index[GHA >= TIME_1]
        index_TIME_2 = index[GHA < TIME_2]
    elif GHA_or_LST == "LST":
        index_TIME_1 = index[m[:, 3] >= TIME_1]
        index_TIME_2 = index[m[:, 3] < TIME_2]
    # Sun elevation, Moon elevation, ambient humidity, and receiver temperature
    index_SUN = index[m[:, 6] <= sun_el_max]
    index_MOON = index[m[:, 8] <= moon_el_max]
    index_HUM = index[m[:, 10] <= amb_hum_max]
    index_Trec = index[
        (m[:, 11] >= min_receiver_temp) & (m[:, 11] <= max_receiver_temp)
    ]  #
    # Combined index
    if TIME_1 < TIME_2:
        index1 = np.intersect1d(index_TIME_1, index_TIME_2)

    elif TIME_1 > TIME_2:
        index1 = np.union1d(index_TIME_1, index_TIME_2)

    index2 = np.intersect1d(index_SUN, index_MOON)
    index3 = np.intersect1d(index2, index_HUM)
    index4 = np.intersect1d(index3, index_Trec)
    return np.intersect1d(index1, index4)


def level4read(path_file):
    with h5py.File(path_file, "r") as hf:
        hfX = hf.get("frequency")
        f = np.array(hfX)

        hfX = hf.get("parameters")
        p_all = np.array(hfX)

        hfX = hf.get("residuals")
        r_all = np.array(hfX)

        hfX = hf.get("weights")
        w_all = np.array(hfX)

        hfX = hf.get("index")
        index = np.array(hfX)

        hfX = hf.get("gha_edges")
        gha = np.array(hfX)

        hfX = hf.get("year_day")
        yd = np.array(hfX)

    return f, p_all, r_all, w_all, index, gha, yd


def level4_binned_read(path_file):
    with h5py.File(path_file, "r") as hf:
        hfX = hf.get("frequency")
        fb = np.array(hfX)

        hfX = hf.get("residuals")
        rb = np.array(hfX)

        hfX = hf.get("weights")
        wb = np.array(hfX)

        hfX = hf.get("stddev")
        sb = np.array(hfX)

        hfX = hf.get("gha_edges")
        gha = np.array(hfX)

        hfX = hf.get("year_day")
        yd = np.array(hfX)

    return fb, rb, wb, sb, gha, yd


def level4_save_averaged_spectra(case, GHA_case, first_day, last_day):
    if case == 2:
        header_text = (
            "f [MHz], t_ant (GHA=0-23) [K], std (GHA=0-23) [K], Nsamples (GHA=0-23)"
        )
        file_path = (
            edges_folder
            + "mid_band/spectra/level4/calibration_2019_10_no_ground_loss_no_beam_corrections"
            "/binned_averages/"
        )
        file_name = "GHA_every_1hr.txt"
    elif case == 3:
        header_text = (
            "f [MHz], t_ant (GHA=0-23) [K], std (GHA=0-23) [K], Nsamples (GHA=0-23)"
        )
        file_path = (
            edges_folder + "mid_band/spectra/level4/case_nominal_50"
            "-150MHz_no_ground_loss_no_beam_corrections"
            "/binned_averages/"
        )
        file_name = "GHA_every_1hr.txt"
    elif case == 406:
        header_text = (
            "f [MHz], t_ant (GHA=0-23) [K], std (GHA=0-23) [K], Nsamples (GHA=0-23)"
        )
        file_path = (
            edges_folder
            + "mid_band/spectra/level4/case_nominal_50-150MHz_LNA1_a2_h2_o2_s1_sim2"
            "/binned_averages/"
        )
        file_name = "GHA_every_1hr.txt"
    elif case == 5:
        header_text = (
            "f [MHz], t_ant (GHA=0-23) [K], std (GHA=0-23) [K], Nsamples (GHA=0-23)"
        )
        file_path = (
            edges_folder + "mid_band/spectra/level4/case_nominal_14_14_terms_55"
            "-150MHz_no_ground_loss_no_beam_corrections/binned_averages/"
        )
        file_name = "GHA_every_1hr.txt"
    elif case == 501:
        header_text = (
            "f [MHz], t_ant (GHA=0-23) [K], std (GHA=0-23) [K], Nsamples (GHA=0-23)"
        )
        file_path = (
            edges_folder + "mid_band/spectra/level4/case_nominal_50"
            "-150MHz_LNA2_a2_h2_o2_s1_sim2_all_lc_yes_bc"
            "/binned_averages/"
        )
        file_name = "GHA_every_1hr.txt"
    start = 0

    if GHA_case == 24:
        for i in range(24):
            fb, tb, rb, wb, sb = tools.level4_integration(
                case, [i], first_day, last_day, 55, 150, 5
            )
            if start == 0:
                outT = np.zeros((1 + 3 * 24, len(fb)))
                outT[0, :] = fb
                start = 1

            outT[i + 1, :] = tb
            outT[i + 1 + 24, :] = sb
            outT[i + 1 + 48, :] = wb

    out = outT.T
    np.savetxt(file_path + file_name, out, header=header_text)

    return out


def level4_foreground_fits_read(path_file):
    with h5py.File(path_file, "r") as hf:
        hfX = hf.get("fref")
        fref = np.array(hfX)

        hfX = hf.get("fit2")
        fit2 = np.array(hfX)

        hfX = hf.get("fit3")
        fit3 = np.array(hfX)

        hfX = hf.get("fit4")
        fit4 = np.array(hfX)

        hfX = hf.get("fit5")
        fit5 = np.array(hfX)

    return fref, fit2, fit3, fit4, fit5


def calibration_RMS_read(path_file):

    with h5py.File(path_file, "r") as hf:

        hfX = hf.get("RMS")
        RMS = np.array(hfX)

        hfX = hf.get("index_cterms")
        cterms = np.array(hfX)

        hfX = hf.get("index_wterms")
        wterms = np.array(hfX)

    return RMS, cterms, wterms
