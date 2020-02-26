from os import makedirs, listdir
from os.path import exists

import h5py
import numpy as np
from edges_cal import (
    EdgesFrequencyRange,
    receiver_calibration_func as rcf,
    modelling as mdl,
)
from edges_io.io import Spectrum

# import src.edges_analysis
from . import io, s11 as s11m, loss, beams, rfi, tools, filters, coordinates
from ..config import config


def level1_to_level2(band, year, day_hour, band_flag=None):
    # Paths and files
    path_level1 = config["home_folder"] + f"/EDGES/spectra/level1/{band}/300_350/"
    path_logs = config["MRO_folder"]
    save_file = (
        config["home_folder"] + f"/EDGES/spectra/level2/{band}/{year}_{day_hour}.hdf5"
    )

    band_flag = band_flag or band.replace("_band", "")
    level1_file = path_level1 + f"level1_{year}_{day_hour}_{band_flag}_300_350.mat"

    if (int(year) < 2017) or ((int(year) == 2017) and (int(day_hour[0:3]) < 330)):
        weather_file = path_logs + "/weather_upto_20171125.txt"

    if (int(year) == 2018) or ((int(year) == 2017) and (int(day_hour[0:3]) > 331)):
        weather_file = path_logs + "/weather2.txt"

    thermlog_file = (
        path_logs + f"/thermlog{'_' + band_flag if band != 'high_band' else ''}"
    )

    # Loading data

    # Frequency and indices
    if band == "high_band":
        f_low = 65
        f_high = 195
    else:
        f_low = 50
        f_high = 199

    freq = EdgesFrequencyRange(f_low=f_low, f_high=f_high)
    fe = freq.freq

    ds, dd = Spectrum._read_mat(level1_file)
    tt = ds[:, freq.mask]
    ww = np.ones((len(tt[:, 0]), len(tt[0, :])))

    # ------------ Meta -------------#
    # Seconds into measurement
    seconds_data = (
        3600 * dd[:, 3].astype(float)
        + 60 * dd[:, 4].astype(float)
        + dd[:, 5].astype(float)
    )

    # EDGES coordinates
    EDGES_LAT = -26.7
    EDGES_LON = 116.6

    # LST
    LST = coordinates.utc2lst(dd, EDGES_LON)
    LST_column = LST.reshape(-1, 1)

    # Year and day
    year = int(year)
    day = int(day_hour[0:3])

    year_column = year * np.ones((len(LST), 1))
    day_column = day * np.ones((len(LST), 1))

    if len(day_hour) > 3:
        fraction_int = int(day_hour[4::])
    elif len(day_hour) == 3:
        fraction_int = 0

    fraction_column = fraction_int * np.ones((len(LST), 1))

    # Galactic Hour Angle
    LST_gc = 17 + (45 / 60) + (40.04 / (60 * 60))  # LST of Galactic Center
    GHA = LST - LST_gc
    for i in range(len(GHA)):
        if GHA[i] < -12.0:
            GHA[i] = GHA[i] + 24
    GHA_column = GHA.reshape(-1, 1)

    sun_moon_azel = coordinates.sun_moon_azel(EDGES_LAT, EDGES_LON, dd)

    # Sun/Moon coordinates
    aux1, aux2 = io.auxiliary_data(weather_file, thermlog_file, band, year, day)

    amb_temp_interp = np.interp(seconds_data, aux1[:, 0], aux1[:, 1]) - 273.15
    amb_hum_interp = np.interp(seconds_data, aux1[:, 0], aux1[:, 2])
    rec1_temp_interp = np.interp(seconds_data, aux1[:, 0], aux1[:, 3]) - 273.15

    if len(aux2) == 1:
        rec2_temp_interp = 25 * np.ones(len(seconds_data))
    else:
        rec2_temp_interp = np.interp(seconds_data, aux2[:, 0], aux2[:, 1])

    amb_rec = np.array(
        [amb_temp_interp, amb_hum_interp, rec1_temp_interp, rec2_temp_interp]
    ).T

    # Meta
    meta = np.array(
        [
            year_column,
            day_column,
            fraction_column,
            LST_column,
            GHA_column,
            sun_moon_azel,
            amb_rec,
        ]
    )

    # Save
    with h5py.File(save_file, "w") as hf:
        hf.create_dataset("frequency", data=fe)
        hf.create_dataset("antenna_temperature", data=tt)
        hf.create_dataset("weights", data=ww)
        hf.create_dataset("metadata", data=meta)

    return fe, tt, ww, meta


def level2_to_level3(
    band,
    year_day_hdf5,
    flag_folder="test",
    rcv_file="",
    s11_path="antenna_s11_2018_147_17_04_33.txt",
    antenna_s11_Nfit=15,
    antenna_correction=1,
    balun_correction=1,
    ground_correction=1,
    beam_correction=1,
    beam_correction_case=0,
    f_low=50,
    f_high=150,
    n_fg=7,
):
    fin = 0

    # Load daily data
    # ---------------
    path_data = config["edges_folder"] + band + "/spectra/level2/"
    filename = path_data + year_day_hdf5
    fin_X, t_2D_X, m_2D, w_2D_X = io.level2read(filename)

    # Continue if there are data available
    # ------------------------------------
    if np.sum(t_2D_X) > 0:

        # Cut the frequency range
        # -----------------------
        fin = fin_X[(fin_X >= f_low) & (fin_X <= f_high)]
        t_2D = t_2D_X[:, (fin_X >= f_low) & (fin_X <= f_high)]
        w_2D = w_2D_X[:, (fin_X >= f_low) & (fin_X <= f_high)]

        rcv = np.genfromtxt(rcv_file)

        fX = rcv[:, 0]
        rcv = rcv[(fX >= f_low) & (fX <= f_high)]
        s11_LNA = rcv[:, 1] + 1j * rcv[:, 2]
        C1, C2, TU, TC, TS = rcv.T[3:]

        # Antenna S11
        s11_ant = s11m.antenna_s11_remove_delay(
            s11_path, fin, delay_0=0.17, n_fit=antenna_s11_Nfit
        )

        # Calibrated antenna temperature with losses and beam chromaticity
        tc_with_loss_and_beam = rcf.calibrated_antenna_temperature(
            t_2D, s11_ant, s11_LNA, C1, C2, TU, TC, TS
        )

        # Antenna Loss (interface between panels and balun)
        Ga = 1
        if antenna_correction:
            Ga = loss.antenna_loss(band, fin)

        # Balun+Connector Loss
        Gbc = 1
        if balun_correction:
            Gb, Gc = loss.balun_and_connector_loss(band, fin, s11_ant)
            Gbc = Gb * Gc

        # Ground Loss
        Gg = 1
        if ground_correction:
            Gg = loss.ground_loss(band, fin)

        # Total loss
        G = Ga * Gbc * Gg

        # Remove loss
        Tamb_2D = np.reshape(273.15 + m_2D[:, 9], (-1, 1))
        G_2D = np.repeat(np.reshape(G, (1, -1)), len(m_2D[:, 9]), axis=0)
        tc_with_beam = (tc_with_loss_and_beam - Tamb_2D * (1 - G_2D)) / G_2D

        # Beam factor
        # -----------
        # No beam correction
        bf = 1
        if beam_correction:
            if band == "mid_band":
                if beam_correction_case == 0:
                    beam_factor_filename = "old_case.hdf5"
                elif beam_correction_case == 1:
                    beam_factor_filename = "new_case.hdf5"
                    print("NEW BEAM FACTOR !!!")
                else:
                    raise ValueError("beam_correction_case must be 0 or 1")
            elif band == "low_band3":
                beam_factor_filename = "table_hires_low_band3_50-120MHz_85deg_alan_haslam_2.5_2.62_reffreq_76MHz.hdf5"
            else:
                raise ValueError("band must be mid_band or low_band3")

            f_table, lst_table, bf_table = beams.beam_factor_table_read(
                f"{config['edges_folder']}{band}/calibration/beam_factors/table/{beam_factor_filename}"
            )
            bfX = beams.beam_factor_table_evaluate(
                f_table, lst_table, bf_table, m_2D[:, 3]
            )

            bf = bfX[:, ((f_table >= f_low) & (f_table <= f_high))]

        # Remove beam chromaticity
        tc = tc_with_beam / bf

        # RFI cleaning
        tt, ww = rfi.excision_raw_frequency(fin, tc, w_2D)

        # Number of spectra
        lt = len(tt)

        # Initializing output arrays
        t_all = np.random.rand(lt, len(fin))
        p_all = np.random.rand(lt, n_fg)
        r_all = np.random.rand(lt, len(fin))
        w_all = np.random.rand(lt, len(fin))
        rms_all = np.random.rand(lt, 3)

        # Foreground models and residuals
        for i, (ti, wi) in enumerate(tt, ww):

            # RFI cleaning
            tti, wwi = rfi.cleaning_polynomial(
                fin, ti, wi, Nterms_fg=n_fg, Nterms_std=5, Nstd=3.5
            )

            # Fitting foreground model to binned version of spectra
            Nsamples = 16  # 48.8 kHz
            fbi, tbi, wbi, sbi = tools.spectral_binning_number_of_samples(
                fin, tti, wwi, nsamples=Nsamples
            )
            par_fg = mdl.fit_polynomial_fourier(
                "LINLOG", fbi / 200, tbi, n_fg, Weights=wbi
            )

            # Evaluating foreground model at raw resolution
            model_i = mdl.model_evaluate("LINLOG", par_fg[0], fin / 200)

            # Residuals
            rri = tti - model_i

            # RMS for two halfs of the spectrum
            IX = int(np.floor(len(fin) / 2))

            F1 = fin[:IX]
            R1 = rri[:IX]
            W1 = wwi[:IX]

            F2 = fin[IX:]
            R2 = rri[IX:]
            W2 = wwi[IX:]

            RMS1 = np.sqrt(np.sum((R1[W1 > 0]) ** 2) / len(F1[W1 > 0]))
            RMS2 = np.sqrt(np.sum((R2[W2 > 0]) ** 2) / len(F2[W2 > 0]))

            # We also compute residuals for 3 terms as an additional filter
            # Fitting foreground model to binned version of spectra
            par_fg_Xt = mdl.fit_polynomial_fourier(
                "LINLOG", fbi / 200, tbi, 3, Weights=wbi
            )

            # Evaluating foreground model at raw resolution
            model_i_Xt = mdl.model_evaluate("LINLOG", par_fg_Xt[0], fin / 200)

            # Residuals
            rri_Xt = tti - model_i_Xt

            # RMS
            RMS3 = np.sqrt(np.sum((rri_Xt[wwi > 0]) ** 2) / len(fin[wwi > 0]))

            # Store
            # -----
            t_all[i] = tti
            p_all[i] = par_fg[0]
            r_all[i] = rri
            w_all[i] = wwi
            rms_all[i, 0] = RMS1
            rms_all[i, 1] = RMS2
            rms_all[i, 2] = RMS3

            print(
                f"{year_day_hdf5}: Spectrum number: {i+1}: RMS: {RMS1}, {RMS2}, {RMS3}"
            )

    # Total power computation
    t1 = t_all[:, (fin >= 60) & (fin <= 90)]
    t2 = t_all[:, (fin >= 90) & (fin <= 120)]
    t3 = t_all[:, (fin >= 60) & (fin <= 120)]

    tp1 = np.sum(t1, axis=1)
    tp2 = np.sum(t2, axis=1)
    tp3 = np.sum(t3, axis=1)

    tp_all = np.zeros((lt, 3))
    tp_all[:, 0] = tp1
    tp_all[:, 1] = tp2
    tp_all[:, 2] = tp3

    # Save
    if band == "mid_band":
        save_folder = (
            config["edges_folder"] + band + "/spectra/level3/" + flag_folder + "/"
        )
    elif band == "low_band3":
        save_folder = (
            f"/media/raul/EXTERNAL_2TB/low_band3/spectra/level3/{flag_folder}/"
        )
    if not exists(save_folder):
        makedirs(save_folder)

    with h5py.File(save_folder + year_day_hdf5, "w") as hf:
        hf.create_dataset("frequency", data=fin)
        hf.create_dataset("antenna_temperature", data=t_all)
        hf.create_dataset("parameters", data=p_all)
        hf.create_dataset("residuals", data=r_all)
        hf.create_dataset("weights", data=w_all)
        hf.create_dataset("rms", data=rms_all)
        hf.create_dataset("total_power", data=tp_all)
        hf.create_dataset("metadata", data=m_2D)

    return fin, t_all, p_all, r_all, w_all, rms_all, m_2D


def level3_to_level4(
    band, case, GHA_edges, sun_el_max, moon_el_max, save_folder_file_name
):
    """
    For instance: One-hour bins -> GHA_edges = np.arange(0, 25, 1)

    or

    GHA_edges = np.arange(0.5, 24, 1)
    GHA_edges = np.insert(GHA_edges,0,23.5)
    """

    # Listing files available
    # ------------------------
    if band == "mid_band":

        # Case 1 calibration: Receiver 2018, Switch 2018
        if (case >= 10) and (case <= 19):
            if case == 10:
                path_files = (
                    config["edges_folder"]
                    + "mid_band/spectra/level3/rcv18_sw18_nominal/"
                )

        # Case 2 calibration: Receiver 2018, Switch 2019
        if (case >= 20) and (case <= 29):
            if case == 20:
                path_files = (
                    config["edges_folder"]
                    + "mid_band/spectra/level3/rcv18_sw19_nominal/"
                )

        # Receiver and switch calibration 2019-10
        if case == 2:
            path_files = (
                config["edges_folder"]
                + "mid_band/spectra/level3/calibration_2019_10_no_ground_loss_no_beam_corrections/"
            )

        # Case 1 calibration: Receiver 2018, Switch 2018, AGAIN
        if case == 3:
            path_files = (
                config["edges_folder"]
                + "mid_band/spectra/level3/case_nominal_50-150MHz_no_ground_loss_no_beam_corrections/"
            )

        # Case 1 calibration: Receiver 2018, Switch 2018, AGAIN
        if case == 5:
            path_files = (
                config["edges_folder"]
                + "mid_band/spectra/level3/case_nominal_14_14_terms_55"
                "-150MHz_no_ground_loss_no_beam_corrections/"
            )

        # Calibration: Receiver 2018, Switch 2018, AGAIN, LNA1
        if case == 406:
            path_files = (
                config["edges_folder"]
                + "mid_band/spectra/level3/case_nominal_50-150MHz_LNA1_a2_h2_o2_s1_sim2/"
            )

        # Calibration: Receiver 2018, Switch 2018, all corrections
        if case == 501:
            path_files = (
                config["edges_folder"]
                + "mid_band/spectra/level3/case_nominal_50-150MHz_LNA2_a2_h2_o2_s1_sim2_all_lc_yes_bc/"
            )

        save_folder = (
            config["edges_folder"]
            + "mid_band/spectra/level4/"
            + save_folder_file_name
            + "/"
        )
        output_file_name_hdf5 = save_folder_file_name + ".hdf5"

    if band == "low_band3":

        if case == 2:
            path_files = "/media/raul/EXTERNAL_2TB/low_band3/spectra/level3/case2/"
            save_folder = config["edges_folder"] + "low_band3/spectra/level4/case2/"
            output_file_name_hdf5 = "case2.hdf5"

    new_list = listdir(path_files)
    new_list.sort()

    index_new_list = range(len(new_list))

    # Loading and cleaning data
    # -------------------------
    flag = -1

    year_day_all = np.zeros((len(index_new_list), 2))

    for i in index_new_list:  # range(4):  #

        # Storing year and day of each file
        year_day_all[i, 0] = float(new_list[i][0:4])

        if len(new_list[i]) == 8:
            year_day_all[i, 1] = float(new_list[i][5::])
        elif len(new_list[i]) > 8:
            year_day_all[i, 1] = float(new_list[i][5:8])

        flag = flag + 1

        # Loading data
        f, ty, py, ry, wy, rmsy, tpy, my = io.level3read(path_files + new_list[i])
        print("----------------------------------------------")

        # Daily index
        daily_index1 = np.arange(len(f))

        # Filtering out high humidity
        amb_hum_max = 40
        IX = io.data_selection(
            my,
            use_gha="GHA",
            time_1=0,
            time_2=24,
            sun_el_max=sun_el_max,
            moon_el_max=moon_el_max,
            amb_hum_max=amb_hum_max,
            min_receiver_temp=0,
            max_receiver_temp=100,
        )

        px = py[IX, :]
        rx = ry[IX, :]
        wx = wy[IX, :]
        rmsx = rmsy[IX, :]
        tpx = tpy[IX, :]
        mx = my[IX, :]
        daily_index2 = daily_index1[IX]
        # master_index[i, IX] = 1

        # Finding index of clean data
        gx = np.copy(mx[:, 4])
        gx[gx < 0] = gx[gx < 0] + 24

        Nsigma = 3
        index_good_rms, i1, i2, i3 = filters.rms_filter(band, case, gx, rmsx, Nsigma)

        # Applying total-power filter
        index_good_total_power, i1, i2, i3 = filters.tp_filter(gx, tpx)

        # Combined filters
        index_good = np.intersect1d(index_good_rms, index_good_total_power)

        # Selecting good data
        p = px[index_good, :]
        r = rx[index_good, :]
        w = wx[index_good, :]
        rms = rmsx[index_good, :]
        m = mx[index_good, :]
        daily_index3 = daily_index2[index_good]

        # Storing GHA and rms of good data
        GHA = m[:, 4]
        GHA[GHA < 0] = GHA[GHA < 0] + 24

        AT = np.vstack((gx, rmsx.T))
        BT = np.vstack((GHA, rms.T))

        A = AT.T
        B = BT.T

        if flag == 0:
            avp_all = np.zeros((len(new_list), len(GHA_edges) - 1, len(p[0, :])))
            avr_all = np.zeros((len(new_list), len(GHA_edges) - 1, len(r[0, :])))
            avw_all = np.zeros((len(new_list), len(GHA_edges) - 1, len(w[0, :])))

            # Creating master array of indices of good-quality spectra used in the final averages
            master_index = np.zeros((len(new_list), len(GHA_edges) - 1, 4000))

            grx_all = np.copy(A)
            gr_all = np.copy(B)

        if flag > 0:
            grx_all = np.vstack((grx_all, A))
            gr_all = np.vstack((gr_all, B))

        # Averaging data within each GHA bin
        for j in range(len(GHA_edges) - 1):

            GHA_LOW = GHA_edges[j]
            GHA_HIGH = GHA_edges[j + 1]

            if GHA_LOW < GHA_HIGH:
                p1 = p[(GHA >= GHA_LOW) & (GHA < GHA_HIGH), :]
                r1 = r[(GHA >= GHA_LOW) & (GHA < GHA_HIGH), :]
                w1 = w[(GHA >= GHA_LOW) & (GHA < GHA_HIGH), :]
                # m1 = m[(GHA >= GHA_LOW) & (GHA < GHA_HIGH), :]
                daily_index4 = daily_index3[(GHA >= GHA_LOW) & (GHA < GHA_HIGH)]

            elif GHA_LOW > GHA_HIGH:
                p1 = p[(GHA >= GHA_LOW) | (GHA < GHA_HIGH), :]
                r1 = r[(GHA >= GHA_LOW) | (GHA < GHA_HIGH), :]
                w1 = w[(GHA >= GHA_LOW) | (GHA < GHA_HIGH), :]
                # m1 = m[(GHA >= GHA_LOW) | (GHA < GHA_HIGH), :]
                daily_index4 = daily_index3[(GHA >= GHA_LOW) | (GHA < GHA_HIGH)]

            print(
                str(new_list[i])
                + ". GHA: "
                + str(GHA_LOW)
                + "-"
                + str(GHA_HIGH)
                + " hr. Number of spectra: "
                + str(len(r1))
            )

            if len(r1) > 0:
                avp = np.mean(p1, axis=0)
                avr, avw = tools.weighted_mean(r1, w1)

                # RFI cleaning of average spectra
                avr_no_rfi, avw_no_rfi = rfi.cleaning_sweep(
                    f, avr, avw, window_width=3, n_poly=2, n_bootstrap=20, n_sigma=2.5,
                )

                # Storing averages
                avp_all[i, j, :] = avp
                avr_all[i, j, :] = avr_no_rfi
                avw_all[i, j, :] = avw_no_rfi
                master_index[i, j, daily_index4] = 1

    print()
    print()

    # Save
    # ----
    if not exists(save_folder):
        makedirs(save_folder)
    with h5py.File(save_folder + output_file_name_hdf5, "w") as hf:
        hf.create_dataset("frequency", data=f)
        hf.create_dataset("parameters", data=avp_all)
        hf.create_dataset("residuals", data=avr_all)
        hf.create_dataset("weights", data=avw_all)
        hf.create_dataset("index", data=master_index)
        hf.create_dataset("gha_edges", data=GHA_edges)
        hf.create_dataset("year_day", data=year_day_all)

    return f, avp_all, avr_all, avw_all, master_index, GHA_edges, year_day_all
