from os import makedirs, listdir
from os.path import exists, dirname

import h5py
import numpy as np
from edges_cal import (
    FrequencyRange,
    receiver_calibration_func as rcf,
    modelling as mdl,
    xrfi as rfi,
    CalibrationObservation,
)
from edges_io.io import Spectrum
import attr
from pathlib import Path
from cached_property import cached_property

# import src.edges_analysis
from . import io, s11 as s11m, loss, beams, tools, filters, coordinates
from ..config import config
from .. import const
from datetime import datetime


@attr.s
class _Level(io.HDF5Object):
    _structure = {"spectra": None, "ancillary": None, "meta": None}
    _prev_level = None

    @classmethod
    def from_previous_level(cls, prev_level, filename=None, **kwargs):
        if not cls._prev_level:
            raise AttributeError(
                f"from_previous_level is not defined for {cls.__name__}"
            )

        if not isinstance(prev_level, cls._prev_level):
            prev_level = cls._prev_level(prev_level)

        data = cls._from_prev_level(prev_level, **kwargs)
        out = cls.from_data(data, filename=filename)

        if filename:
            out.write()
        return out

    @classmethod
    def _from_prev_level(cls, prev_level: [_prev_level], **kwargs):
        pass


@attr.s
class Level1(_Level):
    """Object representing the level-1 stage of processing.

    I think this should actually just be an edges_io.Spectrum.
    # TODO: add actual description.
    """

    @property
    def raw_time_data(self):
        return self.ancillary["times"]

    @cached_property
    def datetimes(self):
        return [datetime(*d) for d in self.raw_time_data]

    @cached_property
    def datetimes_np(self):
        return np.array(self.datetimes, dtype="datetime64[s]")


@attr.s
class Level2(_Level):
    @classmethod
    def _from_prev_level(
        cls, level1, weather_file, thermlog_file, band, year, day, hour=0
    ):
        data, ancillary, meta = level1_to_level2(
            level1, weather_file, thermlog_file, band, year, day, hour
        )
        return {"spectra": data, "ancillary": ancillary, "meta": meta}


def level1_to_level2(level1, weather_file, thermlog_file, band, year, day, hour=0):
    """
    Convert a Level 1 file to a Level 2 file.
    """
    # Paths and files
    # save_file = (
    #     config["home_folder"] + f"/EDGES/spectra/level2/{band}/{year}_{day_hour}.hdf5"
    # )

    if not Path(weather_file).is_absolute():
        weather_file = (Path(config["MRO_folder"]) / Path(weather_file)).absolute()

    if not Path(thermlog_file).is_absolute():
        thermlog_file = (Path(config["MRO_folder"]) / Path(thermlog_file)).absolute()

    if not isinstance(level1, Level1):
        level1 = Level1(level1)
    fe = level1.freq.freq

    tt = level1.spectrum[:, level1.freq.mask]
    ww = np.ones_like(tt)

    # LST
    lst = coordinates.utc2lst(level1.datetimes, const.edges_lon_deg)
    gha = lst - const.galactic_centre_lst
    gha[gha < -12.0] += 24

    # Sun/Moon coordinates
    sun_moon_azel = coordinates.sun_moon_azel(
        const.edges_lat_deg, const.edges_lon_deg, level1.datetimes
    )

    aux1, aux2 = io.auxiliary_data(weather_file, thermlog_file, band, year, day)
    seconds = (level1.datetimes_np - level1.datetimes_np[0]).astype(float)
    amb_temp_interp = np.interp(seconds, aux1[:, 0], aux1[:, 1]) - 273.15
    amb_hum_interp = np.interp(seconds, aux1[:, 0], aux1[:, 2])
    rec1_temp_interp = np.interp(seconds, aux1[:, 0], aux1[:, 3]) - 273.15

    if len(aux2) == 1:
        rec2_temp_interp = 25 * np.ones(len(seconds))
    else:
        rec2_temp_interp = np.interp(seconds, aux2[:, 0], aux2[:, 1])

    # Meta
    meta = {"year": year, "day": day, "hour": hour}

    ancilliary = {
        "ambient_temp": amb_temp_interp,
        "ambient_humidity": amb_hum_interp,
        "receiver1_temp": rec1_temp_interp,
        "recevier2_temp": rec2_temp_interp,
        "lst": lst,
        "gha": gha,
        "sun_moon_azel": sun_moon_azel,
    }

    data = {"frequency": fe, "antenna_temp": tt, "weights": ww}

    return data, ancilliary, meta


class Leve3(_Level):
    def _from_prev_level(cls, level2: [Level2], **kwargs):
        data, ancillary, meta = level2_to_level3(level2, **kwargs)
        return {"spectra": data, "ancillary": ancillary, "meta": meta}


def level2_to_level3(
    level2: [str, Path, Level2],
    calobs: [str, CalibrationObservation],
    band: str,
    s11_path="antenna_s11_2018_147_17_04_33.txt",
    antenna_s11_n_terms=15,
    antenna_correction=True,
    balun_correction=True,
    ground_correction=True,
    beam_file=None,
    f_low: float = 50,
    f_high: float = 150,
    n_fg=7,
) -> dict:
    if not isinstance(level2, Level2):
        level2 = Level2(level2)

    meta = {
        "band": band,
        "s11_path": Path(s11_path).absolute(),
        "antenna_s11_n_terms": antenna_s11_n_terms,
        "antenna_correction": antenna_correction,
        "balun_correction": balun_correction,
        "ground_correction": ground_correction,
        "beam_file": beam_file,
        "f_low": f_low,
        "f_high": f_high,
        "n_fg": n_fg,
    }

    meta = {**meta, **level2["meta"]}

    # Load daily data
    #    fin_X, t_2D_X, m_2D, w_2D_X = io.level2read(filename)

    if np.all(level2["spectra"] == 0):
        raise Exception("The level2 file given has no non-zero spectra!")

    # Cut the frequency range
    freq = FrequencyRange(level2["spectra"]["frequency"], f_low=f_low, f_high=f_high)
    # fin = fin_X[(fin_X >= f_low) & (fin_X <= f_high)]
    spectra = level2["spectra"]["temperature"][:, freq.mask]
    weights = level2["spectra"]["weights"][:, freq.mask]

    if not isinstance(calobs, CalibrationObservation):
        calobs = CalibrationObservation.from_file(calobs)

    #    rcv = np.genfromtxt(rcv_file)

    # fX = rcv[:, 0]
    # rcv = rcv[(fX >= f_low) & (fX <= f_high)]
    # s11_LNA = rcv[:, 1] + 1j * rcv[:, 2]
    # C1, C2, TU, TC, TS = rcv.T[3:]

    # Antenna S11
    s11_ant = s11m.antenna_s11_remove_delay(
        s11_path, freq.freq, delay_0=0.17, n_fit=antenna_s11_n_terms
    )

    # Calibrated antenna temperature with losses and beam chromaticity
    calibrated_temp = rcf.calibrated_antenna_temperature(
        spectra,
        s11_ant,
        calobs.lna.s11_model(freq.freq),
        calobs.C1(freq.freq),
        calobs.C2(freq.freq),
        calobs.Tunc(freq.freq),
        calobs.Tcos(freq.freq),
        calobs.Tsin(freq.freq),
    )

    # Antenna Loss (interface between panels and balun)
    G = np.ones_like(freq.freq)
    if antenna_correction:
        G *= loss.antenna_loss(band, freq.freq)

    # Balun+Connector Loss
    if balun_correction:
        Gb, Gc = loss.balun_and_connector_loss(band, freq.freq, s11_ant)
        G *= Gb * Gc

    # Ground Loss
    if ground_correction:
        G *= loss.ground_loss(band, freq.freq)

    # # Total loss
    # G = Ga * Gbc * Gg
    #
    # Remove loss
    # TODO: it is *not* clear which temperature to use here...
    ambient_temp = 273.15 + level2["ancillary"]["ambient_temp"]
    calibrated_temp = (calibrated_temp - ambient_temp * (1 - G)) / G

    # Beam factor
    if beam_file:
        if not Path(beam_file).exists():
            beam_file = f"{config['edges_folder']}{band}/calibration/beam_factors/table/{beam_file}"

        f_table, lst_table, bf_table = beams.beam_factor_table_read(beam_file)
        bf = beams.beam_factor_table_evaluate(
            f_table, lst_table, bf_table, level2["ancillary"]["lst"]
        )[:, ((f_table >= f_low) & (f_table <= f_high))]

        # Remove beam chromaticity
        calibrated_temp /= bf

    # RFI cleaning
    flags = rfi.excision_raw_frequency(
        freq.freq, rfi_file=dirname(__file__) + "data/known_rfi_channels.yaml"
    )

    calibrated_temp[flags] = 0
    weights[flags] = 0

    # Number of spectra
    #    lt = len(tt)

    # Initializing output arrays
    t_all = np.zeros((len(weights), freq.n))
    p_all = np.zeros((len(weights), n_fg))
    r_all = np.zeros((len(weights), freq.n))
    w_all = np.zeros((len(weights), freq.n))

    # Foreground models and residuals
    for i, (temp_cal, wi) in enumerate(calibrated_temp, weights):

        # RFI cleaning
        flags = rfi.xrfi_poly(
            temp_cal,
            wi,
            f_ratio=freq.max / freq.min,
            n_signal=n_fg,
            n_resid=5,
            n_abs_resid_threshold=3.5,
        )
        # TODO: this should probably be nan, not zero
        temp_cal[flags] = 0
        wi[flags] = 0

        # Fitting foreground model to binned version of spectra
        fbi, tbi, wbi, _ = tools.average_in_frequency(
            temp_cal, freq=freq.freq, weights=wi, resolution=0.0488  # in MHz
        )
        par_fg = mdl.fit_polynomial_fourier(
            "LINLOG", FrequencyRange(fbi).freq_recentred, tbi, n_fg, Weights=wbi
        )[0]

        # Evaluating foreground model at raw resolution
        model = mdl.model_evaluate("LINLOG", par_fg, freq.freq_recentred)

        # Residuals
        rri = temp_cal - model

        # # RMS for two halfs of the spectrum
        # IX = int(np.floor(len(fin) / 2))
        #
        # F1 = fin[:IX]
        # R1 = rri[:IX]
        # W1 = wi[:IX]
        #
        # F2 = fin[IX:]
        # R2 = rri[IX:]
        # W2 = wi[IX:]

        # RMS1 = np.sqrt(np.sum((R1[W1 > 0]) ** 2) / len(F1[W1 > 0]))
        # RMS2 = np.sqrt(np.sum((R2[W2 > 0]) ** 2) / len(F2[W2 > 0]))

        # We also compute residuals for 3 terms as an additional filter
        # Fitting foreground model to binned version of spectra
        #        par_fg_Xt = mdl.fit_polynomial_fourier(
        #            "LINLOG", freq.freq_recentred, tbi, 3, Weights=wbi
        #        )

        # Evaluating foreground model at raw resolution
        #       model_i_Xt = mdl.model_evaluate("LINLOG", par_fg_Xt[0], freq.freq_recentred)

        # Residuals
        #        rri_Xt = temp_cal - model_i_Xt

        # RMS
        # RMS3 = np.sqrt(np.sum((rri_Xt[wi > 0]) ** 2) / np.sum(wi > 0))

        # Store
        # -----
        t_all[i] = temp_cal
        p_all[i] = par_fg
        r_all[i] = rri
        w_all[i] = wi
        # rms_all[i, 0] = RMS1
        # rms_all[i, 1] = RMS2
        # rms_all[i, 2] = RMS3
        #
        # print(
        #     f"{year_day_hdf5}: Spectrum number: {i+1}: RMS: {RMS1}, {RMS2}, {RMS3}"
        # )
    #
    # # Total power computation
    # t1 = t_all[:, (fin >= 60) & (fin <= 90)]
    # t2 = t_all[:, (fin >= 90) & (fin <= 120)]
    # t3 = t_all[:, (fin >= 60) & (fin <= 120)]
    #
    # tp1 = np.sum(t1, axis=1)
    # tp2 = np.sum(t2, axis=1)
    # tp3 = np.sum(t3, axis=1)
    #
    # tp_all = np.zeros((lt, 3))
    # tp_all[:, 0] = tp1
    # tp_all[:, 1] = tp2
    # tp_all[:, 2] = tp3

    # Save
    # if band == "mid_band":
    #     save_folder = (
    #         config["edges_folder"] + band + "/spectra/level3/" + flag_folder + "/"
    #     )
    # elif band == "low_band3":
    #     save_folder = (
    #         f"/media/raul/EXTERNAL_2TB/low_band3/spectra/level3/{flag_folder}/"
    #     )
    # if not exists(save_folder):
    #     makedirs(save_folder)

    ancillary = {"parameters": p_all, "residuals": r_all}

    data = {
        "antenna_temp": t_all,
        "frequency": freq.freq,
        "weights": w_all,
    }
    #
    # with h5py.File(save_folder + year_day_hdf5, "w") as hf:
    #     hf.create_dataset("frequency", data=freq.freq)
    #     hf.create_dataset("antenna_temperature", data=t_all)
    #     hf.create_dataset("parameters", data=p_all)
    #     hf.create_dataset("residuals", data=r_all)
    #     hf.create_dataset("weights", data=w_all)
    #     # hf.create_dataset("rms", data=rms_all)
    #     # hf.create_dataset("total_power", data=tp_all)
    #     hf.create_dataset("metadata", data=m_2D)

    return data, ancillary, meta


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
        index_good_total_power, i1, i2, i3 = filters.total_power_filter(gx, tpx)

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

            if len(r1) > 0:
                avp = np.mean(p1, axis=0)
                avr, avw = tools.weighted_mean(r1, w1)

                # RFI cleaning of average spectra
                flags = rfi.xrfi_poly_filter(
                    avr,
                    avw,
                    window_width=int(3 / (f[1] - f[0])),
                    n_poly=2,
                    n_bootstrap=20,
                    n_sigma=2.5,
                )
                # Storing averages
                avp_all[i, j, :] = avp
                avr_all[i, j, flags] = 0
                avw_all[i, j, flags] = 0
                master_index[i, j, daily_index4] = 1

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
