from os import listdir
from os.path import dirname, join

import numpy as np
import yaml
from edges_cal import modelling as mdl
from edges_cal import xrfi as rfi
from matplotlib import pyplot as plt

from . import tools, io, filters
from .levels import level1_to_level2, level2_to_level3
from ..config import config


CALFILES = {
    1: "nominal/calibration_files/original/calibration_file_receiver1_cterms7_wterms7.txt",
    2: "nominal/calibration_files/original/calibration_file_receiver1_cterms7_wterms8.txt",
    3: "nominal/calibration_files/original/calibration_file_receiver1_cterms7_wterms15.txt",
    4: "nominal/calibration_files/original/calibration_file_receiver1_cterms8_wterms8.txt",
    5: "nominal/calibration_files/original/calibration_file_receiver1_cterms9_wterms9.txt",
    6: "nominal/calibration_files/original/calibration_file_receiver1_cterms10_wterms10.txt",
    7: "nominal_50-150MHz_no_rfi/calibration_files/calibration_file_receiver1_cterms9_wterms9_50-150MHz_no_rfi.txt",
    8: "nominal_50-150MHz_no_rfi/calibration_files/calibration_file_receiver1_cterms8_wterms8_50-150MHz_no_rfi.txt",
    30: "nominal_cleaned_60_120MHz/calibration_files/calibration_file_receiver1_cterms7_wterms6.txt",
    31: "nominal_cleaned_60_120MHz/calibration_files/calibration_file_receiver1_cterms7_wterms8.txt",
    32: "nominal_cleaned_60_120MHz/calibration_files/calibration_file_receiver1_cterms7_wterms9.txt",
    33: "nominal_cleaned_60_120MHz/calibration_files/calibration_file_receiver1_cterms6_wterms4.txt",
    34: "nominal_cleaned_60_120MHz/calibration_files/calibration_file_receiver1_cterms6_wterms8.txt",
    35: "nominal_cleaned_60_120MHz/calibration_files/calibration_file_receiver1_cterms6_wterms9.txt",
    39: "nominal_cleaned_60_120MHz/calibration_files/calibration_file_receiver1_cterms5_wterms9.txt",
    21: "nominal/calibration_files/calibration_file_receiver1_cterms7_wterms7.txt",
    22: "nominal/calibration_files/calibration_file_receiver1_cterms7_wterms8.txt",
    23: "nominal/calibration_files/calibration_file_receiver1_cterms7_wterms9.txt",
    24: "nominal/calibration_files/calibration_file_receiver1_cterms7_wterms10.txt",
    25: "nominal/calibration_files/calibration_file_receiver1_cterms8_wterms8.txt",
    26: "nominal/calibration_files/calibration_file_receiver1_cterms8_wterms11.txt",
    40: "nominal/calibration_files/60_85MHz/calibration_file_receiver1_60_85MHz_cterms4_wterms6.txt",
    88: "nominal/calibration_files/calibration_file_receiver1_50_150MHz_cterms8_wterms8.txt",
    810: "nominal/calibration_files/calibration_file_receiver1_50_150MHz_cterms8_wterms10.txt",
    811: "nominal/calibration_files/calibration_file_receiver1_50_150MHz_cterms8_wterms11.txt",
    100: "nominal/calibration_files/calibration_file_receiver1_50_190MHz_cterms10_wterms13.txt",
    200: "/nominal_2019_12_50-150MHz_try1/calibration_files/calibration_file_receiver1_55_150MHz_cterms14_wterms14.txt",
    201: "/nominal_2019_12_50-150MHz_try1/calibration_files/calibration_file_receiver1_50_150MHz_cterms7_wterms8.txt",
    202: "/nominal_2019_12_50-150MHz_try1/calibration_files/calibration_file_receiver1_50_150MHz_cterms7_wterms11.txt",
    203: "/nominal_2019_12_50-150MHz_try1_LNA_rep1/calibration_files/calibration_file_receiver1_50_150MHz_cterms7_wterms8.txt",
    204: "/nominal_2019_12_50-150MHz_try1_LNA_rep2/calibration_files/calibration_file_receiver1_50_150MHz_cterms7_wterms8.txt",
    205: "/nominal_2019_12_50-150MHz_try1_LNA_rep12/calibration_files/calibration_file_receiver1_50_150MHz_cterms7_wterms8.txt",
    301: "/nominal_2019_12_50-150MHz_try1/calibration_files/calibration_file_receiver1_60_120MHz_cterms7_wterms4.txt",
    302: "/nominal_2019_12_50-150MHz_try1/calibration_files/calibration_file_receiver1_60_120MHz_cterms7_wterms5.txt",
    303: "/nominal_2019_12_50-150MHz_try1/calibration_files/calibration_file_receiver1_60_120MHz_cterms7_wterms9.txt",
    304: "/nominal_2019_12_50-150MHz_try1/calibration_files/calibration_file_receiver1_60_120MHz_cterms8_wterms4.txt",
    305: "/nominal_2019_12_50-150MHz_try1/calibration_files/calibration_file_receiver1_60_120MHz_cterms8_wterms5.txt",
    306: "/nominal_2019_12_50-150MHz_try1/calibration_files/calibration_file_receiver1_60_120MHz_cterms8_wterms9.txt",
    307: "/nominal_2019_12_50-150MHz_try1/calibration_files/calibration_file_receiver1_60_120MHz_cterms5_wterms4.txt",
    308: "/nominal_2019_12_50-150MHz_try1/calibration_files/calibration_file_receiver1_60_120MHz_cterms6_wterms4.txt",
    401: "/nominal_2019_12_50-150MHz_LNA1_a1_h1_o1_s1_sim2/calibration_files/calibration_file_receiver1_50_150MHz_cterms7_wterms8.txt",
    402: "/nominal_2019_12_50-150MHz_LNA1_a1_h2_o1_s1_sim2/calibration_files/calibration_file_receiver1_50_150MHz_cterms7_wterms8.txt",
    403: "/nominal_2019_12_50-150MHz_LNA1_a2_h1_o1_s1_sim2/calibration_files/calibration_file_receiver1_50_150MHz_cterms7_wterms8.txt",
    404: "/nominal_2019_12_50-150MHz_LNA1_a2_h2_o1_s1_sim2/calibration_files/calibration_file_receiver1_50_150MHz_cterms7_wterms8.txt",
    405: "/nominal_2019_12_50-150MHz_LNA1_a2_h2_o1_s2_sim2/calibration_files/calibration_file_receiver1_50_150MHz_cterms7_wterms8.txt",
    406: "/nominal_2019_12_50-150MHz_LNA1_a2_h2_o2_s1_sim2/calibration_files/calibration_file_receiver1_50_150MHz_cterms7_wterms8.txt",
    407: "/nominal_2019_12_50-150MHz_LNA1_a2_h2_o2_s2_sim2/calibration_files/calibration_file_receiver1_50_150MHz_cterms7_wterms8.txt",
}


def integrated_half_hour_level4_many(band, case, GHA_starts=[(13, 1), (14, 0)]):
    # TODO: this should be a script.
    discard = {
        (6, 0): [146, 164, 167, 169],
        (6, 1): [146, 147, 174, 179, 181, 198, 211, 215],
        (7, 0): [146, 147, 157, 166],
        (7, 1): [146, 159],
        (8, 0): [146, 151, 159],
        (8, 1): [146, 159],
        (9, 0): [146, 151, 152, 157, 159, 163, 185],
        (9, 1): [
            146,
            157,
            159,
            167,
            196,
            ([149, 150, 152, 163], (104.5, 110)),
            ([150, 160, 161, 162, 166], (129, 135)),
            (None, (134.5, 140.5)),
        ],
        (10, 0): [152, 157, 166, 159, 196],
        (10, 1): [
            174,
            176,
            204,
            218,
            (None, 101.52, 101.53),
            (None, (102.5, 102.53)),
            (None, (153.02, 153.04)),
            (None, (111.7, 115.4)),
            (None, (121.47, 121.55)),
            (None, (146.5, 148)),
            (None, (150, 150.5)),
            (None, (105.72, 105.74)),
            (None, (106.05, 106.15)),
            (None, (106.42, 106.55)),
        ],
        (11, 0): [
            149,
            165,
            176,
            204,
            ([151, 161], (129, 135)),
            (None, (109, 114.2)),
            (None, (105.72, 105.74)),
            (None, (106.05, 106.15)),
            (None, (106.42, 106.55)),
            (None, (138, 138.4)),
        ],
        (11, 1): [175, 176, 177, 200, 204, 216, (149, (110, 118)), (None, (137, 140))],
        (12, 0): [146, 147, 170, 175, 176, 193, 195, 204, 205, 220],
        (12, 1): [146, 170, 176, 185, 195, 198, 204, 220],
        (13, 0): [152, 174, 176, 182, 185, 195, 204, 208, 214],
        (13, 1): [151, 163, 164, 176, 185, 187, 189, 192, 195, 200, 208, 215, 219],
        (14, 0): [176, 184, 185, 193, 199, 208, 210],
        (14, 1): [166, 174, 177, 185, 199, 200, 201, 208],
        (15, 0): [150, 157, 178, 182, 185, 187, 198, 208],
        (15, 1): [185],
        (16, 0): [184],
        (16, 1): [191, 192],
        (17, 0): [184, 186, 192, 216],
        (17, 1): [186, 192, 196],
        (18, 0): [192, 197],
        (18, 1): [192, 209, 211],
    }

    for i, GHA_start in enumerate(GHA_starts):
        discarded_days = discard[GHA_start]

        (
            fb,
            rb_all,
            wb_all,
            d_all,
            tbn,
            wbn,
            f,
            rr_all,
            wr_all,
            avr,
            avw,
            avp,
            avt,
        ) = tools.integrate_level4_half_hour(
            band, case, 140, 170, discarded_days, GHA_start=GHA_start
        )

        if i == 0:
            avr_all = np.copy(avr)
            avw_all = np.copy(avw)
            avp_all = np.copy(avp)

        else:
            avr_all = np.vstack((avr_all, avr))
            avw_all = np.vstack((avw_all, avw))
            avp_all = np.vstack((avp_all, avp))

    avr, avw = tools.spectral_averaging(avr_all, avw_all)
    avp = np.mean(avp_all, axis=0)
    avt = mdl.model_evaluate("LINLOG", avp, f / 200) + avr

    flags = rfi.xrfi_poly_filter(
        f, window_width=int(3 / (f[1] - f[0])), n_poly=2, n_bootstrap=20, n_sigma=2.5,
    )
    avrn = np.where(flags, 0, avr)
    avwn = np.where(flags, 0, avw)

    fb, rb, wbn = tools.average_in_frequency(avrn, f, avwn, n_samples=16)
    tbn = mdl.model_evaluate("LINLOG", avp, fb / 200) + rb

    return f, avr, avw, avt, avp, fb, tbn, wbn


def batch_level1_to_level2(
    band, batch_file=None, path=None, omit_days=None, day_indx=12
):
    if batch_file:
        with open(batch_file) as fl:
            days = yaml.load(fl, Loader=yaml.FullLoader)["days"]
    else:
        fl_list = sorted(listdir(path))
        days = [fl[day_indx : day_indx + 6] for fl in fl_list]
        if omit_days:
            days = [d for d in days if int(d[:3]) not in omit_days]

    for year, day in days:
        level1_to_level2(band, year, day)


def batch_level2_to_level3(
    band,
    flag_folder,
    first_day,
    last_day,
    receiver_cal_file=2,
    antenna_s11_Nfit=13,
    balun_correction=True,
    ground_correction=True,
    beam_correction=True,
    antenna_correction=True,
    f_low=55,
    beam_correction_case=1,
    f_high=120,
    n_fg=5,
    bad_files=None,
):

    # Listing files to be processed
    path_files = config["edges_folder"] + f"{band}/spectra/level2/"
    files = sorted(listdir(path_files))

    if bad_files is None and band == "mid_band":
        bad_files = dirname(__file__) + "data/bad_files_mid_band_2to3.yaml"

    if isinstance(bad_files, str):
        with open(bad_files) as fl:
            bad_files = yaml.load(fl, Loader=yaml.FullLoader)["bad_files"]

    bad_files = bad_files or []

    files = [f for f in files if f not in bad_files]

    # Processing files
    for fl in files:
        day = int(fl[5:8])

        if (day >= first_day) & (day <= last_day):
            level2_to_level3(
                band,
                fl,
                flag_folder=flag_folder,
                rcv_file=(
                    config["edges_folder"]
                    + "mid_band/calibration/receiver_calibration/receiver1/2018_01_25C/results/"
                    + CALFILES[receiver_cal_file]
                ),
                antenna_s11_n_terms=antenna_s11_Nfit,
                antenna_correction=antenna_correction,
                balun_correction=balun_correction,
                ground_correction=ground_correction,
                beam_correction=beam_correction,
                beam_correction_case=beam_correction_case,
                f_low=f_low,
                f_high=f_high,
                n_fg=n_fg,
            )
