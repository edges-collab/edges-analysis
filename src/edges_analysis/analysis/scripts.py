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


def daily_integrations_and_residuals():
    # TODO: might be old? not very general.

    f, pz, rz, wz, index, gha, ydz = io.level4read(
        config["edges_folder"]
        + "mid_band/spectra/level4/case_nominal/case_nominal.hdf5"
    )

    px = np.delete(pz, 1, axis=0)
    rx = np.delete(rz, 1, axis=0)
    wx = np.delete(wz, 1, axis=0)
    ydx = np.delete(ydz, 1, axis=0)

    # Average the data from the two days 147
    for i in range(len(gha) - 1):
        p147 = np.mean(pz[1:3, i, :], axis=0)
        r147, w147 = tools.spectral_averaging(rz[1:3, i, :], wz[1:3, i, :])

        px[1, i, :] = p147
        rx[1, i, :] = r147
        wx[1, i, :] = w147

    bad_days = np.array(
        [
            [2018, 159],
            [2018, 169],
            [2018, 184],
            [2018, 185],
            [2018, 191],
            [2018, 193],
            [2018, 195],
            [2018, 196],
            [2018, 204],
            [2018, 208],
            [2018, 216],
            [2018, 220],
        ]
    )

    f_low = 57
    f_high = 120
    n_fg = 5
    Nsp = 1

    ll = len(px)
    j = 0
    for i in range(ll):
        if not (ydx[i, 0] in bad_days[:, 0]) and (ydx[i, 1] in bad_days[:, 1]):
            mx = mdl.model_evaluate("LINLOG", px[i, 0, :], f / 200)
            tx = mx + rx[i, 0, :]

            fy = f[(f >= f_low) & (f <= f_high)]
            ty = tx[(f >= f_low) & (f <= f_high)]
            wy = wx[i, 0, (f >= f_low) & (f <= f_high)]

            p = mdl.fit_polynomial_fourier("LINLOG", fy / 200, ty, n_fg, Weights=wy)
            my = mdl.model_evaluate("LINLOG", p[0], fy / 200)
            ry = ty - my

            fb, rb, wb, sb = tools.spectral_binning_number_of_samples(
                fy, ry, wy, nsamples=128
            )

            mb = mdl.model_evaluate("LINLOG", p[0], fb / 200)
            tb = mb + rb

            if j == 0:
                rb_all = np.zeros((ll - len(bad_days), len(fb)))
                tb_all = np.zeros((ll - len(bad_days), len(fb)))
                wb_all = np.zeros((ll - len(bad_days), len(fb)))
                sb_all = np.zeros((ll - len(bad_days), len(fb)))
                yd_all = np.zeros((ll - len(bad_days), 2))
            else:
                tb_all[j] = tb
                rb_all[j] = rb
                wb_all[j] = wb
                sb_all[j] = sb
                yd_all[j] = ydx[i, :]

            j += 1

    K = 0.5
    lb = int(np.floor(len(tb_all[:, 0]) / Nsp))
    plt.figure(figsize=[6.5, 10])
    for i in range(lb):

        rb_i, wb_i = tools.spectral_averaging(
            rb_all[(Nsp * i) : (Nsp * i + Nsp), :],
            wb_all[(Nsp * i) : (Nsp * i + Nsp), :],
        )

        if i < 26 / Nsp:
            cc = "b"
            lw = 2
        else:
            cc = "r"
            lw = 1
        plt.plot(fb, rb_i - K * i, color=cc, linewidth=lw)

        if Nsp == 1:
            plt.text(54, -K * i - 0.3 * K, str(int(yd_all[i, 1])))
        else:
            plt.text(
                52,
                -K * i - 0.3 * K,
                str(int(yd_all[Nsp * i, 1]))
                + "-"
                + str(int(yd_all[Nsp * i + Nsp - 1, 1])),
            )

    plt.ylim([-K * lb, K])
    plt.yticks([])
    plt.xlabel(r"$\nu$ [MHz]", fontsize=13)
    plt.xlim([52, 120])

    return fb, tb_all, rb_all, wb_all, sb_all, yd_all


def integrated_spectrum_level4(
    case,
    index_GHA,
    f_low,
    f_high,
    day_range,
    day_min1,
    day_max1,
    day_min2,
    day_max2,
    n_fg,
    save,
    filename_flag,
):
    """
    Important high level function that averages level4 field data. Used consistently.
    TODO: make this into an actual script. Make it a little bit more general.
    The idea here is just to average together a bunch of level data over many days
    and GHA. Can choose the days.
    """
    cases = {
        100: "rcv18_sw18_nominal_GHA_6_18hr",
        101: "rcv18_sw18_nominal_GHA_every_1hr",
        21: "rcv18_ant19_nominal",
        22: "rcv18_ant19_every_1hr_GHA",
    }

    in_file = f"{config['edges_folder']}mid_band/spectra/level4/{cases[case]}/{cases[case]}.hdf5"
    save_path = dirname(in_file).replace("level4", "level5")
    f, px, rx, wx, index, gha, ydx = io.level4read(in_file)

    if case == 101:
        save_spectrum = (
            "integrated_spectrum_rcv18_sw18_every_1hr_GHA" + filename_flag + ".txt"
        )
    elif case == 22:
        save_spectrum = (
            "integrated_spectrum_rcv18_ant19_every_1hr_GHA" + filename_flag + ".txt"
        )

    # Produce integrated spectrum
    for i in range(len(index_GHA)):
        keep_index = filters.daily_nominal_filter("mid_band", case, index_GHA[i], ydx)

        mask = (keep_index == 1) & (
            ((ydx[:, 1] >= day_min1) & (ydx[:, 1] <= day_max1))
            | ((ydx[:, 1] >= day_min2) & (ydx[:, 1] <= day_max2))
        )

        p_i = px[mask, index_GHA[i]]
        r_i = rx[mask, index_GHA[i]]
        w_i = wx[mask, index_GHA[i]]

        if i == 0:
            p = np.copy(p_i)
            r = np.copy(r_i)
            w = np.copy(w_i)
        else:
            p = np.vstack((p, p_i))
            r = np.vstack((r, r_i))
            w = np.vstack((w, w_i))

    avp = np.mean(p, axis=0)
    m = mdl.model_evaluate("LINLOG", avp, f / 200)

    avr, avw = tools.spectral_averaging(r, w)
    flags = rfi.cleaning_sweep(
        avr,
        avw,
        window_width=int(3 / (f[1] - f[0])),
        n_poly=2,
        n_bootstrap=20,
        n_sigma=3,
    )
    rr = np.where(flags, 0, avr)
    wr = np.where(flags, 0, avw)

    tr = m + rr

    fr1 = 136
    fr2 = 139
    tr[(f >= fr1) & (f <= fr2)] = 0
    wr[(f >= fr1) & (f <= fr2)] = 0

    p = mdl.fit_polynomial_fourier("LINLOG", f / 200, tr, 7, Weights=wr)
    m = mdl.model_evaluate("LINLOG", p[0], f / 200)
    r = tr - m

    NS = 64
    fb, rb, wb, sb = tools.spectral_binning_number_of_samples(f, r, wr, nsamples=NS)

    mb = mdl.model_evaluate("LINLOG", p[0], fb / 200)
    tb = mb + rb
    tb[wb == 0] = 0
    sb[wb == 0] = 0

    # Saving spectrum
    if save and day_range != "daily":
        np.savetxt(
            save_path + save_spectrum,
            np.array([fb, tb, wb, sb]).T,
            header="freq [MHz], temp [K], weight [K], std dev [K]",
        )

    # Computing residuals for plot
    mask = (fb >= f_low) & (fb <= f_high)
    fx, tx, wx, sx = fb[mask], tb[mask], wb[mask], sb[mask]
    ft, tt, st = fx[wx > 0], tx[wx > 0], sx[wx > 0]

    pt = mdl.fit_polynomial_fourier(
        "LINLOG", ft / 200, tt, n_fg, Weights=(1 / (st ** 2))
    )
    mt = mdl.model_evaluate("LINLOG", pt[0], ft / 200)
    rt = tt - mt

    pl = np.polyfit(np.log(ft / 200), np.log(tt), n_fg - 1)
    log_ml = np.polyval(pl, np.log(ft / 200))
    ml = np.exp(log_ml)
    rl = tt - ml

    return ft, tt, st, rt, rl


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

    flags = rfi.cleaning_sweep(
        avr,
        avw,
        window_width=int(3 / (f[1] - f[0])),
        n_poly=2,
        n_bootstrap=20,
        n_sigma=2.5,
    )
    avrn = np.where(flags, 0, avr)
    avwn = np.where(flags, 0, avw)

    fb, rb, wbn = tools.spectral_binning_number_of_samples(f, avrn, avwn)
    tbn = mdl.model_evaluate("LINLOG", avp, fb / 200) + rb

    return f, avr, avw, avt, avp, fb, tbn, wbn


def season_integrated_spectra_GHA(
    band, case, new_gha_edges=np.arange(0, 25, 2), data_save_name_flag="2hr"
):
    """
    Take a whole season of spectra and produce season averages in GHA, with
    changeable range. Not being as strict for cleaning.

    # TODO: move to scripts.
    """
    data_save_path = config["edges_folder"] + f"{band}/spectra/level5/case{case}/"

    # Loading level4 data
    f, p_all, r_all, w_all, gha_edges, yd = io.level4read(
        config["edges_folder"] + f"{band}/spectra/level4/case{case}/case{case}.hdf5"
    )

    # Creating intermediate 1hr-average arrays
    pr_all = np.zeros((len(gha_edges) - 1, len(p_all[0, 0, :])))
    rr_all = np.zeros((len(gha_edges) - 1, len(f)))
    wr_all = np.zeros((len(gha_edges) - 1, len(f)))

    # Looping over every original GHA edges
    for j in range(len(gha_edges) - 1):
        # Looping over day
        index_good = filters.explicit_filter(
            yd,
            bad=join(
                dirname(__file__),
                f"data/bad_hours_{band}{case if case is not None else ''}.yaml",
            ),
        )

        # Selecting good parameters and spectra
        pp = p_all[index_good, j]
        rr = r_all[index_good, j]
        ww = w_all[index_good, j]

        # Average parameters and spectra
        avp = np.mean(pp, axis=0)
        avr, avw = tools.weighted_mean(rr, ww)

        # RFI cleaning of 1-hr season average spectra
        flags = rfi.cleaning_sweep(
            avr,
            avw,
            window_width=int(3 / (f[1] - f[0])),
            n_poly=2,
            n_bootstrap=20,
            n_sigma=2.5,
        )
        avr_no_rfi = np.where(flags, 0, avr)
        avw_no_rfi = np.where(flags, 0, avw)

        # Storing season 1hr-average spectra
        pr_all[j] = avp
        rr_all[j] = avr_no_rfi
        wr_all[j] = avw_no_rfi

        # Frequency binning
        fb, rb, wb = tools.spectral_binning_number_of_samples(f, avr_no_rfi, avw_no_rfi)
        mb = mdl.model_evaluate("LINLOG", avp, fb / 200)
        tb = mb + rb
        tb[wb == 0] = 0

        # Storing binned average spectra
        if j == 0:
            tb_all = np.zeros((len(gha_edges) - 1, len(fb)))
            wb_all = np.zeros((len(gha_edges) - 1, len(fb)))
        else:
            tb_all[j] = tb
            wb_all[j] = wb

    # Averaging data within new GHA edges
    for j in range(len(new_gha_edges) - 1):
        new_gha_start = new_gha_edges[j]
        new_gha_end = new_gha_edges[j + 1]

        flag = True
        for i in range(len(gha_edges) - 1):
            if (
                new_gha_start < new_gha_end
                and ((gha_edges[i] >= new_gha_start) and (gha_edges[i] < new_gha_end))
            ) or ((gha_edges[i] >= new_gha_start) or (gha_edges[i] < new_gha_end)):
                if flag:
                    px_all = pr_all[i]
                    rx_all = rr_all[i]
                    wx_all = wr_all[i]
                    flag = False
                else:
                    px_all = np.vstack((px_all, pr_all[i, :]))
                    rx_all = np.vstack((rx_all, rr_all[i, :]))
                    wx_all = np.vstack((wx_all, wr_all[i, :]))

        if len(px_all.shape) == 1:
            avpx = np.copy(px_all)
            avrx = np.copy(rx_all)
            avwx = np.copy(wx_all)
        elif len(px_all.shape) == 2:
            avpx = np.mean(px_all, axis=0)
            avrx, avwx = tools.weighted_mean(rx_all, wx_all)

        flags = rfi.cleaning_sweep(
            avrx,
            avwx,
            window_width=int(3 / (f[1] - f[0])),
            n_poly=2,
            n_bootstrap=20,
            n_sigma=2.5,
        )
        avrx_no_rfi = np.where(flags, 0, avrx)
        avwx_no_rfi = np.where(flags, 0, avwx)

        # Frequency binning
        fb, rbx, wbx = tools.spectral_binning_number_of_samples(
            f, avrx_no_rfi, avwx_no_rfi
        )
        modelx = mdl.model_evaluate("LINLOG", avpx, fb / 200)
        tbx = modelx + rbx
        tbx[wbx == 0] = 0

        # Storing binned average spectra
        if j == 0:
            tbx_all = np.zeros((len(new_gha_edges) - 1, len(fb)))
            wbx_all = np.zeros((len(new_gha_edges) - 1, len(fb)))

        tbx_all[j, :] = tbx
        wbx_all[j, :] = wbx

        # Saving data
        np.savetxt(
            data_save_path + f"case{case}_frequency.txt", fb, header="Frequency [MHz]."
        )
        np.savetxt(
            data_save_path + f"case{case}_1hr_gha_edges.txt",
            gha_edges,
            header="GHA edges of integrated spectra from 0hr to 23hr in steps of 1hr [hr].",
        )
        np.savetxt(
            data_save_path + f"case{case}_1hr_temperature.txt",
            tb_all,
            header="Rows correspond to different GHAs from 0hr to 23hr in steps of 1hr. Columns correspond to frequency.",
        )
        np.savetxt(
            data_save_path + f"case{case}_1hr_weights.txt",
            wb_all,
            header="Rows correspond to different GHAs from 0hr to 23hr in steps of 1hr. Columns correspond to frequency.",
        )
        np.savetxt(
            data_save_path + f"case{case}_{data_save_name_flag}_gha_edges.txt",
            new_gha_edges,
            header="GHA edges of integrated spectra [hr].",
        )
        np.savetxt(
            data_save_path + f"case{case}_{data_save_name_flag}_temperature.txt",
            tbx_all,
            header="Rows correspond to different GHAs. Columns correspond to frequency.",
        )
        np.savetxt(
            data_save_path + f"case{case}_{data_save_name_flag}_weights.txt",
            wbx_all,
            header="Rows correspond to different GHAs. Columns correspond to frequency.",
        )

    return fb, tb_all, wb_all, tbx_all, wbx_all


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
                antenna_s11_Nfit=antenna_s11_Nfit,
                antenna_correction=antenna_correction,
                balun_correction=balun_correction,
                ground_correction=ground_correction,
                beam_correction=beam_correction,
                beam_correction_case=beam_correction_case,
                f_low=f_low,
                f_high=f_high,
                n_fg=n_fg,
            )
