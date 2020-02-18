from os.path import join

import numpy as np
from edges_cal import modelling as mdl
from edges_cal import reflection_coefficient as rc
from edges_io.io import S1P


def antenna_s11_remove_delay(
    s11_path, f_MHz, delay_0=0.17, model_type="polynomial", Nfit=10
):
    d = np.genfromtxt(s11_path)

    flow = np.min(f_MHz)
    fhigh = np.max(f_MHz)

    d_cut = d[(d[:, 0] >= flow) & (d[:, 0] <= fhigh), :]

    f_orig_MHz = d_cut[:, 0]
    s11 = d_cut[:, 1] + 1j * d_cut[:, 2]

    # Removing delay from S11
    delay = delay_0 * f_orig_MHz
    re_wd = np.abs(s11) * np.cos(delay + np.unwrap(np.angle(s11)))
    im_wd = np.abs(s11) * np.sin(delay + np.unwrap(np.angle(s11)))

    # Fitting data with delay applied
    par_re_wd = np.polyfit(f_orig_MHz, re_wd, Nfit - 1)
    par_im_wd = np.polyfit(f_orig_MHz, im_wd, Nfit - 1)

    # Evaluating models at new input frequency
    model_re_wd = np.polyval(par_re_wd, f_MHz)
    model_im_wd = np.polyval(par_im_wd, f_MHz)

    model_s11_wd = model_re_wd + 1j * model_im_wd
    ra = model_s11_wd * np.exp(-1j * delay_0 * f_MHz)

    return ra


def switch_correction_receiver1(
    ant_s11, f_in, resistance_of_match, base_path, Nt_s11=23, Nt_s12s21=23, Nt_s22=23
):
    """
    This is a stand-in until this is moved to edges_cal
    """

    o_in, f = S1P.read(join(base_path, "Open01.s1p"))
    s_in, f = S1P.read(join(base_path, "Short01.s1p"))
    l_in, f = S1P.read(join(base_path, "Match01.s1p"))

    o_ex, f = S1P.read(join(base_path, "ExternalOpen01.s1p"))
    s_ex, f = S1P.read(join(base_path, "ExternalShort01.s1p"))
    l_ex, f = S1P.read(join(base_path, "ExternalMatch01.s1p"))

    # Standards assumed at the switch
    o_sw = 1 * np.ones(len(f))
    s_sw = -1 * np.ones(len(f))
    l_sw = 0 * np.ones(len(f))

    # Correction at the switch
    o_ex_c, xx1, xx2, xx3 = rc.de_embed(o_sw, s_sw, l_sw, o_in, s_in, l_in, o_ex)
    s_ex_c, xx1, xx2, xx3 = rc.de_embed(o_sw, s_sw, l_sw, o_in, s_in, l_in, s_ex)
    l_ex_c, xx1, xx2, xx3 = rc.de_embed(o_sw, s_sw, l_sw, o_in, s_in, l_in, l_ex)

    # Computation of S-parameters to the receiver input
    oa, sa, la = rc.agilent_85033E(f, resistance_of_match)  # , md)
    xx, s11, s12s21, s22 = rc.de_embed(oa, sa, la, o_ex_c, s_ex_c, l_ex_c, o_ex_c)

    # Polynomial fit of S-parameters from "f" to input frequency vector "f_in"
    # ------------------------------------------------------------------------
    # Frequency normalization
    max_f = np.max(f)
    fn = (f / max_f) - 0.5

    if len(f_in) > 10:
        if f_in[0] > 1e5:
            fn_in = f_in / 1e6
        elif f_in[-1] < 300:
            fn_in = np.copy(f_in)

        fn_in = (fn_in / max_f) - 0.5

    else:
        fn_in = np.copy(fn)

    # Real-Imaginary parts
    real_s11 = np.real(s11)
    imag_s11 = np.imag(s11)
    real_s12s21 = np.real(s12s21)
    imag_s12s21 = np.imag(s12s21)
    real_s22 = np.real(s22)
    imag_s22 = np.imag(s22)

    # Polynomial fits
    p = mdl.fit_polynomial_fourier("fourier", fn, real_s11, Nt_s11)
    fit_real_s11 = mdl.model_evaluate("fourier", p[0], fn)

    p = mdl.fit_polynomial_fourier("fourier", fn, imag_s11, Nt_s11)
    fit_imag_s11 = mdl.model_evaluate("fourier", p[0], fn)

    p = mdl.fit_polynomial_fourier("fourier", fn, real_s12s21, Nt_s12s21)
    fit_real_s12s21 = mdl.model_evaluate("fourier", p[0], fn)

    p = mdl.fit_polynomial_fourier("fourier", fn, imag_s12s21, Nt_s12s21)
    fit_imag_s12s21 = mdl.model_evaluate("fourier", p[0], fn)

    p = mdl.fit_polynomial_fourier("fourier", fn, real_s22, Nt_s22)
    fit_real_s22 = mdl.model_evaluate("fourier", p[0], fn)

    p = mdl.fit_polynomial_fourier("fourier", fn, imag_s22, Nt_s22)
    fit_imag_s22 = mdl.model_evaluate("fourier", p[0], fn)

    fit_s11 = fit_real_s11 + 1j * fit_imag_s11
    fit_s12s21 = fit_real_s12s21 + 1j * fit_imag_s12s21
    fit_s22 = fit_real_s22 + 1j * fit_imag_s22

    # Corrected antenna S11
    corr_ant_s11 = rc.gamma_de_embed(fit_s11, fit_s12s21, fit_s22, ant_s11)

    return (corr_ant_s11, fit_s11, fit_s12s21, fit_s22, f, s11, s12s21, s22)
