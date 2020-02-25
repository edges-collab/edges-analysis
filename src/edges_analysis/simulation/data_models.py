from os import listdir

import numpy as np
from edges_cal import modelling as mdl
from edges_cal import receiver_calibration_func as rcf
from edges_cal import reflection_coefficient as rc
from edges_cal.cal_coefficients import EdgesFrequencyRange

from ..analysis.loss import balun_and_connector_loss
from ..analysis.scripts import models_antenna_s11_remove_delay
from ..estimation import models

edges_folder = ""  # TODO: remove


def simulated_data(
    theta,
    v,
    vr,
    noise_std_at_vr,
    model_type_signal="exp",
    model_type_foreground="exp",
    N21par=4,
    NFGpar=5,
):
    std_dev_vec = noise_std_at_vr * (v / vr) ** (-2.5)

    sigma = np.diag(std_dev_vec ** 2)  # uncertainty covariance matrix
    inv_sigma = np.linalg.inv(sigma)
    det_sigma = np.linalg.det(sigma)

    noise = np.random.multivariate_normal(np.zeros(len(v)), sigma)

    d_no_noise = full_model(
        theta,
        v,
        vr,
        model_type_signal=model_type_signal,
        model_type_foreground=model_type_foreground,
        N21par=N21par,
        NFGpar=NFGpar,
    )
    d = d_no_noise + noise

    return d, sigma, inv_sigma, det_sigma


def real_data(case, FLOW, FHIGH, gap_FLOW=0, gap_FHIGH=0):
    if case == 1:
        dd = np.genfromtxt(
            edges_folder + "mid_band/spectra/level5/one_day_tests/another_one.txt"
        )
    elif case == 101:
        dd = np.genfromtxt(
            edges_folder + "mid_band/spectra/level5/rcv18_sw18_nominal_GHA_every_1hr"
            "/integrated_spectrum_rcv18_sw18_every_1hr_GHA_6-18hr.txt"
        )
    vv = dd[:, 0]
    tt = dd[:, 1]
    ww = dd[:, 2]
    ss = dd[:, 3]

    vp = vv[(vv >= FLOW) & (vv <= FHIGH)]
    tp = tt[(vv >= FLOW) & (vv <= FHIGH)]
    wp = ww[(vv >= FLOW) & (vv <= FHIGH)]
    sp = ss[(vv >= FLOW) & (vv <= FHIGH)]

    # Possibility of removing from analysis the data range between FLOW_gap and FHIGH_gap
    if (gap_FLOW > 0) and (gap_FHIGH > 0):
        vx = np.copy(vp)
        tx = np.copy(tp)
        wx = np.copy(wp)
        sx = np.copy(sp)

        vp = vx[(vx <= gap_FLOW) | (vx >= gap_FHIGH)]
        tp = tx[(vx <= gap_FLOW) | (vx >= gap_FHIGH)]
        wp = wx[(vx <= gap_FLOW) | (vx >= gap_FHIGH)]
        sp = sx[(vx <= gap_FLOW) | (vx >= gap_FHIGH)]

    v = vp[wp > 0]
    t = tp[wp > 0]
    w = wp[wp > 0]
    std_dev_vec = sp[wp > 0]

    sigma = np.diag(std_dev_vec ** 2)  # uncertainty covariance matrix
    inv_sigma = np.linalg.inv(sigma)
    det_sigma = np.linalg.det(sigma)

    return v, t, w, sigma, inv_sigma, det_sigma


def foreground_model(
    model_type, theta_fg, v, vr, ion_abs_coeff="free", ion_emi_coeff="free"
):
    number_of_parameters = len(theta_fg)

    model_fg = 0

    if model_type == "linlog":
        for i in range(number_of_parameters):
            model_fg += theta_fg[i] * ((v / vr) ** (-2.5)) * ((np.log(v / vr)) ** i)

    elif model_type == "powerlog":
        if (ion_abs_coeff == "free") and (ion_emi_coeff == "free"):
            number_astro_parameters = number_of_parameters - 2

        elif (ion_abs_coeff == "free") or (ion_emi_coeff == "free"):
            number_astro_parameters = number_of_parameters - 1

        else:
            number_astro_parameters = number_of_parameters

        astro_exponent = sum(
            theta_fg[i + 1] * ((np.log(v / vr)) ** i)
            for i in range(number_astro_parameters - 1)
        )
        astro_fg = theta_fg[0] * ((v / vr) ** astro_exponent)

        IAC = theta_fg[-2] if ion_abs_coeff == "free" else ion_abs_coeff
        ionos_abs = np.exp(IAC * ((v / vr) ** (-2)))

        IEC = theta_fg[-1] if ion_emi_coeff == "free" else ion_emi_coeff
        ionos_emi = IEC * (1 - ionos_abs)

        model_fg = (astro_fg * ionos_abs) + ionos_emi
    return model_fg


def signal_model(model_type, theta, v):
    return models.model_eor_flattened_gaussian(
        v, model_type=["exp", "tanh"].index(model_type), *theta
    )


def full_model(
    theta, v, vr, model_type_signal="exp", model_type_foreground="exp", N21par=4
):
    # Signal model
    if N21par == 0:
        model_21 = 0
    else:
        model_21 = signal_model(model_type_signal, theta[0:N21par], v)

    # Foreground model
    model_fg = foreground_model(
        model_type_foreground, theta[N21par::], v, vr, ion_abs_coeff=0, ion_emi_coeff=0
    )

    return model_21 + model_fg


def spectrum_channel_to_channel_difference(f, t, w, FLOW, FHIGH, Nfg=5):
    fc = f[(f >= FLOW) & (f <= FHIGH)]
    tc = t[(f >= FLOW) & (f <= FHIGH)]
    wc = w[(f >= FLOW) & (f <= FHIGH)]

    par = mdl.fit_polynomial_fourier("LINLOG", fc, tc, Nfg, Weights=wc)

    rc = tc - par[1]

    x1 = np.arange(0, len(rc), 2)
    x2 = np.arange(1, len(rc), 2)

    return rc[x2] - rc[x1]


def svd_functions(folder_with_spectra, FLOW, FHIGH, method="average_removed"):
    """Computing SVD functions"""

    # Listing files to be processed
    # -----------------------------
    folder = edges_folder + "mid_band/spectra/level5/" + folder_with_spectra + "/"
    list_of_spectra = listdir(folder)
    list_of_spectra.sort()

    # Generate the original matrix of spectra
    # ---------------------------------------
    for i in range(len(list_of_spectra)):
        filename = list_of_spectra[i]
        d = np.genfromtxt(folder + filename)

        if i == 0:
            A = d[:, 1]
            fx = d[:, 0]
        else:
            A = np.vstack((A, d[:, 1]))

    # Cut to desired frequency range
    f = fx[(fx >= FLOW) & (fx <= FHIGH)]
    A = A[:, (fx >= FLOW) & (fx <= FHIGH)]

    # Remove channels with no data
    P = np.prod(A, axis=0)
    f = f[P > 0]
    A = A[:, P > 0]

    if method == "average_removed":
        avA = np.mean(A, axis=0)
        C = A - avA
    elif method == "delta_all":
        flag = 0
        for j in range(len(list_of_spectra) - 1):
            for i in range(len(list_of_spectra) - 1 - j):

                k = i + (j + 1)
                B = A[k] - A[j]

                if flag == 0:
                    C = np.copy(B)

                elif flag > 0:
                    C = np.vstack((C, B))

                print(str(flag) + ": " + str(k) + "-" + str(j))

                flag += 1

    # SVD
    # ---------------------------------------
    u, EValues, EFunctions = np.linalg.svd(C)

    return f, EValues, EFunctions


def signal_edges2018_uncertainties(v):
    # Nominal model reported in Bowman et al. (2018)
    # ----------------------------------------------
    model_nominal = signal_model("exp", [-0.5, 78, 19, 7], v)

    # Computing distribution of models and limits
    # -------------------------------------------
    N_MC = 10000
    dx = 0.003 / 2  # probability tails for 99.7% (3 sigma) probabilities

    # Centers and Widths of Gaussian Parameter Distributions
    uA = (-1 - (-0.2)) / 2 - 0.2
    sA = (0.2 + 0.5) / 6

    uv0 = 78
    sv0 = 1 / 3

    uW = (23 - 17) / 2 + 17
    sW = (23 - 17) / 6

    ut = (12 - 4) / 2 + 4
    st = (12 - 4) / 6

    # Random models
    model_perturbed = np.zeros((N_MC, len(v)))

    for i in range(N_MC):
        x = np.random.multivariate_normal([0, 0, 0, 0], np.diag(np.ones(4)))
        dA = sA * x[0]
        dv0 = sv0 * x[1]
        dW = sW * x[2]
        dt = st * x[3]

        model_perturbed[i, :] = signal_model(
            "exp", [uA + dA, uv0 + dv0, uW + dW, ut + dt], v
        )

    # Limits
    limits = np.zeros((len(v), 2))
    for i in range(len(v)):
        x_low = -1
        x_high = -1
        x = model_perturbed[:, i]
        x_increasing = np.sort(x)
        cx = np.abs(np.cumsum(x_increasing))
        norm_cx = cx / np.max(cx)
        for j in range(len(norm_cx) - 1):
            # print(norm_cx[j])
            if (norm_cx[j] < dx) and (norm_cx[j + 1] >= dx):
                x_low = x_increasing[j + 1]

            if (norm_cx[j] < (1 - dx)) and (norm_cx[j + 1] > (1 - dx)):
                x_high = x_increasing[j]

        if x_low == -1:
            x_low = x_increasing[0]
        if x_high == -1:
            x_high = x_increasing[-1]

        limits[i, 0] = x_low
        limits[i, 1] = x_high

    return model_nominal, model_perturbed, limits


def models_calibration_spectra(case, f, MC_spectra_noise=np.zeros(4)):
    if case == 1:
        path_par_spec = (
            edges_folder
            + "/mid_band/calibration/receiver_calibration/receiver1/2018_01_25C/results"
            "/nominal/spectra/"
        )

        # Loading parameters
        par_spec_amb = np.genfromtxt(path_par_spec + "par_spec_amb.txt")
        par_spec_hot = np.genfromtxt(path_par_spec + "par_spec_hot.txt")
        par_spec_open = np.genfromtxt(path_par_spec + "par_spec_open.txt")
        par_spec_shorted = np.genfromtxt(path_par_spec + "par_spec_shorted.txt")
        RMS_spec = np.genfromtxt(path_par_spec + "RMS_spec.txt")

        # Normalized frequency
        fn = (f - 120) / 60

        # Evaluating models
        Tae = mdl.model_evaluate("fourier", par_spec_amb, fn)
        The = mdl.model_evaluate("fourier", par_spec_hot, fn)
        Toe = mdl.model_evaluate("fourier", par_spec_open, fn)
        Tse = mdl.model_evaluate("fourier", par_spec_shorted, fn)

    # RMS noise
    RMS_Tae = RMS_spec[0]
    RMS_The = RMS_spec[1]
    RMS_Toe = RMS_spec[2]
    RMS_Tse = RMS_spec[3]

    # Adding noise to models
    if MC_spectra_noise[0] > 0:
        Tae = Tae + MC_spectra_noise[0] * RMS_Tae * np.random.normal(
            0, np.ones(len(fn))
        )

    if MC_spectra_noise[1] > 0:
        The = The + MC_spectra_noise[1] * RMS_The * np.random.normal(
            0, np.ones(len(fn))
        )

    if MC_spectra_noise[2] > 0:
        Toe = Toe + MC_spectra_noise[2] * RMS_Toe * np.random.normal(
            0, np.ones(len(fn))
        )

    if MC_spectra_noise[3] > 0:
        Tse = Tse + MC_spectra_noise[3] * RMS_Tse * np.random.normal(
            0, np.ones(len(fn))
        )

    return np.array([Tae, The, Toe, Tse])


def random_signal_perturbation(f, RMS_expectation, Npar_max):
    # Choose randomly, from a Uniform distribution, an integer corresponding to the number of
    # polynomial terms
    Npar = np.random.choice(Npar_max + 1)

    # Choose randomly, from a Gaussian distribution, a number corresponding to the RMS of the
    # perturbation across frequency
    RMS = RMS_expectation * np.random.normal(0, 1)

    # Generate random noise across frequency, with MEAN=0 and STD=1
    noise = np.random.normal(0, 1, size=len(f))

    # Fit a generic polynomial to the noise, using a polynomial with Npar terms
    par = np.polyfit(f, noise, Npar + 1)
    model = np.polyval(par, f)

    # Compute the current RMS of the polynomial
    RMS_model = np.std(model)

    return (RMS / RMS_model) * model


def two_port_network_uncertainties():
    """
    This function propagates the uncertainty in 1-port measurements
    to uncertainties in the 2-port S-parameters of the short cable between the hot load and the
    receiver
    """

    # Simulated measurements at the VNA input
    # ---------------------------------------

    # Reflection standard models
    f = np.arange(50, 181)  # In MHz
    resistance_of_match = 50.1

    Y = rc.agilent_85033E((10 ** 6) * f, resistance_of_match, m=1, md_value_ps=38)
    oa = Y[0]
    sa = Y[1]
    la = Y[2]

    # Simulated measurements at the VNA input
    o1m = 1 * np.ones(len(f))
    s1m = -1 * np.ones(len(f))
    l1m = 0.001 * np.ones(len(f))

    # S-parameters of VNA calibration network
    X, s11V, s12s21V, s22V = rc.de_embed(oa, sa, la, o1m, s1m, l1m, o1m)

    # Simulated measurements at the end of the 2-port network
    # -------------------------------------------------------

    # Load 2-port parameters and models

    path_par_s11 = (
        edges_folder + "/mid_band/calibration/receiver_calibration/receiver1"
        "/2018_01_25C/results/nominal/s11/"
    )

    par_s11_sr_mag = np.genfromtxt(path_par_s11 + "par_s11_sr_mag.txt")
    par_s11_sr_ang = np.genfromtxt(path_par_s11 + "par_s11_sr_ang.txt")

    par_s12s21_sr_mag = np.genfromtxt(path_par_s11 + "par_s12s21_sr_mag.txt")
    par_s12s21_sr_ang = np.genfromtxt(path_par_s11 + "par_s12s21_sr_ang.txt")

    par_s22_sr_mag = np.genfromtxt(path_par_s11 + "par_s22_sr_mag.txt")
    par_s22_sr_ang = np.genfromtxt(path_par_s11 + "par_s22_sr_ang.txt")

    fen = (f - 120) / 60

    s11_mag = mdl.model_evaluate("polynomial", par_s11_sr_mag, fen)
    s11_ang = mdl.model_evaluate("polynomial", par_s11_sr_ang, fen)

    s12s21_mag = mdl.model_evaluate("polynomial", par_s12s21_sr_mag, fen)
    s12s21_ang = mdl.model_evaluate("polynomial", par_s12s21_sr_ang, fen)

    s22_mag = mdl.model_evaluate("polynomial", par_s22_sr_mag, fen)
    s22_ang = mdl.model_evaluate("polynomial", par_s22_sr_ang, fen)

    s11 = s11_mag * (np.cos(s11_ang) + 1j * np.sin(s11_ang))
    s12s21 = s12s21_mag * (np.cos(s12s21_ang) + 1j * np.sin(s12s21_ang))
    s22 = s22_mag * (np.cos(s22_ang) + 1j * np.sin(s22_ang))

    # Measurements as seen at the input of 2-port network
    oX = rc.gamma_shifted(s11, s12s21, s22, oa)
    sX = rc.gamma_shifted(s11, s12s21, s22, sa)
    lX = rc.gamma_shifted(s11, s12s21, s22, la)

    # Measurements at the uncalibrated VNA port
    o2m = rc.gamma_shifted(s11V, s12s21V, s22V, oX)
    s2m = rc.gamma_shifted(s11V, s12s21V, s22V, sX)
    l2m = rc.gamma_shifted(s11V, s12s21V, s22V, lX)

    # Simulating and propagating uncertainties
    # ----------------------------------------
    N = 10000  # MC repetitions

    RMS_mag = 0.0001  # magnitude uncertainty STD
    RMS_ang = 0.1  # phase uncertainty STD
    Npar_max = 15  # maximum number of polynomial terms

    s11_N = np.zeros((N, len(f))) + 1j * 0
    s12s21_N = np.zeros((N, len(f))) + 1j * 0
    s22_N = np.zeros((N, len(f))) + 1j * 0

    for i in range(N):
        print(i)

        # Add perturbations to measurements

        # Open at VNA port
        pert_mag = random_signal_perturbation(f, RMS_mag, Npar_max)
        pert_ang = random_signal_perturbation(f, RMS_ang, Npar_max)

        o1mp_mag = np.abs(o1m) + pert_mag
        o1mp_ang = np.unwrap(np.angle(o1m)) + (np.pi / 180) * pert_ang

        o1mp = o1mp_mag * (np.cos(o1mp_ang) + 1j * np.sin(o1mp_ang))

        # Short at VNA port
        pert_mag = random_signal_perturbation(f, RMS_mag, Npar_max)
        pert_ang = random_signal_perturbation(f, RMS_ang, Npar_max)

        s1mp_mag = np.abs(s1m) + pert_mag
        s1mp_ang = np.unwrap(np.angle(s1m)) + (np.pi / 180) * pert_ang

        s1mp = s1mp_mag * (np.cos(s1mp_ang) + 1j * np.sin(s1mp_ang))

        # Load at VNA port
        pert_mag = random_signal_perturbation(f, RMS_mag, Npar_max)
        pert_ang = random_signal_perturbation(f, RMS_ang, Npar_max)

        l1mp_mag = np.abs(l1m) + pert_mag
        l1mp_ang = np.unwrap(np.angle(l1m)) + (np.pi / 180) * pert_ang

        l1mp = l1mp_mag * (np.cos(l1mp_ang) + 1j * np.sin(l1mp_ang))

        # Open at the end of network
        pert_mag = random_signal_perturbation(f, RMS_mag, Npar_max)
        pert_ang = random_signal_perturbation(f, RMS_ang, Npar_max)

        o2mp_mag = np.abs(o2m) + pert_mag
        o2mp_ang = np.unwrap(np.angle(o2m)) + (np.pi / 180) * pert_ang

        o2mp = o2mp_mag * (np.cos(o2mp_ang) + 1j * np.sin(o2mp_ang))

        # Short at the end of network
        pert_mag = random_signal_perturbation(f, RMS_mag, Npar_max)
        pert_ang = random_signal_perturbation(f, RMS_ang, Npar_max)

        s2mp_mag = np.abs(s2m) + pert_mag
        s2mp_ang = np.unwrap(np.angle(s2m)) + (np.pi / 180) * pert_ang

        s2mp = s2mp_mag * (np.cos(s2mp_ang) + 1j * np.sin(s2mp_ang))

        # Load at the end of network
        pert_mag = random_signal_perturbation(f, RMS_mag, Npar_max)
        pert_ang = random_signal_perturbation(f, RMS_ang, Npar_max)

        l2mp_mag = np.abs(l2m) + pert_mag
        l2mp_ang = np.unwrap(np.angle(l2m)) + (np.pi / 180) * pert_ang

        l2mp = l2mp_mag * (np.cos(l2mp_ang) + 1j * np.sin(l2mp_ang))

        # Calibrate measurements at the end of 2-port network, to VNA port
        o2mx, xa, xb, xc = rc.de_embed(oa, sa, la, o1mp, s1mp, l1mp, o2mp)
        s2mx, xa, xb, xc = rc.de_embed(oa, sa, la, o1mp, s1mp, l1mp, s2mp)
        l2mx, xa, xb, xc = rc.de_embed(oa, sa, la, o1mp, s1mp, l1mp, l2mp)

        # Compute S-parameters of DUT
        l, s11new, s12s21new, s22new = rc.de_embed(oa, sa, la, o2mx, s2mx, l2mx, l2mx)

        # Store S-parameters of DUT
        s11_N[i, :] = s11new
        s12s21_N[i, :] = s12s21new
        s22_N[i, :] = s22new

    # The first three S-parameters are the input, nominal values. The last three are the MC
    # arrays from which the uncertainty STD could be estimated
    return f, s11, s12s21, s22, s11_N, s12s21_N, s22_N


def MC_receiver(
    band, MC_spectra_noise=np.ones(4), MC_s11_syst=np.ones(16), MC_temp=np.ones(4)
):
    # Settings
    case_models = 1  # mid band 2018
    s11_Npar_max = 14

    # band          = 'mid_band'

    f = EdgesFrequencyRange(f_low=50, f_high=150).freq
    # fx, il, ih = ba.frequency_edges(50, 150)
    # f = fx[il : (ih + 1)]
    fn = (f - 120) / 60

    Tamb_internal = 300

    cterms = 7
    wterms = 8

    # MC flags

    # Computing "Perturbed" receiver spectra, reflection coefficients, and physical temperatures

    # Spectra
    if np.sum(MC_spectra_noise) == 0:
        Tunc = np.genfromtxt(
            edges_folder
            + "mid_band/calibration/receiver_calibration/receiver1/2018_01_25C"
            "/results/nominal/data/average_spectra_300_350.txt"
        )
        print(Tunc[:, 0])
        ms = Tunc[:, 1:5].T
        print(ms.shape)
    else:
        ms = models_calibration_spectra(
            case_models, f, MC_spectra_noise=MC_spectra_noise
        )

    # S11
    mr = models_calibration_s11(
        case_models, f, MC_s11_syst=MC_s11_syst, Npar_max=s11_Npar_max
    )

    # Physical temperature
    mt = models_calibration_physical_temperature(
        case_models, f, s_parameters=[mr[2], mr[5], mr[6], mr[7]], MC_temp=MC_temp
    )

    Tae = ms[0]
    The = ms[1]
    Toe = ms[2]
    Tse = ms[3]

    rl = mr[0]
    ra = mr[1]
    rh = mr[2]
    ro = mr[3]
    rs = mr[4]

    Ta = mt[0]
    Thd = mt[1]
    To = mt[2]
    Ts = mt[3]

    # Computing receiver calibration quantities
    C1, C2, TU, TC, TS = rcf.get_calibration_quantities_iterative(
        fn,
        T_raw={"ambient": Tae, "hot_load": The, "short": Tse, "open": Toe},
        gamma_rec=rl,
        gamma_ant={"ambient": ra, "hot_load": rh, "short": rs, "open": ro},
        T_ant={"ambient": Ta, "hot_load": Thd, "short": Ts, "open": To},
        cterms=cterms,
        wterms=wterms,
        Tamb_internal=Tamb_internal,
    )

    return f, rl, C1, C2, TU, TC, TS


def MC_antenna_s11(f, rant, s11_Npar_max=14):
    # rant = models_antenna_s11_remove_delay(band, f, year=2018, day=147, delay_0=0.17,
    # model_type='polynomial', n_fit=14, plot_fit_residuals='no')

    # Producing perturbed antenna reflection coefficient

    RMS_mag = 0.0001  # original value = 0.0001
    RMS_ang = 0.1 * (np.pi / 180)

    pert_mag = random_signal_perturbation(f, RMS_mag, s11_Npar_max)
    pert_ang = random_signal_perturbation(f, RMS_ang, s11_Npar_max)

    rant_mag_MC = np.abs(rant) + pert_mag
    rant_ang_MC = np.unwrap(np.angle(rant)) + pert_ang

    return rant_mag_MC * (np.cos(rant_ang_MC) + 1j * np.sin(rant_ang_MC))


def MC_antenna_loss(f, G_Npar_max=10):
    """
    Output:  dG
    Perturbed MC loss: G_nominal + dG

    where:
    Gb, Gc = balun_and_connector_loss(band, f, rant_MC)
    G_nominal = Gb*Gc  # balun loss x connector loss

    """

    RMS_G = 0.00025  # 5% of typical (i.e., of 0.5%)

    flag = 1
    while flag == 1:
        dG = random_signal_perturbation(f, RMS_G, G_Npar_max)
        if (
            np.max(np.abs(dG)) <= 6 * RMS_G
        ):  # 6 sigma = 0.0015, forcing the loss to stay within reason
            flag = 0

    return dG


def MC_error_propagation():
    band = "mid_band"
    FLOW = 61
    FHIGH = 121

    ff, rl_X, C1_X, C2_X, TU_X, TC_X, TS_X = MC_receiver(
        band,
        MC_spectra_noise=np.zeros(4),
        MC_s11_syst=np.zeros(16),
        MC_temp=np.zeros(4),
    )

    rant1_X = models_antenna_s11_remove_delay(
        "mid_band",
        ff,
        year=2018,
        day=147,
        delay_0=0.17,
        model_type="polynomial",
        Nfit=14,
        plot_fit_residuals=False,
    )
    rant2_X = models_antenna_s11_remove_delay(
        "low_band3",
        ff,
        year=2018,
        day=227,
        delay_0=0.17,
        model_type="polynomial",
        Nfit=14,
        plot_fit_residuals=False,
    )

    Gb, Gc = balun_and_connector_loss("mid_band", ff, rant1_X)
    G1_X = Gb * Gc

    Gb, Gc = balun_and_connector_loss("low_band3", ff, rant1_X)
    G2_X = Gb * Gc

    f = ff[(ff >= FLOW) & (ff <= FHIGH)]
    rl = rl_X[(ff >= FLOW) & (ff <= FHIGH)]
    C1 = C1_X[(ff >= FLOW) & (ff <= FHIGH)]
    C2 = C2_X[(ff >= FLOW) & (ff <= FHIGH)]
    TU = TU_X[(ff >= FLOW) & (ff <= FHIGH)]
    TC = TC_X[(ff >= FLOW) & (ff <= FHIGH)]
    TS = TS_X[(ff >= FLOW) & (ff <= FHIGH)]

    rant1 = rant1_X[(ff >= FLOW) & (ff <= FHIGH)]
    rant2 = rant2_X[(ff >= FLOW) & (ff <= FHIGH)]

    G1 = G1_X[(ff >= FLOW) & (ff <= FHIGH)]
    G2 = G2_X[(ff >= FLOW) & (ff <= FHIGH)]

    tsky1 = 1000 * (f / 100) ** (-2.5)
    tsky2 = 3000 * (f / 100) ** (-2.5)

    Tamb = 273 + 25

    tsky1L1 = tsky1 * G1 + Tamb * (1 - G1)
    tsky2L1 = tsky2 * G1 + Tamb * (1 - G1)

    tsky1L2 = tsky1 * G2 + Tamb * (1 - G2)
    tsky2L2 = tsky2 * G2 + Tamb * (1 - G2)

    tunc1 = rcf.uncalibrated_antenna_temperature(
        tsky1L1, rant1, rl, C1, C2, TU, TC, TS, T_load=300
    )
    tunc2 = rcf.uncalibrated_antenna_temperature(
        tsky1L2, rant2, rl, C1, C2, TU, TC, TS, T_load=300
    )

    tunc3 = rcf.uncalibrated_antenna_temperature(
        tsky2L1, rant1, rl, C1, C2, TU, TC, TS, T_load=300
    )
    tunc4 = rcf.uncalibrated_antenna_temperature(
        tsky2L2, rant2, rl, C1, C2, TU, TC, TS, T_load=300
    )

    ff, rl_Y, C1_Y, C2_Y, TU_Y, TC_Y, TS_Y = MC_receiver(
        band,
        MC_spectra_noise=np.zeros(4),
        MC_s11_syst=np.zeros(16),
        MC_temp=np.zeros(4),
    )

    rant1_Y = MC_antenna_s11(ff, rant1_X, s11_Npar_max=14)
    rant2_Y = MC_antenna_s11(ff, rant2_X, s11_Npar_max=14)

    dG1_Y = MC_antenna_loss(ff, G_Npar_max=10)
    dG2_Y = MC_antenna_loss(ff, G_Npar_max=10)

    G1_Y = G1_X + dG1_Y
    G2_Y = G2_X + dG2_Y

    rl_MC = rl_Y[(ff >= FLOW) & (ff <= FHIGH)]
    C1_MC = C1_Y[(ff >= FLOW) & (ff <= FHIGH)]
    C2_MC = C2_Y[(ff >= FLOW) & (ff <= FHIGH)]
    TU_MC = TU_Y[(ff >= FLOW) & (ff <= FHIGH)]
    TC_MC = TC_Y[(ff >= FLOW) & (ff <= FHIGH)]
    TS_MC = TS_Y[(ff >= FLOW) & (ff <= FHIGH)]

    rant1_MC = rant1_Y[(ff >= FLOW) & (ff <= FHIGH)]
    rant2_MC = rant2_Y[(ff >= FLOW) & (ff <= FHIGH)]

    G1_MC = G1_Y[(ff >= FLOW) & (ff <= FHIGH)]
    G2_MC = G2_Y[(ff >= FLOW) & (ff <= FHIGH)]

    # rant_MC = MC_antenna_s11(f, band)

    tcal1L1 = rcf.calibrated_antenna_temperature(
        tunc1, rant1_MC, rl_MC, C1_MC, C2_MC, TU_MC, TC_MC, TS_MC, T_load=300
    )
    tcal1L2 = rcf.calibrated_antenna_temperature(
        tunc2, rant2_MC, rl_MC, C1_MC, C2_MC, TU_MC, TC_MC, TS_MC, T_load=300
    )

    tcal2L1 = rcf.calibrated_antenna_temperature(
        tunc3, rant1_MC, rl_MC, C1_MC, C2_MC, TU_MC, TC_MC, TS_MC, T_load=300
    )
    tcal2L2 = rcf.calibrated_antenna_temperature(
        tunc4, rant2_MC, rl_MC, C1_MC, C2_MC, TU_MC, TC_MC, TS_MC, T_load=300
    )

    tcal1 = (tcal1L1 - Tamb * (1 - G1_MC)) / G1_MC
    tcal2 = (tcal1L2 - Tamb * (1 - G2_MC)) / G2_MC

    tcal3 = (tcal2L1 - Tamb * (1 - G1_MC)) / G1_MC
    tcal4 = (tcal2L2 - Tamb * (1 - G2_MC)) / G2_MC

    Nfg = 1
    par1 = mdl.fit_polynomial_fourier("LINLOG", f, tcal1, Nfg)
    par2 = mdl.fit_polynomial_fourier("LINLOG", f, tcal2, Nfg)

    par3 = mdl.fit_polynomial_fourier("LINLOG", f, tcal3, Nfg)
    par4 = mdl.fit_polynomial_fourier("LINLOG", f, tcal4, Nfg)

    return (
        f,
        rant1,
        rant2,
        tcal1,
        tcal2,
        tcal3,
        tcal4,
        par1[1],
        par2[1],
        par3[1],
        par4[1],
    )


def models_calibration_s11(case, f, MC_s11_syst=np.zeros(16), Npar_max=15):
    if case == 1:
        path_par_s11 = (
            edges_folder
            + "/mid_band/calibration/receiver_calibration/receiver1/2018_01_25C/results"
            "/nominal/s11/"
        )

    # Loading S11 parameters
    par_s11_LNA_mag = np.genfromtxt(path_par_s11 + "par_s11_LNA_mag.txt")
    par_s11_LNA_ang = np.genfromtxt(path_par_s11 + "par_s11_LNA_ang.txt")

    par_s11_amb_mag = np.genfromtxt(path_par_s11 + "par_s11_amb_mag.txt")
    par_s11_amb_ang = np.genfromtxt(path_par_s11 + "par_s11_amb_ang.txt")

    par_s11_hot_mag = np.genfromtxt(path_par_s11 + "par_s11_hot_mag.txt")
    par_s11_hot_ang = np.genfromtxt(path_par_s11 + "par_s11_hot_ang.txt")

    par_s11_open_mag = np.genfromtxt(path_par_s11 + "par_s11_open_mag.txt")
    par_s11_open_ang = np.genfromtxt(path_par_s11 + "par_s11_open_ang.txt")

    par_s11_shorted_mag = np.genfromtxt(path_par_s11 + "par_s11_shorted_mag.txt")
    par_s11_shorted_ang = np.genfromtxt(path_par_s11 + "par_s11_shorted_ang.txt")

    par_s11_sr_mag = np.genfromtxt(path_par_s11 + "par_s11_sr_mag.txt")
    par_s11_sr_ang = np.genfromtxt(path_par_s11 + "par_s11_sr_ang.txt")

    par_s12s21_sr_mag = np.genfromtxt(path_par_s11 + "par_s12s21_sr_mag.txt")
    par_s12s21_sr_ang = np.genfromtxt(path_par_s11 + "par_s12s21_sr_ang.txt")

    par_s22_sr_mag = np.genfromtxt(path_par_s11 + "par_s22_sr_mag.txt")
    par_s22_sr_ang = np.genfromtxt(path_par_s11 + "par_s22_sr_ang.txt")

    fen = (f - 120) / 60

    # Evaluating S11 models at EDGES frequency
    s11_LNA_mag = mdl.model_evaluate("polynomial", par_s11_LNA_mag, fen)
    s11_LNA_ang = mdl.model_evaluate("polynomial", par_s11_LNA_ang, fen)

    s11_amb_mag = mdl.model_evaluate("fourier", par_s11_amb_mag, fen)
    s11_amb_ang = mdl.model_evaluate("fourier", par_s11_amb_ang, fen)

    s11_hot_mag = mdl.model_evaluate("fourier", par_s11_hot_mag, fen)
    s11_hot_ang = mdl.model_evaluate("fourier", par_s11_hot_ang, fen)

    s11_open_mag = mdl.model_evaluate("fourier", par_s11_open_mag, fen)
    s11_open_ang = mdl.model_evaluate("fourier", par_s11_open_ang, fen)

    s11_shorted_mag = mdl.model_evaluate("fourier", par_s11_shorted_mag, fen)
    s11_shorted_ang = mdl.model_evaluate("fourier", par_s11_shorted_ang, fen)

    s11_sr_mag = mdl.model_evaluate("polynomial", par_s11_sr_mag, fen)
    s11_sr_ang = mdl.model_evaluate("polynomial", par_s11_sr_ang, fen)

    s12s21_sr_mag = mdl.model_evaluate("polynomial", par_s12s21_sr_mag, fen)
    s12s21_sr_ang = mdl.model_evaluate("polynomial", par_s12s21_sr_ang, fen)

    s22_sr_mag = mdl.model_evaluate("polynomial", par_s22_sr_mag, fen)
    s22_sr_ang = mdl.model_evaluate("polynomial", par_s22_sr_ang, fen)

    # ----- Make these input parameters ??
    RMS_expec_mag = 0.0001  # 0.0001
    RMS_expec_ang = 1 * (np.pi / 180)  # 0.1

    # The following were obtained using the function "two_port_network_uncertainties()" also
    # contained in this file
    RMS_expec_mag_2port = 0.0002
    RMS_expec_ang_2port = 2 * (np.pi / 180)

    RMS_expec_mag_s12s21 = 0.0002
    RMS_expec_ang_s12s21 = 0.1 * (np.pi / 180)
    # ---------------------- LNA --------------------------------
    if MC_s11_syst[0] > 0:
        pert_mag = random_signal_perturbation(f, RMS_expec_mag, Npar_max)
        s11_LNA_mag = s11_LNA_mag + MC_s11_syst[0] * pert_mag

    if MC_s11_syst[1] > 0:
        pert_ang = random_signal_perturbation(f, RMS_expec_ang, Npar_max)
        s11_LNA_ang = s11_LNA_ang + MC_s11_syst[1] * pert_ang
    # ---------------------- Amb ---------------------------------
    if MC_s11_syst[2] > 0:
        pert_mag = random_signal_perturbation(f, RMS_expec_mag, Npar_max)
        s11_amb_mag = s11_amb_mag + MC_s11_syst[2] * pert_mag

    if MC_s11_syst[3] > 0:
        pert_ang = random_signal_perturbation(f, RMS_expec_ang, Npar_max)
        s11_amb_ang = s11_amb_ang + MC_s11_syst[3] * pert_ang
    # ---------------------- Hot ---------------------------------
    if MC_s11_syst[4] > 0:
        pert_mag = random_signal_perturbation(f, RMS_expec_mag, Npar_max)
        s11_hot_mag = s11_hot_mag + MC_s11_syst[4] * pert_mag

    if MC_s11_syst[5] > 0:
        pert_ang = random_signal_perturbation(f, RMS_expec_ang, Npar_max)
        s11_hot_ang = s11_hot_ang + MC_s11_syst[5] * pert_ang
    # ---------------------- Open --------------------------------
    if MC_s11_syst[6] > 0:
        pert_mag = random_signal_perturbation(f, RMS_expec_mag, Npar_max)
        s11_open_mag = s11_open_mag + MC_s11_syst[6] * pert_mag

    if MC_s11_syst[7] > 0:
        pert_ang = random_signal_perturbation(f, RMS_expec_ang, Npar_max)
        s11_open_ang = s11_open_ang + MC_s11_syst[7] * pert_ang
    # ---------------------- Shorted -----------------------------
    if MC_s11_syst[8] > 0:
        pert_mag = random_signal_perturbation(f, RMS_expec_mag, Npar_max)
        s11_shorted_mag = s11_shorted_mag + MC_s11_syst[8] * pert_mag

    if MC_s11_syst[9] > 0:
        pert_ang = random_signal_perturbation(f, RMS_expec_ang, Npar_max)
        s11_shorted_ang = s11_shorted_ang + MC_s11_syst[9] * pert_ang
    # ---------------------- S11 short cable -----------------------------------
    if MC_s11_syst[10] > 0:
        pert_mag = random_signal_perturbation(f, RMS_expec_mag_2port, Npar_max)
        s11_sr_mag = s11_sr_mag + MC_s11_syst[10] * pert_mag

    if MC_s11_syst[11] > 0:
        pert_ang = random_signal_perturbation(f, RMS_expec_ang_2port, Npar_max)
        s11_sr_ang = s11_sr_ang + MC_s11_syst[11] * pert_ang
    # ---------------------- S12S21 short cable -----------------------------------
    if MC_s11_syst[12] > 0:
        pert_mag = random_signal_perturbation(f, RMS_expec_mag_s12s21, Npar_max)
        s12s21_sr_mag = s12s21_sr_mag + MC_s11_syst[12] * pert_mag

    if MC_s11_syst[13] > 0:
        pert_ang = random_signal_perturbation(f, RMS_expec_ang_s12s21, Npar_max)
        s12s21_sr_ang = s12s21_sr_ang + MC_s11_syst[13] * pert_ang
    # ---------------------- S22 short cable -----------------------------------
    if MC_s11_syst[14] > 0:
        pert_mag = random_signal_perturbation(f, RMS_expec_mag_2port, Npar_max)
        s22_sr_mag = s22_sr_mag + MC_s11_syst[14] * pert_mag

    if MC_s11_syst[15] > 0:
        pert_ang = random_signal_perturbation(f, RMS_expec_ang_2port, Npar_max)
        s22_sr_ang = s22_sr_ang + MC_s11_syst[15] * pert_ang

    # Output
    # ------
    s11_LNA = s11_LNA_mag * (np.cos(s11_LNA_ang) + 1j * np.sin(s11_LNA_ang))
    s11_amb = s11_amb_mag * (np.cos(s11_amb_ang) + 1j * np.sin(s11_amb_ang))
    s11_hot = s11_hot_mag * (np.cos(s11_hot_ang) + 1j * np.sin(s11_hot_ang))
    s11_open = s11_open_mag * (np.cos(s11_open_ang) + 1j * np.sin(s11_open_ang))
    s11_shorted = s11_shorted_mag * (
        np.cos(s11_shorted_ang) + 1j * np.sin(s11_shorted_ang)
    )
    s11_sr = s11_sr_mag * (np.cos(s11_sr_ang) + 1j * np.sin(s11_sr_ang))
    s12s21_sr = s12s21_sr_mag * (np.cos(s12s21_sr_ang) + 1j * np.sin(s12s21_sr_ang))
    s22_sr = s22_sr_mag * (np.cos(s22_sr_ang) + 1j * np.sin(s22_sr_ang))

    return s11_LNA, s11_amb, s11_hot, s11_open, s11_shorted, s11_sr, s12s21_sr, s22_sr


def models_calibration_physical_temperature(
    case, f, s_parameters=np.zeros(1), MC_temp=np.zeros(4)
):
    if case == 1:
        # Paths
        path = (
            edges_folder
            + "/mid_band/calibration/receiver_calibration/receiver1/2018_01_25C"
            "/results/nominal/temp/"
        )

    # Physical temperatures
    phys_temp = np.genfromtxt(path + "physical_temperatures.txt")
    Ta = phys_temp[0] * np.ones(len(f))
    Th = phys_temp[1] * np.ones(len(f))
    To = phys_temp[2] * np.ones(len(f))
    Ts = phys_temp[3] * np.ones(len(f))

    # MC realizations of physical temperatures
    STD_temp = 0.1
    if MC_temp[0] > 0:
        Ta = Ta + MC_temp[0] * STD_temp * np.random.normal(0, 1)

    if MC_temp[1] > 0:
        Th = Th + MC_temp[1] * STD_temp * np.random.normal(0, 1)

    if MC_temp[2] > 0:
        To = To + MC_temp[2] * STD_temp * np.random.normal(0, 1)

    if MC_temp[3] > 0:
        Ts = Ts + MC_temp[3] * STD_temp * np.random.normal(0, 1)

    # S-parameters of hot load device
    if len(s_parameters) == 1:
        out = models_calibration_s11(case, f)
        rh = out[2]
        s11_sr = out[5]
        s12s21_sr = out[6]
        s22_sr = out[7]

    if len(s_parameters) == 4:
        rh = s_parameters[0]
        s11_sr = s_parameters[1]
        s12s21_sr = s_parameters[2]
        s22_sr = s_parameters[3]

    # reflection coefficient of termination
    rht = rc.gamma_de_embed(s11_sr, s12s21_sr, s22_sr, rh)

    # inverting the direction of the s-parameters,
    # since the port labels have to be inverted to match those of Pozar eqn 10.25
    s11_sr_rev = s22_sr

    # absolute value of S_21
    abs_s21 = np.sqrt(np.abs(s12s21_sr))

    # available power gain
    G = (
        (abs_s21 ** 2)
        * (1 - np.abs(rht) ** 2)
        / ((np.abs(1 - s11_sr_rev * rht)) ** 2 * (1 - (np.abs(rh)) ** 2))
    )

    # temperature
    Thd = G * Th + (1 - G) * Ta

    return np.array([Ta, Thd, To, Ts])
