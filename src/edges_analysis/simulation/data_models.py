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
    n_fgpar=5,
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
        n_21=N21par,
        n_fgpar=n_fgpar,
    )
    d = d_no_noise + noise

    return d, sigma, inv_sigma, det_sigma


def real_data(case, f_low, f_high, gap_f_low=0, gap_f_high=0):
    cases = {
        1: "one_day_tests/another_one.txt",
        2: "rcv18_sw18_nominal_GHA_every_1hr/integrated_spectrum_rcv18_sw18_every_1hr_GHA_6-18hr.txt",
    }
    data = np.genfromtxt(edges_folder + "mid_band/spectra/level5/" + cases[case])
    mask = (data[:, 0] >= f_low) & (data[:, 0] <= f_high)
    data = data[mask]

    # Possibility of removing from analysis the data range between f_low_gap and f_high_gap
    if gap_f_low > 0 and gap_f_high > 0:
        mask = (data[:, 0] <= gap_f_low) | (data[:, 0] >= gap_f_high)
        data = data[mask]

    data = data[data[:, 2] > 0]
    v, t, w, std_dev_vec = data.T

    sigma = np.diag(std_dev_vec ** 2)  # uncertainty covariance matrix
    inv_sigma = np.linalg.inv(sigma)
    det_sigma = np.linalg.det(sigma)

    return v, t, w, sigma, inv_sigma, det_sigma


def foreground_model(
    model_type, theta_fg, v, vr, ion_abs_coeff="free", ion_emi_coeff="free"
):
    n_params = len(theta_fg)

    model_fg = 0

    if model_type == "linlog":
        for i in range(n_params):
            model_fg += theta_fg[i] * ((v / vr) ** (-2.5)) * ((np.log(v / vr)) ** i)

    elif model_type == "powerlog":
        if ion_abs_coeff == "free":
            n_params -= 1
        if ion_emi_coeff == "free":
            n_params -= 1

        astro_exponent = sum(
            theta_fg[i + 1] * ((np.log(v / vr)) ** i) for i in range(n_params - 1)
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
    theta, v, vr, model_type_signal="exp", model_type_foreground="exp", n_21=4
):
    model_21 = 0
    if n_21:
        model_21 = signal_model(model_type_signal, theta[:n_21], v)

    # Foreground model
    model_fg = foreground_model(
        model_type_foreground, theta[n_21:], v, vr, ion_abs_coeff=0, ion_emi_coeff=0
    )

    return model_21 + model_fg


def spectrum_channel_to_channel_difference(f, t, w, f_low, f_high, n_fg=5):
    mask = (f >= f_low) & (f <= f_high)
    par = mdl.fit_polynomial_fourier("LINLOG", f[mask], t[mask], n_fg, Weights=w[mask])
    r = t[mask] - par[1]
    return np.diff(r)


def svd_functions(folder_with_spectra, f_low, f_high, remove_avg=True):
    """Compute SVD functions"""

    # Listing files to be processed
    folder = edges_folder + "mid_band/spectra/level5/" + folder_with_spectra + "/"
    list_of_spectra = sorted(listdir(folder))

    # Generate the original matrix of spectra
    for i in range(len(list_of_spectra)):
        filename = list_of_spectra[i]
        d = np.genfromtxt(folder + filename)

        if i == 0:
            A = d[:, 1]
            f = d[:, 0]
        else:
            A = np.vstack((A, d[:, 1]))

    # Cut to desired frequency range
    mask = (f >= f_low) & (f <= f_high)
    f = f[mask]
    A = A[:, mask]

    # Remove channels with no data
    P = np.prod(A, axis=0)
    f = f[P > 0]
    A = A[:, P > 0]

    if remove_avg:
        avA = np.mean(A, axis=0)
        C = A - avA
    else:
        flag = 0
        for j in range(len(list_of_spectra) - 1):
            for i in range(len(list_of_spectra) - 1 - j):

                k = i + (j + 1)
                B = A[k] - A[j]

                if not flag:
                    C = B
                    flag = True
                else:
                    C = np.vstack((C, B))

    # SVD
    u, eig_values, eig_functions = np.linalg.svd(C)

    return f, eig_values, eig_functions


def signal_edges2018_uncertainties(v):
    # Nominal model reported in Bowman et al. (2018)
    model_nominal = signal_model("exp", [-0.5, 78, 19, 7], v)

    # Compute distribution of models and limits
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


def models_calibration_spectra(path_par_spec, f, MC_spectra_noise=(0, 0, 0, 0)):
    # Loading parameters
    par_spec = {
        kind: np.genfromtxt(path_par_spec + f"par_spec_{kind}.txt")
        for kind in ["amb", "hot", "open", "shorted"]
    }
    RMS_spec = np.genfromtxt(path_par_spec + "RMS_spec.txt")

    # Normalized frequency
    fn = (f - 120) / 60

    # Evaluating models
    models = {
        kind: mdl.model_evaluate("fourier", p, fn) for kind, p in par_spec.items()
    }

    # Adding noise to models
    for i, (kind, model) in models.items():
        if MC_spectra_noise[i] > 0:
            model += MC_spectra_noise[i] * RMS_spec[i] * np.random.normal(size=len(fn))

    return np.array(list(models.values()))


def random_signal_perturbation(f, rms_expectation, n_par_max):
    # Choose randomly, from a Uniform distribution, an integer corresponding to the number of
    # polynomial terms
    n_par = np.random.choice(n_par_max + 1)

    # Choose randomly, from a Gaussian distribution, a number corresponding to the rms of the
    # perturbation across frequency
    rms = rms_expectation * np.random.normal()

    # Generate random noise across frequency, with MEAN=0 and STD=1
    noise = np.random.normal(size=len(f))

    # Fit a generic polynomial to the noise, using a polynomial with n_par terms
    par = np.polyfit(f, noise, n_par + 1)
    model = np.polyval(par, f)

    # Compute the current rms of the polynomial
    rms_model = np.std(model)

    return (rms / rms_model) * model


def two_port_network_uncertainties():
    """
    This function propagates the uncertainty in 1-port measurements
    to uncertainties in the 2-port S-parameters of the short cable between the hot load and the
    receiver
    """
    # Reflection standard models
    f = np.arange(50, 181)  # In MHz
    resistance_of_match = 50.1

    oa, sa, la = rc.agilent_85033E(
        (10 ** 6) * f, resistance_of_match, m=1, md_value_ps=38
    )

    # Simulated measurements at the VNA input
    one_port = [np.ones(len(f)), -np.ones(len(f)), 0.001 * np.ones(len(f))]

    # S-parameters of VNA calibration network
    X, s11V, s12s21V, s22V = rc.de_embed(oa, sa, la, *one_port, one_port[0])

    # Simulated measurements at the end of the 2-port network
    # Load 2-port parameters and models
    path_par_s11 = (
        edges_folder + "/mid_band/calibration/receiver_calibration/receiver1"
        "/2018_01_25C/results/nominal/s11/"
    )

    sxx = get_reflections(f, path_par_s11)

    vna = {}
    for thing, a in zip(["open", "short", "load"], [oa, sa, la]):
        # Measurements as seen at the input of 2-port network
        meas_2port = rc.gamma_shifted(sxx["s11"], sxx["s12s21"], sxx["s22"], a)

        # Measurements at the uncalibrated VNA port
        vna[thing] = rc.gamma_shifted(s11V, s12s21V, s22V, meas_2port)

    # Simulating and propagating uncertainties
    # ----------------------------------------
    N = 10000  # MC repetitions

    RMS_mag = 0.0001  # magnitude uncertainty STD
    RMS_ang = 0.1  # phase uncertainty STD
    Npar_max = 15  # maximum number of polynomial terms

    s11_N = np.zeros((N, len(f))) + 1j * 0
    s12s21_N = np.zeros((N, len(f))) + 1j * 0
    s22_N = np.zeros((N, len(f))) + 1j * 0

    # Add perturbations to measurements
    for i in range(N):
        perturbed = {}
        for key in vna.keys():
            perturbed[key] = {}
            for thing in zip(["one", "two"], [one_port, vna.values()]):
                for meas in thing:
                    # Open at VNA port
                    pert_mag = random_signal_perturbation(f, RMS_mag, Npar_max)
                    pert_ang = random_signal_perturbation(f, RMS_ang, Npar_max)

                    mag = np.abs(meas) + pert_mag
                    ang = np.unwrap(np.angle(meas)) + (np.pi / 180) * pert_ang

                    perturbed[key][thing] = mag * (np.cos(ang) + 1j * np.sin(ang))

        out = {}
        for key in vna.keys():
            out[key] = rc.de_embed(
                oa,
                sa,
                la,
                perturbed[key]["one"],
                perturbed[key]["one"],
                perturbed[key]["one"],
                perturbed[key]["two"],
            )[0]

        # Compute S-parameters of DUT
        l, s11new, s12s21new, s22new = rc.de_embed(
            oa, sa, la, out["open"], out["short"], out["load"], out["load"]
        )

        # Store S-parameters of DUT
        s11_N[i, :] = s11new
        s12s21_N[i, :] = s12s21new
        s22_N[i, :] = s22new

    return f, sxx, s11_N, s12s21_N, s22_N


def get_reflections(f, path_par_s11, kinds=("s11", "s12s21", "s22"), loadname="sr"):
    sxx = {}
    fen = (f - 120) / 60
    for kind in kinds:
        for part in ["mag", "ang"]:
            pars = np.genfromtxt(path_par_s11 + f"par_{kind}_{loadname}_{part}.txt")
            model = mdl.model_evaluate("polynomial", pars, fen)

            if part == "mag":
                sxx[kind] = model
            else:
                sxx[kind] *= np.cos(model) + 1j * np.sin(model)
    return sxx


def MC_receiver(
    band, MC_spectra_noise=np.ones(4), MC_s11_syst=np.ones(16), MC_temp=np.ones(4)
):
    s11_Npar_max = 14

    f = EdgesFrequencyRange(f_low=50, f_high=150).freq
    fn = (f - 120) / 60

    Tamb_internal = 300

    cterms = 7
    wterms = 8

    # Spectra
    if np.sum(MC_spectra_noise) == 0:
        Tunc = np.genfromtxt(
            edges_folder
            + "mid_band/calibration/receiver_calibration/receiver1/2018_01_25C"
            "/results/nominal/data/average_spectra_300_350.txt"
        )
        ms = Tunc[:, 1:5].T
    else:
        ms = models_calibration_spectra(
            (
                edges_folder
                + "/mid_band/calibration/receiver_calibration/receiver1/2018_01_25C/results"
                "/nominal/spectra/"
            ),
            f,
            MC_spectra_noise=MC_spectra_noise,
        )

    # S11
    mr = models_calibration_s11(
        (
            edges_folder
            + "/mid_band/calibration/receiver_calibration/receiver1/2018_01_25C/results"
            "/nominal/s11/"
        ),
        f,
        MC_s11_syst=MC_s11_syst,
        Npar_max=s11_Npar_max,
    )

    # Physical temperature
    mt = models_calibration_physical_temperature(
        (
            edges_folder
            + "/mid_band/calibration/receiver_calibration/receiver1/2018_01_25C"
            "/results/nominal/temp/"
        ),
        f,
        s_parameters=[mr[2], mr[5], mr[6], mr[7]],
        MC_temp=MC_temp,
    )

    rl = mr[0]

    # Computing receiver calibration quantities
    C1, C2, TU, TC, TS = rcf.get_calibration_quantities_iterative(
        fn,
        T_raw={"ambient": ms[0], "hot_load": ms[1], "short": ms[2], "open": ms[3]},
        gamma_rec=mr[0],
        gamma_ant={"ambient": mr[1], "hot_load": mr[2], "short": mr[4], "open": mr[3]},
        T_ant={"ambient": mt[0], "hot_load": mt[1], "short": mt[3], "open": mt[2]},
        cterms=cterms,
        wterms=wterms,
        Tamb_internal=Tamb_internal,
    )

    return f, rl, C1, C2, TU, TC, TS


def MC_antenna_s11(f, rant, s11_Npar_max=14):
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
    dG = RMS_G * 10
    while np.max(np.abs(dG)) > 6 * RMS_G:
        dG = random_signal_perturbation(f, RMS_G, G_Npar_max)

    return dG


def MC_error_propagation(f_low=61, f_high=121, band="mid_band"):

    f, rl, C1, C2, TU, TC, TS = MC_receiver(
        band,
        MC_spectra_noise=np.zeros(4),
        MC_s11_syst=np.zeros(16),
        MC_temp=np.zeros(4),
    )

    mask = (f >= f_low) & (f <= f_high)
    f = f[mask]
    rl = rl[mask]
    C1 = C1[mask]
    C2 = C2[mask]
    TU = TU[mask]
    TC = TC[mask]
    TS = TS[mask]

    Tamb = 273 + 25

    gamma_ant, G, tsky = [], [], []
    for i, (day, temp) in enumerate([147, 227], [1000, 3000]):
        gamma_ant.append(
            models_antenna_s11_remove_delay(
                "mid_band",
                f,
                year=2018,
                day=day,
                delay_0=0.17,
                model_type="polynomial",
                Nfit=14,
                plot_fit_residuals=False,
            )[mask]
        )

        Gb, Gc = balun_and_connector_loss("mid_band", f, gamma_ant[i])
        G.append((Gb * Gc)[mask])
        tsky.append(temp * (f / 100) ** -2.5)

    tunc = []
    for i in range(2):
        for j in range(2):
            tskyl1 = tsky[j] * G[i] + Tamb * (1 - G[i])
            tunc.append(
                rcf.uncalibrated_antenna_temperature(
                    tskyl1, gamma_ant[j], rl, C1, C2, TU, TC, TS, T_load=300
                )
            )

    ff, rl, C1, C2, TU, TC, TS = MC_receiver(
        band,
        MC_spectra_noise=np.zeros(4),
        MC_s11_syst=np.zeros(16),
        MC_temp=np.zeros(4),
    )

    rl_MC = rl[mask]
    C1_MC = C1[mask]
    C2_MC = C2[mask]
    TU_MC = TU[mask]
    TC_MC = TC[mask]
    TS_MC = TS[mask]

    gamma_ant_mc, G_mc = [], []
    for i in range(2):
        gamma_ant_y = MC_antenna_s11(ff, gamma_ant[i], s11_Npar_max=14)
        dG_y = MC_antenna_loss(ff, G_Npar_max=10)
        G_y = G[i] + dG_y
        gamma_ant_mc.append(gamma_ant_y[mask])
        G_mc.append(G_y[mask])

    tcal = []
    for i in range(4):
        tc = rcf.calibrated_antenna_temperature(
            tunc[i],
            gamma_ant_mc[i % 2],
            rl_MC,
            C1_MC,
            C2_MC,
            TU_MC,
            TC_MC,
            TS_MC,
            T_load=300,
        )
        tcal.append((tc - Tamb * (1 - G_mc[i % 2])) / G_mc[i % 2])

    n_fg = 1
    par = [mdl.fit_polynomial_fourier("LINLOG", f, tc, n_fg)[1] for tc in tcal]
    return (f, *gamma_ant, *tcal, *par)


def models_calibration_s11(
    path_par_s11,
    f,
    MC_s11_syst=np.zeros(16),
    Npar_max=15,
    rms_expected=(0.0001, np.pi / 180),
):
    # Loading S11 parameters
    sr = get_reflections(f, path_par_s11)
    loads = {
        name: get_reflections(f, path_par_s11, kinds=("s11",), loadname="amb")
        for name in ["LNA", "amb", "hot", "open", "shorted"]
    }

    # The following were obtained using the function "two_port_network_uncertainties()" also
    # contained in this file
    rms_expec_2port = (0.0002, 2 * (np.pi / 180))
    rms_expec_s12s21 = (0.0002, 0.1 * (np.pi / 180))

    j = 0
    for name in loads:
        this = []
        for i, fnc in enumerate(lambda x: np.abs(x), lambda x: np.angle(x)):  # ang/mag
            if MC_s11_syst[j] > 0:
                pert = random_signal_perturbation(f, rms_expected[i], Npar_max)
                this.append(fnc(loads[name]["s11"]) + MC_s11_syst[j] * pert)
            j += 1
        loads[name] = this[0] * (np.cos(this[1]) + 1j * np.sin(this[1]))

    for name, expec in zip(sr, [rms_expec_2port, rms_expec_s12s21, rms_expec_2port]):
        this = []
        for i, fnc in enumerate(lambda x: np.abs(x), lambda x: np.angle(x)):  # ang/mag
            if MC_s11_syst[j] > 0:
                pert = random_signal_perturbation(f, expec[i], Npar_max)
                this.append(fnc(sr[name]) + MC_s11_syst[j] * pert)
            j += 1
        sr[name] = this[0] * (np.cos(this[1]) + 1j * np.sin(this[1]))

    return (
        loads["LNA"],
        loads["amb"],
        loads["hot"],
        loads["open"],
        loads["shorted"],
        sr["s11"],
        sr["s12s21"],
        sr["s22"],
    )


def models_calibration_physical_temperature(
    path, f, s_parameters=np.zeros(1), MC_temp=np.zeros(4)
):

    # Physical temperatures
    phys_temp = np.genfromtxt(path + "physical_temperatures.txt")
    T = [temp * np.ones(len(f)) for temp in phys_temp]
    # Ta = phys_temp[0] * np.ones(len(f))
    # Th = phys_temp[1] * np.ones(len(f))
    # To = phys_temp[2] * np.ones(len(f))
    # Ts = phys_temp[3] * np.ones(len(f))

    # MC realizations of physical temperatures
    std_temp = 0.1
    for i, t in enumerate(T):
        if MC_temp[i] > 0:
            # TODO: should the random here be of size len(f)?
            t += MC_temp[i] * std_temp * np.random.normal()

    # S-parameters of hot load device
    if len(s_parameters) == 1:
        out = models_calibration_s11(1, f)
        rh = out[2]
        s11_sr = out[5]
        s12s21_sr = out[6]
        s22_sr = out[7]
    elif len(s_parameters) == 4:
        rh, s11_sr, s12s21_sr, s22_sr = s_parameters
    else:
        raise ValueError("s_parameters has wrong size!")

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
    T[1] = G * T[1] + (1 - G) * T[0]

    return np.array(T)
