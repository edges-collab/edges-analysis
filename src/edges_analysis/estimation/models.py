import numpy as np


def model_mirocha2016(theta, pars=None, hmf=None, src=None):
    """
    Usage:

    z, t = model_mirocha2016([np.log10(1e-2), np.log10(1e4), np.log10(1e-2), np.log10(1e40), 20,
    -0.5])
    """
    import ares

    if pars is None:
        pars = ares.util.ParameterBundle("mirocha2016:dpl")
    if hmf is None:
        hmf = ares.physics.HaloMassFunction()
    if src is None:
        src = ares.sources.SynthesisModel(source_Z=0.024, source_sed="eldridge2009")

    # Assign new values to list
    theta_list = {
        "pop_Z{0}": theta[0],  # log10(1e-3, 0.04),
        "pop_Tmin{0}": theta[1],  # log10(500, 5e5),
        "pop_fesc{0}": theta[2],  # log10(1e-3, 0.5),
        "pop_rad_yield{1}": theta[3],  # log10(1e38, 1e42),
        "pop_logN{1}": theta[4],  # (17, 23),
        "pop_rad_yield_Z_index{1}": theta[5],  # (-1, 0),
    }

    # List that defines if parameter is evaluated in log scale or not
    is_log = {
        "pop_Z{0}": True,
        "pop_Tmin{0}": True,
        "pop_fesc{0}": True,
        "pop_rad_yield{1}": True,
        "pop_logN{1}": False,
        "pop_rad_yield_Z_index{1}": False,
    }

    # New list of parameter values
    updates = {
        par: (10 ** val if is_log[par] else val) for par, val in theta_list.items()
    }

    # Create new parameter object
    p = pars.copy()

    # Assign star properties (loaded at the top of the file, IS THIS NECESSARY???????, or can we
    # DO IT ONLY AT THE TOP)
    p["hmf_instance"] = hmf
    p["pop_psm_instance"] = src  # 'psm' is for "population synthesis model"

    # Assign input values to parameter object
    p.update(updates)

    # Run simulation
    sim = ares.simulations.Global21cm(**p)
    sim.run()

    # Extracting brightness temperature and redshift
    hist = sim.history
    t = hist["dTb"]
    z = hist["z"]

    return z, t


def model_eor_flattened_gaussian(v, model_type=1, T21=1, vr=75, dv=20, tau0=4, tilt=0):
    if model_type == 1:
        b = -np.log(-np.log((1 + np.exp(-tau0)) / 2) / tau0)
        K1 = T21 * (1 - np.exp(-tau0 * np.exp((-b * (v - vr) ** 2) / ((dv ** 2) / 4))))
        K2 = 1 + (tilt * (v - vr) / dv)
        K3 = 1 - np.exp(-tau0)
        T = K1 * K2 / K3
    elif model_type == 2:
        tau1 = tau0 if tilt == 0 else tilt
        K1 = np.tanh((1 / (v + dv / 2) - 1 / vr) / (dv / (tau0 * (vr ** 2))))
        K2 = np.tanh((1 / (v - dv / 2) - 1 / vr) / (dv / (tau1 * (vr ** 2))))
        T = -(T21 / 2) * (K1 - K2)
    else:
        raise ValueError("model_type must be 1 or 2")
    return T  # The amplitude is equal to T21, not to -T21


def model_flattened_gaussian_linlog(theta, v, v0, n_fg, n_21=4):
    model_21 = model_eor_flattened_gaussian(model_type=1, *theta[:n_fg])

    model_fg = 0
    if n_fg:
        for i in range(n_fg):
            model_fg += theta[n_21 + i] * ((v / v0) ** (-2.5)) * ((np.log(v / v0)) ** i)

    return model_21 + model_fg
