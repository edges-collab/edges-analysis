import numpy as np
import scipy as sp
from src.edges_analysis.estimation.models import model


def simulated_data(theta, v, v0):
    std_dev_vec = 0.03 * (v / v0) ** (-2.5)

    sigma = np.diag(std_dev_vec ** 2)  # uncertainty covariance matrix
    inv_sigma = sp.linalg.inv(sigma)
    det_sigma = np.linalg.det(sigma)

    noise = np.random.multivariate_normal(np.zeros(len(v)), sigma)

    d_no_noise = model(theta)
    d = d_no_noise + noise

    return v, d, sigma, inv_sigma, det_sigma


def prior(cube):
    """
    A function defining the tranform between the parameterisation in the unit hypercube to the
    true parameters.

    Args: cube (array, list): a list containing the parameters as drawn from a unit hypercube.

    Returns:
    list: the transformed parameters.
    """
    # Unpack the parameters (in their unit hypercube form)
    T21_prime = cube[0]
    vr_prime = cube[1]
    dv_prime = cube[2]
    tau_prime = cube[3]

    a0_prime = cube[4]
    a1_prime = cube[5]
    a2_prime = cube[6]
    a3_prime = cube[7]
    a4_prime = cube[8]

    T21_min = -10  # lower bound on uniform prior
    T21_max = 10  # upper bound on uniform prior

    vr_min = 60  # lower bound on uniform prior
    vr_max = 150  # upper bound on uniform prior

    dv_min = 2  # lower bound on uniform prior
    dv_max = 100  # upper bound on uniform prior

    tau_min = 0  # lower bound on uniform prior
    tau_max = 30  # upper bound on uniform prior

    a0_min = 900  # lower bound on uniform prior
    a0_max = 1100  # upper bound on uniform prior

    a1_min = -1e4  # lower bound on uniform prior
    a1_max = 1e4  # upper bound on uniform prior

    a2_min = -1e4  # lower bound on uniform prior
    a2_max = 1e4  # upper bound on uniform prior

    a3_min = -1e4  # lower bound on uniform prior
    a3_max = 1e4  # upper bound on uniform prior

    a4_min = -1e4  # lower bound on uniform prior
    a4_max = 1e4  # upper bound on uniform prior

    T21 = T21_prime * (T21_max - T21_min) + T21_min
    vr = vr_prime * (vr_max - vr_min) + vr_min
    dv = dv_prime * (dv_max - dv_min) + dv_min
    tau = tau_prime * (tau_max - tau_min) + tau_min

    a0 = a0_prime * (a0_max - a0_min) + a0_min
    a1 = a1_prime * (a1_max - a1_min) + a1_min
    a2 = a2_prime * (a2_max - a2_min) + a2_min
    a3 = a3_prime * (a3_max - a3_min) + a3_min
    a4 = a4_prime * (a4_max - a4_min) + a4_min

    return [T21, vr, dv, tau, a0, a1, a2, a3, a4]


def dumper(live, dead, logweights, logZ, logZerr):
    print(dead[-1])
