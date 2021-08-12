"""An example of running polychord for EDGES data."""
import PyPolyChord
import numpy as np
from PyPolyChord.settings import PolyChordSettings
from edges_analysis.estimation.models import model
import scipy as sp


def simulated_data(theta, v, v0):
    """Simulate some data."""
    std_dev_vec = 0.03 * (v / v0) ** (-2.5)

    sigma = np.diag(std_dev_vec ** 2)  # uncertainty covariance matrix
    inv_sigma = sp.linalg.inv(sigma)
    det_sigma = np.linalg.det(sigma)

    noise = np.random.multivariate_normal(np.zeros(len(v)), sigma)

    d_no_noise = model(theta)
    d = d_no_noise + noise

    return v, d, sigma, inv_sigma, det_sigma


def prior(cube: np.typing.ArrayLike) -> list:
    """
    Define the tranform between the unit hypercube to the true parameters.

    Parameters
    ----------
    cube
        a list containing the parameters as drawn from a unit hypercube.

    Returns
    -------
    params
        the transformed parameters.
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


def dumper(live, dead, logweights, log_z, log_z_err):
    """How to dump stuff."""
    print(dead[-1])


if __name__ == "__main__":

    v = np.arange(60, 151, 1)
    v0 = 100

    N21 = 4
    n_fg = 5

    Nparameters = N21 + n_fg
    Nderived = 0

    # flattened gaussian
    v, d, sigma, inv_sigma, det_sigma = simulated_data(
        [-0.5, 78, 20, 7, 1000, 1, 1, -1, 4], v, v0
    )

    def loglikelihood(theta):
        """The log-likelihood."""
        N = len(v)

        # Evaluating model
        m = model(theta)

        # Log-likelihood
        DELTA = d - m
        lnL2 = (
            -(1 / 2) * np.dot(np.dot(DELTA, inv_sigma), DELTA)
            - (N / 2) * np.log(2 * np.pi)
            - (1 / 2) * np.log(det_sigma)
        )

        # This solves numerical errors
        if np.isnan(lnL2):
            lnL2 = -np.infty

        return lnL2, 0

    def run(root_name):
        """Run the function."""
        settings = PolyChordSettings(Nparameters, Nderived)
        settings.base_dir = "/home/raul/Desktop/"
        settings.file_root = root_name
        settings.do_clustering = True
        settings.read_resume = False
        PyPolyChord.run_polychord(
            loglikelihood, Nparameters, Nderived, settings, prior, dumper
        )

    run("example")
