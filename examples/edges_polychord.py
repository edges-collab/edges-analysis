import numpy as np
import PyPolyChord
from PyPolyChord.settings import PolyChordSettings

from edges_analysis.estimation.models import model
from edges_analysis.estimation.tools import dumper, prior, simulated_data

if __name__ == "__main__":

    v = np.arange(60, 151, 1)
    v0 = 100

    N21 = 4
    Nfg = 5

    Nparameters = N21 + Nfg
    Nderived = 0

    # flattened gaussian
    v, d, sigma, inv_sigma, det_sigma = simulated_data(
        [-0.5, 78, 20, 7, 1000, 1, 1, -1, 4], v, v0
    )

    def loglikelihood(theta):
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
        # in python, or ipython >>

        settings = PolyChordSettings(Nparameters, Nderived)
        settings.base_dir = "/home/raul/Desktop/"
        settings.file_root = root_name
        settings.do_clustering = True
        settings.read_resume = False
        PyPolyChord.run_polychord(
            loglikelihood, Nparameters, Nderived, settings, prior, dumper
        )

    run("example")
