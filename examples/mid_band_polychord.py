#!/usr/bin/python
"""
An example of running polychord for mid-band data.

How to run::

    python mid_band_polychord.py 0 5
"""

import sys
from os import makedirs
from os.path import exists

import numpy as np
import PyPolyChord
from PyPolyChord.settings import PolyChordSettings

from edges_analysis.estimate.tools import dumper
from edges_analysis.simulation import data_models as dm
from edges_analysis.config import config


def prior_list(N21, n_fg, model_type_signal, model_type_foreground):
    pl = np.zeros((Nparameters, 2))
    pl[:, 0] = -1e4
    pl[:, 1] = +1e4

    if model_type_signal in ["exp", "tanh"] and N21 >= 4:

        # Amplitude
        pl[0, 0] = -2
        pl[0, 1] = 2  # 0

        # Center
        pl[1, 0] = 58  # 61
        pl[1, 1] = 128  # 149 #119  #100

        # Width
        pl[2, 0] = 2
        pl[2, 1] = 70  # 60 #30

        # Tau
        pl[3, 0] = 0.01
        pl[3, 1] = 20

        if (model_type_signal == "exp") and (N21 == 5):
            # Tau
            pl[4, 0] = -10
            pl[4, 1] = 10

        if (model_type_signal == "tanh") and (N21 == 5):
            # Tau
            pl[4, 0] = 0.01
            pl[4, 1] = 20

    if model_type_foreground == "linlog":
        # Temperature at reference frequency
        pl[N21, 0] = 100  # lower limit of first parameter, temperature at 100 MHz
        pl[N21, 1] = 10000  # upper limit of first parameter, temperature at 100 MHz
    elif model_type_foreground == "powerlog":
        # Temperature at reference frequency
        pl[N21, 0] = 100  # lower limit of first parameter, temperature at 100 MHz
        pl[N21, 1] = 10000  # upper limit of first parameter, temperature at 100 MHz

        # Spectral index
        pl[N21 + 1, 0] = -2.0
        pl[N21 + 1, 1] = -3.0
    return pl


def loglikelihood(theta):
    N = len(v)

    # Evaluating model
    m = dm.full_model(
        theta,
        v,
        v0,
        model_type_signal=model_type_signal,
        model_type_foreground=model_type_foreground,
        n_21=N21,
        n_fgpar=n_fg,
    )

    # Log-likelihood
    DELTA = t - m
    lnL2 = -(1 / 2) * np.dot(np.dot(DELTA, inv_sigma), DELTA) - (N / 2) * np.log(
        2 * np.pi
    )  # -(1/2)*np.log(det_sigma)
    # lnL2 =  #-(1/2)*np.log(det_sigma)

    # This solves numerical errors
    if np.isnan(lnL2):
        print("True")
        lnL2 = -np.infty

    return lnL2, 0


def prior(cube):
    """

    A function defining the transform between the parameterisation in the unit hypercube to the
    true parameters.

    Args: cube (array, list): a list containing the parameters as drawn from a unit hypercube.

    Returns:
    list: the transformed parameters.

    """

    theta = np.zeros(len(cube))

    pl = prior_list(N21, n_fg, model_type_signal, model_type_foreground)

    for i in range(len(cube)):
        theta[i] = cube[i] * (pl[i, 1] - pl[i, 0]) + pl[i, 0]

    return theta


def run():
    settings = PolyChordSettings(Nparameters, Nderived)
    settings.base_dir = save_folder
    settings.file_root = save_file_name
    settings.do_clustering = True
    settings.read_resume = False
    PyPolyChord.run_polychord(
        loglikelihood, Nparameters, Nderived, settings, prior, dumper
    )


if __name__ == "__main__":
    # Input parameters
    # -----------------------
    save_folder = (
        config["edges_folder"] + "mid_band/polychord/20190910/case101_GHA_6-18hr"
        "/foreground_linlog_5par_signal_exp_4par/"
    )

    data = "real"  # it could be 'real' or 'simulated'
    case = 101  # 0=nominal
    f_low = 58  # 58
    f_high = 118  # 128
    v0 = 90

    n_fg = int(sys.argv[1])
    N21 = int(sys.argv[2])

    gap_f_low = 0  # nominal value: 0
    gap_f_high = 0  # nominal_value: 0

    model_type_foreground = "linlog"  # , 'linlog', 'powerlog'
    model_type_signal = "exp"  # 'exp'  #, 'tanh'

    if not exists(save_folder):
        makedirs(save_folder)
    save_file_name = "chain"  # sys.argv[3]

    # Constants
    # -----------------------
    Nparameters = N21 + n_fg
    Nderived = 0

    # Data
    # -------------------------------------------------
    # Choose to work either with simulated or real data
    if data == "simulated":
        v = np.arange(61, 159, 0.39)

        # t, sigma, inv_sigma, det_sigma = dm.simulated_data([-0.5, 78, 19, 7, 1000, -2.5, -0.1, 1,
        # 1], v, v0, 0.02, model_type_signal='exp', model_type_foreground='exp', n_21=4,
        # n_fgpar=5)
        t, sigma, inv_sigma, det_sigma = dm.simulated_data(
            [1000, -2.5, -0.1, 1, 1],
            v,
            v0,
            0.01,
            model_type_signal="exp",
            model_type_foreground="exp",
            N21par=0,
            n_fgpar=5,
        )

    elif data == "real":
        v, t, w, sigma, inv_sigma, det_sigma = dm.real_data(
            case, f_low, f_high, gap_f_low=gap_f_low, gap_f_high=gap_f_high
        )

    run()
