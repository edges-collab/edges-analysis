import numpy as np
from edges_cal import receiver_calibration_func as rcf

from ..analysis import loss
from ..analysis.s11 import antenna_s11_remove_delay
from ..config import config


def test(band, s11_path, receiver_cal_file=1, f_low=60, f_high=160, antenna_s11_nfit=5):

    # Receiver calibration quantities
    # -------------------------------
    rcv_file = (
        config["edges_folder"]
        + band
        + "/calibration/receiver_calibration/receiver1/2018_01_25C/results/nominal/calibration_files/"
    )

    if receiver_cal_file == 1:
        rcv_file += "calibration_file_mid_band_cfit6_wfit14.txt"
    elif receiver_cal_file == 2:
        rcv_file += "calibration_file_mid_band_cfit10_wfit14.txt"
    else:
        raise ValueError("receiver_cal_file must be 1 or 2")

    rcv = np.genfromtxt(rcv_file)

    fx = rcv[:, 0]
    rcv2 = rcv[(fx >= f_low) & (fx <= f_high), :]

    f = rcv2[:, 0]
    rl = rcv2[:, 1] + 1j * rcv2[:, 2]
    C1, C2, TU, TC, TS = rcv2[:, 3:]

    # Antenna S11
    # -----------
    ra = antenna_s11_remove_delay(
        s11_path,
        f,
        delay_0=0.17,
        n_fit=antenna_s11_nfit,
    )

    # Balun+Connector Loss
    # --------------------
    Gb, Gc = loss.balun_and_connector_loss(f, ra)
    G = Gb * Gc

    # Ambient temperature
    # -------------------
    t_amb = 273.15 + 25

    # Generating calibrated input data
    f0 = 100  # MHz
    t_b = 1000 * (f / f0) ** (-2.5)

    # Uncalibrating input data
    t_lb = G * t_b + (1 - G) * t_amb
    t_3p = rcf.uncalibrated_antenna_temperature(t_lb, ra, rl, C1, C2, TU, TC, TS)

    # Perturbed calibration quantities
    C1 *= 1 + 0.1

    # Calibrated uncalibrated data
    t_lb = rcf.calibrated_antenna_temperature(t_3p, ra, rl, C1, C2, TU, TC, TS)
    return f, t_b, (t_lb - t_amb * (1 - G)) / G
