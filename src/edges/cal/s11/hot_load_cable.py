"""Functions for determining s-parameters of the hot-load calibration cable.

This can be used to compute the loss through the cable.
"""

import numpy as np
from astropy import units as un

from edges.cal.s11.base import CalibratedSParams

from ... import get_data_path
from ... import types as tp
from ...frequencies import get_mask


def read_semi_rigid_cable_sparams_file(
    path: tp.PathLike = ":semi_rigid_s_parameters_WITH_HEADER.txt",
    f_low: tp.FreqType = 0 * un.MHz,
    f_high: tp.FreqType = np.inf * un.MHz,
):
    """Instantiate the HotLoadCorrection from file.

    Parameters
    ----------
    path
        Path to the S-parameters file.
    f_low, f_high
        The min/max frequencies to use in the modelling.
    """
    path = get_data_path(path)

    data = np.genfromtxt(path)
    mask = get_mask(data[:, 0] * un.MHz, low=f_low, high=f_high)
    data = data[mask]
    freq = data[:, 0] * un.MHz

    if data.shape[1] == 7:  # Original file from 2015
        data = data[:, 1::2] + 1j * data[:, 2::2]
    elif data.shape[1] == 6:  # File from 2017
        data = np.array([
            data[:, 1] + 1j * data[:, 2],
            data[:, 3],
            data[:, 4] + 1j * data[:, 5],
        ]).T

    return CalibratedSParams(freqs=freq, s11=data[:, 0], s12=data[:, 1], s22=data[:, 2])
