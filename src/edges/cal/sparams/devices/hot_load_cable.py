"""Functions for determining S-parameters of the hot-load calibration cable.

This can be used to compute the loss through the cable.
"""

import numpy as np
from astropy import units as un

from edges import get_data_path
from edges import types as tp
from edges.frequencies import get_mask
from edges.modeling import ComplexRealImagModel, Polynomial, UnitTransform

from .. import S11ModelParams, SParams


def read_semi_rigid_cable_sparams_file(
    path: tp.PathLike = ":semi_rigid_s_parameters_WITH_HEADER.txt",
    f_low: tp.FreqType = 0 * un.MHz,
    f_high: tp.FreqType = np.inf * un.MHz,
):
    """Read a semi-rigid cable S-parameters file.

    This file is simply a whitespace-separated text file with frequency in MHz
    in the first column, and the S-parameters in the subsequent columns as
    real and imaginary parts. It can have either 6 or 7 columns (the latter
    includes a header row).

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

    return SParams(freqs=freq, s11=data[:, 0], s12=data[:, 1], s22=data[:, 2])


def hot_load_cable_model_params(**kwargs) -> S11ModelParams:
    """Get default model parameters for the hot load cable S11 model."""
    model = kwargs.pop(
        "model", Polynomial(n_terms=21, transform=UnitTransform(range=(0, 1)))
    )

    return S11ModelParams(
        model=model,
        complex_model_type=ComplexRealImagModel,
        set_transform_range=True,
        **kwargs,
    )
