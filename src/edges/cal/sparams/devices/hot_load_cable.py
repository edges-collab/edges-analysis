"""Functions for determining S-parameters of the hot-load calibration cable.

This can be used to compute the loss through the cable.
"""

import numpy as np
from astropy import units as un

from edges import get_data_path
from edges import types as tp
from edges.frequencies import get_mask
from edges.io.calobsdef import HotLoadSemiRigidCable
from edges.modeling import ComplexRealImagModel, Fourier, ZerotooneTransform

from .. import Calkit, CalkitReadings, S11ModelParams, SParams, get_calkit


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


def get_hot_load_semi_rigid_from_filespec(
    filespec: HotLoadSemiRigidCable,
    calkit: Calkit | None = None,
    calkit_overrides: dict | None = None,
) -> SParams:
    """Get the hot load semi-rigid cable S-params from a file spec."""
    osl = CalkitReadings.from_filespec(filespec.osl)

    if calkit is None:
        calkit = get_calkit(
            filespec.calkit, resistance_of_match=filespec.calkit_match_resistance
        )

    if calkit_overrides:
        calkit = calkit.clone(**calkit_overrides)

    return SParams.from_calkit_measurements(
        model=calkit.at_freqs(osl.freqs), measurements=osl
    )


def hot_load_cable_model_params(**kwargs) -> S11ModelParams:
    """Get default model parameters for the hot load cable S11 model."""
    model = kwargs.pop(
        "model",
        Fourier(n_terms=27, transform=ZerotooneTransform(range=(0, 1)), period=1.5),
    )

    return S11ModelParams(
        model=model,
        complex_model_type=ComplexRealImagModel,
        set_transform_range=True,
        **kwargs,
    )
