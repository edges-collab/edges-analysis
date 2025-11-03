"""Functions for creating CalibratedS11 objects from receiver data."""

from collections.abc import Sequence

import numpy as np
from astropy import units as un
from astropy.constants import c as speed_of_light

from edges import types as tp

from ...io import calobsdef
from ...io.vna import SParams
from .. import reflection_coefficient as rc
from ..s11 import CalibratedS11, StandardsReadings


def correct_receiver_for_extra_cable(
    s11_in: tp.ComplexArray,
    freq: tp.FreqType,
    cable_length: tp.LengthType = 0.0 * un.cm,
    cable_loss_percent: float = 0.0,
    cable_dielectric_percent: float = 0.0,
) -> tp.ComplexArray:
    """Correct the receiver S11 measurements to include an extra short cable length.

    Parameters
    ----------
    s11_in
        The un-corrected S11.
    freq
        The frequencies of the un-corrected S11.
    cable_length
        The length of the extra cable to be corrected for.
    cable_loss_percent
        The overall loss of the cable.
    cable_dielectric_percent
        The dielectric of the cable, as a percent.
    """
    _, s11, s12 = rc.path_length_correction_edges3(
        freq=freq,
        delay=cable_length / speed_of_light,
        gamma_in=0,
        lossf=1 + cable_loss_percent * 0.01,
        dielf=1 + cable_dielectric_percent * 0.01,
    )
    smatrix = rc.SMatrix([[s11, s12], [s12, s11]])

    if cable_length > 0.0:
        return rc.gamma_embed(smatrix, s11_in)
    return rc.gamma_de_embed(s11_in, smatrix)


# Constructor Methods
def get_receiver_s11model_from_filespec(
    pathspec: calobsdef.ReceiverS11 | Sequence[calobsdef.ReceiverS11],
    calkit: rc.Calkit = rc.AGILENT_85033E,
    resistance: float | None = None,
    f_low=0.0 * un.MHz,
    f_high=np.inf * un.MHz,
    cable_length: tp.LengthType = 0.0 * un.cm,
    cable_loss_percent: float = 0.0,
    cable_dielectric_percent: float = 0.0,
) -> CalibratedS11:
    """
    Create an instance from a given path.

    Parameters
    ----------
    path : str or Path
        Path to overall Calibration Observation.
    run_num_load : int
        The run to use for the LNA (default latest available).
    run_num_switch : int
        The run to use for the switching state (default lastest available).

    Returns
    -------
    receiver
        The Receiver object.
    """
    if resistance is not None:
        calkit = rc.get_calkit(calkit, resistance_of_match=resistance)

    if not hasattr(pathspec, "__len__"):
        pathspec = [pathspec]

    s11s = []
    for dv in pathspec:
        standards = StandardsReadings.from_filespec(
            dv.calkit, f_low=f_low, f_high=f_high
        )
        receiver_reading = SParams.from_s1p_file(dv.device, f_low=f_low, f_high=f_high)
        freq = standards.freq

        smatrix = rc.SMatrix.from_calkit_and_vna(calkit, standards)
        calibrated_s11 = rc.gamma_de_embed(receiver_reading.s11, smatrix)

        if cable_length != 0 * un.m:
            calibrated_s11 = correct_receiver_for_extra_cable(
                s11_in=calibrated_s11,
                freq=freq,
                cable_length=cable_length,
                cable_dielectric_percent=cable_dielectric_percent,
                cable_loss_percent=cable_loss_percent,
            )

        s11s.append(calibrated_s11)

    return CalibratedS11(s11=np.mean(s11s, axis=0), freqs=freq)
