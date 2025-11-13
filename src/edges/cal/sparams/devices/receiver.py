"""Functions for calibrating the receiver reflection coefficients."""

import numpy as np
from astropy import units as un
from astropy.constants import c as speed_of_light

from edges import types as tp
from edges.io import CalObsDefEDGES2, CalObsDefEDGES3
from edges.io.calobsdef import ReceiverS11
from edges.modeling import ComplexRealImagModel, Fourier, ZerotooneTransform

from .. import (
    AGILENT_85033E,
    Calkit,
    CalkitReadings,
    ReflectionCoefficient,
    S11ModelParams,
    SParams,
    average_reflection_coefficients,
    get_calkit,
)


def path_length_correction_edges3(
    freq: tp.FreqType, delay: tp.TimeType, gamma_in: float, lossf: float, dielf: float
) -> tuple[float, float, float]:
    """
    Calculate the path length correction for the EDGES-3 LNA.

    Notes
    -----
    The 8-position switch memo is 303 and the correction for the path to the
    LNA for the calibration of the LNA s11 is described in memos 367 and 392.

    corrcsv.c corrects lna s11 file for the different vna path to lna args:
    s11.csv -cablen -cabdiel -cabloss outputs c_s11.csv

    The actual numbers are slightly temperature dependent

    corrcsv s11.csv -cablen 4.26 -cabdiel -1.24 -cabloss -91.5

    and need to be determined using a calibration test like that described in
    memos 369 and 361. Basically the path length corrections can be "tuned" by
    minimizing the ripple on the calibrated spectrum of the open or shorted
    cable.

    cablen --> length in inches
    cabloss --> loss correction percentage
    cabdiel --> dielectric correction in percentage
    """
    # TODO: this function should be able to be refactored into the other functions
    # provided in this repo, but this is simply copied directly from the C code.

    freq = freq.to("Hz").value
    length = (delay * speed_of_light).to_value("m")

    b = 0.1175 * 2.54e-2 * 0.5
    a = 0.0362 * 2.54e-2 * 0.5
    diel = 2.05 * dielf  # UT-141C-SP
    # for tinned copper
    d2 = np.sqrt(1.0 / (np.pi * 4.0 * np.pi * 1e-7 * 5.96e07 * 0.8 * lossf))
    # skin depth at 1 Hz for copper
    d = np.sqrt(1.0 / (np.pi * 4.0 * np.pi * 1e-7 * 5.96e07 * lossf))

    L = (4.0 * np.pi * 1e-7 / (2.0 * np.pi)) * np.log(b / a)
    C = 2.0 * np.pi * 8.854e-12 * diel / np.log(b / a)

    La = 4.0 * np.pi * 1e-7 * d / (4.0 * np.pi * a)
    Lb = 4.0 * np.pi * 1e-7 * d2 / (4.0 * np.pi * b)
    disp = (La + Lb) / L
    R = 2.0 * np.pi * L * disp * np.sqrt(freq)
    L = L * (1.0 + disp / np.sqrt(freq))
    G = 2.0 * np.pi * C * freq * 2e-4 if diel > 1.2 else 0
    Zcab = np.sqrt((1j * 2 * np.pi * freq * L + R) / (1j * 2 * np.pi * freq * C + G))
    g = np.sqrt((1j * 2 * np.pi * freq * L + R) * (1j * 2 * np.pi * freq * C + G))

    T = (50.0 - Zcab) / (50.0 + Zcab)
    Vin = np.exp(+g * length) + T * np.exp(-g * length)
    Iin = (np.exp(+g * length) - T * np.exp(-g * length)) / Zcab
    Vout = 1 + T  # Iout = (1 - T)/Zcab
    s11 = ((Vin / Iin) - 50) / ((Vin / Iin) + 50)  # same as s22
    VVin = Vin + 50.0 * Iin
    s12 = 2 * Vout / VVin  # same as s21

    Z = 50.0 * (1 + gamma_in) / (1 - gamma_in)
    T = (Z - Zcab) / (Z + Zcab)
    T = T * np.exp(-g * 2 * length)
    Z = Zcab * (1 + T) / (1 - T)
    T = (Z - 50.0) / (Z + 50.0)

    return T, s11, s12


def correct_receiver_for_extra_cable(
    gamma: ReflectionCoefficient,
    cable_length: tp.LengthType = 0.0 * un.cm,
    cable_loss_percent: float = 0.0,
    cable_dielectric_percent: float = 0.0,
) -> ReflectionCoefficient:
    """Correct the receiver S11 measurements to include an extra short cable length.

    Parameters
    ----------
    gamma
        The receiver reflection coefficient to be corrected.
    cable_length
        The length of the extra cable to be corrected for.
    cable_loss_percent
        The overall loss of the cable.
    cable_dielectric_percent
        The dielectric of the cable, as a percent.
    """
    _, s11, s12 = path_length_correction_edges3(
        freq=gamma.freqs,
        delay=cable_length / speed_of_light,
        gamma_in=0,
        lossf=1 + cable_loss_percent * 0.01,
        dielf=1 + cable_dielectric_percent * 0.01,
    )
    sparams = SParams(freqs=gamma.freqs, s11=s11, s12=s12)
    return gamma.embed(sparams) if cable_length > 0.0 else gamma.de_embed(sparams)


# Constructor Methods
def calibrate_gamma_receiver(
    calkit_measurements: CalkitReadings,
    gamma_receiver: ReflectionCoefficient,
    calkit: Calkit = AGILENT_85033E,
    cable_length: tp.LengthType = 0.0 * un.cm,
    cable_loss_percent: float = 0.0,
    cable_dielectric_percent: float = 0.0,
) -> ReflectionCoefficient:
    """
    Calibrate the receiver reflection coefficient using calkit measurements.

    Parameters
    ----------
    calkit_measurements
        The calkit measurements used to calibrate the receiver.
    gamma_receiver
        The raw receiver reflection coefficient measurements.
    calkit
        The calkit model used for the calibration.
    cable_length
        An optional extra cable length to correct the receiver S11 for. This
        is used in the case of EDGES-3.
    cable_loss_percent
        The loss percentage of the extra cable. Default is 0.0.
        This is used in the case of EDGES-3.
    cable_dielectric_percent
        The dielectric percentage of the extra cable. Default is 0.0.
        This is used in the case of EDGES-3.

    Returns
    -------
    ReflectionCoefficient
        The calibrated receiver reflection coefficient.
    """
    freqs = calkit_measurements.freqs

    # De-embed the small "offset" in the VNA to calibrate the receiver
    # reflection coefficient to the correct reference plane.
    smatrix = SParams.from_calkit_measurements(
        model=calkit.at_freqs(freqs), measurements=calkit_measurements
    )
    gamma_rcv = gamma_receiver.de_embed(smatrix)

    if cable_length != 0 * un.m:
        gamma_rcv = correct_receiver_for_extra_cable(
            gamma=gamma_rcv,
            cable_length=cable_length,
            cable_dielectric_percent=cable_dielectric_percent,
            cable_loss_percent=cable_loss_percent,
        )

    return gamma_rcv


def receiver_model_params(
    find_model_delay: bool = True, complex_model_type=ComplexRealImagModel, **kwargs
) -> S11ModelParams:
    """Get default S11ModelParams for receiver S11 modeling."""
    model = kwargs.pop(
        "model",
        Fourier(n_terms=11, transform=ZerotooneTransform(range=(1, 2)), period=1.5),
    )

    return S11ModelParams(
        model=model,
        find_model_delay=find_model_delay,
        complex_model_type=complex_model_type,
        **kwargs,
    )


def get_gamma_receiver_from_filespec(
    caldef: CalObsDefEDGES2 | CalObsDefEDGES3 | ReceiverS11,
    calkit: Calkit | None = None,
    calkit_overrides: dict | None = None,
    **kwargs,
) -> ReflectionCoefficient:
    """Get the calibrated receiver reflection coeff from a calibration definition."""
    if isinstance(caldef, CalObsDefEDGES2 | CalObsDefEDGES3):
        rcvdef = caldef.receiver_s11
    else:
        rcvdef = caldef

    if not hasattr(rcvdef, "__len__"):
        rcvdef = [rcvdef]

    gamma_rcv = []
    for rcv in rcvdef:
        if calkit is None:
            this_calkit = get_calkit(
                rcv.calkit_name, resistance_of_match=rcv.calkit_match_resistance
            )
        else:
            this_calkit = calkit

        if calkit_overrides is not None:
            this_calkit = get_calkit(this_calkit, **calkit_overrides)

        gamma_rcv.append(
            calibrate_gamma_receiver(
                calkit_measurements=CalkitReadings.from_filespec(rcv.calkit),
                gamma_receiver=ReflectionCoefficient.from_s1p(rcv.device),
                calkit=this_calkit,
                **kwargs,
            )
        )

    return average_reflection_coefficients(gamma_rcv)
