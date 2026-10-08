"""Module with routines for simulating calibration datasets."""

from collections.abc import Callable

import numpy as np
from astropy import units as un

from .. import types as tp
from ..cal import Calibrator, InputSource, noise_waves


def simulate_q(
    *,
    load_s11: np.ndarray,
    receiver_s11: np.ndarray,
    load_temp: tp.TemperatureType,
    t_sca: np.ndarray,
    t_off: np.ndarray,
    t_unc: np.ndarray,
    t_cos: np.ndarray,
    t_sin: np.ndarray,
) -> np.ndarray:
    """Simulate the observed 3-position switch ratio data, Q.

    Parameters
    ----------
    load_s11 : np.ndarray
        The S11 of the input load (antenna, or calibration source) as a function of
        frequency.
    receiver_s11 : np.ndarray
        The S11 of the internal LNA.
    load_temp
        The (calibrated) temperature of the input load.
    t_sca
        The scaling temperature (i.e. ~T_load+ns)
    t_off : np.ndarray
        The offset temperature (i.e. ~T_load)
    t_unc : np.ndarray
        The noise-wave parameter T_uncorrelated
    t_cos : np.ndarray
        The noise-wave parameter T_cos
    t_sin : np.ndarray
        The noise-wave parameter T_sin
    t_load : float, optional
        The fiducial internal load temperature, by default 300.0
    t_load_ns : float, optional
        The internal load + noise source temperature, by default 400.0

    Returns
    -------
    q
        The simulated 3-position switch ratio data.
    """
    a, b = noise_waves.get_linear_coefficients(
        gamma_ant=load_s11,
        gamma_rec=receiver_s11,
        t_sca=t_sca,
        t_off=t_off,
        t_unc=t_unc,
        t_cos=t_cos,
        t_sin=t_sin,
    )

    return (load_temp - b * un.K) / (a * un.K)


def simulate_q_from_calibrator(
    load: InputSource,
    calibrator: Calibrator,
    scale_model: Callable | None = None,
) -> np.ndarray:
    """Simulate the observed 3-position switch ratio, Q, from noise-wave solutions.

    Parameters
    ----------
    calibrator
        The calibration observation that contains the solutions.
    load : str
        The load to simulate.

    Returns
    -------
    np.ndarray
        The 3-position switch values.
    """
    freq = calibrator.freqs

    t_sca = scale_model(freq) if scale_model is not None else calibrator.Tsca
    receiver_s11 = calibrator.receiver_s11
    temp_ave = load.temp_ave

    return simulate_q(
        load_s11=load.reflection_coefficient.s11,
        receiver_s11=receiver_s11,
        load_temp=temp_ave,
        t_sca=t_sca,
        t_off=calibrator.Toff,
        t_unc=calibrator.Tunc,
        t_cos=calibrator.Tcos,
        t_sin=calibrator.Tsin,
    )


def simulate_qant_from_calibrator(
    calibrator: Calibrator,
    ant_s11: np.ndarray,
    ant_temp: np.ndarray,
    scale_model: Callable | None = None,
    loss: np.ndarray | float = 1,
    t_amb: tp.TemperatureType | float = 296 * un.K,
    bm_corr: float | np.ndarray = 1,
) -> np.ndarray:
    """Simulate antenna Q from a calibration observation.

    Parameters
    ----------
    calibrator
        The calibrator that contains the noise-wave solutions.
    ant_s11
        The S11 of the antenna.
    ant_temp
        The true temperature of the beam-weighted sky.
    scale_model
        An optional model of the scale temperature, evaluated at the calibrator's
        frequencies. By default, use ``calibrator.Tsca``.
    loss
        The (antenna/balun/ground) loss, as a power gain between 0 and 1.
    t_amb
        The ambient temperature of the lossy components. Floats are interpreted as
        being in K.
    bm_corr
        The beam-chromaticity correction factor applied to ``ant_temp``.

    Returns
    -------
    np.ndarray
        The simulated 3-position switch ratio, Q.
    """
    freq = calibrator.freqs

    t_sca = scale_model(freq) if scale_model is not None else calibrator.Tsca

    if isinstance(t_amb, un.Quantity):
        t_amb = t_amb.to(un.K, equivalencies=un.temperature())
    else:
        t_amb = t_amb * un.K

    ant_temp = loss * ant_temp * bm_corr + (1 - loss) * t_amb

    lna_s11 = calibrator.receiver_s11

    return simulate_q(
        load_s11=ant_s11,
        receiver_s11=lna_s11,
        load_temp=ant_temp,
        t_sca=t_sca,
        t_off=calibrator.Toff,
        t_unc=calibrator.Tunc,
        t_cos=calibrator.Tcos,
        t_sin=calibrator.Tsin,
    )
