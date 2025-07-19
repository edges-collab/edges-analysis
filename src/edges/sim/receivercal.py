"""Module with routines for simulating calibration datasets."""

from __future__ import annotations

import numpy as np

from ..cal import noise_waves


def simulate_q(
    *,
    load_s11: np.ndarray,
    receiver_s11: np.ndarray,
    load_temp: float | np.ndarray,
    scale: np.ndarray,
    offset: np.ndarray,
    t_unc: np.ndarray,
    t_cos: np.ndarray,
    t_sin: np.ndarray,
    t_load: float = 300.0,
    t_load_ns: float = 400.0,
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
    scale
        The scale polynomial (C1)
    offset : np.ndarray
        The offset polynomial (C2)
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
        sca=scale,
        off=offset,
        t_unc=t_unc,
        t_cos=t_cos,
        t_sin=t_sin,
        t_load=t_load,
    )

    uncal_temp = (load_temp - b) / a
    return (uncal_temp - t_load) / t_load_ns


def simulate_q_from_calobs(
    calobs, load: str, scale_model=None, freq=None
) -> np.ndarray:
    """Simulate the observed 3-position switch ratio, Q, from noise-wave solutions.

    Parameters
    ----------
    calobs : :class:`~edges.cal.cal_coefficients.CalibrationObservation`
        The calibration observation that contains the solutions.
    load : str
        The load to simulate.

    Returns
    -------
    np.ndarray
        The 3-position switch values.
    """
    default_freq = freq is None
    if freq is None:
        freq = calobs.freq

    C1 = scale_model(freq) if scale_model is not None else calobs.C1(freq)
    try:
        receiver_s11 = calobs.receiver.s11_model(freq.to_value("MHz"))
    except AttributeError:
        receiver_s11 = calobs.receiver_s11(freq)

    if not default_freq:
        temp_ave = calobs.loads[load].get_temp_with_loss(freq)
    else:
        temp_ave = calobs.loads[load].temp_ave

    return simulate_q(
        load_s11=calobs.loads[load].reflections.s11_model(freq.to_value("MHz")),
        receiver_s11=receiver_s11,
        load_temp=temp_ave,
        scale=C1,
        offset=calobs.C2(freq),
        t_unc=calobs.Tunc(freq),
        t_cos=calobs.Tcos(freq),
        t_sin=calobs.Tsin(freq),
        t_load=calobs.t_load,
        t_load_ns=calobs.t_load_ns,
    )


def simulate_qant_from_calobs(
    calobs,
    ant_s11: np.ndarray,
    ant_temp: np.ndarray,
    scale_model=None,
    freq=None,
    loss=1,
    t_amb=296,
    bm_corr=1,
) -> np.ndarray:
    """Simulate antenna Q from a calibration observation.

    Parameters
    ----------
    calobs : :class:`~edges.cal.cal_coefficients.CalibrationObservation`
        The calibration observation that contains the solutions.
    ant_s11
        The S11 of the antenna.
    ant_temp
        The true temperature of the beam-weighted sky.

    Returns
    -------
    np.ndarray
        The simulated 3-position switch ratio, Q.
    """
    if freq is None:
        freq = calobs.freq

    scale = scale_model(freq) if scale_model is not None else calobs.C1(freq)

    ant_temp = loss * ant_temp * bm_corr + (1 - loss) * t_amb

    lna_s11 = (
        calobs.receiver_s11(freq)
        if callable(calobs.receiver_s11)
        else calobs.receiver.s11_model(freq.to_value("MHz"))
    )
    return simulate_q(
        load_s11=ant_s11,
        receiver_s11=lna_s11,
        load_temp=ant_temp,
        scale=scale,
        offset=calobs.C2(freq),
        t_unc=calobs.Tunc(freq),
        t_cos=calobs.Tcos(freq),
        t_sin=calobs.Tsin(freq),
        t_load=calobs.t_load,
        t_load_ns=calobs.t_load_ns,
    )


def get_data_from_calobs(
    srcs: Sequence[str],
    calobs,
    tns: mdl.Model | None = None,
    sim: bool = False,
    loads: dict | None = None,
) -> np.ndarray:
    """Generate input data to fit from a calibration observation."""
    if loads is None:
        loads = calobs.loads

    data = []
    for src in srcs:
        load = loads[src]
        if tns is None:
            _tns = calobs.C1() * calobs.t_load_ns
        else:
            _tns = tns(x=calobs.freq)

        q = (
            simulate_q_from_calobs(calobs, load=src)
            if sim
            else load.spectrum.averaged_Q
        )
        c = calobs.get_K()[src][0]
        data.append(_tns * q - c * load.temp_ave)
    return np.concatenate(tuple(data))
