"""Functions to perform reciever calibration given a CalibrationObservation."""

import logging
from collections import deque

import numpy as np
from astropy import units as un

from .. import types as tp
from .calibrator import Calibrator
from .calobs import CalibrationObservation
from .noise_waves import get_calibration_quantities_iterative as _get_cc_iterative

logger = logging.getLogger(__name__)


def get_calcoeffs_iterative(
    calobs: CalibrationObservation,
    *,
    cterms: int = 5,
    wterms: int = 7,
    apply_loss_to_true_temp: bool = True,
    smooth_scale_offset_within_loop: bool = False,
    cable_delay_sweep: np.ndarray = np.array([0]),
    ncal_iter: int = 4,
    fit_method: str = "lstsq",
    scale_offset_poly_spacing: float = 1.0,
    t_load_guess: tp.TemperatureType = 400 * un.K,
    t_load_ns_guess: tp.TemperatureType = 300 * un.K,
) -> Calibrator:
    """Determine noise-wave calibration coefficients iteratively.

    This is the algorithm used for Bowman+2018 to determine the calibration
    coefficients from lab-based observations of the receiver.

    Parameters
    ----------
    cterms
        The number of parameters in the models for the overall scale and offset
        temperatures.
    wterms
        The number of parameters in the models for the noise-wave temperatures.
    apply_loss_to_true_temp
        Whether to apply losses to the true "known" temperature, or inversely
        to the spectra under calibration. Bowman+2018 sets this to False.
    smooth_scale_offset_within_loop
        Whether to smooth the scale and offset temperatures within the iterative
        loop, or only after the loop has converged.
    cable_delay_sweep
        The delays to search to determine the delay of the long cable (used for
        two of the calibration loads).
    ncal_iter
        The number of iterations to use in the main loop.
    fit_method
        The alogirhtm to use in the linear fit.
    scale_offset_poly_spacing
        Sets the indices in the scale/offset polynomial models. Bowman+2018 uses 0.5.
    t_load_guess
        An initial guess for t_load. In principle, this parameter is completely
        ineffective, but it's possible that a good choice makes the iterations more
        stable.
    t_load_ns_guess
        The same as t_load_guess but for the load+noise-source temperature.

    Returns
    -------
    calibrator
        A calibrator object with all the coefficients.
    """
    source_q = {k: source.averaged_q for k, source in calobs.loads.items()}

    if apply_loss_to_true_temp:
        source_true_temps = calobs.source_thermistor_temps
        loss = None
    else:
        source_true_temps = dict(calobs.source_thermistor_temps)
        source_true_temps["hot_load"] = calobs.hot_load.spectrum.temp_ave
        loss = calobs.hot_load.loss

    scale, off, nwp = deque(
        _get_cc_iterative(
            calobs.freqs,
            source_q=source_q,
            receiver_s11=calobs.receiver.s11,
            source_s11s={name: load.s11.s11 for name, load in calobs.loads.items()},
            source_true_temps=source_true_temps,
            cterms=cterms,
            wterms=wterms,
            t_load_guess=t_load_guess,
            t_load_ns_guess=t_load_ns_guess,
            hot_load_loss=loss,
            smooth_scale_offset_within_loop=smooth_scale_offset_within_loop,
            delays_to_fit=cable_delay_sweep,
            niter=ncal_iter,
            poly_spacing=scale_offset_poly_spacing,
            fit_method=fit_method,
        ),
        maxlen=1,
    ).pop()

    fqs = calobs.freqs.to_value("MHz")

    return Calibrator(
        freqs=calobs.freqs,
        Tsca=scale(fqs),
        Toff=off(fqs),
        Tunc=nwp.get_tunc(fqs),
        Tcos=nwp.get_tcos(fqs),
        Tsin=nwp.get_tsin(fqs),
        receiver_s11=calobs.receiver_s11,
    )


def perform_term_sweep(
    calobs: CalibrationObservation,
    delta_rms_thresh: float = 0,
    min_cterms: int = 4,
    min_wterms: int = 4,
    max_cterms: int = 15,
    max_wterms: int = 15,
    **kwargs,
) -> CalibrationObservation:
    """For a given calibration definition, perform a sweep over number of terms.

    Parameters
    ----------
    calobs: :class:`CalibrationObservation` instance
        The definition calibration class. The `cterms` and `wterms` in this instance
        should define the *lowest* values of the parameters to sweep over.
    delta_rms_thresh : float
        The threshold in change in RMS between one set of parameters and the next that
        will define where to cut off. If zero, will run all sets of parameters up to
        the maximum terms specified.
    max_cterms : int
        The maximum number of cterms to trial.
    max_wterms : int
        The maximum number of wterms to trial.
    """
    cterms = range(min_cterms, max_cterms)
    wterms = range(min_wterms, max_wterms)

    winner = np.zeros(len(cterms), dtype=int)
    rms = np.ones((len(cterms), len(wterms))) * np.inf

    best_rms = np.inf

    for i, c in enumerate(cterms):
        for j, w in enumerate(wterms):
            calibrator = get_calcoeffs_iterative(calobs, cterms=c, wterms=w, **kwargs)

            res = calobs.get_rms(calibrator)
            dof = len(calobs.freqs) * len(calobs.loads) - c - w

            rms[i, j] = np.sqrt(sum(v**2 for v in res.values()) / dof).value

            logger.info(f"Nc = {c:02}, Nw = {w:02}; RMS/dof = {rms[i, j]:1.3e}")

            # If we've decreased by more than the threshold, this wterms becomes
            # the new winner (for this number of cterms)
            if j > 0 and rms[i, j] >= rms[i, j - 1] - delta_rms_thresh:
                winner[i] = j - 1
                break

            if rms[i, j] < best_rms:
                best_calibrator = calibrator
                best_rms = rms[i, j]

        if i > 0 and rms[i, winner[i]] >= rms[i - 1, winner[i - 1]] - delta_rms_thresh:
            break

    logger.info(
        f"Best parameters found for Nc={cterms[i - 1]}, "
        f"Nw={wterms[winner[i - 1]]}, "
        f"with RMS = {rms[i - 1, winner[i - 1]]}."
    )

    return best_calibrator
