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


def get_noise_wave_calibration_iterative(
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
            source_s11s={
                name: load.reflection_coefficient.s11
                for name, load in calobs.loads.items()
            },
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


def _sweep_metric(
    rms: dict[str, tp.TemperatureType], nfreq: int, cterms: int, wterms: int
) -> float:
    """Residual RMS per degree of freedom (in K) of a noise-wave calibration.

    This is ``sqrt(chi^2 / dof)`` with unit weights, where ``chi^2`` is the sum of
    squared calibration residuals over all frequencies and sources, and
    ``dof = nfreq * nsources - (2 * cterms + 3 * wterms)``. The noise-wave model has
    ``cterms`` parameters in each of the scale and offset, and ``wterms`` in each of
    the three noise-wave temperatures.
    """
    chi2 = nfreq * sum(v.to_value("K") ** 2 for v in rms.values())
    dof = nfreq * len(rms) - (2 * cterms + 3 * wterms)
    if dof <= 0:
        return np.inf
    return float(np.sqrt(chi2 / dof))


def perform_term_sweep(
    calobs: CalibrationObservation,
    delta_rms_thresh: float = 0,
    min_cterms: int = 4,
    min_wterms: int = 4,
    max_cterms: int = 15,
    max_wterms: int = 15,
    **kwargs,
) -> Calibrator:
    """For a given calibration definition, perform a sweep over number of terms.

    For each number of scale/offset terms, ``cterms``, from ``min_cterms`` to
    ``max_cterms`` (inclusive), the number of noise-wave terms, ``wterms``, is
    increased from ``min_wterms`` up to ``max_wterms`` (inclusive) until the
    goodness-of-fit metric fails to decrease by more than ``delta_rms_thresh``. The
    sweep over ``cterms`` stops when the best metric for a given ``cterms`` fails to
    decrease by more than ``delta_rms_thresh`` compared to the previous ``cterms``.
    The calibrator with the lowest metric of all those tried is returned.

    The goodness-of-fit metric is the residual RMS per degree of freedom, in K:
    ``sqrt(chi^2 / dof)``, where ``chi^2 = Nfreq * sum_s RMS_s^2`` is the sum of the
    squared calibration residuals over all ``Nfreq`` frequencies and all sources
    ``s`` (with ``RMS_s`` from :meth:`CalibrationObservation.get_rms`), and
    ``dof = Nfreq * Nsources - (2 * cterms + 3 * wterms)``, since the scale and
    offset each have ``cterms`` parameters, and each of the three noise-wave
    temperatures has ``wterms`` parameters.

    Parameters
    ----------
    calobs: :class:`CalibrationObservation` instance
        The calibration observation to calibrate.
    delta_rms_thresh : float
        The minimum decrease in the metric (in K) for an extra term to be considered
        an improvement. If zero, terms are added for as long as the metric strictly
        decreases.
    min_cterms : int
        The minimum number of cterms to trial.
    min_wterms : int
        The minimum number of wterms to trial.
    max_cterms : int
        The maximum number of cterms to trial (inclusive).
    max_wterms : int
        The maximum number of wterms to trial (inclusive).
    kwargs
        Passed through to :func:`get_noise_wave_calibration_iterative`.

    Returns
    -------
    calibrator
        The calibrator with the lowest metric found in the sweep.

    Raises
    ------
    RuntimeError
        If no finite metric was found for any number of terms.
    """
    cterms = range(min_cterms, max_cterms + 1)
    wterms = range(min_wterms, max_wterms + 1)
    nfreq = len(calobs.freqs)

    best_metric = np.inf
    best_calibrator = None
    best_terms = None
    prev_c_best = np.inf

    for c in cterms:
        c_best = np.inf
        prev = np.inf
        for w in wterms:
            calibrator = get_noise_wave_calibration_iterative(
                calobs, cterms=c, wterms=w, **kwargs
            )
            metric = _sweep_metric(calobs.get_rms(calibrator), nfreq, c, w)

            logger.info(f"Nc = {c:02}, Nw = {w:02}; RMS/dof = {metric:1.3e}")

            if metric < best_metric:
                best_metric = metric
                best_calibrator = calibrator
                best_terms = (c, w)

            c_best = min(c_best, metric) if np.isfinite(metric) else c_best

            # Stop adding noise-wave terms once they no longer help enough.
            if metric >= prev - delta_rms_thresh:
                break
            prev = metric

        # Stop adding scale/offset terms once they no longer help enough.
        if c_best >= prev_c_best - delta_rms_thresh:
            break
        prev_c_best = c_best

    if best_calibrator is None:
        raise RuntimeError(
            "No finite RMS was found for any number of terms in the sweep "
            f"(cterms in {cterms}, wterms in {wterms})."
        )

    logger.info(
        f"Best parameters found for Nc={best_terms[0]}, Nw={best_terms[1]}, "
        f"with RMS/dof = {best_metric:1.3e}."
    )

    return best_calibrator
