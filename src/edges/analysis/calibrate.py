"""Module defining calibration routines for field data in EDGES."""

from collections.abc import Callable
from pathlib import Path

import numpy as np
from astropy import units as un
from pygsdata import GSData, gsregister

from .. import modeling as mdl
from .. import types as tp
from ..cal import (
    Calibrator,
)
from ..cal.s11 import CalibratedS11
from ..sim import antenna_beam_factor
from ..sim.antenna_beam_factor import BeamFactor


@gsregister("calibrate")
def apply_noise_wave_calibration(
    data: GSData,
    calibrator: Calibrator | tp.PathLike,
    antenna_s11: CalibratedS11,
    tload: float | None = None,
    tns: float | None = None,
) -> GSData:
    """Apply noise-wave calibration to data.

    This function requires a :class:`edges.cal.cal_coefficients.Calibrator` object
    (or a path to a file containing such an object) which must be created beforehand.
    The antenna S11 used is found automatically by searching for the file that has
    the closest match to the time of the data. This can be constrained by passing
    options that match the regex pattern for the S11 files.

    Parameters
    ----------
    data
        Data to be calibrated.
    calobs
        Calibrator object or path to file containing calibrator object.
    antenna_s11

    """
    if data.data_unit not in ("uncalibrated", "uncalibrated_temp"):
        raise ValueError("Data must be uncalibrated to apply calibration!")

    if data.data_unit == "uncalibrated_temp" and (tload is None or tns is None):
        raise ValueError(
            "You need to supply tload and tns if data_unit is uncalibrated_temp"
        )

    if data.nloads != 1:
        raise ValueError("Can only apply noise-wave calibration to single load data!")

    if data.data_unit == "uncalibrated_temp":
        q = (data.data - tload) / tns
    else:
        q = data.data

    new_data = calibrator.calibrate_q(q, ant_s11=antenna_s11.s11, freqs=data.freqs)
    if data.model is not None:
        qmodel = (
            (data.model - tload) / tns
            if data.data_unit == "uncalibrated_temp"
            else data.model
        )
        resids = new_data - calibrator.calibrate_q(
            qmodel, ant_s11=antenna_s11.s11, freq=data.freqs
        )
    else:
        resids = None

    return data.update(
        data=new_data.to_value("K"), data_unit="temperature", residuals=resids
    )


@gsregister("calibrate")
def apply_loss_correction(
    data: GSData,
    ambient_temp: tp.TemperatureType,
    loss: np.ndarray | None = None,
    loss_function: Callable | None = None,
    **kwargs,
) -> GSData:
    """Apply a loss-correction to data.

    Parameters
    ----------
    data
        The GSData object on which to apply the loss-correction.
    ambient_temp
        The ambient temperature at which to apply the loss-correction.
    loss
        An array of losses, where the size of the array must be equal to the number
        of frequencies in the data. If None, a loss function is used to compute
        the losses.
    loss_function
        A function to compute the loss. The function must accept an array of frequencies
        as its first argument, and may accept arbitrary other keyword arguments, which
        will can be passed as kwargs to this function. Either this or loss must be
        specified.

    Notes
    -----
    Loss functions can be stacked, either by multiplying the losses before passing
    them to this function, or by calling this function multiple times, once for each
    loss.
    """
    if loss is None and loss_function is None:
        raise ValueError("Either loss or loss_function must be provided!")

    if loss is None:
        loss = loss_function(data.freqs, **kwargs)

    if data.data_unit != "temperature":
        raise ValueError("Data must be temperature to apply antenna loss correction!")

    a = ambient_temp.to_value(un.K)
    spec = (data.data - np.outer(a, (1 - loss))) / loss

    return data.update(
        data=spec,
        data_unit="temperature",
        residuals=None,
    )


@gsregister("calibrate")
def apply_beam_factor_directly(data: GSData, beam_file: str | Path) -> GSData:
    """Apply a beam correction factor from a file directly to the data.

    This function multiplies the data by the beam correction factor
    from the provided beamfile. It handles data with a single load and updates the data
    unit to "temperature".

    Parameters
    ----------
    data
        The GSData object containing the data to correct.
    beamfile
        The path to the beamfile containing the correction factors. The correction
        factors should be in the fourth column of the csv file, and should have a size
        equal to the number of frequencies in the data.

    Returns
    -------
    data
        A new GSData object with the corrected data and residuals.

    Raises
    ------
    NotImplementedError
        If the data contains more than one load.
    """
    if len(data.loads) > 1:
        raise NotImplementedError(
            "Can only apply beam correction to data with a single load"
        )

    new_data = data.data.copy()
    resids = data.residuals.copy() if data.residuals is not None else None
    bf = np.loadtxt(beam_file)
    new_data *= bf[:, 3]
    if resids is not None:
        resids *= bf[:, 3]
    return data.update(data=new_data, residuals=resids, data_unit="temperature")


@gsregister("calibrate")
def apply_beam_correction(
    data: GSData,
    beam: str | Path | antenna_beam_factor.BeamFactor,
    freq_model: mdl.Model,
    integrate_before_ratio: bool = True,
    oversample_factor: int = 5,
    resample_beam_lsts: bool = True,
    lsts: np.ndarray | None = None,
    cut_to_data_lsts: bool = True,
) -> GSData:
    """Apply beam correction to the data.

    This always applies the beam correction to each time sample in the data. If you want
    to average the data *before* applying the beam correction, you must average the data
    before applying this function to it. The input beam factor object should cover the
    full range of LSTs included in the data itself.

    The beam factor object is defined at a set of LSTs, and by default, the correction
    applied to the data is the *average* beam factor in each LST-bin of the data.
    To use the beam factor *interpolated* to the LSTs of the data instead, set
    ``interpolate_to_lsts`` to True.

    There are two ways to define the average beam factor within an LST-bin: either
    by taking the mean of ratios (of beam-weighted foreground model to *reference*
    beam-weighted foreground model) or the ratio of means. Switch between these
    by using the ``integrate_before_ratio`` parameter.

    Parameters
    ----------
    data
        Data to be calibrated.
    beam
        Either a path to a file containing beam correction coefficients, or the
        BeamFactor object itself.
    freq_model
        The (linear) model to use when evaluating the beam factor at the data freqs.
    integrate_before_ratio
        Whether to integrate (over time) the beam-weighted sky temperature and the
        reference sky temperature individually before taking their ratio to get
        the beam factor.
    oversample_factor
        The number of LST samples to use when interpolating the beam factor to the
        LSTs of the data. For every data LST, ``oversample_factor`` LSTs will be
        interpolated to (regularly spaced between each data LST, regardless of whether
        the data LSTs are regular). This is only used if ``resample_beam_lsts`` is True.
    resample_beam_lsts
        Whether to resample LSTs before averaging (by ``oversample_factor``).
    lsts
        If given, resample to these LSTs exactly, instead of trying to use the
        LST ranges in the data with oversample_factor.
    cut_to_data_lsts
        If True, cut the LSTs at which the beam is sampled to lie within the
        LST ranges of the data in each LST bin. Only set this to False if you have
        a single LST bin in the data and you know precisely the LSTs at which you
        want to sample the beam.
    """
    if isinstance(beam, str | Path):
        beam = BeamFactor.from_file(beam)

    if len(data.loads) > 1:
        raise NotImplementedError(
            "Can only apply beam correction to data with a single load"
        )

    if len(beam.lsts) < 4 and resample_beam_lsts:
        raise ValueError(
            "Your beam has a single LST so you cannot interpolate over LSTs."
        )

    if resample_beam_lsts:
        if lsts is not None:
            beam = beam.at_lsts(lsts)
        else:
            cut_to_data_lsts = True
            new_beam_lsts = []
            for lst0, lst1 in data.lst_ranges[:, 0, :]:
                lst1 = lst1.hour
                if lst1 < lst0.hour:
                    lst1 = lst1 + 24

                new_beam_lsts.append(
                    np.linspace(lst0.hour, lst1, oversample_factor + 1)[:-1]
                )
            new_beam_lsts = np.concatenate(new_beam_lsts)
            beam = beam.at_lsts(new_beam_lsts)

    new_data = data.data.copy()

    resids = data.residuals.copy() if data.residuals is not None else None

    for i, (lst0, lst1) in enumerate(data.lst_ranges[:, 0, :]):
        new = beam.between_lsts(lst0.hour, lst1.hour) if cut_to_data_lsts else beam
        if integrate_before_ratio:
            bf = new.get_integrated_beam_factor(
                model=freq_model, freqs=data.freqs.to_value("MHz")
            )
        else:
            bf = new.get_mean_beam_factor(
                model=freq_model, freqs=data.freqs.to_value("MHz")
            )

        new_data[:, :, i] /= bf
        if resids is not None:
            resids[:, :, i] /= bf

    return data.update(data=new_data, residuals=resids, data_unit="temperature")
