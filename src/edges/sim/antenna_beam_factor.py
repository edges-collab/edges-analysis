from typing import Literal

import astropy.coordinates as apc
import numpy as np
from astropy import units as u

from edges import const
from edges import modelling as mdl
from edges.sim import sky_models
from edges.sim.beams import Beam, BeamFactor

from .. import types as tp
from .simulate import sky_convolution_generator


def compute_antenna_beam_factor(
    beam: Beam,
    ground_loss: np.ndarray | None = None,
    f_low: tp.FreqType = 0 * u.MHz,
    f_high: tp.FreqType = np.inf * u.MHz,
    normalize_beam: bool = True,
    sky_model: sky_models.SkyModel = sky_models.Haslam408(),
    index_model: sky_models.IndexModel = sky_models.GaussianIndex(),
    lsts: np.ndarray | None = None,
    reference_frequency: tp.FreqType | None = None,
    beam_smoothing: bool = True,
    smoothing_model: mdl.Model = mdl.Polynomial(n_terms=12),
    interp_kind: Literal[
        "linear",
        "nearest",
        "slinear",
        "cubic",
        "quintic",
        "pchip",
        "spline",
        "sphere-spline",
    ] = "sphere-spline",
    lst_progress: bool = True,
    freq_progress: bool = True,
    location: apc.EarthLocation = const.edges_location,
    sky_at_reference_frequency: bool = True,
    use_astropy_azel: bool = True,
) -> BeamFactor:
    """
    Calculate the antenna beam factor.

    Parameters
    ----------
    beam
        A :class:`Beam` object.
    ground_loss
        An array of ground-loss values for the beam, shape (Nfreq,).
    f_low
        Minimum frequency to keep in the simulation (frequencies otherwise defined by
        the beam).
    f_high
        Maximum frequency to keep in the simulation (frequencies otherwise defined by
        the beam).
    normalize_beam
        Whether to normalize the beam to be maximum unity.
    sky_model
        A sky model to use.
    index_model
        An :class:`IndexModel` to use to generate different frequencies of the sky
        model.
    twenty_min_per_lst
        How many periods of twenty minutes fit into each LST bin.
    save_dir
        The directory in which to save the output beam factor.
    save_fname
        The filename to save the output beam factor.
    reference_frequency
        The frequency to take as the "reference", i.e. where the chromaticity will
        be by construction unity.
    lst_progress
        Whether to show a progress bar over the LSTs.
    freq_progress
        Whether to show a progress bar over the frequencies.
    location
        The location of the telescope.

    Returns
    -------
    beam_factor : :class`BeamFactor` instance
    """
    beam = beam.between_freqs(f_low, f_high)

    if lsts is None:
        lsts = np.arange(0, 24, 0.5)

    # Get index of reference frequency
    if reference_frequency is None:
        reference_frequency = (f_high + f_low) / 2

    indx_ref_freq = np.argmin(np.abs(beam.frequency - reference_frequency))
    # Don't reset the reference frequency. Alan uses the discrete ref frequency
    # to get the weighted sky temp, then models that, and divides by the model at
    # non-discrete ref frequency to normalize.

    antenna_temperature_above_horizon = np.zeros((len(lsts), len(beam.frequency)))
    if sky_at_reference_frequency:
        convolution_ref = np.zeros((len(lsts),))

    else:
        convolution_ref = np.zeros((len(lsts), len(beam.frequency)))

    loss_fraction = np.zeros((len(lsts), len(beam.frequency)))
    beamsums = np.zeros((len(lsts), len(beam.frequency)))
    for (
        lst_idx,
        freq_idx,
        temperature,
        _,
        sky,
        bm,
        _,
        npix_no_nan,
        _az,
        _el,
        _interp,
    ) in sky_convolution_generator(
        lsts,
        beam=beam,
        sky_model=sky_model,
        index_model=index_model,
        normalize_beam=normalize_beam,
        beam_smoothing=beam_smoothing,
        smoothing_model=smoothing_model,
        ground_loss=ground_loss,
        interp_kind=interp_kind,
        lst_progress=lst_progress,
        freq_progress=freq_progress,
        location=location,
        ref_freq_idx=indx_ref_freq,
        use_astropy_azel=use_astropy_azel,
    ):
        # 'temperature' is the mean beam-weighted foreground above the horizon (single
        #               float for this lst, freq). If normalize_beam is True, it's
        #               normalized by the integral of the beam (at each freq)
        # 'sky' is the full sky model for this LST and freq
        # 'bm' is the full beam model for this freq (same for all LSTs)
        antenna_temperature_above_horizon[lst_idx, freq_idx] = temperature

        if freq_idx == indx_ref_freq:
            ref_bm = bm.copy()
            ref_sky = sky.copy()

            # This updates once per LST, on the first frequency iteration
            """
            sky_at_reference_frequency is a toggle between Eq-4 and Eq-A1 from Sims+23

            """
            if sky_at_reference_frequency:
                convolution_ref[lst_idx] = np.nansum(ref_bm * ref_sky) / npix_no_nan

        if not sky_at_reference_frequency:
            convolution_ref[lst_idx, freq_idx] = np.nansum(ref_bm * sky) / npix_no_nan

        beamsums[lst_idx, freq_idx] = np.nansum(bm) / npix_no_nan

        # Loss fraction
        loss_fraction[lst_idx, freq_idx] = 1 - np.nansum(bm) / npix_no_nan

    return BeamFactor(
        frequencies=beam.frequency.to_value("MHz").astype(float),
        lsts=np.array(lsts).astype(float),
        antenna_temp=(
            antenna_temperature_above_horizon
            if normalize_beam
            else antenna_temperature_above_horizon / beamsums
        ),
        antenna_temp_ref=(
            convolution_ref
            if normalize_beam
            else (convolution_ref.T / beamsums[:, indx_ref_freq]).T
        ),
        loss_fraction=loss_fraction,
        reference_frequency=reference_frequency.to_value("MHz"),
        meta={
            "beam_file": str(beam.raw_file),
            "simulator": beam.simulator,
            "f_low": f_low.to_value("MHz"),
            "f_high": f_high.to_value("MHz"),
            "normalize_beam": bool(normalize_beam),
            "sky_model": sky_model.name,
            "index_model": str(index_model),
            "rotation_from_north": float(90),
        },
    )
