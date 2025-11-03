"""Simulation functions for ideal sky observations."""

from typing import Literal

import astropy.coordinates as apc
import astropy.time as apt
import numpy as np
from astropy import units as un
from pygsdata import coordinates as gscrd
from read_acq import _coordinates as crda
from tqdm import tqdm

from .. import const
from .. import modeling as mdl
from . import sky_models
from .beams import Beam

# Reference UTC observation time. At this time, the LST is 0.1666 (00:10 Hrs LST) at the
# EDGES location. NOTE: this is used by default, but can be changed by the user anywhere
# it is used.
REFERENCE_TIME = apt.Time("2014-01-01T09:39:42", location=const.edges_location)


def sky_convolution_generator(
    lsts: np.ndarray,
    beam: Beam,
    sky_model: sky_models.SkyModel,
    index_model: sky_models.IndexModel,
    normalize_beam: bool,
    beam_smoothing: bool,
    smoothing_model: mdl.Model,
    ground_loss: np.ndarray | None = None,
    location: apc.EarthLocation = const.KNOWN_TELESCOPES["edges-low"].location,
    ref_time: apt.Time = REFERENCE_TIME,
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
    ref_freq_idx: int = 0,
    use_astropy_azel: bool = True,
):
    """
    Iterate through given LSTs and generate a beam*sky product at each freq and LST.

    This is a generator, so it will yield a single item at a time (to save on memory).

    Parameters
    ----------
    lsts
        The LSTs at which to evaluate the convolution.
    ground_loss
        An array of ground-loss values for the beam, shape (Nfreq,).
    beam
        The beam to convolve.
    sky_model
        The sky model to convolve
    index_model
        The spectral index model of the sky model.
    normalize_beam
        Whether to ensure the beam is properly normalised.
    beam_interpolation
        Whether to smooth over freq axis
    interp_kind
        The kind of interpolation to use for the beam. "spline" uses
        :class:`scipy.interpolate.RectBivariateSpline` and "sphere-spline" uses
        :class:`scipy.interpolate.RectSphereBivariateSpline`. All other options use
        :class:`scipy.interpolate.RegularGridInterpolator`. with the given kind as
        ``method``.
    use_astropy_azel
        Whether to use the astropy coordinate system for azimuth and elevation. If
        False, compute the az/el using Alan's method.

    Yields
    ------
    i
        The LST enumerator
    j
        The frequency enumerator
    mean_conv_temp
        The mean temperature after multiplying by the beam (above the horizon)
    conv_temp
        An array containing the temperature after multiuplying by the beam in each pixel
        above the horizon.
    sky
        An array containing the sky temperature in pixel above the horizon.
    beam
        An array containing the interpolatedbeam in pixels above the horizon
    time
        The local time at each LST.
    n_pixels
        The total number of pixels that are not masked.

    Examples
    --------
    Use this function as follows:

    >>> for i, j, mean_t, conv_t, sky, bm, time, npix in sky_convolution_generator():
    >>>     print(conv_t)
    """
    if beam_smoothing:
        beam = beam.smoothed(smoothing_model)

    if ground_loss is None:
        ground_gain = np.ones(len(beam.frequency))
    else:
        ground_gain = np.asarray(ground_loss)

    # Get the local times corresponding to the given LSTs
    times = gscrd.lsts_to_times(lsts * un.hourangle, ref_time, location)

    beam_above_horizon = np.full(sky_model.coords.shape, np.nan)
    interpolators = {}

    for lst_idx, time in tqdm(
        enumerate(times), unit="LST", disable=not lst_progress, total=len(times)
    ):
        # Transform Galactic coordinates of Sky Model to Local coordinates
        if use_astropy_azel:
            altaz = sky_model.coords.transform_to(
                apc.AltAz(location=location, obstime=time)
            )
            az = np.asarray(altaz.az.deg)
            el = np.asarray(altaz.alt.deg)
        else:
            ra, dec = crda.galactic_to_radec(
                sky_model.coords.galactic.b.deg, sky_model.coords.galactic.l.deg
            )
            az, el = crda.radec_azel_from_lst(
                lsts[lst_idx] * np.pi / 12, ra, dec, location.lat.rad
            )
            az *= 180 / np.pi
            el *= 180 / np.pi
        # Number of pixels over FULL SKY (4pi) in the sky model
        n_pix_tot = len(el)

        # Selecting coordinates above the horizon
        horizon_mask = el > 0
        az_above_horizon = az[horizon_mask]
        el_above_horizon = el[horizon_mask]

        # Loop over frequency
        # Using np.roll means we start at the reference frequency and loop around.
        # Starting at ref freq (if there is one) means that we can use the
        # reference beam with other frequencies.
        for freq_idx in tqdm(
            np.roll(range(len(beam.frequency)), -ref_freq_idx),
            unit="Frequency",
            disable=not freq_progress,
        ):
            if freq_idx not in interpolators:
                interpolators[freq_idx] = beam.angular_interpolator(
                    freq_idx, interp_kind=interp_kind
                )

            sky_map = sky_model.at_freq(
                beam.frequency[freq_idx].to_value("MHz"),
                index_model=index_model,
            )
            sky_map[~horizon_mask] = np.nan

            beam_above_horizon *= np.nan

            try:
                beam_above_horizon[horizon_mask] = interpolators[freq_idx](
                    az_above_horizon, el_above_horizon
                )
            except ValueError as e:
                raise ValueError(
                    f"az min/max: {np.min(az_above_horizon), np.max(az_above_horizon)}."
                    f" el min/max: {np.min(el_above_horizon), np.max(el_above_horizon)}"
                ) from e

            # Weight the beam by the pixel resolution of the sky model.
            beam_above_horizon *= sky_model.pixel_res

            # Number of pixels above the horizon that are used.
            n_pix_ok = np.sum(~np.isnan(beam_above_horizon))

            # Number of pixels in the whole sky, but not counting pixels that are
            # above horizon and nan.
            n_pix_tot_no_nan = n_pix_tot - (len(el_above_horizon) - n_pix_ok)

            if normalize_beam:
                solid_angle = np.nansum(beam_above_horizon) / n_pix_tot_no_nan
                beam_above_horizon *= ground_gain[freq_idx] / solid_angle

            antenna_temperature_above_horizon = beam_above_horizon * sky_map
            yield (
                lst_idx,
                freq_idx,
                np.nansum(antenna_temperature_above_horizon) / n_pix_tot_no_nan,
                antenna_temperature_above_horizon,
                sky_map,
                beam_above_horizon,
                time,
                n_pix_tot_no_nan,
                az,
                el,
                interpolators[freq_idx],
            )


def simulate_spectra(
    beam: Beam,
    sky_model: sky_models.SkyModel,
    ground_loss: np.ndarray | None = None,
    f_low: float | None = 0,
    f_high: float | None = np.inf,
    normalize_beam: bool = True,
    index_model: sky_models.IndexModel = sky_models.ConstantIndex(),
    lsts: np.ndarray = None,
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
    use_astropy_azel: bool = True,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Simulate global spectra from sky and beam models.

    Parameters
    ----------
    band
        The band of the antenna (low, mid, high).
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
    lsts
        The LSTs at which to simulate

    Returns
    -------
    antenna_temperature_above_horizon
        The antenna temperature for pixels above the horizon, shape (Nlst, Nfreq)
    freq
        The frequencies at which the simulation is defined.
    lst
        The LSTs at which the sim is defined.
    """
    beam = beam.between_freqs(f_low, f_high)
    if lsts is None:
        lsts = np.arange(0, 24, 0.5)

    antenna_temperature_above_horizon = np.zeros((len(lsts), len(beam.frequency)))
    for i, j, temperature, _, _, _, _, _, _, _, _ in sky_convolution_generator(
        lsts=lsts,
        ground_loss=ground_loss,
        beam=beam,
        sky_model=sky_model,
        index_model=index_model,
        normalize_beam=normalize_beam,
        beam_smoothing=beam_smoothing,
        smoothing_model=smoothing_model,
        interp_kind=interp_kind,
        use_astropy_azel=use_astropy_azel,
    ):
        antenna_temperature_above_horizon[i, j] = temperature

    return antenna_temperature_above_horizon, beam.frequency, lsts
