"""Simulation functions for ideal sky observations."""

from typing import Literal

import astropy.coordinates as apc
import astropy.time as apt
import numpy as np
from astropy import units as un
from astropy.coordinates import Longitude
from pygsdata import GSData, Telescope
from pygsdata import coordinates as gscrd
from read_acq import _coordinates as crda
from rich.progress import track

from edges.types import FreqType

from .. import const
from .. import modeling as mdl
from . import sky_models
from .beams import Beam

# Reference UTC observation time. At this time, the LST is 0.1666 (00:10 Hrs LST) at the
# EDGES location. NOTE: this is used by default, but can be changed by the user anywhere
# it is used.
REFERENCE_TIME = apt.Time("2014-01-01T09:39:42", location=const.edges_location)


def _ground_loss_in_band(
    ground_loss: np.ndarray | None,
    beam: Beam,
    f_low: FreqType = 0 * un.MHz,
    f_high: FreqType = np.inf * un.MHz,
) -> np.ndarray | None:
    """Select the ground loss at the beam frequencies kept between f_low and f_high.

    This uses the same frequency selection as :meth:`Beam.between_freqs`, so that the
    output is aligned with ``beam.between_freqs(f_low, f_high).frequency``.

    Parameters
    ----------
    ground_loss
        The ground loss (as a multiplicative factor, i.e. one for no loss) at each
        frequency of ``beam``, shape ``(Nfreq,)``. An array that already has one entry
        per frequency kept between ``f_low`` and ``f_high`` (and not one per beam
        frequency) is assumed to be aligned with the kept frequencies, and returned
        as is.
    beam
        The beam, *before* restricting it to the frequency range.
    f_low
        Minimum frequency to keep.
    f_high
        Maximum frequency to keep.

    Returns
    -------
    ground_loss
        The ground loss at the kept frequencies, or None if ``ground_loss`` is None.

    Raises
    ------
    ValueError
        If ``ground_loss`` is not 1D with one entry per beam frequency (or per kept
        frequency).
    """
    if ground_loss is None:
        return None

    ground_loss = np.asarray(ground_loss, dtype=float)
    mask = (beam.frequency >= f_low) & (beam.frequency <= f_high)
    if ground_loss.shape == (len(beam.frequency),):
        return ground_loss[mask]
    if ground_loss.shape == (np.sum(mask),):
        return ground_loss

    raise ValueError(
        f"ground_loss must have shape ({len(beam.frequency)},), one entry per beam "
        f"frequency (or ({np.sum(mask)},), one per frequency between f_low and "
        f"f_high). Got shape {ground_loss.shape}."
    )


def _time_independent_coords(coords: apc.SkyCoord) -> apc.SkyCoord:
    """Transform coordinates to the frame from which they are transformed to AltAz.

    Astropy transforms Galactic coordinates to AltAz via ICRS, and the Galactic to
    ICRS step does not depend on the observation time. Doing it once, rather than
    for every LST, gives the same az/el. Coordinates in other frames are returned
    unchanged.

    Parameters
    ----------
    coords
        The coordinates of the sky model.

    Returns
    -------
    coords
        The same coordinates, in ICRS if they were Galactic.
    """
    if isinstance(coords.frame, apc.Galactic):
        return coords.icrs
    return coords


def sky_convolution_generator(
    lsts: Longitude,
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
        The ground loss (as a multiplicative factor, i.e. one for no loss) at each
        frequency of ``beam``, shape ``(Nfreq,)``. It multiplies the beam, whether or
        not the beam is normalised.
    beam
        The beam to convolve.
    sky_model
        The sky model to convolve
    index_model
        The spectral index model of the sky model.
    normalize_beam
        Whether to normalise the beam to unit mean over the (non-blank) sky above the
        horizon. If True, ``mean_conv_temp`` is the beam-weighted mean sky
        temperature, ``int B T dOmega / int B dOmega``. If False, the beam is used
        as given, and ``mean_conv_temp`` is ``(1/4pi) int B T dOmega``, which does
        not depend on the resolution of the sky model.
    beam_smoothing
        Whether to smooth the beam over the frequency axis (with
        ``smoothing_model``) before interpolating it.
    smoothing_model
        The model used to smooth the beam over frequency. Only used if
        ``beam_smoothing`` is True.
    location
        The location of the telescope.
    ref_time
        A reference time around which the times corresponding to ``lsts`` are found.
    interp_kind
        The kind of interpolation to use for the beam. "spline" uses
        :class:`scipy.interpolate.RectBivariateSpline` and "sphere-spline" uses
        :class:`scipy.interpolate.RectSphereBivariateSpline`. All other options use
        :class:`scipy.interpolate.RegularGridInterpolator`. with the given kind as
        ``method``.
    lst_progress
        Whether to show a progress bar over LSTs.
    freq_progress
        Whether to show a progress bar over frequencies.
    ref_freq_idx
        The index of the frequency at which to start the frequency loop (the
        frequencies are iterated cyclically starting from this index).
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
        The beam-weighted sky temperature above the horizon (see ``normalize_beam``),
        equal to ``nansum(conv_temp) / n_pixels``.
    conv_temp
        An array containing the temperature after multiplying by the beam in each pixel
        above the horizon.
    sky
        An array containing the sky temperature in pixel above the horizon.
    beam
        An array containing the interpolated beam in pixels above the horizon (NaN
        elsewhere, and where the sky is NaN), weighted by the solid angle of each
        pixel and scaled (see ``normalize_beam``) so that
        ``nansum(beam * sky) / n_pixels`` is ``mean_conv_temp``. A new array is
        yielded on every iteration.
    time
        The local time at each LST.
    n_pixels
        The total number of pixels that are not masked.
    az
        The azimuth (degrees) of every sky pixel at this LST.
    el
        The elevation (degrees) of every sky pixel at this LST.
    interpolator
        The angular beam interpolator used for this frequency.

    Examples
    --------
    Use this function as follows:

    >>> for i, j, mean_t, conv_t, sky, bm, time, npix, az, el, interp in (
    ...     sky_convolution_generator(...)
    ... ):
    ...     print(conv_t)
    """
    if beam_smoothing:
        beam = beam.smoothed(smoothing_model)

    if ground_loss is None:
        ground_gain = np.ones(len(beam.frequency))
    else:
        ground_gain = np.asarray(ground_loss, dtype=float)
        if ground_gain.shape != (len(beam.frequency),):
            raise ValueError(
                f"ground_loss must have shape ({len(beam.frequency)},), one entry per "
                f"beam frequency. Got shape {ground_gain.shape}."
            )

    # Get the local times corresponding to the given LSTs
    times = gscrd.lsts_to_times(lsts, ref_time, location)

    interpolators = {}

    # The parts of the transformation to local coordinates that do not depend on
    # time are done only once, before looping over the LSTs.
    if use_astropy_azel:
        coords = _time_independent_coords(sky_model.coords)
    else:
        gal = sky_model.coords.galactic
        ra, dec = crda.galactic_to_radec(gal.b.deg, gal.l.deg)
        sin_dec, cos_dec = np.sin(dec), np.cos(dec)
        lat = location.lat.rad
        sin_lat, cos_lat = np.sin(lat), np.cos(lat)

    for lst_idx, time in track(
        enumerate(times), description="LSTs", disable=not lst_progress, total=len(times)
    ):
        # Transform Galactic coordinates of Sky Model to Local coordinates
        if use_astropy_azel:
            altaz = coords.transform_to(apc.AltAz(location=location, obstime=time))
            az = np.asarray(altaz.az.deg)
            el = np.asarray(altaz.alt.deg)
        else:
            # This is crda.radec_azel_from_lst, with the sines and cosines of the
            # declination computed once for all LSTs.
            az, el = crda.radec_azel2(
                lsts[lst_idx].rad - ra, sin_dec, cos_dec, sin_lat, cos_lat
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
        for freq_idx in track(
            np.roll(range(len(beam.frequency)), -ref_freq_idx),
            description="Frequencies",
            disable=not freq_progress,
            total=len(beam.frequency),
            transient=True,
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

            # A fresh array on every iteration, so that the yielded arrays are not
            # overwritten by later iterations (e.g. when calling ``list()`` on this).
            beam_above_horizon = np.full(sky_model.coords.shape, np.nan)

            try:
                beam_above_horizon[horizon_mask] = interpolators[freq_idx](
                    az_above_horizon, el_above_horizon
                )
            except ValueError as e:
                raise ValueError(
                    f"az min/max: {np.min(az_above_horizon), np.max(az_above_horizon)}."
                    f" el min/max: {np.min(el_above_horizon), np.max(el_above_horizon)}"
                ) from e

            # Blank (NaN) sky pixels are excluded from the beam as well, so that they
            # are consistently left out of both the beam normalisation and the
            # beam-weighted sky.
            beam_above_horizon[np.isnan(sky_map)] = np.nan

            # Weight the beam by the pixel resolution of the sky model.
            beam_above_horizon *= sky_model.pixel_res

            # Number of pixels above the horizon that are used.
            n_pix_ok = np.sum(~np.isnan(beam_above_horizon))

            # Number of pixels in the whole sky, but not counting pixels that are
            # above horizon and nan (in either the beam or the sky).
            n_pix_tot_no_nan = n_pix_tot - (len(el_above_horizon) - n_pix_ok)

            if normalize_beam:
                solid_angle = np.nansum(beam_above_horizon) / n_pix_tot_no_nan
                beam_above_horizon *= ground_gain[freq_idx] / solid_angle
            else:
                # Apply the ground loss, and scale so that the mean over pixels below
                # gives (1/4pi) * int B T dOmega, independent of the sky resolution.
                beam_above_horizon *= (
                    ground_gain[freq_idx] * n_pix_tot_no_nan / (4 * np.pi)
                )

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
    lsts: Longitude,
    sky_model: sky_models.SkyModel,
    ground_loss: np.ndarray | None = None,
    f_low: FreqType = 0 * un.MHz,
    f_high: FreqType = np.inf * un.MHz,
    normalize_beam: bool = True,
    index_model: sky_models.IndexModel = sky_models.ConstantIndex(),
    beam_smoothing: bool = True,
    smoothing_model: mdl.Model = mdl.Polynomial(n_terms=12),
    telescope: Telescope = const.KNOWN_TELESCOPES["edges-low"],
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
    use_astropy_azel: bool = True,
) -> GSData:
    """
    Simulate global spectra from sky and beam models.

    Parameters
    ----------
    beam
        A :class:`Beam` object.
    lsts
        The LSTs at which to simulate
    sky_model
        A sky model to use.
    ground_loss
        The ground loss (as a multiplicative factor, i.e. one for no loss) at each
        frequency of ``beam``, shape ``(Nfreq,)``. It is restricted to the
        frequencies between ``f_low`` and ``f_high`` along with the beam (an array
        with one entry per kept frequency is also accepted). It multiplies the beam,
        whether or not the beam is normalised.
    f_low
        Minimum frequency to keep in the simulation (frequencies otherwise defined by
        the beam).
    f_high
        Maximum frequency to keep in the simulation (frequencies otherwise defined by
        the beam).
    normalize_beam
        Whether to normalize the beam to unit integral over the visible sky. If True,
        the spectra are the beam-weighted mean sky temperature,
        ``int B T dOmega / int B dOmega``. If False, the beam is used as given, and
        the spectra are ``(1/4pi) int B T dOmega``.
    index_model
        An :class:`IndexModel` to use to generate different frequencies of the sky
        model.
    beam_smoothing
        Whether to smooth the beam in frequency before interpolating. This can help
        with interpolation artifacts, but can also make the simulation less accurate.
    smoothing_model
        The model to use for smoothing the beam in frequency. Only used if
        ``beam_smoothing`` is True.
    telescope
        The telescope to use for the simulation. Specifies the location and
        polarizations of the simulation.
    ref_time
        The reference time to use for the simulation. This is used to convert between
        LST and local time. By default, this is set to a time when the LST is 0.1666
        (00:10 Hrs LST) at the EDGES location, but can be set to any time.
    interp_kind
        The kind of interpolation to use for the beam. "spline" uses
        :class:`scipy.interpolate.RectBivariateSpline` and "sphere-spline" uses
        :class:`scipy.interpolate.RectSphereBivariateSpline`. All other options use
        :class:`scipy.interpolate.RegularGridInterpolator`. with the given kind as
        ``method``.
    use_astropy_azel
        Whether to use the astropy coordinate system for azimuth and elevation. If
        False, compute the az/el using Alan's method.

    Returns
    -------
    spectra
        A :class:`GSData` object containing the simulated spectra. The data array has
        shape (1, 1, Ntimes, Nfreqs), where the first two dimensions are singleton
        dimensions for loads and polarizations. The frequencies are taken from the beam,
        but only those between ``f_low`` and ``f_high`` are kept.

    """
    ground_loss = _ground_loss_in_band(ground_loss, beam, f_low, f_high)
    beam = beam.between_freqs(f_low, f_high)

    antenna_temperature_above_horizon = np.zeros((len(lsts), len(beam.frequency)))
    times = []
    for i, j, temperature, _, _, _, time, _, _, _, _ in sky_convolution_generator(
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
        location=telescope.location,
        ref_time=ref_time,
    ):
        antenna_temperature_above_horizon[i, j] = temperature

        # Only append the time for a single frequency on each loop
        if j == 0:
            times.append(time)

    return GSData(
        data=antenna_temperature_above_horizon[None, None],
        freqs=beam.frequency,
        lsts=lsts[:, None],
        times=apt.Time(times)[:, None],
        telescope=telescope,
        pols=telescope.pols,
        loads=("ant",),
    )
