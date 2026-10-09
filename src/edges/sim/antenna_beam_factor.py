"""Functions for computing the antenna beam factor."""

from typing import Literal, Self

import astropy.coordinates as apc
import attrs
import numpy as np
import scipy.interpolate as spi
from astropy import units as u
from astropy.coordinates import Longitude

from edges import const
from edges import modeling as mdl
from edges.io.serialization import hickleable
from edges.sim import sky_models
from edges.sim.beams import Beam

from .. import types as tp
from .simulate import _ground_loss_in_band, sky_convolution_generator


@hickleable
@attrs.define(slots=False)
class BeamFactor:
    """A non-interpolated beam factor.

    This class holds the attributes necessary to compute beam factors at particular
    LSTs and frequencies, namely the antenna temperature (beam-weighted integral of the
    sky) and the same at a particular reference frequency. We hold these separately
    to enable computing the beam factor in different ways from these basic quantities.

    Attributes
    ----------
    frequencies: np.ndarray
        The frequencies at which the beam-weighted sky integrals are defined.
    lsts: np.ndarray
        The LSTs at which the beam-weighted sky integrals are defined.
    reference_frequency: float
        The reference frequency.
    antenna_temp: np.ndarray
        The beam-weighted sky integrals at each frequency and LST, i.e. the numerator
        of the beam factor. The beam is always at each frequency. As computed by
        :func:`compute_antenna_beam_factor`, the sky is at the reference frequency if
        ``sky_at_reference_frequency`` is True (Eq. 4 of Sims+23; so this is *not*
        the antenna temperature that would be observed), and at each frequency
        otherwise (Eq. A1 of Sims+23; the observed antenna temperature).
    antenna_temp_ref: np.ndarray
        The beam-weighted sky integrals with the beam at the reference frequency,
        i.e. the denominator of the beam factor. As computed by
        :func:`compute_antenna_beam_factor`, this has shape ``(nlst,)`` (with the sky
        at the reference frequency) if ``sky_at_reference_frequency`` is True, and
        shape ``(nlst, nfreq)`` (with the sky at each frequency) otherwise.
    loss_fraction: np.ndarray
        One minus the ground loss times the mean beam over the whole sky,
        ``1 - ground_loss * (1/4pi) int B dOmega`` (with the beam zero below the
        horizon and at blank sky pixels), at each LST and frequency, as computed by
        :func:`compute_antenna_beam_factor`. When the beam is normalised, it is
        instead simply ``1 - ground_loss`` (i.e. zero if no ground loss was given).
        Note that it is *not* the fraction of the sky signal that is lost below the
        horizon.
    meta
        A dictionary of metadata.
    """

    frequencies: np.ndarray = attrs.field(converter=np.asarray)
    lsts: np.ndarray = attrs.field(converter=np.asarray)
    reference_frequency: float = attrs.field(
        converter=float,
    )
    antenna_temp: np.ndarray = attrs.field(converter=np.asarray)
    antenna_temp_ref: np.ndarray = attrs.field(converter=np.asarray)
    loss_fraction: np.ndarray | None = attrs.field(default=None)
    meta: dict = attrs.field(factory=dict, converter=dict)

    # The attributes that have a leading LST axis (and get interpolated/selected
    # when changing LSTs).
    _per_lst_fields = ("antenna_temp", "antenna_temp_ref", "loss_fraction")

    @property
    def nfreq(self) -> int:
        """The number of frequencies in the beam factor."""
        return len(self.frequencies)

    @property
    def nlst(self) -> int:
        """The number of LSTs in the beam factor."""
        return len(self.lsts)

    @frequencies.validator
    def _check_frequencies(self, attribute: attrs.Attribute, value: np.ndarray) -> None:
        if not np.all(np.diff(value) > 0):
            raise ValueError("Frequencies must be monotonically increasing.")

    @lsts.validator
    def _check_lsts(self, attribute: attrs.Attribute, value: np.ndarray) -> None:
        # LSTs can wrap around 24 hours, but only once.
        if np.sum(np.diff(value) < 0) > 1:
            raise ValueError("LSTs must be monotonically increasing.")

    @reference_frequency.validator
    def _check_reference_frequency(
        self, attribute: attrs.Attribute, value: float
    ) -> None:
        if value <= 0:
            raise ValueError("Reference frequency must be positive.")

    @antenna_temp.validator
    @loss_fraction.validator
    def _check_antenna_temp(
        self, attribute: attrs.Attribute, value: np.ndarray
    ) -> None:
        if attribute.name == "loss_fraction" and value is None:
            return

        if value.ndim != 2:
            raise ValueError(f"{attribute.name} must be a 2D array.")

        if value.shape != (self.nlst, self.nfreq):
            raise ValueError(f"{attribute.name} must have shape (nlst, nfreq).")

    @antenna_temp.validator
    def _check_positive(self, attribute: str, value: np.ndarray) -> None:
        if np.any(value < 0):
            raise ValueError(
                f"Antenna temperature must be positive, got min of {np.nanmin(value)}"
            )

    @antenna_temp_ref.validator
    def _check_antenna_temp_ref(self, attribute: str, value: np.ndarray) -> None:
        if value.ndim not in (1, 2):
            raise ValueError("Reference antenna temperature must be a 1D or 2D array.")

        if value.ndim == 1 and value.shape != (self.nlst,):
            raise ValueError(
                "If Reference antenna temperature is 1D, it must have shape (nlst,)."
                f"Got shape {value.shape} instead of {(self.nlst,)}"
            )

        if value.ndim == 2 and value.shape != (self.nlst, self.nfreq):
            raise ValueError(
                "If Reference antenna temperature is 2D, it must have shape "
                "(nlst, nfreq)."
            )

        if np.any(value < 0):
            raise ValueError(
                "Reference antenna temperature must be positive, "
                f"got min of {np.nanmin(value)}"
            )

    def at_lsts(self, lsts: np.ndarray, interp_kind: int | str = "cubic") -> Self:
        """Return a new BeamFactor interpolated to the given LSTs.

        If the LSTs of this object cover the full day (i.e. the gap from the last LST
        around to the first is no more than 1.5 times the largest spacing between
        consecutive LSTs), the interpolation wraps around 24 hours. Otherwise, the
        requested LSTs must lie within the range of LSTs covered by this object.

        Parameters
        ----------
        lsts
            The LSTs (in hours) at which to evaluate the beam factor.
        interp_kind
            The kind of interpolation, passed to :class:`scipy.interpolate.interp1d`.

        Raises
        ------
        ValueError
            If the LSTs of this object do not cover the full day, and some of the
            requested LSTs are outside the covered range.
        """
        tol = 1e-10
        lsts = np.asarray(lsts)

        these_lsts = self.lsts % 24
        while np.any(these_lsts < these_lsts[0]):
            these_lsts[these_lsts < these_lsts[0]] += 24

        use_lsts = np.array(lsts % 24, dtype=float)
        use_lsts[use_lsts < these_lsts[0] - tol] += 24

        wrap_gap = these_lsts[0] + 24 - these_lsts[-1]
        full_day = self.nlst > 1 and wrap_gap <= 1.5 * np.max(np.diff(these_lsts))

        if full_day:
            these_lsts = np.append(these_lsts, these_lsts[0] + 24)
        elif np.any(use_lsts > these_lsts[-1] + tol):
            raise ValueError(
                "The LSTs of this BeamFactor do not cover the full day, and some "
                "requested LSTs are outside the covered range "
                f"({these_lsts[0]:.4f}h to {these_lsts[-1] % 24:.4f}h)."
            )
        use_lsts = np.clip(use_lsts, these_lsts[0], these_lsts[-1])

        out = {}
        for k in self._per_lst_fields:
            val = getattr(self, k)
            if val is None:
                continue
            if full_day:
                val = np.concatenate((val, val[:1]), axis=0)

            out[k] = spi.interp1d(these_lsts, val, axis=0, kind=interp_kind)(use_lsts)

        return attrs.evolve(self, lsts=lsts, **out)

    def between_lsts(self, lst0: float, lst1: float) -> Self:
        """Return a new BeamFactor including only LSTs between those given.

        Parameters
        ----------
        lst0
            Lower edge of lsts in hours.
        lst1
            Upper edge of lsts in hours.
        """
        if lst1 < lst0:
            lst1 += 24

        these_lsts = self.lsts % 24
        these_lsts[these_lsts < lst0] += 24

        mask = np.logical_and(these_lsts >= lst0, these_lsts < lst1)
        if not np.any(mask):
            raise ValueError(
                f"BeamFactor does not contain any LSTs between {lst0} and {lst1}."
            )
        out = {
            k: getattr(self, k)[mask]
            for k in self._per_lst_fields
            if getattr(self, k) is not None
        }
        return attrs.evolve(self, lsts=these_lsts[mask], **out)

    def get_beam_factor(
        self, model: mdl.Model, freqs: np.ndarray | None = None
    ) -> np.ndarray:
        """Return the beam factor as a function of LST and frequency.

        This will always be normalized to unity at the reference frequency, via
        a model fit.
        """
        if freqs is None:
            freqs = self.frequencies

        bf = (self.antenna_temp.T / self.antenna_temp_ref.T).T

        fixed_model = model.at(x=self.frequencies)
        ref_bf = np.zeros(self.nlst)
        out = np.zeros((self.nlst, len(freqs)))
        for i, ibf in enumerate(bf):
            fit = fixed_model.fit(ibf)
            ref_bf = fit.evaluate(self.reference_frequency)
            out[i] = fit.evaluate(freqs) / ref_bf

        return out

    def get_mean_beam_factor(
        self, model: mdl.Model, freqs: np.ndarray | None
    ) -> np.ndarray:
        """Return the mean beam factor over all LSTs."""
        return np.mean(self.get_beam_factor(model, freqs), axis=0)

    def get_integrated_beam_factor(
        self, model: mdl.Model, freqs: np.ndarray | None = None, **fit_kwargs
    ) -> np.ndarray:
        """Return the beam factor integrated over the LST range.

        This is the ratio of summed LSTs, rather than the sum of the ratio at each LST,
        i.e. it is not the same as the mean beam factor over the LST range.
        It is normalized to unity at the reference frequency via a model fit.
        """
        if freqs is None:
            freqs = self.frequencies

        bf = np.sum(self.antenna_temp, axis=0) / np.sum(self.antenna_temp_ref, axis=0)
        fit = model.fit(self.frequencies, bf, **fit_kwargs)
        return fit.evaluate(freqs) / fit.evaluate(self.reference_frequency)


def compute_antenna_beam_factor(
    beam: Beam,
    sky_model: sky_models.SkyModel,
    ground_loss: np.ndarray | None = None,
    f_low: tp.FreqType = 0 * u.MHz,
    f_high: tp.FreqType = np.inf * u.MHz,
    normalize_beam: bool = True,
    index_model: sky_models.IndexModel = sky_models.GaussianIndex(),
    lsts: Longitude | np.ndarray | None = None,
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
    ] = "cubic",
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
    sky_model
        A sky model to use.
    ground_loss
        The ground loss (as a multiplicative factor, i.e. one for no loss) at each
        frequency of ``beam``, shape ``(Nfreq,)``. It is restricted to the
        frequencies between ``f_low`` and ``f_high`` along with the beam (an array
        with one entry per kept frequency is also accepted). It multiplies the beam
        and enters ``loss_fraction``. With ``normalize_beam=True`` it also enters the
        beam factor (as ``ground_loss(nu) / ground_loss(nu_ref)``); with
        ``normalize_beam=False`` it cancels in the beam factor.
    f_low
        Minimum frequency to keep in the simulation (frequencies otherwise defined by
        the beam).
    f_high
        Maximum frequency to keep in the simulation (frequencies otherwise defined by
        the beam).
    normalize_beam
        Whether to normalize the beam to unit integral over the visible sky.
    index_model
        An :class:`IndexModel` to use to generate different frequencies of the sky
        model.
    lsts
        The LSTs at which to compute the beam factor. Plain numbers are interpreted
        as hours. By default, every half hour over the full day.
    reference_frequency
        The frequency to take as the "reference", i.e. where the chromaticity will
        be by construction unity. By default, ``(f_low + f_high) / 2``, or, if that
        is not finite (e.g. with the default ``f_high``), the mid-point of the range
        of beam frequencies between ``f_low`` and ``f_high``.
    beam_smoothing
        Whether to smooth the beam over frequency before interpolating.
    smoothing_model
        The model used to smooth the beam over frequency, if ``beam_smoothing``.
    interp_kind
        The kind of angular interpolation of the beam. See
        :func:`~edges.sim.simulate.sky_convolution_generator`.
    lst_progress
        Whether to show a progress bar over the LSTs.
    freq_progress
        Whether to show a progress bar over the frequencies.
    location
        The location of the telescope.
    sky_at_reference_frequency
        Whether to hold the sky at the reference frequency (Eq. 4 of Sims+23), or to
        use the sky at each frequency (Eq. A1 of Sims+23). If True, the beam factor
        at frequency ``nu`` is ``int B(nu) T_sky(nu_ref) dOmega / int B(nu_ref)
        T_sky(nu_ref) dOmega``. If False, it is ``int B(nu) T_sky(nu) dOmega /
        int B(nu_ref) T_sky(nu) dOmega``. Either way, an achromatic beam gives a
        beam factor of exactly one. See :class:`BeamFactor` for what is stored in
        each case.
    use_astropy_azel
        Whether to use astropy to compute the azimuth and elevation of the sky
        pixels. If False, use Alan's method.

    Returns
    -------
    beam_factor : :class`BeamFactor` instance
    """
    ground_loss = _ground_loss_in_band(ground_loss, beam, f_low, f_high)
    beam = beam.between_freqs(f_low, f_high)

    if lsts is None:
        lsts = Longitude(np.arange(0, 24, 0.5) * u.hour)
    elif isinstance(lsts, u.Quantity):
        lsts = Longitude(lsts)
    else:
        lsts = Longitude(np.asarray(lsts, dtype=float) * u.hour)

    # Get index of reference frequency
    if reference_frequency is None:
        reference_frequency = (f_high + f_low) / 2
        if not np.isfinite(reference_frequency):
            reference_frequency = (beam.frequency.min() + beam.frequency.max()) / 2

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
        # The generator starts each LST at the reference frequency, so the reference
        # beam and sky are always set before they are used below.
        if freq_idx == indx_ref_freq:
            ref_bm = bm.copy()
            ref_sky = sky.copy()

        # sky_at_reference_frequency toggles between Eq. 4 and Eq. A1 of Sims+23.
        if sky_at_reference_frequency:
            # Eq. 4: the sky is held at the reference frequency in both the numerator
            # (beam at each frequency) and the denominator (beam at the reference
            # frequency), so that the beam factor contains no foreground spectrum.
            antenna_temperature_above_horizon[lst_idx, freq_idx] = (
                np.nansum(bm * ref_sky) / npix_no_nan
            )
            if freq_idx == indx_ref_freq:
                convolution_ref[lst_idx] = np.nansum(ref_bm * ref_sky) / npix_no_nan
        else:
            # Eq. A1: the sky is at each frequency in both the numerator (beam at each
            # frequency) and the denominator (beam at the reference frequency).
            antenna_temperature_above_horizon[lst_idx, freq_idx] = temperature
            convolution_ref[lst_idx, freq_idx] = np.nansum(ref_bm * sky) / npix_no_nan

        beamsums[lst_idx, freq_idx] = np.nansum(bm) / npix_no_nan

        # Loss fraction
        loss_fraction[lst_idx, freq_idx] = 1 - np.nansum(bm) / npix_no_nan

    return BeamFactor(
        frequencies=beam.frequency.to_value("MHz").astype(float),
        lsts=lsts.hour,
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
            "sky_at_reference_frequency": bool(sky_at_reference_frequency),
            "sky_model": sky_model.name,
            "index_model": str(index_model),
            "rotation_from_north": float(90),
        },
    )
