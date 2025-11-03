"""Functions for computing the antenna beam factor."""

from typing import Literal, Self

import astropy.coordinates as apc
import attrs
import numpy as np
import scipy.interpolate as spi
from astropy import units as u

from edges import const
from edges import modeling as mdl
from edges.io.serialization import hickleable
from edges.sim import sky_models
from edges.sim.beams import Beam

from .. import types as tp
from .simulate import sky_convolution_generator


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
        The beam-weighted sky integrals at each frequency and LST.
    antenna_temp_ref: np.ndarray
        The beam-weighted sky integrals at the reference frequency and each LST.
    loss_fraction: np.ndarray
        The fraction of the sky signal lost below the horizon.
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
        """Return a new BeamFactor at the given LSTs."""
        d = attrs.asdict(self)

        lst_like = [
            k
            for k, v in d.items()
            if isinstance(v, np.ndarray) and v.shape[0] == self.nlst
            if k != "lsts"
        ]

        these_lsts = self.lsts % 24
        while np.any(these_lsts < these_lsts[0]):
            these_lsts[these_lsts < these_lsts[0]] += 24

        use_lsts = lsts % 24
        use_lsts[use_lsts < these_lsts[0]] += 24
        these_lsts = np.append(these_lsts, these_lsts[0] + 24)
        out = {}
        for k in lst_like:
            if d[k].ndim == 2:
                val = np.vstack((d[k], d[k][0]))
            elif d[k].ndim == 1:
                val = np.concatenate((d[k], [d[k][0]]))

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
        d = attrs.asdict(self)
        lst_like = [
            k
            for k, v in d.items()
            if isinstance(v, np.ndarray) and v.shape[0] == self.nlst  # and v.ndim == 2
            if k != "lsts"
        ]

        out = {k: getattr(self, k)[mask] for k in lst_like}
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
