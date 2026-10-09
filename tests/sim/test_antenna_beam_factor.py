"""Physical and behavioural tests of the antenna beam factor."""

import numpy as np
import pytest
from astropy import units as un
from astropy.coordinates import Longitude
from sim_helpers import make_achromatic_beam

from edges import modeling as mdl
from edges.sim import compute_antenna_beam_factor
from edges.sim.antenna_beam_factor import BeamFactor
from edges.sim.beams import Beam
from edges.sim.sky_models import ConstantIndex, SkyModel

FAST = {
    "lst_progress": False,
    "freq_progress": False,
    "beam_smoothing": False,
    "interp_kind": "linear",
    "index_model": ConstantIndex(index=2.5),
}


def _achromatic_bf(galaxy_sky, *, sky_at_reference_frequency: bool) -> BeamFactor:
    return compute_antenna_beam_factor(
        beam=make_achromatic_beam(np.arange(50.0, 101.0, 5.0)),
        sky_model=galaxy_sky,
        lsts=Longitude([0.0, 6.0, 12.0, 18.0] * un.hour),
        reference_frequency=75 * un.MHz,
        sky_at_reference_frequency=sky_at_reference_frequency,
        use_astropy_azel=False,
        **FAST,
    )


def test_achromatic_beam_factor_is_unity_eq_a1(galaxy_sky):
    """With sky_at_reference_frequency=False, an achromatic beam gives BF == 1."""
    bf = _achromatic_bf(galaxy_sky, sky_at_reference_frequency=False)
    np.testing.assert_allclose(bf.antenna_temp / bf.antenna_temp_ref, 1.0, rtol=1e-12)
    np.testing.assert_allclose(
        bf.get_beam_factor(mdl.Polynomial(n_terms=4)), 1.0, rtol=1e-8
    )


def test_achromatic_beam_factor_is_unity_eq_4(galaxy_sky):
    """With sky_at_reference_frequency=True, an achromatic beam also gives BF == 1."""
    bf = _achromatic_bf(galaxy_sky, sky_at_reference_frequency=True)
    assert bf.antenna_temp_ref.shape == (bf.nlst,)
    assert bf.meta["sky_at_reference_frequency"]
    np.testing.assert_allclose(
        bf.antenna_temp / bf.antenna_temp_ref[:, None], 1.0, rtol=1e-12
    )
    np.testing.assert_allclose(
        bf.get_beam_factor(mdl.Polynomial(n_terms=4)), 1.0, rtol=1e-8
    )


def _chromatic_beam(freqs: np.ndarray) -> Beam:
    """A beam that narrows with frequency (and is azimuthally asymmetric)."""
    az = np.arange(0, 360, 5.0)
    el = np.arange(0, 91, 5.0)
    power = 2 * (freqs / 75.0)
    pattern = np.sin(np.radians(el))[None, :, None] ** power[:, None, None] * (
        1 + 0.3 * np.cos(2 * np.radians(az))
    )
    return Beam(
        frequency=freqs * un.MHz,
        azimuth=az,
        elevation=el,
        beam=pattern,
        simulator="analytic",
    )


@pytest.mark.parametrize("normalize_beam", [True, False])
def test_eq_4_beam_factor_independent_of_sky_spectrum(galaxy_sky, normalize_beam):
    """With the sky at the reference frequency, the sky's spectral index is irrelevant.

    The beam factor is int B(nu) T_sky(nu_ref) / int B(nu_ref) T_sky(nu_ref), which
    contains no foreground spectrum, so it is the same for any spectral index (but
    not identically one, since the beam is chromatic). The reference frequency is
    the sky model's own frequency, so that the CMB offset in the sky model does not
    make the reference sky depend on the spectral index.
    """
    out = {}
    for beta in (2.0, 2.5, 3.0):
        bf = compute_antenna_beam_factor(
            beam=_chromatic_beam(np.arange(55.0, 96.0, 10.0)),
            sky_model=galaxy_sky,
            lsts=Longitude([0.0, 17.76] * un.hour),
            reference_frequency=galaxy_sky.frequency * un.MHz,
            sky_at_reference_frequency=True,
            normalize_beam=normalize_beam,
            use_astropy_azel=False,
            **{**FAST, "index_model": ConstantIndex(index=beta)},
        )
        out[beta] = bf.antenna_temp / bf.antenna_temp_ref[:, None]

    np.testing.assert_allclose(out[2.0], out[2.5], rtol=1e-12)
    np.testing.assert_allclose(out[3.0], out[2.5], rtol=1e-12)
    np.testing.assert_allclose(out[2.5][:, 2], 1.0, rtol=1e-12)  # nu_ref = 75 MHz
    assert np.ptp(out[2.5]) > 0.01


def test_default_lsts_and_reference_frequency(galaxy_sky):
    """The defaults for lsts and reference_frequency work (and are sensible)."""
    beam = make_achromatic_beam(np.array([50.0, 60.0, 70.0]), delta_az=10, delta_el=10)
    bf = compute_antenna_beam_factor(
        beam=beam, sky_model=galaxy_sky, use_astropy_azel=False, **FAST
    )
    np.testing.assert_allclose(bf.lsts, np.arange(0, 24, 0.5), atol=1e-10)
    assert bf.reference_frequency == pytest.approx(60.0, rel=1e-12)


def test_default_reference_frequency_with_finite_bounds(galaxy_sky):
    beam = make_achromatic_beam(np.array([50.0, 60.0, 70.0]), delta_az=10, delta_el=10)
    bf = compute_antenna_beam_factor(
        beam=beam,
        sky_model=galaxy_sky,
        lsts=[1.0],
        f_low=40 * un.MHz,
        f_high=100 * un.MHz,
        use_astropy_azel=False,
        **FAST,
    )
    assert bf.reference_frequency == pytest.approx(70.0, rel=1e-12)


@pytest.mark.parametrize(
    "lsts",
    [
        np.array([1.5, 7.25]),
        [1.5, 7.25],
        Longitude([1.5, 7.25] * un.hour),
        Longitude([22.5, 108.75] * un.deg),
    ],
)
@pytest.mark.parametrize("use_astropy_azel", [True, False])
def test_lsts_stored_in_hours(lsts, use_astropy_azel, galaxy_sky):
    """LSTs given as floats (hours) or as angles of any unit are stored in hours."""
    beam = make_achromatic_beam(np.array([50.0, 60.0, 70.0]), delta_az=10, delta_el=10)
    bf = compute_antenna_beam_factor(
        beam=beam,
        sky_model=galaxy_sky,
        lsts=lsts,
        use_astropy_azel=use_astropy_azel,
        **FAST,
    )
    np.testing.assert_allclose(bf.lsts, [1.5, 7.25], rtol=1e-12)

    ref = compute_antenna_beam_factor(
        beam=beam,
        sky_model=galaxy_sky,
        lsts=Longitude([1.5, 7.25] * un.hour),
        use_astropy_azel=use_astropy_azel,
        **FAST,
    )
    np.testing.assert_allclose(bf.antenna_temp, ref.antenna_temp, rtol=1e-12)


@pytest.mark.parametrize("nside", [4, 8])
def test_unnormalised_loss_fraction_is_one_minus_mean_beam(nside):
    """Without normalisation, a unit beam above the horizon covers half the sky."""
    sky = SkyModel.uniform_healpix(408.0, temperature=1000.0, nside=nside)
    bf = compute_antenna_beam_factor(
        beam=Beam.uniform(f_low=50, f_high=56, delta_f=2, delta_az=2, delta_el=2),
        sky_model=sky,
        lsts=[2.0],
        reference_frequency=52 * un.MHz,
        normalize_beam=False,
        use_astropy_azel=False,
        **FAST,
    )
    # The fraction of HEALPix pixel centres above the horizon is ~1/2.
    np.testing.assert_allclose(bf.loss_fraction, 0.5, atol=0.01)
    np.testing.assert_allclose(bf.antenna_temp / bf.antenna_temp_ref[:, None], 1.0)


@pytest.mark.parametrize("normalize_beam", [True, False])
def test_ground_loss_aligned_in_beam_factor(normalize_beam):
    """The ground loss is cut with the beam, and multiplies the beam.

    For a normalised beam, it enters the beam factor as g(nu)/g(nu_ref); for an
    unnormalised beam it cancels in the beam factor. Either way it enters the loss
    fraction.
    """
    beam = Beam.uniform(f_low=50, f_high=58, delta_f=2, delta_az=2, delta_el=2)
    sky = SkyModel.uniform_healpix(408.0, temperature=1000.0, nside=4)
    kw = {
        "beam": beam,
        "sky_model": sky,
        "lsts": [2.0],
        "f_low": 52 * un.MHz,
        "reference_frequency": 54 * un.MHz,
        "normalize_beam": normalize_beam,
        "use_astropy_azel": False,
        **FAST,
    }
    ground_loss = np.array([1.0, 0.9, 0.8, 0.7])  # at 50, 52, 54, 56 MHz
    with_loss = compute_antenna_beam_factor(ground_loss=ground_loss, **kw)
    without = compute_antenna_beam_factor(**kw)

    np.testing.assert_allclose(with_loss.frequencies, [52, 54, 56])
    gain_ratio = ground_loss[1:] / 0.8 if normalize_beam else 1.0
    np.testing.assert_allclose(
        with_loss.antenna_temp / with_loss.antenna_temp_ref[:, None],
        gain_ratio * without.antenna_temp / without.antenna_temp_ref[:, None],
        rtol=1e-12,
    )
    np.testing.assert_allclose(
        1 - with_loss.loss_fraction,
        ground_loss[1:] * (1 - without.loss_fraction),
        rtol=1e-12,
    )


def _square_bf(lsts: np.ndarray, *, with_extras: bool = True) -> BeamFactor:
    """A BeamFactor with nfreq == nlst, varying smoothly in LST and frequency."""
    n = len(lsts)
    freqs = np.linspace(50, 100, n)
    at = (1 + 0.1 * np.sin(2 * np.pi * lsts / 24))[:, None] * (freqs / 75)[None, :]
    return BeamFactor(
        frequencies=freqs,
        lsts=lsts,
        reference_frequency=75.0,
        antenna_temp=at,
        antenna_temp_ref=at[:, n // 2],
        loss_fraction=0.01 * at if with_extras else None,
    )


def test_at_lsts_with_nfreq_equal_nlst():
    bf = _square_bf(np.arange(0, 24, 1.0))
    assert bf.nfreq == bf.nlst
    new = bf.at_lsts(np.array([0.5, 6.0, 23.5]), interp_kind="linear")
    np.testing.assert_array_equal(new.frequencies, bf.frequencies)
    assert new.antenna_temp.shape == (3, bf.nfreq)
    assert new.antenna_temp_ref.shape == (3,)
    assert new.loss_fraction.shape == (3, bf.nfreq)
    np.testing.assert_allclose(new.antenna_temp[1], bf.antenna_temp[6], rtol=1e-12)
    # 23.5 wraps around between 23 and 0 (== 24)
    np.testing.assert_allclose(
        new.antenna_temp[2],
        (bf.antenna_temp[23] + bf.antenna_temp[0]) / 2,
        rtol=1e-12,
    )


def test_at_lsts_without_loss_fraction():
    bf = _square_bf(np.arange(0, 24, 1.0), with_extras=False)
    new = bf.at_lsts(np.array([0.5, 6.0]))
    assert new.loss_fraction is None


def test_between_lsts_with_nfreq_equal_nlst():
    bf = _square_bf(np.arange(0, 24, 1.0))
    new = bf.between_lsts(3, 8)
    np.testing.assert_array_equal(new.frequencies, bf.frequencies)
    np.testing.assert_array_equal(new.lsts, [3, 4, 5, 6, 7])
    np.testing.assert_array_equal(new.antenna_temp, bf.antenna_temp[3:8])
    np.testing.assert_array_equal(new.antenna_temp_ref, bf.antenna_temp_ref[3:8])
    np.testing.assert_array_equal(new.loss_fraction, bf.loss_fraction[3:8])


def test_at_lsts_partial_coverage():
    """With partial LST coverage, there is no wrapping/extrapolation across the gap."""
    bf = _square_bf(np.arange(0, 12, 1.0))
    new = bf.at_lsts(np.array([0.0, 5.5, 11.0]), interp_kind="linear")
    np.testing.assert_allclose(
        new.antenna_temp[1], (bf.antenna_temp[5] + bf.antenna_temp[6]) / 2, rtol=1e-12
    )
    np.testing.assert_allclose(new.antenna_temp[2], bf.antenna_temp[-1], rtol=1e-12)

    with pytest.raises(ValueError, match="do not cover the full day"):
        bf.at_lsts(np.array([18.0]))
    with pytest.raises(ValueError, match="do not cover the full day"):
        bf.at_lsts(np.array([11.5]))


def test_at_lsts_partial_coverage_across_midnight():
    lsts = np.array([20.0, 21, 22, 23, 24, 25, 26])
    bf = _square_bf(lsts)
    new = bf.at_lsts(np.array([23.5, 0.5]), interp_kind="linear")
    np.testing.assert_allclose(
        new.antenna_temp[1], (bf.antenna_temp[4] + bf.antenna_temp[5]) / 2, rtol=1e-12
    )
    with pytest.raises(ValueError, match="do not cover the full day"):
        bf.at_lsts(np.array([12.0]))
