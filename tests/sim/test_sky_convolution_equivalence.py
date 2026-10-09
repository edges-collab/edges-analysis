"""Check that the optimised sky convolution matches a straightforward implementation.

The reference generator below is the plain implementation of
:func:`edges.sim.simulate.sky_convolution_generator`: it transforms the full sky to
az/el, evaluates the sky model and interpolates the beam at every (LST, frequency),
without hoisting or caching anything.
"""

import astropy.coordinates as apc
import numpy as np
import pytest
from astropy import units as un
from astropy.coordinates import Longitude
from pygsdata import coordinates as gscrd
from read_acq import _coordinates as crda
from sim_helpers import make_galaxy_sky

from edges import const
from edges import modeling as mdl
from edges.sim import simulate
from edges.sim.beams import Beam
from edges.sim.simulate import REFERENCE_TIME, sky_convolution_generator
from edges.sim.sky_models import ConstantIndex, GaussianIndex, SkyModel, StepIndex

LOCATION = const.KNOWN_TELESCOPES["edges-low"].location


def _reference_generator(
    lsts,
    beam,
    sky_model,
    index_model,
    normalize_beam,
    beam_smoothing,
    smoothing_model,
    ground_loss=None,
    location=LOCATION,
    ref_time=REFERENCE_TIME,
    interp_kind="sphere-spline",
    ref_freq_idx=0,
    use_astropy_azel=True,
):
    """Straightforward implementation of the sky convolution generator."""
    if beam_smoothing:
        beam = beam.smoothed(smoothing_model)

    ground_gain = (
        np.ones(len(beam.frequency))
        if ground_loss is None
        else np.asarray(ground_loss, dtype=float)
    )
    times = gscrd.lsts_to_times(lsts, ref_time, location)

    for lst_idx, time in enumerate(times):
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
                lsts[lst_idx].rad, ra, dec, location.lat.rad
            )
            az *= 180 / np.pi
            el *= 180 / np.pi
        n_pix_tot = len(el)
        horizon_mask = el > 0

        for freq_idx in np.roll(range(len(beam.frequency)), -ref_freq_idx):
            interp = beam.angular_interpolator(freq_idx, interp_kind=interp_kind)
            sky_map = sky_model.at_freq(
                beam.frequency[freq_idx].to_value("MHz"), index_model=index_model
            )
            sky_map[~horizon_mask] = np.nan

            bm = np.full(sky_model.coords.shape, np.nan)
            bm[horizon_mask] = interp(az[horizon_mask], el[horizon_mask])
            bm[np.isnan(sky_map)] = np.nan
            bm *= sky_model.pixel_res

            n_pix_ok = np.sum(~np.isnan(bm))
            n_pix_tot_no_nan = n_pix_tot - (np.sum(horizon_mask) - n_pix_ok)

            if normalize_beam:
                solid_angle = np.nansum(bm) / n_pix_tot_no_nan
                bm *= ground_gain[freq_idx] / solid_angle
            else:
                bm *= ground_gain[freq_idx] * n_pix_tot_no_nan / (4 * np.pi)

            conv = bm * sky_map
            yield (
                lst_idx,
                freq_idx,
                np.nansum(conv) / n_pix_tot_no_nan,
                conv,
                sky_map,
                bm,
                time,
                n_pix_tot_no_nan,
                az,
                el,
                interp,
            )


@pytest.fixture(scope="module")
def feko_beam() -> Beam:
    return Beam.from_file("low").between_freqs(50 * un.MHz, 58 * un.MHz)


@pytest.fixture(scope="module")
def sky_with_blanks() -> SkyModel:
    """A non-uniform sky with a few blank (NaN) pixels."""
    sky = make_galaxy_sky(nside=8)
    temp = sky.temperature.copy()
    temp[::37] = np.nan
    return SkyModel(
        frequency=sky.frequency,
        temperature=temp,
        coords=sky.coords,
        healpix=sky.healpix,
        name="blanks",
    )


def _assert_same_output(new, ref):
    """Assert that two generator outputs are identical (bit for bit)."""
    new = list(new)
    ref = list(ref)
    assert len(new) == len(ref)

    rng = np.random.default_rng(0)
    az = rng.uniform(0, 360, 50)
    el = rng.uniform(0, 90, 50)
    for n, r in zip(new, ref, strict=True):
        assert n[:2] == r[:2]
        np.testing.assert_array_equal(n[2], r[2])
        for i in (3, 4, 5, 8, 9):
            np.testing.assert_array_equal(n[i], r[i])
        assert n[6] == r[6]
        assert n[7] == r[7]
        np.testing.assert_array_equal(n[10](az, el), r[10](az, el))


@pytest.mark.parametrize("use_astropy_azel", [True, False])
@pytest.mark.parametrize(
    "interp_kind",
    [
        "linear",
        "nearest",
        "slinear",
        "cubic",
        "quintic",
        "pchip",
        "spline",
        "sphere-spline",
    ],
)
def test_generator_matches_reference(
    use_astropy_azel, interp_kind, feko_beam, sky_with_blanks
):
    """The generator gives exactly the outputs of the straightforward implementation."""
    kw = {
        "lsts": Longitude([0.3, 7.1, 17.76] * un.hour),
        "beam": feko_beam,
        "sky_model": sky_with_blanks,
        "index_model": GaussianIndex(),
        "normalize_beam": True,
        "beam_smoothing": False,
        "smoothing_model": None,
        "interp_kind": interp_kind,
        "ref_freq_idx": 2,
        "use_astropy_azel": use_astropy_azel,
    }
    _assert_same_output(
        sky_convolution_generator(**kw, lst_progress=False, freq_progress=False),
        _reference_generator(**kw),
    )


@pytest.mark.parametrize("index_model", [ConstantIndex(), StepIndex()])
@pytest.mark.parametrize("normalize_beam", [True, False])
@pytest.mark.parametrize("cache_sky_maps", [True, False])
def test_generator_matches_reference_options(
    index_model, normalize_beam, cache_sky_maps, feko_beam, monkeypatch
):
    """Other options (smoothing, ground loss, index models) also give exact results."""
    if not cache_sky_maps:
        monkeypatch.setattr(simulate, "_MAX_SKY_CACHE_BYTES", 0)
    rng = np.random.default_rng(1)
    kw = {
        "lsts": Longitude([1.0, 13.5] * un.hour),
        "beam": feko_beam,
        "sky_model": make_galaxy_sky(nside=8),
        "index_model": index_model,
        "normalize_beam": normalize_beam,
        "beam_smoothing": True,
        "smoothing_model": mdl.Polynomial(n_terms=3),
        "ground_loss": rng.uniform(0.98, 1.0, len(feko_beam.frequency)),
        "interp_kind": "cubic",
        "ref_freq_idx": 1,
    }
    _assert_same_output(
        sky_convolution_generator(**kw, lst_progress=False, freq_progress=False),
        _reference_generator(**kw),
    )


@pytest.mark.parametrize("frame", ["galactic", "icrs", "fk5"])
def test_time_independent_coords(frame):
    """Hoisting the time-independent transformation does not change the az/el."""
    coords = make_galaxy_sky(nside=16).coords.transform_to(frame)
    time = REFERENCE_TIME + 3.7 * un.hour
    altaz = apc.AltAz(location=LOCATION, obstime=time)

    direct = coords.transform_to(altaz)
    hoisted = simulate._time_independent_coords(coords).transform_to(altaz)
    np.testing.assert_array_equal(hoisted.az.deg, direct.az.deg)
    np.testing.assert_array_equal(hoisted.alt.deg, direct.alt.deg)


@pytest.mark.parametrize("batch_bytes", [1, 8 * 768 * 2])
@pytest.mark.parametrize("interp_kind", ["cubic", "nearest"])
def test_generator_matches_reference_freq_chunks(
    batch_bytes, interp_kind, feko_beam, monkeypatch
):
    """Interpolating the beam in chunks of frequencies gives exact results too."""
    # With a sky of 768 pixels, chunks of one and two frequencies.
    monkeypatch.setattr(simulate, "_MAX_BEAM_BATCH_BYTES", batch_bytes)
    kw = {
        "lsts": Longitude([4.0, 20.0] * un.hour),
        "beam": feko_beam,
        "sky_model": make_galaxy_sky(nside=8),
        "index_model": ConstantIndex(),
        "normalize_beam": True,
        "beam_smoothing": False,
        "smoothing_model": None,
        "interp_kind": interp_kind,
        "ref_freq_idx": 3,
    }
    _assert_same_output(
        sky_convolution_generator(**kw, lst_progress=False, freq_progress=False),
        _reference_generator(**kw),
    )


@pytest.mark.parametrize("interp_kind", ["linear", "cubic"])
def test_generator_out_of_bounds_beam(interp_kind):
    """A beam that does not cover the sky above the horizon gives a clear error."""
    el = np.arange(0, 81, 10.0)
    az = np.arange(0, 360, 10.0)
    beam = Beam(
        frequency=np.array([50.0, 60.0]) * un.MHz,
        azimuth=az,
        elevation=el,
        beam=np.ones((2, len(el), len(az))),
    )
    gen = sky_convolution_generator(
        lsts=Longitude([1.0] * un.hour),
        beam=beam,
        sky_model=make_galaxy_sky(nside=8),
        index_model=ConstantIndex(),
        normalize_beam=True,
        beam_smoothing=False,
        smoothing_model=None,
        interp_kind=interp_kind,
        lst_progress=False,
        freq_progress=False,
    )
    with pytest.raises(ValueError, match="el min/max"):
        next(gen)
