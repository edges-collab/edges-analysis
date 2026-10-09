import numpy as np
import pytest
from astropy import units as un
from astropy.coordinates import Angle, Galactic, Longitude, SkyCoord
from sim_helpers import make_achromatic_beam, make_galaxy_sky

from edges import const
from edges.sim import Beam, simulate_spectra
from edges.sim.simulate import sky_convolution_generator
from edges.sim.sky_models import ConstantIndex, GaussianIndex, SkyModel

T_CMB = 2.725  # the value hard-coded in SkyModel.at_freq

FAST = {
    "beam_smoothing": False,
    "interp_kind": "linear",
}


def test_simulate_spectra():
    beam = Beam.from_file("low")

    # Do a really small simulation
    spectra = simulate_spectra(
        beam=beam,
        f_low=50 * un.MHz,
        f_high=55 * un.MHz,
        lsts=Longitude(np.arange(0, 24, 12) * un.hour),
        sky_model=SkyModel.uniform_healpix(408, nside=8),
        use_astropy_azel=False,
        index_model=GaussianIndex(),
    )

    assert np.all(spectra.data >= 0)
    assert spectra.ntimes == 2
    assert spectra.nfreqs == 3  # df = 2 MHz
    assert spectra.nloads == 1
    assert spectra.npols == 1


@pytest.fixture(scope="module")
def feko_beam() -> Beam:
    return Beam.from_file("low").between_freqs(50 * un.MHz, 54 * un.MHz)


def _get_beam(kind: str, feko_beam: Beam) -> Beam:
    if kind == "uniform":
        return Beam.uniform(f_low=50, f_high=56, delta_f=2, delta_az=2, delta_el=2)
    if kind == "gaussian":
        return Beam.gaussian(
            dish_size=1.0, f_low=50, f_high=56, delta_f=2, delta_az=2, delta_el=2
        )
    if kind == "achromatic":
        return make_achromatic_beam(np.array([50.0, 52.0, 54.0]))
    return feko_beam


@pytest.mark.parametrize("beam_kind", ["uniform", "gaussian", "achromatic", "feko"])
@pytest.mark.parametrize("use_astropy_azel", [True, False])
def test_uniform_sky_gives_sky_temperature(beam_kind, use_astropy_azel, feko_beam):
    """A uniform sky seen through any normalised beam is just the sky temperature."""
    beam = _get_beam(beam_kind, feko_beam)
    sky = SkyModel.uniform_healpix(frequency=408.0, temperature=1000.0, nside=4)

    spectra = simulate_spectra(
        beam=beam,
        lsts=Longitude([2.0, 17.76] * un.hour),
        sky_model=sky,
        index_model=ConstantIndex(),
        use_astropy_azel=use_astropy_azel,
        **FAST,
    )
    expected = np.array([
        sky.at_freq(f, index_model=ConstantIndex())[0]
        for f in beam.frequency.to_value("MHz")
    ])
    np.testing.assert_allclose(
        spectra.data[0, 0], np.broadcast_to(expected, (2, len(expected))), rtol=1e-6
    )


@pytest.mark.parametrize("use_astropy_azel", [True, False])
def test_achromatic_beam_power_law_sky(use_astropy_azel, galaxy_sky, achromatic_beam):
    """An achromatic beam on a constant-index sky gives T - T_cmb propto nu^-beta."""
    beta = 2.5
    spectra = simulate_spectra(
        beam=achromatic_beam,
        lsts=Longitude([0.0, 6.0, 17.76] * un.hour),
        sky_model=galaxy_sky,
        index_model=ConstantIndex(index=beta),
        use_astropy_azel=use_astropy_azel,
        normalize_beam=True,
        **FAST,
    )
    nu = achromatic_beam.frequency.to_value("MHz")
    amp = (spectra.data[0, 0] - T_CMB) * (nu / nu[0]) ** beta
    np.testing.assert_allclose(amp, amp[:, :1] * np.ones_like(amp), rtol=1e-10)

    # And the sky actually varies with LST (i.e. the test is not trivial).
    assert np.ptp(amp[:, 0]) / amp[0, 0] > 0.1


def test_sidereal_periodicity(galaxy_sky, achromatic_beam):
    """LST and LST + 24h (one sidereal day later) give the same antenna temperature."""
    lsts = Angle([3.0, 27.0] * un.hourangle)
    temps = {}
    times = {}
    for i, j, t, *_, time, _npix, _az, _el, _interp in sky_convolution_generator(
        lsts=lsts,
        beam=achromatic_beam,
        sky_model=galaxy_sky,
        index_model=ConstantIndex(),
        normalize_beam=True,
        beam_smoothing=False,
        smoothing_model=None,
        interp_kind="linear",
        lst_progress=False,
        freq_progress=False,
        use_astropy_azel=True,
        location=const.edges_location,
    ):
        temps[i, j] = t
        times[i] = time

    # The two LSTs really are a sidereal day apart in time.
    dt = (times[1] - times[0]).to_value("hour")
    np.testing.assert_allclose(dt, 23.9344696, atol=1e-3)
    for j in range(len(achromatic_beam.frequency)):
        np.testing.assert_allclose(temps[0, j], temps[1, j], rtol=1e-5)


@pytest.mark.parametrize("use_astropy_azel", [True, False])
def test_sky_hot_only_below_horizon_gives_zero(use_astropy_azel, achromatic_beam):
    """A sky that is only bright below the horizon contributes nothing."""
    lsts = Longitude([5.0] * un.hour)
    sky = make_galaxy_sky(nside=8, frequency=50.0)

    # Get the elevation of each sky pixel at this LST from the simulator itself.
    gen = sky_convolution_generator(
        lsts=lsts,
        beam=achromatic_beam,
        sky_model=sky,
        index_model=ConstantIndex(index=0.0),
        normalize_beam=True,
        beam_smoothing=False,
        smoothing_model=None,
        interp_kind="linear",
        lst_progress=False,
        freq_progress=False,
        use_astropy_azel=use_astropy_azel,
        location=const.edges_location,
    )
    el = next(gen)[9]

    # Bright (with a margin of 1 deg below the horizon), and zero above it. With
    # spectral index 0, T_cmb drops out of the frequency scaling.
    hot = SkyModel(
        frequency=50.0,
        temperature=np.where(el < -1.0, 1e4, 0.0),
        coords=sky.coords,
    )
    assert np.sum(hot.temperature > 0) > 0.4 * len(el)

    spectra = simulate_spectra(
        beam=achromatic_beam,
        lsts=lsts,
        sky_model=hot,
        index_model=ConstantIndex(index=0.0),
        use_astropy_azel=use_astropy_azel,
        telescope=const.KNOWN_TELESCOPES["edges-low"],
        **FAST,
    )
    np.testing.assert_allclose(spectra.data, 0.0, atol=1e-12)


@pytest.mark.parametrize("use_astropy_azel", [True, False])
def test_galactic_centre_transit_lst(use_astropy_azel):
    """The Galactic centre transits the meridian at LST ~17.76 h at EDGES.

    At the EDGES latitude the GC culminates ~2 deg south of the zenith, so its azimuth
    swings rapidly through 180 deg at transit.
    """
    beam = Beam.uniform(f_low=50, f_high=52, delta_f=2)
    coords = SkyCoord(l=[0.0, 90.0] * un.deg, b=[0.0, 0.0] * un.deg, frame=Galactic())
    sky = SkyModel(frequency=50.0, temperature=np.ones(2), coords=coords)
    lst_grid = np.arange(17.5, 18.0, 0.02)

    az_el = np.array([
        (out[8][0], out[9][0])
        for out in sky_convolution_generator(
            lsts=Longitude(lst_grid * un.hour),
            beam=beam,
            sky_model=sky,
            index_model=ConstantIndex(),
            normalize_beam=True,
            beam_smoothing=False,
            smoothing_model=None,
            interp_kind="linear",
            lst_progress=False,
            freq_progress=False,
            use_astropy_azel=use_astropy_azel,
            location=const.edges_location,
        )
    ])
    az = az_el[:, 0]
    (cross,) = np.where(np.diff(np.sign(az - 180.0)) != 0)
    assert len(cross) == 1
    k = cross[0]
    transit = lst_grid[k] + (180 - az[k]) / (az[k + 1] - az[k]) * (
        lst_grid[k + 1] - lst_grid[k]
    )
    assert transit == pytest.approx(const.galactic_centre_lst, abs=0.05)
    assert np.max(az_el[:, 1]) > 85.0


@pytest.mark.xfail(
    strict=True,
    reason=(
        "With normalize_beam=False the output is sum(B*dOmega*T)/Npix, which "
        "depends on the sky resolution and is not (1/4pi) * int(B*T dOmega); "
        "fix pending (result-changing)"
    ),
)
def test_unnormalised_beam_is_resolution_independent():
    """(1/4pi) int B T dOmega for B=1 above the horizon and uniform T is T/2."""
    beam = Beam.uniform(f_low=50, f_high=54, delta_f=2)
    out = {}
    for nside in (4, 8):
        sky = SkyModel.uniform_healpix(408.0, temperature=1000.0, nside=nside)
        expected = np.array([
            sky.at_freq(f, index_model=ConstantIndex())[0]
            for f in beam.frequency.to_value("MHz")
        ])
        out[nside] = simulate_spectra(
            beam=beam,
            lsts=Longitude([2.0] * un.hour),
            sky_model=sky,
            index_model=ConstantIndex(),
            normalize_beam=False,
            use_astropy_azel=False,
            **FAST,
        ).data[0, 0, 0]
        np.testing.assert_allclose(out[nside], expected / 2, rtol=0.02)
    np.testing.assert_allclose(out[4], out[8], rtol=0.02)


@pytest.mark.xfail(
    strict=True,
    reason=(
        "ground_loss is indexed by the frequency index after the f_low cut, so "
        "it is misaligned with the beam frequencies; fix pending (result-changing)"
    ),
)
def test_ground_loss_aligned_after_flow_cut():
    beam = Beam.uniform(f_low=50, f_high=58, delta_f=2)  # 50, 52, 54, 56 MHz
    sky = SkyModel.uniform_healpix(408.0, temperature=1000.0, nside=4)
    ground_loss = np.array([1.0, 0.5, 0.5, 0.5])
    spectra = simulate_spectra(
        beam=beam,
        lsts=Longitude([2.0] * un.hour),
        sky_model=sky,
        ground_loss=ground_loss,
        f_low=52 * un.MHz,
        index_model=ConstantIndex(),
        use_astropy_azel=False,
        **FAST,
    )
    expected = np.array([
        sky.at_freq(f, index_model=ConstantIndex())[0] for f in (52.0, 54.0, 56.0)
    ])
    np.testing.assert_allclose(spectra.data[0, 0, 0], 0.5 * expected, rtol=1e-6)


@pytest.mark.xfail(
    strict=True,
    reason=(
        "ground_loss is silently ignored when normalize_beam=False; "
        "fix pending (result-changing)"
    ),
)
def test_ground_loss_applied_when_unnormalised():
    beam = Beam.uniform(f_low=50, f_high=54, delta_f=2)
    sky = SkyModel.uniform_healpix(408.0, temperature=1000.0, nside=4)
    kw = {
        "beam": beam,
        "lsts": Longitude([2.0] * un.hour),
        "sky_model": sky,
        "index_model": ConstantIndex(),
        "normalize_beam": False,
        "use_astropy_azel": False,
        **FAST,
    }
    with_loss = simulate_spectra(ground_loss=np.array([0.5, 0.5]), **kw)
    without = simulate_spectra(ground_loss=None, **kw)
    np.testing.assert_allclose(with_loss.data, 0.5 * without.data, rtol=1e-10)


def test_generator_yields_independent_beam_arrays():
    """Collecting the generator output must not alias the per-frequency beam arrays."""
    beam = Beam.uniform(f_low=50, f_high=54, delta_f=2, delta_az=5, delta_el=5)
    beam = Beam(
        frequency=beam.frequency,
        azimuth=beam.azimuth,
        elevation=beam.elevation,
        beam=beam.beam * np.array([1.0, 2.0])[:, None, None],
    )
    sky = SkyModel.uniform_healpix(408.0, nside=4)
    outs = list(
        sky_convolution_generator(
            lsts=Longitude([1.0] * un.hour),
            beam=beam,
            sky_model=sky,
            index_model=ConstantIndex(),
            normalize_beam=False,
            beam_smoothing=False,
            smoothing_model=None,
            interp_kind="linear",
            lst_progress=False,
            freq_progress=False,
        )
    )
    bm0, bm1 = outs[0][5], outs[1][5]
    assert bm0 is not bm1
    np.testing.assert_allclose(np.nan_to_num(bm1), 2 * np.nan_to_num(bm0), rtol=1e-12)


@pytest.mark.parametrize("use_astropy_azel", [True, False])
def test_nan_sky_pixels_excluded_from_normalisation(use_astropy_azel, achromatic_beam):
    """Blank (NaN) sky pixels above the horizon must not bias the normalised result."""
    sky = SkyModel.uniform_healpix(frequency=408.0, temperature=1000.0, nside=8)
    temp = sky.temperature.copy()
    # Blank a cap around the south celestial pole, which is always above the horizon
    # at the EDGES site (like the blank pixels in the Guzman 45 MHz map).
    temp[sky.coords.icrs.dec.deg < -60] = np.nan
    blanked = SkyModel(
        frequency=408.0, temperature=temp, coords=sky.coords, healpix=sky.healpix
    )
    assert np.sum(np.isnan(temp)) > 10

    spectra = simulate_spectra(
        beam=achromatic_beam,
        lsts=Longitude([0.0, 12.0] * un.hour),
        sky_model=blanked,
        index_model=ConstantIndex(),
        use_astropy_azel=use_astropy_azel,
        **FAST,
    )
    expected = np.array([
        sky.at_freq(f, index_model=ConstantIndex())[0]
        for f in achromatic_beam.frequency.to_value("MHz")
    ])
    np.testing.assert_allclose(
        spectra.data[0, 0], np.broadcast_to(expected, (2, len(expected))), rtol=1e-10
    )
