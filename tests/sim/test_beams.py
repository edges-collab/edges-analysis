"""Test the beams module."""

from pathlib import Path

import numpy as np
import pytest
from astropy import constants
from astropy import units as un
from astropy.coordinates import Longitude

import edges.sim
import edges.sim.antenna_beam_factor
from edges import const
from edges import modeling as mdl
from edges.sim import beams
from edges.sim.sky_models import ConstantIndex, Haslam408AllNoh, SkyModel, StepIndex


@pytest.fixture(scope="module")
def beam() -> beams.Beam:
    return beams.Beam.from_file("low")


def test_bad_beam_shapes():
    with pytest.raises(ValueError, match="Beam must be a 3D array"):
        beams.Beam(
            frequency=np.array([50, 75, 100]) * un.MHz,
            azimuth=np.array([0, 90, 180]),
            elevation=np.array([0, 45, 90]),
            beam=np.zeros((3, 3)),
        )

    with pytest.raises(
        ValueError, match="First dimension of beam must match length of frequency"
    ):
        beams.Beam(
            frequency=np.array([50, 75, 90]) * un.MHz,
            azimuth=np.array([0, 90]),
            elevation=np.array([0, 45]),
            beam=np.zeros((2, 2, 2)),
        )

    with pytest.raises(
        ValueError, match="Second dimension of beam must match length of elevation"
    ):
        beams.Beam(
            frequency=np.array([50, 75]) * un.MHz,
            azimuth=np.array([0, 90]),
            elevation=np.array([0, 45, 90]),
            beam=np.zeros((2, 2, 2)),
        )

    with pytest.raises(
        ValueError, match="Third dimension of beam must match length of azimuth"
    ):
        beams.Beam(
            frequency=np.array([50, 75]) * un.MHz,
            azimuth=np.array([0, 45, 90]),
            elevation=np.array([0, 45]),
            beam=np.zeros((2, 2, 2)),
        )


def test_beam_from_feko(beam):
    assert beam.frequency.min() == 40.0 * un.MHz
    assert beam.frequency.max() == 100.0 * un.MHz

    assert beam.beam.max() > 0

    beam2 = beam.between_freqs(50 * un.MHz, 70 * un.MHz)
    assert beam2.frequency.min() >= 50 * un.MHz
    assert beam2.frequency.max() <= 70 * un.MHz


def test_beam_from_raw_feko(sim_data_path: Path):
    beam = beams.Beam.from_feko_raw(
        sim_data_path / "lowband_dielectric1-new-90orient_simple",
        ext="txt",
        f_low=40 * un.MHz,
        f_high=48 * un.MHz,
        freq_p=5,
        theta_p=181,
        phi_p=361,
    )

    assert beam.frequency.min() == 40.0 * un.MHz
    assert beam.frequency.max() == 48.0 * un.MHz

    assert beam.beam.max() > 0

    beam2 = beam.smoothed()
    assert len(beam2.frequency) == 5
    assert (beam2.frequency == beam.frequency).all()


def test_feko_interp(beam):
    beam2 = beam.at_freq(np.linspace(50, 60, 5) * un.MHz)
    assert (beam2.frequency == np.linspace(50, 60, 5) * un.MHz).all()

    indx_50 = list(beam.frequency).index(50.0 * un.MHz)
    az, el = np.meshgrid(beam.azimuth, beam.elevation[:-1])
    interp = beam2.angular_interpolator(0)(az.flatten(), el.flatten())

    print("Max Error:", np.abs(interp - beam.beam[indx_50, :-1].flatten()).max())
    print(
        "Max Rel. Error:",
        np.abs(
            (interp - beam.beam[indx_50, :-1].flatten())
            / beam.beam[indx_50, :-1].flatten()
        ).max(),
    )

    assert np.allclose(interp, beam.beam[indx_50, :-1].flatten(), rtol=1e-2, atol=1e-5)
    interp_zenith = beam2.angular_interpolator(0)(0, 90)
    assert np.isclose(interp_zenith, beam.beam[indx_50, -1, 0], rtol=1e-2, atol=0)
    print(
        "Error (abs/frac) at zenith: ",
        interp_zenith - beam.beam[indx_50, -1, 0],
        (interp_zenith - beam.beam[indx_50, -1, 0]) / beam.beam[indx_50, -1, 0],
    )


def test_uniform_beam():
    beam = beams.Beam.uniform()
    assert np.allclose(beam.beam, 1)
    az, el = np.meshgrid(beam.azimuth, beam.elevation[:-1])
    assert np.allclose(beam.angular_interpolator(0)(az, el), 1)


def _off_axis_azimuth_averaged_response(
    beam: beams.Beam, freq_indx: int, el: int
) -> float:
    el_indx = np.where(beam.elevation == el)[0][0]
    beam_slice = np.asarray(beam.beam[freq_indx])

    return float(np.nanmean(beam_slice[el_indx]))


def test_gaussian_beam_wider_at_lower_frequency():
    beam = beams.Beam.gaussian(
        dish_size=3.0,
        f_low=50,
        f_high=150,
        delta_f=50,
        delta_el=1,
        delta_az=5,
    )

    low_freq_response = _off_axis_azimuth_averaged_response(beam, freq_indx=0, el=80)
    high_freq_response = _off_axis_azimuth_averaged_response(beam, freq_indx=-1, el=80)

    assert low_freq_response > high_freq_response


def test_airy_beam_wider_at_lower_frequency():
    beam = beams.Beam.airy(
        dish_size=3.0,
        f_low=50,
        f_high=150,
        delta_f=50,
        delta_el=1,
        delta_az=5,
    )

    low_freq_response = _off_axis_azimuth_averaged_response(beam, freq_indx=0, el=85)
    high_freq_response = _off_axis_azimuth_averaged_response(beam, freq_indx=-1, el=85)

    assert low_freq_response > high_freq_response


@pytest.mark.parametrize("beam_constructor", ["gaussian", "airy"])
def test_gaussian_and_airy_beams_no_nans(beam_constructor):
    """Verify that both gaussian and airy constructors produce no NaN values."""
    beam = getattr(beams.Beam, beam_constructor)(
        dish_size=3.0,
        f_low=50,
        f_high=150,
    )

    assert not np.any(np.isnan(beam.beam)), "beam contains NaN values"


@pytest.mark.parametrize("beam_constructor", ["gaussian", "airy", "uniform"])
def test_gaussian_beam_rotationally_symmetric(beam_constructor):
    """Test that Gaussian beams are rotationally symmetric.

    For a rotationally symmetric antenna, the beam response at a given elevation
    should be constant across all azimuths.
    """
    kw = {
        "f_low": 75,
        "f_high": 125,
        "delta_f": 25,
        "delta_el": 10,
        "delta_az": 1,
    }
    if beam_constructor in ("gaussian", "airy"):
        beam = getattr(beams.Beam, beam_constructor)(dish_size=3.0, **kw)
    else:
        beam = beams.Beam.uniform(**kw)

    # Check at multiple frequencies and elevations
    for freq_indx in [0, 1, -1]:
        for el in [0, 30, 60, 80]:
            if el not in beam.elevation:
                continue
            el_indx = np.where(beam.elevation == el)[0][0]
            beam_row = beam.beam[freq_indx][el_indx, :]

            # All azimuths at this elevation should have the same beam value
            assert np.allclose(beam_row, beam_row[0], rtol=1e-10), (
                "Gaussian beam not rotationally symmetric at "
                f"freq_indx={freq_indx}, el={el}"
            )


def test_antenna_beam_factor(beam):
    abf = edges.sim.compute_antenna_beam_factor(
        beam=beam,
        f_low=50 * un.MHz,
        f_high=56 * un.MHz,
        lsts=Longitude(np.arange(0, 24, 6) * un.hour),
        sky_model=SkyModel.uniform_healpix(frequency=75.0, nside=4),
        index_model=StepIndex(),
        use_astropy_azel=False,
        beam_smoothing=False,
    )
    assert isinstance(abf, edges.sim.antenna_beam_factor.BeamFactor)


def test_interp_methods(beam):
    sphere = beam.angular_interpolator(0, "sphere-spline")
    cubic = beam.angular_interpolator(0, "linear")

    assert np.allclose(
        sphere(beam.azimuth[:10], beam.elevation[:10]),
        cubic(beam.azimuth[:10], beam.elevation[:10]),
    )

    rspline = beam.angular_interpolator(0, "spline")
    assert np.allclose(
        sphere(beam.azimuth[:10], beam.elevation[:10]),
        rspline(beam.azimuth[:10], beam.elevation[:10]),
    )


def test_beam_solid_angle():
    beam = beams.Beam.uniform(f_low=50, f_high=60, delta_el=0.1, delta_az=0.1)
    np.testing.assert_allclose(
        beam.get_beam_solid_angle(), 2 * np.pi, rtol=1e-3
    )  # half the sky


def test_bad_beamfactor_args():
    with pytest.raises(ValueError, match="Frequencies must be monotonically"):
        edges.sim.antenna_beam_factor.BeamFactor(
            frequencies=np.array([1, 10, 2, 7]),
            lsts=np.array([0, 1, 2]),
            reference_frequency=75.0,
            antenna_temp=np.zeros((3, 4)),
            antenna_temp_ref=np.zeros((3, 4)),
        )

    with pytest.raises(ValueError, match="LSTs must be monotonically increasing"):
        edges.sim.antenna_beam_factor.BeamFactor(
            frequencies=np.linspace(50, 100, 10),
            lsts=np.array([0, np.pi, 2.0, 2 * np.pi, 0.1]),
            reference_frequency=75.0,
            antenna_temp=np.zeros((5, 10)),
            antenna_temp_ref=np.zeros((5, 10)),
        )

    with pytest.raises(ValueError, match="Reference frequency must be positive"):
        edges.sim.antenna_beam_factor.BeamFactor(
            frequencies=np.linspace(50, 100, 10),
            lsts=np.array([0, 1, 2]),
            reference_frequency=-75.0,
            antenna_temp=np.zeros((3, 10)),
            antenna_temp_ref=np.zeros((3, 10)),
        )

    with pytest.raises(ValueError, match="antenna_temp must be a 2D array"):
        edges.sim.antenna_beam_factor.BeamFactor(
            frequencies=np.linspace(50, 100, 10),
            lsts=np.array([0, 1, 2]),
            reference_frequency=75.0,
            antenna_temp=np.zeros((3,)),
            antenna_temp_ref=np.zeros((3, 10)),
        )

    with pytest.raises(ValueError, match="antenna_temp must have shape"):
        edges.sim.antenna_beam_factor.BeamFactor(
            frequencies=np.linspace(50, 100, 10),
            lsts=np.array([0, 1, 2]),
            reference_frequency=75.0,
            antenna_temp=np.zeros((3, 11)),
            antenna_temp_ref=np.zeros((3, 10)),
        )

    with pytest.raises(ValueError, match="Antenna temperature must be positive"):
        edges.sim.antenna_beam_factor.BeamFactor(
            frequencies=np.linspace(50, 100, 10),
            lsts=np.array([0, 1, 2]),
            reference_frequency=75.0,
            antenna_temp=-np.ones((3, 10)),
            antenna_temp_ref=np.zeros((3, 10)),
        )

    with pytest.raises(ValueError, match="Reference antenna temperature must be a 1D"):
        edges.sim.antenna_beam_factor.BeamFactor(
            frequencies=np.linspace(50, 100, 10),
            lsts=np.array([0, 1, 2]),
            reference_frequency=75.0,
            antenna_temp=np.zeros((3, 10)),
            antenna_temp_ref=np.zeros((3, 10, 7)),
        )

    with pytest.raises(ValueError, match="If Reference antenna temperature is 1D"):
        edges.sim.antenna_beam_factor.BeamFactor(
            frequencies=np.linspace(50, 100, 10),
            lsts=np.array([0, 1, 2]),
            reference_frequency=75.0,
            antenna_temp=np.zeros((3, 10)),
            antenna_temp_ref=np.zeros((10,)),
        )

    with pytest.raises(ValueError, match="If Reference antenna temperature is 2D"):
        edges.sim.antenna_beam_factor.BeamFactor(
            frequencies=np.linspace(50, 100, 10),
            lsts=np.array([0, 1, 2]),
            reference_frequency=75.0,
            antenna_temp=np.zeros((3, 10)),
            antenna_temp_ref=np.zeros((10, 3)),
        )

    with pytest.raises(ValueError, match="Reference antenna temperature must be pos"):
        edges.sim.antenna_beam_factor.BeamFactor(
            frequencies=np.linspace(50, 100, 10),
            lsts=np.array([0, 1, 2]),
            reference_frequency=75.0,
            antenna_temp=np.zeros((3, 10)),
            antenna_temp_ref=-np.ones((3, 10)),
        )


def test_beamfactor_at_lsts_null():
    bf = edges.sim.antenna_beam_factor.BeamFactor(
        frequencies=np.linspace(50, 100, 10),
        lsts=np.linspace(3, 26, 24),
        reference_frequency=75.0,
        antenna_temp=np.ones((24, 10)),
        antenna_temp_ref=np.ones((24, 10)),
    )
    new = bf.at_lsts(np.linspace(10.5, 33.5, 24))
    assert np.allclose(new.antenna_temp, 1.0)


def test_beamfactor_between_lsts():
    bf = edges.sim.antenna_beam_factor.BeamFactor(
        frequencies=np.linspace(50, 100, 10),
        lsts=np.linspace(3, 26, 24),
        reference_frequency=75.0,
        antenna_temp=np.ones((24, 10)),
        antenna_temp_ref=np.ones((24, 10)),
    )
    new = bf.between_lsts(23, 3)
    assert len(new.lsts) == 4


def test_beamfactor_between_lsts_nodata():
    bf = edges.sim.antenna_beam_factor.BeamFactor(
        frequencies=np.linspace(50, 100, 10),
        lsts=np.linspace(3, 7, 24),
        reference_frequency=75.0,
        antenna_temp=np.ones((24, 10)),
        antenna_temp_ref=np.ones((24, 10)),
    )
    with pytest.raises(ValueError, match="BeamFactor does not contain any LSTs"):
        bf.between_lsts(8, 10)


def test_beamfactor_get():
    freq = np.linspace(50, 100, 51)
    poly = mdl.Polynomial(n_terms=4).at(x=freq)

    bf = edges.sim.antenna_beam_factor.BeamFactor(
        frequencies=freq,
        lsts=np.linspace(3, 7, 5),
        reference_frequency=75.0,
        antenna_temp=np.array([poly(parameters=[i * 100, 2, 3, 4]) for i in range(5)]),
        antenna_temp_ref=np.array([
            poly(parameters=[i * 100, 2.5, 3, 4]) for i in range(5)
        ]),
    )

    bf_ = bf.get_beam_factor(poly.model, freq)

    assert np.allclose(bf_[:, 25], 1.0)

    meanbf = bf.get_mean_beam_factor(poly.model, freq)
    assert np.isclose(meanbf[25], 1.0)

    intgbf = bf.get_integrated_beam_factor(poly.model)
    intgbf2 = bf.get_integrated_beam_factor(poly.model, freq)
    assert np.allclose(intgbf, intgbf2)


def test_beam_factor_alan_azel(beam):
    defaults = {
        "beam": beam,
        "f_low": 40 * un.MHz,
        "f_high": 100 * un.MHz,
        "lsts": Longitude([12.0 * un.hour]),
        "sky_model": Haslam408AllNoh(),
        "index_model": ConstantIndex(),
        "normalize_beam": False,
        "reference_frequency": 75 * un.MHz,
        "beam_smoothing": False,
        "interp_kind": "nearest",
        "freq_progress": False,
        "location": const.KNOWN_TELESCOPES["edges-low-alan"].location,
        "sky_at_reference_frequency": False,
        "use_astropy_azel": True,
    }

    default = edges.sim.compute_antenna_beam_factor(**defaults)
    alanazel = edges.sim.compute_antenna_beam_factor(**{
        **defaults,
        "use_astropy_azel": False,
    })
    poly = mdl.Polynomial(n_terms=10)
    assert np.isclose(
        default.get_mean_beam_factor(poly, np.array([55.0])),
        alanazel.get_mean_beam_factor(poly, np.array([55.0])),
        atol=1e-2,
    )


@pytest.fixture(scope="module")
def random_beam_maps() -> np.ndarray:
    rng = np.random.default_rng(42)
    return rng.uniform(size=(2, 91, 360))


def test_shift_beam_maps_full_rotation_is_identity(random_beam_maps):
    np.testing.assert_array_equal(
        beams.Beam.shift_beam_maps(360, random_beam_maps), random_beam_maps
    )
    np.testing.assert_array_equal(
        beams.Beam.shift_beam_maps(0, random_beam_maps), random_beam_maps
    )


@pytest.mark.parametrize("angle", [1, 37, 90, 271])
def test_shift_beam_maps_inverse_and_roll(angle, random_beam_maps):
    shifted = beams.Beam.shift_beam_maps(angle, random_beam_maps)
    np.testing.assert_array_equal(
        beams.Beam.shift_beam_maps(-angle, shifted), random_beam_maps
    )
    np.testing.assert_array_equal(shifted, np.roll(random_beam_maps, angle, axis=2))


def test_gaussian_beam_value_at_known_zenith_angle():
    """The Gaussian beam has FWHM = 1.22 wavelength / dish_size."""
    dish_size = 3.0
    beam = beams.Beam.gaussian(dish_size=dish_size, f_low=50, f_high=52, delta_f=2)
    fwhm = 1.22 * (constants.c.to_value("m/s") / 50e6) / dish_size  # radians
    sigma = fwhm / (2 * np.sqrt(2 * np.log(2)))
    za = 20.0
    el_indx = np.where(beam.elevation == 90 - za)[0][0]
    expected = np.exp(-0.5 * (np.radians(za) / sigma) ** 2)
    np.testing.assert_allclose(beam.beam[0, el_indx], expected, rtol=1e-10)


def test_gaussian_beam_half_power_at_half_fwhm():
    """B(za = FWHM / 2) = 0.5, for a dish sized to give a FWHM of 20 degrees."""
    fwhm = np.radians(20.0)
    dish_size = 1.22 * (constants.c.to_value("m/s") / 50e6) / fwhm
    beam = beams.Beam.gaussian(dish_size=dish_size, f_low=50, f_high=52, delta_f=2)
    np.testing.assert_allclose(beam.beam[0, beam.elevation == 90], 1.0, rtol=1e-12)
    np.testing.assert_allclose(beam.beam[0, beam.elevation == 80], 0.5, rtol=1e-10)


def test_sphere_spline_zenith_value():
    az = np.arange(0, 360, 2.0)
    el = np.arange(0, 91, 1.0)
    pattern = np.repeat((el / 90)[:, None], len(az), axis=1)  # 1 at zenith
    beam = beams.Beam(
        frequency=np.array([50.0]) * un.MHz,
        azimuth=az,
        elevation=el,
        beam=pattern[None],
    )
    interp = beam.angular_interpolator(0, "sphere-spline")
    np.testing.assert_allclose(
        interp(np.array([0.0, 123.0]), np.array([90.0, 90.0])),
        beam.beam[0, -1, 0],
        rtol=1e-4,
    )


def test_sphere_spline_without_zenith_row():
    """Without a zenith row, the pole is not pinned to the highest row's values."""
    az = np.arange(0, 360, 2.0)
    el = np.arange(0, 81, 2.0)
    pattern = np.repeat((1 + el / 90)[:, None], len(az), axis=1)
    beam = beams.Beam(
        frequency=np.array([50.0]) * un.MHz,
        azimuth=az,
        elevation=el,
        beam=pattern[None],
    )
    interp = beam.angular_interpolator(0, "sphere-spline")
    azz, ell = np.meshgrid(az, el)
    np.testing.assert_allclose(
        interp(azz.ravel(), ell.ravel()), pattern.ravel(), rtol=1e-6
    )
    # Between the last row and the zenith, the (linear) trend continues, rather than
    # flattening to the last row's value at the zenith.
    np.testing.assert_allclose(
        interp(np.array([0.0, 90.0]), np.array([85.0, 85.0])), 1 + 85 / 90, rtol=1e-4
    )


@pytest.mark.xfail(
    strict=True,
    reason=(
        "get_beam_solid_angle uses a rectangle rule that includes both "
        "elevation endpoints, biasing the solid angle high by ~d_el/2 (0.9% at 1 deg, "
        "4.3% at 5 deg); fix pending (result-changing)"
    ),
)
@pytest.mark.parametrize("delta", [1, 5])
def test_isotropic_beam_solid_angle_coarse_grid(delta):
    beam = beams.Beam.uniform(f_low=50, f_high=54, delta_el=delta, delta_az=delta)
    np.testing.assert_allclose(beam.get_beam_solid_angle(), 2 * np.pi, rtol=2e-3)


def test_shift_beam_maps_float_angle(random_beam_maps):
    np.testing.assert_array_equal(
        beams.Beam.shift_beam_maps(90.0, random_beam_maps),
        np.roll(random_beam_maps, 90, axis=2),
    )


def test_shift_beam_maps_non_unit_grid():
    maps = np.arange(72.0)[None, None, :] * np.ones((2, 3, 1))  # 5 deg az grid
    az = np.arange(0, 360, 5.0)
    np.testing.assert_array_equal(
        beams.Beam.shift_beam_maps(90, maps, azimuth=az), np.roll(maps, 18, axis=2)
    )
    with pytest.raises(ValueError, match="integer multiple"):
        beams.Beam.shift_beam_maps(92.5, maps, azimuth=az)
    with pytest.raises(ValueError, match="full circle"):
        beams.Beam.shift_beam_maps(90, maps[..., :-1], azimuth=az[:-1])


def test_get_beam_path():
    default = beams.Beam.get_beam_path("low")
    assert default.name == "default.txt"
    assert default.parent.name == "low"
    assert default.exists()

    other = beams.Beam.get_beam_path("low", kind="45deg")
    assert other.name == "45deg.txt"
    assert other.parent.name == "low"

    with pytest.raises(FileNotFoundError, match="No beam exists"):
        beams.Beam.get_beam_path("low", kind="nonexistent")
    with pytest.raises(FileNotFoundError, match="No beam exists"):
        beams.Beam.get_beam_path("nonexistent-band")


def _pattern(az_deg: np.ndarray, el_deg: np.ndarray) -> np.ndarray:
    """Dipole-like pattern, zero below the horizon."""
    return np.clip(np.sin(np.radians(el_deg)), 0, None) ** 2 * (
        1 + 0.3 * np.cos(2 * np.radians(az_deg))
    )


@pytest.mark.parametrize("linear", [True, False])
def test_beam_from_hfss(tmp_path, linear):
    phi = np.arange(-180, 180, 1.0)
    theta = np.arange(0, 181, 1.0)
    pp, tt = np.meshgrid(phi, theta)
    power = _pattern(pp, 90 - tt) + 1e-3
    if not linear:
        power = 10 * np.log10(power)

    fname = tmp_path / "beam.csv"
    np.savetxt(
        fname,
        np.c_[pp.ravel(), tt.ravel(), power.ravel()],
        delimiter=",",
        header="phi,theta,power",
    )
    beam = beams.Beam.from_hfss(fname, frequency=75 * un.MHz, linear=linear)

    assert beam.frequency.shape == (1,)
    assert beam.frequency[0] == 75 * un.MHz
    np.testing.assert_array_equal(beam.elevation, np.arange(0, 91))
    np.testing.assert_array_equal(beam.azimuth, np.arange(0, 360))
    az, el = np.meshgrid(beam.azimuth, beam.elevation)
    np.testing.assert_allclose(beam.beam[0], _pattern(az, el) + 1e-3, rtol=1e-6)

    if linear:  # from_file always reads HFSS files as linear
        beam2 = beams.Beam.from_file(
            None, simulator="hfss", beam_file=fname, frequency=75 * un.MHz
        )
        np.testing.assert_allclose(beam2.beam, beam.beam, rtol=1e-12)


def test_beam_from_file_hfss_needs_frequency(tmp_path):
    with pytest.raises(ValueError, match="frequency must be given"):
        beams.Beam.from_file(None, simulator="hfss", beam_file=tmp_path / "x.csv")


@pytest.mark.parametrize("az_antenna_axis", [0, 90])
def test_beam_from_wipld(tmp_path, az_antenna_axis):
    phi = np.arange(0, 360, 10.0)
    el = np.arange(-90, 91, 10.0)
    freqs = [50.0, 60.0]

    fname = tmp_path / "beam.wipld"
    with fname.open("w") as fl:
        for i, fr in enumerate(freqs):
            fl.write(f"  > Frequency =    {fr:12.6f}\n")
            for e in el:
                for p in phi:
                    gain = (1 + i) * _pattern(p, e)
                    fl.write(f"{p} {e} 0 0 0 0 {gain}\n")

    beam = beams.Beam.from_wipld(fname, az_antenna_axis=az_antenna_axis)
    np.testing.assert_allclose(beam.frequency.to_value("MHz"), freqs)
    np.testing.assert_array_equal(beam.elevation, np.arange(0, 91, 10))
    np.testing.assert_array_equal(beam.azimuth, phi)
    azz, ell = np.meshgrid(phi, beam.elevation)
    expected = np.array([(1 + i) * _pattern(azz, ell) for i in range(2)])
    np.testing.assert_allclose(
        beam.beam,
        np.roll(expected, az_antenna_axis // 10, axis=2),
        rtol=1e-6,
        atol=1e-12,
    )

    beam2 = beams.Beam.from_file(
        None,
        simulator="wipl-d",
        beam_file=fname,
        rotation_from_north=az_antenna_axis,
    )
    np.testing.assert_allclose(beam2.beam, beam.beam)
