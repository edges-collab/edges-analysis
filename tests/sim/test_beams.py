"""Test the beams module."""

from pathlib import Path

import numpy as np
import pytest
from astropy import units as u

import edges.sim
import edges.sim.antenna_beam_factor
from edges import const
from edges import modeling as mdl
from edges.sim import beams
from edges.sim.sky_models import ConstantIndex, Haslam408AllNoh, SkyModel, StepIndex


@pytest.fixture(scope="module")
def beam() -> beams.Beam:
    return beams.Beam.from_file("low")


def test_beam_from_feko(beam):
    assert beam.frequency.min() == 40.0 * u.MHz
    assert beam.frequency.max() == 100.0 * u.MHz

    assert beam.beam.max() > 0

    beam2 = beam.between_freqs(50 * u.MHz, 70 * u.MHz)
    assert beam2.frequency.min() >= 50 * u.MHz
    assert beam2.frequency.max() <= 70 * u.MHz


def test_beam_from_raw_feko(sim_data_path: Path):
    beam = beams.Beam.from_feko_raw(
        sim_data_path / "lowband_dielectric1-new-90orient_simple",
        ext="txt",
        f_low=40 * u.MHz,
        f_high=48 * u.MHz,
        freq_p=5,
        theta_p=181,
        phi_p=361,
    )

    assert beam.frequency.min() == 40.0 * u.MHz
    assert beam.frequency.max() == 48.0 * u.MHz

    assert beam.beam.max() > 0

    beam2 = beam.smoothed()
    assert len(beam2.frequency) == 5
    assert (beam2.frequency == beam.frequency).all()


def test_feko_interp(beam):
    beam2 = beam.at_freq(np.linspace(50, 60, 5) * u.MHz)
    assert (beam2.frequency == np.linspace(50, 60, 5) * u.MHz).all()

    indx_50 = list(beam.frequency).index(50.0 * u.MHz)
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


def test_antenna_beam_factor(beam):
    abf = edges.sim.compute_antenna_beam_factor(
        beam=beam,
        f_low=50 * u.MHz,
        f_high=56 * u.MHz,
        lsts=np.arange(0, 24, 6),
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
        "f_low": 40 * u.MHz,
        "f_high": 100 * u.MHz,
        "lsts": [12.0],
        "sky_model": Haslam408AllNoh(),
        "index_model": ConstantIndex(),
        "normalize_beam": False,
        "reference_frequency": 75 * u.MHz,
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
