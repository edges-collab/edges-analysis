from edges_analysis.analysis import beams, loss
from edges_analysis.analysis.sky_models import Haslam408
import numpy as np
from pathlib import Path


def test_beam_from_feko():
    beam = beams.Beam.from_file("low")

    assert beam.frequency.min() == 40.0
    assert beam.frequency.max() == 100.0

    assert beam.beam.max() > 0

    beam2 = beam.between_freqs(50, 70)
    assert beam2.frequency.min() >= 50
    assert beam2.frequency.max() <= 70


def test_beam_from_raw_feko(beam_settings: Path):
    beam = beams.Beam.from_feko_raw(
        beam_settings / "lowband_dielectric1-new-90orient_simple",
        "txt",
        40,
        48,
        5,
        181,
        361,
    )

    assert beam.frequency.min() == 40.0
    assert beam.frequency.max() == 48.0

    assert beam.beam.max() > 0

    beam2 = beam.smoothed()
    assert len(beam2.frequency) == 5
    assert beam2.frequency.all() == beam.frequency.all()


def test_feko_interp():
    beam = beams.Beam.from_file("low")

    beam2 = beam.at_freq(np.linspace(50, 60, 5))
    assert (beam2.frequency == np.linspace(50, 60, 5)).all()

    indx_50 = list(beam.frequency).index(50.0)
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


def test_simulate_spectra():
    beam = beams.Beam.from_file("low")

    # Do a really small simulation
    sky_map, freq, lst = beams.simulate_spectra(
        beam,
        f_low=50,
        f_high=55,
        lsts=np.arange(0, 24, 12),
        sky_model=Haslam408(max_res=3),
    )

    assert sky_map.shape == (len(lst), len(freq))
    assert np.all(sky_map >= 0)


def test_uniform_beam():
    beam = beams.Beam.from_ideal()
    assert np.allclose(beam.beam, 1)
    az, el = np.meshgrid(beam.azimuth, beam.elevation[:-1])
    assert np.allclose(beam.angular_interpolator(0)(az, el), 1)


def test_antenna_beam_factor():
    beam = beams.Beam.from_file("low")
    abf = beams.antenna_beam_factor(
        beam=beam,
        f_low=50,
        f_high=56,
        lsts=np.arange(0, 24, 12),
        sky_model=Haslam408(max_res=3),
    )
    assert isinstance(abf, beams.BeamFactor)


def test_ground_loss_from_beam(beam_settings: Path):
    beam = beams.Beam.from_feko_raw(
        beam_settings / "lowband_dielectric1-new-90orient_simple",
        "txt",
        40,
        48,
        5,
        181,
        361,
    )

    loss_fraction = 1 - loss.ground_loss(filename=None, beam=beam, freq=beam.frequency)
    assert loss_fraction.max() < 1
