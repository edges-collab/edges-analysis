from edges_analysis.analysis import beams
from edges_analysis.analysis import DATA
import numpy as np


def test_beam_from_feko():
    beam = beams.Beam.from_file("low")

    assert beam.frequency.min() == 40.0
    assert beam.frequency.max() == 100.0

    assert beam.beam.max() > 0

    beam2 = beam.between_freqs(50, 70)
    assert beam2.frequency.min() >= 50
    assert beam2.frequency.max() <= 70


def test_feko_interp():
    beam = beams.Beam.from_file("low")

    beam2 = beam.at_freq(np.linspace(50, 60, 5))
    assert (beam2.frequency == np.linspace(50, 60, 5)).all()

    indx_50 = list(beam.frequency).index(50.0)
    assert np.isclose(
        beam2.angular_interpolator(0)(beam.azimuth[0], beam.elevation[0]), beam.beam[indx_50, 0, 0]
    )


def test_simulate_spectra():
    beam = beams.Beam.from_file("low")

    # Do a really small simulation
    map, freq, lst = beams.simulate_spectra(beam, f_low=50, f_high=55, twenty_min_per_lst=12)

    assert map.shape == (len(lst), len(freq))
    assert np.all(map >= 0)
