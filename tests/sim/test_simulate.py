import numpy as np
from astropy import units as un

from edges.sim import Beam, simulate_spectra
from edges.sim.sky_models import GaussianIndex, SkyModel


def test_simulate_spectra():
    beam = Beam.from_file("low")

    # Do a really small simulation
    sky_map, freq, lst = simulate_spectra(
        beam=beam,
        f_low=50 * un.MHz,
        f_high=55 * un.MHz,
        lsts=np.arange(0, 24, 12),
        sky_model=SkyModel.uniform_healpix(408, nside=8),
        use_astropy_azel=False,
        index_model=GaussianIndex(),
    )

    assert sky_map.shape == (len(lst), len(freq))
    assert np.all(sky_map >= 0)
