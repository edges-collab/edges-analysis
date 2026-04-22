import numpy as np
from astropy import units as un
from astropy.coordinates import Longitude

from edges.sim import Beam, simulate_spectra
from edges.sim.sky_models import GaussianIndex, SkyModel


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
