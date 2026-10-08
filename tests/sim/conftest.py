from pathlib import Path

import pytest
from sim_helpers import make_achromatic_beam, make_galaxy_sky

from edges.sim.beams import Beam
from edges.sim.sky_models import SkyModel


@pytest.fixture(scope="session")
def sim_data_path(testdata_path: Path) -> Path:
    """Path to simulation data."""
    return testdata_path / "sim"


@pytest.fixture(scope="session")
def galaxy_sky() -> SkyModel:
    return make_galaxy_sky()


@pytest.fixture(scope="session")
def achromatic_beam() -> Beam:
    return make_achromatic_beam()
