from pathlib import Path

import pytest


@pytest.fixture(scope="session")
def sim_data_path() -> Path:
    """Path to simulation data."""
    return Path(__file__).parent / "data"


@pytest.fixture(scope="session")
def beam_settings() -> Path:
    return Path(__file__).parent / "data"
