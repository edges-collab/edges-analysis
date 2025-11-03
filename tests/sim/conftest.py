from pathlib import Path

import pytest


@pytest.fixture(scope="session")
def sim_data_path(testdata_path: Path) -> Path:
    """Path to simulation data."""
    return testdata_path / "sim"
