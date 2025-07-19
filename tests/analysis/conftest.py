"""Pytest configuration for the edges-analysis package."""

from __future__ import annotations

from pathlib import Path
from subprocess import run

import matplotlib as mpl
import pytest
from click.testing import CliRunner

from edges.config import config

from . import DATA_PATH

runner = CliRunner()


def invoke(cmd, args, **kwargs):
    result = runner.invoke(cmd, args, **kwargs)
    print(result.output)
    if result.exit_code > 0:
        raise result.exc_info[1]

    return result


@pytest.fixture(scope="session", autouse=True)
def _set_mpl_backend():
    """Set the matplotlib backend for faster tests."""
    mpl.use("pdf")


@pytest.fixture(scope="session")
def integration_test_data() -> Path:
    tmp_path = Path(
        "/tmp/edges-analysis-pytest"
    )  # tmp_path_factory.mktemp("integration-data", numbered=False)
    repo = tmp_path / "edges-analysis-test-data"

    if repo.exists():
        run(["git", "-C", str(repo), "pull"])
    else:
        run([
            "git",
            "clone",
            "https://github.com/edges-collab/edges-analysis-test-data",
            str(repo),
            "--depth",
            "1",
        ])
    return repo


@pytest.fixture(scope="session")
def edges_config(tmp_path_factory):
    new_path = tmp_path_factory.mktemp("edges-levels")

    old_paths = config["paths"]
    new_paths = {**old_paths, "field_products": new_path}

    with config.use(paths=new_paths) as cfg:
        yield cfg


@pytest.fixture(scope="session")
def settings() -> Path:
    return Path(__file__).parent / "settings"


@pytest.fixture(scope="session")
def workflow_dir(tmp_path_factory) -> Path:
    return tmp_path_factory.mktemp("integration-workflow")


@pytest.fixture(scope="session")
def calpath(integration_test_data: Path) -> Path:
    return DATA_PATH / "specal.h5"  # "calfile_v0_hickled.h5"


@pytest.fixture(scope="session")
def s11path(integration_test_data: Path) -> Path:
    return DATA_PATH / "2015_ants11_modelled_redone.h5"


@pytest.fixture(scope="session")
def beamfile(integration_test_data: Path) -> Path:
    return integration_test_data / "alan_beam_factor.h5"
