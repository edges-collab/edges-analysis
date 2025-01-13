"""Pytest configuration for the edges-analysis package."""

from __future__ import annotations

from pathlib import Path
from subprocess import run

import matplotlib as mpl
import numpy as np
import pytest
from astropy import units as un
from astropy.time import Time
from click.testing import CliRunner
from edges_cal import modelling as mdl
from pygsdata import GSData

from edges_analysis import const
from edges_analysis.averaging import lstbin
from edges_analysis.calibration.calibrate import dicke_calibration
from edges_analysis.config import config
from edges_analysis.datamodel import add_model

from . import DATA_PATH
from .mock_gsdata import create_mock_edges_data

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
def beam_settings() -> Path:
    return Path(__file__).parent / "data"


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


@pytest.fixture(scope="session")
def gsd_ones():
    nload, npol, ntime, nfreq = 1, 2, 10, 26
    return GSData(
        data=np.ones((nload, npol, ntime, nfreq)),
        freqs=np.linspace(50, 100, nfreq) * un.MHz,
        times=Time(
            np.linspace(2459856, 2459857, ntime + 1)[:-1, None],
            format="jd",
            scale="utc",
        ),
        telescope=const.KNOWN_TELESCOPES["edges-low"],
        loads=("ant",),
    )


@pytest.fixture(scope="session")
def gsd_ones_power():
    nload, npol, ntime, nfreq = 3, 2, 10, 26
    times = np.linspace(2459856, 2459857, ntime + 1)[:-1]
    times = np.array([times, times, times]).T

    return GSData(
        data=np.ones((nload, npol, ntime, nfreq)),
        freqs=np.linspace(50, 100, nfreq) * un.MHz,
        times=Time(times, format="jd", scale="utc"),
        telescope=const.KNOWN_TELESCOPES["edges-low"],
        loads=("p0", "p1", "p2"),
        data_unit="power",
    )


@pytest.fixture(scope="session")
def mock() -> GSData:
    return create_mock_edges_data(add_noise=True)


@pytest.fixture(scope="session")
def mock_power() -> GSData:
    return create_mock_edges_data(add_noise=True, as_power=True)


@pytest.fixture(scope="session")
def mock_with_model(mock) -> GSData:
    return add_model(data=mock, model=mdl.LinLog(n_terms=2))


@pytest.fixture(scope="session")
def mock_lstbinned(mock: GSData) -> GSData:
    return lstbin.lst_bin(
        mock,
        binsize=0.02,
        first_edge=mock.lsts.min().hour,
        max_edge=mock.lsts.max().hour,
    )


@pytest.fixture(scope="session")
def mock_season() -> list[GSData]:
    """A mock 'season' with three days."""
    return [
        create_mock_edges_data(add_noise=True, as_power=True, time0=2459900.27),
        create_mock_edges_data(add_noise=True, as_power=True, time0=2459901.27),
        create_mock_edges_data(add_noise=True, as_power=True, time0=2459902.27),
    ]


@pytest.fixture(scope="session")
def mock_season_dicke(mock_season: list[GSData]) -> list[GSData]:
    """Dicke-calibrated mock season."""
    return [dicke_calibration(m) for m in mock_season]


@pytest.fixture(scope="session")
def mock_season_modelled(mock_season_dicke: list[GSData]) -> list[GSData]:
    """Dicke-calibrated mock season."""
    return [add_model(m, model=mdl.LinLog(n_terms=2)) for m in mock_season_dicke]
