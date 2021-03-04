# -*- coding: utf-8 -*-
import pytest
from pathlib import Path
from subprocess import run
from edges_analysis.analysis import Level1, Level2, Level3, Level4
import yaml
from typing import Tuple
import numpy as np
from edges_cal.modelling import LinLog
import datetime as dt
from edges_io import __version__
from edges_analysis import __version__ as eav


@pytest.fixture(scope="session")
def integration_test_data(tmp_path_factory) -> Path:
    tmp_path = tmp_path_factory.mktemp("integration-data")

    run(
        [
            "git",
            "clone",
            "https://github.com/edges-collab/edges-analysis-test-data",
            str(tmp_path / "edges-analysis-test-data"),
        ]
    )
    return tmp_path / "edges-analysis-test-data"


@pytest.fixture(scope="session")
def settings() -> Path:
    return Path(__file__).parent / "settings"


@pytest.fixture(scope="session")
def level1_settings(integration_test_data: Path) -> Path:
    settings = {
        "band": "low",
        "f_low": 50,
        "f_high": 100,
        "calfile": str(integration_test_data / "cal_file_Rcv01_2015_09.h5"),
        "s11_path": str(integration_test_data / "s11"),
        "switch_state_dir": str(integration_test_data / "SwitchingState01"),
        "balun_correction": True,
        "antenna_correction": False,
        "ground_correction": ":",
        "beam_file": str(integration_test_data / "feko_Haslam408_ref70.00.h5"),
        "thermlog_file": str(integration_test_data / "thermlog_low.txt"),
        "weather_file": str(integration_test_data / "weather.txt"),
    }

    out = integration_test_data / "level1_settings.yaml"
    with open(out, "w") as fl:
        yaml.dump(settings, fl)

    return out


@pytest.fixture(scope="session")
def level1(integration_test_data: Path, level1_settings: Path) -> Tuple[Level1, Level1]:
    with open(level1_settings) as fl:
        settings = yaml.load(fl, Loader=yaml.FullLoader)

    settings["s11_path"] = str(integration_test_data / "s11")

    l1 = Level1.from_acq(
        integration_test_data / "2016_292_00_small.acq",
        out_file=integration_test_data / "level1/292.h5",
        **settings
    )
    l1.write()

    l2 = Level1.from_acq(
        integration_test_data / "2016_295_00_small.acq",
        out_file=integration_test_data / "level1/295.h5",
        **settings
    )
    l2.write()

    return l1, l2


@pytest.fixture(scope="session")
def level2(level1: Level1, settings: Path, integration_test_data: Path):
    with open(settings / "level2.yml") as fl:
        s = yaml.load(fl, Loader=yaml.FullLoader)

    return Level2.from_previous_level(level1, filename=integration_test_data / "level2/out.h5", **s)


@pytest.fixture(scope="session")
def level3(level2: Level2, settings: Path, integration_test_data: Path):
    with open(settings / "level3.yml") as fl:
        s = yaml.load(fl, Loader=yaml.FullLoader)

    return Level3.from_previous_level(level2, filename=integration_test_data / "level3/out.h5", **s)


@pytest.fixture(scope="session")
def level4(level3: Level3, settings: Path, integration_test_data: Path):
    with open(settings / "level4.yml") as fl:
        s = yaml.load(fl, Loader=yaml.FullLoader)

    return Level4.from_previous_level(level3, filename=integration_test_data / "level4/out.h5", **s)


@pytest.fixture(scope="session")
def mock_level1_list(tmp_path_factory) -> Level1:
    np.random.seed(1234)
    tmp_path = tmp_path_factory.mktemp("mock-data")

    freq = np.linspace(50, 100, 100)
    n_gha = 50

    start_time = dt.datetime(year=2015, month=1, day=1)
    timedelta = dt.timedelta(hours=12) / n_gha

    time_strings = np.array(
        [(start_time + i * timedelta).strftime("%Y:%j:%H:%M:%S") for i in range(n_gha)], dtype="S17"
    )

    anc = Level1.get_ancillary_coords(Level1.get_datetimes(time_strings))
    anc["times"] = time_strings
    gha_model = 10000 * (1 + np.sin(2 * np.pi * (anc["gha"] - 18) / 24))

    mdl = LinLog(default_x=freq, n_terms=2)

    sky = np.array([mdl(parameters=[gg, 0]) for gg in gha_model])
    noise = np.random.normal(0, scale=sky / 100)

    data = {
        "spectrum": sky + noise,
        "switch_powers": np.concatenate((sky, sky, sky)).reshape((3, n_gha, len(freq))),
        "weights": np.ones_like(sky),
        "Q": (sky + noise - 300) / 400,
    }

    return Level1.from_data(
        {
            "frequency": freq,
            "spectra": data,
            "ancillary": anc,
            "meta": {
                "year": 2015,
                "day": 1,
                "hour": 1,
                "band": "low",
                "xrfi_pipe": {},
                "write_time": dt.datetime.now(),
                "edges_io_version": __version__,
                "object_name": "Level1",
                "edges_analysis_version": eav,
                "message": "",
            },
        },
        filename=tmp_path / "mock_level1_0.h5",
    )
