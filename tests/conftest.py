# -*- coding: utf-8 -*-
import pytest
from pathlib import Path
from subprocess import run
from edges_analysis.analysis import Level1, Level2, Level3, Level4
import yaml
from typing import Tuple


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
