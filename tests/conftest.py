import pytest
from pathlib import Path
from subprocess import run
from edges_analysis.analysis import (
    CalibratedData,
    RawData,
    CombinedData,
    CombinedBinnedData,
    DayAveragedData,
    BinnedData,
    ModelData,
    read_step,
)
import yaml
from typing import Tuple
import numpy as np
from edges_cal.modelling import LinLog
import datetime as dt
from edges_analysis.config import config
from click.testing import CliRunner
from edges_analysis import cli

runner = CliRunner()


def invoke(cmd, args, **kwargs):
    result = runner.invoke(cmd, args, **kwargs)
    print(result.output)
    if result.exit_code > 0:
        raise result.exc_info[1]

    return result


@pytest.fixture(scope="session")
def integration_test_data(tmp_path_factory) -> Path:
    tmp_path = tmp_path_factory.mktemp("integration-data")

    run(
        [
            "git",
            "clone",
            "https://github.com/edges-collab/edges-analysis-test-data",
            str(tmp_path / "edges-analysis-test-data"),
            "--depth",
            "1",
        ]
    )
    return tmp_path / "edges-analysis-test-data"


@pytest.fixture(scope="session", autouse=True)
def edges_config(tmp_path_factory):
    new_path = tmp_path_factory.mktemp("edges-levels")

    old_paths = config["paths"]
    new_paths = {**old_paths, **{"field_products": new_path}}

    with config.use(paths=new_paths) as cfg:
        yield cfg


@pytest.fixture(scope="session")
def settings() -> Path:
    return Path(__file__).parent / "settings"


@pytest.fixture(scope="session")
def beam_settings() -> Path:
    return Path(__file__).parent / "data"


@pytest.fixture(scope="session")
def raw_settings(integration_test_data: Path) -> Path:
    settings = {
        "band": "low",
        "thermlog_file": str(integration_test_data / "thermlog_low.txt"),
        "weather_file": str(integration_test_data / "weather.txt"),
        "f_low": 40.0,
    }

    out = integration_test_data / "raw.yaml"
    with open(out, "w") as fl:
        yaml.dump(settings, fl)

    return out


@pytest.fixture(scope="session")
def calibrate_settings(integration_test_data: Path) -> Path:
    settings = {
        "calfile": str(integration_test_data / "calfile_Rcv_2017_05.h5"),
        "s11_path": str(integration_test_data / "s11"),
        "balun_correction": True,
        "antenna_correction": False,
        "ground_correction": ":",
        "beam_file": str(integration_test_data / "feko_Haslam408_ref70.00.h5"),
    }

    out = integration_test_data / "calibrate.yaml"
    with open(out, "w") as fl:
        yaml.dump(settings, fl)

    return out


@pytest.fixture(scope="session")
def calibrate_settings2(integration_test_data: Path) -> Path:
    settings = {
        "calfile": str(integration_test_data / "calfile_Rcv_2017_05.h5"),
        "s11_path": str(
            integration_test_data / "S11_blade_low_band_2015_342_03_14.txt.csv"
        ),
        "balun_correction": True,
        "antenna_correction": False,
        "ground_correction": ":",
        "beam_file": "",
    }

    out = integration_test_data / "calibrate1.yaml"
    with open(out, "w") as fl:
        yaml.dump(settings, fl)

    return out


@pytest.fixture(scope="session")
def raw_step(
    integration_test_data: Path,
    raw_settings: Path,
    edges_config: dict,
) -> Tuple[RawData, RawData]:
    invoke(
        cli.process,
        [
            "raw",
            str(raw_settings),
            "-i",
            str(integration_test_data / "2016_*_00_small.acq"),
            "-l",
            "raw",
        ],
    )

    return tuple(
        read_step(fl)
        for fl in sorted((edges_config["paths"]["field_products"] / "raw").glob("*.h5"))
    )


@pytest.fixture(scope="session")
def cal_step(
    raw_step: Tuple[RawData, RawData],
    calibrate_settings: Path,
    edges_config: dict,
    settings: Path,
) -> Tuple[CalibratedData, CalibratedData]:
    invoke(
        cli.process,
        [
            "calibrate",
            str(calibrate_settings),
            "-i",
            str(raw_step[0].filename.parent / "*.h5"),
            "-l",
            "calibrated",
        ],
    )

    invoke(
        cli.filter,
        [
            str(settings / "xrfi.yml"),
            "-i",
            str(raw_step[0].filename.parent / "calibrated/*.h5"),
        ],
    )

    return tuple(
        read_step(fl)
        for fl in sorted((raw_step[0].filename.parent / "calibrated").glob("*.h5"))
    )


@pytest.fixture(scope="session")
def cal_step2(
    raw_step: Tuple[RawData, RawData],
    calibrate_settings2: Path,
    edges_config: dict,
    settings: Path,
) -> Tuple[CalibratedData, CalibratedData]:
    invoke(
        cli.process,
        [
            "calibrate",
            str(calibrate_settings2),
            "-i",
            str(raw_step[0].filename.parent / "*.h5"),
            "-l",
            "calibrated2",
        ],
    )

    return tuple(
        read_step(fl)
        for fl in sorted((raw_step[0].filename.parent / "calibrated2").glob("*.h5"))
    )


@pytest.fixture(scope="session")
def model_step(cal_step, settings: Path, integration_test_data: Path):
    with open(settings / "model.yml") as fl:
        s = yaml.load(fl, Loader=yaml.FullLoader)

    return [
        ModelData.promote(
            obj, filename=integration_test_data / f"model/{obj.day}.h5", **s
        )
        for obj in cal_step
    ]


@pytest.fixture(scope="session")
def combo_step(model_step, settings: Path, integration_test_data: Path):
    with open(settings / "combine.yml") as fl:
        s = yaml.load(fl, Loader=yaml.FullLoader)

    return CombinedData.promote(
        model_step, filename=integration_test_data / "combined.h5", **s
    )


@pytest.fixture(scope="session")
def combo_bin_step(combo_step, settings: Path, integration_test_data: Path):
    with open(settings / "combine_bin.yml") as fl:
        s = yaml.load(fl, Loader=yaml.FullLoader)

    return CombinedBinnedData.promote(
        combo_step, filename=integration_test_data / "combinedbinned.h5", **s
    )


@pytest.fixture(scope="session")
def combo_bin2_step(combo_bin_step, settings: Path, integration_test_data: Path):
    with open(settings / "combine_bin2.yml") as fl:
        s = yaml.load(fl, Loader=yaml.FullLoader)

    return CombinedBinnedData.promote(
        combo_bin_step, filename=integration_test_data / "combinedbinned2.h5", **s
    )


@pytest.fixture(scope="session")
def day_step(combo_step: CombinedData, settings: Path, integration_test_data: Path):
    with open(settings / "day_average.yml") as fl:
        s = yaml.load(fl, Loader=yaml.FullLoader)

    return DayAveragedData.promote(
        combo_step, filename=integration_test_data / "day_averaged.h5", **s
    )


@pytest.fixture(scope="session")
def gha_step(day_step: DayAveragedData, settings: Path, integration_test_data: Path):
    with open(settings / "gha_average.yml") as fl:
        s = yaml.load(fl, Loader=yaml.FullLoader)

    return BinnedData.promote(
        day_step, filename=integration_test_data / "gha_averaged.h5", **s
    )


@pytest.fixture(scope="session")
def mock_calibrated_data(tmp_path_factory) -> CalibratedData:
    np.random.seed(1234)
    tmp_path = tmp_path_factory.mktemp("mock-data")

    freq = np.linspace(50, 100, 100)
    n_gha = 50

    start_time = dt.datetime(year=2015, month=1, day=1)
    timedelta = dt.timedelta(hours=12) / n_gha

    time_strings = np.array(
        [(start_time + i * timedelta).strftime("%Y:%j:%H:%M:%S") for i in range(n_gha)],
        dtype="S17",
    )

    anc = CalibratedData.get_ancillary_coords(
        CalibratedData.get_datetimes(time_strings)
    )
    anc["times"] = time_strings
    anc["ambient_hum"] = np.zeros(len(time_strings))
    anc["receiver_temp"] = np.ones(len(time_strings)) * 25

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

    return CalibratedData.from_data(
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
            },
        },
        filename=tmp_path / "mock_calibrated_data_0.h5",
        validate=False,
    )
