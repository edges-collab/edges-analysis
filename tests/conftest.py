import pytest
from pathlib import Path
from subprocess import run
from edges_analysis.analysis import (
    CalibratedData,
    CombinedData,
    DayAveragedData,
    BinnedData,
    ModelData,
    filters,
)
import yaml
from typing import Tuple
import numpy as np
from edges_cal.modelling import LinLog
import datetime as dt
from edges_analysis.config import config


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


@pytest.fixture(scope="function")
def tmpconfig(tmp_path_factory):
    new_path = tmp_path_factory.mktemp("edges-levels")
    config["paths"]["field_products"] = str(new_path)
    return config


@pytest.fixture(scope="session")
def settings() -> Path:
    return Path(__file__).parent / "settings"


@pytest.fixture(scope="session")
def calibrate_settings(integration_test_data: Path) -> Path:
    settings = {
        "band": "low",
        "calfile": str(integration_test_data / "calfile_Rcv_2017_05.h5"),
        "s11_path": str(integration_test_data / "s11"),
        "balun_correction": True,
        "antenna_correction": False,
        "ground_correction": ":",
        "beam_file": str(integration_test_data / "feko_Haslam408_ref70.00.h5"),
        "thermlog_file": str(integration_test_data / "thermlog_low.txt"),
        "weather_file": str(integration_test_data / "weather.txt"),
    }

    out = integration_test_data / "calibrate.yaml"
    with open(out, "w") as fl:
        yaml.dump(settings, fl)

    return out


@pytest.fixture(scope="session")
def cal_step(
    integration_test_data: Path, calibrate_settings, settings
) -> Tuple[CalibratedData, CalibratedData]:
    with open(calibrate_settings) as fl:
        cal_settings = yaml.load(fl, Loader=yaml.FullLoader)

    with open(settings / "xrfi.yml") as fl:
        xrfi_pipe = yaml.load(fl, Loader=yaml.FullLoader)

    cals = []
    for day in ("292", "295"):
        cal = CalibratedData.promote(
            integration_test_data / f"2016_{day}_00_small.acq",
            filename=integration_test_data / f"calibrate/{day}.h5",
            **cal_settings,
        )
        filters.rfi_model_filter(data=[cal], in_place=True, **xrfi_pipe["xrfi_model"])
        cals.append(cal)

    return cals


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
