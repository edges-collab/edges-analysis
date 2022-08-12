import datetime as dt
from pathlib import Path
from subprocess import run
from typing import Tuple

import numpy as np
import pytest
import yaml
from click.testing import CliRunner
from edges_cal.modelling import LinLog
from jinja2 import Template

from edges_analysis import cli
from edges_analysis.config import config
from edges_analysis.gsdata import GSData

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


def get_workflow(
    name: str, settings, tmp_path_factory, integration_test_data, beam_file, s11_path
):
    with open(settings / "integration_workflow.yaml") as fl:
        workflow = Template(fl.read())

    tmp_path = tmp_path_factory.mktemp("integration-workflow")

    txt = workflow.render(
        weather_file=str(integration_test_data / "weather.txt"),
        thermlog_file=str(integration_test_data / "thermlog_low.txt"),
        calfile=str(integration_test_data / "calfile_2015.h5"),
        s11_path=s11_path,
        beam_file=beam_file,
    )

    wf = yaml.load(txt, Loader=yaml.FullLoader)

    if not beam_file:
        wf["steps"] = tuple(
            x for x in wf["steps"] if x["function"] != "apply_beam_correction"
        )

    txt = yaml.dump(wf)

    with open(tmp_path / f"workflow_{name}.yaml", "w") as fl:
        fl.write(txt)

    return tmp_path / "cal_workflow_nobeam.yaml"


@pytest.fixture(scope="session")
def workflow(integration_test_data: Path, settings: Path, tmp_path_factory) -> Path:
    return get_workflow(
        "main",
        settings,
        tmp_path_factory,
        integration_test_data,
        str(integration_test_data / "feko_Haslam408_ref70.00.h5"),
        str(integration_test_data / "s11"),
    )


@pytest.fixture(scope="session")
def cal_workflow_nobeam(
    integration_test_data: Path, settings: Path, tmp_path_factory
) -> Path:
    return get_workflow(
        "nobeam",
        settings,
        tmp_path_factory,
        integration_test_data,
        "",
        str(integration_test_data / "S11_blade_low_band_2015_342_03_14.txt.csv"),
    )


@pytest.fixture(scope="session")
def cal_workflow_s11format(
    integration_test_data: Path, settings: Path, tmp_path_factory
) -> Path:
    return get_workflow(
        "s11format",
        settings,
        tmp_path_factory,
        integration_test_data,
        "",
        str(integration_test_data / "average_2015_342_03_14.txt"),
    )


@pytest.fixture(scope="session", autouse=True)
def run_workflow(
    workflow: Path,
    integration_test_data: Path,
) -> Path:
    invoke(
        cli.process,
        [
            str(workflow),
            "-i",
            str(integration_test_data / "2016_*_00_small.acq"),
            "-o",
            str(workflow.parent / "main"),
            "--no-mem-check",
        ],
    )

    return workflow.parent / "main"


@pytest.fixture(scope="session", autouse=True)
def run_workflow_nobeam(
    cal_workflow_nobeam: Path,
    run_workflow: Path,
) -> Path:
    out = cal_workflow_nobeam.parent / "nobeam"
    invoke(
        cli.process,
        [
            str(cal_workflow_nobeam),
            "-i",
            str(run_workflow / "*.gsh5"),
            "-o",
            str(out),
            "--start",
            "dicke_calibration",
            "--no-mem-check",
        ],
    )

    return out


@pytest.fixture(scope="session", autouse=True)
def run_workflow_s11format(
    cal_workflow_s11format: Path,
    run_workflow: Path,
) -> Path:
    out = cal_workflow_s11format.parent / "s11format"
    invoke(
        cli.process,
        [
            str(cal_workflow_s11format),
            "-i",
            str(run_workflow / "*.gsh5"),
            "-o",
            str(out),
            "--start",
            "dicke_calibration",
        ],
    )

    return out


@pytest.fixture(scope="session")
def raw_step(run_workflow: Path) -> Tuple[GSData, GSData]:
    globs = sorted(list(run_workflow.glob("*.gsh5")))
    return tuple(GSData.from_file(iter(globs)))


@pytest.fixture(scope="session")
def cal_step(run_workflow: Path) -> Tuple[GSData, GSData]:
    globs = sorted((run_workflow / "cal").glob("*.gsh5"))
    return tuple(GSData.from_file(fl) for fl in globs)


@pytest.fixture(scope="session")
def cal_step_nobeam(run_workflow_nobeam: Path) -> Tuple[GSData, GSData]:
    globs = sorted(
        x
        for x in (run_workflow_nobeam / "cal").glob("*.gsh5")
        if x.name.count(".") == 1
    )

    return tuple(GSData.from_file(fl) for fl in globs)


@pytest.fixture(scope="session")
def cal_step_s11format(run_workflow_s11format: Path) -> Tuple[GSData, GSData]:
    globs = sorted(
        x
        for x in (run_workflow_s11format / "cal").glob("*.gsh5")
        if x.name.count(".") == 1
    )

    return tuple(GSData.from_file(fl) for fl in globs)


@pytest.fixture(scope="session")
def model_step(run_workflow: Path) -> Tuple[GSData, GSData]:
    globs = sorted(
        x for x in (run_workflow / "cal").glob("*.gsh5") if ".linlog." in x.name
    )

    return tuple(GSData.from_file(fl) for fl in globs)


@pytest.fixture(scope="session")
def lstbin_step(run_workflow: Path) -> Tuple[GSData, GSData]:
    globs = sorted(
        x for x in (run_workflow / "cal").glob("*.gsh5") if ".L15min." in x.name
    )

    return tuple(GSData.from_file(fl) for fl in globs)


@pytest.fixture(scope="session")
def lstavg_step(run_workflow: Path) -> tuple[GSData]:
    return (GSData.from_file(run_workflow / "cal/lst-avg/lst_average.gsh5"),)


@pytest.fixture(scope="session")
def lstbin24_step(run_workflow: Path) -> tuple[GSData]:
    return (GSData.from_file(run_workflow / "cal/lst-avg/lstbin24hr.gsh5"),)


@pytest.fixture(scope="session")
def final_step(run_workflow: Path) -> tuple[GSData]:
    return (GSData.from_file(run_workflow / "cal/lst-avg/lst_average.400kHz.gsh5"),)


# @pytest.fixture(scope="session")
# def mock_calibrated_data(tmp_path_factory) -> CalibratedData:
#     np.random.seed(1234)
#     tmp_path = tmp_path_factory.mktemp("mock-data")

#     freq = np.linspace(50, 100, 100)
#     n_gha = 50

#     start_time = dt.datetime(year=2015, month=1, day=1)
#     timedelta = dt.timedelta(hours=12) / n_gha

#     time_strings = np.array(
#         [(start_time + i * timedelta).strftime("%Y:%j:%H:%M:%S") for i in range(n_gha)],
#         dtype="S17",
#     )

#     anc = CalibratedData.get_ancillary_coords(
#         CalibratedData.get_datetimes(time_strings)
#     )
#     anc["times"] = time_strings
#     anc["ambient_hum"] = np.zeros(len(time_strings))
#     anc["receiver_temp"] = np.ones(len(time_strings)) * 25

#     gha_model = 10000 * (1 + np.sin(2 * np.pi * (anc["gha"] - 18) / 24))

#     mdl = LinLog(default_x=freq, n_terms=2)

#     sky = np.array([mdl(parameters=[gg, 0]) for gg in gha_model])
#     noise = np.random.normal(0, scale=sky / 100)

#     data = {
#         "spectrum": sky + noise,
#         "switch_powers": np.concatenate((sky, sky, sky)).reshape((3, n_gha, len(freq))),
#         "weights": np.ones_like(sky),
#         "Q": (sky + noise - 300) / 400,
#     }

#     return CalibratedData.from_data(
#         {
#             "frequency": freq,
#             "spectra": data,
#             "ancillary": anc,
#             "meta": {
#                 "year": 2015,
#                 "day": 1,
#                 "hour": 1,
#                 "band": "low",
#                 "xrfi_pipe": {},
#             },
#         },
#         filename=tmp_path / "mock_calibrated_data_0.h5",
#         validate=False,
#     )
