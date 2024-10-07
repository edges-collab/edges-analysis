"""Pytest configuration for the edges-analysis package."""

from __future__ import annotations

from pathlib import Path
from subprocess import run

import numpy as np
import pytest
import yaml
from astropy import units as un
from astropy.time import Time
from click.testing import CliRunner
from edges_cal import modelling as mdl
from jinja2 import Template

from edges_analysis import cli, const
from edges_analysis.averaging import lstbin
from edges_analysis.calibration.calibrate import dicke_calibration
from edges_analysis.config import config
from edges_analysis.datamodel import add_model
from edges_analysis.gsdata import GSData

from .mock_gsdata import create_mock_edges_data

runner = CliRunner()


def invoke(cmd, args, **kwargs):
    result = runner.invoke(cmd, args, **kwargs)
    print(result.output)
    if result.exit_code > 0:
        raise result.exc_info[1]

    return result


@pytest.fixture(scope="session")
def integration_test_data() -> Path:
    tmp_path = Path(
        "/tmp/edges-analysis-pytest"
    )  # tmp_path_factory.mktemp("integration-data", numbered=False)
    repo = tmp_path / "edges-analysis-test-data"

    if repo.exists():
        run(["git", "-C", str(repo), "pull"])
    else:
        run(
            [
                "git",
                "clone",
                "https://github.com/edges-collab/edges-analysis-test-data",
                str(repo),
                "--depth",
                "1",
            ]
        )
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
    return integration_test_data / "calfile_2015.h5"


@pytest.fixture(scope="session")
def s11path(integration_test_data: Path) -> Path:
    return integration_test_data / "s11"


@pytest.fixture(scope="session")
def beamfile(integration_test_data: Path) -> Path:
    return integration_test_data / "alan_beam_factor.h5"


def get_workflow(
    name: str, settings, workflow_dir, integration_test_data, beam_file, s11_path
):
    with (settings / "integration_workflow.yaml").open("r") as fl:
        workflow = Template(fl.read())

    txt = workflow.render(
        weather_file=str(integration_test_data / "weather.txt"),
        thermlog_file=str(integration_test_data / "thermlog_low.txt"),
        calfile=str(integration_test_data / "calfile_2015.h5"),
        s11_path=s11_path,
        beam_file=beam_file,
    )

    if not beam_file:
        print("loading up the yaml...")
        wf = yaml.load(txt, Loader=yaml.FullLoader)
        wf["steps"] = tuple(
            x for x in wf["steps"] if x["function"] != "apply_beam_correction"
        )
        txt = yaml.dump(wf)
    with (workflow_dir / f"workflow_{name}.yaml").open("w") as fl:
        fl.write(txt)

    return workflow_dir / f"workflow_{name}.yaml"


@pytest.fixture(scope="session")
def workflow(integration_test_data: Path, settings: Path, workflow_dir: Path) -> Path:
    return get_workflow(
        "main",
        settings,
        workflow_dir,
        integration_test_data,
        str(integration_test_data / "alan_beam_factor.h5"),
        str(integration_test_data / "s11"),
    )


@pytest.fixture(scope="session")
def cal_workflow_nobeam(
    integration_test_data: Path, settings: Path, workflow_dir: Path
) -> Path:
    return get_workflow(
        "nobeam",
        settings,
        workflow_dir,
        integration_test_data,
        "",
        str(integration_test_data / "S11_blade_low_band_2015_342_03_14.txt.csv"),
    )


@pytest.fixture(scope="session")
def cal_workflow_s11format(
    integration_test_data: Path, settings: Path, workflow_dir: Path
) -> Path:
    return get_workflow(
        "s11format",
        settings,
        workflow_dir,
        integration_test_data,
        "",
        str(integration_test_data / "average_2015_342_03_14.txt"),
    )


@pytest.fixture(scope="session")
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


@pytest.fixture(scope="session")
def run_workflow_nobeam(
    cal_workflow_nobeam: Path,
    run_workflow: Path,
) -> Path:
    out = cal_workflow_nobeam.parent / "nobeam"
    invoke(cli.fork, [str(cal_workflow_nobeam), str(run_workflow), "-o", str(out)])

    invoke(
        cli.process,
        [
            str(cal_workflow_nobeam),
            "-o",
            str(out),
            "--no-mem-check",
        ],
    )

    return out


@pytest.fixture(scope="session")
def run_workflow_s11format(
    cal_workflow_s11format: Path,
    run_workflow: Path,
) -> Path:
    out = cal_workflow_s11format.parent / "s11format"
    invoke(cli.fork, [str(cal_workflow_s11format), str(run_workflow), "-o", str(out)])

    invoke(
        cli.process,
        [
            str(cal_workflow_s11format),
            "-o",
            str(out),
            "--no-mem-check",
        ],
    )

    return out


@pytest.fixture(scope="session")
def raw_step(run_workflow: Path) -> tuple[GSData, GSData]:
    globs = sorted(run_workflow.glob("*.gsh5"))
    return tuple(GSData.from_file(fl) for fl in globs)


@pytest.fixture(scope="session")
def cal_step(run_workflow: Path) -> tuple[GSData, GSData]:
    globs = sorted((run_workflow / "cal").glob("*.gsh5"))
    return tuple(GSData.from_file(fl) for fl in globs)


@pytest.fixture(scope="session")
def cal_step_nobeam(run_workflow_nobeam: Path) -> tuple[GSData, GSData]:
    globs = sorted((run_workflow_nobeam / "cal").glob("*.gsh5"))
    return tuple(GSData.from_file(fl) for fl in globs)


@pytest.fixture(scope="session")
def cal_step_s11format(run_workflow_s11format: Path) -> tuple[GSData, GSData]:
    globs = sorted((run_workflow_s11format / "cal").glob("*.gsh5"))
    return tuple(GSData.from_file(fl) for fl in globs)


@pytest.fixture(scope="session")
def model_step(run_workflow: Path) -> tuple[GSData, GSData]:
    globs = sorted((run_workflow / "cal/linlog").glob("*.gsh5"))

    return tuple(GSData.from_file(fl) for fl in globs)


@pytest.fixture(scope="session")
def lstbin_step(run_workflow: Path) -> tuple[GSData, GSData]:
    globs = sorted((run_workflow / "cal/linlog/L15min/").glob("*.gsh5"))

    return tuple(GSData.from_file(fl) for fl in globs)


@pytest.fixture(scope="session")
def lstavg_step(run_workflow: Path) -> tuple[GSData]:
    return (
        GSData.from_file(run_workflow / "cal/linlog/L15min/lst-avg/lst_average.gsh5"),
    )


@pytest.fixture(scope="session")
def lstbin24_step(run_workflow: Path) -> tuple[GSData]:
    return (
        GSData.from_file(run_workflow / "cal/linlog/L15min/lst-avg/lstbin24hr.gsh5"),
    )


@pytest.fixture(scope="session")
def final_step(run_workflow: Path) -> tuple[GSData]:
    return (
        GSData.from_file(
            run_workflow / "cal/linlog/L15min/lst-avg/lstbin24hr.400kHz.gsh5"
        ),
    )


@pytest.fixture(scope="session")
def gsd_ones():
    nload, npol, ntime, nfreq = 1, 2, 10, 26
    return GSData(
        data=np.ones((nload, npol, ntime, nfreq)),
        freq_array=np.linspace(50, 100, nfreq) * un.MHz,
        time_array=Time(
            np.linspace(2459856, 2459857, ntime + 1)[:-1, None],
            format="jd",
            scale="utc",
        ),
        telescope_location=const.edges_location,
        loads=("ant",),
    )


@pytest.fixture(scope="session")
def gsd_ones_power():
    nload, npol, ntime, nfreq = 3, 2, 10, 26
    times = np.linspace(2459856, 2459857, ntime + 1)[:-1]
    times = np.array([times, times, times]).T

    return GSData(
        data=np.ones((nload, npol, ntime, nfreq)),
        freq_array=np.linspace(50, 100, nfreq) * un.MHz,
        time_array=Time(times, format="jd", scale="utc"),
        telescope_location=const.edges_location,
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
        first_edge=mock.lst_array.min().hour,
        max_edge=mock.lst_array.max().hour,
    )


@pytest.fixture(scope="session")
def mock_season() -> list[GSData]:
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
