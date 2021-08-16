from click.testing import CliRunner
from pathlib import Path
from edges_analysis.cli import process, filter

runner = CliRunner()


def test_process(
    calibrate_settings: Path, settings: Path, integration_test_data: Path, tmpconfig
):
    out = runner.invoke(
        process,
        [
            "calibrate",
            str(calibrate_settings),
            "-i",
            str(integration_test_data / "2016_292_00_small.acq"),
            "-l",
            "test",
        ],
    )
    print(out.output)
    assert out.exit_code == 0

    out = runner.invoke(
        filter,
        [
            str(settings / "xrfi.yml"),
            "-i",
            tmpconfig["paths"]["field_products"] + "/test/2016_292_00.h5",
        ],
    )
    print(out.output)
    assert out.exit_code == 0
