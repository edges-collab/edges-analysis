import json
import traceback
from pathlib import Path

from click.testing import CliRunner

from edges.cal.cli import compare, report, run


def test_run(cal_data_path: Path, tmpdir: Path):
    runner = CliRunner()
    outdir = tmpdir / "cli-out"
    if not outdir.exists():
        outdir.mkdir()
    result = runner.invoke(
        run,
        [
            str(cal_data_path / "settings.yaml"),
            str(cal_data_path / "Receiver01_25C_2019_11_26_040_to_200MHz"),
            "--out",
            str(outdir),
            "--global-config",
            json.dumps({"cal": {"cache-dir": str(outdir)}}),
            "--plot",
            "--simulators",
            "AntSim2",
        ],
    )

    if result.exit_code:
        print(result.exception)
        print(traceback.print_exception(*result.exc_info))

    print(result.output)
    assert result.exit_code == 0


def test_report(cal_data_path: Path, tmpdir: Path):
    runner = CliRunner()
    outdir = tmpdir / "cli-out"
    if not outdir.exists():
        outdir.mkdir()

    result = runner.invoke(
        report,
        [
            str(cal_data_path / "settings.yaml"),
            str(cal_data_path / "Receiver01_25C_2019_11_26_040_to_200MHz"),
            "--out",
            str(outdir),
            "--global-config",
            json.dumps({"cal": {"cache-dir": str(outdir)}}),
            "--no-pdf",
        ],
        catch_exceptions=False,
    )

    if result.exit_code:
        print(result.exception)
        print(traceback.print_exception(*result.exc_info))

    print(result.output)

    assert result.exit_code == 0


def test_compare(cal_data_path: Path, tmpdir: Path):
    runner = CliRunner()
    outdir = tmpdir / "cli-out"
    if not outdir.exists():
        outdir.mkdir()

    result = runner.invoke(
        compare,
        [
            str(cal_data_path / "settings.yaml"),
            str(cal_data_path / "Receiver01_25C_2019_11_26_040_to_200MHz"),
            str(cal_data_path / "settings.yaml"),
            str(cal_data_path / "Receiver01_25C_2019_11_26_040_to_200MHz"),
            "--out",
            str(outdir),
            "--global-config",
            json.dumps({"cal": {"cache-dir": str(outdir)}}),
            "--no-pdf",
        ],
        catch_exceptions=False,
    )

    if result.exit_code:
        print(result.exception)
        print(traceback.print_exception(*result.exc_info))

    print(result.output)

    assert result.exit_code == 0
