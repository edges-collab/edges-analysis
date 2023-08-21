from __future__ import annotations

import shutil
from click.testing import CliRunner

from edges_analysis import cli
from edges_analysis.gsdata import GSData

runner = CliRunner()


def invoke(cmd, args, **kwargs):
    result = runner.invoke(cmd, args, **kwargs)
    print(result.output)
    if result.exit_code > 0:
        raise result.exc_info[1]

    return result


def test_add_file(workflow, integration_test_data):
    res = invoke(
        cli.process,
        [
            str(workflow),
            "-i",
            str(integration_test_data / "2016_292_00_small.acq"),
            "-o",
            str(workflow.parent / "here"),
            "--no-mem-check",
        ],
    )

    # Now run a second time. We should have both files there now.
    res2 = invoke(
        cli.process,
        [
            str(workflow),
            "-i",
            str(integration_test_data / "2016_295_00_small.acq"),
            "-o",
            str(workflow.parent / "here"),
            "--no-mem-check",
        ],
    )

    assert "2016_295" not in res.output
    assert "2016_295" in res2.output
    assert "2016_292" in res2.output

    # Now restart the workflow.
    invoke(
        cli.process,
        [
            str(workflow),
            "-i",
            str(integration_test_data / "2016_295_00_small.acq"),
            "-o",
            str(workflow.parent / "here"),
            "--no-mem-check",
            "--restart",
        ],
    )

    assert any(
        fl.name.startswith("2016_295") for fl in (workflow.parent / "here").glob("*")
    )


def test_stop(workflow, integration_test_data):
    workdir = workflow.parent / "stop"
    res = invoke(
        cli.process,
        [
            str(workflow),
            "-i",
            str(integration_test_data / "2016_292_00_small.acq"),
            "-o",
            str(workdir),
            "--no-mem-check",
            "--stop",
            "linlog",
        ],
    )

    assert (workdir / "cal/linlog/2016_292_00_small.gsh5").exists()
    assert not (workdir / "cal/linlog/L15min/2016_292_00_small.gsh5").exists()

    res = invoke(
        cli.process,
        [
            str(workflow),
            "-o",
            str(workdir),
            "--no-mem-check",
        ],
    )

    assert "linlog does not need to run on any files" in res.output
    assert (workdir / "cal/linlog/L15min/2016_292_00_small.gsh5").exists()

    # Now remove the last file...
    (workdir / "cal/linlog/L15min/lst-avg/lstbin24hr.400kHz.gsh5").unlink()

    # Run again, but this time add extra file
    res = invoke(
        cli.process,
        [
            str(workflow),
            "-i",
            str(integration_test_data / "2016_295_00_small.acq"),
            "-o",
            str(workdir),
            "--no-mem-check",
        ],
    )
    assert "does not need to run any files" not in res.output
    assert (workdir / "cal/linlog/L15min/lst-avg/lstbin24hr.400kHz.gsh5").exists()


def test_stop_at_filter(workflow, integration_test_data):
    workdir = workflow.parent / "stop_at_filter"
    res = invoke(
        cli.process,
        [
            str(workflow),
            "-i",
            str(integration_test_data / "2016_29*_00_small.acq"),
            "-o",
            str(workdir),
            "--no-mem-check",
            "--stop",
            "negative_power_filter",
        ],
    )

    assert (workdir / "2016_292_00_small.gsh5").exists()
    assert (workdir / "2016_295_00_small.gsh5").exists()

    data = GSData.from_file(workdir / "2016_292_00_small.gsh5")

    assert "negative_power_filter" in data.flags

    res = invoke(
        cli.process,
        [
            str(workflow),
            "-o",
            str(workdir),
            "--no-mem-check",
        ],
    )

    assert "negative_power_filter does not need to run on any files" in res.output
    assert (workdir / "cal/linlog/L15min/2016_292_00_small.gsh5").exists()


def test_delete_file(workflow, integration_test_data):
    workdir = workflow.parent / "delete-file"
    res = invoke(
        cli.process,
        [
            str(workflow),
            "-i",
            str(integration_test_data / "2016_29*_00_small.acq"),
            "-o",
            str(workdir),
            "--no-mem-check",
        ],
    )

    shutil.copy(workdir / "progressfile.yaml", workdir / "progressfile.yaml.bak")

    assert (workdir / "2016_292_00_small.gsh5").exists()
    assert (workdir / "2016_295_00_small.gsh5").exists()

    data = GSData.from_file(workdir / "2016_292_00_small.gsh5")

    assert "negative_power_filter" in data.flags

    # Remove the calibration files
    (workdir / "cal/2016_292_00_small.gsh5").unlink()
    (workdir / "cal/2016_295_00_small.gsh5").unlink()

    res = invoke(
        cli.process,
        [
            str(workflow),
            "-o",
            str(workdir),
            "--no-mem-check",
        ],
    )

    assert "crop_50mhz does not need to run on any files" not in res.output
    assert "negative_power_filter does not need to run on any files" in res.output

    assert (workdir / "cal/linlog/L15min/2016_292_00_small.gsh5").exists()
    assert (workdir / "cal/linlog/L15min/2016_295_00_small.gsh5").exists()


def test_start(workflow, integration_test_data):
    workdir = workflow.parent / "delete-file"
    res = invoke(
        cli.process,
        [
            str(workflow),
            "-i",
            str(integration_test_data / "2016_29*_00_small.acq"),
            "-o",
            str(workdir),
            "--no-mem-check",
        ],
    )

    res = invoke(
        cli.process,
        [
            str(workflow),
            "-o",
            str(workdir),
            "--no-mem-check",
            "--start",
            "negative_power_filter",
        ],
    )

    assert "negative_power_filter does not need to run on any files" not in res.output

    assert (workdir / "cal/linlog/L15min/2016_292_00_small.gsh5").exists()
    assert (workdir / "cal/linlog/L15min/2016_295_00_small.gsh5").exists()
