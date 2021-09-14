from click.testing import CliRunner
from pathlib import Path
from edges_analysis import cli
from edges_analysis.analysis import read_step
import numpy as np

runner = CliRunner()


def test_flag_idx_filter(edges_config: dict, cal_step, settings: Path):
    fnames = [str(obj.filename) for obj in cal_step]

    print("Hey I'm Here...")
    # Run second xrfi filter on same data...
    result = runner.invoke(
        cli.filter,
        [
            str(settings / "xrfi.yml"),
            "-i",
            fnames[0],
            "-i",
            fnames[1],
            "-l",
            "should-be-same",
            "--flag-idx",
            "0",
        ],
    )
    print(result.output)

    filtered_files = sorted(
        (cal_step[0].filename.parent / "should-be-same").glob("*.h5")
    )
    assert len(filtered_files) == 2

    obj = read_step(filtered_files[0])
    assert len(obj.filters_applied) == 1
    assert obj.filters_applied["rfi_model_filter"] == 0

    assert np.all(cal_step[0].weights == obj.weights)

    # Do it again, but this time clobber the result
    result = runner.invoke(
        cli.filter,
        [
            str(settings / "xrfi.yml"),
            "-i",
            fnames[0],
            "-i",
            fnames[1],
            "-l",
            "should-be-same",
            "--flag-idx",
            "0",
            "--clobber",
        ],
    )
    print(result.output)

    filtered_files2 = sorted(
        (cal_step[0].filename.parent / "should-be-same").glob("*.h5")
    )
    assert len(filtered_files2) == 2

    obj2 = read_step(filtered_files2[0])
    assert len(obj2.filters_applied) == 1
    assert obj2.filters_applied["rfi_model_filter"] == 0

    assert np.all(cal_step[0].weights == obj2.weights)
    assert obj.get_filter_meta("rfi_model_filter") == obj2.get_filter_meta(
        "rfi_model_filter"
    )
