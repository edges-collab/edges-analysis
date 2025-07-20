"""
Integration tests that ensure we can reproduce Alan's C-code results.

The data that we compare to is produced by Alan's pipeline using the
https://github.com/edges-collab/edges3-day300-310-test repo and the
docal_316test script with both the fittp-fix and pi-fix turned ON.
"""

import shutil
import traceback
from pathlib import Path

import numpy as np
import pytest
from click.testing import CliRunner

from edges.cal import alanmode as am
from edges.cal.cli import alancal

DATA_PATH = Path(__file__).parent / "data"

datadir = Path("/data5/edges/data/EDGES3_data/MRO")
if not datadir.exists():
    # We're not on enterprise, so we can only do the S11 stuff.
    datadir = DATA_PATH / "edges3-data-for-alan-comparison"

loads = ["amb", "hot", "open", "short"]
alandata = DATA_PATH / "edges3-2022-316-alan"


@pytest.fixture(scope="module")
def edges3_2022_316(data_path, tmp_path_factory):
    out = tmp_path_factory.mktemp("day316")

    runner = CliRunner()
    on_enterprise = "/data5/" in str(datadir)

    if not on_enterprise:
        avg_spec_files = sorted(datadir.glob("sp*.txt"))
        for fl in avg_spec_files:
            shutil.copy(fl, out / fl.name)

    args = [
        "2022_319_14",
        "2022",
        "316",
        "--out",
        str(out.absolute()),
        "-res",
        "49.8",
        "-ps",
        "33",
        "-cablen",
        "4.26",
        "-cabloss",
        "-91.5",
        "-cabdiel",
        "-1.24",
        "-fstart",
        "48",
        "-fstop",
        "198",
        "-smooth",
        "8",
        "-tload",
        "300",
        "-tcal",
        "1000",
        "-Lh",
        "-1",
        "-wfstart",
        "50.0",
        "-wfstop",
        "190.0",
        "-tcold",
        "306.5",
        "-thot",
        393.22,
        "-tcab",
        "306.5",
        "-cfit",
        "7",
        "-wfit",
        "7",
        "-nfit3",
        "10",
        "-nfit2",
        "27",
        "--no-plot",
        "--datadir",
        str(datadir),
    ]

    result = runner.invoke(alancal, args)

    if result.exit_code:
        print(result.exception)
        print(traceback.print_exception(*result.exc_info))

    print(result.output)
    assert result.exit_code == 0

    return out


@pytest.mark.parametrize("load", loads)
def test_spectra(edges3_2022_316: Path, load):
    # Test the raw spectra
    data, _ = am.read_spec_txt(f"{alandata}/sp{load}.txt")
    spfreq = data["freq"]
    alanspec = data["spectra"]
    data, _ = am.read_spec_txt(f"{edges3_2022_316}/sp{load}.txt")
    ourspec = data["spectra"]
    np.testing.assert_allclose(spfreq, data["freq"])

    # We don't currently get the very edges of the smoothing correct, but it doesn't
    # matter because we never use the very edges anyway. We test within these edges.
    np.testing.assert_allclose(alanspec[20:-20], ourspec[20:-20], atol=1e-6)


@pytest.mark.parametrize("load", [*loads, "lna"])
def test_unmodelled_s11(edges3_2022_316: Path, load):
    # Test the calibrated (unmodelled) S11s
    s11freq, alans11 = am.read_s11_csv(f"{alandata}/s11{load}.csv")
    ourfreq, ours11 = am.read_s11_csv(f"{edges3_2022_316}/s11{load}.csv")

    np.testing.assert_allclose(s11freq, ourfreq)
    np.testing.assert_allclose(alans11.real, ours11.real, atol=1e-10)
    np.testing.assert_allclose(alans11.imag, ours11.imag, atol=1e-10)


def test_modelled_s11(edges3_2022_316):
    # Test modelled S11s
    _alans11m = np.genfromtxt(f"{alandata}/s11_modelled.txt", comments="#", names=True)
    _ours11m = np.genfromtxt(
        f"{edges3_2022_316}/s11_modelled.txt", comments="#", names=True
    )

    for k in _alans11m.dtype.names:
        print(f"Modelled S11 {k}")

        # We clip the ends here, because they are slightly extrapolated in the default
        # case.
        np.testing.assert_allclose(_alans11m[k], _ours11m[k], atol=4e-9, rtol=0)


def test_specal(edges3_2022_316: Path):
    # Test final calibration
    acal = am.read_specal(f"{alandata}/specal.txt")
    ourcal = am.read_specal(f"{edges3_2022_316}/specal.txt")

    np.testing.assert_allclose(acal["freq"], ourcal["freq"])
    np.testing.assert_allclose(acal["C1"], ourcal["C1"], atol=2e-6)
    np.testing.assert_allclose(acal["C2"], ourcal["C2"], atol=4e-5)
    np.testing.assert_allclose(acal["Tunc"], ourcal["Tunc"], atol=1e-5)
    np.testing.assert_allclose(acal["Tcos"], ourcal["Tcos"], atol=2e-5)
    np.testing.assert_allclose(acal["Tsin"], ourcal["Tsin"], atol=2e-5)
