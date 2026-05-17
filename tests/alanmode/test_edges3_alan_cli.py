"""
Integration tests that ensure we can reproduce Alan's C-code results.

The data that we compare to is produced by Alan's pipeline using the
https://github.com/edges-collab/edges3-day300-310-test repo and the
docal_316test script with both the fittp-fix and pi-fix turned ON.
"""

import shutil
from pathlib import Path

import numpy as np
import pytest

from edges import alanmode as am
from edges.alanmode.cli import amode as _amode


def amode(*args, **kwargs):
    """Return the CLI app with SystemExit disabled for testing."""
    return _amode(*args, **kwargs, result_action="return_value")


@pytest.fixture(scope="module")
def alandata(alanmode_data_path: Path) -> Path:
    return alanmode_data_path / "edges3-2022-316-alan"


@pytest.fixture(scope="module")
def edges3_2022_316(tmp_path_factory, alanmode_data_path):
    out = tmp_path_factory.mktemp("day316")

    datadir = Path("/data5/edges/data/EDGES3_data/MRO")
    if not datadir.exists():
        # We're not on enterprise, so we can only do the S11 stuff.
        datadir = alanmode_data_path / "edges3-data-for-alan-comparison"

    on_enterprise = "/data5/" in str(datadir)

    if not on_enterprise:
        avg_spec_files = sorted(datadir.glob("sp*.txt"))
        for fl in avg_spec_files:
            shutil.copy(fl, out / fl.name)

    amode(
        "cal-edges3 "
        f"--out '{out.absolute()}' "
        f"--data.datadir '{datadir}' "
        "--data.s11date 2022_319_14 "
        "--data.specyear 2022 "
        "--data.specday 316 "
        "--data.res 49.8 "
        "--data.ps 33 "
        "--data.cablen 4.26 "
        "--data.cabloss -91.5 "
        "--data.cabdiel -1.24 "
        "--avg.fstart 48 "
        "--avg.fstop 198 "
        "--avg.smooth 8 "
        "--avg.tload 300 "
        "--avg.tcal 1000 "
        "--cal.Lh -1 "
        "--cal.wfstart 50.0 "
        "--cal.wfstop 190.0 "
        "--cal.tcold 306.5 "
        "--cal.thot 393.22 "
        "--cal.tcab 306.5 "
        "--cal.cfit 7 "
        "--cal.wfit 7 "
        "--cal.nfit3 10 "
        "--cal.nfit2 27 "
        "--no-plot "
    )
    return out


@pytest.mark.parametrize("load", am.LOADMAP.keys())
def test_spectra(edges3_2022_316: Path, load, alandata: Path):
    # Test the raw spectra
    data = am.read_spec_txt(f"{alandata}/sp{load}.txt")
    spfreq = data.freqs
    alanspec = data.data
    data = am.read_spec_txt(f"{edges3_2022_316}/sp{am.LOADMAP[load]}.txt")
    ourspec = data.data
    np.testing.assert_allclose(spfreq, data.freqs)

    # We don't currently get the very edges of the smoothing correct, but it doesn't
    # matter because we never use the very edges anyway. We test within these edges.
    np.testing.assert_allclose(alanspec[20:-20], ourspec[20:-20], atol=1e-6)


@pytest.mark.parametrize("load", [*list(am.LOADMAP.keys()), "lna"])
def test_unmodelled_s11(edges3_2022_316: Path, load, alandata: Path):
    # Test the calibrated (unmodelled) S11s

    s11freq, alans11 = am.read_s11_csv(f"{alandata}/s11{load}.csv")
    key = "lna" if load == "lna" else am.LOADMAP[load]
    ourfreq, ours11 = am.read_s11_csv(f"{edges3_2022_316}/s11{key}.csv")

    # Need to mask the C-code frequencies, because they get written out before
    # being cut to wfstart-wfstop, whereas we only have access to the raw s11
    # *after* this cut in python.
    mask = (s11freq >= ourfreq.min()) & (s11freq <= ourfreq.max())

    np.testing.assert_allclose(s11freq[mask], ourfreq)
    np.testing.assert_allclose(alans11[mask].real, ours11.real, atol=1e-10)
    np.testing.assert_allclose(alans11[mask].imag, ours11.imag, atol=1e-10)


def test_modelled_s11(edges3_2022_316, alandata: Path):
    # Test modelled S11s
    alans11m = np.genfromtxt(f"{alandata}/s11_modelled.txt", comments="#", names=True)
    ours11m = np.genfromtxt(
        f"{edges3_2022_316}/s11_modelled.txt", comments="#", names=True
    )

    for k in alans11m.dtype.names:
        print(f"Modelled S11 {k}")

        # We clip the ends here, because they are slightly extrapolated in the default
        # case.
        np.testing.assert_allclose(alans11m[k], ours11m[k], atol=5e-9, rtol=0)


def test_specal(edges3_2022_316: Path, alandata: Path):
    # Test final calibration
    alan = am.read_specal(f"{alandata}/specal.txt", t_load=300, t_load_ns=1000)
    ours = am.read_specal(f"{edges3_2022_316}/specal.txt", t_load=300, t_load_ns=1000)

    np.testing.assert_allclose(ours.Tsca, alan.Tsca, rtol=3e-6)
    np.testing.assert_allclose(ours.Toff, alan.Toff, atol=4e-5)
    np.testing.assert_allclose(ours.Tunc, alan.Tunc, atol=1.1e-6)
    np.testing.assert_allclose(ours.Tcos, alan.Tcos, atol=2e-5)
    np.testing.assert_allclose(ours.Tsin, alan.Tsin, atol=1.4e-5)
