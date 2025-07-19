"""Test that we recover the calibration of Alan for the H2 case in B2018.

The data files that we compare to here were computed using the alans-pipeline code,
specifically the memo-208 branch, detailed in ASU Memo #208. The data tested against
is that computed WITH the fittp fix described in that memo.

We do *not* test the acqplot7amoon function here, because the data for that is too large
to carry around in our tests. Instead, the averaged spectra are included in the tests
directly.
"""

import shutil
import traceback
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from astropy import units as un
from click.testing import CliRunner

from edges.cal.alanmode import read_specal, read_specal_as_calibrator
from edges.cal.cli import alancal2

DATA = Path(__file__).parent / "data" / "b18_cal_test"


@pytest.fixture(scope="module")
def b18cal(tmpdir):
    tmpdir = tmpdir / "outputs"
    tmpdir.mkdir()

    runner = CliRunner()

    # Put the averaged spectrum files into the working directory.
    specfiles = DATA.glob("sp*.txt")
    for fl in specfiles:
        shutil.copy(fl, f"{tmpdir}/")

    # Note that all the ACQs here point to their original names, but they are not
    # actually used in this test.
    result = runner.invoke(
        alancal2,
        args=[
            "--s11-path",
            f"{DATA}/s11_calibration_low_band_LNA25degC_2015-09-16-12-30-29_simulator2_long.txt",
            "--ambient-acqs",
            f"{DATA}/Ambient_01_2015_245_02_00_00_lab.acq",
            "--ambient-acqs",
            f"{DATA}/Ambient_01_2015_246_02_00_00_lab.acq",
            "--hotload-acqs",
            f"{DATA}/HotLoad_01_2015_246_04_00_00_lab.acq",
            "--hotload-acqs",
            f"{DATA}/HotLoad_01_2015_247_00_00_00_lab.acq",
            "--open-acqs",
            f"{DATA}/LongCableOpen_01_2015_243_14_00_00_lab.acq",
            "--open-acqs",
            f"{DATA}/LongCableOpen_01_2015_244_00_00_00_lab.acq",
            "--open-acqs",
            f"{DATA}/LongCableOpen_01_2015_245_00_00_00_lab.acq",
            "--short-acqs",
            f"{DATA}/LongCableShorted_01_2015_241_00_00_00_lab.acq",
            "--short-acqs",
            f"{DATA}/LongCableShorted_01_2015_242_00_00_00_lab.acq",
            "--short-acqs",
            f"{DATA}/LongCableShorted_01_2015_243_00_00_00_lab.acq",
            "-fstart=40.0",
            "-fstop=110.0",
            "-wfstart=50.0",
            "-wfstop=100.0",
            "-smooth=8",
            "-tload=300.0",
            "-tstart=0",
            "-tstop=23",
            "-delaystart=7200",
            "-Lh=-2",
            "-tcold=296",
            "-thot=399",
            "-cfit=6",
            "-wfit=5",
            "-nfit2=27",
            "-nfit3=11",
            "--lna-poly=0",
            "--no-spectra",
            "--no-plot",
            "--out",
            str(tmpdir),
        ],
    )

    if result.exit_code:
        print(result.exception)
        print(traceback.print_exception(*result.exc_info))

    print(result.output)
    assert result.exit_code == 0

    return tmpdir


@pytest.fixture(scope="module")
def mask():
    alancalfreq, _ = read_s11m(f"{DATA}/s11_modelled_alan.txt")
    return (alancalfreq >= 50.0) & (alancalfreq <= 100.0)


def read_s11m(pth) -> tuple[np.ndarray, pd.DataFrame]:
    _s11m = np.genfromtxt(pth, comments="#", names=True)
    s11m = {}
    freq = _s11m["freq"]
    for name in _s11m.dtype.names:
        if name == "freq":
            continue

        bits = name.split("_")
        cmp = bits[-1]
        load = "_".join(bits[:-1])

        if load not in s11m:
            s11m[load] = np.zeros(len(_s11m), dtype=complex)
        if cmp == "real":
            s11m[load] += _s11m[name]
        else:
            s11m[load] += _s11m[name] * 1j

    return freq, pd.DataFrame(s11m)


def test_s11_modelled(b18cal: Path, mask):
    alancalfreq, alan = read_s11m(f"{DATA}/s11_modelled_alan.txt")
    calfreq, ours = read_s11m(b18cal / "s11_modelled.txt")

    assert np.allclose(alancalfreq[mask], calfreq)
    for load in ours:
        print(load)
        np.testing.assert_allclose(
            ours[load].to_numpy(), alan[load].to_numpy()[mask], rtol=0, atol=1e-7
        )


def test_hot_load_loss(b18cal: Path, mask):
    alans = np.genfromtxt(f"{DATA}/hot_load_loss.txt")
    ours = np.genfromtxt(b18cal / "hot_load_loss.txt")

    np.testing.assert_allclose(ours[:, 1], alans[mask, 1], atol=1e-7, rtol=0)


def test_calcoeffs(b18cal: Path, mask):
    ours = read_specal(b18cal / "specal.txt")
    alan = read_specal(f"{DATA}/specal_alan.txt")[mask]

    for name in ours.dtype.names:
        print(name)
        np.testing.assert_allclose(ours[name], alan[name], rtol=0, atol=1.1e-6)


def test_read_specal_as_calibrator():
    data = read_specal(DATA / "specal_alan.txt")
    data = data[data["weight"] > 0]
    calobs = read_specal_as_calibrator(DATA / "specal_alan.txt", nfit1=51)

    f = data["freq"] * un.MHz
    np.testing.assert_allclose(calobs.C1(f), data["C1"], atol=1e-4)
    np.testing.assert_allclose(calobs.C2(f), data["C2"], atol=1e-4)
    np.testing.assert_allclose(calobs.Tunc(f), data["Tunc"], atol=1e-4)
    np.testing.assert_allclose(calobs.Tcos(f), data["Tcos"], atol=1e-4)
    np.testing.assert_allclose(calobs.Tsin(f), data["Tsin"], atol=1e-4)
    np.testing.assert_allclose(
        calobs.receiver_s11(f).real, data["s11lna_real"], atol=1e-4
    )
    np.testing.assert_allclose(
        calobs.receiver_s11(f).imag, data["s11lna_imag"], atol=1e-4
    )
