"""Test that we recover the calibration of Alan for the H2 case in B2018.

The data files that we compare to here were computed using the alans-pipeline code,
specifically the memo-208 branch, detailed in ASU Memo #208. The data tested against
is that computed WITH the fittp fix described in that memo.

We do *not* test the acqplot7amoon function here, because the data for that is too large
to carry around in our tests. Instead, the averaged spectra are included in the tests
directly.
"""

import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from edges.alanmode import read_specal
from edges.alanmode.alanmode import (
    ACQPlot7aMoonParams,
    Edges2CalobsParams,
    EdgesScriptParams,
)
from edges.alanmode.cli import AlanCalOpts, alancal_edges2


@pytest.fixture(scope="module")
def b18cal(tmpdir, testdata_path):
    tmpdir = tmpdir / "outputs"
    tmpdir.mkdir()

    DATA = testdata_path / "alanmode" / "b18_cal_test"

    # Put the averaged spectrum files into the working directory.
    specfiles = DATA.glob("sp*.txt")
    for fl in specfiles:
        shutil.copy(fl, f"{tmpdir}/")

    # Note that all the ACQs here point to their original names, but they are not
    # actually used in this test.
    defparams = Edges2CalobsParams(
        s11_path=DATA
        / "s11_calibration_low_band_LNA25degC_2015-09-16-12-30-29_simulator2_long.txt",
        ambient_acqs=[
            DATA / "Ambient_01_2015_245_02_00_00_lab.acq",
            DATA / "Ambient_01_2015_246_02_00_00_lab.acq",
        ],
        hotload_acqs=[
            DATA / "HotLoad_01_2015_246_04_00_00_lab.acq",
            DATA / "HotLoad_01_2015_247_00_00_00_lab.acq",
        ],
        open_acqs=[
            DATA / "LongCableOpen_01_2015_243_14_00_00_lab.acq",
            DATA / "LongCableOpen_01_2015_244_00_00_00_lab.acq",
            DATA / "LongCableOpen_01_2015_245_00_00_00_lab.acq",
        ],
        short_acqs=[
            DATA / "LongCableShorted_01_2015_241_00_00_00_lab.acq",
            DATA / "LongCableShorted_01_2015_242_00_00_00_lab.acq",
            DATA / "LongCableShorted_01_2015_243_00_00_00_lab.acq",
        ],
    )
    acqparams = ACQPlot7aMoonParams(
        fstart=40.0,
        fstop=110.0,
        smooth=8,
        tload=300.0,
        tstart=0,
        tstop=23,
        delaystart=7200,
    )
    calparams = EdgesScriptParams(
        wfstart=50.0,
        wfstop=100.0,
        Lh=-2,
        tcold=296,
        thot=399,
        cfit=6,
        wfit=5,
        nfit2=27,
        nfit3=11,
        lna_poly=0,
    )

    alancal_edges2(
        data=defparams,
        opts=AlanCalOpts(
            avg=acqparams,
            cal=calparams,
            plot=False,
            out=tmpdir,
            redo_spectra=False,
            redo_cal=True,
        ),
    )

    return tmpdir


@pytest.fixture(scope="module")
def mask(testdata_path):
    DATA = testdata_path / "alanmode" / "b18_cal_test"

    alancalfreq, _ = read_s11m(DATA / "s11_modelled_alan.txt")
    return (alancalfreq >= 50.0) & (alancalfreq <= 100.0)


def read_s11m(pth) -> tuple[np.ndarray, pd.DataFrame]:
    raw_s11m = np.genfromtxt(pth, comments="#", names=True)
    s11m = {}
    freq = raw_s11m["freq"]
    for name in raw_s11m.dtype.names:
        if name == "freq":
            continue

        bits = name.split("_")
        cmp = bits[-1]
        load = "_".join(bits[:-1])

        if load not in s11m:
            s11m[load] = np.zeros(len(raw_s11m), dtype=complex)
        if cmp == "real":
            s11m[load] += raw_s11m[name]
        else:
            s11m[load] += raw_s11m[name] * 1j

    return freq, pd.DataFrame(s11m)


def test_s11_modelled(b18cal: Path, mask, testdata_path):
    DATA = testdata_path / "alanmode" / "b18_cal_test"

    alancalfreq, alan = read_s11m(DATA / "s11_modelled_alan.txt")
    calfreq, ours = read_s11m(b18cal / "s11_modelled.txt")

    assert np.allclose(alancalfreq[mask], calfreq)
    for load in ours:
        print(f"Testing {load}")
        np.testing.assert_allclose(
            ours[load].to_numpy(), alan[load].to_numpy()[mask], rtol=0, atol=1e-7
        )


def test_hot_load_loss(b18cal: Path, mask, testdata_path):
    DATA = testdata_path / "alanmode" / "b18_cal_test"

    alans = np.genfromtxt(DATA / "hot_load_loss.txt")
    ours = np.genfromtxt(b18cal / "hot_load_loss.txt")
    np.testing.assert_allclose(ours[:, 1], alans[mask, 1], atol=1e-7, rtol=0)


def test_calcoeffs(b18cal: Path, mask, testdata_path):
    DATA = testdata_path / "alanmode" / "b18_cal_test"

    ours = read_specal(b18cal / "specal.txt", t_load=300.0, t_load_ns=1000.0)
    alan = read_specal(DATA / "specal_alan.txt", t_load=300.0, t_load_ns=1000.0)

    np.testing.assert_allclose(
        ours.Tsca, alan.Tsca, rtol=3e-6
    )  # since we mutliply by tcal=1000, use 1e-3
    np.testing.assert_allclose(ours.Toff, alan.Toff, atol=1.1e-6)
    np.testing.assert_allclose(ours.Tunc, alan.Tunc, atol=1.1e-6)
    np.testing.assert_allclose(ours.Tcos, alan.Tcos, atol=1.1e-6)
    np.testing.assert_allclose(ours.Tsin, alan.Tsin, atol=1.1e-6)
