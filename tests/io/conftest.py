from pathlib import Path

import numpy as np
import pytest
from astropy import units as apu
from astropy.time import Time
from pygsdata import KNOWN_TELESCOPES, GSData, Telescope
from read_acq.gsdata import write_gsdata_to_acq

from edges.io import TEST_DATA_PATH, CalObsDefEDGES3


@pytest.fixture(scope="session")
def tmpdir(tmp_path_factory):
    return tmp_path_factory.mktemp("edges-io-tests")


@pytest.fixture(scope="session")
def edgeslow():
    return KNOWN_TELESCOPES["edges-low"]


@pytest.fixture(scope="session")
def small_gsdata_obj(edgeslow: Telescope):
    ntimes = 2
    nfreqs = 32768
    npols = 1
    nloads = 3
    return GSData(
        data=np.zeros((nloads, npols, ntimes, nfreqs)),
        freqs=np.linspace(0, 200, nfreqs) * apu.MHz,
        times=Time([
            ["2020:001:01:01:01", "2020:001:01:01:01", "2020:001:01:01:01"],
            ["2020:001:01:02:01", "2020:001:01:02:01", "2020:001:01:02:01"],
        ]),
        telescope=edgeslow,
        data_unit="power",
        auxiliary_measurements={
            "adcmax": np.zeros((ntimes, nloads)),
            "adcmin": np.zeros((ntimes, nloads)),
            "data_drops": np.zeros((ntimes, nloads), dtype="int"),
        },
    )


@pytest.fixture(scope="session", autouse=True)
def fastspec_spectrum_fl(tmpdir, small_gsdata_obj: GSData):
    """An auto-generated empty Fastspec h5 format file."""
    flname = tmpdir / "fastspec_example_file.acq"
    write_gsdata_to_acq(small_gsdata_obj, flname)
    return flname


@pytest.fixture(scope="session")
def datadir() -> Path:
    return TEST_DATA_PATH


@pytest.fixture(scope="module")
def smallcaldef_edges3(datadir: Path) -> CalObsDefEDGES3:
    return CalObsDefEDGES3.from_standard_layout(
        rootdir=datadir / "edges3-mock-root",
        year=2023,
        day=70,
    )
