from pathlib import Path

import numpy as np
import pytest
from astropy import units as un
from astropy.time import Time
from pygsdata import GSData

from edges import const
from edges import modeling as mdl
from edges.analysis.datamodel import add_model
from edges.averaging import lstbin
from edges.cal import CalibratedS11, CalibrationObservation, Load, LoadSpectrum
from edges.cal.calibrator import Calibrator
from edges.cal.dicke import dicke_calibration
from edges.cal.receiver_cal import get_calcoeffs_iterative
from edges.frequencies import edges_raw_freqs
from edges.io import calobsdef
from edges.testing import create_mock_edges_data


@pytest.fixture(scope="session")
def testdata_path() -> Path:
    return Path(__file__).parent / "data"


@pytest.fixture(scope="session", autouse=True)
def cal_data_path(testdata_path: Path) -> Path:
    """Path to test data."""
    return testdata_path / "cal"


@pytest.fixture(scope="session", autouse=True)
def alanmode_data_path(testdata_path: Path) -> Path:
    """Path to test data."""
    return testdata_path / "alanmode"


@pytest.fixture(scope="session", autouse=True)
def anl_data_path(testdata_path: Path) -> Path:
    """Path to test data."""
    return testdata_path / "analysis"


@pytest.fixture(scope="session", autouse=True)
def sim_data_path(testdata_path: Path) -> Path:
    """Path to test data."""
    return testdata_path / "sim"


@pytest.fixture(scope="session")
def cal_data(cal_data_path: Path) -> Path:
    return cal_data_path / "Receiver01_25C_2019_11_26_040_to_200MHz"


@pytest.fixture(scope="session", autouse=True)
def tmpdir(tmp_path_factory) -> Path:
    return tmp_path_factory.mktemp("edges")


@pytest.fixture(scope="session")
def caldef(cal_data: Path):
    return calobsdef.CalObsDefEDGES2.from_standard_layout(cal_data)


@pytest.fixture(scope="session")
def calobs(caldef: calobsdef.CalObsDefEDGES2):
    return CalibrationObservation.from_edges2_caldef(
        caldef,
        f_low=50 * un.MHz,
        f_high=100 * un.MHz,
    )


@pytest.fixture(scope="session")
def calibrator(calobs: CalibrationObservation) -> Calibrator:
    return get_calcoeffs_iterative(calobs)


def make_gsd_ones(freqs, ntime: int = 10, data_unit: str = "uncalibrated"):
    nload, npol, nfreq = 1, 1, freqs.size
    if data_unit == "power":
        nload = 3

    times = np.linspace(2459856, 2459857, ntime + 1)[:-1, None]
    if nload == 3:
        times = np.hstack((times,) * 3)

    return GSData(
        data=np.ones((nload, npol, ntime, nfreq)),
        freqs=freqs,
        times=Time(times, format="jd", scale="utc"),
        telescope=const.KNOWN_TELESCOPES["edges-low"],
        loads=("ant",) if nload == 1 else ("p0", "p1", "p2"),
        data_unit=data_unit,
    )


@pytest.fixture(scope="session")
def gsd_ones():
    freqs = np.linspace(50, 100, 26) * un.MHz
    return make_gsd_ones(freqs, ntime=10)


@pytest.fixture(scope="session")
def gsd_ones_power():
    freqs = np.linspace(50, 100, 26) * un.MHz
    return make_gsd_ones(freqs, ntime=10, data_unit="power")


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
    with pytest.warns(UserWarning, match="Auxiliary measurements cannot be binned"):
        return lstbin.lst_bin(
            mock,
            binsize=0.02,
            first_edge=mock.lsts.min().hour,
            max_edge=mock.lsts.max().hour,
        )


@pytest.fixture(scope="session")
def mock_season() -> list[GSData]:
    """A mock 'season' with three days."""
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


@pytest.fixture(scope="session")
def identity_calibrator(ideal_calobs: CalibrationObservation):
    fq = ideal_calobs.freqs
    return Calibrator(
        freqs=fq,
        Tsca=np.ones(fq.size),
        Toff=np.zeros(fq.size),
        Tunc=np.zeros(fq.size),
        Tcos=np.zeros(fq.size),
        Tsin=np.zeros(fq.size),
        receiver_s11=np.zeros(fq.size, dtype=complex),
    )


@pytest.fixture(scope="session")
def ideal_calobs():
    """An idealized calibration observation with S11's of zero."""
    fqs = edges_raw_freqs(f_low=50 * un.MHz, f_high=100 * un.MHz)[::8]

    gsd = make_gsd_ones(fqs, ntime=1, data_unit="uncalibrated")

    def make_load():
        return Load(
            spectrum=LoadSpectrum(q=gsd, temp_ave=1.0 * un.K),
            s11=CalibratedS11(s11=np.zeros(fqs.size, dtype=complex), freqs=fqs),
        )

    return CalibrationObservation(
        receiver=CalibratedS11(s11=np.zeros(fqs.size, dtype=complex), freqs=fqs),
        loads={
            "ambient": make_load(),
            "hot_load": make_load(),
            "open": make_load(),
            "short": make_load(),
        },
    )
