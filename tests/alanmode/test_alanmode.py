"""Unit-tests of functions in alanmode."""

from pathlib import Path

import numpy as np
import pytest
from astropy import units as un
from astropy.time import Time
from pygsdata import KNOWN_TELESCOPES, GSData
from read_acq.gsdata import write_gsdata_to_acq

from edges import alanmode as am
from edges.frequencies import edges_raw_freqs


def test_read_write_spec_loop(alanmode_data_path: Path, tmpdir: Path):
    spamb = am.read_spec_txt(
        alanmode_data_path / "edges3-data-for-alan-comparison/spambient.txt"
    )

    am.write_spec_txt(
        spamb.freqs,
        n=int(np.mean(spamb.nsamples)),
        spec=spamb.data.squeeze(),
        fname=tmpdir / "spamb.txt",
    )
    spamb2 = am.read_spec_txt(tmpdir / "spamb.txt")
    np.testing.assert_allclose(spamb.freqs, spamb2.freqs)
    np.testing.assert_allclose(spamb.data, spamb2.data)
    np.testing.assert_allclose(spamb.nsamples, spamb2.nsamples)


NTIME = 24
FREQS = edges_raw_freqs()


@pytest.fixture(scope="module")
def unity_acq(tmpdir):
    data = np.ones((3, 1, NTIME, len(FREQS))) / 1000.0
    data[0] *= 4.0
    data[2] *= 4.0

    times = np.linspace(2459856, 2459857, NTIME + 1)[:-1]
    times = np.array([times, times + 0.1, times + 0.2]).T
    times = Time(times, format="jd", scale="utc")

    data = GSData(
        data=data,
        freqs=FREQS,
        times=times,
        telescope=KNOWN_TELESCOPES["edges-low"],
        auxiliary_measurements={
            "adcmax": np.zeros_like(times),
            "adcmin": np.zeros_like(times),
        },
    )
    write_gsdata_to_acq(data, tmpdir / "unity.acq")

    return tmpdir / "unity.acq"


class TestACQPlot7AMoon:
    def test_unity_default(self, unity_acq):
        meanspec = am.acqplot7amoon(
            unity_acq, fstart=0, fstop=np.inf, smooth=0, tload=300, tcal=1000
        )

        assert meanspec.nfreqs == FREQS.size
        assert meanspec.nsamples.max() == NTIME
        np.testing.assert_allclose(meanspec.data, 1300)

    def test_unity_one_hour(self, unity_acq):
        meanspec = am.acqplot7amoon(
            unity_acq,
            fstart=0,
            fstop=np.inf,
            smooth=0,
            tload=300,
            tcal=1000,
            tstart=12.0,
            tstop=12.0,
        )

        assert meanspec.nfreqs == FREQS.size
        assert meanspec.nsamples.max() == 1
        np.testing.assert_allclose(meanspec.data, 1300)

    def test_unity_delaystart(self, unity_acq):
        meanspec = am.acqplot7amoon(
            unity_acq,
            fstart=0,
            fstop=np.inf,
            smooth=0,
            tload=300,
            tcal=1000,
            delaystart=1.0,  # second
        )

        assert meanspec.nfreqs == FREQS.size
        assert meanspec.nsamples.max() == NTIME - 1
        np.testing.assert_allclose(meanspec.data, 1300)

    def test_unity_smooth(self, unity_acq):
        meanspec = am.acqplot7amoon(
            unity_acq,
            fstart=0,
            fstop=np.inf,
            smooth=8,
            tload=300,
            tcal=1000,
        )

        assert meanspec.nfreqs == FREQS.size // 8
        assert np.isclose(meanspec.nsamples.mean(), NTIME * 8, rtol=0.08)
        np.testing.assert_allclose(meanspec.data, 1300)


class TestCorrcsv:
    def test_zero_cable_length(self):
        freq = np.linspace(50, 100, 101) * un.MHz
        s11 = np.zeros(len(freq), dtype=complex)

        corr = am.corrcsv(freq=freq, s11=s11, cablen=0, cabdiel=0, cabloss=0)
        np.testing.assert_allclose(corr, 0, atol=1e-15)
