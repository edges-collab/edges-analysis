import numpy as np
import pytest
from astropy import units as u
from astropy import units as un

from edges import modeling as mdl
from edges.cal import s11
from edges.cal.s11.base import CalibratedS11
from edges.cal.s11.s11model import S11ModelParams
from edges.io import SParams


class TestS11Model:
    @pytest.mark.parametrize(
        "cmod", [mdl.ComplexMagPhaseModel, mdl.ComplexRealImagModel]
    )
    def test_use_spline(self, cmod):
        rng = np.random.default_rng()
        freq = np.linspace(50, 100, 100) * un.MHz
        mfreq = 75 * un.MHz
        raw_data = (
            (freq / mfreq) ** -2.5
            + 1j * (freq / mfreq) ** 0.5
            + rng.normal(scale=0.1, size=100)
        )

        sp = CalibratedS11(freqs=freq, s11=raw_data)
        sp_smooth = sp.smoothed(
            S11ModelParams(use_spline=True, complex_model_type=cmod)
        )

        assert np.allclose(sp_smooth.s11, raw_data)


class TestReceiverS11Model:
    def test_receiver(self, caldef):
        rcv = s11.CalibratedS11.from_receiver_filespec(caldef.receiver_s11)
        assert np.iscomplexobj(rcv.s11)
        assert np.all(np.abs(rcv.s11) < 1)
        assert len(np.unique(rcv.s11)) > 25


class TestStandardsReadings:
    def test_different_freqs_in_standards(self):
        freq = np.linspace(50, 100, 100) * u.MHz
        s = np.linspace(0, 1, 100) + 0j

        vna1 = SParams(freq=freq, s11=s)
        vna2 = SParams(freq=freq[:80], s11=s[:80])

        with pytest.raises(
            ValueError, match="short standard does not have same frequencies"
        ):
            s11.StandardsReadings(open=vna1, short=vna2, match=vna1)

        with pytest.raises(
            ValueError, match="match standard does not have same frequencies"
        ):
            s11.StandardsReadings(open=vna1, short=vna1, match=vna2)

        sr = s11.StandardsReadings(open=vna1, short=vna1, match=vna1)
        assert np.all(sr.freq == vna1.freq)


class TestSparamsFromSemiRigid:
    def test_2017_semi_rigid(self):
        hlc = s11.CalibratedSParams.from_hot_load_semi_rigid(
            path=":semi_rigid_s_parameters_2017.txt"
        )
        assert hlc.s12.dtype == complex
