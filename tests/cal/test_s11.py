import numpy as np
import pytest
from astropy import units as u

from edges import modelling as mdl
from edges.cal import s11
from edges.cal.s11.base import CalibratedS11
from edges.cal.s11.s11model import S11ModelParams
from edges.io import calobsdef, SParams
from astropy import units as un

class TestS11Model:    
    @pytest.mark.parametrize("cmod", (mdl.ComplexMagPhaseModel, mdl.ComplexRealImagModel))
    def test_use_spline(self, cmod):
        freq = np.linspace(50, 100, 100) * un.MHz
        mfreq = 75 * un.MHz
        raw_data = (
            (freq / mfreq) ** -2.5
            + 1j * (freq / mfreq) ** 0.5
            + np.random.normal(scale=0.1, size=100)
        )

        sp = CalibratedS11(freqs=freq, s11=raw_data)
        sp_smooth = sp.smoothed(S11ModelParams(use_spline=True, complex_model_type=cmod))
            
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


# def test_get_k_matrix():
#     freq = np.linspace(50, 100, 100) * u.MHz
#     mfreq = 75 * u.MHz
#     raw_data = (
#         (freq / mfreq).value ** -2.5
#         + 1j * (freq / mfreq).value ** 0.5
#         + np.random.normal(scale=0.1, size=100)
#     )

#     int_switch = s11.InternalSwitch(
#         s11_data=raw_data, s12_data=raw_data, s22_data=raw_data, freq=freq
#     )
#     rcv = s11.Receiver(raw_s11=raw_data, freq=freq)
#     s11m = s11.LoadS11(freq=freq, raw_s11=raw_data, internal_switch=int_switch)

#     K = s11m.get_k_matrix(rcv)
#     assert np.array(K).shape == (4, freq.size)
