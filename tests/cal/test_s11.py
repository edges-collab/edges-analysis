import numpy as np
import pytest
from astropy import units as un

from edges.cal import sparams as sp


class TestReceiverS11Model:
    def test_receiver(self, caldef):
        rcv = sp.get_gamma_receiver_from_filespec(caldef)
        assert np.iscomplexobj(rcv.s11)
        assert np.all(np.abs(rcv.s11) < 1)
        assert len(np.unique(rcv.s11)) > 25


class TestCalkitReadings:
    def test_different_freqs_in_standards(self):
        freqs = np.linspace(50, 100, 100) * un.MHz
        s = np.linspace(0, 1, 100) + 0j

        vna1 = sp.ReflectionCoefficient(freqs=freqs, reflection_coefficient=s)
        vna2 = sp.ReflectionCoefficient(freqs=freqs[:80], reflection_coefficient=s[:80])

        with pytest.raises(
            ValueError, match="short standard does not have same frequencies"
        ):
            sp.CalkitReadings(open=vna1, short=vna2, match=vna1)

        with pytest.raises(
            ValueError, match="match standard does not have same frequencies"
        ):
            sp.CalkitReadings(open=vna1, short=vna1, match=vna2)

        sr = sp.CalkitReadings(open=vna1, short=vna1, match=vna1)
        assert np.all(sr.freqs == vna1.freqs)


class TestSparamsFromSemiRigid:
    def test_2017_semi_rigid(self):
        hlc = sp.read_semi_rigid_cable_sparams_file(
            path=":semi_rigid_s_parameters_2017.txt"
        )
        assert hlc.s12.dtype == complex
