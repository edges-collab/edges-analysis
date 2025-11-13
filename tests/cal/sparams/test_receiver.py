import numpy as np

from edges.cal import sparams as sp


class TestReceiver:
    def test_receiver(self, caldef):
        rcv = sp.get_gamma_receiver_from_filespec(caldef)
        assert np.iscomplexobj(rcv.s11)
        assert np.all(np.abs(rcv.s11) < 1)
        assert len(np.unique(rcv.s11)) > 25
