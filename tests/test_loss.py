"""Test the loss module."""

import hickle
import numpy as np
import pytest
from astropy import units as un

from edges_analysis.calibration import loss

from . import DATA_PATH


def test_no_band():
    with pytest.raises(ValueError, match="you must provide 'band'"):
        loss.ground_loss(filename=":", freq=np.linspace(50, 100, 100))


class TestLow2BalunConnectorLoss:
    @pytest.mark.parametrize("use_approx_eps0", [True, False])
    def test_happy_path(self, use_approx_eps0):
        fq = np.linspace(50, 100, 51) * un.MHz
        ants11 = np.ones(51)

        bcloss = loss.low2_balun_connector_loss(
            fq, ants11, use_approx_eps0=use_approx_eps0
        )

        assert bcloss.shape == fq.shape

    def test_other_s11_inputs(self):
        fq = np.linspace(50, 100, 51) * un.MHz
        s11file = DATA_PATH / "2015_ants11_modelled_redone.h5"
        bcloss = loss.low2_balun_connector_loss(fq, ants11=s11file)
        assert bcloss.shape == fq.shape

        bcloss2 = loss.low2_balun_connector_loss(fq, hickle.load(s11file))
        assert np.allclose(bcloss, bcloss2)
