"""Test the loss module."""

from pathlib import Path

import numpy as np
import pytest
from astropy import units as un

from edges.analysis import loss
from edges.cal.s11.base import CalibratedS11


def test_no_band():
    with pytest.raises(ValueError, match="you must provide 'band'"):
        loss.ground_loss(filename=":", freq=np.linspace(50, 100, 100))


class TestLow2BalunConnectorLoss:
    @pytest.mark.parametrize("use_approx_eps0", [True, False])
    def test_happy_path(self, use_approx_eps0):
        fq = np.linspace(50, 100, 51) * un.MHz
        ants11 = np.ones(51) / 2

        bcloss = loss.low2_balun_connector_loss(
            fq, ants11, use_approx_eps0=use_approx_eps0
        )

        assert bcloss.shape == fq.shape

    def test_other_s11_inputs(self, tmp_path: Path):
        s11 = CalibratedS11(
            freqs=np.linspace(50, 100, 100) * un.MHz, s11=np.zeros(100, dtype=complex)
        )
        fq = np.linspace(50, 100, 100) * un.MHz

        # S11Model object
        bcloss = loss.low2_balun_connector_loss(fq, ants11=s11)
        assert bcloss.shape == fq.shape

        # Hickled file
        s11.write(tmp_path / "tmp-ant.h5")
        bcloss2 = loss.low2_balun_connector_loss(fq, tmp_path / "tmp-ant.h5")
        assert np.allclose(bcloss, bcloss2)
