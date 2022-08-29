import pytest

import numpy as np

from edges_analysis.calibration import loss


def test_no_band():
    with pytest.raises(ValueError):
        loss.ground_loss(":", freq=np.linspace(50, 100, 100))
