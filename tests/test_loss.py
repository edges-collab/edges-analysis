import pytest

from edges_analysis.analysis import loss
import numpy as np


def test_no_band():
    with pytest.raises(ValueError):
        loss.ground_loss(":", freq=np.linspace(50, 100, 100))
