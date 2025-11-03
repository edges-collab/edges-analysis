from pathlib import Path

import numpy as np
import pytest
from astropy import units as un

from edges.io.vna import SParams


def test_s1p_read(datadir: Path):
    fl = (
        datadir / "Receiver01_25C_2019_11_26_040_to_200MHz/S11/Ambient01/External01.s1p"
    )
    s1p = SParams.from_s1p_file(fl)

    assert np.all(np.iscomplex(s1p.s11))
    assert len(s1p.s11) == len(s1p.freq)


def test_s1_read_db(datadir: Path):
    fl = datadir / "s11_db.s1p"
    s1p = SParams.from_s1p_file(fl)

    assert np.all(np.iscomplex(s1p.s11))
    assert len(s1p.s11) == len(s1p.freq)


def test_s2p_read(datadir: Path):
    fl = datadir / "example.s2p"
    s1p = SParams.from_s1p_file(fl)
    assert s1p.s12 is not None
    assert s1p.s21 is not None
    assert s1p.s22 is not None


def test_bad_s11_input_to_vna():
    with pytest.raises(ValueError, match="s-parameters must be complex"):
        SParams(freq=np.linspace(50, 100, 100) * un.MHz, s11=np.linspace(0, 1, 100))

    with pytest.raises(ValueError, match="Shape of s11 does not match"):
        SParams(freq=np.linspace(50, 100, 100) * un.MHz, s11=np.linspace(0, 1, 70) + 0j)
