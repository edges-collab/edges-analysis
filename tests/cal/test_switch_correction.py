import pytest

from edges.cal import s11 as s11
from edges.io.calobsdef import SwitchingState


@pytest.fixture(scope="module")
def internal_switch(cal_data) -> SwitchingState:
    return SwitchingState(cal_data / "S11" / "SwitchingState01")


def test_bad_nterms(internal_switch: SwitchingState) -> None:
    with pytest.raises(ValueError, match="n_terms must be >0"):
        s11.InternalSwitch.from_io(internal_switch, n_terms=(1, 1, 0))
