"""Round-trip specal.txt -> Calibrator -> h5 -> Calibrator.from_calfile."""

from pathlib import Path

from edges.alanmode import read_specal
from edges.cal import Calibrator

SPECAL_TXT = (
    Path(__file__).parent.parent
    / "data"
    / "alanmode"
    / "edges3-2022-316-alan"
    / "specal.txt"
)


def test_specal_txt_write_h5_read_from_calfile(tmp_path):
    """read_specal builds a Calibrator; write h5; from_calfile restores it."""
    cal = read_specal(SPECAL_TXT, t_load=300.0, t_load_ns=1000.0)

    h5_path = tmp_path / "specal.h5"
    cal.write(h5_path)
    loaded = Calibrator.from_calfile(h5_path)

    assert loaded == cal
