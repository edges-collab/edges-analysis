import pytest
from astropy import units as un
from astropy.table import QTable

from edges.io import CalObsDefEDGES3, templogs


@pytest.fixture(scope="module")
def templog_table(smallcaldef_edges3: CalObsDefEDGES3) -> QTable:
    with pytest.warns(UserWarning, match="Error parsing temperature log entry"):
        # THere is a single entry that is corrupted, and this should not make the
        # reader break.
        return templogs.read_temperature_log(smallcaldef_edges3.ambient.templog)


def test_temperature_read(templog_table: QTable):
    assert "time" in templog_table.columns
    assert "hot_load_temperature" in templog_table.columns
    assert "amb_load_temperature" in templog_table.columns
    assert "front_end_temperature" in templog_table.columns
    assert "inner_box_temperature" in templog_table.columns
    assert "thermal_control" in templog_table.columns
    assert "battery_voltage" in templog_table.columns
    assert "pr59_current" in templog_table.columns

    assert all(
        templog_table[col].unit == un.K
        for col in templog_table.columns
        if col.endswith("temperature")
    )


def test_get_mean_temperature(templog_table: QTable):
    mean_temp = templogs.get_mean_temperature(templog_table, load="amb")
    assert mean_temp.unit == un.K
    assert mean_temp.value == pytest.approx(300, abs=15)

    mean_temp0 = templogs.get_mean_temperature(
        templog_table, load="amb", start_time=templog_table["time"].min()
    )
    assert mean_temp0 == mean_temp

    mean_temp0 = templogs.get_mean_temperature(
        templog_table, load="open", end_time=templog_table["time"].max()
    )
    assert mean_temp0 == mean_temp

    mean_temp0 = templogs.get_mean_temperature(templog_table, load="hot")
    assert mean_temp0 > mean_temp

    with pytest.raises(ValueError, match="Unknown load fake"):
        templogs.get_mean_temperature(templog_table, load="fake")
