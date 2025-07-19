import pytest
from astropy import units as un

from edges.io import CalObsDefEDGES3, templogs


def test_temperature_read(smallcaldef_edges3: CalObsDefEDGES3):
    temp_table = templogs.read_temperature_log(smallcaldef_edges3.templog)

    assert "time" in temp_table.columns
    assert "hot_load_temperature" in temp_table.columns
    assert "amb_load_temperature" in temp_table.columns
    assert "front_end_temperature" in temp_table.columns
    assert "inner_box_temperature" in temp_table.columns
    assert "thermal_control" in temp_table.columns
    assert "battery_voltage" in temp_table.columns
    assert "pr59_current" in temp_table.columns

    assert all(
        temp_table[col].unit == un.K
        for col in temp_table.columns
        if col.endswith("temperature")
    )


def test_get_mean_temperature(smallcaldef_edges3: CalObsDefEDGES3):
    temp_table = templogs.read_temperature_log(smallcaldef_edges3.templog)
    mean_temp = templogs.get_mean_temperature(temp_table, load="amb")
    assert mean_temp.unit == un.K
    assert mean_temp.value == pytest.approx(300, abs=15)

    mean_temp0 = templogs.get_mean_temperature(
        temp_table, load="amb", start_time=temp_table["time"].min()
    )
    assert mean_temp0 == mean_temp

    mean_temp0 = templogs.get_mean_temperature(
        temp_table, load="open", end_time=temp_table["time"].max()
    )
    assert mean_temp0 == mean_temp

    mean_temp0 = templogs.get_mean_temperature(temp_table, load="hot")
    assert mean_temp0 > mean_temp

    with pytest.raises(ValueError, match="Unknown load fake"):
        templogs.get_mean_temperature(temp_table, load="fake")
