from pathlib import Path

import numpy as np

from edges.io import thermistor


class TestOldStyleRead:
    def test_read_old_header(self, datadir: Path):
        header, _nlines = thermistor._read_old_style_csv_header(
            datadir / "old_resistance_file.csv"
        )

        assert header["Start Time"] == "9/14/2017 2:16:45 PM"

    def test_read_old(self, datadir: Path):
        fl = datadir / "old_resistance_file.csv"
        data = thermistor.read_old_style_csv(fl)

        assert len(data) == 11
        assert len(data.columns) == 12
        assert not np.any(np.isnan(data["load_resistance"]))


class TestNewStyleRead:
    def test_read_new(self, datadir: Path):
        fl = datadir / (
            "Receiver01_25C_2019_11_26_040_to_200MHz/Resistance/"
            "Ambient_01_2019_329_16_02_35_lab.csv"
        )
        data = thermistor.read_new_style_csv(fl)
        assert len(data) == 9
        assert len(data.columns) == 11
