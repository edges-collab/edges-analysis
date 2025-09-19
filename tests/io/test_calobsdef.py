from pathlib import Path

import pytest

from edges.io import calobsdef


@pytest.fixture(scope="module")
def test_dir(tmp_path_factory):
    return mock_calobs_dir(tmp_path_factory)


@pytest.fixture
def mock_calobs_dir(tmp_path_factory) -> Path:
    # Create an ideal observation file using tmp_path_factory
    path_list = ["Spectra", "Resistance", "S11"]
    s11_list = [
        "Ambient01",
        "AntSim301",
        "HotLoad01",
        "LongCableOpen01",
        "LongCableShorted01",
        "ReceiverReading01",
        "ReceiverReading02",
        "SwitchingState01",
        "SwitchingState02",
    ]
    root_dir = tmp_path_factory.mktemp("Test_Obs")
    obs_dir = root_dir / "Receiver01_25C_2020_01_01_010_to_200MHz"
    obs_dir.mkdir()
    note = obs_dir / "Notes.txt"
    note.touch()
    dlist = []
    slist = []
    for i, p in enumerate(path_list):
        dlist.append(obs_dir / p)
        dlist[i].mkdir()
        if p == "Resistance":
            print("Making Resistance files")
            file_list = [
                "Ambient",
                "AntSim3",
                "HotLoad",
                "LongCableOpen",
                "LongCableShorted",
            ]
            for filename in file_list:
                name1 = f"{filename}_01_2020_001_01_01_01_lab.csv"
                file1 = dlist[i] / name1
                file1.touch()
        elif p == "S11":
            print("Making S11 files")
            for k, s in enumerate(s11_list):
                slist.append(dlist[i] / s)
                slist[k].mkdir()
                if s[:-2] == "ReceiverReading":
                    file_list = ["ReceiverReading", "Match", "Open", "Short"]
                elif s[:-2] == "SwitchingState":
                    file_list = [
                        "ExternalOpen",
                        "ExternalMatch",
                        "ExternalShort",
                        "Match",
                        "Open",
                        "Short",
                    ]
                else:
                    file_list = ["External", "Match", "Open", "Short"]
                for filename in file_list:
                    name1 = f"{filename}01.s1p"
                    name2 = filename + "02.s1p"
                    file1 = slist[k] / name1
                    file1.write_text(
                        "# Hz S RI R 50\n"
                        "40000000        0.239144887761343       0.934085904901478\n"
                        "40000000        0.239144887761343       0.934085904901478"
                    )
                    file2 = slist[k] / name2
                    file2.write_text(
                        "# Hz S RI R 50\n"
                        "40000000        0.239144887761343       0.934085904901478\n"
                        "40000000        0.239144887761343       0.934085904901478"
                    )

        elif p == "Spectra":
            print("Making Spectra files")
            file_list = [
                "Ambient",
                "AntSim3",
                "HotLoad",
                "LongCableOpen",
                "LongCableShorted",
            ]
            for filename in file_list:
                name1 = filename + "_01_2020_001_01_01_01_lab.acq"
                file1 = dlist[i] / name1
                file1.touch()
    return obs_dir


# function to make observation object
@pytest.fixture(scope="session")
def calio(mock_calobs_dir):
    return calobsdef.CalObsDefEDGES2.from_standard_layout(mock_calobs_dir)


class TestFromStandardLayout:
    """Tests of the from_standard_layout factory method."""

    def test_bad_dirname_obs(self, mock_calobs_dir):
        # test that incorrect directories fail

        test_dir = mock_calobs_dir
        base = test_dir.parent

        wrong_dir = base / "Receiver_2020_01_01_010_to_200MHz"
        with pytest.raises(FileNotFoundError, match="does not exist"):
            calobsdef.CalObsDefEDGES2.from_standard_layout(rootdir=wrong_dir)

    def test_run_num_not_exist(self, datadir: Path):
        direc = datadir / "Receiver01_25C_2019_11_26_040_to_200MHz"
        with pytest.warns(UserWarning, match="using ReceiverReading01"):
            calobsdef.CalObsDefEDGES2.from_standard_layout(rootdir=direc, run_num=3)
