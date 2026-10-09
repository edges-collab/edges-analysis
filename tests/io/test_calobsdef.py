import shutil
from pathlib import Path

import pytest

from edges.io import calobsdef


@pytest.fixture(scope="module")
def test_dir(tmp_path_factory):
    return mock_calobs_dir(tmp_path_factory)


@pytest.fixture(scope="module")
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
                    file_list = [
                        "ReceiverReading",
                        "Match",
                        "Open",
                        "Short",
                        "metadata.yaml",
                    ]
                elif s[:-2] == "SwitchingState":
                    file_list = [
                        "ExternalOpen",
                        "ExternalMatch",
                        "ExternalShort",
                        "Match",
                        "Open",
                        "Short",
                        "metadata.yaml",
                    ]
                else:
                    file_list = ["External", "Match", "Open", "Short"]
                for filename in file_list:
                    if filename == "metadata.yaml":
                        file1 = slist[k] / filename
                        file1.write_text(
                            "calkit: AGILENT_85033E\n"
                            "calkit_match_resistance: 49.98  # ohm\n"
                        )
                        continue

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
@pytest.fixture(scope="module")
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


def test_metadata(calio: calobsdef.CalObsDefEDGES2):
    with pytest.raises(TypeError, match="Value must be a dict"):
        calobsdef.InternalSwitch(
            internal=calio.internal_switch.internal,
            external=calio.internal_switch.external,
            metadata=3,
        )

    isw = calobsdef.InternalSwitch(
        internal=calio.internal_switch.internal,
        external=calio.internal_switch.external,
        metadata={"external_calkit": "calkit-name"},
    )

    assert isw.temperature is None
    assert isw.external_calkit == "calkit-name"


@pytest.mark.xfail(
    strict=True,
    reason=(
        "The EDGES-2 standard layout names the hot load 'HotLoad', but the default "
        "semi-rigid cable S-parameter file is only set for a load named 'hot_load', "
        "so no hot-load cable loss is ever applied; fix pending (result-changing)"
    ),
)
def test_standard_layout_hot_load_has_semirigid_sparams(
    calio: calobsdef.CalObsDefEDGES2,
):
    assert calio.hot_load.sparams_file is not None


def _reverse_glob(monkeypatch: pytest.MonkeyPatch) -> None:
    """Make Path.glob yield its matches in reverse-sorted order."""
    original = Path.glob

    def _glob(self, pattern, *args, **kwargs):
        return iter(sorted(original(self, pattern, *args, **kwargs), reverse=True))

    monkeypatch.setattr(Path, "glob", _glob)


class TestGlobOrderIndependence:
    """File discovery must not depend on the order the filesystem lists files."""

    def test_standard_layout(self, datadir: Path, monkeypatch):
        direc = datadir / "Receiver01_25C_2019_11_26_040_to_200MHz"
        forward = calobsdef.CalObsDefEDGES2.from_standard_layout(direc)
        _reverse_glob(monkeypatch)
        backward = calobsdef.CalObsDefEDGES2.from_standard_layout(direc)

        assert backward == forward
        assert len(forward.hot_load.spectra) == 4
        for load in forward.loads.values():
            assert load.spectra == sorted(load.spectra)

    def test_calkit_fallback_repeat(self, datadir: Path, monkeypatch):
        direc = datadir / "Receiver01_25C_2019_11_26_040_to_200MHz/S11/Ambient01"
        with pytest.warns(UserWarning, match="using Open01.s1p"):
            forward = calobsdef.CalkitFileSpec.from_edges2_layout(direc, repeat_num=5)
        _reverse_glob(monkeypatch)
        with pytest.warns(UserWarning, match="using Open01.s1p"):
            backward = calobsdef.CalkitFileSpec.from_edges2_layout(direc, repeat_num=5)
        assert backward == forward


class TestMissingFiles:
    """Missing files raise a clear FileNotFoundError (not AssertionError etc.)."""

    @pytest.fixture
    def obsdir(self, mock_calobs_dir: Path, tmp_path: Path) -> Path:
        out = tmp_path / "obs"
        shutil.copytree(mock_calobs_dir, out)
        return out

    @pytest.mark.parametrize("sub", ["Resistance", "S11", "Spectra"])
    def test_missing_subdir(self, obsdir: Path, sub: str):
        shutil.rmtree(obsdir / sub)
        with pytest.raises(FileNotFoundError, match=f"no '{sub}' subdirectory"):
            calobsdef.CalObsDefEDGES2.from_standard_layout(obsdir)
        with pytest.raises(FileNotFoundError, match=f"no '{sub}' subdirectory"):
            calobsdef.LoadDefEDGES2.from_standard_layout(obsdir, "Ambient")

    def test_missing_resistance(self, obsdir: Path):
        for fl in (obsdir / "Resistance").glob("HotLoad_*.csv"):
            fl.unlink()
        with pytest.raises(FileNotFoundError, match="HotLoad_01_"):
            calobsdef.CalObsDefEDGES2.from_standard_layout(obsdir)

    def test_missing_spectra(self, obsdir: Path):
        for fl in (obsdir / "Spectra").glob("Ambient_*.acq"):
            fl.unlink()
        with pytest.raises(FileNotFoundError, match="Ambient_01_"):
            calobsdef.LoadDefEDGES2.from_standard_layout(obsdir, "Ambient")

    def test_multiple_resistance_files_warn(self, obsdir: Path):
        first = obsdir / "Resistance" / "Ambient_01_2020_001_01_01_01_lab.csv"
        (obsdir / "Resistance" / "Ambient_01_2020_002_01_01_01_lab.csv").touch()
        with pytest.warns(UserWarning, match="Found 2 Resistance files for Ambient"):
            load = calobsdef.LoadDefEDGES2.from_standard_layout(obsdir, "Ambient")
        assert load.thermistor == first

    def test_no_open_files(self, obsdir: Path):
        direc = obsdir / "S11" / "Ambient01"
        for fl in direc.glob("Open*.s1p"):
            fl.unlink()
        with pytest.raises(FileNotFoundError, match="Open"):
            calobsdef.CalkitFileSpec.from_edges2_layout(direc)

    def test_no_other_repeat_allowed(self, obsdir: Path):
        direc = obsdir / "S11" / "Ambient01"
        with pytest.raises(FileNotFoundError, match="Could not find Open05"):
            calobsdef.CalkitFileSpec.from_edges2_layout(
                direc, repeat_num=5, allow_other=False
            )

    def test_no_receiver_reading(self, obsdir: Path):
        for direc in (obsdir / "S11").glob("ReceiverReading*"):
            shutil.rmtree(direc)
        with pytest.raises(FileNotFoundError, match="ReceiverReading"):
            calobsdef.CalObsDefEDGES2.from_standard_layout(obsdir)

    def test_no_switching_state(self, obsdir: Path):
        for direc in (obsdir / "S11").glob("SwitchingState*"):
            shutil.rmtree(direc)
        with pytest.raises(FileNotFoundError, match="SwitchingState"):
            calobsdef.CalObsDefEDGES2.from_standard_layout(obsdir)


def test_missing_receiver_reading_warns_loads_switched(datadir: Path):
    direc = datadir / "Receiver01_25C_2019_11_26_040_to_200MHz"
    with pytest.warns(UserWarning, match="loads .* will also use run 01"):
        caldef = calobsdef.CalObsDefEDGES2.from_standard_layout(direc, run_num=3)
    assert caldef.run_num == 1
    for load in caldef.loads.values():
        assert load.thermistor.name.split("_")[1] == "01"
