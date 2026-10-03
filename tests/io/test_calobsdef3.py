import re
import shutil
from itertools import pairwise
from pathlib import Path

import numpy as np
import pytest

from edges.io import CalObsDefEDGES3, calobsdef3, templogs


@pytest.fixture(scope="module")
def mockroot(datadir: Path) -> Path:
    return datadir / "edges3-mock-root"


def test_all_files_present(smallcaldef_edges3: CalObsDefEDGES3):
    # Really we just needed to make the thing to test it.
    print(smallcaldef_edges3)


def test_default_templog(smallcaldef_edges3: CalObsDefEDGES3, mockroot: Path):
    for load in smallcaldef_edges3.loads.values():
        assert load.templog == mockroot / "temperature_logger/temperature.log"


@pytest.mark.parametrize(
    ("year", "day", "ddays", "expected"),
    [
        (2023, 70, 1, (2023, 71)),
        (2023, 365, 1, (2024, 1)),
        (2024, 1, -1, (2023, 365)),
        (2024, 365, 1, (2024, 366)),
        (2024, 366, 1, (2025, 1)),
        (2025, 1, -1, (2024, 366)),
    ],
)
def test_shift_day(year: int, day: int, ddays: int, expected: tuple[int, int]):
    assert calobsdef3._shift_day(year, day, ddays) == expected


def _make_s11s(direc: Path, prefix: str, labels=("O", "S", "L")):
    direc.mkdir(parents=True, exist_ok=True)
    for label in labels:
        (direc / f"{prefix}_{label}.s1p").touch()


class TestGetSingleS1PFile:
    def test_exact_day(self, tmp_path: Path):
        _make_s11s(tmp_path, "2023_070_11")
        fl = calobsdef3._get_single_s1p_file(tmp_path, 2023, 70, "O")
        assert fl == tmp_path / "2023_070_11_O.s1p"

    def test_across_year_boundary_forward(self, tmp_path: Path):
        _make_s11s(tmp_path, "2024_001_11")
        fl = calobsdef3._get_single_s1p_file(tmp_path, 2023, 365, "O", allow_closest=1)
        assert fl == tmp_path / "2024_001_11_O.s1p"

    def test_across_year_boundary_backward(self, tmp_path: Path):
        # 2024 is a leap year, so the day before 2025_001 is 2024_366.
        _make_s11s(tmp_path, "2024_366_11")
        fl = calobsdef3._get_single_s1p_file(tmp_path, 2025, 2, "O", allow_closest=2)
        assert fl == tmp_path / "2024_366_11_O.s1p"

    def test_year_dir_template(self, tmp_path: Path):
        _make_s11s(tmp_path / "s11" / "2022", "2022_365_03")
        fl = calobsdef3._get_single_s1p_file(
            tmp_path, 2023, 1, "S", allow_closest=1, s11_dir="s11/{year}"
        )
        assert fl == tmp_path / "s11/2022/2022_365_03_S.s1p"

    def test_allow_closest_is_inclusive(self, tmp_path: Path):
        _make_s11s(tmp_path, "2023_070_11")
        fl = calobsdef3._get_single_s1p_file(tmp_path, 2023, 75, "O", allow_closest=5)
        assert fl == tmp_path / "2023_070_11_O.s1p"

        with pytest.raises(FileNotFoundError, match="within 4 days of 2023_075"):
            calobsdef3._get_single_s1p_file(tmp_path, 2023, 75, "O", allow_closest=4)

    def test_closest_prefers_later_day_on_tie(self, tmp_path: Path):
        _make_s11s(tmp_path, "2023_069_11")
        _make_s11s(tmp_path, "2023_071_11")
        fl = calobsdef3._get_single_s1p_file(tmp_path, 2023, 70, "O", allow_closest=1)
        assert fl == tmp_path / "2023_071_11_O.s1p"

    def test_first_and_last_hour(self, tmp_path: Path):
        _make_s11s(tmp_path, "2023_070_02")
        _make_s11s(tmp_path, "2023_070_11")
        get = calobsdef3._get_single_s1p_file
        assert get(tmp_path, 2023, 70, "O").name == "2023_070_02_O.s1p"
        assert get(tmp_path, 2023, 70, "O", hour="last").name == "2023_070_11_O.s1p"
        assert get(tmp_path, 2023, 70, "O", hour=11).name == "2023_070_11_O.s1p"

    def test_ambiguous_hour(self, tmp_path: Path):
        _make_s11s(tmp_path, "2023_070_02")
        _make_s11s(tmp_path, "2023_070_11")
        with pytest.raises(OSError, match="More than one file"):
            calobsdef3._get_single_s1p_file(tmp_path, 2023, 70, "O", hour=None)


def _split_log(log: Path, outdir: Path, cuts: list[slice]) -> list[Path]:
    """Split a temperature log into several (possibly overlapping) files."""
    lines = log.read_text(encoding="latin-1").splitlines(keepends=True)
    starts = [i for i, ln in enumerate(lines) if re.match(r"^\d{4}_\d{3}_\d{2}$", ln)]
    entries = ["".join(lines[a:b]) for a, b in pairwise([*starts, len(lines)])]
    outdir.mkdir(parents=True, exist_ok=True)
    out = []
    for i, cut in enumerate(cuts):
        fl = outdir / f"part{i}.log"
        fl.write_text("".join(entries[cut]), encoding="latin-1")
        out.append(fl)
    return out


@pytest.fixture
def custom_layout(tmp_path: Path, mockroot: Path) -> Path:
    """The mock data, rearranged into a non-MRO layout."""
    s11dir = tmp_path / "s11" / "2023"
    s11dir.mkdir(parents=True)
    for fl in mockroot.glob("*.s1p"):
        shutil.copy(fl, s11dir)

    for fl in mockroot.glob("mro/*/2023/*.acq"):
        load = fl.parent.parent.name
        (tmp_path / "spectra" / load).mkdir(parents=True)
        shutil.copy(fl, tmp_path / "spectra" / load)

    _split_log(
        mockroot / "temperature_logger/temperature.log",
        tmp_path / "logs",
        [slice(None, 500), slice(400, None)],
    )
    return tmp_path


class TestCustomLayout:
    def test_custom_dirs(self, custom_layout: Path):
        caldef = CalObsDefEDGES3.from_standard_layout(
            rootdir=custom_layout,
            year=2023,
            day=70,
            s11_dir="s11/{year}",
            spectrum_dir="spectra/{load}",
            templog=["logs/part0.log", custom_layout / "logs/part1.log"],
        )
        assert (
            caldef.receiver_s11.device == custom_layout / "s11/2023/2023_070_11_lna.s1p"
        )
        assert (
            caldef.ambient.s11.external
            == custom_layout / "s11/2023/2023_070_11_amb.s1p"
        )
        assert caldef.hot_load.spectra == [
            custom_layout / "spectra/hot/2023_070_15_39_25_hot.acq"
        ]
        for load in caldef.loads.values():
            assert load.templog == [
                custom_layout / "logs/part0.log",
                custom_layout / "logs/part1.log",
            ]

    def test_no_default_templog(self, custom_layout: Path):
        caldef = CalObsDefEDGES3.from_standard_layout(
            rootdir=custom_layout,
            year=2023,
            day=70,
            s11_dir="s11/{year}",
            spectrum_dir="spectra/{load}",
        )
        assert caldef.ambient.templog is None

    def test_empty_templog_list(self, mockroot: Path):
        caldef = CalObsDefEDGES3.from_standard_layout(
            rootdir=mockroot, year=2023, day=70, templog=[]
        )
        assert caldef.ambient.templog is None

    def test_missing_templog(self, custom_layout: Path):
        with pytest.raises(ValueError, match="templog path does not exist"):
            CalObsDefEDGES3.from_standard_layout(
                rootdir=custom_layout,
                year=2023,
                day=70,
                s11_dir="s11/{year}",
                spectrum_dir="spectra/{load}",
                templog="logs/nonexistent.log",
            )

    def test_split_templogs_match_full_log(self, custom_layout: Path, mockroot: Path):
        caldef = CalObsDefEDGES3.from_standard_layout(
            rootdir=custom_layout,
            year=2023,
            day=70,
            s11_dir="s11/{year}",
            spectrum_dir="spectra/{load}",
            templog=["logs/part0.log", "logs/part1.log"],
        )
        with pytest.warns(UserWarning, match="malformed lines"):
            full = templogs.read_temperature_log(
                mockroot / "temperature_logger/temperature.log"
            )
        with pytest.warns(UserWarning, match="malformed lines"):
            merged = templogs.read_temperature_log(caldef.ambient.templog)

        assert len(merged) == len(full)
        assert np.all(merged["time"] == full["time"])
        np.testing.assert_allclose(
            merged["amb_load_temperature"].value,
            full["amb_load_temperature"].value,
            rtol=0,
            atol=1e-12,
        )

    def test_multiple_spectra(self, custom_layout: Path):
        ambdir = custom_layout / "spectra" / "amb"
        shutil.copy(
            ambdir / "2023_070_12_43_06_amb.acq", ambdir / "2023_070_23_00_00_amb.acq"
        )
        kw = {
            "rootdir": custom_layout,
            "year": 2023,
            "day": 70,
            "s11_dir": "s11/{year}",
            "spectrum_dir": "spectra/{load}",
        }
        with pytest.raises(OSError, match="allow_multiple=True"):
            CalObsDefEDGES3.from_standard_layout(**kw)

        caldef = CalObsDefEDGES3.from_standard_layout(**kw, allow_multiple_spectra=True)
        assert caldef.ambient.spectra == [
            ambdir / "2023_070_12_43_06_amb.acq",
            ambdir / "2023_070_23_00_00_amb.acq",
        ]
        assert len(caldef.hot_load.spectra) == 1

    def test_warn_midnight_continuation(self, custom_layout: Path):
        shortdir = custom_layout / "spectra" / "short"
        shutil.copy(
            shortdir / "2023_070_21_32_02_short.acq",
            shortdir / "2023_071_00_00_02_short.acq",
        )
        with pytest.warns(UserWarning, match="2023_071_00_00_02_short.acq") as rec:
            caldef = CalObsDefEDGES3.from_standard_layout(
                rootdir=custom_layout,
                year=2023,
                day=70,
                s11_dir="s11/{year}",
                spectrum_dir="spectra/{load}",
            )
        assert len(rec) == 1
        assert caldef.short.spectra == [shortdir / "2023_070_21_32_02_short.acq"]


class TestGetSpectrumFiles:
    def _touch(self, root: Path, *names: str):
        for name in names:
            year = name[:4]
            (root / "mro" / "short" / year).mkdir(parents=True, exist_ok=True)
            (root / "mro" / "short" / year / name).touch()

    def test_no_warning_for_later_start(self, tmp_path: Path, recwarn):
        self._touch(
            tmp_path, "2023_070_21_32_02_short.acq", "2023_071_00_05_00_short.acq"
        )
        files = calobsdef3.get_spectrum_files("short", tmp_path, 2023, 70)
        assert [fl.name for fl in files] == ["2023_070_21_32_02_short.acq"]
        assert not [w for w in recwarn if "midnight" in str(w.message)]

    def test_no_files(self, tmp_path: Path):
        self._touch(tmp_path, "2023_071_21_32_02_short.acq")
        with pytest.raises(FileNotFoundError, match="No files found"):
            calobsdef3.get_spectrum_files("short", tmp_path, 2023, 70)

    def test_warn_continuation_across_year(self, tmp_path: Path):
        self._touch(
            tmp_path, "2023_365_22_00_00_short.acq", "2024_001_00_00_05_short.acq"
        )
        with pytest.warns(UserWarning, match="2024_001_00_00_05_short.acq"):
            calobsdef3.get_spectrum_files("short", tmp_path, 2023, 365)


class TestFromFiles:
    @pytest.fixture
    def s11_files(self, mockroot: Path) -> list[Path]:
        return sorted(mockroot.glob("*.s1p"))

    @pytest.fixture
    def spectra(self, mockroot: Path) -> dict[str, list[Path]]:
        return {
            load: sorted(mockroot.glob(f"mro/{load}/2023/*.acq"))
            for load in ("amb", "hot", "open", "short")
        }

    def test_matches_standard_layout(
        self, smallcaldef_edges3: CalObsDefEDGES3, s11_files, spectra, mockroot
    ):
        caldef = CalObsDefEDGES3.from_files(
            s11_files=s11_files,
            spectra=spectra,
            templog=mockroot / "temperature_logger/temperature.log",
        )
        assert caldef == smallcaldef_edges3

    def test_long_load_names_and_single_files(
        self, smallcaldef_edges3: CalObsDefEDGES3, s11_files, spectra, mockroot
    ):
        caldef = CalObsDefEDGES3.from_files(
            s11_files=s11_files,
            spectra={
                "ambient": spectra["amb"][0],
                "hot_load": spectra["hot"],
                "open": spectra["open"],
                "short": spectra["short"],
            },
            templog=mockroot / "temperature_logger/temperature.log",
        )
        assert caldef == smallcaldef_edges3

    def test_multiple_templogs(self, s11_files, spectra, custom_layout: Path):
        logs = sorted((custom_layout / "logs").glob("*.log"))
        caldef = CalObsDefEDGES3.from_files(
            s11_files=s11_files, spectra=spectra, templog=logs
        )
        assert caldef.open.templog == logs

    def test_missing_s11(self, s11_files, spectra):
        s11_files = [fl for fl in s11_files if not fl.stem.endswith("lna_S")]
        with pytest.raises(FileNotFoundError, match="lna_S"):
            CalObsDefEDGES3.from_files(s11_files=s11_files, spectra=spectra)

    def test_duplicate_s11(self, s11_files, spectra, tmp_path: Path):
        extra = tmp_path / "2023_071_02_hot.s1p"
        extra.touch()
        with pytest.raises(ValueError, match="label 'hot'"):
            CalObsDefEDGES3.from_files(s11_files=[*s11_files, extra], spectra=spectra)

    def test_badly_named_s11(self, s11_files, spectra, tmp_path: Path):
        extra = tmp_path / "hot.s1p"
        extra.touch()
        with pytest.raises(ValueError, match="not named like"):
            CalObsDefEDGES3.from_files(s11_files=[*s11_files, extra], spectra=spectra)

    def test_missing_spectra(self, s11_files, spectra):
        del spectra["short"]
        with pytest.raises(FileNotFoundError, match="short"):
            CalObsDefEDGES3.from_files(s11_files=s11_files, spectra=spectra)
