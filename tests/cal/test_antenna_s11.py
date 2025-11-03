import pytest
from astropy.time import Time

from edges.cal.s11.antenna import _get_closest_s11_time, get_antenna_s11_paths


class TestGetClosestS11Time:
    """Tests of the _get_closest_s11_time method."""

    def _setup_dir(self, tmp_path, year: int = 2015):
        d = tmp_path / "s11"
        if not d.exists():
            d.mkdir()

        # Create a few empty files
        find_these_files = [
            (d / f"{year}_312_00_0.s1p").absolute(),
            (d / f"{year}_312_00_1.s1p").absolute(),
            (d / f"{year}_312_00_2.s1p").absolute(),
            (d / f"{year}_312_00_3.s1p").absolute(),
        ]
        for fl in find_these_files:
            fl.touch()

        return d, find_these_files

    def test_happy_path(self, tmp_path):
        d, find_these_files = self._setup_dir(tmp_path)

        # Find these files
        out = _get_closest_s11_time(
            s11_dir=d,
            time=Time("2015:312:00:00"),
            fileglob="*.s1p",
            dateformat="%Y_%j_%H",
            date_slice=slice(0, 11),
        )
        assert out == sorted(find_these_files)

        # Add some more files
        self._setup_dir(tmp_path, year=2011)
        out = _get_closest_s11_time(
            s11_dir=d,
            time=Time("2015:312:00:00"),
            fileglob="*.s1p",
            dateformat="%Y_%j_%H",
            date_slice=slice(0, 11),
        )
        assert len(out) == 4
        assert out == sorted(find_these_files)

    def test_no_files_exist(self, tmp_path):
        with pytest.raises(FileNotFoundError, match="No files found"):
            _get_closest_s11_time(
                tmp_path,
                time=Time("2015:312:00:00"),
                fileglob="*.s1p",
                dateformat="%Y_%j_%H",
                date_slice=slice(0, 11),
            )

    def test_wrong_number_of_files(self, tmp_path):
        d, find_these_files = self._setup_dir(tmp_path)
        find_these_files[-1].unlink()

        with pytest.raises(
            FileNotFoundError, match="There need to be four input S1P files"
        ):
            _get_closest_s11_time(
                d,
                fileglob="*.s1p",
                time=Time("2015:312:00:00"),
                dateformat="%Y_%j_%H",
                date_slice=slice(0, 11),
            )

    def test_include_files_that_dont_match(self, tmp_path):
        d, find_these_files = self._setup_dir(tmp_path)
        (d / "unmatching.s1p").touch()

        with pytest.warns(UserWarning, match="Only 4 of 5 were parseable"):
            out = _get_closest_s11_time(
                d,
                fileglob="*.s1p",
                time=Time("2015:312:00:00"),
                dateformat="%Y_%j_%H",
                date_slice=slice(0, 11),
            )
        assert out == sorted(find_these_files)

    def test_ignore_file(self, tmp_path):
        d, find_these_files = self._setup_dir(tmp_path)
        (d / "2015_001_00_0.s1p").touch()

        out = _get_closest_s11_time(
            d,
            fileglob="*.s1p",
            time=Time("2015:312:00:00"),
            dateformat="%Y_%j_%H",
            date_slice=slice(0, 11),
            ignore_files="2015_001",
        )
        assert out == sorted(find_these_files)


class TestGetS11Paths:
    """Tests of the get_s11_paths method."""

    def _setup_dir(self, tmp_path, year: int = 2015):
        d = tmp_path / "s11"
        if not d.exists():
            d.mkdir()

        # Create a few empty files
        find_these_files = [
            (d / f"{year}_312_00_0.s1p"),
            (d / f"{year}_312_00_1.s1p"),
            (d / f"{year}_312_00_2.s1p"),
            (d / f"{year}_312_00_3.s1p"),
        ]
        for fl in find_these_files:
            fl.touch()

        return d, find_these_files

    def test_with_sequence(self, tmp_path):
        d = tmp_path / "s11"
        d.mkdir()

        paths = []
        for i in range(4):
            paths.append(d / f"{i}.s1p")
            paths[-1].touch()

        out = get_antenna_s11_paths(paths)
        assert out == paths

        with pytest.raises(ValueError, match="length must be 4"):
            get_antenna_s11_paths(paths[:-1])

    def test_single_file(self, tmp_path):
        fl = tmp_path / "file.csv"
        fl.touch()
        out = get_antenna_s11_paths(fl)
        assert out == [fl]

    def test_get_closest_time(self, tmp_path):
        d, find_these_files = self._setup_dir(tmp_path)

        # Find these files
        out = get_antenna_s11_paths(
            d,
            time=Time("2015:312:00:00"),
            fileglob="*.s1p",
            dateformat="%Y_%j_%H",
            date_slice=slice(0, 11),
        )
        assert out == sorted(find_these_files)

    def test_find_directly(self, tmp_path):
        d, find_these_files = self._setup_dir(tmp_path)

        # Find these files
        out = get_antenna_s11_paths(
            f"{d}/2015_312_00_{{load}}.s1p",
        )
        assert out == sorted(find_these_files)

        find_these_files[-1].unlink()
        with pytest.raises(FileNotFoundError, match="There are not exactly four"):
            get_antenna_s11_paths(
                f"{d}/2015_312_00_{{load}}.s1p",
            )
