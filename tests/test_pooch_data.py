import tempfile
from pathlib import Path

import py7zr
import pytest
import requests

from edges.data import _pooch
from edges.data.cli import app


def test_fetch_b18_testing(capsys):
    app("data fetch_b18 --dataset testing", result_action="return_value")
    out = capsys.readouterr().out
    assert "/S11" in out
    assert "/Resistance" in out
    assert "Fetched B18 calibration testing data files:" in out


def test_fetch_b18_none(capsys):
    app("data fetch_b18 --dataset none", result_action="return_value")
    out = capsys.readouterr().out
    assert "No files fetched." in out


def _make_archive(archive: Path, content: str = "data") -> Path:
    """Write a 7z archive with a single top-level directory named for its stem."""
    archive.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as tmp:
        src = Path(tmp) / archive.stem
        src.mkdir()
        (src / "a.txt").write_text(content)
        with py7zr.SevenZipFile(archive, "w") as z:
            z.writeall(src, arcname=archive.stem)
    return archive


@pytest.fixture
def calobs_dir(tmp_path: Path, monkeypatch) -> Path:
    out = tmp_path / "B18-cal-raw-data"
    monkeypatch.setattr(_pooch, "_UNZIPPED_B18CAL_OBS", out)
    return out


@pytest.mark.parametrize("processor", ["calobs", "next_to_archive"])
class TestUnpack:
    """The pooch post-processors that unpack 7z archives (no network needed)."""

    @pytest.fixture
    def archive(self, tmp_path: Path) -> Path:
        return _make_archive(tmp_path / "cache" / "Spectra.7z", "v1")

    @pytest.fixture
    def dest(self, processor: str, archive: Path, calobs_dir: Path) -> Path:
        return calobs_dir if processor == "calobs" else archive.parent

    @staticmethod
    def _unpack(processor: str, archive: Path, action: str) -> Path:
        if processor == "calobs":
            return _pooch._unpack_to_calobs(str(archive), action, None)
        return _pooch._unpack7z(str(archive), action, None)

    def test_unpack_and_redownload(self, processor, archive, dest):
        out = self._unpack(processor, archive, "download")
        assert out == dest / "Spectra"
        assert (out / "a.txt").read_text() == "v1"
        (out / "stale.txt").write_text("stale")

        # Already extracted and not re-downloaded: left alone.
        assert self._unpack(processor, archive, "fetch") == out
        assert (out / "stale.txt").exists()

        # Re-downloaded: the existing directory is replaced (this used to crash,
        # calling unlink() on a directory).
        _make_archive(archive, "v2")
        assert self._unpack(processor, archive, "update") == out
        assert (out / "a.txt").read_text() == "v2"
        assert not (out / "stale.txt").exists()
        assert not list(dest.glob(".Spectra-extracting-*"))

    def test_interrupted_extraction(self, processor, archive, dest, monkeypatch):
        original = py7zr.unpack_7zarchive

        def _interrupted(archive, path, extra=None):
            (Path(path) / "Spectra").mkdir()
            (Path(path) / "Spectra" / "partial.txt").write_text("")
            raise KeyboardInterrupt

        monkeypatch.setattr(py7zr, "unpack_7zarchive", _interrupted)
        with pytest.raises(KeyboardInterrupt):
            self._unpack(processor, archive, "download")

        # Nothing partial is left that would be taken as a complete extraction.
        assert not (dest / "Spectra").exists()
        assert not list(dest.glob(".Spectra-extracting-*"))

        # So the next fetch (without a download) extracts it again.
        monkeypatch.setattr(py7zr, "unpack_7zarchive", original)
        out = self._unpack(processor, archive, "fetch")
        assert (out / "a.txt").read_text() == "v1"
        assert not (out / "partial.txt").exists()


def test_calibrated_s11_uses_registry_name(monkeypatch, tmp_path, calobs_dir):
    fetched = []

    def _fetch(name, *args, **kwargs):
        fetched.append(name)
        fl = tmp_path / name
        fl.write_text("s11")
        return str(fl)

    monkeypatch.setattr(_pooch.B18CAL_REPO, "fetch", _fetch)
    out = _pooch.fetch_b18cal_calibrated_s11s(in_obs=True)
    assert fetched == [_pooch._S11FILE]
    assert _pooch._S11FILE in _pooch.B18CAL_REPO.registry
    assert out == calobs_dir / "S11" / _pooch._S11FILE
    assert out.read_text() == "s11"


class _FlakyDownloader:
    """A pooch downloader that fails ``nfail`` times, then writes ``content``."""

    def __init__(self, nfail: int, content: bytes, error: Exception):
        self.nfail = nfail
        self.content = content
        self.error = error
        self.calls = 0

    def __call__(self, url, output_file, pooch_obj):
        self.calls += 1
        if self.calls <= self.nfail:
            raise self.error
        Path(output_file).write_bytes(self.content)


@pytest.fixture
def b18_repo(tmp_path: Path, monkeypatch):
    """A repository like the B18 one (same retries), with a single small file."""
    import hashlib

    import pooch

    monkeypatch.setattr(pooch.core.time, "sleep", lambda _: None)
    content = b"some data"
    repo = pooch.create(
        path=tmp_path,
        base_url=_pooch.B18CAL_REPO.base_url,
        registry={"file.txt": f"md5:{hashlib.md5(content).hexdigest()}"},
        retry_if_failed=_pooch.B18CAL_REPO.retry_if_failed,
    )
    return repo, content


def test_b18_repo_retries_failed_downloads():
    assert _pooch.B18CAL_REPO.retry_if_failed == _pooch.DOWNLOAD_RETRIES >= 3


@pytest.mark.parametrize(
    "error",
    [
        requests.exceptions.HTTPError("502 Server Error: Bad Gateway"),
        requests.exceptions.ChunkedEncodingError("Connection broken: IncompleteRead"),
        requests.exceptions.ConnectionError("Remote end closed connection"),
    ],
)
def test_transient_download_failures_are_retried(b18_repo, error):
    repo, content = b18_repo
    downloader = _FlakyDownloader(_pooch.DOWNLOAD_RETRIES, content, error)
    path = repo.fetch("file.txt", downloader=downloader)
    assert Path(path).read_bytes() == content
    assert downloader.calls == _pooch.DOWNLOAD_RETRIES + 1


def test_persistent_download_failure_raises(b18_repo):
    repo, content = b18_repo
    error = requests.exceptions.HTTPError("502 Server Error: Bad Gateway")
    downloader = _FlakyDownloader(_pooch.DOWNLOAD_RETRIES + 1, content, error)
    with pytest.raises(requests.exceptions.HTTPError, match="502"):
        repo.fetch("file.txt", downloader=downloader)
    assert downloader.calls == _pooch.DOWNLOAD_RETRIES + 1
