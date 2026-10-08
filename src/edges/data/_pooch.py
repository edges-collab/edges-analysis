"""Module defining external datasets used throughout tests and tutorials."""

import shutil
import tempfile
from pathlib import Path

import pooch
import py7zr
from platformdirs import PlatformDirs

dirs = PlatformDirs("edges", "edges-collab")

# Don't "auto" populate the registry, because that requires an API call, and sometimes
# we want to be running this in massive parallel.
_S11FILE = "s11_calibration_low_band_LNA25degC_2015-09-16-12-30-29_simulator2_long.txt"
B18CAL_REPO = pooch.create(
    path=dirs.user_cache_dir,
    base_url="doi:10.5281/zenodo.18091240",
    registry={
        "LegacyPipelineOutputs.7z": "md5:df5f573d390f0cff46157ce157849925",
        "Resistance.7z": "md5:9ac40bdd7a009ec722af4235dadc6162",
        "S11.7z": "md5:e04bd94977fb0a2d330290c92d0c19f8",
        _S11FILE: "md5:eaf078330589b4adedf4ce0b687f4add",
        "Spectra.7z": "md5:c81c5d7ae127d0306b7c7dcaf5510304",
        "AntennaS11.7z": "md5:09d67ed167e293864150afa92edd32b2",
    },
)

_UNZIPPED_B18CAL_OBS = Path(dirs.user_cache_dir) / "B18-cal-raw-data"


def _extract_7z(archive: Path, dest: Path, name: str, force: bool) -> Path:
    """Extract a 7z archive whose top-level entry is ``name`` into ``dest``.

    The archive is extracted into a temporary directory inside ``dest`` and then
    moved into place, so an interrupted extraction never leaves a partial
    ``dest / name`` that would later be taken as complete. Existing entries are
    replaced (whether files or directories).

    Parameters
    ----------
    archive
        The archive to extract.
    dest
        The directory to extract into.
    name
        The name of the top-level entry of the archive (its stem), whose path is
        returned.
    force
        Whether to extract even if ``dest / name`` already exists.

    Returns
    -------
    Path
        The path to the extracted ``dest / name``.
    """
    target = dest / name
    if target.exists() and not force:
        return target

    dest.mkdir(parents=True, exist_ok=True)
    tmpdir = Path(tempfile.mkdtemp(prefix=f".{name}-extracting-", dir=dest))
    try:
        py7zr.unpack_7zarchive(str(archive), str(tmpdir))
        # Move the target last, so that it only exists once everything is in place.
        entries = sorted(tmpdir.iterdir(), key=lambda pth: pth.name == name)
        for entry in entries:
            out = dest / entry.name
            if out.is_dir() and not out.is_symlink():
                shutil.rmtree(out)
            elif out.exists() or out.is_symlink():
                out.unlink()
            entry.rename(out)
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)

    return target


def _unpack7z(fname: str, action: str, pup: pooch.Pooch) -> Path:
    """Post-processing hook to unzip a file next to it and return the unzipped path."""
    path = Path(fname)
    # Don't unzip if file already exists and is not being downloaded
    return _extract_7z(
        path, path.parent, path.stem, force=action in ("update", "download")
    )


def _unpack_to_calobs(fname: str, action: str, pup: pooch.Pooch) -> Path:
    """Post-processing hook to unzip a file and return the unzipped file name."""
    path = Path(fname)
    # Don't unzip if file already exists and is not being downloaded
    return _extract_7z(
        path,
        _UNZIPPED_B18CAL_OBS,
        path.stem,
        force=action in ("update", "download"),
    )


def _fetch_b18cal_data(kind: str) -> Path:
    return B18CAL_REPO.fetch(
        f"{kind}.7z", progressbar=True, processor=_unpack_to_calobs
    )


def fetch_b18cal_spectra() -> Path:
    return _fetch_b18cal_data("Spectra")


def fetch_b18cal_resistances() -> Path:
    return _fetch_b18cal_data("Resistance")


def fetch_b18cal_s11s() -> Path:
    return _fetch_b18cal_data("S11")


def fetch_b18cal_ants11s() -> Path:
    return _fetch_b18cal_data("AntennaS11")


def fetch_b18cal_calibrated_s11s(in_obs: bool = False) -> Path:
    fl = Path(B18CAL_REPO.fetch(_S11FILE))
    if in_obs:
        (_UNZIPPED_B18CAL_OBS / "S11").mkdir(parents=True, exist_ok=True)

        shutil.copyfile(fl, _UNZIPPED_B18CAL_OBS / "S11" / fl.name)
        return _UNZIPPED_B18CAL_OBS / "S11" / fl.name
    return fl


def fetch_b18cal_full() -> Path:
    fetch_b18cal_resistances()
    fetch_b18cal_s11s()
    fetch_b18cal_ants11s()
    fetch_b18cal_spectra()
    fetch_b18cal_calibrated_s11s(in_obs=True)
    return _UNZIPPED_B18CAL_OBS


def fetch_b18_cal_outputs() -> Path:
    return Path(B18CAL_REPO.fetch("LegacyPipelineOutputs.7z", processor=_unpack7z))
