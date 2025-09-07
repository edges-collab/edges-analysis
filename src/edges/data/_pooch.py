"""Module defining external datasets used throughout tests and tutorials."""

import shutil
from pathlib import Path

import pooch
import py7zr
from platformdirs import PlatformDirs

dirs = PlatformDirs("edges", "edges-collab")

B18CAL_REPO = POOCH = pooch.create(
    path=dirs.user_cache_dir,
    # Use the figshare DOI
    base_url="doi:10.5281/zenodo.16883743",
    registry=None,
)

# Automatically populate the registry
B18CAL_REPO.load_registry_from_doi()

_UNZIPPED_B18CAL_OBS = Path(dirs.user_cache_dir) / "B18-cal-raw-data"


def _unpack7z(fname: str, action: str, pup: pooch.Pooch):
    path = Path(fname)

    newpath = path.parent / path.stem

    # Don't unzip if file already exists and is not being downloaded
    if action in ("update", "download") or not newpath.exists():
        py7zr.unpack_7zarchive(path, path.parent)

    return newpath


def _unpack_to_calobs(fname: str, action: str, pup: pooch.Pooch):
    """Post-processing hook to unzip a file and return the unzipped file name."""
    path = Path(fname)
    # Create a new name for the unzipped file. Appending something to the
    # name is a relatively safe way of making sure there are no clashes
    # with other files in the registry.
    newpath = _UNZIPPED_B18CAL_OBS / path.stem

    # Don't unzip if file already exists and is not being downloaded
    if action in ("update", "download") or not newpath.exists():
        _UNZIPPED_B18CAL_OBS.mkdir(parents=True, exist_ok=True)
        py7zr.unpack_7zarchive(path, _UNZIPPED_B18CAL_OBS)

    return newpath


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


def fetch_b18cal_calibrated_s11s(in_obs: bool = False) -> Path:
    fl = Path(
        B18CAL_REPO.fetch(
            "s11_calibration_low_band_LNA25degC_2015-09-16-12-30-29_simulator2_long.txt"
        )
    )
    if in_obs:
        (_UNZIPPED_B18CAL_OBS / "S11").mkdir(parents=True, exist_ok=True)

        shutil.copyfile(fl, _UNZIPPED_B18CAL_OBS / "S11" / fl.name)
        return _UNZIPPED_B18CAL_OBS / "S11" / fl.name
    return fl


def fetch_b18cal_full() -> Path:
    fetch_b18cal_resistances()
    fetch_b18cal_s11s()
    fetch_b18cal_spectra()
    fetch_b18cal_calibrated_s11s(in_obs=True)
    return _UNZIPPED_B18CAL_OBS


def fetch_b18_cal_outputs() -> Path:
    return Path(B18CAL_REPO.fetch("LegacyPipelineOutputs.7z", processor=_unpack7z))
