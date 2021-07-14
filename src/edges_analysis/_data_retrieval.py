import requests
from .config import config
from pathlib import Path

# The google drive ID for the folder containing these files:
# 14hH-zBhHGddVacc0ncqRWq7ofhGLWfND
_FILE_IDS = {
    "beam_model": {
        "low": {
            "": "1EwAUi-piwtL9zvLbvjbwJ583XK9GR53e",
            "45deg": "11DHCK_RmDfHqiyoPXtrIodbndinGhYHa",
        },
        "high": {
            "": "1MGnT14__Ooc1Y33_1Urxe3LgQOoBX7n_",
        },
        "mid": {"": "1URXyPqOiPNApnDzwWFZZC9rv4_F1Rwg7"},
    }
}


def retrieve_beam(band, configuration=""):
    """Retrieve a beam model from Google Drive and cache it locally.

    Parameters
    ----------
    band
        The instrument to download the model for (low, mid, high).
    configuration
        Any extra configuration string (eg '45' for `low45`).
    """
    fname = "builtin"
    if configuration:
        fname += f"_{configuration}.txt"
    abspath = (
        Path(config["paths"]["beams"]).expanduser() / f"{band}/simulations/feko" / fname
    )

    if not abspath.exists():
        # Get it from gdrive
        urlbase = "https://drive.google.com/uc?export=download"
        fileid = _FILE_IDS["beam_model"][band][configuration]
        print(f"Downloading new beam from Google Drive to {abspath}.")

        r = requests.get(
            urlbase, params={"id": fileid, "authuser": 0, "export": "download"}
        )
        if not abspath.parent.exists():
            abspath.parent.mkdir(parents=True)

        with open(abspath, "wb") as ofile:
            ofile.write(r.content)

    return f":{fname}"
