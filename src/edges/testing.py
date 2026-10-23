"""Create very simple mock GSData objects for testing purposes."""

import functools
from typing import Any

import numpy as np
from astropy import units as un
from astropy.time import Time
from pygsdata import KNOWN_TELESCOPES, GSData
from pygsdata.coordinates import lst2gha
from scipy.interpolate import interp1d

from .const import edges_location
from .filters import DATA_PATH
from .frequencies import edges_raw_freqs

# To get "reasonable" data values, we use a model of the Haslam sky convolved with
# the 30x30m ground-plane beam that is shipped in the filters data folder. It is only
# read (lazily) to generate mock data, and contains nothing but plain float arrays.
_SKY_MODEL_FILE = DATA_PATH / "Lowband_30mx30m_Haslam_2p5_20minlst_50_100.npz"


@functools.cache
def _sky_model() -> dict[str, Any]:
    """Load the sky model and a spline of its 75 MHz slice versus GHA (cached)."""
    with np.load(_SKY_MODEL_FILE, allow_pickle=False) as fl:
        model = {k: fl[k] for k in fl.files}

    msky = model["sky_temp"][:, 25]  # this is the 75 MHz slice
    mgha = model["lsts"]
    return {
        "model": model,
        "msky": msky,
        "mgha": mgha,
        "spl75": interp1d(mgha, msky),
    }


def create_mock_edges_data(
    flow: un.Quantity[un.MHz] = 50 * un.MHz,
    fhigh: un.Quantity[un.MHz] = 100 * un.MHz,
    ntime: int = 100,
    time0: float = 2459900.27,
    dt: un.Quantity[un.s] = 40.0 * un.s,
    add_noise: bool = False,
    as_power: bool = False,
) -> GSData:
    """Create mock GSData objects."""
    dt = dt.to(un.day)
    freqs = edges_raw_freqs(f_low=flow, f_high=fhigh)
    times = Time(
        np.arange(time0, (ntime - 0.1) * dt.value + time0, dt.value)[:, None],
        format="jd",
    )

    lsts = times.sidereal_time("apparent", longitude=edges_location.lon)
    gha = lst2gha(lsts)[:, 0]

    skydata = (
        _sky_model()["spl75"](gha.hour)[:, None]
        * ((freqs / (75 * un.MHz)) ** (-2.5))[None, :]
    )

    if add_noise:
        rng = np.random.default_rng()
        skydata += rng.normal(0, 0.001, skydata.shape) * skydata

    data = skydata[None, None]

    if as_power:
        p1 = np.ones_like(data)
        p2 = np.ones_like(data) * 100
        data = np.concatenate((data, p1, p2), axis=0)
        times = Time(
            np.hstack((
                times.jd,
                times.jd + dt.to_value("day") / 3,
                times.jd + 2 * dt.to_value("day") / 3,
            )),
            format="jd",
        )
    return GSData(
        data=data,
        freqs=freqs,
        times=times,
        telescope=KNOWN_TELESCOPES["edges-low"],
        nsamples=np.ones_like(data),
        effective_integration_time=dt / 3,
        data_unit="power" if as_power else "temperature",
        auxiliary_measurements={
            "ambient_hum": np.linspace(10.0, 90.0, ntime),
            "receiver_temp": np.linspace(10.0, 90.0, ntime),
        },
    )
