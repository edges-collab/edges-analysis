"""Create very simple mock GSData objects for testing purposes."""

import numpy as np
from astropy import units as un
from astropy.time import Time
from edges_cal.tools import FrequencyRange

from edges_analysis.const import edges_location
from edges_analysis.coordinates import lst2gha
from edges_analysis.gsdata import GSData


def create_mock_edges_data(
    flow: un.Quantity[un.MHz] = 50 * un.MHz,
    fhigh: un.Quantity[un.MHz] = 100 * un.MHz,
    ntime: int = 100,
    time0: float = 2459900.0,
    dt: un.Quantity[un.s] = 40.0 * un.s,
    add_noise: bool = False,
):
    dt = dt.to(un.day)
    freqs = FrequencyRange.from_edges(f_low=flow, f_high=fhigh, keep_full=False)
    times = Time(
        np.arange(time0, (ntime - 0.1) * dt.value + time0, dt.value)[:, None],
        format="jd",
    )

    lsts = times.sidereal_time("apparent", longitude=edges_location.lon)
    gha = lst2gha(lsts.hour)[:, 0]

    skydata = (
        1e4
        * (np.cos(gha * np.pi / 12)[:, None] + 1.2)
        * ((freqs.freq / (75 * un.MHz)) ** (-2.5))[None, :]
    )

    if add_noise:
        skydata += np.random.normal(0, 0.001, skydata.shape) * skydata

    data = skydata[None, None]
    print("DATA UNITS", data.unit)
    return GSData(
        data=data,
        freq_array=freqs.freq,
        time_array=times,
        telescope_location=edges_location,
        loads=("ant",),
        nsamples=np.ones_like(data),
        effective_integration_time=dt / 3,
        telescope_name="Mock-EDGES",
        data_unit="temperature",
        auxiliary_measurements={
            "ambient_hum": np.linspace(10.0, 90.0, ntime),
            "receiver_temp": np.linspace(10.0, 90.0, ntime),
        },
    )
