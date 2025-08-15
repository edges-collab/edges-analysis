"""Functions for dicke-switch calibration."""

import numpy as np
from pygsdata import GSData, gsregister


@gsregister("calibrate")
def dicke_calibration(data: GSData) -> GSData:
    """Calibrate field data using the Dicke switch data.

    Assumes that the data has the three loads "ant", "internal_load" and
    "internal_load_plus_noise_source". The data is calibrated using the
    Dicke switch model (i.e.
    ``(ant - internal_load)/(internal_load_plus_noise_source - internal_load)``).
    """
    iant = data.loads.index("ant")
    iload = data.loads.index("internal_load")
    ilns = data.loads.index("internal_load_plus_noise_source")

    with np.errstate(divide="ignore", invalid="ignore"):
        q = (data.data[iant] - data.data[iload]) / (data.data[ilns] - data.data[iload])

    return data.update(
        data=q[np.newaxis],
        data_unit="uncalibrated",
        times=data.times[:, [iant]],
        time_ranges=data.time_ranges[:, [iant]],
        effective_integration_time=data.effective_integration_time[[iant]],
        lsts=data.lsts[:, [iant]],
        lst_ranges=data.lst_ranges[:, [iant]],
        loads=("ant",),
        nsamples=data.nsamples[[iant]],
        flags={name: flag.any(axis="load") for name, flag in data.flags.items()},
        residuals=None,
    )
