"""Functions for applying calibration solutions to data."""

from pygsdata import GSData, gsregister


@gsregister("calibrate")
def approximate_temperature(
    data: GSData, *, tload: float, tns: float, reverse: bool = False
):
    """Convert an uncalibrated object to an uncalibrated_temp object.

    This uses a guess for T0 and T1 that provides an approximate temperature spectrum.
    One does not need this step to perform actual calibration, and if actual calibration
    is done following applying this function, you will need to provide the same tload
    and tns as used here.
    """
    if data.data_unit != "uncalibrated_temp" and reverse:
        raise ValueError(
            "data_unit must be 'uncalibrated_temp' to decalibrate from approximate "
            "temperature"
        )

    if data.data_unit != "uncalibrated" and not reverse:
        raise ValueError(
            "data_unit must be 'uncalibrated' to calculate approximate temperature"
        )

    if reverse:
        udata = (data.data - tload) / tns
        resid = data.residuals / tns if data.residuals is not None else None
    else:
        udata = data.data * tns + tload
        resid = data.residuals * tns if data.residuals is not None else None

    return data.update(
        data=udata,
        data_unit="uncalibrated" if reverse else "uncalibrated_temp",
        residuals=resid,
    )
