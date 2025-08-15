"""Functions for working with data from a thermistor."""

from collections.abc import Sequence
from typing import Self

import attrs
import numpy as np
from astropy import units as un
from astropy.table import QTable
from astropy.time import Time, TimeDelta

from .. import types as tp
from ..io.serialization import hickleable
from ..io.thermistor import read_thermistor_csv

IgnoreTimesType = int | un.Quantity[un.percent] | un.Quantity[un.s]


def ignore_ntimes(times: Time, ignore_times: IgnoreTimesType) -> int:
    """Number of time integrations to ignore from the start of the observation."""
    if isinstance(ignore_times, int):
        n = ignore_times
    elif ignore_times.unit.is_equivalent(un.second):
        time_since_start = (times - times[0]).to("second")
        n = np.where(time_since_start > ignore_times)[0][0]
    elif ignore_times.unit.is_equivalent(un.percent):
        n = int(len(times) * ignore_times.to(un.dimensionless_unscaled))
    else:
        raise TypeError("ignore_times is not a valid type!")

    return n


def get_temperature_thermistor(
    resistance: tp.OhmType,
    coeffs: str | Sequence[float] = "oven_industries_TR136_170",
) -> tp.TemperatureType:
    """
    Convert resistance of a thermistor to temperature.

    Uses a pre-defined set of standard coefficients.

    Parameters
    ----------
    resistance
    coeffs : str or len-3 iterable of floats, optional
        If str, should be an identifier of a standard set of coefficients, otherwise,
        should specify the coefficients.

    Returns
    -------
    temperature
        The temperature for each `resistance` given.
    """
    # Steinhart-Hart coefficients
    coeffs_ = {"oven_industries_TR136_170": [1.03514e-3, 2.33825e-4, 7.92467e-8]}

    if isinstance(coeffs, str):
        coeffs = coeffs_[coeffs]

    assert len(coeffs) == 3

    # TK in Kelvin
    return (
        1
        * un.K
        / (
            coeffs[0]
            + coeffs[1] * np.log(resistance.to_value("ohm"))
            + coeffs[2] * (np.log(resistance.to_value("ohm"))) ** 3
        )
    )


@hickleable
@attrs.define
class ThermistorReadings:
    """
    Object containing thermistor readings.

    Parameters
    ----------
    data
        The data array containing the readings.
    """

    data: QTable = attrs.field()

    @data.validator
    def _data_vld(self, att, val):
        if "times" not in val.colnames:
            raise ValueError("'times' must be in the data for ThermistorReadings")
        if "load_resistance" not in val.colnames:
            raise ValueError("'load_resistance' must be in data for THermistorReadings")

    def ignore_times(self, ignore_times: IgnoreTimesType):
        """Number of time integrations to ignore from the start of the observation."""
        n = ignore_ntimes(self.data["times"], ignore_times=ignore_times)
        return attrs.evolve(self, data=self.data[n:])

    @classmethod
    def from_csv(
        cls,
        path: tp.PathLike,
        ignore_times: IgnoreTimesType = 0,
    ) -> Self:
        """Generate the object from an io.Resistance object."""
        return ThermistorReadings(data=read_thermistor_csv(path)).ignore_times(
            ignore_times
        )

    def get_physical_temperature(self) -> tp.TemperatureType:
        """The associated thermistor temperature in K."""
        return get_temperature_thermistor(self.data["load_resistance"])

    def get_thermistor_indices(self, timestamps: Time) -> list[int | None]:
        """Get the index of the closest therm measurement for each spectrum."""
        closest = []
        indx = 0
        thermistor_timestamps = self.data["times"]

        deltat = thermistor_timestamps[1] - thermistor_timestamps[0]

        for d in timestamps:
            if indx >= len(thermistor_timestamps):
                closest.append(np.nan)
                continue

            for i, td in enumerate(thermistor_timestamps[indx:], start=indx):
                if d - td >= TimeDelta(0 * un.s):
                    if d - td <= deltat:
                        closest.append(i)
                        break
                    indx += 1

            else:
                closest.append(np.nan)

        return closest
