"""Various useful type-hint definitions."""

from pathlib import Path

from astropy import units
from numpy.typing import NDArray

LengthType = units.Quantity["length"]
Conductivity = units.Quantity["electrical conductivity"]
InductanceType = units.Quantity["electromagnetic field strength"]
TemperatureType = units.Quantity["temperature"]


PathLike = str | Path
ImpedanceType = units.Quantity["electrical impedance"]
OhmType = units.Quantity["ohm"]
DimlessType = units.Quantity["dimensionless"]
FreqType = units.Quantity["frequency"]
TimeType = units.Quantity["time"]


FloatArray = NDArray[float]
ComplexArray = NDArray[complex]
