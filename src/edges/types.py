"""Various useful type-hint definitions."""

# from pygsdata.types import *
from pathlib import Path

from astropy import units

LengthType = units.Quantity["length"]
Conductivity = units.Quantity["electrical conductivity"]
InductanceType = units.Quantity["electromagnetic field strength"]
TemperatureType = units.Quantity["temperature"]


PathLike = str | Path
ImpedanceType = units.Quantity["electrical impedance"]
OhmType = units.Quantity[units.ohm]
DimlessType = units.Quantity["dimensionless"]
FreqType = units.Quantity[units.MHz]
