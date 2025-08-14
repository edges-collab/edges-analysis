"""A sub-package for reading and writing EDGES data formats.

The package also contains a number of methods for finding data that are generally
laid out in a certain way on disk.
"""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version(__name__)
except PackageNotFoundError:
    # package is not installed
    __version__ = "unknown"
finally:
    del version, PackageNotFoundError

from pathlib import Path

__all__ = [
    "TEST_DATA_PATH",
    "CalObsDefEDGES2",
    "CalObsDefEDGES3",
    "CalkitFileSpec",
    "CalkitFileSpec",
    "LoadDefEDGES2",
    "LoadDefEDGES3",
    "LoadS11",
    "SParams",
    "SwitchingState",
    "get_mean_temperature",
    "hickleable",
    "read_auxiliary_data",
    "read_s1p",
    "read_temperature_log",
    "read_thermistor_csv",
    "read_thermlog_file",
    "read_weather_file",
]
from . import time_formats
from .auxiliary import read_auxiliary_data, read_thermlog_file, read_weather_file
from .calobsdef import (
    CalkitFileSpec,
    CalObsDefEDGES2,
    LoadDefEDGES2,
    LoadS11,
    SwitchingState,
)
from .calobsdef3 import CalObsDefEDGES3, LoadDefEDGES3
from .serialization import hickleable
from .templogs import get_mean_temperature, read_temperature_log
from .thermistor import read_thermistor_csv
from .vna import SParams, read_s1p

TEST_DATA_PATH = Path(__file__).parent / "test_data"

del Path
