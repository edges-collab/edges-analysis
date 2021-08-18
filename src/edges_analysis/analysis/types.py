"""Simple type definitions for use internally."""
from typing import Union, Type
from pathlib import Path
from edges_cal.modelling import Model

PathLike = Union[str, Path]
Modelable = Union[str, Type[Model]]
