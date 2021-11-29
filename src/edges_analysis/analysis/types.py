"""Simple type definitions for use internally."""
from edges_cal.modelling import Model
from pathlib import Path
from typing import Type, Union

PathLike = Union[str, Path]
Modelable = Union[str, Type[Model]]
