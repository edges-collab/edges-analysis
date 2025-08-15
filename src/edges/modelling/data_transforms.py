"""Module defining transforms of data that occur pre-fitting."""

from abc import ABCMeta, abstractmethod
from typing import Self

import attrs
import numpy as np
import yaml

from ..io.serialization import hickleable


def _transform_yaml_constructor(loader: yaml.SafeLoader, node: yaml.nodes.MappingNode):
    mapping = loader.construct_mapping(node, deep=True)
    return DataTransform.get(node.tag[4:])(**mapping)


def _transform_yaml_representer(dumper: yaml.SafeDumper, tr) -> yaml.nodes.MappingNode:
    dct = attrs.asdict(tr, recurse=False)
    return dumper.represent_mapping(f"!dt.{tr.__class__.__name__}", dct)


@hickleable
@attrs.define(frozen=True, kw_only=True, slots=False)
class DataTransform(metaclass=ABCMeta):
    """A base class for model transforms.

    A DataTransform must implement *both* the `transform` and `inverse` methods.
    These methods may take both the response variable (x) and the data and return
    some transformed data. This transformed data will be what is actually fit by the
    model (and the inverse transform will be applied to fit evaluations to take them
    back to the space of the data).
    """

    _models = {}

    def __init_subclass__(cls, is_meta=False, **kwargs):
        """Initialize a subclass and add it to the registered models."""
        super().__init_subclass__(**kwargs)

        yaml.add_constructor(f"!dt.{cls.__name__}", _transform_yaml_constructor)

        if not is_meta:
            cls._models[cls.__name__.lower()] = cls

    @abstractmethod
    def transform(self, x: np.ndarray, data: np.ndarray) -> np.ndarray:
        """Transform the data."""

    @abstractmethod
    def inverse(self, x: np.ndarray, data: np.ndarray) -> np.ndarray:
        """Transform the data."""

    @classmethod
    def get(cls, model: str) -> type[Self]:
        """Get a ModelTransform class."""
        return cls._models[model.lower()]

    def __getstate__(self):
        """Get the state for pickling."""
        return attrs.asdict(self)


@hickleable
@attrs.define(frozen=True, kw_only=True, slots=False)
class IdentityTransform(DataTransform):
    """A transform that does nothing."""

    def transform(self, x: np.ndarray, data: np.ndarray) -> np.ndarray:
        """Transform the data."""
        return data

    def inverse(self, x: np.ndarray, trns: np.ndarray) -> np.ndarray:
        """Inverse transform the data."""
        return trns


yaml.add_multi_representer(DataTransform, _transform_yaml_representer)
