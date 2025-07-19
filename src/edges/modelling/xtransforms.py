"""Module defining x-variable transforms for modelling."""

from __future__ import annotations

from abc import ABCMeta, abstractmethod
from functools import cached_property

import attrs
import numpy as np
import yaml

from ..io.serialization import hickleable


def _transform_yaml_constructor(
    loader: yaml.SafeLoader, node: yaml.nodes.MappingNode
) -> XTransform:
    mapping = loader.construct_mapping(node, deep=True)
    return XTransform.get(node.tag[1:])(**mapping)


def _transform_yaml_representer(
    dumper: yaml.SafeDumper, tr: XTransform
) -> yaml.nodes.MappingNode:
    dct = attrs.asdict(tr, recurse=False)
    return dumper.represent_mapping(f"!{tr.__class__.__name__}", dct)


@hickleable
@attrs.define(frozen=True, kw_only=True, slots=False)
class XTransform(metaclass=ABCMeta):
    _models = {}

    def __init_subclass__(cls, is_meta=False, **kwargs):
        """Initialize a subclass and add it to the registered models."""
        super().__init_subclass__(**kwargs)

        yaml.add_constructor(f"!{cls.__name__}", _transform_yaml_constructor)

        if not is_meta:
            cls._models[cls.__name__.lower()] = cls

    @abstractmethod
    def transform(self, x: np.ndarray) -> np.ndarray:
        """Transform the coordinates."""

    @classmethod
    def get(cls, model: str) -> type[XTransform]:
        """Get a ModelTransform class."""
        return cls._models[model.lower()]

    def __call__(self, x: np.ndarray) -> np.ndarray:
        """Transform the coordinates."""
        return self.transform(x)

    def __getstate__(self):
        """Get the state for pickling."""
        return attrs.asdict(self)


@hickleable
@attrs.define(frozen=True, kw_only=True, slots=False)
class IdentityTransform(XTransform):
    def transform(self, x: np.ndarray) -> np.ndarray:
        """Transform the coordinates."""
        return x


@hickleable
@attrs.define(frozen=True, kw_only=True, slots=False)
class ScaleTransform(XTransform):
    scale: float = attrs.field(converter=float)

    def transform(self, x: np.ndarray) -> np.ndarray:
        """Transform the coordinates."""
        return x / self.scale


def tuple_converter(x):
    """Convert input to tuple of floats."""
    return tuple(float(xx) for xx in x)


@hickleable
@attrs.define(frozen=True, kw_only=True, slots=False)
class CentreTransform(XTransform):
    range: tuple[float, float] = attrs.field(converter=tuple_converter)
    centre: float = attrs.field(default=0.0, converter=float)

    def transform(self, x: np.ndarray) -> np.ndarray:
        """Transform the coordinates."""
        return x - self.range[0] - (self.range[1] - self.range[0]) / 2 + self.centre


@hickleable
@attrs.define(frozen=True, kw_only=True, slots=False)
class ShiftTransform(XTransform):
    shift: float = attrs.field(converter=float, default=0.0)

    def transform(self, x: np.ndarray) -> np.ndarray:
        """Transform the coordinates."""
        return x - self.shift


@hickleable
@attrs.define(frozen=True, kw_only=True, slots=False)
class UnitTransform(XTransform):
    """A transform that takes the input range down to -1 to 1."""

    range: tuple[float, float] = attrs.field(converter=tuple_converter)

    @cached_property
    def _centre(self):
        return CentreTransform(centre=0, range=self.range)

    def transform(self, x: np.ndarray) -> np.ndarray:
        """Transform the coordinates."""
        return 2 * self._centre.transform(x) / (self.range[1] - self.range[0])


@hickleable
@attrs.define(frozen=True, kw_only=True, slots=False)
class LogTransform(XTransform):
    """A transform that takes the logarithm of the input."""

    scale: float = attrs.field(default=1.0)

    def transform(self, x: np.ndarray) -> np.ndarray:
        """Transform the coordinates."""
        return np.log(x / self.scale)


@hickleable
@attrs.define(frozen=True, kw_only=True, slots=False)
class Log10Transform(LogTransform):
    """A transform that takes the base10 logarithm of the input."""

    def transform(self, x: np.ndarray) -> np.ndarray:
        """Transform the coordinates."""
        return np.log10(x / self.scale)


@hickleable
@attrs.define(frozen=True, kw_only=True, slots=False)
class ZerotooneTransform(UnitTransform):
    """A transform that takes an input range down to (0,1)."""

    def transform(self, x: np.ndarray) -> np.ndarray:
        """Transform the coordinates."""
        return (x - self.range[0]) / (self.range[1] - self.range[0])


yaml.add_multi_representer(XTransform, _transform_yaml_representer)
