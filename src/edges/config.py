"""The global configuration for all of edges-analysis."""

import contextlib
import copy
from pathlib import Path

import attrs
import cattrs
import yaml
from platformdirs import PlatformDirs

dirs = PlatformDirs("edges", "edges-collab")


@attrs.define(frozen=False, kw_only=True)
class Config:
    """Simple over-ride of dict that adds a context manager.

    Allows to specify extra config options, but ensures that all specified options
    are defined.
    """

    raw_field_data: Path | None = attrs.field(
        default=None, converter=attrs.converters.optional(Path)
    )
    raw_lab_data: Path | None = attrs.field(
        default=None, converter=attrs.converters.optional(Path)
    )
    beams: Path = attrs.field(
        default=Path(dirs.user_cache_dir) / "beams", converter=Path
    )
    antenna: Path = attrs.field(
        default=Path(dirs.user_cache_dir) / "antenna", converter=Path
    )
    sky_models: Path = attrs.field(
        default=Path(dirs.user_cache_dir) / "sky-models", converter=Path
    )

    @contextlib.contextmanager
    def use(self, **kwargs):
        """Context manager for using certain configuration options for a set time."""
        avail = list(attrs.fields_dict(self.__class__).keys())

        for k in kwargs:
            if k not in avail:
                raise KeyError(
                    f"Cannot use {k} in config, as it doesn't exist. "
                    f"Available keys: {avail}."
                )
        backup = copy.deepcopy(self)
        for k, v in kwargs.items():
            setattr(self, k, v)

        yield self

        for k in kwargs:
            setattr(self, k, getattr(backup, k))

    def write(self, fname=None):
        """Write current configuration to file to make it permanent."""
        fname = Path(fname or self.path)

        with fname.open("w") as fl:
            yaml.dump(cattrs.unstructure(self), fl)

    @classmethod
    def load(cls, file_name):
        """Create a Config object from a config file."""
        with Path(file_name).open("r") as fl:
            config = yaml.load(fl, Loader=yaml.FullLoader)

        return cattrs.structure(config, cls)


_config_filename = Path(dirs.user_config_dir) / "config.yaml"

try:  # pragma: no cover
    config = Config.load(_config_filename)
except FileNotFoundError:  # pragma: no cover
    config = Config()

default_config = copy.deepcopy(config)
