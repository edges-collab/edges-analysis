"""The global configuration for all of edges-analysis."""

import contextlib
import copy
import warnings
from pathlib import Path
from typing import Self

import attrs
import cattrs
import yaml
from platformdirs import PlatformDirs

dirs = PlatformDirs("edges", "edges-collab")


@attrs.define(frozen=False, kw_only=True)
class Config:
    """The configuration options of edges-analysis.

    Each attribute is an option. Options can be changed temporarily with the
    :meth:`use` context manager (which only accepts existing options), and saved
    to a file with :meth:`write`. The global configuration, ``edges.config.config``,
    is read from the user's config file when edges is imported.
    """

    raw_field_data: Path | None = attrs.field(
        default=None, converter=attrs.converters.optional(Path)
    )
    raw_lab_data: Path | None = attrs.field(
        default=None, converter=attrs.converters.optional(Path)
    )
    edges3_data: Path = attrs.field(
        default=Path("/data5/edges/data/EDGES3_data/MRO/"), converter=Path
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

        try:
            yield self
        finally:
            for k in kwargs:
                setattr(self, k, getattr(backup, k))

    def write(self, fname: str | Path | None = None) -> None:
        """Write current configuration to file to make it permanent.

        Parameters
        ----------
        fname
            The file to write to. By default, the user's config file, which is
            read when edges is imported.
        """
        fname = Path(fname or _config_filename)
        fname.parent.mkdir(parents=True, exist_ok=True)

        with fname.open("w") as fl:
            yaml.safe_dump(cattrs.unstructure(self), fl)

    @classmethod
    def load(cls, file_name: str | Path) -> Self:
        """Create a Config object from a config file.

        Parameters
        ----------
        file_name
            The YAML file to read. Options it does not set take their default
            values; an empty file gives the default configuration.

        Raises
        ------
        TypeError
            If the file does not contain a mapping of options.
        """
        with Path(file_name).open("r") as fl:
            config = yaml.safe_load(fl)

        if config is None:
            config = {}
        if not isinstance(config, dict):
            raise TypeError(
                f"Config file {file_name} must contain a mapping of options, "
                f"not {type(config).__name__}."
            )
        return cattrs.structure(config, cls)


_config_filename = Path(dirs.user_config_dir) / "config.yaml"


def _load_user_config(path: Path) -> Config:
    """Load the user's config file, falling back to the defaults if it is unusable.

    A missing file gives the defaults silently. A malformed file gives the defaults
    with a warning, so that it does not prevent edges from being imported.
    """
    try:
        return Config.load(path)
    except FileNotFoundError:
        return Config()
    except (
        OSError,
        yaml.YAMLError,
        cattrs.BaseValidationError,
        TypeError,
        ValueError,
    ) as e:
        warnings.warn(
            f"Could not read the edges config file {path} ({e!r}). Using the default "
            "configuration instead.",
            stacklevel=2,
        )
        return Config()


config = _load_user_config(_config_filename)

default_config = copy.deepcopy(config)
