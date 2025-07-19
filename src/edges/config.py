"""The global configuration for all of edges-analysis."""

from __future__ import annotations

import contextlib
import copy
import os
import warnings
from pathlib import Path
from typing import ClassVar

import yaml


class ConfigurationError(Exception):
    pass


class Config(dict):
    """Simple over-ride of dict that adds a context manager.

    Allows to specify extra config options, but ensures that all specified options
    are defined.
    """

    _defaults: ClassVar = {
        "cal": {"cache-dir": str(Path("~/.edges-cal-cache").expanduser())},
        "paths": {
            "raw_field_data": "",
            "raw_lab_data": "",
            "lab_products": os.path.expanduser("~/edges-calibrations"),
            "field_products": os.path.expanduser("~/edges-field-levels"),
            "beams": os.path.expanduser("~/edges-beams"),
            "antenna": os.path.expanduser("~/edges-antenna-meta"),
            "sky_models": os.path.expanduser("~/edges-sky-models"),
        },
    }

    # The following gives a way to change the keys of defaults over time,
    # and update the base config file.
    _aliases: ClassVar = {}

    def __init__(
        self,
        path: str | Path | None = None,
        _loaded_from_file: bool = False,
        *args,
        **kwargs,
    ):
        self.path = Path(path) if path is not None else None
        self._loaded_from_file = _loaded_from_file
        if self._loaded_from_file and (not self.path or not self.path.exists()):
            raise ValueError("cannot have been loaded from file as it doesn't exist.")
        super().__init__(*args, **kwargs)
        self._migrate()

    def _migrate(self):
        # Ensure the keys that got read in are the right keys for the current version

        def check(k, v, selfdict):
            if k in selfdict:
                updated = False
                if isinstance(v, dict):
                    for kk, vv in v.items():
                        updated |= check(kk, vv, selfdict[k])
                return updated

            # Otherwise, we must update selfdict in some way.

            # First way: we have the key under a different name. In this case, we
            # change the name of the key in the instance to match the schema.
            if k in self._aliases:
                for alias in self._aliases[k]:
                    if alias in selfdict:
                        warnings.warn(
                            f"Your configuration spec has old key '{alias}' which has "
                            f"been re-named '{k}'.",
                            stacklevel=2,
                        )
                        selfdict[k] = selfdict[alias]
                        del selfdict[alias]

            # If the key still isn't there, it mustn't have existed as an alias. In this
            # case, we just write its default into the instance.
            if k not in selfdict:
                selfdict[k] = v

            # Now, if the value is a dict, we need to recurse into it to check it.
            if isinstance(v, dict):
                for kk, vv in v.items():
                    check(kk, vv, selfdict[k])

            return True

        updated = False
        for k, v in self._defaults.items():
            updated |= check(k, v, self)

        if updated and self.path:
            self.write()

    def _add_to_schema(self, new: dict):
        """Add more keys/defaults to the schema.

        Not to be called by users.
        """
        self._defaults.update(new)
        self._migrate()

    @contextlib.contextmanager
    def use(self, **kwargs):
        """Context manager for using certain configuration options for a set time."""
        for k in kwargs:
            if k not in self:
                raise KeyError(
                    f"Cannot use {k} in config, as it doesn't exist. "
                    f"Available keys: {list(self.keys())}."
                )
        backup = copy.deepcopy(self)
        for k, v in kwargs.items():
            if isinstance(self[k], dict):
                self[k].update(v)
            else:
                self[k] = v
        yield self
        for k in kwargs:
            self[k] = backup[k]

    def write(self, fname=None):
        """Write current configuration to file to make it permanent."""
        fname = Path(fname or self.path)

        with fname.open("w") as fl:
            yaml.dump(self._as_dict(), fl)
        self.path = fname

    def _as_dict(self):
        """Get the plain dict defining the instance."""
        return dict(self.items())

    @classmethod
    def load(cls, file_name):
        """Create a Config object from a config file."""
        with Path(file_name).open("r") as fl:
            config = yaml.load(fl, Loader=yaml.FullLoader)
        return cls(file_name, _loaded_from_file=True, **config)


_config_filename = Path("~/.edges.yml").expanduser()

try:  # pragma: no cover
    config = Config.load(_config_filename)
except FileNotFoundError:  # pragma: no cover
    config = Config()
    config.file_name = _config_filename

    with contextlib.suppress(Exception):
        config.write()

default_config = copy.deepcopy(config)
