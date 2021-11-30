"""The global configuration for all of edges-analysis."""
from __future__ import annotations

import contextlib
import warnings
import yaml
from pathlib import Path
from typing import Any


class ConfigurationError(Exception):
    pass


def _path_constructor(_loader, node):
    return Path(node.value).expanduser()


def _path_representer(dumper, data):
    return dumper.represent_scalar("!path", str(data))


yaml.add_constructor("!path", _path_constructor)
yaml.add_multi_representer(Path, _path_representer)


class Config(dict):
    """Simple over-ride of dict that adds a context manager."""

    _defaults = {
        "paths": {
            "raw_field_data": "",
            "raw_lab_data": "",
            "lab_products": Path("~/edges-calibrations").expanduser(),
            "field_products": Path("~/edges-field-levels").expanduser(),
            "beams": Path("~/edges-beams").expanduser(),
            "antenna": Path("~/edges-antenna-meta").expanduser(),
            "sky_models": Path("~/edges-sky-models").expanduser(),
        }
    }

    # The following gives a way to change the keys of defaults over time,
    # and update the base config file.
    _aliases = {}

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._migrate()

    @classmethod
    def _check_key(cls, key: str, val: Any, selfdict: Config) -> bool:
        do_write = False

        if key in selfdict and isinstance(val, dict):
            for kk, vv in val.items():
                do_write |= cls._check_key(kk, vv, selfdict[key])
        elif key not in cls._aliases:
            warnings.warn("Your configuration file is out of date. Updating...")
            do_write = True
            selfdict[key] = val

        else:
            for alias in cls._aliases[key]:
                if alias in selfdict:
                    do_write = True
                    warnings.warn(
                        f"Your configuration file has old key '{alias}' which has "
                        f"been re-named '{key}'. Updating..."
                    )
                    selfdict[key] = selfdict[alias]
                    del selfdict[alias]

                    if isinstance(val, dict):
                        for kk, vv in val.items():
                            do_write |= cls._check_key(kk, vv, selfdict[kk])

            if not do_write:
                raise ConfigurationError(
                    f"The configuration file has key '{key}' which is not known."
                )
        return do_write

    def _migrate(self) -> None:
        # Ensure the keys that got read in are the right keys for the current version
        do_write = False
        for k, v in self._defaults.items():
            do_write |= self._check_key(k, v, self)

        if do_write:
            self.write()

    @contextlib.contextmanager
    def use(self, **kwargs):
        """Context manager for using certain configuration options for a set time."""
        backup = self.copy()
        for k, v in kwargs.items():
            self[k] = v
        yield self
        for k in kwargs:
            self[k] = backup[k]

    def write(self, fname=None):
        """Write current configuration to file to make it permanent."""
        fname = fname or self.file_name
        with open(fname, "w") as fl:
            yaml.dump(self._as_dict(), fl)

    def _as_dict(self):
        """The plain dict defining the instance."""
        return {k: v for k, v in self.items()}

    @classmethod
    def load(cls, file_name):
        """Create a Config object from a config file."""
        cls.file_name = file_name
        with open(file_name) as fl:
            config = yaml.load(fl, Loader=yaml.FullLoader)
        return cls(config)

    def __eq__(self, other) -> bool:
        """Test equality."""
        if not isinstance(other, Config):
            return False
        for k, v in self.items():
            if not isinstance(v, dict) and other[k] != v:
                return False
            elif isinstance(v, dict):
                if not all(vv == other[k][kk] for kk, vv in self[k].items()):
                    return False

        return True


config_filename = Path("~/.edges.yml").expanduser()

try:
    config = Config.load(config_filename)
except FileNotFoundError:
    config = Config()
    config.file_name = config_filename
    config.write()
