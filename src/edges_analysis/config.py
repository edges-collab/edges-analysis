# This is the GLOBAL configuration for all of edges-analysis.
import os
import yaml
import contextlib
import warnings
from edges_cal.config import config as cal_config


class ConfigurationError(Exception):
    pass


class Config(dict):
    """Simple over-ride of dict that adds a context manager."""

    _defaults = {
        "raw_antenna_data_dir": "",
        "raw_lab_data_dir": "",
        "lab_products_dir": cal_config["cache_dir"],
        "field_products_dir": os.path.expanduser("~/edges-cal-field-data"),
    }

    # The following gives a way to change the keys of defaults over time,
    # and update the base config file.
    _aliases = {}

    def __init__(self, *args, **kwargs):

        super().__init__(*args, **kwargs)

        self._migrate()

    def _migrate(self):
        # Ensure the keys that got read in are the right keys for the current version
        do_write = False
        for k, v in self._defaults.items():
            if k in self:
                continue

            if k not in self._aliases:
                warnings.warn("Your configuration file is out of date. Updating...")
                do_write = True
                self[k] = v

            else:
                for alias in self._aliases[k]:
                    if alias in self:
                        do_write = True
                        warnings.warn(
                            f"Your configuration file has old key '{alias}' which has been re-named "
                            f"'{k}'. Updating..."
                        )
                        self[k] = self[alias]
                        del self[alias]
                if not do_write:
                    raise ConfigurationError(
                        f"The configuration file has key '{alias}' which is not known."
                    )

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
        with open(file_name, "r") as fl:
            config = yaml.load(fl)
        return cls(config)


config_filename = os.path.expanduser(os.path.join("~", ".edges-analysis", "config.yml"))

try:
    config = Config.load(config_filename)
except FileNotFoundError:
    config = Config()
    config.file_name = config_filename
    config.write()
