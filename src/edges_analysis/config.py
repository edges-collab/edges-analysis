"""The global configuration for all of edges-analysis."""
import os
import yaml
import contextlib
import warnings


class ConfigurationError(Exception):
    pass


class Config(dict):
    """Simple over-ride of dict that adds a context manager."""

    _defaults = {
        "paths": {
            "raw_field_data": "",
            "raw_lab_data": "",
            "lab_products": os.path.expanduser("~/edges-calibrations"),
            "field_products": os.path.expanduser("~/edges-field-levels"),
            "beams": os.path.expanduser("~/edges-beams"),
            "antenna": os.path.expanduser("~/edges-antenna-meta"),
            "sky_models": os.path.expanduser("~/edges-sky-models"),
        }
    }

    # The following gives a way to change the keys of defaults over time,
    # and update the base config file.
    _aliases = {}

    def __init__(self, *args, **kwargs):

        super().__init__(*args, **kwargs)

        self._migrate()

    def _migrate(self):
        # Ensure the keys that got read in are the right keys for the current version

        def check(k, v, selfdict):
            do_write = False

            if k in selfdict:
                if isinstance(v, dict):
                    for kk, vv in v.items():
                        do_write |= check(kk, vv, selfdict[k])
            elif k not in self._aliases:
                warnings.warn("Your configuration file is out of date. Updating...")
                do_write = True
                selfdict[k] = v

            else:
                for alias in self._aliases[k]:
                    if alias in selfdict:
                        do_write = True
                        warnings.warn(
                            f"Your configuration file has old key '{alias}' which has "
                            f"been re-named '{k}'. Updating..."
                        )
                        selfdict[k] = selfdict[alias]
                        del selfdict[alias]

                        if isinstance(v, dict):
                            for kk, vv in v.items():
                                do_write |= check(kk, vv, selfdict[kk])

                if not do_write:
                    raise ConfigurationError(
                        f"The configuration file has key '{alias}' which is not known."
                    )
            return do_write

        do_write = False
        for k, v in self._defaults.items():
            do_write |= check(k, v, self)

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


config_filename = os.path.expanduser(os.path.join("~", ".edges.yml"))

try:
    config = Config.load(config_filename)
except FileNotFoundError:
    config = Config()
    config.file_name = config_filename
    config.write()
