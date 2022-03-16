"""Global configuration options."""
from edges_io.config import config, default_config
import os

_new_defaults = {
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

config._add_to_schema(_new_defaults)
default_config._add_to_schema(_new_defaults)
