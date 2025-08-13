"""Methods for filtering/flagging data."""

from pathlib import Path

__all__ = [
    "apply_flags",
    "aux_filter",
    "explicit_day_filter",
    "filter_150mhz",
    "filters",
    "flag_frequency_ranges",
    "lst_model",
    "maxfm_filter",
    "moon_filter",
    "negative_power_filter",
    "peak_orbcomm_filter",
    "peak_power_filter",
    "prune_flagged_integrations",
    "rfi_model_filter",
    "rfi_model_nonlinear_window",
    "rfi_model_sweep_filter",
    "rfi_watershed_filter",
    "rms_filter",
    "rmsf_filter",
    "sun_filter",
]

DATA_PATH = Path(__file__).parent / "data"

from . import filters, lst_model
from .filters import (
    apply_flags,
    aux_filter,
    explicit_day_filter,
    filter_150mhz,
    flag_frequency_ranges,
    maxfm_filter,
    moon_filter,
    negative_power_filter,
    peak_orbcomm_filter,
    peak_power_filter,
    prune_flagged_integrations,
    rfi_model_filter,
    rfi_model_nonlinear_window,
    rfi_model_sweep_filter,
    rfi_watershed_filter,
    rms_filter,
    rmsf_filter,
    sun_filter,
)
