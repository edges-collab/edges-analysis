"""A module defining a few new sub-formats for astropy times."""

from astropy.time import TimeISO


class TimeISOWithSlashes(TimeISO):
    """A time format similar to ISO but using slashes instead of dashes."""

    name = "iso_custom"  # Unique format name

    subfmts = (
        (
            "date_hms_space",
            "%m/%d/%Y %H:%M:%S",
            "{mon:02d}/{day:02d}/{year:04d} {hour:02d}:{min:02d}:{sec:02d}",
        ),
        (
            "date_hms_colon",
            "%m/%d/%Y:%H:%M:%S",
            "{mon:02d}/{day:02d}/{year:04d}:{hour:02d}:{min:02d}:{sec:02d}",
        ),
    )
