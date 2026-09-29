"""Functions for reading lists of spectrum files."""

import logging
import warnings
from collections.abc import Sequence
from pathlib import Path

import numpy as np
from astropy import units as un
from astropy.time import Time
from pygsdata import GSData, Telescope
from pygsdata.utils import time_concat
from read_acq.gsdata import read_acq_to_gsdata
from read_acq.read_acq import Ancillary

from ..const import KNOWN_TELESCOPES

logger = logging.getLogger(__name__)

# Values of the fastspec ``--instrument`` header item for EDGES-3. Unlike EDGES-2
# (where it is the band, e.g. "low2"), EDGES-3 writes one file per load, named by
# the load being measured.
EDGES3_LOADS = frozenset({"ant", "amb", "hot", "open", "short"})

# Fastspec ``--site`` values of EDGES-3 deployments that have a telescope preset.
EDGES3_SITES = {"mro": "edges3", "hmp": "edges3-devon"}


def read_acq_header(path: str | Path) -> dict:
    """Read the metadata header of an ACQ file.

    Parameters
    ----------
    path
        The ACQ file to read.

    Returns
    -------
    dict
        The header items. Files written by fastspec have entries such as ``site``,
        ``instrument``, ``datadir``, ``samples_per_accumulation`` and
        ``acquisition_rate``. Older files written by pxspec have no header, so these
        entries are missing.
    """
    with warnings.catch_warnings():
        # Header items without a value (e.g. --output_file) raise warnings that are
        # irrelevant here, and are re-raised when the file is actually read.
        warnings.simplefilter("ignore", UserWarning)
        return Ancillary(path).meta


def infer_telescope_from_acq_header(meta: dict) -> str | None:
    """Infer the telescope that wrote an ACQ file from its header, if possible.

    Only EDGES-3 at MRO (``edges3``) and on Devon Island (``edges3-devon``) can be
    identified. Files written by pxspec have no header, and fastspec headers from
    EDGES-2, lab measurements and other EDGES-3 systems do not identify a telescope
    that has a preset in ``KNOWN_TELESCOPES``.

    Parameters
    ----------
    meta
        The header of the file, as returned by :func:`read_acq_header`.

    Returns
    -------
    str or None
        The key of the telescope in ``KNOWN_TELESCOPES``, or None if it cannot be
        determined.
    """
    instrument = meta.get("instrument")
    datadir = str(meta.get("datadir", ""))

    if instrument not in EDGES3_LOADS:
        return None

    # The Adak system reported site=mro during its first days at Adak.
    if "edges3_adak" in datadir:
        return None

    return EDGES3_SITES.get(meta.get("site"))


def get_acq_integration_time(meta: dict) -> un.Quantity[un.s] | None:
    """Get the effective integration time per switch position from an ACQ header.

    This is the number of ADC samples accumulated into each spectrum divided by the
    sampling rate. It does not include switching or processing overhead, so it is
    shorter than the wall-clock time spent on each switch position.

    Parameters
    ----------
    meta
        The header of the file, as returned by :func:`read_acq_header`.

    Returns
    -------
    Quantity or None
        The integration time, or None if the header does not define it (e.g. for
        files written by pxspec).
    """
    nsamples = meta.get("samples_per_accumulation")
    rate = meta.get("acquisition_rate")
    if not nsamples or not rate:
        return None
    return (nsamples / (rate * un.MHz)).to(un.s)


def read_spectra(
    files: Sequence[Path], telescope: Telescope | str | None = None
) -> GSData:
    """Read common spectrum file formats.

    Parameters
    ----------
    files
        The files to read. All must have the same format, and they are concatenated
        along the time axis.
    telescope
        The telescope that took the data, either as a :class:`pygsdata.Telescope` or
        the name of one in ``KNOWN_TELESCOPES``. Only used for ``.acq`` files (other
        formats store the telescope). If not given, it is inferred from the header of
        each ``.acq`` file (see :func:`infer_telescope_from_acq_header`). If that
        fails, ``edges-low`` is assumed and a warning is raised.

    Returns
    -------
    GSData
        The data. For ``.acq`` files, the time ranges are derived from the time
        stamps (see :func:`acq_time_ranges`), and for files written by fastspec the
        effective integration time is read from the header. Neither is taken from
        the telescope, except as a fallback.
    """
    fmt = files[0].suffix

    if fmt in (".h5", ".gsh5"):
        return GSData.from_file(files, concat_axis="time")
    if fmt == ".acq":
        return _read_acq_spectra(files, telescope)
    raise ValueError(f"File format '{fmt}' not supported.")


def _read_acq_spectra(
    files: Sequence[Path], telescope: Telescope | str | None
) -> GSData:
    metas = [read_acq_header(fl) for fl in files]

    if isinstance(telescope, str):
        telescope = KNOWN_TELESCOPES[telescope]
    elif telescope is None:
        telescope = _infer_telescope(files, metas)

    kwargs = {}
    intg_times = {
        None if (t := get_acq_integration_time(meta)) is None else t.to_value(un.s)
        for meta in metas
    }
    if len(intg_times) > 1:
        raise ValueError(
            "Files have different integration times in their headers (seconds): "
            f"{intg_times}. Read them separately."
        )
    intg_time = intg_times.pop()
    if intg_time is not None:
        kwargs["effective_integration_time"] = intg_time * un.s

    data = read_acq_to_gsdata(files, telescope=telescope, **kwargs)

    # A cycle is (nloads x integration time) plus switching and processing overhead,
    # which is ~15-40% for EDGES-2 and EDGES-3. Anything much longer is a gap.
    cycle_intg = telescope.integration_time if intg_time is None else intg_time * un.s
    max_cycle = 2 * data.nloads * cycle_intg
    time_ranges = acq_time_ranges(data.times, max_cycle_duration=max_cycle)
    if time_ranges is None:
        warnings.warn(
            f"Cannot determine the cycle duration from the time stamps in {files}, "
            "as no two consecutive switch cycles are within "
            f"{max_cycle.to(un.s):.1f} of each other. Time ranges will use the "
            f"telescope's integration time ({telescope.integration_time}).",
            stacklevel=3,
        )
        return data

    # lst_ranges must be recomputed as well, since update() keeps existing fields.
    return data.update(
        time_ranges=time_ranges,
        lst_ranges=time_ranges.sidereal_time("apparent", telescope.location),
    )


def acq_time_ranges(
    times: Time,
    max_cycle_duration: un.Quantity[un.s] | None = None,
    max_gap_factor: float = 1.5,
) -> Time | None:
    """Compute the time range of each switch position from ACQ time stamps.

    ACQ files have one time stamp per switch cycle (shared by all switch positions),
    marking the start of the cycle. Each cycle is taken to last until the next
    cycle starts, and is split equally between its switch positions, in order.
    The resulting time ranges tile the observation with no gaps or overlaps.

    The last cycle has no following time stamp, and a gap in the data (e.g. between
    files, or from dropped cycles) would make a cycle look too long. These cycles,
    and any whose following time stamp is not later than its own, are assigned the
    median cycle duration instead. The median is taken over cycles no longer than
    ``max_cycle_duration``, so that it is not set by the gaps between files when
    each file has only a few cycles.

    Parameters
    ----------
    times
        The time stamps of the data, with shape ``(ntimes, nloads)``, where the
        loads are in the order of the switch positions.
    max_cycle_duration
        The longest plausible cycle. Longer intervals between time stamps are
        treated as gaps. By default, there is no limit.
    max_gap_factor
        Cycles longer than this factor times the median cycle duration are assigned
        the median duration.

    Returns
    -------
    Time or None
        The time ranges, with shape ``(ntimes, nloads, 2)``. None if the cycle
        duration cannot be determined, i.e. no two consecutive time stamps are a
        plausible cycle apart.
    """
    start = times[:, 0]
    nloads = times.shape[1]

    duration = (start[1:] - start[:-1]).to_value(un.s)
    valid = duration > 0
    if max_cycle_duration is not None:
        valid &= duration <= max_cycle_duration.to_value(un.s)
    if not np.any(valid):
        return None
    typical = np.median(duration[valid])

    duration = np.append(duration, typical)
    duration[(duration <= 0) | (duration > max_gap_factor * typical)] = typical

    step = duration / nloads
    offsets = np.arange(nloads + 1) * step[:, None]  # (ntimes, nloads + 1)
    edges = start[:, None] + offsets * un.s
    return time_concat((edges[:, :-1, None], edges[:, 1:, None]), axis=-1)


def _infer_telescope(files: Sequence[Path], metas: Sequence[dict]) -> Telescope:
    names = {infer_telescope_from_acq_header(meta) for meta in metas}
    if len(names) > 1:
        raise ValueError(
            "Files were written by different telescopes: "
            f"{sorted(str(n) for n in names)} (None means unknown). Read them "
            "separately, or pass `telescope` explicitly."
        )
    name = names.pop()

    if name is None:
        warnings.warn(
            f"Could not determine the telescope from the header of {files[0]}. "
            "Assuming edges-low. Pass `telescope` explicitly to read_spectra.",
            stacklevel=4,
        )
        return KNOWN_TELESCOPES["edges-low"]

    meta = metas[0]
    logger.info(
        f"Inferred telescope {name!r} from the header of {files[0]} "
        f"(site={meta.get('site')!r}, instrument={meta.get('instrument')!r}). "
        "Pass `telescope` to read_spectra to override."
    )
    return KNOWN_TELESCOPES[name]
