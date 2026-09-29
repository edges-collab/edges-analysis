"""Functions for reading lists of spectrum files."""

import warnings
from collections.abc import Sequence
from pathlib import Path

from astropy import units as un
from pygsdata import GSData, Telescope
from read_acq.gsdata import read_acq_to_gsdata
from read_acq.read_acq import Ancillary

from ..const import KNOWN_TELESCOPES

# Values of the fastspec ``--instrument`` header item for EDGES-3 at MRO. EDGES-3
# writes one file per switch-cycle target, named by the load being measured.
EDGES3_INSTRUMENTS = frozenset({"ant", "amb", "hot", "open", "short"})


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
        ``instrument``, ``samples_per_accumulation`` and ``acquisition_rate``. Older
        files written by pxspec have no header, so these entries are missing.
    """
    with warnings.catch_warnings():
        # Header items without a value (e.g. --output_file) raise warnings that are
        # irrelevant here, and are re-raised when the file is actually read.
        warnings.simplefilter("ignore", UserWarning)
        return Ancillary(path).meta


def infer_telescope_from_acq_header(meta: dict) -> Telescope:
    """Infer the telescope that wrote an ACQ file from its header.

    Parameters
    ----------
    meta
        The header of the file, as returned by :func:`read_acq_header`.

    Returns
    -------
    Telescope
        The inferred telescope. Files without a header (written by pxspec) and lab
        measurements are assigned to ``edges-low``. Unrecognized headers are also
        assigned to ``edges-low``, with a warning.
    """
    site = meta.get("site")
    instrument = meta.get("instrument")

    if site is None and instrument is None:
        # Written by pxspec, which was only ever used for EDGES-2.
        return KNOWN_TELESCOPES["edges-low"]

    if site == "mro" and isinstance(instrument, str):
        if instrument.startswith("low"):
            return KNOWN_TELESCOPES["edges-low"]
        if instrument in EDGES3_INSTRUMENTS:
            return KNOWN_TELESCOPES["edges3"]

    if site == "asulab" and instrument == "lab":
        return KNOWN_TELESCOPES["edges-low"]

    warnings.warn(
        f"Could not infer telescope from ACQ header (site={site!r}, "
        f"instrument={instrument!r}). Assuming edges-low. Pass `telescope` "
        "explicitly to silence this warning.",
        stacklevel=3,
    )
    return KNOWN_TELESCOPES["edges-low"]


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
        each ``.acq`` file (see :func:`infer_telescope_from_acq_header`).

    Returns
    -------
    GSData
        The data. For ``.acq`` files written by fastspec, the effective integration
        time is read from the file header rather than taken from the telescope.
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
        telescopes = {infer_telescope_from_acq_header(meta).name for meta in metas}
        if len(telescopes) > 1:
            raise ValueError(
                f"Files were written by different telescopes: {sorted(telescopes)}. "
                "Read them separately, or pass `telescope` explicitly."
            )
        telescope = KNOWN_TELESCOPES[telescopes.pop()]

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

    return read_acq_to_gsdata(files, telescope=telescope, **kwargs)
