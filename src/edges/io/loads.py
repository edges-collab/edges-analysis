"""Names of the calibration loads, and how they are spelled in the file layouts.

Throughout ``edges``, the four calibration loads are called by their *canonical*
names, :data:`LOAD_NAMES`: ``ambient``, ``hot_load``, ``open`` and ``short``. These
are the attribute names of :class:`~edges.io.CalObsDefEDGES2` and
:class:`~edges.io.CalObsDefEDGES3`, the ``name`` of their load definitions, the keys
of :attr:`edges.cal.CalibrationObservation.loads` and the keys in HDF5 files.

On disk, the loads are spelled differently: the EDGES-2 layout uses
:data:`EDGES2_LAYOUT_NAMES` (e.g. ``HotLoad``) and the EDGES-3 layout uses
:data:`EDGES3_LAYOUT_NAMES` (e.g. ``hot``). Functions that take the name of a load
from a user accept any of these spellings (and a few other known variants, such as
``Hot_Load``), converting them with :func:`canonical_load_name`.
"""

from collections.abc import Container, Mapping
from typing import Literal, TypeVar, get_args

from bidict import bidict

#: The canonical name of a calibration load.
LoadName = Literal["ambient", "hot_load", "open", "short"]

#: The canonical names of the four calibration loads.
LOAD_NAMES: tuple[LoadName, ...] = get_args(LoadName)

#: The names of the loads in the EDGES-2 file layout, keyed by canonical name. These
#: name the Resistance, Spectra and S11 files and directories of each load.
EDGES2_LAYOUT_NAMES: bidict[LoadName, str] = bidict({
    "ambient": "Ambient",
    "hot_load": "HotLoad",
    "open": "LongCableOpen",
    "short": "LongCableShorted",
})

#: The names of the loads in the EDGES-3 file layout, keyed by canonical name. These
#: label the S11 and spectrum files of each load (and are the ``--instrument`` header
#: item of EDGES-3 spectrum files). They are also used by Alan's C code.
EDGES3_LAYOUT_NAMES: bidict[LoadName, str] = bidict({
    "ambient": "amb",
    "hot_load": "hot",
    "open": "open",
    "short": "short",
})

# Other known spellings of the loads (mostly found in old EDGES-2 file names).
_ALIASES: dict[LoadName, tuple[str, ...]] = {
    "ambient": ("AmbientLoad", "Ambient_Load", "ambient_load", "Ambient_load"),
    "hot_load": ("Hot_Load",),
    "open": (
        "cable_open",
        "long_cable_open",
        "Long_Cable_Open",
        "Open_Cable",
        "open_cable",
    ),
    "short": (
        "LongCableShort",
        "cable_short",
        "long_cable_short",
        "long_cable_shorted",
        "cable_shorted",
        "Short_cable",
        "Short_Cable",
        "shorted_cable",
    ),
}

_SPELLINGS: dict[str, LoadName] = {
    spelling: name
    for name in LOAD_NAMES
    for spelling in (
        name,
        EDGES2_LAYOUT_NAMES[name],
        EDGES3_LAYOUT_NAMES[name],
        *_ALIASES[name],
    )
}

_T = TypeVar("_T")


def canonical_load_name(name: str) -> LoadName:
    """Return the canonical name of a calibration load.

    Parameters
    ----------
    name
        The name of the load: a canonical name (e.g. ``hot_load``), its name in the
        EDGES-2 or EDGES-3 file layout (e.g. ``HotLoad`` or ``hot``), or another
        known spelling (e.g. ``Hot_Load``). Matching is case-sensitive.

    Returns
    -------
    LoadName
        The canonical name of the load, one of :data:`LOAD_NAMES`.

    Raises
    ------
    ValueError
        If ``name`` is not a known spelling of a calibration load.
    """
    try:
        return _SPELLINGS[name]
    except (KeyError, TypeError):
        raise ValueError(
            f"Unknown calibration load '{name}'. Use one of {LOAD_NAMES} (or their "
            f"names in the EDGES-2 or EDGES-3 file layout)."
        ) from None


def canonical_load_keys(
    mapping: Mapping[str, _T], other_keys: Container[str] = ()
) -> dict[str, _T]:
    """Return a copy of a mapping keyed by load name, with canonical keys.

    Parameters
    ----------
    mapping
        A mapping whose keys are names of calibration loads (in any spelling accepted
        by :func:`canonical_load_name`).
    other_keys
        Keys that are not load names, which are kept unchanged (e.g. ``"default"``).

    Returns
    -------
    dict
        A new dictionary with the same values, keyed by the canonical load names.

    Raises
    ------
    ValueError
        If a key is not a known spelling of a calibration load (nor in
        ``other_keys``), or if more than one key names the same load.
    """
    out = {}
    for key, val in mapping.items():
        name = key if key in other_keys else canonical_load_name(key)
        if name in out:
            raise ValueError(f"More than one key given for load '{name}'.")
        out[name] = val
    return out
