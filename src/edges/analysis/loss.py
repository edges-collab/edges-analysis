"""Functions defining expected losses from the instruments."""

from pathlib import Path

import attrs
import numpy as np
import numpy.typing as npt
from astropy import units as un

from ..cal import loss
from ..cal import sparams as sp
from ..config import config
from ..data import LOSS_PATH


def low2_balun_connector_loss(
    ants11: sp.ReflectionCoefficient | str | Path,
    use_approx_eps0: bool = True,
    connector: sp.CoaxialCable = sp.KNOWN_CABLES["SC3792 Connector"],
    balun: sp.CoaxialCable = sp.KNOWN_CABLES["lowband-balun-tube"],
) -> npt.NDArray:
    """Obtain the balun and connector loss for the low-2 instrument on-site at MRO.

    Parameters
    ----------
    ants11
        The antenna S11, either as a :class:`edges.cal.sparams.ReflectionCoefficient`
        or a path to a file containing one. The loss is computed at its frequencies.
    use_approx_eps0 : bool, optional
        Whether to approximate the vacuum electric permittivity as 8.854e-12 F/m
        instead of the ~10-digit accuracy it has from astropy.
        This is mainly for backward compatibility with previous results.
        Default is True.
    connector
        The connector to use. Default is the SC3792 connector.
    balun
        The balun to use. Default is the lowband-balun-tube.

    Returns
    -------
    loss : np.ndarray
        The balun+connector loss as a function of frequency (same length
        as the frequencies of ``ants11``).
    """
    if use_approx_eps0:
        connector = attrs.evolve(connector, eps0=8.854e-12 * un.F / un.m)
        balun = attrs.evolve(balun, eps0=8.854e-12 * un.F / un.m)

    # Get the antenna s11
    if isinstance(ants11, str | Path):
        ants11 = sp.ReflectionCoefficient.from_file(ants11)

    mdl = loss.get_cable_loss_model([connector, balun])
    return mdl(ants11)


def _get_loss(fname: str | Path, freq: np.ndarray, n_terms: int) -> np.ndarray:
    """Fit a polynomial with ``n_terms`` terms to a tabulated loss, and evaluate it.

    The file has two columns: frequency (MHz) and the fractional loss. The returned
    value is the gain, ``1 - loss``.
    """
    gr = np.genfromtxt(fname)
    fr = gr[:, 0]
    dr = gr[:, 1]

    par = np.polyfit(fr, dr, n_terms - 1)
    model = np.polyval(par, freq)

    return 1 - model


def _get_loss_from_datafile(
    filename: str | Path,
    freq: np.ndarray,
    instrument: str | None,
    configuration: str,
    loss_type: str,
    n_terms: int,
) -> np.ndarray:
    if str(filename).startswith(":"):
        if instrument is None:
            raise ValueError(
                f"For non-absolute path {filename}, you must provide 'band'."
            )
        if str(filename) == ":":
            # Use the built-in loss files
            if configuration:
                loss_type += f"_{configuration}"
            filename = LOSS_PATH / instrument / f"{loss_type}.txt"
            if not filename.exists():
                available = sorted(
                    str(p.relative_to(LOSS_PATH)) for p in LOSS_PATH.glob("*/*.txt")
                )
                raise FileNotFoundError(
                    f"No built-in {loss_type} loss file for instrument "
                    f"'{instrument}' ({filename} does not exist). Available built-in "
                    f"loss files: {available}."
                )
        else:
            # Find the file in the standard directory structure
            filename = config.antenna / instrument / "loss" / str(filename)[1:]
    else:
        filename = Path(filename)

    return _get_loss(str(filename), freq, n_terms)


def ground_loss(
    freq: np.ndarray,
    filename: str | Path,
    instrument: str | None = None,
    configuration: str = "",
) -> np.ndarray:
    """
    Calculate ground loss of a particular antenna at given frequencies.

    The tabulated loss is fit with a 9-term (degree-8) polynomial in frequency.

    Parameters
    ----------
    freq : array-like
        Frequency in MHz. For mid-band (low-band), between 50 and 150 (120) MHz.
    filename : path
        File in which value of the ground loss for this instrument are tabulated.
        If it is exactly ``":"``, the built-in file for ``instrument`` (and
        ``configuration``) is used, from ``edges/data/loss/<instrument>/``. If it
        starts with ``":"``, the rest of it is a filename in the standard directory
        structure for the instrument.
    instrument : str, optional
        The instrument to find the ground loss for (e.g. ``"low"`` or ``"mid"``).
        Only required if ``filename`` starts with ``":"``.
    configuration : str, optional
        The configuration of the instrument. A string, such as "45deg", which defines
        the orientation or other configuration parameters of the instrument, which may
        affect the ground loss.

    Returns
    -------
    np.ndarray
        The ground gain, ``1 - loss``, at each frequency.

    Raises
    ------
    FileNotFoundError
        If a built-in loss file is requested (``filename=":"``) that does not exist
        for the given instrument and configuration.
    """
    return _get_loss_from_datafile(
        filename,
        freq=freq,
        instrument=instrument,
        configuration=configuration,
        loss_type="ground",
        n_terms=9,
    )


def antenna_loss(
    freq: np.ndarray,
    filename: str | Path,
    instrument: str | None = None,
    configuration: str = "",
) -> np.ndarray:
    """
    Calculate antenna loss of a particular antenna at given frequencies.

    The tabulated loss is fit with an 11-term (degree-10) polynomial in frequency.

    Parameters
    ----------
    freq : array-like
        Frequency in MHz. For mid-band (low-band), between 50 and 150 (120) MHz.
    filename : path
        File in which value of the antenna loss for this instrument are tabulated.
        If it is exactly ``":"``, the built-in antenna-loss file for ``instrument``
        (and ``configuration``) is used, from ``edges/data/loss/<instrument>/``. If
        it starts with ``":"``, the rest of it is a filename in the standard
        directory structure for the instrument.
    instrument
        The instrument to find the antenna loss for (e.g. ``"mid"``). Only required
        if ``filename`` starts with ``":"``.
    configuration : str, optional
        The configuration of the instrument. A string, such as "45deg", which defines
        the orientation or other configuration parameters of the instrument, which may
        affect the antenna loss.

    Returns
    -------
    np.ndarray
        The antenna gain, ``1 - loss``, at each frequency.

    Raises
    ------
    FileNotFoundError
        If a built-in loss file is requested (``filename=":"``) that does not exist
        for the given instrument and configuration (only ``"mid"`` has one).
    """
    return _get_loss_from_datafile(
        filename,
        freq=freq,
        instrument=instrument,
        configuration=configuration,
        loss_type="antenna",
        n_terms=11,
    )
