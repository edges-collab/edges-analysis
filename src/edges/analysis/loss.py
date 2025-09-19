"""Functions defining expected losses from the instruments."""

import warnings
from pathlib import Path

import attrs
import numpy as np
import numpy.typing as npt
from astropy import units as un

from ..cal import ee, loss
from ..cal.s11 import CalibratedS11
from ..config import config


def low2_balun_connector_loss(
    freq: un.Quantity[un.MHz],
    ants11: np.ndarray | CalibratedS11 | str | Path,
    use_approx_eps0: bool = True,
    connector: ee.CoaxialCable = ee.KNOWN_CABLES["SC3792 Connector"],
    balun: ee.CoaxialCable = ee.KNOWN_CABLES["lowband-balun-tube"],
) -> npt.NDArray:
    """Obtain the balun and connector loss for the low-2 instrument on-site at MRO.

    Parameters
    ----------
    freq
        An array of frequencies at which to calculate the balun+connector loss.
    ants11
        The antenna S11, either as a numpy array, a path to a file containing the
        S11, or a S11Model instance.
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
        as `freq`).
    """
    if use_approx_eps0:
        connector = attrs.evolve(connector, eps0=8.854e-12 * un.F / un.m)
        balun = attrs.evolve(balun, eps0=8.854e-12 * un.F / un.m)

    # Get the antenna s11
    if isinstance(ants11, str | Path):
        ants11 = CalibratedS11.from_file(ants11).s11
    elif isinstance(ants11, CalibratedS11):
        ants11 = ants11.s11

    try:
        ants11 = np.asarray(ants11)
    except Exception as e:
        raise ValueError(
            "ants11 must be a numpy array or a S11Model instance or path."
        ) from e

    mdl = loss.get_cable_loss_model([connector, balun])
    return mdl(freq, ants11)


def _get_loss(fname: str | Path, freq: np.ndarray, n_terms: int) -> np.ndarray:
    gr = np.genfromtxt(fname)
    fr = gr[:, 0]
    dr = gr[:, 1]

    par = np.polyfit(fr, dr, n_terms)
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
            filename = (
                Path(__file__).parent
                / "data"
                / "loss"
                / instrument
                / f"{loss_type}.txt"
            )
            if not filename.exists():
                warnings.warn(
                    f"Ground loss file {filename} does not exist. Returning ones.",
                    stacklevel=2,
                )
                return np.ones(freq.shape)
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

    Parameters
    ----------
    filename : path
        File in which value of the ground loss for this instrument are tabulated.
    freq : array-like
        Frequency in MHz. For mid-band (low-band), between 50 and 150 (120) MHz.
    instrument : str, optional
        The instrument to find the ground loss for. Only required if `filename`
        doesn't exist and isn't an absolute path (in which case the standard directory
        structure will be searched using ``band``).
    configuration : str, optional
        The configuration of the instrument. A string, such as "45deg", which defines
        the orientation or other configuration parameters of the instrument, which may
        affect the ground loss.
    """
    return _get_loss_from_datafile(
        filename,
        freq=freq,
        instrument=instrument,
        configuration=configuration,
        loss_type="ground",
        n_terms=8,
    )


def antenna_loss(
    freq: np.ndarray,
    filename: str | Path,
    instrument: str | None = None,
    configuration: str = "",
):
    """
    Calculate antenna loss of a particular antenna at given frequencies.

    Parameters
    ----------
    filename : path
        File in which value of the antenna loss for this instrument are tabulated.
    freq : array-like
        Frequency in MHz. For mid-band (low-band), between 50 and 150 (120) MHz.
    instrument
        The instrument to find the antenna loss for. Only required if `filename`
        starts with the magic ':' (in which case the standard directory
        structure will be searched using ``band``).
    configuration : str, optional
        The configuration of the instrument. A string, such as "45deg", which defines
        the orientation or other configuration parameters of the instrument, which may
        affect the antenna loss.
    """
    return _get_loss_from_datafile(
        filename,
        freq=freq,
        instrument=instrument,
        configuration=configuration,
        loss_type="ground",
        n_terms=11,
    )
