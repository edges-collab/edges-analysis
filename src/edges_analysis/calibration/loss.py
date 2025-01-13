"""Functions defining expected losses from the instruments."""

from __future__ import annotations

from pathlib import Path

import attrs
import hickle
import numpy as np
import numpy.typing as npt
from astropy import units as un
from edges_cal import ee, loss
from edges_cal.s11 import LoadS11
from scipy import integrate

from ..config import config


def low2_balun_connector_loss(
    freq: un.Quantity[un.MHz],
    ants11: np.ndarray | LoadS11 | str | Path,
    use_approx_eps0: bool = True,
) -> npt.NDArray:
    """Obtain the balun and connector loss for the low-2 instrument on-site at MRO.

    Parameters
    ----------
    freq
        An array of frequencies at which to calculate the balun+connector loss.
    use_approx_eps0 : bool, optional
        Whether to approximate the vacuum electric permittivity as 8.854e-12 F/m
        instead of the ~10-digit accuracy it has from astropy.
    """
    connector = ee.KNOWN_CABLES["SC3792 Connector"]
    balun = ee.KNOWN_CABLES["lowband-balun-tube"]

    if use_approx_eps0:
        connector = attrs.evolve(connector, eps0=8.854e-12 * un.F / un.m)
        balun = attrs.evolve(balun, eps0=8.854e-12 * un.F / un.m)

    # Get the antenna s11
    if isinstance(ants11, str | Path):
        ants11 = hickle.load(ants11).s11_model(freq)
    elif isinstance(ants11, LoadS11):
        ants11 = ants11.s11_model(freq)

    try:
        ants11 = np.asarray(ants11)
    except Exception as e:
        raise ValueError(
            "ants11 must be a numpy array or a LoadS11 instance or path."
        ) from e

    mdl = loss.get_cable_loss_model([connector, balun])
    return mdl(freq, ants11)


def ground_loss_from_beam(beam, deg_step: float) -> np.ndarray:
    """
    Calculate ground loss from a given beam instance.

    Parameters
    ----------
    beam : Beam instance
        The beam to use for the calculation.
    deg_step : float
        Frequency in MHz. For mid-band (low-band), between 50 and 150 (120) MHz.

    Returns
    -------
    gain: array of the gain values
    """
    p_in = np.zeros_like(beam.beam)
    gain_t = np.zeros((np.shape(beam.beam)[0], np.shape(beam.beam)[2]))

    gain = np.zeros(np.shape(beam.beam)[0])

    for k in range(np.shape(beam.frequency)[0]):
        p_in[k] = (
            np.sin((90 - np.transpose([beam.elevation] * 360)) * deg_step * np.pi / 180)
            * beam.beam[k]
        )

        gain_t[k] = integrate.trapezoid(p_in[k], dx=deg_step * np.pi / 180, axis=0)

        gain[k] = integrate.trapezoid(gain_t[k], dx=deg_step * np.pi / 180, axis=0)
        gain[k] = gain[k] / (4 * np.pi)
    return gain


def _get_loss(fname: str | Path, freq: np.ndarray, n_terms: int) -> np.ndarray:
    gr = np.genfromtxt(fname)
    fr = gr[:, 0]
    dr = gr[:, 1]

    par = np.polyfit(fr, dr, n_terms)
    model = np.polyval(par, freq)

    return 1 - model


def ground_loss(
    freq: np.ndarray,
    filename: str | Path,
    beam=None,
    deg_step: float = 1.0,
    band: str | None = None,
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
    beam
        A :class:`Beam` instance with which the ground loss may be computed.
    deg_step
        The steps (in degrees) of the azimuth angle in the beam (if given).
    band : str, optional
        The instrument to find the ground loss for. Only required if `filename`
        doesn't exist and isn't an absolute path (in which case the standard directory
        structure will be searched using ``band``).
    configuration : str, optional
        The configuration of the instrument. A string, such as "45deg", which defines
        the orientation or other configuration parameters of the instrument, which may
        affect the ground loss.
    """
    if beam is not None:
        return ground_loss_from_beam(beam, deg_step)

    elif str(filename).startswith(":"):
        if band is None:
            raise ValueError(
                f"For non-absolute path {filename}, you must provide 'band'."
            )
        if str(filename) == ":":
            # Use the built-in loss files
            fl = "ground"
            if configuration:
                fl += f"_{configuration}"
            filename = Path(__file__).parent / "data" / "loss" / band / f"{fl}.txt"
            if not filename.exists():
                return np.ones_like(freq)
        else:
            # Find the file in the standard directory structure
            filename = (
                Path(config["paths"]["antenna"]) / band / "loss" / str(filename)[1:]
            )
        return _get_loss(str(filename), freq, 8)
    else:
        filename = Path(filename)
        return _get_loss(str(filename), freq, 8)


def antenna_loss(
    freq: np.ndarray,
    filename: str | Path | bool,
    band: str | None = None,
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
    band : str, optional
        The instrument to find the antenna loss for. Only required if `filename`
        starts with the magic ':' (in which case the standard directory
        structure will be searched using ``band``).
    configuration : str, optional
        The configuration of the instrument. A string, such as "45deg", which defines
        the orientation or other configuration parameters of the instrument, which may
        affect the antenna loss.
    """
    if str(filename).startswith(":"):
        if str(filename) == ":":
            # Use the built-in loss files
            fl = "antenna"
            if configuration:
                fl += f"_{configuration}"
            filename = Path(__file__).parent / "data" / "loss" / band / f"{fl}.txt"
            if not filename.exists():
                return np.zeros_like(freq)
        else:
            # Find the file in the standard directory structure
            filename = (
                Path(config["paths"]["antenna"]) / band / "loss" / str(filename)[1:]
            )
    else:
        filename = Path(filename)

    return _get_loss(str(filename), freq, 11)
