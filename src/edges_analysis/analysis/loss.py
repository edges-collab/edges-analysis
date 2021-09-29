"""Functions defining expected losses from the instruments."""
from __future__ import annotations

from pathlib import Path
import numpy as np
from edges_cal import reflection_coefficient as rc
from ..config import config
from scipy import integrate


def balun_and_connector_loss(
    band: str,
    freq,
    gamma_ant,
    monte_carlo_flags=(False, False, False, False, False, False, False, False),
):
    """
    Compute balun and connector losses.

    Parameters
    ----------
    band : str {'low3', 'mid'}
        Parameters of the loss are different for each antenna.
    freq : array-like
        Frequency in MHz
    gamma_ant: float
        Reflection coefficient of antenna at the reference plane, the LNA input.
    monte_carlo_flags : tuple of bool
        Which parameters to add a random offset to, in order:
        * tube_inner_radius
        * tube_outer_radius
        * tube_length
        * connector_inner_radius
        * connector_outer_radius
        * connector_length
        * metal_conductivity
        * teflon_permittivity

    Returns
    -------
    Gb : float or array-like
        The balun loss
    Gc : float or array-like
        The connector loss
    """
    # Angular frequency
    w = 2 * np.pi * freq * 1e6

    # Inch-to-meters conversion
    inch2m = 1 / 39.370

    # Conductivity of copper
    # Pozar 3rd edition. Alan uses a different number. What
    sigma_copper0 = 5.96 * 10 ** 7

    # Metal conductivity
    sigma_copper = 1 * sigma_copper0
    sigma_brass = 0.24 * sigma_copper0

    sigma_xx_inner = 0.24 * sigma_copper0
    sigma_xx_outer = 0.024 * sigma_copper0

    # Permeability
    u0 = (
        4 * np.pi * 10 ** (-7)
    )  # permeability of free space (same for copper, brass, etc., all nonmagnetic
    ur_air = 1  # relative permeability of air
    u_air = u0 * ur_air

    # Permittivity
    c = 299792458  # speed of light
    e0 = 1 / (u0 * c ** 2)  # permittivity of free space

    parameters = {
        "low": {
            "balun_length": 43.6 * inch2m,
            "connector_length": 0.8 * inch2m,
            "er_air": 1.07,
            "ric_b": ((5 / 16) * inch2m) / 2,
            "roc_b": ((3 / 4) * inch2m) / 2,
            "roc_c": (0.16 * inch2m) / 2,
        },
        "mid": {
            "balun_length": 35 * inch2m,
            "connector_length": 0.03,
            "er_air": 1.2,
            "ric_b": ((16 / 32) * inch2m) / 2,
            "roc_b": ((1.25) * inch2m) / 2,
            "roc_c": (0.161 * inch2m) / 2,
        },
    }

    ep_air = e0 * parameters[band]["er_air"]
    tan_delta_air = 0
    epp_air = ep_air * tan_delta_air

    er_teflon = 2.05  # why Alan????
    ep_teflon = e0 * er_teflon
    # http://www.kayelaby.npl.co.uk/general_physics/2_6/2_6_5.html
    tan_delta_teflon = 0.0002
    epp_teflon = ep_teflon * tan_delta_teflon

    ur_teflon = 1  # relative permeability of teflon
    u_teflon = u0 * ur_teflon

    ric_b = parameters[band]["ric_b"]
    if monte_carlo_flags[0]:
        # 1-sigma of 3%
        ric_b *= 1 + 0.03 * np.random.normal()

    roc_b = parameters[band]["roc_b"]
    if monte_carlo_flags[1]:
        # 1-sigma of 3%
        roc_b *= 1 + 0.03 * np.random.normal()

    l_b = parameters[band]["balun_length"]  # length in meters
    if monte_carlo_flags[2]:
        l_b += 0.001 * np.random.normal()  # 1-sigma of 1 mm

    # Connector dimensions
    ric_c = (0.05 * inch2m) / 2  # radius of outer wall of inner conductor
    if monte_carlo_flags[3]:
        # 1-sigma of 3%, about < 0.04 mm
        ric_c *= 1 + 0.03 * np.random.normal()

    roc_c = parameters[band]["roc_c"]
    if monte_carlo_flags[4]:
        # 1-sigma of 3%
        roc_c *= 1 + 0.03 * np.random.normal()

    l_c = parameters[band]["connector_length"]
    if monte_carlo_flags[5]:
        l_c += 0.0001 * np.random.normal()

    if monte_carlo_flags[6]:
        sigma_copper *= 1 + 0.01 * np.random.normal()
        sigma_brass *= 1 + 0.01 * np.random.normal()
        sigma_xx_inner *= 1 + 0.01 * np.random.normal()
        sigma_xx_outer *= 1 + 0.01 * np.random.normal()

    if monte_carlo_flags[7] == 1:
        # 1-sigma of 1%
        epp_teflon *= 1 + 0.01 * np.random.normal()

    # Skin Depth
    skin_depth_copper = np.sqrt(2 / (w * u0 * sigma_copper))
    skin_depth_brass = np.sqrt(2 / (w * u0 * sigma_brass))

    skin_depth_xx_inner = np.sqrt(2 / (w * u0 * sigma_xx_inner))
    skin_depth_xx_outer = np.sqrt(2 / (w * u0 * sigma_xx_outer))

    # Surface resistance
    Rs_copper = 1 / (sigma_copper * skin_depth_copper)
    Rs_brass = 1 / (sigma_brass * skin_depth_brass)

    Rs_xx_inner = 1 / (sigma_xx_inner * skin_depth_xx_inner)
    Rs_xx_outer = 1 / (sigma_xx_outer * skin_depth_xx_outer)

    def get_induc_cap_res_cond_prop(
        ric, roc, skin_depth_inner, skin_depth_outer, rs_inner, rs_outer, u, ep, epp
    ):
        L_inner = u0 * skin_depth_inner / (4 * np.pi * ric)
        L_dielec = (u / (2 * np.pi)) * np.log(roc / ric)
        L_outer = u0 * skin_depth_outer / (4 * np.pi * roc)
        L = L_inner + L_dielec + L_outer
        C = 2 * np.pi * ep / np.log(roc / ric)
        R = (rs_inner / (2 * np.pi * ric)) + (rs_outer / (2 * np.pi * roc))
        G = 2 * np.pi * w * epp / np.log(roc / ric)

        return (
            np.sqrt((R + 1j * w * L) * (G + 1j * w * C)),
            np.sqrt((R + 1j * w * L) / (G + 1j * w * C)),
        )

    # Inductance per unit length
    gamma_b, Zchar_b = get_induc_cap_res_cond_prop(
        ric_b,
        roc_b,
        skin_depth_copper,
        skin_depth_brass,
        Rs_copper,
        Rs_brass,
        u_air,
        ep_air,
        epp_air,
    )

    gamma_c, Zchar_c = get_induc_cap_res_cond_prop(
        ric_c,
        roc_c,
        skin_depth_xx_inner,
        skin_depth_xx_outer,
        Rs_xx_inner,
        Rs_xx_outer,
        u_teflon,
        ep_teflon,
        epp_teflon,
    )

    # Impedance of Agilent terminations
    Zref = 50
    Ropen, Rshort, Rmatch = rc.agilent_85033E(freq * 1e6, Zref, 1)

    def get_gamma(r):
        Z = rc.gamma2impedance(r, Zref)
        Zin_b = rc.input_impedance_transmission_line(Zchar_b, gamma_b, l_b, Z)
        Zin_c = rc.input_impedance_transmission_line(Zchar_c, gamma_c, l_c, Z)
        Rin_b = rc.impedance2gamma(Zin_b, Zref)
        Rin_c = rc.impedance2gamma(Zin_c, Zref)
        return Rin_b, Rin_c

    Rin_b_open, Rin_c_open = get_gamma(Ropen)
    Rin_b_short, Rin_c_short = get_gamma(Rshort)
    Rin_b_match, Rin_c_match = get_gamma(Rmatch)

    # S-parameters (it has to be done in this order, first the Connector+Bend, then the
    # Balun)
    ra_c, S11c, S12S21c, S22c = rc.de_embed(
        Ropen, Rshort, Rmatch, Rin_c_open, Rin_c_short, Rin_c_match, gamma_ant
    )

    # Reflection of antenna only, at the input of bend+connector
    ra_b, S11b, S12S21b, S22b = rc.de_embed(
        Ropen, Rshort, Rmatch, Rin_b_open, Rin_b_short, Rin_b_match, ra_c
    )

    def get_g(S11_rev, S12S21, ra_x, ra_y):
        return (
            np.abs(S12S21)
            * (1 - np.abs(ra_x) ** 2)
            / ((np.abs(1 - S11_rev * ra_x)) ** 2 * (1 - (np.abs(ra_y)) ** 2))
        )

    Gb = get_g(S22b, S12S21b, ra_b, ra_c)
    Gc = get_g(S22c, S12S21c, ra_c, gamma_ant)

    return Gb, Gc


def _get_loss(fname, freq, n_terms):
    gr = np.genfromtxt(fname)
    fr = gr[:, 0]
    dr = gr[:, 1]

    par = np.polyfit(fr, dr, n_terms)
    model = np.polyval(par, freq)

    return 1 - model


def ground_loss_from_beam(beam, deg_step):
    """
    Calculate ground loss from a given beam instance.

    Parameters
    ----------
    beam : instance

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

        gain_t[k] = integrate.trapz(p_in[k], dx=deg_step * np.pi / 180, axis=0)

        gain[k] = integrate.trapz(gain_t[k], dx=deg_step * np.pi / 180, axis=0)
        gain[k] = gain[k] / (4 * np.pi)
    return gain


def ground_loss(
    filename: str | Path | bool,
    freq: np.ndarray,
    beam=None,
    deg_step: float = 1.0,
    band: str | None = None,
    configuration: str = "",
):
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
        if str(filename) == ":":
            # Use the built-in loss files
            fl = "ground"
            if configuration:
                fl += "_" + configuration
            filename = Path(__file__).parent / "data" / "loss" / band / (fl + ".txt")
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
    filename: [str, Path, bool],
    freq: [np.ndarray],
    band: [None, str] = None,
    configuration: [str] = "",
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
                fl += "_" + configuration
            filename = Path(__file__).parent / "data" / "loss" / band / (fl + ".txt")
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
