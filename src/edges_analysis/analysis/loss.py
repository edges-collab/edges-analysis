import numpy as np
from edges_cal import reflection_coefficient as rc
from ..config import config


def balun_and_connector_loss(
    band, f, ra, MC=(False, False, False, False, False, False, False, False)
):
    """
    f:    frequency in MHz
    ra: reflection coefficient of antenna at the reference plane, the LNA input

    MC Switches:
    -------------------------
    MC[0] = tube_inner_radius
    MC[1] = tube_outer_radius
    MC[2] = tube_length
    MC[3] = connector_inner_radius
    MC[4] = connector_outer_radius
    MC[5] = connector_length
    MC[6] = metal_conductivity
    MC[7] = teflon_permittivity
    """
    # Angular frequency
    w = 2 * np.pi * f * 1e6

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
        "low_band3": {
            "balun_length": 43.6,
            "connector_length": 0.8,
            "er_air": 1.07,
            "ric_b": ((5 / 16) * inch2m) / 2,
            "roc_b": ((3 / 4) * inch2m) / 2,
            "roc_c": (0.16 * inch2m) / 2,
        },
        "mid_band": {
            "balun_length": 35,  # inches
            "connector_length": 0.03 / inch2m,
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
    if MC[0]:
        # 1-sigma of 3%
        ric_b *= 1 + 0.03 * np.random.normal()

    roc_b = parameters[band]["roc_b"]
    if MC[1]:
        # 1-sigma of 3%
        roc_b *= 1 + 0.03 * np.random.normal()

    l_b = parameters[band]["balun_length"] * inch2m  # length in meters
    if MC[2]:
        l_b += 0.001 * np.random.normal()  # 1-sigma of 1 mm

    # Connector dimensions
    ric_c = (0.05 * inch2m) / 2  # radius of outer wall of inner conductor
    if MC[3]:
        # 1-sigma of 3%, about < 0.04 mm
        ric_c += 0.03 * ric_c * np.random.normal()

    roc_c = parameters[band]["roc_c"]
    if MC[4]:
        # 1-sigma of 3%
        roc_c *= 1 + 0.03 * np.random.normal()

    l_c = parameters[band]["connector_length"] * inch2m  # length
    if MC[5]:
        l_c += 0.0001 * np.random.normal()

    if MC[6]:
        sigma_copper *= 1 + 0.01 * np.random.normal()
        sigma_brass *= 1 + 0.01 * np.random.normal()
        sigma_xx_inner *= 1 + 0.01 * np.random.normal()
        sigma_xx_outer *= 1 + 0.01 * np.random.normal()

    if MC[7] == 1:
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

    def get_induc_cap_res_cond_prop(ric, roc, skin_depth_inner, skin_depth_outer, u):
        L_inner = u0 * skin_depth_inner / (4 * np.pi * ric)
        L_dielec = (u / (2 * np.pi)) * np.log(roc / ric)
        L_outer = u0 * skin_depth_outer / (4 * np.pi * roc)
        L = L_inner + L_dielec + L_outer
        C = 2 * np.pi * ep_air / np.log(roc / ric)
        R = (Rs_copper / (2 * np.pi * ric)) + (Rs_brass / (2 * np.pi * roc))
        G = 2 * np.pi * w * epp_air / np.log(roc / ric)
        gamma = np.sqrt((R + 1j * w * L) * (G + 1j * w * C))
        return L, C, R, G, gamma

    # Inductance per unit length
    Lb, Cb, Rb, Gb, gamma_b = get_induc_cap_res_cond_prop(
        ric_b, roc_b, skin_depth_copper, skin_depth_brass, u_air
    )
    Lc, Cc, Rc, Gc, gamma_c = get_induc_cap_res_cond_prop(
        ric_c, roc_c, skin_depth_xx_inner, skin_depth_xx_outer, u_teflon
    )

    # Complex Cable Impedance
    Zchar_b = gamma_b
    Zchar_c = gamma_c

    # Impedance of Agilent terminations
    Zref = 50
    Ropen, Rshort, Rmatch = rc.agilent_85033E(f * 1e6, Zref, 1)

    def get_gamma(R):
        Z = rc.gamma2impedance(R, Zref)
        Zin_b = rc.input_impedance_transmission_line(Zchar_b, gamma_b, l_b, Z)
        Zin_c = rc.input_impedance_transmission_line(Zchar_c, gamma_c, l_c, Z)
        Rin_b = rc.impedance2gamma(Zin_b, Zref)
        Rin_c = rc.impedance2gamma(Zin_c, Zref)
        return Rin_b, Rin_c

    Rin_b_open, Rin_c_open = get_gamma(Ropen)
    Rin_b_short, Rin_c_short = get_gamma(Rshort)
    Rin_b_match, Rin_c_match = get_gamma(Rmatch)

    # S-parameters (it has to be done in this order, first the Connector+Bend, then the Balun)
    ra_c, S11c, S12S21c, S22c = rc.de_embed(
        Ropen, Rshort, Rmatch, Rin_c_open, Rin_c_short, Rin_c_match, ra
    )

    # Reflection of antenna only, at the input of bend+connector
    ra_b, S11b, S12S21b, S22b = rc.de_embed(
        Ropen, Rshort, Rmatch, Rin_b_open, Rin_b_short, Rin_b_match, ra_c
    )

    def get_G(S11_rev, S12S21, ra_x, ra_y):
        return (
            np.abs(S12S21)
            * (1 - np.abs(ra_x) ** 2)
            / ((np.abs(1 - S11_rev * ra_b)) ** 2 * (1 - (np.abs(ra_y)) ** 2))
        )

    Gb = get_G(S22b, S12S21b, ra_b, ra_c)
    Gc = get_G(S22c, S12S21c, ra_c, ra)

    return Gb, Gc


def _get_loss(fname, f_MHz, n_terms):
    gr = np.genfromtxt(fname)
    fr = gr[:, 0]
    dr = gr[:, 1]

    par = np.polyfit(fr, dr, n_terms)
    model = np.polyval(par, f_MHz)

    return 1 - model


def ground_loss(band, f_MHz):
    """
    f: frequency in MHz. For mid-band (low-band), between 50 and 150 (120) MHz.
    """
    if band == "low_band":
        fname = (
            config["edges_folder"]
            + "calibration/loss/low_band/ground_loss/lowband_loss_on_30x30m_two_columns.txt"
        )
    elif band == "mid_band":
        fname = (
            config["edges_folder"] + "mid_band/calibration/ground_loss/loss_column.txt"
        )
    else:
        raise ValueError("band must be low_band or mid_band")

    return _get_loss(fname, f_MHz, 8)


def antenna_loss(band, f_MHz):
    """
    f: frequency in MHz. For mid-band (low-band), between 50 and 150 (120) MHz.
    """
    if band == "mid_band":
        fname = (
            config["edges_folder"]
            + "mid_band/calibration/antenna_loss/loss_mid_ant_column.txt"
        )
    else:
        raise ValueError("only mid_band allowed for band")

    return _get_loss(fname, f_MHz, 11)
