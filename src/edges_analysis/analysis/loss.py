import numpy as np
from edges_cal import reflection_coefficient as rc

edges_folder = ""  # TODO: remove
edges_folder_v1 = ""  # TODO: remove


def balun_and_connector_loss(band, f, ra, MC=[0, 0, 0, 0, 0, 0, 0, 0]):
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

    if band == "low_band3":
        print("Balun loss model: " + band)

        # Angular frequency
        w = 2 * np.pi * f * 1e6

        # Inch-to-meters conversion
        inch2m = 1 / 39.370

        # Conductivity of copper
        sigma_copper0 = (
            5.96 * 10 ** 7
        )  # Pozar 3rd edition. Alan uses a different number. What
        # These are valid for the low-band 1 antenna
        balun_length = 43.6  # inches
        connector_length = 0.8  # inch

        # Balun dimensions
        ric_b = ((5 / 16) * inch2m) / 2  # radius of outer wall of inner conductor
        if MC[0] == 1:
            ric_b = (
                ric_b + 0.03 * ric_b * np.random.normal()
            )  # 1-sigma of 3%, about 0.04 mm

        roc_b = ((3 / 4) * inch2m) / 2  # radius of inner wall of outer conductor
        if MC[1] == 1:
            roc_b = (
                roc_b + 0.03 * roc_b * np.random.normal()
            )  # 1-sigma of 3%, about 0.08 mm

        l_b = balun_length * inch2m  # length in meters
        if MC[2] == 1:
            l_b = l_b + 0.001 * np.random.normal()  # 1-sigma of 1 mm

        # Connector dimensions
        ric_c = (0.05 * inch2m) / 2  # radius of outer wall of inner conductor
        if MC[3] == 1:
            ric_c = (
                ric_c + 0.03 * ric_c * np.random.normal()
            )  # 1-sigma of 3%, about < 0.04 mm

        roc_c = (0.16 * inch2m) / 2  # radius of inner wall of outer conductor
        if MC[4] == 1:
            roc_c = (
                roc_c + 0.03 * roc_c * np.random.normal()
            )  # 1-sigma of 3%, about 0.04 mm

        l_c = connector_length * inch2m  # length
        if MC[5] == 1:
            l_c = l_c + 0.0001 * np.random.normal()  # 1-sigma of 0.1 mm

        # Metal conductivity
        sigma_copper = 1 * sigma_copper0
        sigma_brass = 0.24 * sigma_copper0

        sigma_xx_inner = 0.24 * sigma_copper0
        sigma_xx_outer = 0.024 * sigma_copper0

        if MC[6] == 1:
            sigma_copper = (
                sigma_copper + 0.01 * sigma_copper * np.random.normal()
            )  # 1-sigma of
            # 1% of value
            sigma_brass = (
                sigma_brass + 0.01 * sigma_brass * np.random.normal()
            )  # 1-sigma of 1%
            # of value
            sigma_xx_inner = (
                sigma_xx_inner + 0.01 * sigma_xx_inner * np.random.normal()
            )  #
            # 1-sigma of 1% of value
            sigma_xx_outer = (
                sigma_xx_outer + 0.01 * sigma_xx_outer * np.random.normal()
            )  #
        # Permeability
        u0 = (
            4 * np.pi * 10 ** (-7)
        )  # permeability of free space (same for copper, brass, etc., all nonmagnetic
        ur_air = 1  # relative permeability of air
        u_air = u0 * ur_air

        ur_teflon = 1  # relative permeability of teflon
        u_teflon = u0 * ur_teflon

        # Permittivity
        c = 299792458  # speed of light
        e0 = 1 / (u0 * c ** 2)  # permittivity of free space

        er_air = 1.07  # why Alan???? shouldn't it be closer to 1 ?
        ep_air = e0 * er_air
        tan_delta_air = 0
        epp_air = ep_air * tan_delta_air

        er_teflon = 2.05  # why Alan????
        ep_teflon = e0 * er_teflon
        tan_delta_teflon = (
            0.0002  # http://www.kayelaby.npl.co.uk/general_physics/2_6/2_6_5.html
        )
        epp_teflon = ep_teflon * tan_delta_teflon

        if MC[7] == 1:
            epp_teflon = (
                epp_teflon + 0.01 * epp_teflon * np.random.normal()
            )  # 1-sigma of 1%

    elif band == "mid_band":
        # Angular frequency
        w = 2 * np.pi * f * 1e6

        # Inch-to-meters conversion
        inch2m = 1 / 39.370

        # Conductivity of copper
        sigma_copper0 = (
            5.96 * 10 ** 7
        )  # Pozar 3rd edition. Alan uses a different number. What
        # These are valid for the mid-band antenna
        balun_length = 35  # inches
        connector_length = 0.03 / inch2m  # (3 cm <-> 1.18 inch) # Fairview SC3792

        # Balun dimensions
        ric_b = ((16 / 32) * inch2m) / 2  # radius of outer wall of inner conductor
        if MC[0] == 1:
            ric_b = (
                ric_b + 0.03 * ric_b * np.random.normal()
            )  # 1-sigma of 3%, about 0.04 mm

        roc_b = ((1.25) * inch2m) / 2  # radius of inner wall of outer conductor
        if MC[1] == 1:
            roc_b = (
                roc_b + 0.03 * roc_b * np.random.normal()
            )  # 1-sigma of 3%, about 0.08 mm

        l_b = balun_length * inch2m  # length in meters
        if MC[2] == 1:
            l_b = l_b + 0.001 * np.random.normal()  # 1-sigma of 1 mm

        # Connector dimensions (Fairview SC3792)
        ric_c = (0.05 * inch2m) / 2  # radius of outer wall of inner conductor
        if MC[3] == 1:
            ric_c = (
                ric_c + 0.03 * ric_c * np.random.normal()
            )  # 1-sigma of 3%, about < 0.04 mm

        roc_c = (0.161 * inch2m) / 2  # radius of inner wall of outer conductor
        if MC[4] == 1:
            roc_c = (
                roc_c + 0.03 * roc_c * np.random.normal()
            )  # 1-sigma of 3%, about 0.04 mm

        l_c = connector_length * inch2m  # length    in meters
        if MC[5] == 1:
            l_c = l_c + 0.0001 * np.random.normal()  # 1-sigma of 0.1 mm

        # Metal conductivity
        sigma_copper = 1 * sigma_copper0
        sigma_brass = 0.24 * sigma_copper0

        sigma_xx_inner = 0.24 * sigma_copper0
        sigma_xx_outer = 0.024 * sigma_copper0

        if MC[6] == 1:
            sigma_copper = (
                sigma_copper + 0.01 * sigma_copper * np.random.normal()
            )  # 1-sigma of
            # 1% of value
            sigma_brass = (
                sigma_brass + 0.01 * sigma_brass * np.random.normal()
            )  # 1-sigma of 1%
            # of value
            sigma_xx_inner = (
                sigma_xx_inner + 0.01 * sigma_xx_inner * np.random.normal()
            )  #
            # 1-sigma of 1% of value
            sigma_xx_outer = (
                sigma_xx_outer + 0.01 * sigma_xx_outer * np.random.normal()
            )  #
            # 1-sigma of 1% of value

        # Permeability
        u0 = (
            4 * np.pi * 10 ** (-7)
        )  # permeability of free space (same for copper, brass, etc., all nonmagnetic
        ur_air = 1  # relative permeability of air
        u_air = u0 * ur_air

        ur_teflon = 1  # relative permeability of teflon
        u_teflon = u0 * ur_teflon

        # Permittivity
        c = 299792458  # speed of light
        e0 = 1 / (u0 * c ** 2)  # permittivity of free space

        er_air = (
            1.2  # Question for Alan. Why this value ???? shouldn't it be closer to 1 ?
        )
        ep_air = e0 * er_air
        tan_delta_air = 0
        epp_air = ep_air * tan_delta_air

        er_teflon = 2.05  # why Alan????
        ep_teflon = e0 * er_teflon
        tan_delta_teflon = (
            0.0002  # http://www.kayelaby.npl.co.uk/general_physics/2_6/2_6_5.html
        )
        epp_teflon = ep_teflon * tan_delta_teflon

        if MC[7] == 1:
            epp_teflon = (
                epp_teflon + 0.01 * epp_teflon * np.random.normal()
            )  # 1-sigma of 1%

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

    # Inductance per unit length
    Lb_inner = u0 * skin_depth_copper / (4 * np.pi * ric_b)
    Lb_dielec = (u_air / (2 * np.pi)) * np.log(roc_b / ric_b)
    Lb_outer = u0 * skin_depth_brass / (4 * np.pi * roc_b)
    Lb = Lb_inner + Lb_dielec + Lb_outer

    # Capacitance per unit length
    Cb = 2 * np.pi * ep_air / np.log(roc_b / ric_b)

    # Resistance per unit length
    Rb = (Rs_copper / (2 * np.pi * ric_b)) + (Rs_brass / (2 * np.pi * roc_b))

    # Conductance per unit length
    Gb = 2 * np.pi * w * epp_air / np.log(roc_b / ric_b)

    # Inductance per unit length
    Lc_inner = u0 * skin_depth_xx_inner / (4 * np.pi * ric_c)
    Lc_dielec = (u_teflon / (2 * np.pi)) * np.log(roc_c / ric_c)
    Lc_outer = u0 * skin_depth_xx_outer / (4 * np.pi * roc_c)
    Lc = Lc_inner + Lc_dielec + Lc_outer

    # Capacitance per unit length
    Cc = 2 * np.pi * ep_teflon / np.log(roc_c / ric_c)

    # Resistance per unit length
    Rc = (Rs_xx_inner / (2 * np.pi * ric_c)) + (Rs_xx_outer / (2 * np.pi * roc_c))

    # Conductance per unit length
    Gc = 2 * np.pi * w * epp_teflon / np.log(roc_c / ric_c)

    # Propagation constant
    gamma_b = np.sqrt((Rb + 1j * w * Lb) * (Gb + 1j * w * Cb))
    gamma_c = np.sqrt((Rc + 1j * w * Lc) * (Gc + 1j * w * Cc))

    # Complex Cable Impedance
    Zchar_b = np.sqrt((Rb + 1j * w * Lb) / (Gb + 1j * w * Cb))
    Zchar_c = np.sqrt((Rc + 1j * w * Lc) / (Gc + 1j * w * Cc))

    # Impedance of Agilent terminations
    Zref = 50
    Ropen, Rshort, Rmatch = rc.agilent_85033E(f * 1e6, Zref, 1)
    Zopen = rc.gamma2impedance(Ropen, Zref)
    Zshort = rc.gamma2impedance(Rshort, Zref)
    Zmatch = rc.gamma2impedance(Rmatch, Zref)

    # Impedance of terminated transmission lines
    Zin_b_open = rc.input_impedance_transmission_line(Zchar_b, gamma_b, l_b, Zopen)
    Zin_b_short = rc.input_impedance_transmission_line(Zchar_b, gamma_b, l_b, Zshort)
    Zin_b_match = rc.input_impedance_transmission_line(Zchar_b, gamma_b, l_b, Zmatch)

    Zin_c_open = rc.input_impedance_transmission_line(Zchar_c, gamma_c, l_c, Zopen)
    Zin_c_short = rc.input_impedance_transmission_line(Zchar_c, gamma_c, l_c, Zshort)
    Zin_c_match = rc.input_impedance_transmission_line(Zchar_c, gamma_c, l_c, Zmatch)

    # Reflection of terminated transmission lines
    Rin_b_open = rc.impedance2gamma(Zin_b_open, Zref)
    Rin_b_short = rc.impedance2gamma(Zin_b_short, Zref)
    Rin_b_match = rc.impedance2gamma(Zin_b_match, Zref)

    Rin_c_open = rc.impedance2gamma(Zin_c_open, Zref)
    Rin_c_short = rc.impedance2gamma(Zin_c_short, Zref)
    Rin_c_match = rc.impedance2gamma(Zin_c_match, Zref)

    # S-parameters (it has to be done in this order, first the Connector+Bend, then the Balun)
    ra_c, S11c, S12S21c, S22c = rc.de_embed(
        Ropen, Rshort, Rmatch, Rin_c_open, Rin_c_short, Rin_c_match, ra
    )  # Reflection of antenna + balun, at the input of
    # bend+connector
    ra_b, S11b, S12S21b, S22b = rc.de_embed(
        Ropen, Rshort, Rmatch, Rin_b_open, Rin_b_short, Rin_b_match, ra_c
    )  # Reflection of antenna only, at the input of
    # Inverting S11 and S22
    S11b_rev = S22b

    S11c_rev = S22c

    # Absolute value of S_21
    abs_S21b = np.sqrt(np.abs(S12S21b))
    abs_S21c = np.sqrt(np.abs(S12S21c))

    # Available Power Gain (Gain Factor, also known as Loss Factor)
    Gb = (
        (abs_S21b ** 2)
        * (1 - np.abs(ra_b) ** 2)
        / ((np.abs(1 - S11b_rev * ra_b)) ** 2 * (1 - (np.abs(ra_c)) ** 2))
    )
    Gc = (
        (abs_S21c ** 2)
        * (1 - np.abs(ra_c) ** 2)
        / ((np.abs(1 - S11c_rev * ra_c)) ** 2 * (1 - (np.abs(ra)) ** 2))
    )

    return Gb, Gc


def ground_loss(band, f_MHz):
    """
    f_MHz: frequency in MHz. For mid-band (low-band), between 50 and 150 (120) MHz.
    """

    if band == "low_band":
        gr = np.genfromtxt(
            edges_folder_v1
            + "calibration/loss/low_band/ground_loss/lowband_loss_on_30x30m_two_columns.txt"
        )

        fr = gr[:, 0]
        dr = gr[:, 1]

        par = np.polyfit(fr, dr, 8)  # 7 terms are sufficient
        model = np.polyval(par, f_MHz)

    elif band == "mid_band":
        gr = np.genfromtxt(
            edges_folder + "mid_band/calibration/ground_loss/loss_column.txt"
        )

        fr = gr[:, 0]
        dr = gr[:, 1]

        par = np.polyfit(fr, dr, 8)  # 7 terms are sufficient
        model = np.polyval(par, f_MHz)

    return 1 - model


def antenna_loss(band, f_MHz):
    """
    Dec 7, 2019
    """

    if band == "mid_band":
        d = np.genfromtxt(
            edges_folder + "mid_band/calibration/antenna_loss/loss_mid_ant_column.txt"
        )

        fr = d[:, 0]
        dr = d[:, 1]

        par = np.polyfit(fr, dr, 11)  # 7 terms are sufficient
        model = np.polyval(par, f_MHz)
    else:
        raise ValueError("only mid_band allowed for band")

    return 1 - model
