import datetime as dt

import astropy.coordinates as apc
import astropy.time as apt
import astropy.units as apu
import h5py
import matplotlib.pyplot as plt
import numpy as np
import scipy.interpolate as spi

from . import coordinates as coords
from .loss import ground_loss
from .sky_models import (
    LW_150MHz_map,
    guzman_45MHz_map,
    haslam_408MHz_map,
    remazeilles_408MHz_map,
)

edges_folder = ""  # TODO: get rid of this


def hfss_read(
    path_to_file,
    dB_or_linear,
    theta_min=0,
    theta_max=180,
    theta_resolution=1,
    phi_min=0,
    phi_max=359,
    phi_resolution=1,
):
    """"""

    d = np.genfromtxt(path_to_file, skip_header=1, delimiter=",")
    theta = np.arange(theta_min, theta_max + theta_resolution, theta_resolution)
    phi = np.arange(phi_min, phi_max + phi_resolution, phi_resolution)

    beam_map = np.zeros((len(theta), len(phi)))
    for i in range(len(theta)):

        phiX = d[d[:, 1] == theta[i], 0]
        powerX = d[d[:, 1] == theta[i], 2]

        phiX[phiX < 0] = phiX[phiX < 0] + 360
        phiX, iX = np.unique(phiX, axis=0, return_index=True)
        powerX = powerX[iX]

        iY = np.argsort(phiX)
        powerY = powerX[iY]

        if dB_or_linear == "dB":
            linear_power = 10 ** (powerY / 10)

        elif dB_or_linear == "linear":
            linear_power = np.copy(powerY)

        linear_power[np.isnan(linear_power)] = 0
        beam_map[i, :] = linear_power

    return theta, phi, beam_map


def wipld_read(filename, AZ_antenna_axis=0):
    with open(filename) as fn:

        file_length = 0
        number_of_frequencies = 0

        flag_columns = 0
        frequencies_list = []
        for line in fn:
            file_length += 1
            if line[2] == ">":
                number_of_frequencies += 1
                frequencies_list.append(float(line[19:32]))
                print(line)

            if (line[2] != ">") and (flag_columns == 0):
                line_splitted = line.split()
                number_of_columns = len(line_splitted)
                flag_columns = 1

        rows_per_frequency = (
            file_length - number_of_frequencies
        ) / number_of_frequencies

        print(file_length)
        print(int(number_of_frequencies))
        print(int(rows_per_frequency))
        print(int(number_of_columns))

        output = np.zeros(
            (
                int(number_of_frequencies),
                int(rows_per_frequency),
                int(number_of_columns),
            )
        )

    frequencies = np.array(frequencies_list)

    with open(filename) as fn:
        i = -1
        for line in fn:

            if line[2] == ">":
                i += 1
                j = -1

                print(line)
            else:
                j = j + 1
                line_splitted = line.split()
                line_array = np.array(line_splitted, dtype=float)

                output[i, j, :] = line_array
    # Rearranging data
    # ----------------
    phi_u = np.unique(output[0, :, 0])
    theta_u = np.unique(output[0, :, 1])

    beam = np.zeros((len(frequencies), len(theta_u), len(phi_u)))

    for i in range(len(frequencies)):
        out_2D = output[i, :, :]

        phi = out_2D[:, 0]
        theta = 90 - out_2D[:, 1]  # theta is zero at the zenith, and goes to 180 deg
        gain = out_2D[:, 6]

        theta_u = np.unique(theta)
        it = np.argsort(theta_u)
        theta_a = theta_u[it]

        for j in range(len(theta_a)):
            # print(theta_a[j])

            phi_j = phi[theta == theta_a[j]]
            gain_j = gain[theta == theta_a[j]]

            ip = np.argsort(phi_j)
            gp = gain_j[ip]

            beam[i, j, :] = gp

    # Flip beam from theta to elevation
    # ---------------------------------
    beam_maps = beam[:, ::-1, :]

    # Change coordinates from theta/phi, to AZ/EL
    # -------------------------------------------
    EL = np.arange(0, 91)  # , dtype='uint32')
    AZ = np.arange(0, 360)  # , dtype='uint32')

    # Shifting beam relative to true AZ (referenced at due North)
    # Due to angle of orientation of excited antenna panels relative to due North
    # ---------------------------------------------------------
    print("AZ_antenna_axis = " + str(AZ_antenna_axis) + " deg")
    if AZ_antenna_axis < 0:
        AZ_index = -AZ_antenna_axis
        bm1 = beam_maps[:, :, AZ_index::]
        bm2 = beam_maps[:, :, 0:AZ_index]
        beam_maps_shifted = np.append(bm1, bm2, axis=2)

    elif AZ_antenna_axis > 0:
        AZ_index = AZ_antenna_axis
        bm1 = beam_maps[:, :, 0:(-AZ_index)]
        bm2 = beam_maps[:, :, (360 - AZ_index) : :]
        beam_maps_shifted = np.append(bm2, bm1, axis=2)

    elif AZ_antenna_axis == 0:
        beam_maps_shifted = np.copy(beam_maps)

    return frequencies, AZ, EL, beam_maps_shifted


def antenna_beam_factor(
    band,
    name_save,
    FLOW=50,
    FHIGH=200,
    beam_file=0,
    normalize_mid_band_beam=True,
    sky_model="haslam",
    rotation_from_north=90,
    index_model="gaussian",
    sigma_deg=8.5,
    index_center=2.4,
    index_pole=2.65,
    band_deg=10,
    index_inband=2.5,
    index_outband=2.6,
    reference_frequency=100,
    convolution_computation="old",
    sky_plots=False,
):
    """
    2019-04-16

    band                      : 'mid_band'
    sky_model                 : 'haslam', 'LW', 'guzman'

    index_model               : 'gaussian' (default), or 'step'
    parameters for 'gaussian' :  sigma_deg=8.5, index_center=2.4, index_pole=2.65
    parameters for 'step'     :  band_deg=10, index_inband=2.5, index_outband=2.6

    Example:


    [101] :   o = cal.antenna_beam_factor('mid_band',
    'mid_band_50-200MHz_90deg_alan0_haslam_gaussian_index_2.4_2.65_sigma_deg_8.5_reffreq_90MHz',
    beam_file=0, sky_model='haslam', rotation_from_north=90, index_model='gaussian',
    sigma_deg=8.5, index_center=2.4, index_pole=2.65, reference_frequency=90)

    """
    # Data paths
    path_save = edges_folder + band + "/calibration/beam_factors/raw/"
    path_plots = path_save + "plots/"

    # Antenna beam
    # ---------------------------------------------------------------------------
    AZ_beam = np.arange(0, 360)
    EL_beam = np.arange(0, 91)

    # FEKO blade beam
    # Fixing rotation angle due to diferent rotation (by 90deg) in Nivedita's map
    if (band == "mid_band") and (beam_file == 100):
        rotation_from_north = rotation_from_north - 90

    # Best case, Feb 20, 2019
    if ((band == "mid_band") and (beam_file <= 100)) or (band == "low_band3"):
        beam_all_X = feko_blade_beam(
            band, beam_file, AZ_antenna_axis=rotation_from_north
        )

    # Beams from WIPL-D
    elif (band == "mid_band") and (beam_file > 100):
        if beam_file == 101:
            filename = (
                edges_folder + "/others/beam_simulations/wipl-d/20191030"
                "/blade_dipole_infinite_soil_real_metal_GP_30mx30m.ra1"
            )

        elif beam_file == 102:
            filename = (
                edges_folder
                + "/others/beam_simulations/wipl-d/20191124/mid_band_perf_30x30_5mm_wire_.ra1"
            )

        elif beam_file == 103:
            filename = (
                edges_folder + "/others/beam_simulations/wipl-d/20191124"
                "/mid_band_perf_30x30_5mm_wire_no_soil_conductivity.ra1"
            )

        freq_array_X, AZ_beam, EL_beam, beam_all_X = wipld_read(
            filename, AZ_antenna_axis=rotation_from_north
        )

    # Frequency array
    if band == "mid_band":
        if beam_file == 0:  # Best case, Feb 20, 2019
            # ALAN #0
            freq_array_X = np.arange(50, 201, 2, dtype="uint32")

        elif beam_file == 1:
            # ALAN #1
            freq_array_X = np.arange(50, 201, 2, dtype="uint32")

        elif beam_file == 2:
            # ALAN #2
            freq_array_X = np.arange(50, 201, 2, dtype="uint32")

        elif beam_file == 100:
            # NIVEDITA
            freq_array_X = np.arange(60, 201, 2, dtype="uint32")

    elif band == "low_band3":
        if beam_file == 1:
            freq_array_X = np.arange(50, 121, 2, dtype="uint32")

    elif band == "low_band":
        if beam_file == 2:
            freq_array_X = np.arange(40, 121, 2, dtype="uint32")

    # Selecting frequency range
    if band == "mid_band" and normalize_mid_band_beam:  # Beam normalization
        FLOW = 50
        FHIGH = 150
    freq_array = freq_array_X[(freq_array_X >= FLOW) & (freq_array_X <= FHIGH)]
    beam_all = beam_all_X[(freq_array_X >= FLOW) & (freq_array_X <= FHIGH), :, :]

    ground_gain = ground_loss("mid_band", freq_array)

    print(beam_all.shape)
    print(np.max(beam_all))

    # Index of reference frequency
    index_freq_array = np.arange(len(freq_array))
    irf = index_freq_array[freq_array == reference_frequency]
    print("Reference frequency: " + str(freq_array[irf][0]) + " MHz")

    # Sky map
    # ------------------------------------------------------------------
    if sky_model == "haslam":
        # Loading Haslam map
        map_orig, lon, lat, GALAC_COORD_object = haslam_408MHz_map()
        v0 = 408

    if sky_model == "remazeilles":

        # Loading Remazeilles map
        map_orig, lon, lat, GALAC_COORD_object = remazeilles_408MHz_map()
        v0 = 408

    elif sky_model == "LW":

        # Loading LW map
        map_orig, lon, lat, GALAC_COORD_object = LW_150MHz_map()
        v0 = 150

    elif sky_model == "guzman":

        # Loading Guzman map
        map_orig, lon, lat, GALAC_COORD_object = guzman_45MHz_map()
        v0 = 45

    print(v0)

    # Scaling sky map (the map contains the CMB, which has to be removed and then added back)
    # ---------------------------------------------------------------------------------------
    if index_model == "gaussian":
        index = index_pole - (index_pole - index_center) * np.exp(
            -(1 / 2) * (np.abs(lat) / sigma_deg) ** 2
        )

    if index_model == "step":
        index = np.zeros(len(lat))
        index[np.abs(lat) <= band_deg] = index_inband
        index[np.abs(lat) > band_deg] = index_outband

    Tcmb = 2.725
    sky_map = np.zeros((len(map_orig), len(freq_array)))
    for i in range(len(freq_array)):
        sky_map[:, i] = (map_orig - Tcmb) * (freq_array[i] / v0) ** (-index) + Tcmb

    # for i in range(len(freq_array)):

    ## Band of the Galactic center, using spectral index
    # sky_map[(lat >= -band_deg) & (lat <= band_deg), i] = (map_orig - Tcmb)[(lat >= -band_deg) &
    # (lat <= band_deg)] * (freq_array[i]/v0)**(-index_inband) + Tcmb

    ## Range outside the Galactic center, using second spectral index
    # sky_map[(lat < -band_deg) | (lat > band_deg), i]   = (map_orig - Tcmb)[(lat < -band_deg) | (
    # lat > band_deg)] * (freq_array[i]/v0)**(-index_outband) + Tcmb

    # Calculation
    # --------------------------------------------------------------------------------------

    # EDGES location
    EDGES_lat_deg = -26.714778  # MARS: 79.43   # GUADALUPE: 29   # PATAGONIA -47.1
    EDGES_lon_deg = 116.605528
    EDGES_location = apc.EarthLocation(
        lat=EDGES_lat_deg * apu.deg, lon=EDGES_lon_deg * apu.deg
    )

    # Reference UTC observation time. At this time, the LST is 0.1666 (00:10 Hrs LST) at the
    # EDGES location (it was wrong before, now it is correct)
    Time_iter = np.array([2014, 1, 1, 9, 39, 42])
    Time_iter_dt = dt.datetime(
        Time_iter[0],
        Time_iter[1],
        Time_iter[2],
        Time_iter[3],
        Time_iter[4],
        Time_iter[5],
    )

    # Looping over LST
    LST = np.zeros(72)
    convolution_ref = np.zeros((len(LST), len(beam_all[:, 0, 0])))
    antenna_temperature_above_horizon = np.zeros((len(LST), len(beam_all[:, 0, 0])))
    loss_fraction = np.zeros((len(LST), len(beam_all[:, 0, 0])))

    # for i in range(2):
    for i in range(len(LST)):

        print(name_save + ". LST: " + str(i + 1) + " out of 72")

        # Advancing time ( 19:57 minutes UTC correspond to 20 minutes LST )
        minutes_offset = 19
        seconds_offset = 57
        if i > 0:
            Time_iter_dt = Time_iter_dt + dt.timedelta(
                minutes=minutes_offset, seconds=seconds_offset
            )
            Time_iter = np.array(
                [
                    Time_iter_dt.year,
                    Time_iter_dt.month,
                    Time_iter_dt.day,
                    Time_iter_dt.hour,
                    Time_iter_dt.minute,
                    Time_iter_dt.second,
                ]
            )

        # LST
        LST[i] = coords.utc2lst(Time_iter, EDGES_lon_deg)

        # Transforming Galactic coordinates of Sky to Local coordinates
        altaz = GALAC_COORD_object.transform_to(
            apc.AltAz(
                location=EDGES_location,
                obstime=apt.Time(Time_iter_dt, format="datetime"),
            )
        )
        AZ = np.asarray(altaz.az)
        EL = np.asarray(altaz.alt)

        # Selecting coordinates above the horizon
        AZ_above_horizon = AZ[EL >= 0]
        EL_above_horizon = EL[EL >= 0]

        # Selecting sky data above the horizon
        sky_above_horizon = sky_map[EL >= 0, :]
        sky_ref_above_horizon = sky_above_horizon[:, irf].flatten()

        # Plotting sky in local coordinates
        if sky_plots:

            LAT_DEG = np.copy(EDGES_lat_deg)

            AZ_plot = np.copy(AZ_above_horizon)
            AZ_plot[AZ_plot > 180] = AZ_plot[AZ_plot > 180] - 360

            EL_plot = np.copy(EL_above_horizon)
            SKY_plot = np.copy(sky_ref_above_horizon)

            max_log10sky = np.max(np.log10(sky_map[:, irf]))
            min_log10sky = np.min(np.log10(sky_map[:, irf]))

            marker_size = 10

            LST_gc = 17 + (45 / 60) + (40.04 / (60 * 60))  # LST of Galactic Center
            GHA = LST[i] - LST_gc
            if GHA < 0:
                GHA = GHA + 24

            plot_format = "polar"

            if plot_format == "rect":
                plt.close()
                plt.figure(figsize=[19, 6])
                plt.scatter(
                    AZ_plot,
                    EL_plot,
                    edgecolors="none",
                    s=marker_size,
                    c=np.log10(SKY_plot),
                    vmin=min_log10sky,
                    vmax=max_log10sky,
                )
                plt.xticks(np.arange(-180, 181, 30))
                plt.yticks([0, 15, 30, 45, 60, 75, 90])
                cbar = plt.colorbar()
                cbar.set_label("log10( Tsky @ 50MHz [K] )", rotation=90)
                plt.xlabel("AZ [deg]")
                plt.ylabel("EL [deg]")
                plt.title(
                    "LAT="
                    + str(np.round(LAT_DEG, 3))
                    + " [deg] \n\n LST="
                    + str(np.round(LST[i], 3)).ljust(5, "0")
                    + " hr        GHA="
                    + str(np.round(GHA, 3)).ljust(5, "0")
                    + " hr"
                )

            if plot_format == "polar":
                plt.close()
                fig = plt.figure(figsize=[11.5, 10])
                ax = fig.add_subplot(111, projection="polar")
                c = ax.scatter(
                    (np.pi / 180) * AZ_plot,
                    90 - EL_plot,
                    edgecolors="none",
                    s=marker_size,
                    c=np.log10(SKY_plot),
                    vmin=min_log10sky,
                    vmax=5,
                )  # , vmin=min_log10sky, vmax=max_log10sky)
                ax.set_theta_offset(-np.pi / 2)
                ax.set_ylim([0, 90])
                ax.set_yticks([0, 30, 60, 90])
                ax.set_yticklabels(["90", "60", "30", "0"])
                plt.text(-2 * (np.pi / 180), 101, "AZ", fontsize=14, fontweight="bold")
                plt.text(22 * (np.pi / 180), 95, "EL", fontsize=14, fontweight="bold")
                plt.text(
                    45 * (np.pi / 180),
                    143,
                    "Raul Monsalve",
                    fontsize=8,
                    color=[0.5, 0.5, 0.5],
                )
                plt.title(
                    "LAT="
                    + str(np.round(LAT_DEG, 3))
                    + " [deg] \n\n LST="
                    + str(np.round(LST[i], 3)).ljust(5, "0")
                    + " [hr]        GHA="
                    + str(np.round(GHA, 3)).ljust(5, "0")
                    + " [hr]",
                    fontsize=14,
                    fontweight="bold",
                )
                # plt.xticks(np.arange(-180, 181, 30))
                # plt.yticks([0,15,30,45,60,75,90])
                cbar_ax = fig.add_axes([0.9, 0.3, 0.02, 0.4])
                hcbar = fig.colorbar(c, cax=cbar_ax)
                # cbar = ax.colorbar()
                hcbar.set_label("log10( Tsky @ 50MHz [K] )", rotation=90)
            # plt.xlabel('AZ [deg]')
            # plt.ylabel('EL [deg]')

            plt.savefig(
                path_plots + "LST_" + str(np.round(LST[i], 3)).ljust(5, "0") + "hr.png",
                bbox_inches="tight",
            )
            plt.close()

        # Arranging AZ and EL arrays corresponding to beam model
        az_array = np.tile(AZ_beam, 91)
        el_array = np.repeat(EL_beam, 360)
        az_el_original = np.array([az_array, el_array]).T
        az_el_above_horizon = np.array([AZ_above_horizon, EL_above_horizon]).T

        # Loop over frequency
        for j in range(len(freq_array)):

            print(
                name_save
                + ", Freq: "
                + str(j)
                + " out of "
                + str(len(beam_all[:, 0, 0]))
            )

            beam_array = beam_all[j, :, :].reshape(1, -1)[0]
            beam_above_horizon = spi.griddata(
                az_el_original, beam_array, az_el_above_horizon, method="cubic"
            )  # interpolated beam

            no_nan_array = np.ones(len(AZ_above_horizon)) - np.isnan(beam_above_horizon)
            index_no_nan = np.nonzero(no_nan_array)[0]

            sky_above_horizon_ff = sky_above_horizon[:, j].flatten()

            # Normalization only above the horizon, frequency-by-frequency
            if convolution_computation == "old":

                # Convolution and Antenna temperature OLD 'incorrect' WAY
                # -------------------------------------------------------

                # Convolution between (beam at all frequencies) and (sky at reference frequency)
                convolution_ref[i, j] = np.sum(
                    beam_above_horizon[index_no_nan]
                    * sky_ref_above_horizon[index_no_nan]
                ) / np.sum(beam_above_horizon[index_no_nan])

                # Antenna temperature, i.e., Convolution between (beam at all frequencies) and (
                # sky at all frequencies)
                antenna_temperature_above_horizon[i, j] = np.sum(
                    beam_above_horizon[index_no_nan]
                    * sky_above_horizon_ff[index_no_nan]
                ) / np.sum(beam_above_horizon[index_no_nan])

                loss_fraction[i, j] = 0

            # Normalization to 4pi, constant in frequency
            elif convolution_computation == "new":

                # Convolution and Antenna temperature NEW 'correct' WAY
                # -----------------------------------------------------
                # Number of pixels over 4pi that are not 'nan'
                Npixels_total = len(EL)
                Npixels_above_horizon_nan = len(EL_above_horizon) - len(
                    EL_above_horizon[index_no_nan]
                )  # The nans are only above the horizon
                Npixels_total_no_nan = Npixels_total - Npixels_above_horizon_nan

                if normalize_mid_band_beam:
                    SOLID_ANGLE = (
                        np.sum(beam_above_horizon[index_no_nan]) / Npixels_total_no_nan
                    )
                    NORMALIZED_BEAM_ABOVE_HORIZON = (
                        ground_gain[j] / SOLID_ANGLE
                    ) * beam_above_horizon
                else:
                    NORMALIZED_BEAM_ABOVE_HORIZON = np.copy(beam_above_horizon)

                # Convolution between (beam at all frequencies) and (sky at reference frequency)
                convolution_ref[i, j] = np.sum(
                    NORMALIZED_BEAM_ABOVE_HORIZON[index_no_nan]
                    * sky_ref_above_horizon[index_no_nan]
                )

                # 'Correct' antenna temperature above the horizon, i.e., Convolution between (beam
                # at all frequencies) and (sky at all frequencies)
                antenna_temperature_above_horizon[i, j] = (
                    np.sum(
                        NORMALIZED_BEAM_ABOVE_HORIZON[index_no_nan]
                        * sky_above_horizon_ff[index_no_nan]
                    )
                    / Npixels_total_no_nan
                )

                # Loss fraction
                loss_fraction[i, j] = 1 - (
                    np.sum(NORMALIZED_BEAM_ABOVE_HORIZON[index_no_nan])
                    / Npixels_total_no_nan
                )

    # if i == 0:
    # integrated_gain_above_horizon[j] = np.sum(beam_above_horizon[index_no_nan])

    # if j == 0:
    # pixels_above_horizon[i]          = len(beam_above_horizon[index_no_nan])

    # numerator[i,j] = np.sum(beam_above_horizon[index_no_nan]*sky_above_horizon_ff[
    # index_no_nan])

    # Beam factor
    # -------------------------------
    beam_factor_T = convolution_ref.T / convolution_ref[:, irf].T
    beam_factor = beam_factor_T.T

    # Saving
    # ---------------------------------------------------------
    np.savetxt(path_save + name_save + "_freq.txt", freq_array)
    np.savetxt(path_save + name_save + "_LST.txt", LST)
    np.savetxt(path_save + name_save + "_tant.txt", antenna_temperature_above_horizon)
    np.savetxt(path_save + name_save + "_loss.txt", loss_fraction)
    np.savetxt(path_save + name_save + "_beam_factor.txt", beam_factor)

    return 0


def antenna_beam_factor_interpolation(band, case, lst_hires, fnew, Npar_freq=15):
    """

    For Mid-Band, over 50-200MHz, we have to use Npar_freq=15

    Here, "case" is not the same as in the FEKO... function

    """

    if band == "low_band3":
        file_path = edges_folder + "low_band3/calibration/beam_factors/raw/"

        bf_old = np.genfromtxt(
            file_path
            + "low_band3_50-120MHz_85deg_alan_haslam_2.5_2.62_reffreq_76MHz_data.txt"
        )
        freq = np.genfromtxt(
            file_path
            + "low_band3_50-120MHz_85deg_alan_haslam_2.5_2.62_reffreq_76MHz_freq.txt"
        )
        lst_old = np.genfromtxt(
            file_path
            + "low_band3_50-120MHz_85deg_alan_haslam_2.5_2.62_reffreq_76MHz_LST.txt"
        )
    elif band == "mid_band":
        file_path = edges_folder + "mid_band/calibration/beam_factors/raw/"

        if case == 0:
            bf_old = np.genfromtxt(
                file_path + "mid_band_50-200MHz_90deg_alan0_haslam_gaussian_index_2.4_2"
                ".65_sigma_deg_8.5_reffreq_90MHz_data.txt"
            )
            freq = np.genfromtxt(
                file_path + "mid_band_50-200MHz_90deg_alan0_haslam_gaussian_index_2.4_2"
                ".65_sigma_deg_8.5_reffreq_90MHz_freq.txt"
            )
            lst_old = np.genfromtxt(
                file_path + "mid_band_50-200MHz_90deg_alan0_haslam_gaussian_index_2.4_2"
                ".65_sigma_deg_8.5_reffreq_90MHz_LST.txt"
            )
        elif case == 1:
            bf_old = np.genfromtxt(
                file_path + "mid_band_50-200MHz_90deg_alan0_haslam_flat_index_2"
                ".56_reffreq_90MHz_data.txt"
            )
            freq = np.genfromtxt(
                file_path + "mid_band_50-200MHz_90deg_alan0_haslam_flat_index_2"
                ".56_reffreq_90MHz_freq.txt"
            )
            lst_old = np.genfromtxt(
                file_path + "mid_band_50-200MHz_90deg_alan0_haslam_flat_index_2"
                ".56_reffreq_90MHz_LST.txt"
            )
        elif case == 10:
            bf_old = np.genfromtxt(
                file_path
                + "NORMALIZED_mid_band_50-150MHz_90deg_alan0_haslam_gaussian_index_2"
                ".4_2.65_sigma_deg_8.5_reffreq_90MHz_beam_factor.txt"
            )
            freq = np.genfromtxt(
                file_path
                + "NORMALIZED_mid_band_50-150MHz_90deg_alan0_haslam_gaussian_index_2"
                ".4_2.65_sigma_deg_8.5_reffreq_90MHz_freq.txt"
            )
            lst_old = np.genfromtxt(
                file_path
                + "NORMALIZED_mid_band_50-150MHz_90deg_alan0_haslam_gaussian_index_2"
                ".4_2.65_sigma_deg_8.5_reffreq_90MHz_LST.txt"
            )
        elif case == 2:
            bf_old = np.genfromtxt(
                file_path + "mid_band_50-200MHz_90deg_alan0_LW_gaussian_index_2.4_2"
                ".65_sigma_deg_8.5_reffreq_90MHz_data.txt"
            )
            freq = np.genfromtxt(
                file_path + "mid_band_50-200MHz_90deg_alan0_LW_gaussian_index_2.4_2"
                ".65_sigma_deg_8.5_reffreq_90MHz_freq.txt"
            )
            lst_old = np.genfromtxt(
                file_path + "mid_band_50-200MHz_90deg_alan0_LW_gaussian_index_2.4_2"
                ".65_sigma_deg_8.5_reffreq_90MHz_LST.txt"
            )
        elif case == 3:
            bf_old = np.genfromtxt(
                file_path + "mid_band_50-200MHz_90deg_alan0_guzman_gaussian_index_2.4_2"
                ".65_sigma_deg_8.5_reffreq_90MHz_data.txt"
            )
            freq = np.genfromtxt(
                file_path + "mid_band_50-200MHz_90deg_alan0_guzman_gaussian_index_2.4_2"
                ".65_sigma_deg_8.5_reffreq_90MHz_freq.txt"
            )
            lst_old = np.genfromtxt(
                file_path + "mid_band_50-200MHz_90deg_alan0_guzman_gaussian_index_2.4_2"
                ".65_sigma_deg_8.5_reffreq_90MHz_LST.txt"
            )
        elif case == 4:
            bf_old = np.genfromtxt(
                file_path + "mid_band_50-200MHz_90deg_alan1_haslam_gaussian_index_2.4_2"
                ".65_sigma_deg_8.5_reffreq_90MHz_data.txt"
            )
            freq = np.genfromtxt(
                file_path + "mid_band_50-200MHz_90deg_alan1_haslam_gaussian_index_2.4_2"
                ".65_sigma_deg_8.5_reffreq_90MHz_freq.txt"
            )
            lst_old = np.genfromtxt(
                file_path + "mid_band_50-200MHz_90deg_alan1_haslam_gaussian_index_2.4_2"
                ".65_sigma_deg_8.5_reffreq_90MHz_LST.txt"
            )
        elif case == 5:
            bf_old = np.genfromtxt(
                file_path + "mid_band_50-200MHz_90deg_alan0_haslam_gaussian_index_2.4_2"
                ".65_sigma_deg_8.5_reffreq_120MHz_data.txt"
            )
            freq = np.genfromtxt(
                file_path + "mid_band_50-200MHz_90deg_alan0_haslam_gaussian_index_2.4_2"
                ".65_sigma_deg_8.5_reffreq_120MHz_freq.txt"
            )
            lst_old = np.genfromtxt(
                file_path + "mid_band_50-200MHz_90deg_alan0_haslam_gaussian_index_2.4_2"
                ".65_sigma_deg_8.5_reffreq_120MHz_LST.txt"
            )
    # Wrap beam factor and LST for 24-hr interpolation
    bf = np.vstack((bf_old[-1, :], bf_old, bf_old[0, :]))
    lst0 = np.append(lst_old[-1] - 24, lst_old)
    lst = np.append(lst0, lst_old[0] + 24)

    # Arranging original arrays in preparation for interpolation
    freq_array = np.tile(freq, len(lst))
    lst_array = np.repeat(lst, len(freq))
    bf_array = bf.reshape(1, -1)[0]
    freq_lst_original = np.array([freq_array, lst_array]).T

    # Producing high-resolution array of LSTs (frequencies are the same as the original)
    freq_hires = np.copy(freq)
    freq_hires_array = np.tile(freq_hires, len(lst_hires))
    lst_hires_array = np.repeat(lst_hires, len(freq_hires))
    freq_lst_hires = np.array([freq_hires_array, lst_hires_array]).T

    # Interpolating beam factor to high LST resolution
    bf_hires_array = spi.griddata(
        freq_lst_original, bf_array, freq_lst_hires, method="cubic"
    )
    bf_2D = bf_hires_array.reshape(len(lst_hires), len(freq_hires))

    # Interpolating beam factor to high frequency resolution
    for i in range(len(bf_2D[:, 0])):
        par = np.polyfit(freq_hires, bf_2D[i, :], Npar_freq - 1)
        bf_single = np.polyval(par, fnew)

        if i == 0:
            bf_2D_hires = np.copy(bf_single)
        elif i > 0:
            bf_2D_hires = np.vstack((bf_2D_hires, bf_single))

    return bf_2D_hires, bf_2D  # beam_factor_model  #, freq_hires, bf_lst_average


def antenna_beam_factor_interpolation_v2(band, case, lst_hires, fnew):
    """
    2019-12-17

    """

    if band == "mid_band":
        file_path = edges_folder + "mid_band/calibration/beam_factors/raw/"

        if case == 0:
            bf_old = np.genfromtxt(
                file_path + "mid_band_50-200MHz_90deg_alan0_haslam_gaussian_index_2.4_2"
                ".65_sigma_deg_8.5_reffreq_120MHz_data.txt"
            )
            freq = np.genfromtxt(
                file_path + "mid_band_50-200MHz_90deg_alan0_haslam_gaussian_index_2.4_2"
                ".65_sigma_deg_8.5_reffreq_120MHz_freq.txt"
            )
            lst_old = np.genfromtxt(
                file_path + "mid_band_50-200MHz_90deg_alan0_haslam_gaussian_index_2.4_2"
                ".65_sigma_deg_8.5_reffreq_120MHz_LST.txt"
            )
        elif case == 1:
            bf_old = np.genfromtxt(
                file_path
                + "NORMALIZED_mid_band_50-150MHz_90deg_alan0_haslam_gaussian_index_2"
                ".4_2.65_sigma_deg_8.5_reffreq_90MHz_beam_factor.txt"
            )
            freq = np.genfromtxt(
                file_path
                + "NORMALIZED_mid_band_50-150MHz_90deg_alan0_haslam_gaussian_index_2"
                ".4_2.65_sigma_deg_8.5_reffreq_90MHz_freq.txt"
            )
            lst_old = np.genfromtxt(
                file_path
                + "NORMALIZED_mid_band_50-150MHz_90deg_alan0_haslam_gaussian_index_2"
                ".4_2.65_sigma_deg_8.5_reffreq_90MHz_LST.txt"
            )
    # Wrap beam factor and LST for 24-hr interpolation
    bf = np.vstack((bf_old[-1, :], bf_old, bf_old[0, :]))
    lst0 = np.append(lst_old[-1] - 24, lst_old)
    lst = np.append(lst0, lst_old[0] + 24)

    new_bf = np.zeros((len(lst_hires), len(freq)))
    for i in range(len(freq)):
        fun = spi.interp1d(lst, bf[:, i], kind="cubic")
        new_bf[:, i] = fun(lst_hires)

    new_bf2 = np.zeros((len(lst_hires), len(fnew)))
    for i in range(len(lst_hires)):
        fun = spi.interp1d(freq, new_bf[i, :], kind="cubic", fill_value="extrapolate")
        new_bf2[i, :] = fun(fnew)

    return new_bf2, bf_old, freq, lst_old


def beam_factor_table_computation(band, case, f, N_lst, file_name_hdf5):
    """
    Frequency vector
    ------------------
    ff, i1, i2 = ba.frequency_edges(50, 150)
    f = ff[i1:i2+1]

    High and Low LST resolution
    ------------------------------------------
    high:   N_lst = 6000   # number of LST points within 24 hours
    low:    N_lst = 300    # number of LST points within 24 hours
    """

    lst_hires = np.arange(0, 24, 24 / N_lst)
    bf, bf_orig, f_orig, lst_orig = antenna_beam_factor_interpolation_v2(
        band, case, lst_hires, f
    )

    # Save
    # ----
    file_path = edges_folder + band + "/calibration/beam_factors/table/"
    with h5py.File(file_path + file_name_hdf5, "w") as hf:
        hf.create_dataset("frequency", data=f)
        hf.create_dataset("lst", data=lst_hires)
        hf.create_dataset("beam_factor", data=bf)

    return 0


def beam_factor_table_read(path_file):
    # path_file = home_folder + '/EDGES/calibration/beam_factors/mid_band/file.hdf5'

    # Show keys (array names inside HDF5 file)
    with h5py.File(path_file, "r") as hf:
        hf_freq = hf.get("frequency")
        f = np.array(hf_freq)

        hf_lst = hf.get("lst")
        lst = np.array(hf_lst)

        hf_bf = hf.get("beam_factor")
        bf = np.array(hf_bf)

    return f, lst, bf


def beam_factor_table_evaluate(f_table, lst_table, bf_table, lst_in):
    beam_factor = np.zeros((len(lst_in), len(f_table)))

    for i in range(len(lst_in)):
        d = np.abs(lst_table - lst_in[i])

        index = np.argsort(d)
        IX = index[0]

        beam_factor[i, :] = bf_table[IX, :]

    return beam_factor


def HFSS_integrated_beam_directivity():
    path_to_file = (
        "/run/media/raul/WD_RED_6TB/EDGES_vol2/others/beam_simulations/20190911"
        "/test4_0.04Sm/MROsoil_vacuum_120MHz.csv"
    )
    dB_or_linear = "linear"
    theta, phi, beam1 = hfss_read(
        path_to_file,
        dB_or_linear,
        theta_min=0,
        theta_max=180,
        theta_resolution=1,
        phi_min=0,
        phi_max=359,
        phi_resolution=1,
    )

    path_to_file = (
        "/run/media/raul/WD_RED_6TB/EDGES_vol2/others/beam_simulations/20190911"
        "/test4_0.04Sm/MROsoil_MROsoil_120MHz.csv"
    )
    dB_or_linear = "linear"
    theta, phi, beam2 = hfss_read(
        path_to_file,
        dB_or_linear,
        theta_min=0,
        theta_max=180,
        theta_resolution=1,
        phi_min=0,
        phi_max=359,
        phi_resolution=1,
    )

    sin_theta = np.sin((np.pi / 180) * theta)
    sin_theta_A_T = np.array([sin_theta])
    sin_theta_A = sin_theta_A_T.T
    sin_theta_2D = np.tile(sin_theta_A, len(phi))

    beam = np.copy(beam1)
    # beam[0:91,:] = beam1[0:91,:]
    beam[91::, :] = beam2[91::, :]

    total_radiated_power = np.sum(beam * sin_theta_2D)

    directivity = (4 * np.pi) * beam / total_radiated_power

    directivity_above = np.sum(directivity[0:91, :] * sin_theta_2D[0:91, :])
    directivity_below = np.sum(directivity[91::, :] * sin_theta_2D[91::, :])

    return theta, phi, beam, directivity_above, directivity_below


def beam_solid_angle(gain_map):
    # Theta vector valid for the FEKO beams. In the beam, dimension 1 increases from index 0 to
    # 90 corresponding to elevation, which is 90-theta
    theta = np.arange(90, -1, -1)

    sin_theta = np.sin(theta * (np.pi / 180))
    sin_theta_2D_T = np.tile(sin_theta, (360, 1))
    sin_theta_2D = sin_theta_2D_T.T

    beam_integration = np.sum(gain_map * sin_theta_2D)
    return (1 / (4 * np.pi)) * ((np.pi / 180) ** 2) * beam_integration


def beam_normalization(f_X, input_beam_X, FLOW=50, FHIGH=150):
    """
    2019-Nov-29

    input_beam_X = cal.FEKO_blade_beam('mid_band', 0, AZ_antenna_axis=90)
    f_X          = np.arange(50,201,2)
    f, original_solid_angle, normalized_beam = cal.beam_normalization(f_X, input_beam_X)
    """

    # Select data in the range 50-150 MHz, where we have ground loss data available.
    f = f_X[(f_X >= FLOW) & (f_X <= FHIGH)]
    input_beam = input_beam_X[(f_X >= FLOW) & (f_X <= FHIGH), :, :]

    # Definitions
    g = ground_loss("mid_band", f)
    output_beam = np.copy(input_beam)
    original_solid_angle = np.zeros(len(f))

    # Frequency-by-frequency correction
    for i in range(len(f)):
        m = input_beam[i, :, :]
        osa = beam_solid_angle(m)
        original_solid_angle[i] = osa

        output_beam[i, :, :] = (g[i] / osa) * m

    return f, original_solid_angle, output_beam


def feko_blade_beam(
    band,
    beam_file,
    frequency_interpolation=False,
    frequency=np.array([0]),
    AZ_antenna_axis=0,
):
    """
    AZ_antenna_axis = 0    #  Angle of orientation (in integer degrees) of excited antenna panels
    relative to due North. Best value is around X ???
    """
    if band == "low_band3":
        data_folder = edges_folder + "/low_band3/calibration/beam/alan/"

        if beam_file == 1:
            # FROM ALAN, 50-120 MHz
            print("BEAM MODEL #1 FROM ALAN")
            ff = data_folder + "azelq_low3.txt"
            f_original = np.arange(
                50, 121, 2
            )  # between 50 and 120 MHz in steps of 2 MHz

    elif band == "mid_band":
        data_folder = edges_folder + "/mid_band/calibration/beam/alan/"

        if beam_file == 0:
            # FROM ALAN, 50-200 MHz
            print("BEAM MODEL #0 FROM ALAN")
            ff = data_folder + "azelq_blade9perf7mid_1.9in.txt"
            f_original = np.arange(
                50, 201, 2
            )  # between 50 and 200 MHz in steps of 2 MHz
        elif beam_file == 1:
            # FROM ALAN, 50-200 MHz
            print("BEAM MODEL #1 FROM ALAN")
            ff = data_folder + "azelq_blade9mid0.78.txt"
            f_original = np.arange(
                50, 201, 2
            )  # between 50 and 200 MHz in steps of 2 MHz
        elif beam_file == 100:
            # FROM NIVEDITA, 60-200 MHz
            print("BEAM MODEL FROM NIVEDITA")
            ff = data_folder + "FEKO_midband_realgnd_Simple-blade_niv.txt"
            f_original = np.arange(
                60, 201, 2
            )  # between 60 and 200 MHz in steps of 2 MHz
        elif beam_file == 2:
            # FROM ALAN, 50-200 MHz
            print("BEAM MODEL #2 FROM ALAN")
            ff = data_folder + "azelq_blade9perf7mid.txt"
            f_original = np.arange(
                50, 201, 2
            )  # between 50 and 200 MHz in steps of 2 MHz
    data = np.genfromtxt(ff)

    # Loading data and convert to linear representation
    beam_maps = np.zeros((len(f_original), 91, 360))
    for i in range(len(f_original)):
        beam_maps[i, :, :] = (10 ** (data[(i * 360) : ((i + 1) * 360), 2::] / 10)).T

    # Frequency interpolation
    if frequency_interpolation:

        interp_beam = np.zeros(
            (len(frequency), len(beam_maps[0, :, 0]), len(beam_maps[0, 0, :]))
        )
        for j in range(len(beam_maps[0, :, 0])):
            for i in range(len(beam_maps[0, 0, :])):
                # print('Elevation: ' + str(j) + ', Azimuth: ' + str(i))
                par = np.polyfit(f_original / 200, beam_maps[:, j, i], 13)
                model = np.polyval(par, frequency / 200)
                interp_beam[:, j, i] = model

        beam_maps = np.copy(interp_beam)

    # Shifting beam relative to true AZ (referenced at due North)
    # Due to angle of orientation of excited antenna panels relative to due North
    print("AZ_antenna_axis = " + str(AZ_antenna_axis) + " deg")
    if AZ_antenna_axis < 0:
        AZ_index = -AZ_antenna_axis
        bm1 = beam_maps[:, :, AZ_index::]
        bm2 = beam_maps[:, :, 0:AZ_index]
        beam_maps_shifted = np.append(bm1, bm2, axis=2)

    elif AZ_antenna_axis > 0:
        AZ_index = AZ_antenna_axis
        bm1 = beam_maps[:, :, 0:(-AZ_index)]
        bm2 = beam_maps[:, :, (360 - AZ_index) : :]
        beam_maps_shifted = np.append(bm2, bm1, axis=2)

    elif AZ_antenna_axis == 0:
        beam_maps_shifted = np.copy(beam_maps)

    return beam_maps_shifted
