import datetime as dt
import warnings

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
    linear=True,
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
        mask = d[:, 1] == theta[i]

        this_phi = d[mask, 0]
        this_power = d[mask, 2]

        this_phi[this_phi < 0] += 360
        this_phi, unique_inds = np.unique(this_phi, axis=0, return_index=True)
        this_power = this_power[unique_inds]

        this_power = this_power[np.argsort(this_phi)]

        if not linear:
            this_power = 10 ** (this_power / 10)

        this_power[np.isnan(this_power)] = 0
        beam_map[i, :] = this_power

    return theta, phi, beam_map


def wipld_read(filename, az_antenna_axis=0):
    with open(filename) as fn:

        file_length = 0
        number_of_frequencies = 0

        flag_columns = False
        frequencies_list = []
        for line in fn:
            file_length += 1
            if line[2] == ">":
                number_of_frequencies += 1
                frequencies_list.append(float(line[19:32]))
            elif not flag_columns:
                line_splitted = line.split()
                number_of_columns = len(line_splitted)
                flag_columns = True

        rows_per_frequency = (
            file_length - number_of_frequencies
        ) / number_of_frequencies

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
            else:
                j += 1
                line_splitted = line.split()
                line_array = np.array(line_splitted, dtype=float)
                output[i, j, :] = line_array

    # Re-arrange data
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
            phi_j = phi[theta == theta_a[j]]
            gain_j = gain[theta == theta_a[j]]

            ip = np.argsort(phi_j)
            gp = gain_j[ip]

            beam[i, j, :] = gp

    # Flip beam from theta to elevation
    beam_maps = beam[:, ::-1, :]

    # Change coordinates from theta/phi, to AZ/EL
    el = np.arange(0, 91)
    az = np.arange(0, 360)

    # Shifting beam relative to true AZ (referenced at due North)
    beam_maps_shifted = shift_beam_maps(az_antenna_axis, beam_maps)

    return frequencies, az, el, beam_maps_shifted


def shift_beam_maps(az_antenna_axis, beam_maps):
    if az_antenna_axis < 0:
        index = -az_antenna_axis
        bm1 = beam_maps[:, :, index::]
        bm2 = beam_maps[:, :, 0:index]
        beam_maps_shifted = np.append(bm1, bm2, axis=2)
    elif az_antenna_axis > 0:
        index = az_antenna_axis
        bm1 = beam_maps[:, :, 0:(-index)]
        bm2 = beam_maps[:, :, (360 - index) : :]
        beam_maps_shifted = np.append(bm2, bm1, axis=2)
    else:
        beam_maps_shifted = beam_maps

    return beam_maps_shifted


def antenna_beam_factor(
    band,
    name_save,
    f_low=None,
    f_high=None,
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
    plot_format="polar",
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
    az_beam = np.arange(0, 360)
    el_beam = np.arange(0, 91)
    freq_array = None
    if (
        band == "mid_band" and beam_file <= 100
    ) or band == "low_band3":  # FEKO blade beam
        # Fixing rotation angle due to different rotation (by 90deg) in Nivedita's map
        if band == "mid_band" and beam_file == 100:
            rotation_from_north -= 90

        beam_all = feko_blade_beam(band, beam_file, az_antenna_axis=rotation_from_north)
    elif band == "mid_band":  # Beams from WIPL-D
        prefix = edges_folder + "/others/beam_simulations/wipl-d/"
        filenames = {
            101: "20191030/blade_dipole_infinite_soil_real_metal_GP_30mx30m.ra1",
            102: "20191124/mid_band_perf_30x30_5mm_wire_.ra1",
            103: "20191124/mid_band_perf_30x30_5mm_wire_no_soil_conductivity.ra1",
        }

        if beam_file not in filenames:
            raise ValueError(
                "If band=='mid_band' and beam_file>100, "
                "beam_file must be in {}".format(filenames.keys())
            )

        freq_array, az_beam, el_beam, beam_all = wipld_read(
            prefix + filenames[beam_file], az_antenna_axis=rotation_from_north
        )
    else:
        raise ValueError("Incompatible combination of band and beam_file")

    # Frequency array
    if band == "mid_band":
        if beam_file in (0, 1, 2):  # Best case, Feb 20, 2019
            freq_array = np.arange(50, 201, 2, dtype="uint32")
        elif beam_file == 100:
            freq_array = np.arange(60, 201, 2, dtype="uint32")
    elif band == "low_band3" and beam_file == 1:
        freq_array = np.arange(50, 121, 2, dtype="uint32")
    elif band == "low_band" and beam_file == 2:
        freq_array = np.arange(40, 121, 2, dtype="uint32")

    if freq_array is None:
        raise ValueError("Incompatible combination of band and beam_file")

    # Selecting frequency range
    if band == "mid_band" and normalize_mid_band_beam:  # Beam normalization
        if f_low is not None or f_high is not None:
            warnings.warn(
                "Your selected frequency range is being over-ridden due to"
                " band='mid_band' and normalize_mid_band_beam=True."
            )
        f_low = 50
        f_high = 150
    else:
        f_low = f_low or 50
        f_high = f_high or 200

    freq_mask = freq_array >= f_low & freq_array <= f_high
    freq_array = freq_array[freq_mask]
    beam_all = beam_all[freq_mask, :, :]

    ground_gain = ground_loss("mid_band", freq_array)

    # Index of reference frequency
    index_freq_array = np.arange(len(freq_array))
    irf = index_freq_array[freq_array == reference_frequency]

    sky_models = {
        "haslam": (haslam_408MHz_map, 408),
        "remazeilles": (remazeilles_408MHz_map, 408),
        "LW": (LW_150MHz_map, 150),
        "guzman": (guzman_45MHz_map, 45),
    }
    if sky_model not in sky_models:
        raise ValueError("sky_model must be one of {}".format(sky_models.keys()))

    map_orig, lon, lat, galac_coord_object = sky_models[sky_model][0]()
    v0 = sky_models[sky_model][1]

    # Scale sky map (the map contains the CMB, which has to be removed and then added back)
    if index_model == "gaussian":
        index = index_pole - (index_pole - index_center) * np.exp(
            -(1 / 2) * (np.abs(lat) / sigma_deg) ** 2
        )
    elif index_model == "step":
        index = np.zeros(len(lat))
        index[np.abs(lat) <= band_deg] = index_inband
        index[np.abs(lat) > band_deg] = index_outband
    else:
        raise ValueError("index_model must be either 'gaussian' or 'step'")

    Tcmb = 2.725
    sky_map = np.zeros((len(map_orig), len(freq_array)))
    for i in range(len(freq_array)):
        sky_map[:, i] = (map_orig - Tcmb) * (freq_array[i] / v0) ** (-index) + Tcmb

    # EDGES location
    edges_lat_deg = -26.714778
    edges_lon_deg = 116.605528
    edges_location = apc.EarthLocation(
        lat=edges_lat_deg * apu.deg, lon=edges_lon_deg * apu.deg
    )

    # Reference UTC observation time. At this time, the LST is 0.1666 (00:10 Hrs LST) at the
    # EDGES location.
    ref_time = [2014, 1, 1, 9, 39, 42]
    ref_time_dt = dt.datetime(*ref_time)

    # Looping over LST
    lst = np.zeros(72)  # TODO: magic number
    convolution_ref = np.zeros((len(lst), len(beam_all[:, 0, 0])))
    antenna_temperature_above_horizon = np.zeros((len(lst), len(beam_all[:, 0, 0])))
    loss_fraction = np.zeros((len(lst), len(beam_all[:, 0, 0])))

    # TODO: consider adding progress bar.
    for i in range(len(lst)):
        # Advancing time ( 19:57 minutes UTC correspond to 20 minutes LST )
        minutes_offset = 19
        seconds_offset = 57
        if i > 0:
            ref_time_dt = ref_time_dt + dt.timedelta(
                minutes=minutes_offset, seconds=seconds_offset
            )
            ref_time = [
                ref_time_dt.year,
                ref_time_dt.month,
                ref_time_dt.day,
                ref_time_dt.hour,
                ref_time_dt.minute,
                ref_time_dt.second,
            ]

        lst[i] = coords.utc2lst(np.array(ref_time), edges_lon_deg)

        # Transforming Galactic coordinates of Sky to Local coordinates
        altaz = galac_coord_object.transform_to(
            apc.AltAz(
                location=edges_location,
                obstime=apt.Time(ref_time_dt, format="datetime"),
            )
        )
        az = np.asarray(altaz.az)
        el = np.asarray(altaz.alt)

        # Selecting coordinates above the horizon
        horizon_mask = el >= 0
        az_above_horizon = az[horizon_mask]
        el_above_horizon = el[horizon_mask]

        # Selecting sky data above the horizon
        sky_above_horizon = sky_map[horizon_mask, :]
        sky_ref_above_horizon = sky_above_horizon[:, irf].flatten()

        # Plotting sky in local coordinates
        if sky_plots:
            LAT_DEG = np.copy(edges_lat_deg)

            AZ_plot = np.copy(az_above_horizon)
            AZ_plot[AZ_plot > 180] = AZ_plot[AZ_plot > 180] - 360

            EL_plot = np.copy(el_above_horizon)
            SKY_plot = np.copy(sky_ref_above_horizon)

            max_log10sky = np.max(np.log10(sky_map[:, irf]))
            min_log10sky = np.min(np.log10(sky_map[:, irf]))

            marker_size = 10

            LST_gc = 17 + (45 / 60) + (40.04 / (60 * 60))  # LST of Galactic Center
            GHA = lst[i] - LST_gc
            if GHA < 0:
                GHA = GHA + 24

            if plot_format == "rect":
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
                    + str(np.round(lst[i], 3)).ljust(5, "0")
                    + " hr        GHA="
                    + str(np.round(GHA, 3)).ljust(5, "0")
                    + " hr"
                )
            elif plot_format == "polar":
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
                )
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
                    + str(np.round(lst[i], 3)).ljust(5, "0")
                    + " [hr]        GHA="
                    + str(np.round(GHA, 3)).ljust(5, "0")
                    + " [hr]",
                    fontsize=14,
                    fontweight="bold",
                )
                cbar_ax = fig.add_axes([0.9, 0.3, 0.02, 0.4])
                hcbar = fig.colorbar(c, cax=cbar_ax)
                hcbar.set_label("log10( Tsky @ 50MHz [K] )", rotation=90)
            else:
                raise ValueError("plot_format must be either 'polar' or 'rect'.")

            plt.savefig(
                path_plots + "LST_" + str(np.round(lst[i], 3)).ljust(5, "0") + "hr.png",
                bbox_inches="tight",
            )

        # Arranging AZ and EL arrays corresponding to beam model
        az_array = np.tile(az_beam, 91)
        el_array = np.repeat(el_beam, 360)
        az_el_original = np.array([az_array, el_array]).T
        az_el_above_horizon = np.array([az_above_horizon, el_above_horizon]).T

        # Loop over frequency
        # TODO: consider adding progress bar.
        for j in range(len(freq_array)):
            beam_array = beam_all[j, :, :].reshape(1, -1)[0]
            beam_above_horizon = spi.griddata(
                az_el_original, beam_array, az_el_above_horizon, method="cubic"
            )  # interpolated beam

            no_nan_array = np.ones(len(az_above_horizon)) - np.isnan(beam_above_horizon)
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
                Npixels_total = len(el)
                Npixels_above_horizon_nan = len(el_above_horizon) - len(
                    el_above_horizon[index_no_nan]
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
            else:
                raise ValueError("convolution_computation must be 'old' or 'new'")

    # Beam factor
    beam_factor = convolution_ref.T / convolution_ref[:, irf].T

    # Saving
    np.savetxt(path_save + name_save + "_freq.txt", freq_array)
    np.savetxt(path_save + name_save + "_LST.txt", lst)
    np.savetxt(path_save + name_save + "_tant.txt", antenna_temperature_above_horizon)
    np.savetxt(path_save + name_save + "_loss.txt", loss_fraction)
    np.savetxt(path_save + name_save + "_beam_factor.txt", beam_factor.T)


def antenna_beam_factor_interpolation(band, case, lst_hires, fnew, Npar_freq=15):
    """
    For Mid-Band, over 50-200MHz, we have to use Npar_freq=15

    Here, "case" is not the same as in the FEKO... function
    """
    # TODO: is this functio deprecated/unused since there's a v2 below?

    if band not in ("low_band3", "mid_band"):
        raise ValueError("band must be 'low_band3' or 'mid_band'")

    direc = edges_folder + "{}/calibration/beam_factors/raw/".format(band)

    if band == "low_band3":
        pth = "low_band3_50-120MHz_85deg_alan_haslam_2.5_2.62_reffreq_76MHz_{}.txt"

    elif band == "mid_band":
        paths = {
            0: (
                "mid_band_50-200MHz_90deg_alan0_haslam_gaussian_index_2.4_2"
                ".65_sigma_deg_8.5_reffreq_90MHz_{}.txt"
            ),
            1: (
                "mid_band_50-200MHz_90deg_alan0_haslam_flat_index_2"
                ".56_reffreq_90MHz_{}.txt"
            ),
            10: (
                "NORMALIZED_mid_band_50-150MHz_90deg_alan0_haslam_gaussian_index_2"
                ".4_2.65_sigma_deg_8.5_reffreq_90MHz_{}.txt"
            ),
            2: (
                "mid_band_50-200MHz_90deg_alan0_LW_gaussian_index_2.4_2"
                ".65_sigma_deg_8.5_reffreq_90MHz_{}.txt"
            ),
            3: (
                "mid_band_50-200MHz_90deg_alan0_guzman_gaussian_index_2.4_2"
                ".65_sigma_deg_8.5_reffreq_90MHz_{}.txt"
            ),
            4: (
                "mid_band_50-200MHz_90deg_alan1_haslam_gaussian_index_2.4_2"
                ".65_sigma_deg_8.5_reffreq_90MHz_{}.txt"
            ),
            5: (
                "mid_band_50-200MHz_90deg_alan0_haslam_gaussian_index_2.4_2"
                ".65_sigma_deg_8.5_reffreq_120MHz_{}.txt"
            ),
        }

        if case not in paths:
            raise ValueError(
                "For mid_band, the case must be one of {}".format(paths.keys())
            )

        pth = paths[case]

    bf_old = np.genfromtxt(direc + pth.format("data"))
    freq = np.genfromtxt(direc + pth.format("freq"))
    lst_old = np.genfromtxt(direc + pth.format("LST"))

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
    bf_2D_hires = np.zeros((len(bf_2D), len(fnew)))
    for i, bf in enumerate(bf_2D):
        par = np.polyfit(freq_hires, bf, Npar_freq - 1)
        bf_2D_hires[i] = np.polyval(par, fnew)

    return bf_2D_hires, bf_2D


def antenna_beam_factor_interpolation_v2(band, case, lst_hires, fnew):
    if band not in ("mid_band",):
        raise NotImplementedError("only 'mid_band' implemented for this function.")

    direc = edges_folder + "{}/calibration/beam_factors/raw/".format(band)

    if band == "mid_band":
        paths = {
            0: (
                "mid_band_50-200MHz_90deg_alan0_haslam_gaussian_index_2.4_2"
                ".65_sigma_deg_8.5_reffreq_120MHz_{}.txt"
            ),
            1: (
                "NORMALIZED_mid_band_50-150MHz_90deg_alan0_haslam_gaussian_index_2"
                ".4_2.65_sigma_deg_8.5_reffreq_90MHz_{}.txt"
            ),
        }

        if case not in paths:
            raise ValueError("case must be in {}".format(paths.keys()))

        pth = paths[case]

    bf_old = np.genfromtxt(direc + pth.format("data"))
    freq = np.genfromtxt(direc + pth.format("freq"))
    lst_old = np.genfromtxt(direc + pth.format("LST"))

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

    file_path = edges_folder + band + "/calibration/beam_factors/table/"
    with h5py.File(file_path + file_name_hdf5, "w") as hf:
        hf.create_dataset("frequency", data=f)
        hf.create_dataset("lst", data=lst_hires)
        hf.create_dataset("beam_factor", data=bf)


def beam_factor_table_read(path_file):
    # Show keys (array names inside HDF5 file)
    with h5py.File(path_file, "r") as hf:
        f = np.array(hf.get("frequency"))
        lst = np.array(hf.get("lst"))
        bf = np.array(hf.get("beam_factor"))

    return f, lst, bf


def beam_factor_table_evaluate(f_table, lst_table, bf_table, lst_in):
    beam_factor = np.zeros((len(lst_in), len(f_table)))

    for i, lst in enumerate(lst_in):
        d = np.abs(lst_table - lst)
        index = np.argsort(d)[0]
        beam_factor[i, :] = bf_table[index, :]

    return beam_factor


def hfss_integrated_beam_directivity():
    kwargs = dict(
        linear=True,
        theta_min=0,
        theta_max=180,
        theta_resolution=1,
        phi_min=0,
        phi_max=359,
        phi_resolution=1,
    )

    pth = (
        "/run/media/raul/WD_RED_6TB/EDGES_vol2/others/beam_simulations/20190911"
        "/test4_0.04Sm/MROsoil_{}_120MHz.csv"
    )

    theta, phi, beam1 = hfss_read(pth.format("vacuum"), **kwargs)
    beam2 = hfss_read(pth.format("MROsoil"), **kwargs)[-1]

    sin_theta = np.array([np.sin((np.pi / 180) * theta)]).T
    sin_theta_2D = np.tile(sin_theta, len(phi))

    # TODO: explain why we do this...
    beam = beam1
    beam[91:, :] = beam2[91:, :]

    total_radiated_power = np.sum(beam * sin_theta_2D)

    directivity = (4 * np.pi) * beam / total_radiated_power
    directivity_above = np.sum(directivity[:91, :] * sin_theta_2D[:91, :])
    directivity_below = np.sum(directivity[91:, :] * sin_theta_2D[91:, :])

    return theta, phi, beam, directivity_above, directivity_below


def beam_solid_angle(gain_map):
    # Theta vector valid for the FEKO beams. In the beam, dimension 1 increases from index 0 to
    # 90 corresponding to elevation, which is 90-theta
    theta = np.arange(90, -1, -1)

    sin_theta = np.sin(theta * (np.pi / 180))
    sin_theta_2D = np.tile(sin_theta, (360, 1)).T

    beam_integration = np.sum(gain_map * sin_theta_2D)
    return (1 / (4 * np.pi)) * ((np.pi / 180) ** 2) * beam_integration


def beam_normalization(freqs, input_beam, f_low=50, f_high=150):
    """
    2019-Nov-29

    input_beam_X = cal.FEKO_blade_beam('mid_band', 0, az_antenna_axis=90)
    f_X          = np.arange(50,201,2)
    f, original_solid_angle, normalized_beam = cal.beam_normalization(f_X, input_beam_X)
    """

    # Select data in the range 50-150 MHz, where we have ground loss data available.
    mask = (freqs >= f_low) & (freqs <= f_high)
    f = freqs[mask]
    input_beam = input_beam[mask, :, :]

    # Definitions
    g = ground_loss("mid_band", f)
    output_beam = np.zeros_like(input_beam)
    original_solid_angle = np.zeros_like(f)

    # Frequency-by-frequency correction
    for i, m in enumerate(input_beam):
        osa = beam_solid_angle(m)
        original_solid_angle[i] = osa
        output_beam[i, :, :] = (g[i] / osa) * m

    return f, original_solid_angle, output_beam


def feko_blade_beam(
    band,
    beam_file,
    frequency_interpolation=False,
    frequency=np.array([0]),
    az_antenna_axis=0,
):
    """
    az_antenna_axis = 0    #  Angle of orientation (in integer degrees) of excited antenna panels
    relative to due North. Best value is around X ???
    """
    data_folder = edges_folder + "/{}/calibration/beam/alan/".format(band)

    if band == "low_band3":
        if beam_file == 1:
            ff = data_folder + "azelq_low3.txt"
            f_original = np.arange(50, 121, 2)
        else:
            raise ValueError("for low_band3 beam_file must be 1.")
    elif band == "mid_band":
        if beam_file == 0:
            ff = data_folder + "azelq_blade9perf7mid_1.9in.txt"
            f_original = np.arange(50, 201, 2)
        elif beam_file == 1:
            ff = data_folder + "azelq_blade9mid0.78.txt"
            f_original = np.arange(50, 201, 2)
        elif beam_file == 100:
            ff = data_folder + "FEKO_midband_realgnd_Simple-blade_niv.txt"
            f_original = np.arange(60, 201, 2)
        elif beam_file == 2:
            ff = data_folder + "azelq_blade9perf7mid.txt"
            f_original = np.arange(50, 201, 2)
        else:
            raise ValueError("for low_band3 beam_file must be 0,1,100 or 2.")
    else:
        raise ValueError("band must be either 'low_band3' or 'mid_band'.")

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
        beam_maps = interp_beam

    # Shifting beam relative to true AZ (referenced at due North)
    # Due to angle of orientation of excited antenna panels relative to due North
    return shift_beam_maps(az_antenna_axis, beam_maps)
