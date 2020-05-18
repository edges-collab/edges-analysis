import datetime as dt
import warnings
from pathlib import Path
from typing import Optional

import astropy.coordinates as apc
import astropy.time as apt
import h5py
import numpy as np
import scipy.interpolate as spi
from attr.validators import instance_of

from . import coordinates as coords
from .loss import ground_loss
from .plots import plot_beam_factor
from . import io
from .sky_models import (
    LW_150MHz_map,
    guzman_45MHz_map,
    haslam_408MHz_map,
    remazeilles_408MHz_map,
    gsm_map,
)
from ..config import config
from .._data_retrieval import retrieve_beam
from .. import const
from edges_cal import FrequencyRange


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


def feko_read(
    filename: [str, Path], frequency=None, frequency_out=None, az_antenna_axis=0,
):
    """
    Read a FEKO beam file.

    Parameters
    ----------
    filename : path
        The path to the file.
    frequency : array-like, optional
        The frequencies of the data. This usually must be given, as they are not
        included in the data file itself. By default, uses range(50, 121, 2).
    frequency_out : array-like, optional
        If given, input frequencies will be interpolated to these frequencies.
    az_antenna_axis : int, optional

    Returns
    -------
    beam_maps : ndarray
        An ndarray with first axis being frequency, second axis elevation, and third
        axis azimuth.
    """
    filename = Path(filename)

    data = np.genfromtxt(str(filename))
    if frequency is None:
        frequency = np.arange(50, 121, 2)

    freq = FrequencyRange(frequency)

    # Loading data and convert to linear representation
    beam_maps = np.zeros((len(frequency), 91, 360))
    for i in range(len(frequency)):
        beam_maps[i] = (10 ** (data[(i * 360) : ((i + 1) * 360), 2::] / 10)).T

    # Frequency interpolation
    if frequency_out is not None:
        interp_beam = np.zeros(
            (len(frequency_out), beam_maps.shape[1], beam_maps.shape[2])
        )
        for i, bm in enumerate(beam_maps.T):
            for j, b in enumerate(bm):
                par = np.polyfit(freq.freq_recentred, b, 13)
                model = np.polyval(par, freq.normalize(frequency_out))
                interp_beam[:, j, i] = model
        beam_maps = interp_beam

    # Shifting beam relative to true AZ (referenced at due North)
    # Due to angle of orientation of excited antenna panels relative to due North
    return shift_beam_maps(az_antenna_axis, beam_maps)


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


class BeamFactor(io.HDF5Object):
    """A non-interpolated beam factor."""

    _structure = {
        "frequency": instance_of(np.ndarray),
        "lst": instance_of(np.ndarray),
        "antenna_temp_above_horizon": instance_of(np.ndarray),
        "loss_fraction": instance_of(np.ndarray),
        "beam_factor": instance_of(np.ndarray),
    }


def antenna_beam_factor(
    band: str,
    beam_file: [str, Path],
    simulator: str,
    ground_loss_file: [str, Path],
    configuration="",
    save_dir: [None, str, Path] = None,
    save_fname: [None, str, Path] = None,
    f_low: [None, float] = None,
    f_high: [None, float] = None,
    normalize_mid_band_beam: bool = True,
    sky_model: str = "gsm",
    rotation_from_north: float = 90,
    index_model: str = "gaussian",
    sigma_deg: float = 8.5,
    index_center: float = 2.4,
    index_pole: float = 2.65,
    band_deg: float = 10,
    index_inband: float = 2.5,
    index_outband: float = 2.6,
    reference_frequency: float = 100,
    convolution_computation: str = "old",
    plot_format: str = "polar",
    sky_plot_path: [None, str, Path] = None,
):
    """
    Calculate the antenna beam factor.

    Parameters
    ----------
    band
    simulator
    name_save
    f_low
    f_high
    beam_file
    normalize_mid_band_beam
    sky_model
    rotation_from_north
    index_model
    sigma_deg
    index_center
    index_pole
    band_deg
    index_inband
    index_outband
    reference_frequency
    convolution_computation
    sky_plots
    plot_format

    Returns
    -------
    beam_factor : :class`BeamFactor` instance
    """
    if not save_fname:
        raise NotImplementedError(
            "You have to pass 'save_fname' until we figure out an automatic scheme."
        )

    if not save_dir:
        save_dir = Path(config["paths"]["beams"]) / f"{band}/beam_factors/"

    if str(save_fname).startswith(":"):
        save_fname = Path(save_dir).absolute() / str(save_fname)[1:]

    if str(beam_file) == ":":
        # Get the default beam file.
        beam_file = (
            Path(__file__).parent
            / "data"
            / "beams"
            / band
            / ("default.txt" if not configuration else f"{configuration}.txt")
        )

    elif str(beam_file).startswith(":"):
        # Use a beam file in the standard directory.
        beam_file = (
            Path(config["paths"]["beams"]).expanduser()
            / f"{band}/simulations/{simulator}/{str(beam_file)[1:]}"
        ).absolute()
    else:
        beam_file = Path(beam_file).absolute()

    # Antenna beam
    az_beam = np.arange(0, 360)
    el_beam = np.arange(0, 91)
    freq_array = None
    if simulator == "feko":
        rotation_from_north -= 90
        beam_all = feko_read(beam_file, az_antenna_axis=rotation_from_north)

        # TODO: move this to actual beam reading/storing.
        if len(beam_all) == 76:
            freq_array = np.arange(50, 201, 2, dtype="uint32")
        elif len(beam_all) == 71:
            freq_array = np.arange(60, 201, 2, dtype="uint32")
        elif len(beam_all) == 36:
            freq_array = np.arange(50, 121, 2, dtype="uint32")
        elif len(beam_all) == 41:
            freq_array = np.arange(40, 201, 2, dtype="uint32")

    elif simulator == "wipl-d":  # Beams from WIPL-D
        freq_array, az_beam, el_beam, beam_all = wipld_read(
            beam_file, az_antenna_axis=rotation_from_north
        )
    else:
        raise ValueError(f"Unknown value for simulator: '{simulator}'")

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

    freq_mask = (freq_array >= f_low) & (freq_array <= f_high)
    freq_array = freq_array[freq_mask]
    beam_all = beam_all[freq_mask, :, :]

    ground_gain = ground_loss(ground_loss_file, band=band, freq=freq_array)

    # Index of reference frequency
    index_freq_array = np.arange(len(freq_array))
    irf = index_freq_array[freq_array == reference_frequency]

    sky_models = {
        "haslam": (haslam_408MHz_map, 408),
        "remazeilles": (remazeilles_408MHz_map, 408),
        "LW": (LW_150MHz_map, 150),
        "guzman": (guzman_45MHz_map, 45),
        "gsm": (gsm_map, 408),
    }
    if sky_model not in sky_models:
        raise ValueError("sky_model must be one of {}".format(sky_models.keys()))

    map_orig, (lon, lat, galac_coord_object) = sky_models[sky_model][0]()
    nu0 = sky_models[sky_model][1]

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
        sky_map[:, i] = (map_orig - Tcmb) * (freq_array[i] / nu0) ** (-index) + Tcmb

    # Reference UTC observation time. At this time, the LST is 0.1666 (00:10 Hrs LST) at the
    # EDGES location.
    ref_time = [2014, 1, 1, 9, 39, 42]
    ref_time_dt = dt.datetime(*ref_time)

    # Looping over LST
    lst = np.zeros(72)  # TODO: magic number
    convolution_ref = np.zeros((len(lst), len(beam_all[:, 0, 0])))
    antenna_temperature_above_horizon = np.zeros((len(lst), len(beam_all[:, 0, 0])))
    loss_fraction = np.zeros((len(lst), len(beam_all[:, 0, 0])))

    # Advancing time ( 19:57 minutes UTC correspond to 20 minutes LST )
    minutes_offset = 19
    seconds_offset = 57

    # TODO: consider adding progress bar.
    for i in range(len(lst)):
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

        lst[i] = coords.utc2lst(np.array(ref_time), const.edges_lon_deg)

        # Transforming Galactic coordinates of Sky to Local coordinates
        altaz = galac_coord_object.transform_to(
            apc.AltAz(
                location=const.edges_location,
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
        if sky_plot_path:
            plot_beam_factor(
                az_above_horizon,
                const.edges_lat_deg,
                el_above_horizon,
                irf,
                lst[i],
                sky_plot_path,
                plot_format,
                sky_map,
                sky_ref_above_horizon,
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

            if convolution_computation == "new":
                # Convolution and Antenna temperature NEW 'correct' WAY
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
            elif convolution_computation == "old":
                # Convolution and Antenna temperature OLD 'incorrect' WAY
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

            else:
                raise ValueError("convolution_computation must be 'old' or 'new'")

    # Beam factor
    beam_factor = convolution_ref.T / convolution_ref[:, irf].T

    out = {
        "frequency": freq_array,
        "lst": lst,
        "antenna_temp_above_horizon": antenna_temperature_above_horizon,
        "loss_fraction": loss_fraction,
        "beam_factor": beam_factor,
        "meta": {
            "beam_file": beam_file,
            "simulator": simulator,
            "f_low": f_low,
            "f_high": f_high,
            "normalize_mid_band_beam": normalize_mid_band_beam,
            "sky_model": sky_model,
            "rotation_from_north": rotation_from_north,
            "index_model": index_model,
            "sigma_deg": sigma_deg,
            "index_center": index_center,
            "index_pole": index_pole,
            "band_deg": band_deg,
            "index_inband": index_inband,
            "index_outband": index_outband,
            "reference_frequency": reference_frequency,
            "convolution_computation": convolution_computation,
        },
    }

    bf = BeamFactor.from_data(out, filename=save_fname)
    bf.write()

    return bf


class InterpolatedBeamFactor(io.HDF5Object):
    _structure = {
        "beam_factor": instance_of(np.ndarray),
        "frequency": instance_of(np.ndarray),
        "lst": instance_of(np.ndarray),
    }

    @classmethod
    def from_beam_factor(
        cls,
        beam_factor_file,
        band: Optional[str] = None,
        lst_new: Optional[np.ndarray] = None,
        f_new: Optional[np.ndarray] = None,
    ):
        """
        Interpolate beam factor to a new set of LSTs and frequencies.

        The LST interpolation is done using `griddata`, whilst the frequency interpolation
        is done using a polynomial fit.

        Parameters
        ----------
        beam_factor_file : path
            Path to a file containing beam factors produced by :func:`antenna_beam_factor`.
            If just a filename (no path), the `beams/band/beam_factors/` directory will
            be searched (dependent on the configured "beams" directory).
        band : str, optional
            If `beam_factor_file` is relative, the band is required to find the file.
        lst_new : array-like, optional
            The LSTs to interpolate to. By default, keep same LSTs as input.
        f_new : array-like, optional
            The frequencies to interpolate to. By default, keep same frequencies as input.
        """
        beam_factor_file = Path(beam_factor_file)
        if not beam_factor_file.is_absolute():
            beam_factor_file = (
                Path(config["paths"]["beams"])
                / band
                / "beam_factors"
                / beam_factor_file
            )

        if not beam_factor_file.exists():
            raise ValueError(
                "The beam factor file {} does not exist!".format(beam_factor_file)
            )

        with h5py.File(beam_factor_file, "r") as fl:
            beam_factor = fl["beam_factor"][...]
            freq = fl["frequency"][...]
            lst = fl["lst"][...]

        # Wrap beam factor and LST for 24-hr interpolation
        beam_factor = np.vstack((beam_factor[-1], beam_factor, beam_factor[0]))
        lst0 = np.append(lst[-1] - 24, lst)
        lst = np.append(lst0, lst[0] + 24)

        if lst_new:
            beam_factor_lst = np.zeros((len(lst_new), len(freq)))
            for i, bf in enumerate(beam_factor.T):
                beam_factor_lst[:, i] = spi.interp1d(lst, bf, kind="cubic")(lst_new)
            lst = lst_new
        else:
            beam_factor_lst = beam_factor

        if f_new:
            # Interpolating beam factor to high frequency resolution
            beam_factor_freq = np.zeros((len(beam_factor_lst), len(f_new)))
            for i, bf in enumerate(beam_factor_lst):
                beam_factor_freq[i] = spi.interp1d(freq, bf, kind="cubic")(f_new)

            freq = f_new
        else:
            beam_factor_freq = beam_factor_lst

        return cls.from_data(
            {"beam_factor": beam_factor_freq, "frequency": freq, "lst": lst}
        )

    def evaluate(self, lst):
        """Fast nearest-neighbour evaluation of the beam factor for a given LST."""
        beam_factor = np.zeros((len(lst), len(self["frequency"])))

        for i, lst in enumerate(lst):
            index = np.argmin(np.abs(self["lst"] - lst))
            beam_factor[i] = self["beam_factor"][index]

        return beam_factor


def hfss_integrated_beam_directivity(soil_fname, vacuum_fname):
    kwargs = dict(
        linear=True,
        theta_min=0,
        theta_max=180,
        theta_resolution=1,
        phi_min=0,
        phi_max=359,
        phi_resolution=1,
    )

    theta, phi, beam1 = hfss_read(vacuum_fname, **kwargs)
    beam2 = hfss_read(soil_fname, **kwargs)[-1]

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
    input_beam_X = cal.FEKO_blade_beam('mid_band', 0, az_antenna_axis=90)
    f_X          = np.arange(50,201,2)
    f, original_solid_angle, normalized_beam = cal.beam_normalization(f_X, input_beam_X)
    """

    # Select data in the range 50-150 MHz, where we have ground loss data available.
    mask = (freqs >= f_low) & (freqs <= f_high)
    f = freqs[mask]
    input_beam = input_beam[mask]

    # Definitions
    g = ground_loss("mid_band", f)
    output_beam = np.zeros_like(input_beam)
    original_solid_angle = np.zeros_like(f)

    # Frequency-by-frequency correction
    for i, m in enumerate(input_beam):
        osa = beam_solid_angle(m)
        original_solid_angle[i] = osa
        output_beam[i] = (g[i] / osa) * m

    return f, original_solid_angle, output_beam


def FEKO_low_band_blade_beam(**kwargs):
    # TODO: find this function.
    try:
        return feko_read(**kwargs)
    except Exception:
        raise NotImplementedError("yeah this function actually doesn't work.")


def gain_derivative(beam_file):
    b_all = FEKO_low_band_blade_beam(
        beam_file=beam_file,
        frequency_interpolation=False,
        frequency=np.array([0]),
        AZ_antenna_axis=0,
    )
    return np.diff(b_all, axis=0)
