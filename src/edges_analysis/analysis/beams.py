from __future__ import annotations

import datetime as dt
import warnings
from pathlib import Path
from typing import Optional, Tuple
import logging
import hashlib
from methodtools import lru_cache
import astropy.coordinates as apc
import astropy.time as apt
import h5py
import numpy as np
import scipy.interpolate as spi
from tqdm import tqdm
import attr

from . import coordinates as coords
from .loss import ground_loss
from .plots import plot_beam_factor
from . import sky_models

from edges_io.h5 import HDF5Object

from ..config import config
from .. import const
from edges_cal import FrequencyRange

logger = logging.getLogger(__name__)


@attr.s
class Beam:
    beam = attr.ib()
    frequency = attr.ib()
    elevation = attr.ib()
    azimuth = attr.ib()
    simulator = attr.ib(default=None, converter=str)
    instrument = attr.ib(default=None, converter=str)
    raw_file = attr.ib(default=None, converter=str)

    @classmethod
    def from_hfss(
        cls,
        path: [str, Path],
        frequency: float,
        linear: bool = True,
        theta_min: float = 0,
        theta_max: float = 180,
        theta_resolution: float = 1,
        phi_min: float = 0,
        phi_max: float = 359,
        phi_resolution: float = 1,
    ) -> Beam:
        """
        Create a Beam object from a HFSS file.

        Parameters
        ----------
        path
            Path to the file. Use :meth:`resolve_file` to get the absolute path
            from a a relative path (starting with ':').
        linear
        theta_min
        theta_max
        theta_resolution
        phi_min
        phi_max
        phi_resolution

        Returns
        -------
        beam
            The beam object.
        """

        d = np.genfromtxt(path, skip_header=1, delimiter=",")
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

        return Beam(
            frequency=np.array([frequency]),
            elevation=theta,
            azimuth=phi,
            beam=beam_map,
            simulator="hfss",
            raw_file=path,
        )

    @classmethod
    def from_wipld(cls, path: [str, Path], az_antenna_axis: float = 0) -> Beam:
        with open(path) as fn:
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

            rows_per_frequency = (file_length - number_of_frequencies) / number_of_frequencies

            output = np.zeros(
                (
                    int(number_of_frequencies),
                    int(rows_per_frequency),
                    int(number_of_columns),
                )
            )

        frequencies = np.array(frequencies_list)

        with open(path) as fn:
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
        beam_maps_shifted = cls.shift_beam_maps(az_antenna_axis, beam_maps)

        return Beam(
            frequency=frequencies,
            azimuth=az,
            elevation=el,
            beam=beam_maps_shifted,
            simulator="wipl-d",
            raw_file=path,
        )

    @classmethod
    def from_feko(
        cls,
        path: [str, Path],
        az_antenna_axis: float = 0,
        ideal: int = 1
    ) -> Beam:
        """
        Read a FEKO beam file.

        Parameters
        ----------
        filename
            The path to the file.
        az_antenna_axis
            The azimuth of the primary antenna axis, in degrees.

        Returns
        -------
        beam_maps
            A ``(Nfreq, Nel, Naz)`` array giving values of the beam. Note that elevation
            and azimuth are always in 1-degree increments.
        freq
            The frequencies at which the beam is defined.
        """
        filename = Path(path)

        data = np.genfromtxt(str(filename))
        frequency = []
        with open(filename) as fl:
            for line in fl.readlines():
                if line.startswith("#FREQUENCY"):
                    line = line.split(" ")
                    indx = line.index("MHz")
                    frequency.append(float(line[indx - 1]))
        freq = FrequencyRange(np.array(frequency))

        # Loading data and convert to linear representation
        beam_maps = np.zeros((len(frequency), 91, 360))
        for i in range(len(frequency)):
            beam_maps[i] = (10 ** (data[(i * 360) : ((i + 1) * 360), 2::] / 10)).T

        # Shifting beam relative to true AZ (referenced at due North)
        # Due to angle of orientation of excited antenna panels relative to due North
        beam_maps = cls.shift_beam_maps(az_antenna_axis, beam_maps)
        if ideal ==1:
            beam_maps = np.ones((len(frequency), 91, 360))
        return Beam(
            frequency=freq.freq,
            beam=beam_maps,
            azimuth=np.arange(0, 360),
            elevation=np.arange(0, 91),
            simulator="feko",
            raw_file=path,
        )

    def at_freq(self, freq: [np.ndarray]) -> Beam:
        """
        Interpolate the beam to a new set of frequencies.

        Parameters
        ----------
        freq
            Frequencies to interpolate to.

        Returns
        -------
        beam
            The Beam object at the new frequencies.
        """
        if len(self.frequency) < 3:
            raise ValueError("Can't freq-interpolate beam that has fewer than three frequencies.")

        freq_obj = FrequencyRange(self.frequency)

        # Frequency interpolation
        interp_beam = np.zeros((len(freq), len(self.elevation), len(self.azimuth)))

        for i, bm in enumerate(self.beam.T):
            for j, b in enumerate(bm):
                par = np.polyfit(freq_obj.freq_recentred, b, 13)
                model = np.polyval(par, freq_obj.normalize(freq))
                interp_beam[:, j, i] = model
        return Beam(
            frequency=freq, azimuth=self.azimuth, elevation=self.elevation, beam=interp_beam
        )

    @staticmethod
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

    @classmethod
    def resolve_file(
        cls,
        path: [str, Path],
        band: [None, str] = None,
        configuration: str = "default",
        simulator: str = "feko",
    ) -> Path:
        if str(path) == ":":
            if band is None:
                raise ValueError("band must be given if path starts with a colon (:)")

            # Get the default beam file.
            return Path(__file__).parent / "data" / "beams" / band / f"{configuration}.txt"
        elif str(path).startswith(":"):
            if band is None:
                raise ValueError("band must be given if path starts with a colon (:)")

            # Use a beam file in the standard directory.
            return (
                Path(config["paths"]["beams"]).expanduser()
                / f"{band}/simulations/{simulator}/{str(path)[1:]}"
            ).absolute()
        else:
            return Path(path).absolute()

    @classmethod
    def from_file(
        cls,
        band: [None, str],
        simulator: str = "feko",
        beam_file: [str, Path] = ":",
        configuration: [str] = "default",
        rotation_from_north: float = 90,
    ) -> Beam:
        beam_file = cls.resolve_file(beam_file, band, configuration, simulator)

        if simulator == "feko":
            rotation_from_north -= 90
            out = cls.from_feko(beam_file, az_antenna_axis=rotation_from_north)
        elif simulator == "wipl-d":  # Beams from WIPL-D
            out = cls.from_wipld(beam_file, rotation_from_north)
        elif simulator == "hfss":
            out = cls.from_hfss(beam_file)
        else:
            raise ValueError(f"Unknown value for simulator: '{simulator}'")

        out.instrument = band
        return out

    @lru_cache(maxsize=1000)
    def angular_interpolator(self, freq_indx):
        # Interpolate the beam for faster evaluation onto the sky model array.
        az_array, el_array = np.meshgrid(self.azimuth, self.elevation)

        return spi.CloughTocher2DInterpolator(
            np.array([az_array.flatten(), el_array.flatten()]).T,
            self.beam[freq_indx].flatten(),
        )

    def between_freqs(self, low=0, high=np.inf):
        mask = (self.frequency >= low) & (self.frequency <= high)
        return attr.evolve(self, frequency=self.frequency[mask], beam=self.beam[mask])

    def beam_solid_angle(self):
        # Theta vector valid for the FEKO beams. In the beam, dimension 1 increases from index 0 to
        # 90 corresponding to elevation, which is 90-theta
        theta = self.elevation[::-1]

        sin_theta = np.sin(theta * (np.pi / 180))
        sin_theta_2D = np.tile(sin_theta, (360, 1)).T

        beam_integration = np.sum(self.beam * sin_theta_2D)
        return (1 / (4 * np.pi)) * ((np.pi / 180) ** 2) * beam_integration


class BeamFactor(HDF5Object):
    """A non-interpolated beam factor."""

    _structure = {
        "frequency": lambda x: (x.ndim == 1 and x.dtype == float),
        "lst": lambda x: (x.ndim == 1 and x.dtype == float),
        "antenna_temp_above_horizon": lambda x: (x.ndim == 2 and x.dtype == float),
        "loss_fraction": lambda x: (x.ndim == 2 and x.dtype == float),
        "beam_factor": lambda x: (x.ndim == 2 and x.dtype == float),
        "meta": {
            "beam_file": lambda x: isinstance(x, str),
            "simulator": lambda x: isinstance(x, str),
            "f_low": lambda x: isinstance(x, float),
            "f_high": lambda x: isinstance(x, float),
            "normalize_beam": lambda x: isinstance(x, (bool, np.bool_)),
            "sky_model": lambda x: isinstance(x, str),
            "rotation_from_north": lambda x: isinstance(x, float),
            "index_model": lambda x: isinstance(x, str),
            "reference_frequency": lambda x: isinstance(x, float),
            "max_nside": lambda x: isinstance(x, (int, np.int64)),
        },
    }


def _iterate_through_lst_freq(
    lst, ground_loss_file, beam, twenty_min_per_lst, sky_model, index_model, normalize_beam
):
    sky_map = sky_model.at_freq(
        beam.frequency,
        index_model=index_model,
    )

    ground_gain = ground_loss(ground_loss_file, band=beam.instrument, freq=beam.frequency)
    galactic_coords = sky_model.get_sky_coords()

    # Reference UTC observation time. At this time, the LST is 0.1666 (00:10 Hrs LST) at the
    # EDGES location.
    ref_time = dt.datetime(2014, 1, 1, 9, 39, 42)

    # Advancing time (19:57 minutes UTC corresponds to 20 minutes LST)
    twenty_lst_min = dt.timedelta(minutes=19, seconds=57)
    convol = []
    antenna_temperature_above_horizon_list = np.zeros((len(lst), len(beam.frequency), len(sky_map)))
    sky_above_horizon_list = np.zeros((len(lst), len(beam.frequency), len(sky_map)))
    reftime_list = []
    for i in tqdm(range(len(lst)), unit="LST"):
        if i > 0:
            ref_time += twenty_lst_min * twenty_min_per_lst

        lst[i] = coords.utc2lst([ref_time], const.edges_lon_deg)

        # Transform Galactic coordinates of Sky Model to Local coordinates
        altaz = galactic_coords.transform_to(
            apc.AltAz(
                location=const.edges_location,
                obstime=apt.Time(ref_time, format="datetime"),
            )
        )
        az = np.asarray(altaz.az)
        el = np.asarray(altaz.alt)

        # Number of pixels over 4pi that are not 'nan'
        n_pix_tot = len(el)

        # Selecting coordinates above the horizon
        horizon_mask = el >= 0
        az_above_horizon = az[horizon_mask]
        el_above_horizon = el[horizon_mask]
        
        reftime_list.append(ref_time)
        # Selecting sky data above the horizon
        sky_above_horizon = sky_map[horizon_mask, :]

        # Arranging AZ and EL arrays corresponding to beam model
        az_el_above_horizon = np.array([az_above_horizon, el_above_horizon]).T

        # Loop over frequency
        for j in tqdm(range(len(beam.frequency)), unit="Frequency"):
            beam_above_horizon = beam.angular_interpolator(j)(az_el_above_horizon)

            n_pix_ok = np.sum(~np.isnan(beam_above_horizon))

            # The nans are only above the horizon
            n_pix_tot_no_nan = n_pix_tot - (len(el_above_horizon) - n_pix_ok)

            if normalize_beam:
                solid_angle = np.nansum(beam_above_horizon) / n_pix_tot_no_nan
                beam_above_horizon *= ground_gain[j] / solid_angle
                #convol.append(beam_above_horizon* sky_above_horizon[:, j])
                antenna_temperature_above_horizon_list[i,j,horizon_mask]=(beam_above_horizon * sky_above_horizon[:,j])
                sky_above_horizon_list[i,j,horizon_mask]=sky_above_horizon[:,j]
            # Convolution between (beam at all frequencies) and (sky at all frequencies)
            yield (
                i,
                j,
                np.nansum(beam_above_horizon * sky_above_horizon[:, j]) / n_pix_tot_no_nan,
                antenna_temperature_above_horizon_list,
                sky_above_horizon_list,
                reftime_list,
                n_pix_tot_no_nan,
            )


def simulate_spectra(
    beam: Beam,
    ground_loss_file: [str, Path] = ":",
    f_low: [None, float] = 0,
    f_high: [None, float] = np.inf,
    normalize_beam: bool = True,
    sky_model: sky_models.SkyModel = sky_models.Haslam408(),
    index_model: sky_models.IndexModel = sky_models.ConstantIndex(),
    twenty_min_per_lst: int = 1,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """

    Parameters
    ----------
    band
        The band of the antenna (low, mid, high).
    beam
        A :class:`Beam` object.
    ground_loss_file
        A file pointing to a ground-loss model. By default, gets the default ground loss
        model for the given ``band``.
    f_low
        Minimum frequency to keep in the simulation (frequencies otherwise defined by the
        beam).
    f_high
        Maximum frequency to keep in the simulation (frequencies otherwise defined by the
        beam).
    normalize_beam
        Whether to normalize the beam to be maximum unity.
    sky_model
        A sky model to use.
    index_model
        An :class:`IndexModel` to use to generate different frequencies of the sky model.
    twenty_min_per_lst
        How many periods of twenty minutes fit into each LST bin.

    Returns
    -------
    antenna_temperature_above_horizon
        The antenna temperature for pixels above the horizon, shape (Nlst, Nfreq)
    freq
        The frequencies at which the simulation is defined.
    lst
        The LSTs at which the sim is defined.
    """
    beam = beam.between_freqs(f_low, f_high)
    lst = np.zeros(72 // twenty_min_per_lst)  # TODO: magic number

    antenna_temperature_above_horizon = np.zeros((len(lst), len(beam.frequency)))
    for i, j, temperature, convolution_product, sky_list,ref_time, _ in _iterate_through_lst_freq(
        lst, ground_loss_file, beam, twenty_min_per_lst, sky_model, index_model, normalize_beam
    ):
        antenna_temperature_above_horizon[i, j] = temperature

    return antenna_temperature_above_horizon, beam.frequency, lst, convolution_product, sky_list,ref_time


def antenna_beam_factor(
    beam: Beam,
    ground_loss_file: [str, Path] = ":",
    f_low: float = 0,
    f_high: float = np.inf,
    normalize_beam: bool = True,
    sky_model: sky_models.SkyModel = sky_models.Haslam408(),
    index_model: sky_models.IndexModel = sky_models.GaussianIndex(),
    twenty_min_per_lst=1,
    save_dir: [None, str, Path] = None,
    save_fname: [None, str, Path, bool] = None,
    reference_frequency: [None, float] = None,
):
    """
    Calculate the antenna beam factor.

    Parameters
    ----------
    beam
        A :class:`Beam` object.
    ground_loss_file
        A file pointing to a ground-loss model. By default, gets the default ground loss
        model for the given ``band``.
    f_low
        Minimum frequency to keep in the simulation (frequencies otherwise defined by the
        beam).
    f_high
        Maximum frequency to keep in the simulation (frequencies otherwise defined by the
        beam).
    normalize_beam
        Whether to normalize the beam to be maximum unity.
    sky_model
        A sky model to use.
    index_model
        An :class:`IndexModel` to use to generate different frequencies of the sky model.
    twenty_min_per_lst
        How many periods of twenty minutes fit into each LST bin.
    save_dir
        The directory in which to save the output beam factor.
    save_fname
        The filename to save the output beam factor.
    reference_frequency
        The frequency to take as the "reference", i.e. where the chromaticity will
        be by construction unity.

    Returns
    -------
    beam_factor : :class`BeamFactor` instance
    """
    if not save_dir:
        save_dir = Path(config["paths"]["beams"]) / f"{beam.instrument}/beam_factors/"

    if str(save_fname).startswith(":"):
        save_fname = Path(save_dir).absolute() / str(save_fname)[1:]

    beam = beam.between_freqs(f_low, f_high)
    lst = np.zeros(72 // twenty_min_per_lst)  # TODO: magic number

    # Get index of reference frequency
    if reference_frequency is None:
        reference_frequency = (f_high + f_low) / 2

    indx_ref_freq = np.argwhere(beam.frequency >= reference_frequency)[0][0]
    reference_frequency = beam.frequency[indx_ref_freq]

    antenna_temperature_above_horizon = np.zeros((len(lst), len(beam.frequency)))
    convolution_ref = np.zeros((len(lst), len(beam.frequency)))
    loss_fraction = np.zeros((len(lst), len(beam.frequency)))

    for i, j, temperature, sky, bm, npix_no_nan in _iterate_through_lst_freq(
        lst,
        beam=beam,
        twenty_min_per_lst=twenty_min_per_lst,
        sky_model=sky_model,
        index_model=index_model,
        normalize_beam=normalize_beam,
        ground_loss_file=ground_loss_file,
    ):
        antenna_temperature_above_horizon[i, j] = temperature
        # Convolution between (beam at all frequencies) and (sky at reference frequency)
        convolution_ref[i, j] = np.nansum(bm * sky[:, indx_ref_freq])

        # Loss fraction
        loss_fraction[i, j] = 1 - np.nansum(bm) / npix_no_nan

    # Beam factor
    beam_factor = (convolution_ref.T / convolution_ref[:, indx_ref_freq]).T

    out = {
        "frequency": beam.frequency.astype(np.float),
        "lst": lst,
        "antenna_temp_above_horizon": antenna_temperature_above_horizon,
        "loss_fraction": loss_fraction,
        "beam_factor": beam_factor,
        "meta": {
            "beam_file": str(beam.raw_file),
            "simulator": beam.simulator,
            "f_low": float(f_low),
            "f_high": float(f_high),
            "normalize_beam": bool(normalize_beam),
            "sky_model": str(sky_model),
            "index_model": str(index_model),
            "reference_frequency": float(reference_frequency),
            "rotation_from_north": float(90),
            "max_nside": int(sky_model.max_res or 0) ,
                    },
    }

    if save_fname is None:
        hsh = hashlib.md5(repr(out["meta"]).encode()).hexdigest()
        save_fname = (
            save_dir
            / f"{beam.simulator}_{sky_model.__class__.__name__}_ref{reference_frequency:.2f}_{hsh}.h5"
        )

    logger.info(f"Writing out beam file to {save_fname}")
    bf = BeamFactor.from_data(out, filename=save_fname)
    bf.write(clobber=True)

    return bf


class InterpolatedBeamFactor(HDF5Object):
    _structure = {
        "beam_factor": None,
        "frequency": None,
        "lst": None,
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
        beam_factor_file = Path(beam_factor_file).expanduser()
        if str(beam_factor_file).startswith(":"):
            beam_factor_file = (
                Path(config["paths"]["beams"]) / band / "beam_factors" / str(beam_factor_file)[1:]
            )

        if not beam_factor_file.exists():
            raise ValueError(f"The beam factor file {beam_factor_file} does not exist!")

        with h5py.File(beam_factor_file, "r") as fl:
            beam_factor = fl["beam_factor"][...]
            freq = fl["frequency"][...]
            lst = fl["lst"][...]

        # Wrap beam factor and LST for 24-hr interpolation
        beam_factor = np.vstack((beam_factor[-1], beam_factor, beam_factor[0]))
        lst0 = np.append(lst[-1] - 24, lst)
        lst = np.append(lst0, lst[0] + 24)

        if lst_new is not None:
            beam_factor_lst = np.zeros((len(lst_new), len(freq)))
            for i, bf in enumerate(beam_factor.T):
                beam_factor_lst[:, i] = spi.interp1d(lst, bf, kind="cubic")(lst_new)
            lst = lst_new
        else:
            beam_factor_lst = beam_factor

        if f_new is not None:
            # Interpolating beam factor to high frequency resolution
            beam_factor_freq = np.zeros((len(beam_factor_lst), len(f_new)))
            for i, bf in enumerate(beam_factor_lst):
                beam_factor_freq[i] = spi.interp1d(freq, bf, kind="cubic")(f_new)

            freq = f_new
        else:
            beam_factor_freq = beam_factor_lst

        return cls.from_data({"beam_factor": beam_factor_freq, "frequency": freq, "lst": lst})

    def evaluate(self, lst):
        """Fast nearest-neighbour evaluation of the beam factor for a given LST."""
        beam_factor = np.zeros((len(lst), len(self["frequency"])))

        for i, lst in enumerate(lst):
            index = np.argmin(np.abs(self["lst"] - lst))
            beam_factor[i] = self["beam_factor"][index]

        return beam_factor
