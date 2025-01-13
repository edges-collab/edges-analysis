"""Beam models and chromaticity corrections."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Literal

import astropy.coordinates as apc
import astropy.time as apt
import attrs
import numpy as np
import scipy.interpolate as spi
from astropy import units as u
from edges_cal import FrequencyRange
from edges_cal import modelling as mdl
from edges_cal.tools import vld_unit
from edges_io import types as tp
from hickleable import hickleable
from pygsdata import coordinates as gscrd
from read_acq import _coordinates as crda
from tqdm import tqdm

from . import const, sky_models
from .calibration.loss import ground_loss
from .config import config
from .data import BEAM_PATH

logger = logging.getLogger(__name__)

# Reference UTC observation time. At this time, the LST is 0.1666 (00:10 Hrs LST) at the
# EDGES location. NOTE: this is used by default, but can be changed by the user anywhere
# it is used.
REFERENCE_TIME = apt.Time("2014-01-01T09:39:42", location=const.edges_location)


@attrs.define(kw_only=True)
class Beam:
    """A beam model object.

    Attributes
    ----------
    beam
        The beam model, as a function of frequency, elevation, and azimuth.
    frequency
        The frequencies at which the beam is defined.
    elevation
        The elevation angles at which the beam is defined.
    azimuth
        The azimuth angles at which the beam is defined.
    simulator
        The simulator used to generate the beam.
    instrument
        The instrument for which the beam is defined.
    raw_file
        The path to the raw file from which the beam was read.
    """

    beam: np.ndarray = attrs.field()
    frequency: tp.FreqType = attrs.field(validator=vld_unit("frequency"))
    elevation: np.ndarray = attrs.field(converter=np.asarray)
    azimuth: np.ndarray = attrs.field(converter=np.asarray)
    simulator: str | None = attrs.field(default=None, converter=str)
    instrument: str | None = attrs.field(default=None, converter=str)
    raw_file: str | None = attrs.field(default=None, converter=str)

    @classmethod
    def from_hfss(
        cls,
        path: tp.PathLike,
        frequency: tp.FreqType,
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
            Whether the beam values are in linear units (or decibels)
        theta_min, theta_max
            Min/Max of the zenith angle (degrees)
        theta_resolution
            Resolution of the zenith angle.
        phi_min, phi_max
            Min/Max of the azimuth angles (degrees)
        phi_resolution
            The resolution of the azimuth angle.

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
    def from_wipld(cls, path: tp.PathLike, az_antenna_axis: float = 0) -> Beam:
        """Read a WIPL-D beam.

        Parameters
        ----------
        path
            The path to the file.
        az_antenna_axis
            The azimuth of the primary antenna axis, in degrees.
        """
        with Path(path).open("r") as fn:
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

            output = np.zeros((
                int(number_of_frequencies),
                int(rows_per_frequency),
                int(number_of_columns),
            ))

        frequencies = np.array(frequencies_list)

        with Path(path).open("r") as fn:
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
            out2d = output[i, :, :]

            phi = out2d[:, 0]
            theta = 90 - out2d[:, 1]  # theta is zero at the zenith, and goes to 180 deg
            gain = out2d[:, 6]

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
    def from_ideal(cls, delta_f=2, f_low=40, f_high=200, delta_az=1, delta_el=1):
        """Create an ideal beam that is completely unity."""
        freq = np.arange(f_low, f_high, delta_f)
        az = np.arange(0, 360, delta_az)
        el = np.arange(0, 90 + 0.1 * delta_el, delta_el)
        return Beam(
            frequency=freq * u.MHz,
            azimuth=az,
            elevation=el,
            beam=np.ones((len(freq), len(el), len(az))),
            simulator="ideal",
        )

    @classmethod
    def from_feko(cls, path: str | Path, az_antenna_axis: float = 0) -> Beam:
        """
        Read a FEKO beam file.

        Parameters
        ----------
        filename
            The path to the file.
        az_antenna_axis
            The azimuth of the primary antenna axis, in degrees.
        """
        filename = Path(path)

        data = np.genfromtxt(str(filename))
        frequency = []
        with filename.open("r") as fl:
            for line in fl:
                if line.startswith("#FREQUENCY"):
                    line = line.split("MHz")[0].strip().split(" ")
                    frequency.append(float(line[-1]))
        freq = FrequencyRange(np.array(frequency) * u.MHz)

        # Loading data and convert to linear representation
        beam_maps = np.zeros((len(frequency), 91, 360))
        for i in range(len(frequency)):
            beam_maps[i] = (10 ** (data[(i * 360) : ((i + 1) * 360), 2::] / 10)).T

        # Shifting beam relative to true AZ (referenced at due North)
        # Due to angle of orientation of excited antenna panels relative to due North
        beam_maps = cls.shift_beam_maps(az_antenna_axis, beam_maps)

        return Beam(
            frequency=freq.freq,
            beam=beam_maps,
            azimuth=np.arange(0, 360),
            elevation=np.arange(0, 91),
            simulator="feko",
            raw_file=path,
        )

    @classmethod
    def from_cst(
        cls,
        file_name_prefix: tp.PathLike,
        f_low: int = 40,
        f_high: int = 100,
        freq_p: int = 61,
        theta_p: float = 181,
        phi_p: float = 361,
        az_antenna_axis: float = 0,
    ) -> Beam:
        """
        Read a CST beam file.

        Parameters
        ----------
        filename
            The path to the file.
        az_antenna_axis
            The azimuth of the primary antenna axis, in degrees.
        f_low, f_high
            lower and higher frequency bounds
        freq_p
            Number of frequency points in the simulation file.
        theta_p
            Number of zenith angle points in the simulation file.
        phi_p
            Number of azimuth points in the simulation file.

        Returns
        -------
        beam
            The beam object.
        """
        phi_t = np.zeros((freq_p, theta_p * phi_p))
        frequency = np.linspace(f_low, f_high, freq_p)
        beam_square = np.zeros((freq_p, theta_p, phi_p))
        res = (f_high - f_low) / (freq_p - 1)

        for i in range(freq_p):
            n = int(i * res + f_low)
            thispath = Path(f"{file_name_prefix}farfield (f={n}) [1].txt")
            with thispath.open("rb") as fc_file:
                for x, line in enumerate(fc_file):
                    if x > 1:
                        check = 0
                        for o in range(len(line)):
                            if line[o] != "":
                                check = check + 1
                                if check == 3:
                                    phi_t[i][x - 2] = float(line.split()[2])

        for i in range(freq_p):
            for x in range(phi_p):
                beam_square[i, :, x] = phi_t[i, x * theta_p : (x + 1) * theta_p]
            beam_square[i, :, 0 : int(phi_p / 2)] = beam_square[
                i, :, int(phi_p / 2) : phi_p - 1
            ]

        freq = FrequencyRange(np.array(frequency) * u.MHz)
        beam_square[:, 91:, :] = 0
        beam_maps = np.flip(beam_square[:, :91, :360], axis=1)
        # Shifting beam relative to true AZ (referenced at due North)
        # Due to angle of orientation of excited antenna panels relative to due North
        beam_maps = cls.shift_beam_maps(az_antenna_axis, beam_maps)

        return Beam(
            frequency=freq.freq,
            beam=beam_maps,
            azimuth=np.arange(0, 360),
            elevation=np.arange(0, 91),
            simulator="cst",
            raw_file=file_name_prefix,
        )

    @classmethod
    def from_feko_raw(
        cls,
        file_name_prefix: tp.PathLike,
        ext: str = "txt",
        f_low: int = 40,
        f_high: int = 100,
        freq_p: int = 61,
        theta_p: float = 181,
        phi_p: float = 361,
        az_antenna_axis: float = 0,
    ) -> Beam:
        """
        Read a FEKO beam file.

        Parameters
        ----------
        filename
            The path to the file.
        az_antenna_axis
            The azimuth of the primary antenna axis, in degrees.
        f_low, f_high
            lower and higher frequency bounds
        freq_p
            Number of frequency points in the simulation file.
        theta_p
            Number of zenith angle points in the simulation file.
        phi_p
            Number of azimuth points in the simulation file.

        Returns
        -------
        beam
            The beam object.
        """
        beam_square = np.zeros((freq_p, theta_p, phi_p))
        frequency = np.linspace(f_low, f_high, freq_p)

        # TODO: change this to no.of theta * no.of phi + No.of header lines
        z = theta_p * 91 + 10

        def read_file(path, idx: int):
            with path.open("r") as fl:
                for index, line in enumerate(fl):
                    if index % z == 0:
                        co = 0
                    if index % z >= 10:
                        x = list(map(float, line.split()))
                        here = int(index / z), co % theta_p, int(co / theta_p) + idx
                        beam_square[here] = 10 ** (x[8] / 10)
                        co += 1

        read_file(Path(f"{file_name_prefix}_0-90.{ext}"), 0)

        z = theta_p * 90 + 10  #

        read_file(Path(f"{file_name_prefix}_91-180.{ext}"), 91)
        read_file(Path(f"{file_name_prefix}_181-270.{ext}"), 181)
        read_file(Path(f"{file_name_prefix}_271-360.{ext}"), 271)

        freq = FrequencyRange(np.array(frequency) * u.MHz)
        beam_square[:, 90:, :] = 0
        beam_maps = np.flip(beam_square[:, :91, :360], axis=1)
        # Shifting beam relative to true AZ (referenced at due North)
        # Due to angle of orientation of excited antenna panels relative to due North
        beam_maps = cls.shift_beam_maps(az_antenna_axis, beam_maps)

        return Beam(
            frequency=freq.freq,
            beam=beam_maps,
            azimuth=np.arange(0, 360),
            elevation=np.arange(0, 91),
            simulator="feko",
            raw_file=file_name_prefix,
        )

    @classmethod
    def get_beam_path(cls, band: str, kind: str | None = None) -> Path:
        """Get a standard path to a beam file."""
        pth = BEAM_PATH / band / "default.txt" if not kind else f"{kind}.txt"
        if not pth.exists():
            raise FileNotFoundError(f"No beam exists for band={band}.")
        return pth

    def select_freqs(self, indx: tp.Sequence[int]) -> Beam:
        """Select a subset of frequencies.

        Parameters
        ----------
        indx
            The indices of the frequencies to select.
        """
        if not hasattr(indx, "__len__"):
            indx = [indx]

        return attrs.evolve(
            self,
            frequency=self.frequency[indx],
            beam=self.beam[indx],
        )

    def at_freq(
        self,
        freq: tp.FreqType,
        model: mdl.Model = mdl.Polynomial(
            n_terms=13, transform=mdl.ScaleTransform(scale=75.0)
        ),
        **fit_kwargs,
    ) -> Beam:
        """
        Interpolate the beam to a new set of frequencies.

        Parameters
        ----------
        freq
            Frequencies to interpolate to.
        model
            The model to use for interpolation.
        fit_kwargs
            Keyword arguments to pass to the model fit.

        Returns
        -------
        beam
            The Beam object at the new frequencies.
        """
        if len(self.frequency) < 3:
            raise ValueError(
                "Can't freq-interpolate beam that has fewer than three frequencies."
            )

        # Frequency interpolation
        interp_beam = np.zeros((len(freq), len(self.elevation), len(self.azimuth)))
        cached_model = model.at(x=self.frequency.to_value("MHz"))
        for i, bm in enumerate(self.beam.T):
            for j, b in enumerate(bm):
                model_fit = cached_model.fit(ydata=b, **fit_kwargs)
                interp_beam[:, j, i] = model_fit.evaluate(freq.to_value("MHz"))

        return Beam(
            frequency=freq,
            azimuth=self.azimuth,
            elevation=self.elevation,
            beam=interp_beam,
        )

    def smoothed(
        self, model: mdl.Model = mdl.Polynomial(n_terms=12), **fit_kwargs
    ) -> Beam:
        """
        Return a new beam, smoothed over the frequency axis, but without decimation.

        Parameters
        ----------
        model
            The model to use for smoothing.
        fit_kwargs
            Keyword arguments to pass to the model fit.

        Returns
        -------
        beam
            The Beam object smoothed over its same frequencies.
        """
        if len(self.frequency) < 3:
            raise ValueError(
                "Can't freq-interpolate beam that has fewer than three frequencies."
            )

        # Frequency smoothing
        smooth_beam = np.zeros_like(self.beam)
        cached_model = model.at(x=self.frequency.to_value("MHz"))
        for i, bm in enumerate(self.beam.T):
            for j, b in enumerate(bm):
                model_fit = cached_model.fit(ydata=b, **fit_kwargs)
                smooth_beam[:, j, i] = model_fit.evaluate()
        return Beam(
            frequency=self.frequency,
            azimuth=self.azimuth,
            elevation=self.elevation,
            beam=smooth_beam,
        )

    @staticmethod
    def shift_beam_maps(az_antenna_axis: float, beam_maps: np.ndarray) -> np.ndarray:
        """Rotate beam maps around an axis.

        Parameters
        ----------
        az_antenna_axis
            The aximuth angle of the antenna axis.
        beam_maps
            Beam maps as a function of frequency, za and az.

        Returns
        -------
        beam maps
            Array of the same shape as the input, but rotated.
        """
        if az_antenna_axis < 0:
            index = -az_antenna_axis
            bm1 = beam_maps[:, :, index::]
            bm2 = beam_maps[:, :, 0:index]
            return np.append(bm1, bm2, axis=2)
        elif az_antenna_axis > 0:
            index = az_antenna_axis
            bm1 = beam_maps[:, :, 0:(-index)]
            bm2 = beam_maps[:, :, (360 - index) :]
            return np.append(bm2, bm1, axis=2)
        else:
            return beam_maps

    @classmethod
    def resolve_file(
        cls,
        path: tp.PathLike,
        band: str | None = None,
        configuration: str = "default",
        simulator: str = "feko",
    ) -> Path:
        """Resolve a file path to a standard location."""
        if str(path) == ":":
            if band is None:
                raise ValueError("band must be given if path starts with a colon (:)")

            # Get the default beam file.
            return BEAM_PATH / band / f"{configuration}.txt"
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
        band: str | None,
        simulator: str = "feko",
        beam_file: tp.PathLike = ":",
        configuration: str = "default",
        rotation_from_north: float = 90,
    ) -> Beam:
        """Read a beam from file."""
        beam_file = cls.resolve_file(beam_file, band, configuration, simulator)

        if simulator == "feko":
            out = cls.from_feko(beam_file, az_antenna_axis=rotation_from_north)
        elif simulator == "wipl-d":  # Beams from WIPL-D
            out = cls.from_wipld(beam_file, rotation_from_north)
        elif simulator == "hfss":
            out = cls.from_hfss(beam_file)
        else:
            raise ValueError(f"Unknown value for simulator: '{simulator}'")

        out.instrument = band
        return out

    def angular_interpolator(
        self,
        freq_indx: int,
        interp_kind: Literal[
            "linear",
            "nearest",
            "slinear",
            "quintic",
            "pchip",
            "spline",
            "sphere-spline",
        ] = "sphere-spline",
    ) -> callable[[np.ndarray, np.ndarray], np.ndarray]:
        """Return a callable function that interpolates the beam.

        The returned function has the signature ``interp(az, el)``, where ``az`` is
        azimuth in degrees, and ``el`` is elevation in degrees. They may be arrays,
        in which case they should be the same length.
        """
        beam = self.beam[freq_indx]

        if interp_kind == "sphere-spline":
            el = self.elevation * np.pi / 180 + np.pi / 2

            if el[-1] > 0.999 * np.pi:
                el = el[:-1]
                beam = beam[:-1]

            spl = spi.RectSphereBivariateSpline(
                el,
                self.azimuth * np.pi / 180,
                beam,
                pole_values=(None, beam[-1]),
                pole_exact=True,
            )
            return lambda az, el: spl(
                el * np.pi / 180 + np.pi / 2, az * np.pi / 180, grid=False
            )
        else:
            az = np.concatenate([self.azimuth, [self.azimuth[0] + 360]])
            beam = np.hstack((beam, beam[:, 0][:, None]))
            if interp_kind == "spline":
                spl = spi.RectBivariateSpline(
                    az, self.elevation, beam.T, kx=3, ky=3, s=0
                )
                return lambda az, el: spl(az, el, grid=False)
            else:
                spl = spi.RegularGridInterpolator(
                    (az, self.elevation),
                    beam.T,
                    method=interp_kind,
                )
                return lambda az, el: spl(np.array([az, el]).T)

    def between_freqs(
        self, low: tp.FreqType = 0 * u.MHz, high: tp.Freqtype = np.inf * u.MHz
    ) -> Beam:
        """Return a new :class:`Beam` object restricted to a given frequency range."""
        mask = (self.frequency >= low) & (self.frequency <= high)
        return attrs.evolve(self, frequency=self.frequency[mask], beam=self.beam[mask])

    def get_beam_solid_angle(self) -> float:
        """Calculate the integrated beam solid angle."""
        sin_theta = np.cos(self.elevation * (np.pi / 180))
        sin_theta = np.tile(sin_theta, (len(self.azimuth), 1)).T

        beam_integration = np.sum(self.beam * sin_theta, axis=(1, 2))

        d_el = self.elevation[1] - self.elevation[0]
        d_az = self.azimuth[1] - self.azimuth[0]
        return d_el * d_az * (np.pi / 180) ** 2 * beam_integration


@hickleable()
@attrs.define(slots=False)
class BeamFactor:
    """A non-interpolated beam factor.

    This class holds the attributes necessary to compute beam factors at particular
    LSTs and frequencies, namely the antenna temperature (beam-weighted integral of the
    sky) and the same at a particular reference frequency. We hold these separately
    to enable computing the beam factor in different ways from these basic quantities.

    Attributes
    ----------
    frequencies: np.ndarray
        The frequencies at which the beam-weighted sky integrals are defined.
    lsts: np.ndarray
        The LSTs at which the beam-weighted sky integrals are defined.
    reference_frequency: float
        The reference frequency.
    antenna_temp: np.ndarray
        The beam-weighted sky integrals at each frequency and LST.
    antenna_temp_ref: np.ndarray
        The beam-weighted sky integrals at the reference frequency and each LST.
    loss_fraction: np.ndarray
        The fraction of the sky signal lost below the horizon.
    meta
        A dictionary of metadata.
    """

    frequencies: np.ndarray = attrs.field(converter=np.asarray)
    lsts: np.ndarray = attrs.field(converter=np.asarray)
    reference_frequency: float = attrs.field(
        converter=float,
    )
    antenna_temp: np.ndarray = attrs.field(converter=np.asarray)
    antenna_temp_ref: np.ndarray = attrs.field(converter=np.asarray)
    loss_fraction: np.ndarray | None = attrs.field(default=None)
    meta: dict[str, tp.Any] = attrs.field(factory=dict, converter=dict)

    @property
    def nfreq(self) -> int:
        """The number of frequencies in the beam factor."""
        return len(self.frequencies)

    @property
    def nlst(self) -> int:
        """The number of LSTs in the beam factor."""
        return len(self.lsts)

    @frequencies.validator
    def _check_frequencies(self, attribute: attrs.Attribute, value: np.ndarray) -> None:
        if not np.all(np.diff(value) > 0):
            raise ValueError("Frequencies must be monotonically increasing.")

    @lsts.validator
    def _check_lsts(self, attribute: attrs.Attribute, value: np.ndarray) -> None:
        # LSTs can wrap around 24 hours, but only once.
        if np.sum(np.diff(value) < 0) > 1:
            raise ValueError("LSTs must be monotonically increasing.")

    @reference_frequency.validator
    def _check_reference_frequency(
        self, attribute: attrs.Attribute, value: float
    ) -> None:
        if value <= 0:
            raise ValueError("Reference frequency must be positive.")

    @antenna_temp.validator
    @loss_fraction.validator
    def _check_antenna_temp(
        self, attribute: attrs.Attribute, value: np.ndarray
    ) -> None:
        if attribute.name == "loss_fraction" and value is None:
            return

        if value.ndim != 2:
            raise ValueError(f"{attribute.name} must be a 2D array.")

        if value.shape != (self.nlst, self.nfreq):
            raise ValueError(f"{attribute.name} must have shape (nlst, nfreq).")

    @antenna_temp.validator
    def _check_positive(self, attribute: str, value: np.ndarray) -> None:
        if np.any(value < 0):
            raise ValueError(
                f"Antenna temperature must be positive, got min of {np.nanmin(value)}"
            )

    @antenna_temp_ref.validator
    def _check_antenna_temp_ref(self, attribute: str, value: np.ndarray) -> None:
        if value.ndim not in (1, 2):
            raise ValueError("Reference antenna temperature must be a 1D or 2D array.")

        if value.ndim == 1 and value.shape != (self.nlst,):
            raise ValueError(
                "If Reference antenna temperature is 1D, it must have shape (nlst,)."
                f"Got shape {value.shape} instead of {(self.nlst,)}"
            )

        if value.ndim == 2 and value.shape != (self.nlst, self.nfreq):
            raise ValueError(
                "If Reference antenna temperature is 2D, it must have shape "
                "(nlst, nfreq)."
            )

        if np.any(value < 0):
            raise ValueError(
                "Reference antenna temperature must be positive, "
                f"got min of {np.nanmin(value)}"
            )

    def at_lsts(self, lsts: np.ndarray, interp_kind: int | str = "cubic") -> BeamFactor:
        """Return a new BeamFactor at the given LSTs."""
        d = attrs.asdict(self)

        lst_like = [
            k
            for k, v in d.items()
            if isinstance(v, np.ndarray) and v.shape[0] == self.nlst
            if k != "lsts"
        ]

        these_lsts = self.lsts % 24
        while np.any(these_lsts < these_lsts[0]):
            these_lsts[these_lsts < these_lsts[0]] += 24

        use_lsts = lsts % 24
        use_lsts[use_lsts < these_lsts[0]] += 24
        these_lsts = np.append(these_lsts, these_lsts[0] + 24)
        out = {}
        for k in lst_like:
            if d[k].ndim == 2:
                val = np.vstack((d[k], d[k][0]))
            elif d[k].ndim == 1:
                val = np.concatenate((d[k], [d[k][0]]))

            out[k] = spi.interp1d(these_lsts, val, axis=0, kind=interp_kind)(use_lsts)

        return attrs.evolve(self, lsts=lsts, **out)

    def between_lsts(self, lst0: float, lst1: float) -> BeamFactor:
        """Return a new BeamFactor including only LSTs between those given.

        Parameters
        ----------
        lst0
            Lower edge of lsts in hours.
        lst1
            Upper edge of lsts in hours.
        """
        if lst1 < lst0:
            lst1 += 24

        these_lsts = self.lsts % 24
        these_lsts[these_lsts < lst0] += 24

        mask = np.logical_and(these_lsts >= lst0, these_lsts < lst1)
        if not np.any(mask):
            raise ValueError(
                f"BeamFactor does not contain any LSTs between {lst0} and {lst1}."
            )
        d = attrs.asdict(self)
        lst_like = [
            k
            for k, v in d.items()
            if isinstance(v, np.ndarray) and v.shape[0] == self.nlst  # and v.ndim == 2
            if k != "lsts"
        ]

        out = {k: getattr(self, k)[mask] for k in lst_like}
        return attrs.evolve(self, lsts=these_lsts[mask], **out)

    def get_beam_factor(
        self, model: mdl.Model, freqs: np.ndarray | None = None
    ) -> np.ndarray:
        """Return the beam factor as a function of LST and frequency.

        This will always be normalized to unity at the reference frequency, via
        a model fit.
        """
        if freqs is None:
            freqs = self.frequencies

        bf = (self.antenna_temp.T / self.antenna_temp_ref.T).T

        fixed_model = model.at(x=self.frequencies)
        ref_bf = np.zeros(self.nlst)
        out = np.zeros((self.nlst, len(freqs)))
        for i, ibf in enumerate(bf):
            fit = fixed_model.fit(ibf)
            ref_bf = fit.evaluate(self.reference_frequency)
            out[i] = fit.evaluate(freqs) / ref_bf

        return out

    def get_mean_beam_factor(
        self, model: mdl.Model, freqs: np.ndarray | None
    ) -> np.ndarray:
        """Return the mean beam factor over all LSTs."""
        return np.mean(self.get_beam_factor(model, freqs), axis=0)

    def get_integrated_beam_factor(
        self, model: mdl.Model, freqs: np.ndarray | None = None
    ) -> np.ndarray:
        """Return the beam factor integrated over the LST range.

        This is the ratio of summed LSTs, rather than the sum of the ratio at each LST,
        i.e. it is not the same as the mean beam factor over the LST range.
        It is normalized to unity at the reference frequency via a model fit.
        """
        if freqs is None:
            freqs = self.frequencies

        bf = np.sum(self.antenna_temp, axis=0) / np.sum(self.antenna_temp_ref, axis=0)
        fit = model.fit(self.frequencies, bf)
        return fit.evaluate(freqs) / fit.evaluate(self.reference_frequency)


def sky_convolution_generator(
    lsts: np.ndarray,
    ground_loss_file: str,
    beam: Beam,
    sky_model: sky_models.SkyModel,
    index_model: sky_models.IndexModel,
    normalize_beam: bool,
    beam_smoothing: bool,
    smoothing_model: mdl.Model,
    location: apc.EarthLocation = const.KNOWN_TELESCOPES["edges-low"].location,
    ref_time: apt.Time = REFERENCE_TIME,
    interp_kind: Literal[
        "linear",
        "nearest",
        "slinear",
        "cubic",
        "quintic",
        "pchip",
        "spline",
        "sphere-spline",
    ] = "sphere-spline",
    lst_progress: bool = True,
    freq_progress: bool = True,
    ref_freq_idx: int = 0,
    use_astropy_azel: bool = True,
):
    """
    Iterate through given LSTs and generate a beam*sky product at each freq and LST.

    This is a generator, so it will yield a single item at a time (to save on memory).

    Parameters
    ----------
    lsts
        The LSTs at which to evaluate the convolution.
    ground_loss_file
        A path to a file containing ground loss information.
    beam
        The beam to convolve.
    sky_model
        The sky model to convolve
    index_model
        The spectral index model of the sky model.
    normalize_beam
        Whether to ensure the beam is properly normalised.
    beam_interpolation
        Whether to smooth over freq axis
    interp_kind
        The kind of interpolation to use for the beam. "spline" uses
        :class:`scipy.interpolate.RectBivariateSpline` and "sphere-spline" uses
        :class:`scipy.interpolate.RectSphereBivariateSpline`. All other options use
        :class:`scipy.interpolate.RegularGridInterpolator`. with the given kind as
        ``method``.
    use_astropy_azel
        Whether to use the astropy coordinate system for azimuth and elevation. If
        False, compute the az/el using Alan's method.

    Yields
    ------
    i
        The LST enumerator
    j
        The frequency enumerator
    mean_conv_temp
        The mean temperature after multiplying by the beam (above the horizon)
    conv_temp
        An array containing the temperature after multiuplying by the beam in each pixel
        above the horizon.
    sky
        An array containing the sky temperature in pixel above the horizon.
    beam
        An array containing the interpolatedbeam in pixels above the horizon
    time
        The local time at each LST.
    n_pixels
        The total number of pixels that are not masked.

    Examples
    --------
    Use this function as follows:

    >>> for i, j, mean_t, conv_t, sky, bm, time, npix in sky_convolution_generator():
    >>>     print(conv_t)
    """
    if beam_smoothing:
        beam = beam.smoothed(smoothing_model)

    if ground_loss_file is not None:
        ground_gain = ground_loss(
            filename=ground_loss_file,
            band=beam.instrument,
            freq=beam.frequency.to_value("MHz"),
        )
    else:
        ground_gain = np.ones(beam.frequency.size)

    # Get the local times corresponding to the given LSTs
    times = gscrd.lsts_to_times(lsts * u.hourangle, ref_time, location)

    beam_above_horizon = np.full(sky_model.coords.shape, np.nan)
    interpolators = {}

    for lst_idx, time in tqdm(enumerate(times), unit="LST", disable=not lst_progress):
        # Transform Galactic coordinates of Sky Model to Local coordinates
        if use_astropy_azel:
            altaz = sky_model.coords.transform_to(
                apc.AltAz(location=location, obstime=time)
            )
            az = np.asarray(altaz.az.deg)
            el = np.asarray(altaz.alt.deg)
        else:
            ra, dec = crda.galactic_to_radec(
                sky_model.coords.galactic.b.deg, sky_model.coords.galactic.l.deg
            )
            az, el = crda.radec_azel_from_lst(
                lsts[lst_idx] * np.pi / 12, ra, dec, location.lat.rad
            )
            az *= 180 / np.pi
            el *= 180 / np.pi
        # Number of pixels over FULL SKY (4pi) in the sky model
        n_pix_tot = len(el)

        # Selecting coordinates above the horizon
        horizon_mask = el > 0
        az_above_horizon = az[horizon_mask]
        el_above_horizon = el[horizon_mask]

        # Loop over frequency
        # Using np.roll means we start at the reference frequency and loop around.
        # Starting at ref freq (if there is one) means that we can use the
        # reference beam with other frequencies.
        for freq_idx in tqdm(
            np.roll(range(len(beam.frequency)), -ref_freq_idx),
            unit="Frequency",
            disable=not freq_progress,
        ):
            if freq_idx not in interpolators:
                interpolators[freq_idx] = beam.angular_interpolator(
                    freq_idx, interp_kind=interp_kind
                )

            sky_map = sky_model.at_freq(
                beam.frequency[freq_idx].to_value("MHz"),
                index_model=index_model,
            )
            sky_map[~horizon_mask] = np.nan

            beam_above_horizon *= np.nan

            try:
                beam_above_horizon[horizon_mask] = interpolators[freq_idx](
                    az_above_horizon, el_above_horizon
                )
            except ValueError as e:
                raise ValueError(
                    f"az min/max: {np.min(az_above_horizon), np.max(az_above_horizon)}."
                    f" el min/max: {np.min(el_above_horizon), np.max(el_above_horizon)}"
                ) from e

            # Weight the beam by the pixel resolution of the sky model.
            beam_above_horizon *= sky_model.pixel_res

            # Number of pixels above the horizon that are used.
            n_pix_ok = np.sum(~np.isnan(beam_above_horizon))

            # Number of pixels in the whole sky, but not counting pixels that are
            # above horizon and nan.
            n_pix_tot_no_nan = n_pix_tot - (len(el_above_horizon) - n_pix_ok)

            if normalize_beam:
                solid_angle = np.nansum(beam_above_horizon) / n_pix_tot_no_nan
                beam_above_horizon *= ground_gain[freq_idx] / solid_angle

            antenna_temperature_above_horizon = beam_above_horizon * sky_map

            yield (
                lst_idx,
                freq_idx,
                np.nansum(antenna_temperature_above_horizon) / n_pix_tot_no_nan,
                antenna_temperature_above_horizon,
                sky_map,
                beam_above_horizon,
                time,
                n_pix_tot_no_nan,
                az,
                el,
                interpolators[freq_idx],
            )


def simulate_spectra(
    beam: Beam,
    ground_loss_file: [str, Path] = ":",
    f_low: [None, float] = 0,
    f_high: [None, float] = np.inf,
    normalize_beam: bool = True,
    sky_model: sky_models.SkyModel = sky_models.Haslam408(),
    index_model: sky_models.IndexModel = sky_models.ConstantIndex(),
    lsts: np.ndarray = None,
    beam_smoothing: bool = True,
    smoothing_model: mdl.Model = mdl.Polynomial(n_terms=12),
    interp_kind: Literal[
        "linear",
        "nearest",
        "slinear",
        "cubic",
        "quintic",
        "pchip",
        "spline",
        "sphere-spline",
    ] = "sphere-spline",
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Simulate global spectra from sky and beam models.

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
        Minimum frequency to keep in the simulation (frequencies otherwise defined by
        the beam).
    f_high
        Maximum frequency to keep in the simulation (frequencies otherwise defined by
        the beam).
    normalize_beam
        Whether to normalize the beam to be maximum unity.
    sky_model
        A sky model to use.
    index_model
        An :class:`IndexModel` to use to generate different frequencies of the sky
        model.
    lsts
        The LSTs at which to simulate

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
    if lsts is None:
        lsts = np.arange(0, 24, 0.5)

    antenna_temperature_above_horizon = np.zeros((len(lsts), len(beam.frequency)))
    for i, j, temperature, _, _, _, _, _, _, _, _ in sky_convolution_generator(
        lsts,
        ground_loss_file,
        beam,
        sky_model,
        index_model,
        normalize_beam,
        beam_smoothing,
        smoothing_model,
        interp_kind=interp_kind,
    ):
        antenna_temperature_above_horizon[i, j] = temperature

    return antenna_temperature_above_horizon, beam.frequency, lsts


def antenna_beam_factor(
    beam: Beam,
    ground_loss_file: tp.PathLike | None = ":",
    f_low: tp.FreqType = 0 * u.MHz,
    f_high: tp.Freqtype = np.inf * u.MHz,
    normalize_beam: bool = True,
    sky_model: sky_models.SkyModel = sky_models.Haslam408(),
    index_model: sky_models.IndexModel = sky_models.GaussianIndex(),
    lsts: np.ndarray | None = None,
    reference_frequency: tp.FreqType | None = None,
    beam_smoothing: bool = True,
    smoothing_model: mdl.Model = mdl.Polynomial(n_terms=12),
    interp_kind: Literal[
        "linear",
        "nearest",
        "slinear",
        "cubic",
        "quintic",
        "pchip",
        "spline",
        "sphere-spline",
    ] = "sphere-spline",
    lst_progress: bool = True,
    freq_progress: bool = True,
    location: apc.EarthLocation = const.edges_location,
    sky_at_reference_frequency: bool = True,
    use_astropy_azel: bool = True,
) -> BeamFactor:
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
        Minimum frequency to keep in the simulation (frequencies otherwise defined by
        the beam).
    f_high
        Maximum frequency to keep in the simulation (frequencies otherwise defined by
        the beam).
    normalize_beam
        Whether to normalize the beam to be maximum unity.
    sky_model
        A sky model to use.
    index_model
        An :class:`IndexModel` to use to generate different frequencies of the sky
        model.
    twenty_min_per_lst
        How many periods of twenty minutes fit into each LST bin.
    save_dir
        The directory in which to save the output beam factor.
    save_fname
        The filename to save the output beam factor.
    reference_frequency
        The frequency to take as the "reference", i.e. where the chromaticity will
        be by construction unity.
    lst_progress
        Whether to show a progress bar over the LSTs.
    freq_progress
        Whether to show a progress bar over the frequencies.
    location
        The location of the telescope.

    Returns
    -------
    beam_factor : :class`BeamFactor` instance
    """
    beam = beam.between_freqs(f_low, f_high)

    if lsts is None:
        lsts = np.arange(0, 24, 0.5)

    # Get index of reference frequency
    if reference_frequency is None:
        reference_frequency = (f_high + f_low) / 2

    indx_ref_freq = np.argmin(np.abs(beam.frequency - reference_frequency))
    # Don't reset the reference frequency. Alan uses the discrete ref frequency
    # to get the weighted sky temp, then models that, and divides by the model at
    # non-discrete ref frequency to normalize.

    antenna_temperature_above_horizon = np.zeros((len(lsts), len(beam.frequency)))
    if sky_at_reference_frequency:
        convolution_ref = np.zeros((len(lsts),))

    else:
        convolution_ref = np.zeros((len(lsts), len(beam.frequency)))

    loss_fraction = np.zeros((len(lsts), len(beam.frequency)))
    beamsums = np.zeros((len(lsts), len(beam.frequency)))
    for (
        lst_idx,
        freq_idx,
        temperature,
        _,
        sky,
        bm,
        _,
        npix_no_nan,
        _az,
        _el,
        _interp,
    ) in sky_convolution_generator(
        lsts,
        beam=beam,
        sky_model=sky_model,
        index_model=index_model,
        normalize_beam=normalize_beam,
        beam_smoothing=beam_smoothing,
        smoothing_model=smoothing_model,
        ground_loss_file=ground_loss_file,
        interp_kind=interp_kind,
        lst_progress=lst_progress,
        freq_progress=freq_progress,
        location=location,
        ref_freq_idx=indx_ref_freq,
        use_astropy_azel=use_astropy_azel,
    ):
        # 'temperature' is the mean beam-weighted foreground above the horizon (single
        #               float for this lst, freq). If normalize_beam is True, it's
        #               normalized by the integral of the beam (at each freq)
        # 'sky' is the full sky model for this LST and freq
        # 'bm' is the full beam model for this freq (same for all LSTs)
        antenna_temperature_above_horizon[lst_idx, freq_idx] = temperature

        if freq_idx == indx_ref_freq:
            ref_bm = bm.copy()
            ref_sky = sky.copy()

            # This updates once per LST, on the first frequency iteration
            """
            sky_at_reference_frequency is a toggle between Eq-4 and Eq-A1 from Sims+23

            """
            if sky_at_reference_frequency:
                convolution_ref[lst_idx] = np.nansum(ref_bm * ref_sky) / npix_no_nan

        if not sky_at_reference_frequency:
            convolution_ref[lst_idx, freq_idx] = np.nansum(ref_bm * sky) / npix_no_nan

        beamsums[lst_idx, freq_idx] = np.nansum(bm) / npix_no_nan

        # Loss fraction
        loss_fraction[lst_idx, freq_idx] = 1 - np.nansum(bm) / npix_no_nan

    return BeamFactor(
        frequencies=beam.frequency.to_value("MHz").astype(float),
        lsts=np.array(lsts).astype(float),
        antenna_temp=(
            antenna_temperature_above_horizon
            if normalize_beam
            else antenna_temperature_above_horizon / beamsums
        ),
        antenna_temp_ref=(
            convolution_ref
            if normalize_beam
            else (convolution_ref.T / beamsums[:, indx_ref_freq]).T
        ),
        loss_fraction=loss_fraction,
        reference_frequency=reference_frequency.to_value("MHz"),
        meta={
            "beam_file": str(beam.raw_file),
            "simulator": beam.simulator,
            "f_low": f_low.to_value("MHz"),
            "f_high": f_high.to_value("MHz"),
            "normalize_beam": bool(normalize_beam),
            "sky_model": sky_model.name,
            "index_model": str(index_model),
            "rotation_from_north": float(90),
        },
    )
