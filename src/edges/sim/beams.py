"""Beam models and chromaticity corrections."""

import logging
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Literal, Self

import attrs
import numpy as np
import scipy.interpolate as spi
from astropy import units as u

from edges.data import BEAM_PATH

from .. import modeling as mdl
from .. import types as tp
from ..config import config
from ..units import vld_unit

logger = logging.getLogger(__name__)


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
    ) -> Self:
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
    def from_wipld(cls, path: tp.PathLike, az_antenna_axis: float = 0) -> Self:
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
    def uniform(cls, delta_f=2, f_low=40, f_high=200, delta_az=1, delta_el=1):
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
    def from_feko(cls, path: str | Path, az_antenna_axis: float = 0) -> Self:
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
        data = np.loadtxt(str(filename), usecols=list(range(2, 93)))
        frequency = []
        with filename.open("r") as fl:
            for line in fl:
                if line.startswith("#FREQUENCY"):
                    line = line.split("MHz")[0].strip().split(" ")
                    frequency.append(float(line[-1]))
        freq = np.array(frequency) * u.MHz

        # Loading data and convert to linear representation
        beam_maps = np.zeros((len(frequency), 91, 360))
        for i in range(len(frequency)):
            beam_maps[i] = (10 ** (data[(i * 360) : ((i + 1) * 360)] / 10)).T

        # Shifting beam relative to true AZ (referenced at due North)
        # Due to angle of orientation of excited antenna panels relative to due North
        beam_maps = cls.shift_beam_maps(az_antenna_axis, beam_maps)

        return Beam(
            frequency=freq,
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
    ) -> Self:
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

        freq = np.array(frequency) * u.MHz
        beam_square[:, 91:, :] = 0
        beam_maps = np.flip(beam_square[:, :91, :360], axis=1)
        # Shifting beam relative to true AZ (referenced at due North)
        # Due to angle of orientation of excited antenna panels relative to due North
        beam_maps = cls.shift_beam_maps(az_antenna_axis, beam_maps)

        return Beam(
            frequency=freq,
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
    ) -> Self:
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

        freq = np.array(frequency) * u.MHz
        beam_square[:, 90:, :] = 0
        beam_maps = np.flip(beam_square[:, :91, :360], axis=1)
        # Shifting beam relative to true AZ (referenced at due North)
        # Due to angle of orientation of excited antenna panels relative to due North
        beam_maps = cls.shift_beam_maps(az_antenna_axis, beam_maps)

        return Beam(
            frequency=freq,
            beam=beam_maps,
            azimuth=np.arange(0, 360),
            elevation=np.arange(0, 91),
            simulator="feko",
            raw_file=file_name_prefix,
        )

    @classmethod
    def get_beam_path(cls, band: str, kind: str | None = None) -> Path:
        """Get a standard path to a beam file."""
        pth = f"{kind}.txt" if kind else BEAM_PATH / band / "default.txt"
        if not pth.exists():
            raise FileNotFoundError(f"No beam exists for band={band}.")
        return pth

    def select_freqs(self, indx: Sequence[int]) -> Self:
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
    ) -> Self:
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
    ) -> Self:
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
        if az_antenna_axis > 0:
            index = az_antenna_axis
            bm1 = beam_maps[:, :, 0:(-index)]
            bm2 = beam_maps[:, :, (360 - index) :]
            return np.append(bm2, bm1, axis=2)
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
        if str(path).startswith(":"):
            if band is None:
                raise ValueError("band must be given if path starts with a colon (:)")

            # Use a beam file in the standard directory.
            return (
                config.beams / f"{band}/simulations/{simulator}/{str(path)[1:]}"
            ).absolute()
        return Path(path).absolute()

    @classmethod
    def from_file(
        cls,
        band: str | None,
        simulator: str = "feko",
        beam_file: tp.PathLike = ":",
        configuration: str = "default",
        rotation_from_north: float = 90,
    ) -> Self:
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
        ] = "linear",
    ) -> Callable[[np.ndarray, np.ndarray], np.ndarray]:
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
        az = np.concatenate([self.azimuth, [self.azimuth[0] + 360]])
        beam = np.hstack((beam, beam[:, 0][:, None]))
        if interp_kind == "spline":
            spl = spi.RectBivariateSpline(az, self.elevation, beam.T, kx=3, ky=3, s=0)
            return lambda az, el: spl(az, el, grid=False)
        spl = spi.RegularGridInterpolator(
            (az, self.elevation),
            beam.T,
            method=interp_kind,
        )
        return lambda az, el: spl(np.array([az, el]).T)

    def between_freqs(
        self, low: tp.FreqType = 0 * u.MHz, high: tp.FreqType = np.inf * u.MHz
    ) -> Self:
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

    def compute_ground_loss(self):
        """Compute the ground loss for the beam."""
        raise NotImplementedError("We need to get this right...")
