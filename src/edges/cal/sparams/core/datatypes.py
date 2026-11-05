"""Dataclasses representing S-parameters and reflection coefficients.

These small classes are used throughout the rest of the ``sparams`` sub-package.
"""

from functools import cached_property
from typing import TYPE_CHECKING, Self

import attrs
import numpy as np
from astropy import units as un
from pygsdata.attrs import npfield

from edges import types as tp
from edges.frequencies import get_mask
from edges.io import calobsdef, read_s1p
from edges.io.serialization import hickleable
from edges.tools import linear_to_decibels

if TYPE_CHECKING:
    from .network_component_models import Calkit
    from .s11model import S11ModelParams


@hickleable
@attrs.define(frozen=True, kw_only=True, slots=False)
class SParams:
    """A scattering matrix for a two-port network.

    This class eases some of the computations performed with S-parameters. Most of the
    methods are based on https://en.wikipedia.org/wiki/Scattering_parameters.

    See also https://en.wikipedia.org/wiki/Two-port_network#Interrelation_of_parameters
    """

    freqs: tp.FreqType = npfield(dtype=float, possible_ndims=(1,), unit=un.MHz)
    s11: np.ndarray = npfield(
        possible_ndims=(1,),
        dtype=complex,
    )
    s12: np.ndarray = npfield(
        possible_ndims=(1,),
        dtype=complex,
    )
    s22: np.ndarray = npfield(
        possible_ndims=(1,),
        dtype=complex,
    )
    s21: np.ndarray = npfield(
        possible_ndims=(1,),
        dtype=complex,
    )

    @property
    def nfreqs(self):
        """The number of frequencies in the S-matrix (the last axis)."""
        return len(self.freqs)

    @s11.validator
    @s12.validator
    @s22.validator
    @s21.validator
    def _s_vld(self, att, val):
        if val.size != self.nfreqs:
            raise ValueError(
                "Number of frequencies in S-matrix must match freqs length."
            )

    @s22.default
    def _s22_default(self):
        return self.s11

    @s21.default
    def _s21_default(self):
        return self.s12

    @property
    def s(self) -> np.ndarray:
        """The S-matrix as a 2x2xN array."""
        return np.array([[self.s11, self.s12], [self.s21, self.s22]])

    @property
    def smatrix(self) -> np.ndarray:
        """The S-Matrix representation.

        An alias for Sparams.s
        """
        return self.s

    @classmethod
    def from_smatrix(cls, freqs: tp.FreqType, s: np.ndarray) -> Self:
        """Create an SParams from an S-Matrix.

        This merely allows to pass in the S-parameters in matrix form.

        Parameters
        ----------
        s : np.ndarray
            The S-Matrix, shape (2,2,N).
        """
        return cls(
            freqs=freqs,
            s11=s[0, 0],
            s12=s[0, 1],
            s21=s[1, 0],
            s22=s[1, 1],
        )

    @classmethod
    def from_transfer_matrix(cls, freqs: tp.FreqType, t: np.ndarray) -> Self:
        """Create an SMatrix from a transfer matrix.

        See https://en.wikipedia.org/wiki/Scattering_parameters#Scattering_transfer_parameters.

        Parameters
        ----------
        t : np.ndarray
            The transfer matrix, shape (2,2,N).
        """
        detT = t[0, 0] * t[1, 1] - t[0, 1] * t[1, 0]

        return cls.from_smatrix(
            freqs=freqs,
            s=np.array([
                [t[0, 1] / t[1, 1], detT / t[1, 1]],
                [1 / t[1, 1], -t[1, 0] / t[1, 1]],
            ]),
        )

    @cached_property
    def determinant(self):
        """The determinant of the SMatrix."""
        return self.s11 * self.s22 - self.s12 * self.s21

    def as_transfer_matrix(self) -> np.ndarray:
        """Return a transfer matrix from the SMatrix.

        See https://en.wikipedia.org/wiki/Scattering_parameters#Scattering_transfer_parameters.
        Transfer matrices are useful as they are easier to apply to cascading 2-port
        networks (i.e. to get the total effect of several components linked together).
        """
        s = self.s

        return np.array([
            [-self.determinant / s[1, 0], s[0, 0] / s[1, 0]],
            [-s[1, 1] / s[1, 0], 1 / s[1, 0]],
        ])

    def cascade_with(self, other: Self) -> Self:
        """Return a new TwoPortNetwork from the conjunction of two two-port networks.

        To achieve this, we first convert to Transfer matrices, then perform
        matrix multiplication, and then convert back to a SMatrix.
        """
        if self.s.shape != other.s.shape:
            raise ValueError("Both SMatrices must have the same shape to add them.")

        t1 = self.as_transfer_matrix()
        t2 = other.as_transfer_matrix()
        t = np.matmul(t1.transpose((2, 0, 1)), t2.transpose((2, 0, 1))).transpose((
            1,
            2,
            0,
        ))

        return SParams.from_transfer_matrix(freqs=self.freqs, t=t)

    def is_reciprocal(self) -> bool:
        """Whether the S-matrix describes a reciprocal network.

        A reciprocal network has a symmetric S-matrix, i.e. S12 == S21. (This says
        nothing about whether the network is passive, nor whether S11 == S22.)

        See https://en.wikipedia.org/wiki/Scattering_parameters#Reciprocity
        """
        return np.allclose(self.s, self.s.transpose((1, 0, 2)))

    def is_lossless(self):
        """Whether the S-matrix describes a lossless network.

        See https://en.wikipedia.org/wiki/Scattering_parameters#Lossless_networks
        """
        product = np.matmul(
            self.s.transpose((2, 0, 1)), self.s.transpose((2, 1, 0)).conj()
        )
        return np.allclose(product, np.eye(2))

    @property
    def complex_linear_gain(self) -> tp.ComplexArray:
        """The complex linear (forward) gain of the network, i.e. S21."""
        return self.s[1, 0]

    @property
    def scalar_linear_gain(self) -> tp.FloatArray:
        """The abs value of the linear gain of the network, ``|S21|``."""
        return np.abs(self.complex_linear_gain)

    @property
    def scalar_logarithmic_gain(self) -> tp.FloatArray:
        """The scalar gain, ``|S21|``, in decibels."""
        return linear_to_decibels(self.scalar_linear_gain)

    @property
    def insertion_loss(self):
        """The insertion loss of the network.

        See https://en.wikipedia.org/wiki/Scattering_parameters#Insertion_loss
        """
        return -self.scalar_logarithmic_gain

    @property
    def input_return_loss(self):
        """The loss, ``1/|S11|`` in decibels."""
        return -linear_to_decibels(self.s[0, 0])

    @property
    def output_return_loss(self):
        """The output loss, ``1/|S22|``, in decibels."""
        return -linear_to_decibels(self.s[1, 1])

    @property
    def reverse_gain(self):
        """The reverse gain, ``|S12|``, in decibels."""
        return linear_to_decibels(self.s[0, 1])

    @property
    def reverse_isolation(self):
        """The reverse isolation, ``1/|S12|``, in decibels.

        This is the negative of :attr:`reverse_gain`, and is positive for a passive
        network (``|S12| < 1``).
        """
        return -self.reverse_gain

    def voltage_standing_wave_ratio_in(self):
        """The Voltage Standing Wave Ratio (VSWR) of the network input."""
        return (1 + np.abs(self.s11)) / (1 - np.abs(self.s11))

    def voltage_standing_wave_ratio_out(self):
        """The Voltage Standing Wave Ratio (VSWR) of the network output."""
        return (1 + np.abs(self.s22)) / (1 - np.abs(self.s22))

    @staticmethod
    def from_calkit_measurements(
        measurements: "CalkitReadings",
        model: "CalkitReadings | Calkit | None" = None,
    ) -> "SParams":
        """Compute S-parameters of a 2-port network from calkit measurements.

        This is a thin wrapper around
        :func:`~edges.cal.sparams.core.sparam_calibration.sparams_from_calkit_measurements`
        (Eq. 3 of Monsalve et al., 2016).

        Parameters
        ----------
        measurements
            The actual measurements of the calkit standards.
        model
            A model of the calkit standards: either the model reflection
            coefficients at the measured frequencies (:class:`CalkitReadings`), or a
            :class:`~edges.cal.sparams.Calkit`, which is evaluated at the measured
            frequencies. If None, ideal standards are assumed.

        Returns
        -------
        sparams
            The S-parameters of the network between the reference plane of the
            measurements and the standards. Only the product S12*S21 is determined
            by the calibration; S12 and S21 are both set to its square root, taken
            on the branch that is continuous across frequency.
        """
        from .sparam_calibration import sparams_from_calkit_measurements

        return sparams_from_calkit_measurements(measurements, model)

    def smoothed(
        self,
        params: "S11ModelParams | dict[str, S11ModelParams]",
        freqs: tp.FreqType | None = None,
    ) -> "SParams":
        """Smooth all S-parameters using the S11 modeling procedure.

        This is a thin wrapper around
        :func:`~edges.cal.sparams.core.s11model.smooth_sparams`.

        Parameters
        ----------
        params
            The set of parameters defining the model used to smooth/interpolate,
            either one set for all S-parameters, or a dict of them keyed by
            ``"s11"``, ``"s12"``, ``"s21"`` and ``"s22"``.
        freqs
            Optional new frequencies onto which to interpolate. If not given,
            retain the same set of frequencies.

        Returns
        -------
        smoothed_sparams
            A new SParams object that has been smoothed.
        """
        from .s11model import smooth_sparams

        return smooth_sparams(self, params, freqs=freqs)


@hickleable
@attrs.define(kw_only=True, frozen=True)
class ReflectionCoefficient:
    """A small dataclass to represent a reflection coefficient.

    Reflection coefficients are complex-valued functions of frequency,
    typically denoted by the Greek letter Gamma (Γ). They represent the ratio
    of the reflected wave amplitude to the incident wave amplitude at a given frequency,
    at a specific reference plane in a network.

    This class is a generic class that could represent a measurement or a model,
    and its reference plane is not explicitly defined here, so it could be anywhere.
    """

    freqs: tp.FreqType = npfield(dtype=float, possible_ndims=(1,), unit=un.MHz)
    reflection_coefficient: np.ndarray = npfield(
        dtype=complex,
        possible_ndims=(1,),
    )

    @reflection_coefficient.validator
    def _fv(self, att, val):
        if val.size != len(self.freqs):
            raise ValueError(
                f"len(freq) != len(raw_s11) [{len(val)},{len(self.freqs)}]"
            )

    def select_frequencies(
        self,
        f_low: tp.FreqType = 0 * un.MHz,
        f_high: tp.FreqType = np.inf * un.MHz,
    ) -> Self:
        """Return a ReflectionCoefficient with only frequencies in the given range."""
        mask = get_mask(self.freqs, low=f_low, high=f_high)
        return ReflectionCoefficient(
            freqs=self.freqs[mask],
            reflection_coefficient=self.reflection_coefficient[mask],
        )

    def remove_delay(self, delay: tp.TimeType) -> Self:
        """Return a new ReflectionCoefficient with the delay removed.

        Parameters
        ----------
        delay
            The delay to apply, in time units (e.g., seconds).
        """
        phase_shift = np.exp(2j * np.pi * self.freqs * delay).to_value("")
        return ReflectionCoefficient(
            freqs=self.freqs,
            reflection_coefficient=self.reflection_coefficient * phase_shift,
        )

    @classmethod
    def from_s1p(
        cls,
        path: tp.PathLike,
        f_low: tp.FreqType = 0 * un.MHz,
        f_high: tp.FreqType = np.inf * un.MHz,
    ) -> Self:
        """Create a ReflectionCoefficient by reading a .s1p file.

        Parameters
        ----------
        path
            The path to the .s1p file.
        f_low
            The minimum frequency to keep.
        f_high
            The maximum frequency to keep.
        """
        sparams = read_s1p(path, f_low=f_low, f_high=f_high)
        return cls(
            freqs=sparams["frequency"],
            reflection_coefficient=sparams["s11"],
        )

    @classmethod
    def from_csv(
        cls,
        path: tp.PathLike,
        freq_unit: un.Unit = un.Hz,
    ):
        """Create a ReflectionCoefficient by reading a CSV or space-delimited file.

        The file is expected to have three columns: frequency, real part of Gamma,
        imaginary part of Gamma. The first row is assumed to be a header and skipped.

        Parameters
        ----------
        path
            The path to the CSV or space-delimited file.
        freq_unit
            The unit of the frequency column in the file. Default is Hz.
        """
        delimiter = "," if str(path).endswith(".csv") else " "

        f_orig, gamma_real, gamma_imag = np.loadtxt(
            path,
            skiprows=1,
            delimiter=delimiter,
            unpack=True,
            comments=["BEGIN", "END", "#"],
        )

        return cls(
            reflection_coefficient=gamma_real + 1j * gamma_imag,
            freqs=f_orig * freq_unit,
        )

    @property
    def s11(self) -> np.ndarray:
        """Alias for reflection_coefficient."""
        return self.reflection_coefficient

    def de_embed(self, sparams: SParams) -> "ReflectionCoefficient":
        """Remove the effect of a 2-port network from this reflection coefficient.

        See Eq. 2 of Monsalve et al., 2016 or
        https://en.wikipedia.org/wiki/Scattering_parameters#S-parameters_in_amplifier_design

        Parameters
        ----------
        sparams
            The S-matrix of the 2-port network / subsystem, defined at the same
            frequencies as this reflection coefficient.

        Returns
        -------
        gamma_de_embedded
            The reflection coefficient at the desired reference plane, on the other
            side of the 2-port network.

        See Also
        --------
        embed
            The inverse operation.

        Notes
        -----
        Given this reflection coefficient, observed at a reference plane on one
        side of an electrical component/subsystem, this returns the reflection
        coefficient at the reference plane on the other side of the subsystem::

           ---         ------------
          |VNA| ---|---| SUBSYTEM |---|---
           ---         ------------
                   ^                  ^
                   |                  |
               MEAS. REF.          DESIRED REF.
                 PLANE               PLANE

        """
        gamma_in = self.reflection_coefficient
        return ReflectionCoefficient(
            freqs=self.freqs,
            reflection_coefficient=(gamma_in - sparams.s11)
            / (sparams.s22 * (gamma_in - sparams.s11) + sparams.s12 * sparams.s21),
        )

    def embed(self, sparams: SParams) -> "ReflectionCoefficient":
        """Add the effect of a 2-port network to this reflection coefficient.

        This is the inverse of :meth:`de_embed`.

        Parameters
        ----------
        sparams
            The S-matrix of the 2-port network, defined at the same frequencies as
            this reflection coefficient.

        Returns
        -------
        gamma_ref
            The reflection coefficient at the reference plane on the other side of
            the 2-port network.

        See Also
        --------
        de_embed
            The inverse operation.
        """
        gamma_in = self.reflection_coefficient
        return ReflectionCoefficient(
            freqs=self.freqs,
            reflection_coefficient=(
                sparams.s11
                + (sparams.s12 * sparams.s21 * gamma_in) / (1 - sparams.s22 * gamma_in)
            ),
        )

    def smoothed(
        self,
        params: "S11ModelParams",
        freqs: tp.FreqType | None = None,
    ) -> "ReflectionCoefficient":
        """Return a new, smoothed/modelled, reflection coefficient.

        This is a thin wrapper around
        :func:`~edges.cal.sparams.core.s11model.new_s11_modelled`.

        Parameters
        ----------
        params
            The set of parameters defining the model used to smooth/interpolate.
        freqs
            Optional new frequencies onto which to interpolate. If not given,
            retain the same set of frequencies.

        Returns
        -------
        modelled_s11
            A new ReflectionCoefficient object that has been smoothed.
        """
        from .s11model import new_s11_modelled

        return new_s11_modelled(self, params, freqs=freqs)


@attrs.define
class CalkitReadings:
    """A class representing the full set of calkit measurements.

    This includes the open, short, and match standards.

    Parameters
    ----------
    open
        The open standard S-parameters.
    short
        The short standard S-parameters.
    match
        The match standard S-parameters.
    """

    open: ReflectionCoefficient = attrs.field(
        validator=attrs.validators.instance_of(ReflectionCoefficient)
    )
    short: ReflectionCoefficient = attrs.field(
        validator=attrs.validators.instance_of(ReflectionCoefficient)
    )
    match: ReflectionCoefficient = attrs.field(
        validator=attrs.validators.instance_of(ReflectionCoefficient)
    )

    @short.validator
    @match.validator
    def _vld(self, att, val):
        if val.freqs.size != self.open.freqs.size:
            raise ValueError(f"{att.name} standard does not have same frequencies")

        if np.any(val.freqs != self.open.freqs):
            raise ValueError(
                f"{att.name} standard does not have same frequencies as open standard!"
            )

    @property
    def freqs(self) -> tp.FreqType:
        """Frequencies of the standards measurements."""
        return self.open.freqs

    @classmethod
    def from_filespec(cls, paths: calobsdef.CalkitFileSpec, **kwargs) -> Self:
        """Instantiate from a given Calkit I/O object.

        Other Parameters
        ----------------
        kwargs
            Everything else is passed to :meth:`ReflectionCoefficient.from_s1p` for
            each standard, i.e. ``f_low`` and ``f_high``.
        """
        return cls(
            open=ReflectionCoefficient.from_s1p(paths.open, **kwargs),
            short=ReflectionCoefficient.from_s1p(paths.short, **kwargs),
            match=ReflectionCoefficient.from_s1p(paths.match, **kwargs),
        )

    @classmethod
    def ideal(cls, freqs: tp.FreqType) -> Self:
        """Create an ideal calkit standards reading.

        An ideal calkit has:
        - Open: reflection coefficient of 1
        - Short: reflection coefficient of -1
        - Match: reflection coefficient of 0

        Parameters
        ----------
        freqs
            The frequencies at which to define the ideal standards.
        """
        n = len(freqs)
        return cls(
            open=ReflectionCoefficient(
                freqs=freqs, reflection_coefficient=np.ones(n, dtype=complex)
            ),
            short=ReflectionCoefficient(
                freqs=freqs, reflection_coefficient=-np.ones(n, dtype=complex)
            ),
            match=ReflectionCoefficient(
                freqs=freqs, reflection_coefficient=np.zeros(n, dtype=complex)
            ),
        )

    def de_embed(self, sparams: SParams) -> "CalkitReadings":
        """Remove the effect of a 2-port network from each standard measurement.

        This applies :meth:`ReflectionCoefficient.de_embed` to each of the open,
        short and match readings.

        Parameters
        ----------
        sparams
            The S-parameters of the 2-port network to de-embed.

        Returns
        -------
        readings
            The calkit readings referenced to the other side of the 2-port network.
        """
        return CalkitReadings(**{
            kind: getattr(self, kind).de_embed(sparams)
            for kind in ("open", "short", "match")
        })
