"""Dataclasses representing S-parameters and reflection coefficients.

These small classes are used throughout the rest of the ``sparams`` sub-package.
"""

from functools import cached_property
from typing import Self

import attrs
import numpy as np
from astropy import units as un
from pygsdata.attrs import npfield

from edges import types as tp
from edges.frequencies import get_mask
from edges.io import calobsdef, read_s1p
from edges.io.serialization import hickleable
from edges.tools import linear_to_decibels


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

        Defined as a network that is passive and symmetric.

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
        """The complex linear gain of the network, i.e. S12."""
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
        """The reverse gain, ``S|12|``, in decibels."""
        return linear_to_decibels(self.s[0, 1])

    @property
    def reverse_isolation(self):
        """The reverse isolation, ``1/|S12|``, in decibels."""
        return -np.abs(self.reverse_gain)

    def voltage_standing_wave_ratio_in(self):
        """The Voltage Standing Wave Ratio (VSWR) of the network input."""
        return (1 + np.abs(self.s11)) / (1 - np.abs(self.s11))

    def voltage_standing_wave_ratio_out(self):
        """The Voltage Standing Wave Ratio (VSWR) of the network output."""
        return (1 + np.abs(self.s22)) / (1 - np.abs(self.s22))


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
    def from_s1p(cls, path: tp.PathLike) -> Self:
        """Create a ReflectionCoefficient by reading a .s1p file.

        Parameters
        ----------
        path
            The path to the .s1p file.
        """
        sparams = read_s1p(path)
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
        delimiter = "," if path.endswith(".csv") else " "

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
            Everything else is passed to the :class:`SParams` objects. This includes
            f_low and f_high.
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
        return cls(
            open=ReflectionCoefficient(
                freqs=freqs,
                reflection_coefficient=np.ones_like(freqs, dtype=complex),
            ),
            short=ReflectionCoefficient(
                freqs=freqs,
                reflection_coefficient=-1 * np.ones_like(freqs, dtype=complex),
            ),
            match=ReflectionCoefficient(
                freqs=freqs,
                reflection_coefficient=np.zeros_like(freqs, dtype=complex),
            ),
        )
