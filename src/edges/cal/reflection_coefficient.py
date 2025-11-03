"""Functions for working with reflection coefficients.

Most of the functions in this module follow the formalism/notation of

    Monsalve et al., 2016, "One-Port Direct/Reverse Method for Characterizing
    VNA Calibration Standards", IEEE Transactions on Microwave Theory and
    Techniques, vol. 64, issue 8, pp. 2631-2639, https://arxiv.org/pdf/1606.02446.pdf

They represent basic relations between physical parameters of circuits, as measured
with internal standards.
"""

from collections.abc import Callable
from functools import cached_property

from ..units import unit_converter

try:
    # only available on py311+
    from typing import Self
except ImportError:
    from typing import Self

import attrs
import numpy as np
import numpy.typing as npt
from astropy import units
from astropy.constants import c as speed_of_light
from pygsdata.attrs import npfield
from scipy.optimize import minimize
from scipy.signal.windows import blackmanharris

from .. import modeling as mdl
from .. import types as tp
from ..io.serialization import hickleable
from ..tools import linear_to_decibels
from . import ee


def impedance2gamma(
    z: float | np.ndarray,
    z0: float | np.ndarray,
) -> float | np.ndarray:
    """Convert impedance to reflection coefficient.

    See Eq. 19 of Monsalve et al. 2016.

    Parameters
    ----------
    z
        Impedance.
    z0
        Reference impedance.

    Returns
    -------
    gamma
        The reflection coefficient.
    """
    return (z - z0) / (z + z0)


def gamma2impedance(
    gamma: float | np.ndarray,
    z0: float | np.ndarray,
) -> float | np.ndarray:
    """Convert reflection coeffient to impedance.

    See Eq. 19 of Monsalve et al. 2016.

    Parameters
    ----------
    gamma
        Reflection coefficient.
    z0
        Reference impedance.

    Returns
    -------
    z
        The impedance.
    """
    return z0 * (1 + gamma) / (1 - gamma)


@hickleable
@attrs.define(frozen=True, slots=False, kw_only=True)
class CalkitStandard:
    """Class representing a calkit standard.

    The standard could be open, short or load/match.
    See the Appendix of Monsalve et al. 2016 for details.

    For all parameters, 'offset' refers to the small transmission
    line section of the standard (not an offset in the parameter).

    Parameters
    ----------
    resistance
        The resistance of the standard termination, either assumed or measured.
    offset_impedance
        Impedance of the transmission line, in Ohms.
    offset_delay
        One-way delay of the transmission line, in picoseconds.
    offset_loss
        One-way loss of the transmission line, unitless.
    """

    resistance: float | tp.ImpedanceType = attrs.field(
        converter=unit_converter(units.ohm)
    )
    offset_impedance: float | tp.ImpedanceType = attrs.field(
        default=50.0 * units.ohm, converter=unit_converter(units.ohm)
    )
    offset_delay: float | tp.TimeType = attrs.field(
        default=30.0 * units.picosecond, converter=unit_converter(units.picosecond)
    )
    offset_loss: float | units.Quantity[units.Gohm / units.s] = attrs.field(
        default=2.2 * units.Gohm / units.s,
        converter=unit_converter(units.Gohm / units.s),
    )

    capacitance_model: Callable | None = attrs.field(default=None)
    inductance_model: Callable | None = attrs.field(default=None)

    @property
    def name(self) -> str:
        """The name of the standard. Inferred from the resistance."""
        if np.abs(self.resistance.to_value("ohm")) > 1000:
            return "open"
        return "short" if np.abs(self.resistance.to_value("ohm")) < 1 else "match"

    @classmethod
    def _verify_freq(cls, freq: np.ndarray | units.Quantity):
        if units.get_physical_type(freq) != "frequency":
            raise TypeError(
                "freq must be a frequency quantity! "
                f"Got {units.get_physical_type(freq)}"
            )

    @property
    def intrinsic_gamma(self) -> float:
        """The intrinsic reflection coefficient of the idealized standard."""
        if np.isinf(self.resistance):
            return 1.0  # np.inf / np.inf
        return impedance2gamma(self.resistance, 50.0 * units.Ohm)

    def termination_impedance(self, freq: tp.FreqType) -> tp.OhmType:
        """The impedance of the termination of the standard.

        See Eq. 22-25 of M16 for open and short standards. The match standard
        uses the input measured resistance as the impedance.
        """
        self._verify_freq(freq)
        freq = freq.to("Hz").value

        if self.capacitance_model is not None:
            return (-1j / (2 * np.pi * freq * self.capacitance_model(freq))) * units.ohm
        if self.inductance_model is not None:
            return 1j * 2 * np.pi * freq * self.inductance_model(freq) * units.ohm
        return self.resistance

    def termination_gamma(self, freq: tp.FreqType) -> tp.DimlessType:
        """Reflection coefficient of the termination.

        Eq. 19 of M16.
        """
        return impedance2gamma(self.termination_impedance(freq), 50 * units.ohm)

    def lossy_characteristic_impedance(self, freq: tp.FreqType) -> tp.OhmType:
        """Obtain the lossy characteristic impedance of the transmission line (offset).

        See Eq. 20 of Monsalve et al., 2016
        """
        self._verify_freq(freq)
        return self.offset_impedance + (1 - 1j) * (
            self.offset_loss / (2 * 2 * np.pi * freq)
        ) * np.sqrt(freq.to("GHz").value)

    def gl(self, freq: tp.FreqType) -> np.ndarray:
        """Obtain the product gamma*length.

        gamma is the propagation constant of the transmission line (offset) and l
        is its length. See Eq. 21 of Monsalve et al. 2016.
        """
        self._verify_freq(freq)

        temp = (
            np.sqrt(freq.to("GHz").value)
            * (self.offset_loss * self.offset_delay)
            / (2 * self.offset_impedance)
        )
        return ((2 * np.pi * freq * self.offset_delay) * 1j + (1 + 1j) * temp).to_value(
            ""
        )

    def offset_gamma(self, freq: tp.FreqType) -> tp.DimlessType:
        """Obtain reflection coefficient of the offset.

        Eq. 19 of M16.
        """
        return impedance2gamma(
            self.lossy_characteristic_impedance(freq), 50 * units.ohm
        )

    def reflection_coefficient(self, freq: tp.FreqType) -> tp.DimlessType:
        """Obtain the combined reflection coefficient of the standard.

        See Eq. 18 of M16.

        Note that, despite looking different to Alan's implementation, this is exactly
        the same as his agilent() function EXCEPT that he doesn't seem to use the
        loss / capacitance models.
        """
        ex = np.exp(-2 * self.gl(freq))
        r1 = self.offset_gamma(freq)
        gamma_termination = self.termination_gamma(freq)
        return (r1 * (1 - ex - r1 * gamma_termination) + ex * gamma_termination) / (
            1 - r1 * (ex * r1 + gamma_termination * (1 - ex))
        ).value

    @classmethod
    def open(cls, resistance=np.inf * units.ohm, **kwargs) -> Self:
        """Create an 'open' calkit standard, with resistance=inf.

        See :class:`CalkitStandard` for all parameters available.
        """
        return cls(resistance=resistance, **kwargs)

    @classmethod
    def short(cls, resistance=0 * units.ohm, **kwargs) -> Self:
        """Create a 'short' calkit standard, with resistance=0.

        See :class:`CalkitStandard` for all parameters available.
        """
        return cls(resistance=resistance, **kwargs)

    @classmethod
    def match(cls, resistance=50.0 * units.ohm, **kwargs) -> Self:
        """Create a 'match' calkit standard.

        See :class:`CalkitStandard` for all possible parameters.
        """
        return cls(resistance=resistance, **kwargs)


@hickleable
@attrs.define(slots=False, frozen=True)
class Calkit:
    """A class holding all calkit standards.

    This is not a class to merely hold Calkit data, but instead to hold electrical
    engineering definitions of calkit standard models.
    """

    open: CalkitStandard = attrs.field()
    short: CalkitStandard = attrs.field()
    match: CalkitStandard = attrs.field()

    @open.validator
    def _open_vld(self, att, val):
        assert val.name == "open"

    @short.validator
    def _short_vld(self, att, val):
        assert val.name == "short"

    @match.validator
    def _match_vld(self, att, val):
        assert val.name == "match"

    def clone(self, *, short=None, open=None, match=None):
        """Return a clone with updated parameters for each standard."""
        return attrs.evolve(
            self,
            open=attrs.evolve(self.open, **(open or {})),
            short=attrs.evolve(self.short, **(short or {})),
            match=attrs.evolve(self.match, **(match or {})),
        )


AGILENT_85033E = Calkit(
    open=CalkitStandard.open(
        offset_impedance=50.0 * units.ohm,
        offset_delay=29.243 * units.picosecond,
        offset_loss=2.2 * units.Gohm / units.s,
        capacitance_model=mdl.Polynomial(
            parameters=[49.43e-15, -310.1e-27, 23.17e-36, -0.1597e-45]
        ),
    ),
    short=CalkitStandard.short(
        offset_impedance=50.0 * units.ohm,
        offset_delay=31.785 * units.picosecond,
        offset_loss=2.36 * units.Gohm / units.s,
        inductance_model=mdl.Polynomial(
            parameters=[2.077e-12, -108.5e-24, 2.171e-33, -0.01e-42]
        ),
    ),
    match=CalkitStandard.match(
        offset_impedance=50.0 * units.ohm,
        offset_delay=38.0 * units.picosecond,
        offset_loss=2.3 * units.Gohm / units.s,
    ),
)

AGILENT_ALAN = Calkit(
    open=CalkitStandard.open(
        offset_impedance=50.0 * units.ohm,
        offset_delay=33 * units.picosecond,
        offset_loss=2.3 * units.Gohm / units.s,
        resistance=1e9 * units.Ohm,
    ),
    short=CalkitStandard.short(
        offset_impedance=50.0 * units.ohm,
        offset_delay=33 * units.picosecond,
        offset_loss=2.3 * units.Gohm / units.s,
        resistance=0 * units.Ohm,
    ),
    match=CalkitStandard.match(
        offset_impedance=50.0 * units.ohm,
        offset_delay=33.0 * units.picosecond,
        offset_loss=2.3 * units.Gohm / units.s,
    ),
)


def get_calkit(
    base,
    resistance_of_match: tp.ImpedanceType | None = None,
    open: dict | None = None,
    short: dict | None = None,
    match: dict | None = None,
):
    """Get a calkit based on a provided base calkit, with given updates.

    Parameters
    ----------
    base
        The base calkit to use, eg. AGILENT_85033E
    resistance_of_match
        The resistance of the match, overwrites default from the base.
    open
        Dictionary of parameters to overwrite the open standard.
    short
        Dictionary of parameters to overwrite the short standard.
    match
        Dictionary of parameters to overwrite the match standard.
    """
    match = match or {}
    if resistance_of_match is not None:
        match.update(resistance=resistance_of_match)
    return base.clone(short=short, open=open, match=match)


@attrs.define(frozen=True, kw_only=False)
class TwoPortNetwork:
    """A matrix-representation of a two-port network.

    This is a matrix representation of a two-port network, defined in terms of
    voltages and currents at ports (in contrast to the SMatrix representation which
    is in terms of reflected waves).

    This class allows for the simple conversion between representations of
    two-port network matrices. The internal representation is the ABCD representation
    (https://en.wikipedia.org/wiki/Two-port_network#ABCD-parameters).
    """

    x: npt.NDArray[complex] = npfield(
        possible_ndims=(3,),
        dtype=complex,
        converter=np.atleast_3d,
    )

    @x.validator
    def _x_vld(self, att, val):
        if val.shape[:2] != (2, 2):
            raise ValueError("Matrix must have shape (2, 2, Nfreq).")

    @classmethod
    def from_zmatrix(cls, z: npt.NDArray) -> Self:
        """Create a TwoPortNetwork from a Z-matrix."""
        z = np.atleast_3d(z)
        nf = z.shape[-1]
        detz = z[0, 0] * z[1, 1] - z[0, 1] * z[1, 0]
        return cls(1 / z[1, 0] * np.array([[z[0, 0], detz], [np.ones(nf), z[1, 1]]]))

    @classmethod
    def from_ymatrix(cls, z: npt.NDArray) -> Self:
        """Create a TwoPortNetwork from a Y-matrix."""
        z = np.atleast_3d(z)
        nf = z.shape[-1]
        detz = z[0, 0] * z[1, 1] - z[0, 1] * z[1, 0]
        return cls(
            1 / z[1, 0] * np.array([[-z[1, 1], -np.ones(nf)], [-detz, -z[0, 0]]])
        )

    @classmethod
    def from_abcd(cls, abcd, inverse: bool = False):
        """Create a TwoPortNetwork from an ABCD representation."""
        return cls(np.linalg.inv(abcd.T).T) if inverse else cls(abcd)

    @property
    def A(self):  # noqa: N802
        """Return the A parameter."""
        return self.x[0, 0]

    @property
    def B(self):  # noqa: N802
        """Return the B parameter."""
        return self.x[0, 1]

    @property
    def C(self):  # noqa: N802
        """Return the C parameter."""
        return self.x[1, 0]

    @property
    def D(self):  # noqa: N802
        """Return the D parameter."""
        return self.x[1, 1]

    @cached_property
    def determinant(self) -> npt.NDArray[float]:
        """The determinant of the ABCD representation, |AD - BC|."""
        return self.A * self.D - self.B * self.C

    def is_reciprocal(self) -> bool:
        """Whether the network is a reciprocal network."""
        return np.allclose(self.determinant, 1)

    def is_symmetric(self) -> bool:
        """Whether the network is symmetric."""
        return np.allclose(self.A, self.D)

    def is_lossless(self) -> bool:
        """Whether the network is lossless."""
        return (
            np.allclose(self.A.imag, 0)
            and np.allclose(self.D.imag, 0)
            and np.allclose(self.B.real, 0)
            and np.allclose(self.C.real, 0)
        )

    def _check_add_args(self, other: Self):
        if not isinstance(other, TwoPortNetwork):
            raise ValueError("Two matrices must be of the same type.")

        if other.x.shape != self.x.shape:
            raise ValueError("Two matrices must have the same dimensions.")

    def add_in_series(self, other: Self) -> Self:
        """Combine two TwoPortNetworks together in series."""
        self._check_add_args(other)
        z = self.zmatrix + other.zmatrix
        return TwoPortNetwork.from_zmatrix(z)

    def add_in_parallel(self, other):
        """Combine two TwoPortNetworks together in parallel."""
        self._check_add_args(other)

        y = self.ymatrix + other.ymatrix
        return TwoPortNetwork.from_ymatrix(y)

    def add_in_series_parallel(self, other):
        """Combine two TwoPortNetworks together in parallel."""
        self._check_add_args(other)

        h = self.hmatrix + other.hmatrix
        return TwoPortNetwork.from_hmatrix(h)

    def cascade_with(self, other: Self) -> Self:
        """Cascade two TwoPortNetworks together."""
        self._check_add_args(other)
        abcd = np.matmul(other.x.T, self.x.T)
        return TwoPortNetwork.from_abcd(abcd.T)

    @property
    def zmatrix(self):
        """Return the Z-matrix (impedance parameters) of the network."""
        nf = self.B.shape[-1]
        return (1 / self.C) * np.array([
            [self.A, self.determinant],
            [np.ones(nf), self.D],
        ])

    @property
    def impedance_matrix(self):
        """Alias of zmatrix."""
        return self.zmatrix

    @property
    def ymatrix(self):
        """Return the Y-matrix (admittance parameters) of the network.

        This is the inverse of the z-matrix.
        """
        nf = self.B.shape[-1]
        return (1 / self.B) * np.array([
            [self.D, -self.determinant],
            [-np.ones(nf), self.A],
        ])

    @property
    def admittance_matrix(self):
        """Alias of ymatrix."""
        return self.ymatrix

    @classmethod
    def from_hmatrix(cls, z: npt.NDArray) -> Self:
        """Create a TwoPortNetwork from a H-matrix."""
        z = np.atleast_3d(z)
        nf = z.shape[-1]
        detz = z[0, 0] * z[1, 1] - z[0, 1] * z[1, 0]
        return cls(
            1 / z[1, 0] * np.array([[-detz, -z[0, 0]], [-z[1, 1], -np.ones(nf)]])
        )

    @property
    def hmatrix(self):
        """Return the H-matrix (hybrid parameters) of the network."""
        nf = self.B.shape[-1]
        return (1 / self.D) * np.array([
            [self.B, self.determinant],
            [-np.ones(nf), self.C],
        ])

    @property
    def hybrid_matrix(self):
        """Alias of hmatrix."""
        return self.hmatrix

    def as_smatrix(
        self, source_impedance: float, load_impedance: float | None = None
    ) -> "SMatrix":
        """Convert the TwoPortNetwork to a SMatrix."""
        if load_impedance is None:
            load_impedance = source_impedance

        a, b, c, d = self.A, self.B, self.C, self.D
        zs = source_impedance
        zl = load_impedance
        denom = (b + c * zs * zl) + (a * zl + d * zs)

        return SMatrix([
            [
                ((b - c * zs * zl) + (a * zl - d * zs)) / denom,
                2 * zs * self.determinant / denom,
            ],
            [
                2 * zl / denom,
                ((b - c * zs * zl) - (a * zl - d * zs)) / denom,
            ],
        ])

    @classmethod
    def from_smatrix(cls, s: "SMatrix", z0: npt.NDArray) -> Self:
        """Compute the network from scattering parameters."""
        denom = 1 / (2 * s.s21)
        xx = s.s21 * s.s12

        return cls(
            denom
            * np.array([
                [
                    (1 + s.s11) * (1 - s.s22) + xx,
                    z0 * ((1 + s.s11) * (1 + s.s22) - xx),
                ],
                [
                    (1 / z0) * ((1 - s.s11) * (1 - s.s22) - xx),
                    (1 - s.s11) * (1 + s.s22) + xx,
                ],
            ])
        )

    @classmethod
    def from_transmission_line(
        cls, line: ee.TransmissionLine, length: tp.LengthType
    ) -> Self:
        """Get a two-port network representation of a transmission line."""
        gl = (line.propagation_constant * length).to_value("")

        cgl = np.cosh(gl)
        sgl = np.sinh(gl)
        return cls(
            np.array([
                [cgl, line.characteristic_impedance * sgl],
                [(1 / line.characteristic_impedance) * sgl, cgl],
            ])
        )


@attrs.define(frozen=True, kw_only=False, slots=False)
class SMatrix:
    """A scattering matrix for a two-port network.

    This class eases some of the computations performed with S-parameters. Most of the
    methods are based on https://en.wikipedia.org/wiki/Scattering_parameters.

    See also https://en.wikipedia.org/wiki/Two-port_network#Interrelation_of_parameters
    """

    s: npt.NDArray[complex] = npfield(
        possible_ndims=(3,),
        dtype=complex,
        converter=np.atleast_3d,
    )

    @s.validator
    def _s_vld(self, att, val):
        if val.shape[:2] != (2, 2):
            raise ValueError("Scattering matrix must have shape (2, 2, Nfreq).")

    @property
    def nfreq(self):
        """The number of frequencies in the S-matrix (the last axis)."""
        return self.s.shape[-1]

    @classmethod
    def from_sparams(cls, s11, s12, s21=None, s22=None):
        """Create a SMatrix from S-parameters.

        Parameters
        ----------
        s11 : array_like
            S11 parameter.
        s12 : array_like
            S12 parameter.
        s21 : array_like, optional
            S21 parameter. If not provided, assumed to be equal to s12.
        s22 : array_like, optional
            S22 parameter. If not provided, assumed to be equal to s11.
        """
        if s21 is None:
            s21 = s12
        if s22 is None:
            s22 = s11

        return cls(np.array([[s11, s12], [s21, s22]]))

    @classmethod
    def from_transfer_matrix(cls, t: npt.NDArray[complex]):
        """Create an SMatrix from a transfer matrix.

        See https://en.wikipedia.org/wiki/Scattering_parameters#Scattering_transfer_parameters.
        """
        detT = t[0, 0] * t[1, 1] - t[0, 1] * t[1, 0]

        return cls(
            np.array([
                [t[0, 1] / t[1, 1], detT / t[1, 1]],
                [1 / t[1, 1], -t[1, 0] / t[1, 1]],
            ])
        )

    @cached_property
    def determinant(self):
        """The determinant of the SMatrix."""
        return self.s11 * self.s22 - self.s12 * self.s21

    def as_transfer_matrix(self) -> npt.NDArray:
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

        return SMatrix.from_transfer_matrix(t)

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
        return np.abs(self.reverse_gain)

    @property
    def s11(self):
        """The S11 coefficient of the network."""
        return self.s[0, 0]

    @property
    def s21(self):
        """The S21 coefficient of the network."""
        return self.s[1, 0]

    @property
    def s12(self):
        """The S12 coefficient of the network."""
        return self.s[0, 1]

    @property
    def s22(self):
        """The S22 coefficient of the network."""
        return self.s[1, 1]

    def voltage_standing_wave_ratio_in(self):
        """The Voltage Standing Wave Ratio (VSWR) of the network input."""
        return (1 + np.abs(self.s11)) / (1 - np.abs(self.s11))

    def voltage_standing_wave_ratio_out(self):
        """The Voltage Standing Wave Ratio (VSWR) of the network output."""
        return (1 + np.abs(self.s22)) / (1 - np.abs(self.s22))

    @classmethod
    def from_transmission_line(
        cls,
        line: ee.TransmissionLine,
        length: tp.LengthType | None,
        load_impedance: tp.ImpedanceType = 50 * units.Ohm,
    ):
        """Generate a new SMatrix from a given transmission line object.

        Parameters
        ----------
        line
            The transmission line object to convert to an SMatrix.
        length
            The length of the transmission line (only required if not set on the
            line itself).
        load_impedance
            The impedance of a load connected to the network.
        """
        Zo = line.characteristic_impedance
        Zp = load_impedance

        if length is None and line.length is None:
            raise ValueError("Either length or line.length must be provided.")
        if length is None:
            length = line.length

        γ = line.propagation_constant
        γl = (γ * length).to_value("")

        denom = (Zo**2 + Zp**2) * np.sinh(γl) + 2 * Zo * Zp * np.cosh(γl)

        s11 = s22 = (Zo**2 - Zp**2) * np.sinh(γl) / denom
        s12 = s21 = 2 * Zo * Zp / denom
        return cls(np.array([[s11, s12], [s21, s22]]))

    @classmethod
    def from_calkit_and_vna(cls, calkit: Calkit, standards: "StandardsReadings"):  # noqa: F821
        """Generate an SMatrix from a Calkit definition, and standards measurements.

        Parameters
        ----------
        calkit
            A Calkit object, representing the calkit used to measure the network.
        standards
            The measurements of the standard OSL performed with a VNA.
        """
        freq = standards.freq

        return get_sparams_from_osl(
            calkit.open.reflection_coefficient(freq),
            calkit.short.reflection_coefficient(freq),
            calkit.match.reflection_coefficient(freq),
            standards.open.s11,
            standards.short.s11,
            standards.match.s11,
        )


def gamma_de_embed(
    gamma_ref: np.typing.ArrayLike,
    smatrix: SMatrix,
) -> np.typing.ArrayLike:
    """Get the reflection coefficient of load attached to a two-port network.

    See Eq. 2 of Monsalve et al., 2016 or
    https://en.wikipedia.org/wiki/Scattering_parameters#S-parameters_in_amplifier_design

    Notes
    -----
    This function gives the intrinsic reflection coefficient of the load attached to a
    2-port network, given the reflection coefficient observed at the reference plane
    of the input port of the network::

        ____________________________o___o
            |             |         |Z|
        PORT 1 |   NETWORK   |  PORT 2 |Z| <LOAD, Gamma_L>
            |             |         |Z|
        _______|_____________|_________|o|
            ^
            |
            REF.
            PLANE

    Parameters
    ----------
    smatrix
        The S-matrix of the 2-port network.
    gamma_ref
        The reflection coefficient of the device
        under test (DUT) measured at the reference plane.

    Returns
    -------
    gamma
        The intrinsic reflection coefficient of the DUT (the Load).

    See Also
    --------
    gamma_embed
        The inverse function to this one.
    """
    return (gamma_ref - smatrix.s11) / (
        smatrix.s22 * (gamma_ref - smatrix.s11) + smatrix.s12 * smatrix.s21
    )


def gamma_embed(
    smatrix: SMatrix,
    gamma: np.typing.ArrayLike,
) -> np.typing.ArrayLike:
    """Obtain intrinsic reflection coefficient of a load attached to a 2-port network.

    See Eq. 2 of Monsalve et al., 2016 or
    https://en.wikipedia.org/wiki/Scattering_parameters#S-parameters_in_amplifier_design

    Notes
    -----
    This function gives the reflection coefficient observed at the reference plane
    of the input port of the 2-port network, given the intrinsic reflection coefficient
    of the DUT / load attached to the output of the 2-port network::


        ____________________________o___o
            |             |         |Z|
        PORT 1 |   NETWORK   |  PORT 2 |Z| <LOAD, Gamma_L>
            |             |         |Z|
        _______|_____________|_________|o|
            ^
            |
            REF.
            PLANE


    Parameters
    ----------
    smatrix
        The S-matrix of the two-port networok
    gamma
        The intrinsic reflection coefficient of the device
        under test (DUT/Load)

    Returns
    -------
    gamma_ref
         The reflection coefficient of the DUT measured at the reference plane.

    See Also
    --------
    gamma_de_embed
        The inverse function to this one.
    """
    return smatrix.s11 + (smatrix.s12 * smatrix.s21 * gamma / (1 - smatrix.s22 * gamma))


def get_sparams_from_osl(
    gamma_open_intr: np.ndarray | float,
    gamma_short_intr: np.ndarray | float,
    gamma_match_intr: np.ndarray | float,
    gamma_open_meas: np.ndarray,
    gamma_short_meas: np.ndarray,
    gamma_match_meas: np.ndarray,
) -> SMatrix:
    """Obtain network S-parameters from OSL standards and intrinsic reflections of DUT.

    See Eq. 3 of Monsalve et al., 2016.

    Parameters
    ----------
    gamma_open_intr
        The intrinsic reflection of the open standard
        (assumed as true) as a function of frequency.
    gamma_shrt_intr
        The intrinsic reflection of the short standard
        (assumed as true) as a function of frequency.
    gamma_load_intr
        The intrinsic reflection of the load standard
        (assumed as true) as a function of frequency.
    gamma_open_meas
        The reflection of the open standard
        measured at port 1 as a function of frequency.
    gamma_shrt_meas
        The reflection of the short standard
        measured at port 1 as a function of frequency.
    gamma_load_meas
        The reflection of the load standard
        measured at port 1 as a function of frequency.

    Returns
    -------
    s11
        The S11 of the network.
    s12s21
        The product `S12*S21` of the network
    s22
        The S22 of the network.
    """
    gamma_open_intr = gamma_open_intr * np.ones_like(gamma_open_meas)
    gamma_short_intr = gamma_short_intr * np.ones_like(gamma_open_meas)
    gamma_match_intr = gamma_match_intr * np.ones_like(gamma_open_meas)

    s11 = np.zeros(len(gamma_open_intr)) + 0j  # 0j added to make array complex
    s12s21 = np.zeros(len(gamma_open_intr)) + 0j
    s22 = np.zeros(len(gamma_open_intr)) + 0j

    for i in range(len(gamma_open_intr)):
        b = np.array([gamma_open_meas[i], gamma_short_meas[i], gamma_match_meas[i]])
        A = np.array([
            [
                1,
                complex(gamma_open_intr[i]),
                complex(gamma_open_intr[i] * gamma_open_meas[i]),
            ],
            [
                1,
                complex(gamma_short_intr[i]),
                complex(gamma_short_intr[i] * gamma_short_meas[i]),
            ],
            [
                1,
                complex(gamma_match_intr[i]),
                complex(gamma_match_intr[i] * gamma_match_meas[i]),
            ],
        ])
        x = np.linalg.lstsq(A, b, rcond=None)[0]

        s11[i] = x[0]
        s12s21[i] = x[1] + x[0] * x[2]
        s22[i] = x[2]

    s12 = np.sqrt(s12s21)
    return SMatrix(np.array([[s11, s12], [s12, s22]]))


def path_length_correction_edges3(
    freq: tp.FreqType, delay: tp.TimeType, gamma_in: float, lossf: float, dielf: float
) -> tuple[float, float, float]:
    """
    Calculate the path length correction for the EDGES-3 LNA.

    Notes
    -----
    The 8-position switch memo is 303 and the correction for the path to the
    LNA for the calibration of the LNA s11 is described in memos 367 and 392.

    corrcsv.c corrects lna s11 file for the different vna path to lna args:
    s11.csv -cablen -cabdiel -cabloss outputs c_s11.csv

    The actual numbers are slightly temperature dependent

    corrcsv s11.csv -cablen 4.26 -cabdiel -1.24 -cabloss -91.5

    and need to be determined using a calibration test like that described in
    memos 369 and 361. Basically the path length corrections can be "tuned" by
    minimizing the ripple on the calibrated spectrum of the open or shorted
    cable.

    cablen --> length in inches
    cabloss --> loss correction percentage
    cabdiel --> dielectric correction in percentage

    """
    freq = freq.to("Hz").value
    length = (delay * speed_of_light).to_value("m")

    b = 0.1175 * 2.54e-2 * 0.5
    a = 0.0362 * 2.54e-2 * 0.5
    diel = 2.05 * dielf  # UT-141C-SP
    # for tinned copper
    d2 = np.sqrt(1.0 / (np.pi * 4.0 * np.pi * 1e-7 * 5.96e07 * 0.8 * lossf))
    # skin depth at 1 Hz for copper
    d = np.sqrt(1.0 / (np.pi * 4.0 * np.pi * 1e-7 * 5.96e07 * lossf))

    L = (4.0 * np.pi * 1e-7 / (2.0 * np.pi)) * np.log(b / a)
    C = 2.0 * np.pi * 8.854e-12 * diel / np.log(b / a)

    La = 4.0 * np.pi * 1e-7 * d / (4.0 * np.pi * a)
    Lb = 4.0 * np.pi * 1e-7 * d2 / (4.0 * np.pi * b)
    disp = (La + Lb) / L
    R = 2.0 * np.pi * L * disp * np.sqrt(freq)
    L = L * (1.0 + disp / np.sqrt(freq))
    G = 2.0 * np.pi * C * freq * 2e-4 if diel > 1.2 else 0
    Zcab = np.sqrt((1j * 2 * np.pi * freq * L + R) / (1j * 2 * np.pi * freq * C + G))
    g = np.sqrt((1j * 2 * np.pi * freq * L + R) * (1j * 2 * np.pi * freq * C + G))

    T = (50.0 - Zcab) / (50.0 + Zcab)
    Vin = np.exp(+g * length) + T * np.exp(-g * length)
    Iin = (np.exp(+g * length) - T * np.exp(-g * length)) / Zcab
    Vout = 1 + T  # Iout = (1 - T)/Zcab
    s11 = ((Vin / Iin) - 50) / ((Vin / Iin) + 50)  # same as s22
    VVin = Vin + 50.0 * Iin
    s12 = 2 * Vout / VVin  # same as s21

    Z = 50.0 * (1 + gamma_in) / (1 - gamma_in)
    T = (Z - Zcab) / (Z + Zcab)
    T = T * np.exp(-g * 2 * length)
    Z = Zcab * (1 + T) / (1 - T)
    T = (Z - 50.0) / (Z + 50.0)

    return T, s11, s12


def rephase(delay: float, freq: np.ndarray, s11: np.ndarray):
    """Rephase an S11 with a given delay."""
    return s11 * np.exp(2 * np.pi * freq * delay * 1j)


def get_rough_delay(freq: np.ndarray, s11: np.ndarray):
    """Calculate the delay of an S11 using FFT."""
    power = np.abs(np.fft.fft(s11 * blackmanharris(len(s11)))) ** 2
    kk = np.fft.fftfreq(len(s11), d=freq[1] - freq[0])

    return -kk[np.argmax(power)]


def get_delay(
    freq: tp.FreqType, s11: np.ndarray, optimize: bool = False
) -> units.Quantity[units.microsecond]:
    """Find the delay of an S11 using a minimization routine."""
    freq = freq.to_value("MHz")  # resulting delay in microsecond

    def _objfun(delay, freq: np.ndarray, s11: np.ndarray):
        reph = rephase(delay, freq, s11)
        return -np.abs(np.sum(reph))

    if optimize:
        start = -get_rough_delay(freq, s11)
        dk = 1 / (freq[1] - freq[0])
        res = minimize(
            _objfun, x0=(start,), bounds=((start - dk, start + dk),), args=(freq, s11)
        )
        return res.x * units.microsecond

    delays = np.arange(-1e-3, 0.1, 1e-4)
    obj = [_objfun(d, freq, s11) for d in delays]
    return delays[np.argmin(obj)] * units.microsecond
