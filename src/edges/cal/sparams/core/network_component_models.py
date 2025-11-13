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

from edges.units import unit_converter

try:
    # only available on py311+
    from typing import Self
except ImportError:
    from typing import Self

import attrs
import numpy as np
import numpy.typing as npt
from astropy import units as un
from astropy.constants import eps0, mu0
from pygsdata.attrs import npfield
from pygsdata.attrs import unit_validator as unv

from edges import modeling as mdl
from edges import types as tp
from edges.io.serialization import hickleable

from . import sparam_calibration as ee
from .datatypes import CalkitReadings, ReflectionCoefficient, SParams


def skin_depth(freq: tp.FreqType, conductivity: tp.Conductivity) -> un.Quantity[un.m]:
    """Calculate the skin depth of a conducting material."""
    return np.sqrt(1.0 / (np.pi * freq * mu0 * conductivity)).to("m")


@attrs.define(frozen=True, slots=False)
class TransmissionLine:
    """A transmission line."""

    freqs: tp.FreqType = attrs.field(validator=unv(un.Hz))
    resistance = attrs.field(validator=unv(un.ohm / un.m))
    inductance = attrs.field(validator=unv(un.ohm * un.s / un.m))
    conductance = attrs.field(validator=unv(un.siemens / un.m))
    capacitance = attrs.field(validator=unv(un.siemens * un.s / un.m))
    length: tp.LengthType = attrs.field(
        validator=attrs.validators.optional(unv(un.m)), default=None
    )

    @cached_property
    def angular_freq(self) -> tp.FreqType:
        """The angular frequencies at which to evaluate the transmission line."""
        return 2 * np.pi * 1j * self.freqs

    @cached_property
    def characteristic_impedance(self) -> tp.ImpedanceType:
        r"""Calculate the characteristic impedance of a transmission line.

        The characteristic impedance Z 0 {\displaystyle Z_{0}} of a transmission line
        is the ratio of the amplitude of a single voltage wave to its current wave.

        https://en.wikipedia.org/wiki/Transmission_line
        """
        return np.sqrt(
            (self.resistance + self.angular_freq * self.inductance)
            / (self.conductance + self.angular_freq * self.capacitance)
        ).to("ohm")

    @cached_property
    def propagation_constant(self) -> un.Quantity[1 / un.m]:
        """Calculate the propagation constant of a transmission line.

        https://en.wikipedia.org/wiki/Transmission_line#General_case_of_a_line_with_losses
        """
        return np.sqrt(
            (self.resistance + self.angular_freq * self.inductance)
            * (self.conductance + self.angular_freq * self.capacitance)
        ).to("1/m")

    def input_impedance(
        self,
        load_impedance: tp.ImpedanceType = 50 * un.Ohm,
        line_length: tp.LengthType | None = None,
    ):
        """Calculate the "input impedance" of a transmission line.

        https://en.wikipedia.org/wiki/Transmission_line#Input_impedance_of_transmission_line

        Parameters
        ----------
        freq : tp.FreqType
            Frequency of the signal.
        """
        if line_length is None:
            line_length = self.length

        if line_length is None:
            raise ValueError("Line length must be provided or set on the instance.")

        return (
            self.characteristic_impedance
            * (
                load_impedance
                + self.characteristic_impedance
                * np.tanh(self.propagation_constant * line_length)
            )
            / (
                self.characteristic_impedance
                + load_impedance * np.tanh(self.propagation_constant * line_length)
            )
        )

    def reflection_coefficient(
        self,
        load_impedance: tp.ImpedanceType = 50 * un.Ohm,
    ):
        """Calculate the reflection coefficient of a transmission line.

        This is the reflections coefficient measured at the load end of a transmission
        line.

        https://en.wikipedia.org/wiki/Transmission_line
          #Input_impedance_of_transmission_line
        """
        return (load_impedance - self.characteristic_impedance) / (
            load_impedance + self.characteristic_impedance
        )

    def scattering_parameters(
        self,
        load_impedance: tp.ImpedanceType = 50 * un.Ohm,
        line_length: tp.LengthType | None = None,
    ) -> SParams:
        """Calculate the S11 parameter of a transmission line.

        This is the reflection coefficient of the transmission line in the case
        of matched loads at each termination.

        https://en.wikipedia.org/wiki/Transmission_line#Scattering_parameters
        """
        if line_length is None:
            line_length = self.length

        if line_length is None:
            raise ValueError("Line length must be provided or set on the instance.")

        Zo = self.characteristic_impedance
        Zp = load_impedance

        γ = self.propagation_constant
        γl = (γ * line_length).to_value("")

        denom = (Zo**2 + Zp**2) * np.sinh(γl) + 2 * Zo * Zp * np.cosh(γl)

        s11 = (Zo**2 - Zp**2) * np.sinh(γl) / denom
        s12 = 2 * Zo * Zp / denom
        return SParams(freqs=self.freqs, s11=s11, s12=s12)


@attrs.define(kw_only=True, frozen=True, slots=False)
class CoaxialCable:
    """Properties of a coaxial cable.

    These properties are those used in the cabl2 function in edges.c.

    Parameters
    ----------
    outer_radius`
        The outer diameter of the cable. Equivalent to b in cabl2.
    inner_radius
        The inner diameter of the cable. Equivalent to a in cabl2.
    dielectric
        The dielectric constant of the cable. Equivalent to diel in cabl2.
    outer_material
        The material that forms the outer conductor of the cable.
    inner_material
        The material that forms the inner conductor of the cable.
    outer_conductivity
        The conductivity of the outer conductor. Used to get the skin depth. Only
        required if the material is not in the known materials.
    inner_conductivity
        The conductivity of the inner conductor. Used to get the skin depth. Only
        required if the material is not in the known materials.
    """

    # These conductivities are taken from Alan's code in cabl2
    conductivities: dict[str, tp.Conductivity] = {  # noqa: RUF008
        "copper": 5.96e07 * un.siemens / un.m,
        "brass": 5.96e07 * 0.29 * un.siemens / un.m,
        "stainless steel": 5.96e07 * 0.024 * un.siemens / un.m,
        "tinned copper": 5.96e07 * 0.8 * un.siemens / un.m,
        "silver plated copper": 5.96e07 * un.siemens / un.m,
    }

    outer_radius: tp.LengthType = attrs.field(
        validator=[unv(un.m), attrs.validators.gt(0)]
    )
    inner_radius: tp.LengthType = attrs.field(
        validator=[unv(un.m), attrs.validators.gt(0)]
    )
    outer_material: str = attrs.field(converter=str)
    inner_material: str = attrs.field(converter=str)
    relative_dielectric: float = attrs.field(
        converter=float, validator=attrs.validators.gt(0)
    )

    outer_conductivity: tp.Conductivity = attrs.field(
        validator=[unv(un.siemens / un.m), attrs.validators.gt(0)]
    )
    inner_conductivity: tp.Conductivity = attrs.field(
        validator=[unv(un.siemens / un.m), attrs.validators.gt(0)]
    )
    relative_conductance_interior: float = attrs.field(
        validator=attrs.validators.ge(0), converter=float, default=2e-4
    )
    length: tp.LengthType = attrs.field(
        validator=attrs.validators.optional([unv(un.m), attrs.validators.gt(0)]),
        default=None,
    )

    # We add _eps0 here so that we can set it equivalently to Alan's code, i.e.
    # to 8.854e-12 (which is an error of ~0.01%)
    _eps0: un.Quantity[un.F / un.m] = attrs.field(
        validator=[unv(un.F / un.m), attrs.validators.gt(0)], default=eps0
    )

    @outer_conductivity.default
    def _default_outer_conductivity(self):
        try:
            return self.conductivities[self.outer_material]
        except KeyError as e:
            raise ValueError(
                f"Unknown material: {self.outer_material}. Either choose from "
                f"{self.conductivities.keys()} or specify outer_condutivity directly."
            ) from e

    @inner_conductivity.default
    def _default_inner_conductivity(self):
        try:
            return self.conductivities[self.inner_material]
        except KeyError as e:
            raise ValueError(
                f"Unknown material: {self.inner_material}. Either choose from "
                f"{self.conductivities.keys()} or specify inner_condutivity directly."
            ) from e

    def outer_skin_depth(self, freq: tp.FreqType) -> tp.LengthType:
        """Get the skin depth of the outer material at a given frequency.

        See https://en.wikipedia.org/wiki/Skin_effect#Examples
        """
        return skin_depth(freq, self.outer_conductivity)

    def inner_skin_depth(self, freq: tp.FreqType) -> tp.LengthType:
        """Get the skin depth of the inner material at a given frequency.

        See https://en.wikipedia.org/wiki/Skin_effect
        """
        return skin_depth(freq, self.inner_conductivity)

    @property
    def inductance_per_metre(self) -> tp.InductanceType:
        """Get the inductance per metre of the cable.

        See https://en.wikipedia.org/wiki/Inductance#Inductance_of_a_coaxial_cable.

        This is equivalent to Alan's "L" in cabl2.
        """
        return ((mu0 / (2 * np.pi)) * np.log(self.outer_radius / self.inner_radius)).to(
            "H/m"
        )

    @property
    def capacitance_per_metre(self) -> tp.Conductivity:
        """Get the capacitance per metre of the cable.

        See https://en.wikipedia.org/wiki/Coaxial_cable#Physical_parameters
        """
        return (
            (2 * np.pi * self._eps0 * self.relative_dielectric)
            / np.log(self.outer_radius / self.inner_radius)
        ).to("F/m")

    def disp(self, freq: tp.FreqType):
        """TODO: what the hell is this."""
        a = mu0 * self.inner_skin_depth(freq) / (4 * np.pi * self.inner_radius)
        b = mu0 * self.outer_skin_depth(freq) / (4 * np.pi * self.outer_radius)
        return (a + b) / self.inductance_per_metre

    def resistance_per_metre(self, freq: tp.FreqType) -> un.Quantity[un.ohm / un.m]:
        """Get the resistance per metre of the cable."""
        return (2 * np.pi * freq * self.inductance_per_metre * self.disp(freq)).to(
            un.ohm / un.m
        )

    def spectral_inductance_per_metre(self, freq: tp.FreqType) -> tp.InductanceType:
        """Get the spectral inductance per metre of the cable."""
        return self.inductance_per_metre * (1 + self.disp(freq))

    def conductance_per_metre(self, freq: tp.FreqType) -> un.Quantity[un.m / un.ohm]:
        """Get the conductance per metre of the cable."""
        return (
            2
            * np.pi
            * self.capacitance_per_metre
            * freq
            * self.relative_conductance_interior
        )  # todo: why the 2e-4?

    def as_transmission_line(
        self, freqs: tp.FreqType, length: tp.LengthType | None = None
    ) -> TransmissionLine:
        """Return a TransmissionLine object for the cable."""
        return TransmissionLine(
            freqs=freqs,
            resistance=self.resistance_per_metre(freqs),
            inductance=self.spectral_inductance_per_metre(freqs),
            conductance=self.conductance_per_metre(freqs),
            capacitance=self.capacitance_per_metre,
            length=length or self.length,
        )

    def characteristic_impedance(
        self, freq: tp.FreqType, length: tp.LengthType | None = None
    ) -> tp.OhmType:
        """Get the characteristic impedance of the cable at a given frequency.

        See https://en.wikipedia.org/wiki/Coaxial_cable#Derived_electrical_parameters
        """
        return self.as_transmission_line(freq, length).characteristic_impedance

    def propagation_constant(
        self, freq: tp.FreqType, length: tp.LengthType | None = None
    ) -> un.Quantity[1 / un.m]:
        """Get the propagation constant of the cable at a given frequency."""
        return self.as_transmission_line(freq, length).propagation_constant

    def scattering_parameters(
        self,
        freqs: tp.FreqType,
        length: tp.LengthType | None = None,
    ):
        """Get the scattering matrix of the cable at a given frequency."""
        return self.as_transmission_line(freqs, length).scattering_parameters()


KNOWN_CABLES = {
    "balun-tube": CoaxialCable(
        outer_radius=0.37 / 2 * un.imperial.inch,
        inner_radius=5 / 64 * un.imperial.inch,
        outer_material="brass",
        inner_material="copper",
        relative_dielectric=1.07,
    ),
    "lowband-balun-tube": CoaxialCable(
        outer_radius=0.75 / 2 * un.imperial.inch,
        inner_radius=5 / 32 * un.imperial.inch,
        outer_material="brass",
        inner_material="copper",
        relative_dielectric=1.07,
        relative_conductance_interior=0,
        length=43.6 * un.imperial.inch,
    ),
    "midband-balun-tube": CoaxialCable(
        outer_radius=1.25 / 2 * un.imperial.inch,
        inner_radius=1 / 4 * un.imperial.inch,
        outer_material="brass",
        inner_material="copper",
        relative_dielectric=1.2,
    ),
    "SC3792 Connector": CoaxialCable(
        outer_radius=0.161 / 2 * un.imperial.inch,
        inner_radius=0.05 / 2 * un.imperial.inch,
        outer_material="stainless steel",
        inner_material="copper",
        inner_conductivity=5.96e07 * 0.24 * un.siemens / un.m,
        relative_dielectric=2.05,
        length=3e-2 * un.m,
    ),
    "SMA Connector": CoaxialCable(
        outer_radius=0.16 / 2 * un.imperial.inch,
        inner_radius=0.05 / 2 * un.imperial.inch,
        outer_material="stainless steel",
        inner_material="copper",
        inner_conductivity=5.96e07 * 0.20 * un.siemens / un.m,
        relative_dielectric=2.05,
        length=2e-2 * un.m,
    ),
    "UT-141C-SP": CoaxialCable(
        outer_radius=0.1175 * un.imperial.inch / 2,
        inner_radius=0.0362 * un.imperial.inch / 2,
        outer_material="tinned copper",
        inner_material="copper",
        relative_dielectric=2.05,
        length=4 * un.imperial.inch,
    ),
    "UT-086C-SP": CoaxialCable(
        outer_radius=1.57e-3 * un.m / 2,
        inner_radius=0.51e-3 * un.m / 2,
        outer_material="tinned copper",
        inner_material="copper",
        relative_dielectric=2.05,
    ),
    "Molex WM10479": CoaxialCable(
        outer_radius=0.1175 * un.imperial.inch / 2,
        inner_radius=0.0453 * un.imperial.inch / 2,
        outer_material="silver plated copper",
        inner_material="silver plated copper",
        relative_dielectric=1.32,
    ),
}


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

    resistance: float | tp.ImpedanceType = attrs.field(converter=unit_converter(un.ohm))
    offset_impedance: float | tp.ImpedanceType = attrs.field(
        default=50.0 * un.ohm, converter=unit_converter(un.ohm)
    )
    offset_delay: float | tp.TimeType = attrs.field(
        default=30.0 * un.picosecond, converter=unit_converter(un.picosecond)
    )
    offset_loss: float | un.Quantity[un.Gohm / un.s] = attrs.field(
        default=2.2 * un.Gohm / un.s,
        converter=unit_converter(un.Gohm / un.s),
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
    def _verify_freq(cls, freq: np.ndarray | un.Quantity):
        if un.get_physical_type(freq) != "frequency":
            raise TypeError(
                f"freq must be a frequency quantity! Got {un.get_physical_type(freq)}"
            )

    @property
    def intrinsic_gamma(self) -> float:
        """The intrinsic reflection coefficient of the idealized standard."""
        if np.isinf(self.resistance):
            return 1.0  # np.inf / np.inf
        return ee.impedance2gamma(self.resistance, 50.0 * un.Ohm)

    def termination_impedance(self, freq: tp.FreqType) -> tp.OhmType:
        """The impedance of the termination of the standard.

        See Eq. 22-25 of M16 for open and short standards. The match standard
        uses the input measured resistance as the impedance.
        """
        self._verify_freq(freq)
        freq = freq.to("Hz").value

        if self.capacitance_model is not None:
            return (-1j / (2 * np.pi * freq * self.capacitance_model(freq))) * un.ohm
        if self.inductance_model is not None:
            return 1j * 2 * np.pi * freq * self.inductance_model(freq) * un.ohm
        return self.resistance

    def termination_gamma(self, freq: tp.FreqType) -> tp.DimlessType:
        """Reflection coefficient of the termination.

        Eq. 19 of M16.
        """
        return ee.impedance2gamma(self.termination_impedance(freq), 50 * un.ohm)

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
        return ee.impedance2gamma(
            self.lossy_characteristic_impedance(freq), 50 * un.ohm
        )

    def reflection_coefficient(self, freqs: tp.FreqType) -> ReflectionCoefficient:
        """Obtain the combined reflection coefficient of the standard.

        See Eq. 18 of M16.

        Note that, despite looking different to Alan's implementation, this is exactly
        the same as his agilent() function EXCEPT that he doesn't seem to use the
        loss / capacitance models.
        """
        ex = np.exp(-2 * self.gl(freqs))
        r1 = self.offset_gamma(freqs)
        gamma_termination = self.termination_gamma(freqs)
        return ReflectionCoefficient(
            freqs=freqs,
            reflection_coefficient=(
                r1 * (1 - ex - r1 * gamma_termination) + ex * gamma_termination
            )
            / (1 - r1 * (ex * r1 + gamma_termination * (1 - ex))).value,
        )

    @classmethod
    def open(cls, resistance=np.inf * un.ohm, **kwargs) -> Self:
        """Create an 'open' calkit standard, with resistance=inf.

        See :class:`CalkitStandard` for all parameters available.
        """
        return cls(resistance=resistance, **kwargs)

    @classmethod
    def short(cls, resistance=0 * un.ohm, **kwargs) -> Self:
        """Create a 'short' calkit standard, with resistance=0.

        See :class:`CalkitStandard` for all parameters available.
        """
        return cls(resistance=resistance, **kwargs)

    @classmethod
    def match(cls, resistance=50.0 * un.ohm, **kwargs) -> Self:
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

    def clone(self, *, short=None, open=None, match=None):  # noqa: A002
        """Return a clone with updated parameters for each standard."""
        return attrs.evolve(
            self,
            open=attrs.evolve(self.open, **(open or {})),
            short=attrs.evolve(self.short, **(short or {})),
            match=attrs.evolve(self.match, **(match or {})),
        )

    def at_freqs(self, freqs: tp.FreqType) -> CalkitReadings:
        """Get the reflection coefficients of each standard at given frequencies."""
        return CalkitReadings(
            open=self.open.reflection_coefficient(freqs),
            short=self.short.reflection_coefficient(freqs),
            match=self.match.reflection_coefficient(freqs),
        )


AGILENT_85033E = Calkit(
    open=CalkitStandard.open(
        offset_impedance=50.0 * un.ohm,
        offset_delay=29.243 * un.picosecond,
        offset_loss=2.2 * un.Gohm / un.s,
        capacitance_model=mdl.Polynomial(
            parameters=[49.43e-15, -310.1e-27, 23.17e-36, -0.1597e-45]
        ),
    ),
    short=CalkitStandard.short(
        offset_impedance=50.0 * un.ohm,
        offset_delay=31.785 * un.picosecond,
        offset_loss=2.36 * un.Gohm / un.s,
        inductance_model=mdl.Polynomial(
            parameters=[2.077e-12, -108.5e-24, 2.171e-33, -0.01e-42]
        ),
    ),
    match=CalkitStandard.match(
        offset_impedance=50.0 * un.ohm,
        offset_delay=38.0 * un.picosecond,
        offset_loss=2.3 * un.Gohm / un.s,
    ),
)

AGILENT_ALAN = Calkit(
    open=CalkitStandard.open(
        offset_impedance=50.0 * un.ohm,
        offset_delay=33 * un.picosecond,
        offset_loss=2.3 * un.Gohm / un.s,
        resistance=1e9 * un.Ohm,
    ),
    short=CalkitStandard.short(
        offset_impedance=50.0 * un.ohm,
        offset_delay=33 * un.picosecond,
        offset_loss=2.3 * un.Gohm / un.s,
        resistance=0 * un.Ohm,
    ),
    match=CalkitStandard.match(
        offset_impedance=50.0 * un.ohm,
        offset_delay=33.0 * un.picosecond,
        offset_loss=2.3 * un.Gohm / un.s,
    ),
)

KNOWN_CALKITS = {"AGILENT_85033E": AGILENT_85033E, "AGILENT_ALAN": AGILENT_ALAN}


def get_calkit(
    base: Calkit | str,
    resistance_of_match: tp.ImpedanceType | None = None,
    open: dict | None = None,  # noqa: A002
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
    if isinstance(base, str):
        base = KNOWN_CALKITS[base]

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

    def as_sparams(
        self,
        freqs: tp.FreqType,
        source_impedance: float,
        load_impedance: float | None = None,
    ) -> SParams:
        """Convert the TwoPortNetwork to an SParams instance."""
        if load_impedance is None:
            load_impedance = source_impedance

        a, b, c, d = self.A, self.B, self.C, self.D
        zs = source_impedance
        zl = load_impedance
        denom = (b + c * zs * zl) + (a * zl + d * zs)

        return SParams(
            freqs=freqs,
            s11=((b - c * zs * zl) + (a * zl - d * zs)) / denom,
            s12=2 * zs * self.determinant / denom,
            s21=2 * zl / denom,
            s22=((b - c * zs * zl) - (a * zl - d * zs)) / denom,
        )

    @classmethod
    def from_smatrix(cls, s: SParams, z0: npt.NDArray) -> Self:
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
        cls, line: TransmissionLine, length: tp.LengthType
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
