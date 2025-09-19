"""Electrical enginerring equations."""

from functools import cached_property

import attrs
import numpy as np
from astropy import constants as cnst
from astropy import units as un
from astropy.constants import eps0, mu0
from pygsdata.attrs import unit_validator as unv

from .. import types as tp


def skin_depth(freq: tp.FreqType, conductivity: tp.Conductivity) -> un.Quantity[un.m]:
    """Calculate the skin depth of a conducting material."""
    return np.sqrt(1.0 / (np.pi * freq * cnst.mu0 * conductivity)).to("m")


@attrs.define(frozen=True, slots=False)
class TransmissionLine:
    """A transmission line."""

    freq: tp.FreqType = attrs.field(validator=unv(un.Hz))
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
        return 2 * np.pi * 1j * self.freq

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
    ):
        """Calculate the S11 parameter of a transmission line.

        This is the reflection coefficient of the transmission line in the case
        of matched loads at each termination.

        https://en.wikipedia.org/wiki/Transmission_line#Scattering_parameters
        """
        if line_length is None:
            line_length = self.length

        if line_length is None:
            raise ValueError("Line length must be provided or set on the instance.")

        from .reflection_coefficient import SMatrix

        return SMatrix.from_transmission_line(
            self, length=line_length, load_impedance=load_impedance
        )


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
        self, freq: tp.FreqType, length: tp.LengthType | None = None
    ) -> TransmissionLine:
        """Return a TransmissionLine object for the cable."""
        return TransmissionLine(
            freq=freq,
            resistance=self.resistance_per_metre(freq),
            inductance=self.spectral_inductance_per_metre(freq),
            conductance=self.conductance_per_metre(freq),
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
    ) -> tp.DimlessType:
        """Get the propagation constant of the cable at a given frequency."""
        return self.as_transmission_line(freq, length).propagation_constant

    def scattering_parameters(
        self,
        freq: tp.FreqType,
        length: tp.LengthType | None = None,
    ):
        """Get the scattering matrix of the cable at a given frequency."""
        return self.as_transmission_line(freq, length).scattering_parameters()


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
