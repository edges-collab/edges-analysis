"""Functions for calibrating S-parameter measurements.

Functions for de-embedding and embedding 2-port networks,
as well as generating S-parameters from calkit measurements.
"""

from collections.abc import Sequence

import numpy as np

from .datatypes import CalkitReadings, ReflectionCoefficient, SParams


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


def gamma_de_embed(
    gamma: ReflectionCoefficient,
    sparams: SParams,
) -> ReflectionCoefficient:
    """Remove the effect of a 2-port network from a reflection coefficient.

    See Eq. 2 of Monsalve et al., 2016 or
    https://en.wikipedia.org/wiki/Scattering_parameters#S-parameters_in_amplifier_design

    Notes
    -----
    Given the reflection coefficient observed at a reference plane on one side of
    an electrical component/subsystem, this function returns the reflection coefficient
    at the reference plane on the other side of the subsystem::

       ---         ------------
      |VNA| ---|---| SUBSYTEM |---|---
       ---         ------------
               ^                  ^
               |                  |
           MEAS. REF.          DESIRED REF.
             PLANE               PLANE

    Parameters
    ----------
    gamma
        The reflection coefficient measured at the reference plane "in front"
        of the 2-port network / subsystem. The shape should be (N,), where N is the
        number of frequency points.
    sparams
        The S-matrix of the 2-port network / subsystem.
        The shape should be (2, 2, N), where N is the number of frequency points.

    Returns
    -------
    gamma_de_embedded
        The reflection coefficient at the desired reference plane, on the other
        side of the 2-port network. The shape is (N,), where N is the number of
        frequency points.

    See Also
    --------
    gamma_embed
        The inverse function to this one.
    """
    gamma_in = gamma.reflection_coefficient
    return ReflectionCoefficient(
        freqs=gamma.freqs,
        reflection_coefficient=(gamma_in - sparams.s11)
        / (sparams.s22 * (gamma_in - sparams.s11) + sparams.s12 * sparams.s21),
    )


def gamma_embed(
    gamma: ReflectionCoefficient,
    sparams: SParams,
) -> ReflectionCoefficient:
    """Add the effect of a 2-port network to a reflection coefficient.

    See notes for :func:`gamma_de_embed`. This is the inverse function to that one.

    Parameters
    ----------
    sparams
        The S-matrix of the two-port networok. Shape should be (2, 2, N), where N is the
        number of frequency points.
    gamma
        The reflection coefficient at the referance plan on one side
        of the 2-port network. Shape should be (N,), where N is the number of
        frequency points.

    Returns
    -------
    gamma_ref
         The reflection coefficient at the reference plane on the other side
         of the 2-port network. Shape is (N,), where N is the number of frequency
         points.

    See Also
    --------
    gamma_de_embed
        The inverse function to this one.
    """
    gamma_in = gamma.reflection_coefficient
    return ReflectionCoefficient(
        freqs=gamma.freqs,
        reflection_coefficient=(
            sparams.s11
            + (sparams.s12 * sparams.s21 * gamma_in) / (1 - sparams.s22 * gamma_in)
        ),
    )


def sparams_from_calkit_measurements(
    measurements: CalkitReadings,
    model: CalkitReadings | None = None,
) -> SParams:
    """Compute S-parameters of a 2-port network from calkit measurements.

    This uses Eq. 3 of Monsalve et al., 2016.

    Parameters
    ----------
    measurements
        The actual measurements of the calkit standards.
    model
        A model of the calkit standards. If None, ideal standards are assumed.
    """
    from .network_component_models import Calkit

    freq = measurements.freqs

    if isinstance(model, Calkit):
        model = model.at_freqs(freq)
    elif model is None:
        model = CalkitReadings.ideal(freqs=freq)

    n = len(freq)

    s11 = np.zeros(n, dtype=complex)
    s12s21 = np.zeros(n, dtype=complex)
    s22 = np.zeros(n, dtype=complex)

    for i in range(n):
        om, sm, mm = (
            model.open.reflection_coefficient[i],
            model.short.reflection_coefficient[i],
            model.match.reflection_coefficient[i],
        )

        b = np.array([
            measurements.open.reflection_coefficient[i],
            measurements.short.reflection_coefficient[i],
            measurements.match.reflection_coefficient[i],
        ])

        A = np.array([
            [1, om, om * b[0]],
            [1, sm, sm * b[1]],
            [1, mm, mm * b[2]],
        ])
        x = np.linalg.lstsq(A, b, rcond=None)[0]

        s11[i] = x[0]
        s12s21[i] = x[1] + x[0] * x[2]
        s22[i] = x[2]

    s12 = np.sqrt(s12s21)
    return SParams(freqs=freq, s11=s11, s12=s12, s21=s12, s22=s22)


def de_embed_network_from_calkit_measurements(
    measurements: CalkitReadings, sparams: SParams
) -> CalkitReadings:
    """Compute the S-parameters of a 2-port network from calkit measurements.

    This is a convenience wrapper around :func:`sparams_from_calkit_measurements`.

    Parameters
    ----------
    measurements
        The actual measurements of the calkit standards.
    model
        A model of the calkit standards. If None, ideal standards are assumed.
    """
    return CalkitReadings(**{
        kind: getattr(measurements, kind).de_embed(sparams)
        for kind in ("open", "short", "match")
    })


def average_reflection_coefficients(
    s: Sequence[ReflectionCoefficient],
) -> ReflectionCoefficient:
    """Average multiple reflection coefficients."""
    return ReflectionCoefficient(
        freqs=s[0].freqs,
        reflection_coefficient=np.mean([ss.reflection_coefficient for ss in s], axis=0),
    )


def average_sparams(s: Sequence[SParams]) -> SParams:
    """Average multiple reflection coefficients."""
    return SParams(
        freqs=s[0].freqs,
        s11=np.mean([ss.s11 for ss in s], axis=0),
        s12=np.mean([ss.s12 for ss in s], axis=0),
        s21=np.mean([ss.s21 for ss in s], axis=0),
        s22=np.mean([ss.s22 for ss in s], axis=0),
    )


# Patch some of these functions onto the class definitions, for convenience.
ReflectionCoefficient.de_embed = gamma_de_embed
ReflectionCoefficient.embed = gamma_embed
SParams.from_calkit_measurements = staticmethod(sparams_from_calkit_measurements)
CalkitReadings.de_embed = de_embed_network_from_calkit_measurements
