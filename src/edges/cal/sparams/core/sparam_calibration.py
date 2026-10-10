"""Functions for calibrating S-parameter measurements.

Functions for de-embedding and embedding 2-port networks,
as well as generating S-parameters from calkit measurements.
"""

from collections.abc import Sequence
from typing import TYPE_CHECKING

import numpy as np

from .datatypes import CalkitReadings, ReflectionCoefficient, SParams

if TYPE_CHECKING:
    from .network_component_models import Calkit


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


def continuous_sqrt(product: np.ndarray) -> np.ndarray:
    """Square root of a complex quantity, on a branch continuous along the last axis.

    The principal square root of a complex quantity whose phase crosses ±π (e.g.
    S12*S21 of a network with appreciable electrical delay) flips sign at each
    crossing. This returns the principal root at the first finite sample and then
    follows the branch that keeps the root continuous along the last axis
    (frequency), i.e. the sign of each sample is chosen so that consecutive finite
    roots differ in phase by less than π/2. Non-finite samples are skipped (they
    remain non-finite) and do not break continuity across them.

    The result squares to exactly the same values as :func:`numpy.sqrt` (only the
    sign of each element can differ), and the choice is only well defined when the
    phase of ``product`` changes by less than π between consecutive samples.

    Parameters
    ----------
    product
        The complex quantity to take the square root of, with frequency on the last
        axis.

    Returns
    -------
    root
        The square root, with the same shape as ``product``.
    """
    root = np.sqrt(np.asarray(product, dtype=complex))
    if root.ndim == 0:
        return root

    flat = root.reshape(-1, root.shape[-1])
    for row in flat:
        idx = np.flatnonzero(np.isfinite(row))
        if idx.size < 2:
            continue
        good = row[idx]
        flips = np.real(good[1:] * np.conj(good[:-1])) < 0
        sign = np.where(np.cumsum(flips) % 2 == 1, -1, 1)
        row[idx[1:]] *= sign
    return flat.reshape(root.shape)


_STANDARDS = ("open", "short", "match")


def sparams_from_calkit_measurements(
    measurements: CalkitReadings,
    model: "CalkitReadings | Calkit | None" = None,
) -> SParams:
    """Compute S-parameters of a 2-port network from calkit measurements.

    This uses Eq. 3 of Monsalve et al., 2016.

    Parameters
    ----------
    measurements
        The actual measurements of the calkit standards.
    model
        A model of the calkit standards: either the model reflection coefficients
        at the measured frequencies (:class:`CalkitReadings`), or a
        :class:`~edges.cal.sparams.Calkit`, which is evaluated at the measured
        frequencies. If None, ideal standards are assumed.

    Returns
    -------
    sparams
        The S-parameters of the network between the reference plane of the
        measurements and the standards. Only the product S12*S21 is determined by
        the calibration; S12 and S21 are both set to its square root, taken on the
        branch that is continuous across frequency (see :func:`continuous_sqrt`), so
        that S12 and S21 do not flip sign where the phase of S12*S21 crosses ±π.
    """
    from .network_component_models import Calkit

    freq = measurements.freqs

    if isinstance(model, Calkit):
        model = model.at_freqs(freq)
    elif model is None:
        model = CalkitReadings.ideal(freqs=freq)

    # One 3x3 linear system per frequency: rows are the (open, short, match)
    # standards, with model reflection m and measured reflection b,
    # [1, m, m*b] @ [e00, e01e10 - e00 e11, e11] = b.
    m = np.stack(
        [np.asarray(getattr(model, k).reflection_coefficient) for k in _STANDARDS],
        axis=-1,
    )
    b = np.stack(
        [
            np.asarray(getattr(measurements, k).reflection_coefficient)
            for k in _STANDARDS
        ],
        axis=-1,
    )
    a = np.stack([np.ones_like(m), m, m * b], axis=-1)
    x = _solve_osl_systems(a, b)

    s11 = x[:, 0]
    s12s21 = x[:, 1] + x[:, 0] * x[:, 2]
    s22 = x[:, 2]

    s12 = continuous_sqrt(s12s21)
    return SParams(freqs=freq, s11=s11, s12=s12, s21=s12, s22=s22)


def _solve_osl_systems(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Solve a stack of square linear systems ``a[i] @ x[i] = b[i]``.

    Each system is solved by LU decomposition (:func:`numpy.linalg.solve`) in one
    batched call. Systems that are rank-deficient by the criterion that
    :func:`numpy.linalg.lstsq` uses with ``rcond=None`` (smallest singular value
    at most ``eps * n`` times the largest) are instead solved individually with
    :func:`numpy.linalg.lstsq`, so that they get the same minimum-norm solution as
    a per-system least-squares solve. Non-finite inputs raise a
    :class:`numpy.linalg.LinAlgError`, as :func:`numpy.linalg.lstsq` does.

    Parameters
    ----------
    a
        The coefficient matrices, shape ``(N, n, n)``.
    b
        The right-hand sides, shape ``(N, n)``.

    Returns
    -------
    x
        The solutions, shape ``(N, n)``.
    """
    sv = np.linalg.svd(a, compute_uv=False)
    rcond = np.finfo(sv.dtype).eps * a.shape[-1]
    full_rank = sv[:, -1] > rcond * sv[:, 0]

    x = np.empty(b.shape, dtype=np.result_type(a, b))
    x[full_rank] = np.linalg.solve(a[full_rank], b[full_rank, :, None])[..., 0]
    for i in np.flatnonzero(~full_rank):
        x[i] = np.linalg.lstsq(a[i], b[i], rcond=None)[0]
    return x


def de_embed_network_from_calkit_measurements(
    measurements: CalkitReadings, sparams: SParams
) -> CalkitReadings:
    """Remove the effect of a 2-port network from each calkit standard measurement.

    This applies :func:`gamma_de_embed` to each of the open, short and match readings.

    Parameters
    ----------
    measurements
        The measurements of the calkit standards.
    sparams
        The S-parameters of the 2-port network to de-embed.

    Returns
    -------
    readings
        The calkit readings referenced to the other side of the 2-port network.
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


def align_transmission_signs(s: Sequence[SParams]) -> list[SParams]:
    """Align the sign of S12 and S21 of several S-parameter sets to the first one.

    Wherever (frequency by frequency) the S21 of a set points away from the S21 of
    the first set by more than 90 degrees in phase, both its S12 and S21 are negated.
    The product S12*S21, and S11 and S22, are unchanged. This puts S-parameters
    whose S12 and S21 are only defined up to a common sign (as from
    :func:`sparams_from_calkit_measurements`) on a common square-root branch, so that
    they can be averaged or interpolated element by element.

    Parameters
    ----------
    s
        The S-parameters to align. All must be defined at the same frequencies.

    Returns
    -------
    list of SParams
        The aligned S-parameters (the first is returned unchanged).
    """
    ref = s[0].s21
    out = []
    for ss in s:
        sign = np.where(np.real(ss.s21 * np.conj(ref)) < 0, -1, 1)
        out.append(
            SParams(
                freqs=ss.freqs,
                s11=ss.s11,
                s12=sign * ss.s12,
                s21=sign * ss.s21,
                s22=ss.s22,
            )
        )
    return out


def average_sparams(s: Sequence[SParams]) -> SParams:
    """Average multiple S-parameters, element-by-element.

    S12 and S21 are often only known up to a common sign (only their product is
    determined by a one-port calibration, see
    :func:`sparams_from_calkit_measurements`). Before averaging, the sign of S12
    and S21 of each set is therefore aligned, frequency by frequency, to the first
    set: wherever an S21 points away from the first set's S21 (by more than 90
    degrees in phase), both its S12 and S21 are negated. This leaves each S12*S21
    unchanged and avoids the cancellation that occurs when the sets sit on
    different square-root branches. S11 and S22 are averaged directly.

    Parameters
    ----------
    s
        The S-parameters to average. All must be defined at the same frequencies.

    Returns
    -------
    SParams
        The averaged S-parameters.
    """
    s = align_transmission_signs(s)
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
