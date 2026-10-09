"""Functions for calculating the S-parameters of the internal switch (EDGES 2)."""

from collections.abc import Sequence
from typing import Any

import numpy as np
from astropy import units as un
from scipy.interpolate import RegularGridInterpolator

from edges import types as tp
from edges.io import CalObsDefEDGES2
from edges.io.calobsdef import InternalSwitch
from edges.modeling import ComplexRealImagModel, Polynomial, ScaleTransform

from .. import (
    AGILENT_85033E,
    Calkit,
    CalkitReadings,
    S11ModelParams,
    SParams,
    average_sparams,
    get_calkit,
)
from ..core.sparam_calibration import align_transmission_signs, continuous_sqrt


def get_internal_switch_sparams(
    internal_osl: CalkitReadings | Sequence[CalkitReadings],
    external_osl: CalkitReadings | Sequence[CalkitReadings],
    external_calkit: Calkit = AGILENT_85033E,
    internal_calkit: Calkit | None = None,
) -> SParams:
    """Compute the S-parameters of the internal switch subsystem for EDGES-2.

    Techically, the "internal switch" subsystem includes both the SP2T and the input
    SMA (see Figure 1 of Murray+2025). To compute the S-parameters of this subsystem, we
    de-embed the SP4T subsystem from the external measurements.

    Parameters
    ----------
    internal_osl
        The internal OSL measurements of the switch. If multiple measurements are
        provided, the S-parameters will be averaged.
    external_osl
        The external OSL measurements of the switch. If multiple measurements are
        provided, the S-parameters will be averaged with :func:`average_sparams`,
        which aligns the sign of each repeat's S12 and S21 before averaging.
    external_calkit
        The calkit model for the external OSL measurements.
    internal_calkit
        The calkit model for the internal OSL measurements. If None, ideal standards are
        assumed.

    Returns
    -------
    SParams
        The S-parameters of the internal switch subsystem.
    """
    if not hasattr(internal_osl, "__len__"):
        internal_osl = [internal_osl]

    if not hasattr(external_osl, "__len__"):
        external_osl = [external_osl]

    freqs = internal_osl[0].freqs

    sparams_sp2t = []
    for internal, external in zip(internal_osl, external_osl, strict=True):
        # Compute the S-parameters of the SP4T subsystem using the
        # internal OSL measurements.
        smtrx_sp4t_subsystem = SParams.from_calkit_measurements(
            model=internal_calkit,
            measurements=internal,
        )

        # Re-reference the external OSL measurements to the input of the SP4T
        # (point r2 in Murray+2025, Figure 1).
        external_at_r2 = external.de_embed(smtrx_sp4t_subsystem)

        # Compute the Sparams of the SP2T subsystem using the external OSL measurements
        # referenced to point r2.
        sparams_sp2t.append(
            SParams.from_calkit_measurements(
                model=external_calkit.at_freqs(freqs),
                measurements=external_at_r2,
            )
        )

    return average_sparams(sparams_sp2t)


def combine_internal_switch_sparams(
    sparams: Sequence[SParams],
    temperatures: tp.TemperatureType,
    measured_temperature: tp.TemperatureType,
    combine_s12s21: bool = True,
) -> SParams:
    """Linearly interpolate internal switch S-parameters to the desired temperature.

    Parameters
    ----------
    sparams
        The S-parameters of the internal switch at each of ``temperatures``.
    temperatures
        The temperatures at which each of ``sparams`` was measured.
    measured_temperature
        The temperature to which to interpolate.
    combine_s12s21
        If True, interpolate the product S12*S21 and set S12 and S21 to its square
        root (on the branch continuous across frequency). Otherwise, interpolate
        S12 and S21 separately, after aligning their signs across temperatures
        with :func:`.align_transmission_signs`.

    Returns
    -------
    SParams
        The interpolated S-parameters.
    """
    if not combine_s12s21:
        sparams = align_transmission_signs(sparams)

    interp_x = np.array([
        np.ones(len(sparams[0].freqs))
        * measured_temperature.to_value("K", un.temperature()),
        sparams[0].freqs.to_value("MHz"),
    ]).T

    keys = ["s11", "s22"] if combine_s12s21 else ["s11", "s22", "s12", "s21"]

    tt = [t.to_value("K", un.temperature()) for t in temperatures]
    interpolated = {
        param: RegularGridInterpolator(
            points=(tt, sparams[0].freqs.to_value("MHz")),
            values=np.array([getattr(sp, param) for sp in sparams]),
            bounds_error=True,
        )(interp_x)
        for param in keys
    }

    if combine_s12s21:
        s12_s21 = RegularGridInterpolator(
            points=(tt, sparams[0].freqs.to_value("MHz")),
            values=np.array([sp.s12 * sp.s21 for sp in sparams]),
            bounds_error=True,
        )(interp_x)
        s12_s21 = continuous_sqrt(s12_s21)
        interpolated["s12"] = s12_s21
        interpolated["s21"] = s12_s21

    return SParams(freqs=sparams[0].freqs, **interpolated)


def get_internal_switch_from_caldef(
    caldef: CalObsDefEDGES2 | InternalSwitch | Sequence[InternalSwitch],
    external_calkit: Calkit | None = None,
    internal_calkit: Calkit | None = None,
    measured_temperature: tp.TemperatureType | None = None,
    calkit_overrides: dict[str, Any] | None = None,
) -> SParams:
    """Obtain the S-parameters of the internal switch from a caldef.

    If the caldef contains multiple internal switch calkit measurements at a set
    of temperatures, the S-parameters will be interpolated to the measured_temperature.

    Parameters
    ----------
    caldef
        The calibration definition object that points to all the required datafiles.
    external_calkit
        The calkit model for the external OSL measurements.
    internal_calkit
        The calkit model for the internal OSL measurements. If None, ideal standards are
        assumed.
    measured_temperature
        The temperature at which to interpolate the internal switch S-parameters.
    """
    intsw = caldef.internal_switch if isinstance(caldef, CalObsDefEDGES2) else caldef

    if not hasattr(intsw, "__len__"):  # single set of measurements
        intsw = [intsw]

    sparams = []
    for sw in intsw:
        if external_calkit is None:
            this_external_calkit = get_calkit(
                sw.external_calkit, resistance_of_match=sw.calkit_match_resistance
            )
        else:
            this_external_calkit = external_calkit

        if calkit_overrides:
            this_external_calkit = this_external_calkit.clone(**calkit_overrides)

        internal_calkit_measurements = CalkitReadings.from_filespec(sw.internal)
        external_calkit_measurements = CalkitReadings.from_filespec(sw.external)

        sparams.append(
            get_internal_switch_sparams(
                internal_osl=internal_calkit_measurements,
                external_osl=external_calkit_measurements,
                external_calkit=this_external_calkit,
                internal_calkit=internal_calkit,
            )
        )

    if len(sparams) > 1:
        return combine_internal_switch_sparams(
            sparams,
            temperatures=[sw.temperature for sw in intsw],
            measured_temperature=measured_temperature,
        )
    return sparams[0]


def internal_switch_model_params(**kwargs) -> S11ModelParams:
    """Return the default internal switch S-parameters for EDGES-2."""
    model = kwargs.pop(
        "model",
        Polynomial(
            n_terms=8,
            transform=ScaleTransform(scale=75),
        ),
    )

    return S11ModelParams(
        model=model,
        complex_model_type=kwargs.pop("complex_model_type", ComplexRealImagModel),
        find_model_delay=kwargs.pop("find_model_delay", False),
        set_transform_range=True,
        **kwargs,
    )
