"""Functions for calculating the S-parameters of the internal switch (EDGES 2)."""

from collections.abc import Sequence

import numpy as np

from edges.modeling import ComplexRealImagModel, Polynomial, UnitTransform

from .. import AGILENT_85033E, Calkit, CalkitReadings, S11ModelParams, SParams


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
        provided, the S-parameters will be averaged.
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

    return SParams(
        freqs=freqs,
        s11=np.mean([sp.s11 for sp in sparams_sp2t], axis=0),
        s12=np.mean([sp.s12 for sp in sparams_sp2t], axis=0),
        s21=np.mean([sp.s21 for sp in sparams_sp2t], axis=0),
        s22=np.mean([sp.s22 for sp in sparams_sp2t], axis=0),
    )


def internal_switch_model_params(**kwargs) -> S11ModelParams:
    """Return the default internal switch S-parameters for EDGES-2."""
    model = kwargs.pop(
        "model",
        Polynomial(
            n_terms=7,
            transform=UnitTransform(range=(0, 1)),
        ),
    )

    return S11ModelParams(
        model=model,
        complex_model_type=kwargs.pop("complex_model_type", ComplexRealImagModel),
        find_model_delay=kwargs.pop("find_model_delay", False),
        set_transform_range=True,
        **kwargs,
    )
