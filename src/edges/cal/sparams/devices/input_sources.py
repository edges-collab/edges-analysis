"""Functions for calibrating input source reflection coefficients."""

from collections.abc import Sequence

from edges.io import CalObsDefEDGES2, CalObsDefEDGES3, LoadS11
from edges.modeling import ComplexRealImagModel, Fourier, ZerotooneTransform

from .. import (
    Calkit,
    CalkitReadings,
    ReflectionCoefficient,
    S11ModelParams,
    SParams,
    average_reflection_coefficients,
)


def calibrate_gamma_src(
    gamma_src: ReflectionCoefficient,
    internal_osl: CalkitReadings,
    internal_switch: SParams | None = None,
    internal_calkit: Calkit | None = None,
) -> ReflectionCoefficient:
    """Calibrate the reflection coefficient of a calibration source input.

    This moves the reference plane of the reflection coefficient measurement from
    the VNA to the input of the calibration source, de-embedding any internal switches
    and cables.

    Parameters
    ----------
    gamma_src
        The reflection coefficient measured at the VNA reference plane.
    internal_osl
        The internal OSL measurements of the calibration source.
    internal_switch
        The S-parameters of any internal switch to de-embed. If None, no switch is
        de-embedded.
    internal_calkit
        The calkit model for the internal OSL measurements. If None, ideal standards are
        assumed.

    Returns
    -------
    gamma_calibrated
        The calibrated reflection coefficient at the input of the calibration source.
    """
    smatrix = SParams.from_calkit_measurements(
        model=internal_calkit, measurements=internal_osl
    )

    # This de-embeds the internal SP4T subsystem
    gamma_src = gamma_src.de_embed(smatrix)

    if internal_switch is not None:
        gamma_src = gamma_src.de_embed(internal_switch)

    return gamma_src


def get_gamma_src_from_filespec(
    caldef: CalObsDefEDGES2 | CalObsDefEDGES3 | LoadS11 | Sequence[LoadS11],
    source: str | None = None,
    **kwargs,
) -> ReflectionCoefficient:
    """Get the calibrated receiver reflection coeff from a calibration definition."""
    if isinstance(caldef, CalObsDefEDGES2 | CalObsDefEDGES3):
        srcdef = getattr(caldef, source)
    else:
        srcdef = caldef

    if not hasattr(srcdef, "__len__"):
        srcdef = [srcdef]

    gamma_src = []
    for src in srcdef:
        gsrc = ReflectionCoefficient.from_s1p(src.external)
        calkit = CalkitReadings.from_filespec(src.calkit)

        gamma_src.append(
            calibrate_gamma_src(
                gamma_src=gsrc,
                internal_osl=calkit,
                **kwargs,
            )
        )

    return average_reflection_coefficients(gamma_src)


def input_source_model_params(
    name: str,
    find_model_delay: bool = True,
    complex_model_type=ComplexRealImagModel,
    **kwargs,
) -> S11ModelParams:
    """Return the default input source S11 model parameters."""
    n_terms = 27  # default_nterms.get(name, 37)

    model = kwargs.pop(
        "model",
        Fourier(
            n_terms=n_terms, transform=ZerotooneTransform(range=(0, 1)), period=1.5
        ),
    )

    return S11ModelParams(
        model=model,
        find_model_delay=find_model_delay,
        complex_model_type=complex_model_type,
        **kwargs,
    )
