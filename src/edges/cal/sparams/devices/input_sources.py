"""Functions for calibrating input source reflection coefficients."""

from edges.modeling import Fourier, UnitTransform

from .. import Calkit, CalkitReadings, ReflectionCoefficient, S11ModelParams, SParams


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


def input_source_model_params(
    name: str, find_model_delay: bool = True, **kwargs
) -> S11ModelParams:
    """Return the default input source S11 model parameters."""
    default_nterms = {
        "ambient": 37,
        "hot_load": 37,
        "open": 105,
        "short": 105,
    }
    n_terms = default_nterms.get(name, 37)

    model = kwargs.pop(
        "model", Fourier(n_terms=n_terms, transform=UnitTransform(range=(0, 1)))
    )

    return S11ModelParams(model=model, find_model_delay=find_model_delay, **kwargs)
