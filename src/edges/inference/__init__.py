"""Functions and methods for performing inference on spectra.

The framework used for specifying models and performing fits is ``yabf``.
"""

from .eor_models import FlattenedGaussian, GaussianAbsorptionProfile
from .fitting import SemiLinearFit
from .foregrounds import (
    Bias,
    DampedOscillations,
    DampedSinusoid,
    IonContrib,
    LinLog,
    LinPoly,
    LogPoly,
    PhysicalHills,
    PhysicalLin,
    PhysicalSmallIonDepth,
    Sinusoid,
)
from .linear_fg_likelihoods import LinearFG
from .partial_linear_model import PartialLinearModel
