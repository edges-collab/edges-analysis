"""A package for performing linear model fits."""

from .composite import ComplexMagPhaseModel, ComplexRealImagModel, CompositeModel
from .core import FixedLinearModel, Model, Modelable, get_mdl, get_mdl_inst
from .data_transforms import DataTransform
from .fitting import ModelFit
from .models import (
    EdgesPoly,
    Foreground,
    Fourier,
    FourierDay,
    LinLog,
    LogPoly,
    PhysicalLin,
    Polynomial,
)
from .xtransforms import (
    IdentityTransform,
    Log10Transform,
    LogTransform,
    ScaleTransform,
    ShiftTransform,
    UnitTransform,
    XTransform,
    ZerotooneTransform,
)
