"""Functions and classes for dealing with reflection coefficients.

The :module:`core` submodule contains the main base classes and functions for working
with reflection coefficients. The :module:`devices` submodule contains functions
for creating calibrated S11 measurements from measurement files of various devices.
"""

from .core.datatypes import CalkitReadings, ReflectionCoefficient, SParams
from .core.network_component_models import (
    AGILENT_85033E,
    AGILENT_ALAN,
    KNOWN_CABLES,
    Calkit,
    CalkitStandard,
    CoaxialCable,
    TransmissionLine,
    TwoPortNetwork,
    get_calkit,
)
from .core.s11model import (
    S11ModelParams,
    get_delay,
    get_s11_model,
    new_s11_modelled,
    smooth_sparams,
)
from .core.sparam_calibration import (
    average_reflection_coefficients,
    average_sparams,
    gamma2impedance,
    gamma_de_embed,
    gamma_embed,
    impedance2gamma,
)
from .devices.hot_load_cable import (
    get_hot_load_semi_rigid_from_filespec,
    hot_load_cable_model_params,
    read_semi_rigid_cable_sparams_file,
)
from .devices.input_sources import (
    calibrate_gamma_src,
    get_gamma_src_from_filespec,
    input_source_model_params,
)
from .devices.internal_switch import (
    combine_internal_switch_sparams,
    get_internal_switch_from_caldef,
    get_internal_switch_sparams,
    internal_switch_model_params,
)
from .devices.receiver import (
    calibrate_gamma_receiver,
    correct_receiver_for_extra_cable,
    get_gamma_receiver_from_filespec,
    receiver_model_params,
)
