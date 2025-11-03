"""Functions for creating CalibratedS11 objects from calibration load data."""

import numpy as np
from astropy import units as un

from edges import types as tp
from edges.cal.s11.s11model import S11ModelParams
from edges.io import CalObsDefEDGES2, CalObsDefEDGES3, LoadS11, SParams, SwitchingState

from .. import reflection_coefficient as rc
from . import StandardsReadings
from .base import CalibratedS11, CalibratedSParams


def calibrate_loads11_with_switch(
    load_s11: np.ndarray,
    internal_switch: CalibratedSParams,
) -> CalibratedS11:
    """Generate the LoadS11 from an uncalibrated load and internal switch."""
    if not hasattr(load_s11, "__len__"):
        load_s11 = [load_s11]

    freq = internal_switch.freqs

    s11s = []

    for load in load_s11:
        gamma = rc.gamma_de_embed(load, internal_switch.smatrix)
        s11s.append(gamma)

    return CalibratedS11(
        freqs=freq,
        s11=np.mean(s11s, axis=0),
    )


def get_loads11_from_load_and_switch(
    loaddef: LoadS11,
    switchdef: SwitchingState | None = None,
    internal_switch_kwargs: dict | None = None,
    calkit: rc.Calkit | None = None,
    load_kw: dict | None = None,
    f_low: tp.FreqType = 0 * un.MHz,
    f_high: tp.FreqType = np.inf * un.MHz,
) -> CalibratedS11:
    """Compute a calibrated S11 of a calibration load given its files and switch.

    The internal switch must be corrected for (since it is a different pathway
    than the VNA used to take the measurements of the calibration load).
    """
    internal_switch_kwargs = internal_switch_kwargs or {}
    internal_switch_kwargs["f_low"] = f_low
    internal_switch_kwargs["f_high"] = f_high
    internal_switch_model_params = internal_switch_kwargs.pop(
        "model_params", S11ModelParams.from_internal_switch_defaults()
    )

    load_kw = load_kw or {}
    load_kw["f_low"] = f_low
    load_kw["f_high"] = f_high

    standards = StandardsReadings.from_filespec(
        loaddef.calkit, f_low=f_low, f_high=f_high
    )
    uncal_load_s11 = SParams.from_s1p_file(loaddef.external, f_low=f_low, f_high=f_high)
    freq = standards.freq

    # Historically we use (1, -1, 0) in EDGES2, and proper calkit in EDGES3
    smatrix = rc.get_sparams_from_osl(
        1 if calkit is None else calkit.open.reflection_coefficient(freq),
        -1 if calkit is None else calkit.short.reflection_coefficient(freq),
        0.0 if calkit is None else calkit.match.reflection_coefficient(freq),
        standards.open.s11,
        standards.short.s11,
        standards.match.s11,
    )
    loads11 = rc.gamma_de_embed(uncal_load_s11.s11, smatrix)

    if switchdef is not None:
        internal_switch = CalibratedSParams.from_internal_switchdef(
            switchdef, **internal_switch_kwargs
        )

        # Smooth the internal switch
        internal_switch = internal_switch.smoothed(
            params=internal_switch_model_params, freqs=freq
        )
        return calibrate_loads11_with_switch(
            load_s11=loads11, internal_switch=internal_switch
        )
    return CalibratedS11(s11=loads11, freqs=freq)


def get_loads11_from_edges2_loaddef(
    caldef: CalObsDefEDGES2, load: str, **kwargs
) -> CalibratedS11:
    """Calculate the calibrated S11 of a calibration load given a datafile spec."""
    return get_loads11_from_load_and_switch(
        loaddef=getattr(caldef, load).s11, switchdef=caldef.switching_state, **kwargs
    )


def get_loads11_from_edges3_loaddef(
    caldef: CalObsDefEDGES3, load: str, calkit: rc.Calkit = rc.AGILENT_ALAN, **kwargs
) -> CalibratedS11:
    """Create a LoadS11 object from the EDGES-3 CalibrationObservation."""
    return get_loads11_from_load_and_switch(
        loaddef=getattr(caldef, load).s11, calkit=calkit, **kwargs
    )
