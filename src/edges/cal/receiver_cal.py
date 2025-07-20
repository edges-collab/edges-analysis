"""Functions to perform reciever calibration given a CalibrationObservation."""
from .calobs import CalibrationObservation
from .calibrator import Calibrator
from collections import deque
from .noise_waves import get_calibration_quantities_iterative as _get_cc_iterative
import numpy as np

def get_calcoeffs_iterative(
    calobs: CalibrationObservation,
    *,
    cterms: int = 5,
    wterms: int = 7,
    apply_loss_to_true_temp: bool = True,
    smooth_scale_offset_within_loop: bool =False,
    cable_delay_sweep: np.ndarray = np.array([0]),
    ncal_iter: int = 4,
    fit_method: str = 'lstsq',
    scale_offset_poly_spacing: float = 1.0,
    t_load: float = 400,
    t_load_ns: float = 300,
) -> Calibrator:
    if (
        hasattr(calobs, "_injected_averaged_spectra")
        and calobs._injected_averaged_spectra is not None
    ):
        ave_spec = calobs._injected_averaged_spectra
    else:
        ave_spec = {
            k: calobs.averaged_spectrum(source) for k, source in calobs.loads.items()
        }

    if calobs.apply_loss_to_true_temp:
        temp_ant = calobs.source_thermistor_temps
        loss = None
    else:
        temp_ant = dict(calobs.source_thermistor_temps)
        temp_ant["hot_load"] = calobs.hot_load.spectrum.temp_ave
        loss = calobs.hot_load.loss()

    scale, off, nwp = deque(
        _get_cc_iterative(
            calobs.freq.to_value("MHz"),
            temp_raw=ave_spec,
            gamma_rec=calobs.receiver.s11_model,
            gamma_ant={name: load.s11_model for name, load in calobs.loads.items()},
            temp_ant=temp_ant,
            cterms=cterms,
            wterms=wterms,
            temp_amb_internal=t_load,
            hot_load_loss=loss,
            smooth_scale_offset_within_loop=smooth_scale_offset_within_loop,
            delays_to_fit=cable_delay_sweep,
            niter=ncal_iter,
            poly_spacing=scale_offset_poly_spacing,
        ),
        maxlen=1,
    ).pop()
    
    Calibrator(
        C1 = scale,
        C2 = off,
        Tunc = nwp.get_tunc(),
        Tcos = nwp.get_tcos(),
        Tsin = nwp.get_tsin(),
    )
    