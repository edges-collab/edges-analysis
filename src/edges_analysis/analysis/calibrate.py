"""Module providing routines for calibration of field data."""
import attr
from edges_cal import Calibration, CalibrationObservation
from . import s11 as s11m
from cached_property import cached_property
from typing import Callable, Union, Tuple, Sequence, Optional, Type
import numpy as np
from pathlib import Path
from edges_cal import receiver_calibration_func as rcf


def optional(tp: Type) -> Callable:
    """Define a function that checks if a value is optionally a cetain type."""
    return lambda x: None if x is None else tp(x)


@attr.s(kw_only=True)
class LabCalibration:
    """Lab calibration of field data."""

    calobs: Union[Calibration, CalibrationObservation] = attr.ib()
    s11_files: Sequence[Union[str, Path]] = attr.ib()
    antenna_s11_n_terms: int = attr.ib(default=15)

    @property
    def internal_switch_s11(self) -> Callable:
        """The internal switch S11 model."""
        try:  # if a calibration observation
            return self.calobs.internal_switch.s11_model
        except AttributeError:  # if a calfile
            return self.calobs.internal_switch_s11

    @property
    def internal_switch_s12(self) -> Callable:
        """The internal switch S12 model."""
        try:
            return self.calobs.internal_switch.s12_model
        except AttributeError:
            return self.calobs.internal_switch_s12

    @property
    def internal_switch_s22(self) -> Callable:
        """The internal switch S22 modle."""
        try:
            return self.calobs.internal_switch.s22_model
        except AttributeError:
            return self.calobs.internal_switch_s22

    @cached_property
    def _antenna_s11(
        self,
    ) -> Tuple[Callable[[np.ndarray], np.ndarray], np.ndarray, np.ndarray]:
        model, raw, raw_freq = s11m.antenna_s11_remove_delay(
            self.s11_files,
            f_low=self.calobs.freq.min,
            f_high=self.calobs.freq.max,
            delay_0=0.17,
            internal_switch_s11=self.internal_switch_s11,
            internal_switch_s12=self.internal_switch_s12,
            internal_switch_s22=self.internal_switch_s22,
        )

        return model, raw, raw_freq

    def antenna_s11_model(self, freq: np.ndarray) -> Callable[[np.ndarray], np.ndarray]:
        """Callable S11 model as a function of frequency."""
        return self._antenna_s11[0](freq)

    @cached_property
    def antenna_s11(self) -> np.ndarray:
        """The antenna S11 at the default frequencies."""
        return self.antenna_s11_model(self.calobs.freq.freq)

    @property
    def raw_antenna_s11(self) -> np.ndarray:
        """The raw antenna S11."""
        return self._antenna_s11[1]

    @property
    def raw_antenna_s11_freq(self) -> np.ndarray:
        """The raw antenna s11 frequencies."""
        return self._antenna_s11[2]

    def get_gamma_coeffs(
        self, freq: Optional[np.ndarray] = None, ant_s11: Optional[np.ndarray] = None
    ):
        """Get the K-vector for calibration that is dependent on LNA and Antenna S11."""
        if freq is None:
            freq = self.calobs.freq.freq

        try:
            lna = self.calobs.lna_s11(freq)  # from Calibration
        except TypeError:
            lna = self.calobs.lna_s11  # from CalibrationObservation

        if ant_s11 is None:
            ant_s11 = self.antenna_s11_model(freq)
        return rcf.get_K(lna, ant_s11)

    def get_linear_coefficients(
        self, freq: Optional[np.ndarray] = None, ant_s11: Optional[np.ndarray] = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Get the linear coeffs that transform uncalibrated to calibrated temp."""
        if freq is None:
            freq = self.calobs.freq.freq

        coeffs = self.get_gamma_coeffs(freq, ant_s11=ant_s11)
        a, b = rcf.get_linear_coefficients_from_K(
            coeffs,
            self.calobs.C1(freq),
            self.calobs.C2(freq),
            self.calobs.Tunc(freq),
            self.calobs.Tcos(freq),
            self.calobs.Tsin(freq),
            t_load=self.calobs.t_load,
        )
        return a, b

    def clone(self, **kwargs):
        """Create a new instance based off this one."""
        return attr.evolve(self, **kwargs)

    def calibrate_q(
        self, q: np.ndarray, freq: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """Convert three-position switch ratio to fully calibrated temperature."""
        if freq is None:
            freq = self.calobs.freq.freq
            ant_s11 = self.antenna_s11
        else:
            ant_s11 = self.antenna_s11_model(freq)

        return self.calobs.calibrate_Q(freq, q, ant_s11)

    def calibrate_temp(
        self, temp: np.ndarray, freq: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """Convert semi-calibrated temperature to fully calibrated temperature."""
        if freq is None:
            freq = self.calobs.freq.freq
            ant_s11 = self.antenna_s11
        else:
            ant_s11 = self.antenna_s11_model(freq)
        return self.calobs.calibrate_temp(freq, temp, ant_s11)

    def decalibrate_temp(
        self, temp: np.ndarray, freq: Optional[np.ndarray] = None, to_q=False
    ) -> np.ndarray:
        """Convert fully-calibrated temp to semi-calibrated temp."""
        if not isinstance(self.calobs, Calibration):
            calobs = self.calobs.to_calfile()
        else:
            calobs = self.calobs

        if freq is None:
            freq = self.calobs.freq.freq
            ant_s11 = self.antenna_s11
        else:
            ant_s11 = self.antenna_s11_model(freq)

        out = calobs.decalibrate_temp(freq, temp, ant_s11)

        if to_q:
            return (out - self.calobs.t_load) / self.calobs.t_load_ns
        else:
            return out