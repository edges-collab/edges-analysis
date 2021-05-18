"""Module providing routines for calibration of field data."""
import attr
from edges_cal import Calibration
from . import s11 as s11m
from cached_property import cached_property
from typing import Callable, Union, Tuple, Sequence, Optional
import numpy as np
from edges_cal import FrequencyRange
from pathlib import Path
import warnings
from edges_cal import receiver_calibration_func as rcf

def optional(type):
    return lambda x: None if x is None else type(x)



@attr.s(kw_only=True)
class LabCalibration:
    """Lab calibration of field data."""

    calobs: Calibration = attr.ib()
    s11_files: Sequence[Union[str, Path]] = attr.ib()
    antenna_s11_n_terms: int = attr.ib(default=15)
    _switch_state_dir: Optional[Union[str, Path]] = attr.ib(default=None, converter=optional(Path))
    _switch_state_repeat_num: Optional[int] = attr.ib(default=None, converter=optional(int))

    @_switch_state_dir.validator
    def _ssd_validator(self, att, val):
        if self.calobs.internal_switch is None and val is None:
            raise ValueError(
                "Internal switch of calobs not found, and no switch_state_dir given!"
            )
        if val is not None and not val.exists():
            raise IOError(f"Provided switch_state_dir does not exist! {val}")


    @property
    def switch_state_dir(self) -> Path:
        """The directory in which switching state data exists."""
        if self._switch_state_dir is None:
            return self.calobs.internal_switch.path

        if self.calobs.internal_switch is not None:
            warnings.warn(
                "You should use the switch state that is inherently in the calibration object."
            )
        return self._switch_state_dir.absolute()

    @property
    def switch_state_repeat_num(self):
        if self._switch_state_repeat_num is not None:
            warnings.warn(
                "You should use the switch state repeat_num that is inherently in the "
                "calibration object."
            )
            return self._switch_state_repeat_num
        else:
            if self.calobs.internal_switch is None:
                return 1
            else:
                return self.calobs.internal_switch.repeat_num

    @cached_property
    def _antenna_s11(self) -> Tuple[Callable[[np.ndarray], np.ndarray], np.ndarray, np.ndarray]:
        model, raw, raw_freq = s11m.antenna_s11_remove_delay(
            self.s11_files,
            f_low=self.calobs.freq.min,
            f_high=self.calobs.freq.max,
            switch_state_dir=self.switch_state_dir,
            delay_0=0.17,
            n_fit=self.antenna_s11_n_terms,
            switch_state_repeat_num=self.switch_state_repeat_num,
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

    def get_K(self, freq: Optional[np.ndarray]=None):
        """Get the K-vector for calibration that is dependent on LNA and Antenna S11.
        """
        if freq is None:
            freq = self.calobs.freq.freq

        lna = self.calobs.lna_s11(freq)
        return rcf.get_K(lna, self.antenna_s11_model(freq))

    def get_linear_coefficients(self, freq: Optional[np.ndarray]=None) -> Tuple[np.ndarray, np.ndarray]:
        """Get the linear coefficients that transform uncalibrated to calibrated temp.
        """
        if freq is None:
            freq = self.calobs.freq.freq

        K = self.get_K(freq)
        a, b = rcf.get_linear_coefficients_from_K(
            K, self.calobs.C1(freq), self.calobs.C2(freq), self.calobs.Tunc(freq), self.calobs.Tcos(freq), self.calobs.Tsin(freq),
            t_load=self.calobs.t_load,
        )
        return a, b

    def clone(self, **kwargs):
        """Create a new instance based off this one."""
        return attr.evolve(self, **kwargs)


    def calibrate_q(self, q: np.ndarray) -> np.ndarray:
        """Convert three-position switch ratio to fully calibrated temperature."""
        return self.calobs.calibrate_Q(self.calobs.freq.freq, q, self.antenna_s11)

    def calibrate_temp(self, temp: np.ndarray) -> np.ndarray:
        """Convert semi-calibrated temperature to fully calibrated temperature."""
        return self.calobs.calibrate_temp(self.calobs.freq.freq, temp, self.antenna_s11)

    def decalibrate_temp(self, temp: np.ndarray) -> np.ndarray:
        """Convert fully-calibrated temp to semi-calibrated temp."""
        return self.calobs.decalibrate_temp(self.calobs.freq.freq, temp, self.antenna_s11)