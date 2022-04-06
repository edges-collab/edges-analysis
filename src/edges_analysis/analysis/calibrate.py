"""Module providing routines for calibration of field data."""
from __future__ import annotations
import attr
from edges_cal import Calibrator, CalibrationObservation, s11
from .s11 import AntennaS11
from cached_property import cached_property
from typing import Callable, Sequence
import numpy as np
from pathlib import Path
from edges_cal import receiver_calibration_func as rcf
from edges_cal import types as tp


def optional(tp: type) -> Callable:
    """Define a function that checks if a value is optionally a cetain type."""
    return lambda x: None if x is None else tp(x)


@attr.s(kw_only=True)
class LabCalibration:
    """Lab calibration of field data."""

    calobs: Calibrator = attr.ib(
        converter=lambda x: x.to_calibrator()
        if isinstance(x, CalibrationObservation)
        else x
    )
    _antenna_s11_model: AntennaS11 | Callable = attr.ib()

    @classmethod
    def from_s11_files(
        cls,
        calobs: Calibrator | CalibrationObservation,
        s11_files: tp.PathLike | Sequence[tp.PathLike],
        **kwargs,
    ):
        """Generate LabCalibration object from files.

        Parameters
        ----------
        calobs
            The calibration observation with which to calibrate the receiver.
        s11_files
            Either four S1P files that represent the 3 internal standards measurements
            plus one external match, or a single file in which the S11 has already been
            calibrated.

        Other Parameters
        ----------------
        All other parameters are passed to :class:`~s11.AntennaS11`. Includes arguments
        like ``n_terms``, ``model`` and ``model_delay``.
        """
        if isinstance(calobs, CalibrationObservation):
            calobs = calobs.to_calibrator()

        if hasattr(s11_files, "__len__") and len(s11_files) == 4:
            ant_s11 = AntennaS11.from_s1p_files(
                files=s11_files,
                internal_switch=calobs.internal_switch,
                f_low=calobs.freq.min,
                f_high=calobs.freq.max,
                **kwargs,
            )
        else:
            if not isinstance(s11_files, (str, Path)):
                s11_files = s11_files[0]

            ant_s11 = AntennaS11.from_single_file(
                s11_files,
                f_low=calobs.freq.min,
                f_high=calobs.freq.max,
                internal_switch=calobs.internal_switch,
                **kwargs,
            )

        return cls(
            calobs=calobs,
            antenna_s11_model=ant_s11,
        )

    @property
    def internal_switch(self) -> s11.InternalSwitch:
        """The internal switch reflection parameters."""
        return self.calobs.internal_switch

    @property
    def antenna_s11_model(self) -> Callable[[np.ndarray], np.ndarray]:
        """Callable S11 model as a function of frequency."""
        if not isinstance(self._antenna_s11_model, AntennaS11):
            return self._antenna_s11_model
        else:
            return self._antenna_s11_model.s11_model

    @cached_property
    def antenna_s11(self) -> np.ndarray:
        """The antenna S11 at the default frequencies."""
        return self.antenna_s11_model(self.calobs.freq.freq.to_value("MHz"))

    def get_gamma_coeffs(
        self, freq: tp.FreqType | None = None, ant_s11: np.ndarray | None = None
    ):
        """Get the K-vector for calibration that is dependent on LNA and Antenna S11."""
        if freq is None:
            freq = self.calobs.freq.freq

        lna = self.calobs.receiver_s11(freq)

        if ant_s11 is None:
            ant_s11 = self.antenna_s11_model(freq)
        return rcf.get_K(lna, ant_s11)

    def get_linear_coefficients(
        self, freq: np.ndarray | None = None, ant_s11: np.ndarray | None = None
    ) -> tuple[np.ndarray, np.ndarray]:
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

    def calibrate_q(self, q: np.ndarray, freq: tp.FreqType | None = None) -> np.ndarray:
        """Convert three-position switch ratio to fully calibrated temperature."""
        if freq is None:
            freq = self.calobs.freq.freq
            ant_s11 = self.antenna_s11
        else:
            ant_s11 = self.antenna_s11_model(freq)

        return self.calobs.calibrate_Q(freq, q, ant_s11)

    def calibrate_temp(
        self, temp: np.ndarray, freq: tp.FreqType | None = None
    ) -> np.ndarray:
        """Convert semi-calibrated temperature to fully calibrated temperature."""
        if freq is None:
            freq = self.calobs.freq.freq
            ant_s11 = self.antenna_s11
        else:
            ant_s11 = self.antenna_s11_model(freq)
        return self.calobs.calibrate_temp(freq, temp, ant_s11)

    def decalibrate_temp(
        self, temp: np.ndarray, freq: tp.FreqType | None = None, to_q=False
    ) -> np.ndarray:
        """Convert fully-calibrated temp to semi-calibrated temp."""
        if freq is None:
            freq = self.calobs.freq.freq
            ant_s11 = self.antenna_s11
        else:
            ant_s11 = self.antenna_s11_model(freq)

        out = self.calobs.decalibrate_temp(freq, temp, ant_s11)

        if to_q:
            return (out - self.calobs.t_load) / self.calobs.t_load_ns
        else:
            return out

    def with_ant_s11(
        self, ant_s11: Callable[[np.ndarray], np.ndarray]
    ) -> LabCalibration:
        """Clone the instance and add a specific antenna S11 model."""
        return attr.evolve(self, antenna_s11_model=ant_s11)
