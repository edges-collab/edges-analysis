import numpy as np
import pytest
from astropy import units as un

from edges.cal import CalibratedS11, CalibrationObservation, Calibrator
from edges.io import CalObsDefEDGES2


class TestCalibrationObservation:
    def test_calobs_bad_input(self, caldef: CalObsDefEDGES2):
        with pytest.raises(ValueError):
            CalibrationObservation.from_edges2_caldef(
                caldef, f_low=100 * un.MHz, f_high=40 * un.MHz
            )

    def test_calibrator_resids(
        self, ideal_calobs: CalibrationObservation, identity_calibrator: Calibrator
    ):
        out = ideal_calobs.get_calibration_residuals(identity_calibrator)

        ld = ideal_calobs.loads
        cmp = {
            k: ideal_calobs.loads[k].averaged_q - ld[k].temp_ave.to_value("K")
            for k, v in out.items()
        }

        np.testing.assert_allclose(out["ambient"], cmp["ambient"])
        np.testing.assert_allclose(out["hot_load"], cmp["hot_load"])
        np.testing.assert_allclose(out["short"], cmp["short"])
        np.testing.assert_allclose(out["open"], cmp["open"])

    def test_rms(
        self, ideal_calobs: CalibrationObservation, identity_calibrator: Calibrator
    ):
        rms = ideal_calobs.get_rms(identity_calibrator)
        assert isinstance(rms, dict)
        assert isinstance(rms["ambient"], un.Quantity)
        assert rms["ambient"].unit.is_equivalent(un.K)

    def test_update(self, calobs: CalibrationObservation):
        c2 = calobs.clone(
            receiver=CalibratedS11(
                s11=np.zeros(calobs.freqs.size, dtype=complex), freqs=calobs.freqs
            )
        )
        assert c2 != calobs

    def test_calobs_equivalence(
        self, calobs: CalibrationObservation, caldef: CalObsDefEDGES2
    ):
        # By construction the same as calobs
        calobs1 = CalibrationObservation.from_edges2_caldef(
            caldef, f_low=50 * un.MHz, f_high=100 * un.MHz
        )

        assert calobs1 == calobs

    def test_load_names(self, calobs):
        assert set(calobs.load_names) == set(calobs.loads.keys())

    def test_load_s11_models(self, calobs):
        models = calobs.load_s11_models
        assert all(name in models for name in calobs.loads)
        assert all(isinstance(val, np.ndarray) for val in models.values())
        assert all(val.size == calobs.freqs.size for val in models.values())

    def test_inject(self, calobs: CalibrationObservation):
        new = calobs.inject(
            receiver=calobs.receiver_s11 * 2,
            source_s11s={
                name: calobs.load_s11_models[name] * 2 for name in calobs.loads
            },
            averaged_q={
                name: load.averaged_q * 2 for name, load in calobs.loads.items()
            },
            thermistor_temp_ave={
                name: load.temp_ave * 2 for name, load in calobs.loads.items()
            },
        )

        np.testing.assert_allclose(new.receiver_s11, 2 * calobs.receiver_s11)

        for name, tmp in new.source_thermistor_temps.items():
            assert np.allclose(tmp, 2 * calobs.source_thermistor_temps[name])

    def test_load_str_to_load(self, calobs):
        assert calobs._load_str_to_load("ambient") == calobs.ambient
        assert calobs._load_str_to_load(calobs.ambient) == calobs.ambient

        with pytest.raises(AttributeError, match="load must be a Load object"):
            calobs._load_str_to_load("non-existent")

        with pytest.raises(AssertionError, match="load must be a Load instance"):
            calobs._load_str_to_load(3)

    def test_hickle_roundtrip(self, calobs, tmpdir):
        calobs.write(tmpdir / "tmp_hickle.h5")
        new = CalibrationObservation.from_file(tmpdir / "tmp_hickle.h5")

        assert new == calobs
