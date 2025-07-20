from pathlib import Path

import hickle
import numpy as np
import pytest
from astropy import units as u
from edges.io import CalObsDefEDGES2
from edges.cal import CalibrationObservation, Calibrator

class TestCalibrationObservation:
    def test_calobs_bad_input(self, caldef: CalObsDefEDGES2):
        with pytest.raises(ValueError):
            CalibrationObservation.from_edges2_caldef(
                caldef, f_low=100 * u.MHz, f_high=40 * u.MHz
            )

    def test_load_resids(self, calobs: CalibrationObservation):
        cal = calobs.calibrate("ambient")

        out = calobs.get_load_residuals()
        mask = ~np.isnan(cal)
        assert np.allclose(out["ambient"][mask], cal[mask] - calobs.ambient.temp_ave)


    def test_rms(calobs):
        rms = calobs.get_rms()
        assert isinstance(rms, dict)
        assert isinstance(rms["ambient"], float)


    def test_update(calobs):
        c2 = calobs.clone(wterms=10)

        assert len(c2.cal_coefficient_models["NW"].model.parameters) == 30
        assert (
            len(c2.cal_coefficient_models["NW"].model.model.models["unc"].parameters) == 10
        )


    def test_calibration_init(calobs, tmpdir: Path):
        calobs.write(tmpdir / "calfile.h5")

        cal = cc.Calibrator.from_calfile(tmpdir / "calfile.h5")

        assert np.allclose(
            cal.receiver_s11(), calobs.receiver.s11_model(calobs.freq.to_value("MHz"))
        )
        assert np.allclose(cal.C1(), calobs.C1())
        assert np.allclose(cal.C2(), calobs.C2())
        assert np.allclose(cal.Tunc(), calobs.Tunc())
        assert np.allclose(cal.Tcos(), calobs.Tcos())
        assert np.allclose(cal.Tsin(), calobs.Tsin())

        temp = calobs.averaged_spectrum(calobs.ambient)
        s11 = calobs.ambient.reflections.s11_model(calobs.freq.to_value("MHz"))
        cal_temp = cal.calibrate_temp(calobs.freq, temp, s11)
        mask = ~np.isnan(cal_temp)
        assert np.allclose(cal_temp[mask], calobs.calibrate("ambient")[mask])

        mask = ~np.isnan(temp)
        assert np.allclose(
            cal.decalibrate_temp(calobs.freq, cal_temp, s11)[mask], temp[mask]
        )




    def test_calobs_equivalence(calobs, io_obs):
        calobs1 = CalibrationObservation.from_io(
            io_obs, f_low=50 * u.MHz, f_high=100 * u.MHz
        )

        assert calobs1.open.spectrum == calobs.open.spectrum
        assert calobs1.open.reflections == calobs.open.reflections
        assert calobs1.open == calobs.open


    def test_basic_s11_properties(calobs):
        assert calobs.open.reflections.load_name == "open"


    def test_inject(calobs):
        new = calobs.inject(
            lna_s11=calobs.receiver_s11 * 2,
            source_s11s={
                name: calobs.s11_correction_models[name] * 2 for name in calobs.loads
            },
            c1=calobs.C1() * 2,
            c2=calobs.C2() * 2,
            t_unc=calobs.Tunc() * 2,
            t_cos=calobs.Tcos() * 2,
            t_sin=calobs.Tsin() * 2,
            averaged_spectra={
                name: calobs.averaged_spectrum(load) * 2
                for name, load in calobs.loads.items()
            },
            thermistor_temp_ave={
                name: load.temp_ave * 2 for name, load in calobs.loads.items()
            },
        )

        np.testing.assert_allclose(new.receiver_s11, 2 * calobs.receiver_s11)
        assert not np.allclose(
            new.get_linear_coefficients("open")[0],
            calobs.get_linear_coefficients("open")[0],
        )

        for name, tmp in new.source_thermistor_temps.items():
            assert np.allclose(tmp, 2 * calobs.source_thermistor_temps[name])

        assert np.allclose(new.C1(), 2 * calobs.C1())
        assert np.allclose(new.C2(), 2 * calobs.C2())
        assert np.allclose(new.Tunc(), 2 * calobs.Tunc())
        assert np.allclose(new.Tcos(), 2 * calobs.Tcos())
        assert np.allclose(new.Tsin(), 2 * calobs.Tsin())

    def test_load_str_to_load(calobs):
        assert calobs._load_str_to_load("ambient") == calobs.ambient
        assert calobs._load_str_to_load(calobs.ambient) == calobs.ambient

        with pytest.raises(AttributeError, match="load must be a Load object"):
            calobs._load_str_to_load("non-existent")

        with pytest.raises(AssertionError, match="load must be a Load instance"):
            calobs._load_str_to_load(3)

    def test_getk_with_freq(calobs):
        k = calobs.get_K()
        k2 = calobs.get_K(calobs.freq)

        assert all(np.allclose(k[key], k2[key]) for key in k)


    def test_hickle_roundtrip(calobs, tmpdir):
        hickle.dump(calobs, tmpdir / "tmp_hickle.h5")
        new = hickle.load(tmpdir / "tmp_hickle.h5")

        assert new == calobs
