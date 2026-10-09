import attrs
import numpy as np
import pytest
from astropy import units as un

from edges import DATA_PATH
from edges import modeling as mdl
from edges.cal import (
    CalibrationObservation,
    Calibrator,
    InputSource,
    LossFunctionGivenSparams,
    ReflectionCoefficient,
)
from edges.cal import sparams as sp
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
            for k in out
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
            receiver=ReflectionCoefficient(
                reflection_coefficient=np.zeros(calobs.freqs.size, dtype=complex),
                freqs=calobs.freqs,
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

    def test_inject_only_averaged_q(self, calobs: CalibrationObservation):
        new = calobs.inject(
            averaged_q={"open": calobs.open.averaged_q * 2},
        )
        np.testing.assert_allclose(
            new.open.averaged_q, 2 * calobs.open.averaged_q, rtol=1e-14
        )
        np.testing.assert_allclose(
            new.open.temp_ave.to_value("K"),
            calobs.open.temp_ave.to_value("K"),
            rtol=0,
            atol=0,
        )
        np.testing.assert_allclose(
            new.short.averaged_q, calobs.short.averaged_q, rtol=0, atol=0
        )

    def test_inject_only_thermistor_temp(self, calobs: CalibrationObservation):
        new = calobs.inject(
            thermistor_temp_ave={"short": calobs.short.temp_ave + 10 * un.K},
        )
        np.testing.assert_allclose(
            new.short.temp_ave.to_value("K"),
            calobs.short.temp_ave.to_value("K") + 10,
            rtol=1e-14,
        )
        np.testing.assert_allclose(
            new.short.averaged_q, calobs.short.averaged_q, rtol=0, atol=0
        )
        np.testing.assert_allclose(
            new.open.temp_ave.to_value("K"),
            calobs.open.temp_ave.to_value("K"),
            rtol=0,
            atol=0,
        )


def _caldef_with_hot_load_sparams(caldef: CalObsDefEDGES2) -> CalObsDefEDGES2:
    return attrs.evolve(
        caldef,
        hot_load=attrs.evolve(
            caldef.hot_load,
            sparams_file=DATA_PATH / "semi_rigid_s_parameters_WITH_HEADER.txt",
        ),
    )


def test_hot_load_sparams_file_default_loss_model_params(caldef: CalObsDefEDGES2):
    """A hot-load sparams file on a different grid is smoothed with the defaults."""
    caldef2 = _caldef_with_hot_load_sparams(caldef)

    loss_models = {"open": None}
    calobs = CalibrationObservation.from_edges2_caldef(
        caldef2, f_low=50 * un.MHz, f_high=100 * un.MHz, loss_models=loss_models
    )
    # The caller's dict is not mutated.
    assert loss_models == {"open": None}

    loss = calobs.hot_load.loss
    assert np.all(loss > 0.9)
    assert np.all(loss < 1)
    for name in ("ambient", "open", "short"):
        np.testing.assert_allclose(calobs.loads[name].loss, 1, rtol=0, atol=0)

    # Passing the default loss-model parameters explicitly gives the same answer.
    internal_switch = sp.get_internal_switch_from_caldef(caldef2)
    hot = InputSource.from_caldef(
        caldef2,
        load_name="hot_load",
        f_low=50 * un.MHz,
        f_high=100 * un.MHz,
        ambient_temperature=calobs.ambient.temp_ave,
        internal_switch=internal_switch.smoothed(
            sp.internal_switch_model_params(), freqs=internal_switch.freqs
        ),
        loss_model=LossFunctionGivenSparams(
            sp.read_semi_rigid_cable_sparams_file(
                caldef2.hot_load.sparams_file, f_low=50 * un.MHz, f_high=100 * un.MHz
            )
        ),
        loss_model_params=sp.hot_load_cable_model_params(),
        restrict_s11_freqs=True,
    )
    np.testing.assert_allclose(hot.loss, loss, rtol=1e-12)


def test_kwargs_reach_all_loads_and_are_not_mutated(caldef: CalObsDefEDGES2):
    """User s11 model_params reach every load, and caller dicts are untouched."""
    custom = sp.input_source_model_params(
        name="custom",
        model=mdl.Fourier(
            n_terms=15, transform=mdl.ZerotooneTransform(range=(0, 1)), period=1.5
        ),
    )
    rcv_params = sp.receiver_model_params()
    s11_kwargs = {"model_params": custom}
    spectrum_kwargs = {"default": {"ignore_times": 5 * un.percent}}
    receiver_kwargs = {"model_params": rcv_params}
    loss_models = {"open": None}

    calobs = CalibrationObservation.from_edges2_caldef(
        caldef,
        f_low=50 * un.MHz,
        f_high=100 * un.MHz,
        s11_kwargs=s11_kwargs,
        spectrum_kwargs=spectrum_kwargs,
        receiver_kwargs=receiver_kwargs,
        loss_models=loss_models,
    )

    for name, load in calobs.loads.items():
        expected = load._raw_s11.smoothed(custom, freqs=load.freqs)
        np.testing.assert_allclose(
            load.reflection_coefficient.s11,
            expected.s11,
            rtol=1e-12,
            err_msg=f"model_params not applied to {name}",
        )

    assert s11_kwargs == {"model_params": custom}
    assert s11_kwargs["model_params"] is custom
    assert spectrum_kwargs == {"default": {"ignore_times": 5 * un.percent}}
    assert receiver_kwargs.keys() == {"model_params"}
    assert receiver_kwargs["model_params"] is rcv_params
    assert loss_models == {"open": None}


@pytest.mark.xfail(
    strict=True,
    reason=(
        "restrict_s11_freqs writes f_low/f_high into s11_kwargs that nothing reads, "
        "so S11 models are always fit over the full measured band; fix pending "
        "(result-changing: restrict_s11_model_freqs defaults to True in "
        "CalibrationObservation)"
    ),
)
def test_restrict_s11_freqs(caldef: CalObsDefEDGES2):
    f_low, f_high = 50 * un.MHz, 100 * un.MHz
    load = InputSource.from_caldef(
        caldef,
        load_name="ambient",
        f_low=f_low,
        f_high=f_high,
        ambient_temperature=300 * un.K,
        restrict_s11_freqs=True,
    )
    expected = load._raw_s11.select_frequencies(f_low, f_high).smoothed(
        sp.input_source_model_params(name="ambient"), freqs=load.freqs
    )
    np.testing.assert_allclose(
        load.reflection_coefficient.s11, expected.s11, rtol=0, atol=1e-10
    )
