"""Tests of creating EDGES-3 style calibration."""

import attrs
import numpy as np
import pytest
from astropy import units as un

from edges.cal import CalibrationObservation, LoadSpectrum
from edges.cal import sparams as sp
from edges.io import TEST_DATA_PATH, calobsdef3


@pytest.fixture(scope="module")
def smallcal() -> calobsdef3.CalObsDefEDGES3:
    return calobsdef3.CalObsDefEDGES3.from_standard_layout(
        rootdir=TEST_DATA_PATH / "edges3-mock-root",
        year=2023,
        day=70,
    )


@pytest.fixture(scope="module")
def calobs(smallcal: calobsdef3.CalObsDefEDGES3) -> CalibrationObservation:
    with pytest.warns(UserWarning, match="has no value"):
        return CalibrationObservation.from_edges3_caldef(
            smallcal,
            f_low=50 * un.MHz,
            f_high=100 * un.MHz,
            spectrum_kwargs={
                "default": {"allow_closest_time": True, "temperature": 300.0 * un.K}
            },
        )


def test_calobs_creation(calobs):
    assert calobs is not None


def test_from_edges3_caldef_forwards_kwargs(smallcal, monkeypatch):
    """Extra kwargs are forwarded to _from_caldef rather than silently dropped."""
    captured = {}

    def fake_from_caldef(cls, **kwargs):
        captured.update(kwargs)
        return "sentinel"

    monkeypatch.setattr(
        CalibrationObservation, "_from_caldef", classmethod(fake_from_caldef)
    )
    params = sp.hot_load_cable_model_params()
    out = CalibrationObservation.from_edges3_caldef(
        smallcal, loss_model_params=params, receiver_kwargs={"cable_length": 0 * un.m}
    )
    assert out == "sentinel"
    assert captured["loss_model_params"] is params
    assert captured["receiver_kwargs"]["cable_length"] == 0 * un.m


def test_from_edges3_caldef_does_not_mutate_inputs(smallcal, monkeypatch):
    monkeypatch.setattr(
        CalibrationObservation, "_from_caldef", classmethod(lambda cls, **kw: kw)
    )
    loss_models = {"open": None}
    receiver_kwargs = {"cable_length": 0 * un.m}
    CalibrationObservation.from_edges3_caldef(
        smallcal, loss_models=loss_models, receiver_kwargs=receiver_kwargs
    )
    assert loss_models == {"open": None}
    assert receiver_kwargs == {"cable_length": 0 * un.m}


def test_from_edges3_caldef_unknown_kwarg(smallcal):
    with (
        pytest.raises(TypeError, match="not_a_parameter"),
        pytest.warns(UserWarning, match="has no value"),
    ):
        CalibrationObservation.from_edges3_caldef(
            smallcal,
            f_low=50 * un.MHz,
            f_high=100 * un.MHz,
            spectrum_kwargs={
                "default": {"allow_closest_time": True, "temperature": 300.0 * un.K}
            },
            not_a_parameter=1,
        )


def test_load_spectrum_from_split_templogs(smallcal, tmp_path):
    """Temperatures from several overlapping templogs match those from the full log."""
    full = smallcal.ambient.templog
    text = full.read_text(encoding="latin-1")
    # Split at an entry boundary, with the two halves overlapping.
    starts = [i for i in range(len(text)) if text.startswith("2023_", i)]
    part0, part1 = tmp_path / "part0.log", tmp_path / "part1.log"
    part0.write_text(text[: starts[500]], encoding="latin-1")
    part1.write_text(text[starts[400] :], encoding="latin-1")

    split = attrs.evolve(smallcal.ambient, templog=[part0, part1])

    kw = {"f_low": 50 * un.MHz, "f_high": 100 * un.MHz}
    # The mock .acq files have empty header items, and the mock log has bad lines.
    with (
        pytest.warns(UserWarning, match="malformed lines"),
        pytest.warns(UserWarning, match="has no value"),
    ):
        ref = LoadSpectrum.from_loaddef(smallcal.ambient, **kw)
    with (
        pytest.warns(UserWarning, match="malformed lines"),
        pytest.warns(UserWarning, match="has no value"),
    ):
        new = LoadSpectrum.from_loaddef(split, **kw)

    assert new.temp_ave.unit == un.K
    assert np.isfinite(ref.temp_ave)
    np.testing.assert_allclose(
        new.temp_ave.value, ref.temp_ave.value, rtol=0, atol=1e-10
    )
