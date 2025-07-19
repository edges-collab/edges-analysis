"""Test that a full calibration gives results that don't change over time."""

from pathlib import Path

import h5py
import hickle
import numpy as np
import pytest
from astropy import units as un

from edges.cal import s11
from edges.cal.calobs import CalibrationObservation, HotLoadCorrection, Load
from edges.io import calobsdef


@pytest.fixture(scope="module")
def data(data_path: Path) -> Path:
    return data_path / "2015-09-data"


# @pytest.fixture(scope="module")
# def s11dir(data) -> calobsdef.S11Dir:
#     return calobsdef.S11Dir(data / "S11", run_num={"receiver_reading": 1})


@pytest.fixture(scope="module")
def calobs_2015(data: Path) -> CalibrationObservation:
    f_low = 50 * un.MHz
    f_high = 100 * un.MHz

    pathspec = calobsdef.ReceiverS11(
        calkit=calobsdef.CalkitEdges2.from_standard_layout(
            direc=data / "S11" / "ReceiverReading01",
        ),
        device=data / "S11" / "ReceiverReading01" / "ReceiverReading01.s1p",
    )

    receiver = s11.Receiver.from_io(
        pathspec,
        resistance=49.98 * un.ohm,
        n_terms=11,
        model_type="polynomial",
    )

    refl = {
        src: s11.LoadS11.from_io(
            s11dir,
            src,
            internal_switch_kwargs={"resistance": 50.12 * un.ohm},
            f_low=f_low,
            f_high=f_high,
        )
        for src in ["ambient", "hot_load", "open", "short"]
    }

    spec = {src: hickle.load(data / f"spectra/{src}.h5") for src in refl}

    loads = {}
    for src in refl:
        if src != "hot_load":
            loads[src] = Load(spectrum=spec[src], reflections=refl[src])
        else:
            loads[src] = Load(
                spectrum=spec[src],
                reflections=refl[src],
                loss_model=HotLoadCorrection.from_file(f_low=f_low, f_high=f_high),
                ambient_temperature=spec["ambient"].temp_ave,
            )

    return CalibrationObservation(
        loads=loads,
        receiver=receiver,
        cterms=6,
        wterms=5,
        apply_loss_to_true_temp=True,  # the ref file was produced with these,
        smooth_scale_offset_within_loop=True,  # but we should re-produce it with False
    )


@pytest.mark.parametrize("p", ["C1", "C2", "Tunc", "Tcos", "Tsin"])
def test_cal_params(data: Path, calobs_2015: CalibrationObservation, p: str):
    # TODO: make the tolerances smaller once we have tightened up the calibration.
    with h5py.File(data / "reference.h5", "r") as fl:
        f = fl["freq"][...]
        if p == "C1":
            np.testing.assert_allclose(
                getattr(calobs_2015, p)(f * un.MHz), fl[p.lower()][...], rtol=1e-3
            )
        else:
            np.testing.assert_allclose(
                getattr(calobs_2015, p)(f * un.MHz),
                fl[p.lower()][...],
                atol=200,
                rtol=0,
            )


def test_receiver(data: Path, calobs_2015: CalibrationObservation):
    with h5py.File(data / "reference.h5", "r") as fl:
        f = fl["freq"][...]
        np.testing.assert_allclose(
            calobs_2015.receiver.raw_s11, fl["receiver_raw"][...], atol=1e-8, rtol=1e-5
        )
        np.testing.assert_allclose(
            calobs_2015.receiver.s11_model(f),
            fl["receiver_modeled"][...],
            atol=1e-8,
            rtol=1e-5,
        )


@pytest.mark.parametrize("p", ["s11", "s12", "s22"])
def test_internal_switch(data: Path, calobs_2015: CalibrationObservation, p: str):
    with h5py.File(data / "reference.h5", "r") as fl:
        f = fl["freq"][...]
        np.testing.assert_allclose(
            getattr(calobs_2015.internal_switch, f"{p}_data"),
            fl[f"isw_raw_{p}"][...],
            atol=1e-8,
            rtol=1e-5,
        )
        np.testing.assert_allclose(
            getattr(calobs_2015.internal_switch, f"{p}_model")(f),
            fl[f"isw_mdl_{p}"][...],
            atol=1e-8,
            rtol=1e-5,
        )


@pytest.mark.parametrize("load", ["ambient", "hot_load", "short", "open"])
def test_load_s11(data: Path, calobs_2015: CalibrationObservation, load: str):
    with h5py.File(data / "reference.h5", "r") as fl:
        f = fl["freq"][...]
        np.testing.assert_allclose(
            getattr(calobs_2015, load).s11_model(f),
            fl[f"{load}_s11"][...],
            atol=1e-8,
            rtol=1e-5,
        )


def make_comparison_data(obspath):
    calio = calobsdef.CalibrationObservation(
        obspath,
        run_num={"receiver_reading": 1},
        repeat_num=1,
    )

    calobs = CalibrationObservation.from_io(
        calio,
        f_low=50.0 * un.MHz,
        f_high=100.0 * un.MHz,
        cterms=6,
        wterms=5,
        spectrum_kwargs={
            "default": {"t_load": 300, "t_load_ns": 350, "ignore_times_percent": 7},
            "hot_load": {"ignore_times_percent": 10},
        },
        receiver_kwargs={"n_terms": 11, "model_type": "polynomial"},
    )

    f = np.linspace(50, 100, 100) * un.MHz

    with h5py.File("data/2015-09-data/reference.h5", "w") as fl:
        fl["freq"] = f

        fl["c1"] = calobs.C1(f)
        fl["c2"] = calobs.C2(f)
        fl["tunc"] = calobs.Tunc(f)
        fl["tcos"] = calobs.Tcos(f)
        fl["tsin"] = calobs.Tsin(f)

        fl["receiver_raw"] = calobs.receiver.raw_s11
        fl["receiver_modeled"] = calobs.receiver.s11_model(f)

        fl["isw_raw_s11"] = calobs.internal_switch.s11_data
        fl["isw_raw_s12"] = calobs.internal_switch.s12_data
        fl["isw_raw_s22"] = calobs.internal_switch.s22_data
        fl["isw_mdl_s11"] = calobs.internal_switch.s11_model(f.to_value("MHz"))
        fl["isw_mdl_s12"] = calobs.internal_switch.s12_model(f.to_value("MHz"))
        fl["isw_mdl_s22"] = calobs.internal_switch.s22_model(f.to_value("MHz"))

        for src, load in calobs.loads.items():
            fl[f"{src}_s11"] = load.s11_model(f)

    for src, load in calobs.loads.items():
        hickle.dump(load.spectrum, f"data/2015-09-data/spectra/{src}.h5")


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1:
        obsdir = sys.argv[1]
    else:
        obsdir = (
            "/data5/edges/data/CalibrationObservations/Receiver01/"
            "Receiver01_25C_2015_09_02_040_to_200MHz"
        )

    make_comparison_data(obsdir)
