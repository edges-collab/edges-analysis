"""Test that the S-parameter calibration functions reproduce B18 results."""

from pathlib import Path

import astropy.units as un
import numpy as np
import pytest

from edges import io
from edges.alanmode import read_modelled_s11s, read_raul_s11_format
from edges.cal import sparams as sp
from edges.data import (
    fetch_b18_cal_outputs,
    fetch_b18cal_calibrated_s11s,
    fetch_b18cal_s11s,
)

# Need to fix the offset delay of the match standard for AGILENT_85033E
CALKITKW = {"match": {"offset_delay": 30 * un.picosecond}}


@pytest.fixture(scope="module")
def s11root() -> Path:
    return fetch_b18cal_s11s()


@pytest.fixture(scope="module")
def legacy_raw_s11s() -> dict:
    return read_raul_s11_format(fetch_b18cal_calibrated_s11s())


@pytest.fixture(scope="module")
def legacy_mdl_s11s() -> dict:
    out = read_modelled_s11s(
        fetch_b18_cal_outputs() / "H2Case-fittpfix" / "s11_modelled.txt"
    )
    mask = (out["freqs"] >= 50 * un.MHz) & (out["freqs"] <= 100 * un.MHz)
    return out[mask]


@pytest.fixture(scope="module")
def internal_switch_smooth(s11root: Path) -> sp.SParams:
    def get_switch_state_filespec(d: Path):
        return io.InternalSwitch(
            internal=io.CalkitFileSpec(
                open=d / "open.S1P",
                short=d / "short.S1P",
                match=d / "load.S1P",
            ),
            external=io.CalkitFileSpec(
                open=d / "open_input.S1P",
                short=d / "short_input.S1P",
                match=d / "load_input.S1P",
            ),
        )

    internal_switch_paths = sorted((s11root / "InternalSwitch").glob("*degC"))
    internal_switch_filespecs = [
        get_switch_state_filespec(d) for d in internal_switch_paths
    ]
    internal_switch = sp.get_internal_switch_from_caldef(
        internal_switch_filespecs,
        calkit_overrides=CALKITKW,
        measured_temperature=27.0 * un.deg_C,
    )
    intsw_model_params = sp.internal_switch_model_params()
    return internal_switch.smoothed(params=intsw_model_params)


def test_receiver(s11root, legacy_raw_s11s, legacy_mdl_s11s):
    """Test that the B18 receiver S11 matches legacy data."""
    lna_paths = sorted(s11root.glob("ReceiverReading*"))

    def get_lna_filespec(d: Path):
        calkit = io.CalkitFileSpec(
            open=d / "open.s1p",
            short=d / "short.s1p",
            match=d / "load.s1p",
        )
        return io.ReceiverS11(calkit=calkit, device=d / "LNA.s1p")

    lna_filespecs = [get_lna_filespec(d) for d in lna_paths]
    gamma_rcv = sp.get_gamma_receiver_from_filespec(
        lna_filespecs, calkit_overrides=CALKITKW
    )

    # Test that the calibrated unsmoothed S11 matches legacy data
    np.testing.assert_allclose(
        gamma_rcv.reflection_coefficient, legacy_raw_s11s["lna"], atol=1e-15
    )

    # Test that the smoothed modelled S11 matches legacy data
    freqs = legacy_mdl_s11s["freqs"]
    smooth = gamma_rcv.smoothed(params=sp.receiver_model_params(), freqs=freqs)
    np.testing.assert_allclose(
        smooth.reflection_coefficient, legacy_mdl_s11s["receiver"], rtol=1e-6
    )


@pytest.mark.parametrize("source", ["ambient", "hot_load", "open", "short"])
def test_gamma_src(
    source: str,
    s11root: Path,
    internal_switch_smooth: sp.SParams,
    legacy_raw_s11s: dict,
    legacy_mdl_s11s: dict,
):
    """Test that the B18 source S11s match legacy data."""
    fname = {
        "open": "LongCableOpen",
        "short": "LongCableShorted",
        "ambient": "Ambient",
        "hot_load": "HotLoad",
    }[source]

    d = s11root / fname

    # Internal calkit measurements
    calkit = io.CalkitFileSpec(
        open=d / "open.S1P",
        short=d / "short.S1P",
        match=d / "load.S1P",
    )

    # Measurement of the external source plugged in at the input (SMA1)
    external = next(iter(d.glob("*_input.S1P")))

    filespec = io.LoadS11(calkit=calkit, external=external)
    gamma_src = sp.get_gamma_src_from_filespec(
        filespec, internal_switch=internal_switch_smooth
    )

    oldname = (
        "amb" if source == "ambient" else "hot" if source == "hot_load" else source
    )
    # Test that the calibrated unsmoothed S11 matches legacy data
    np.testing.assert_allclose(
        gamma_src.reflection_coefficient, legacy_raw_s11s[oldname], atol=1e-13
    )

    # Test that the smoothed modelled S11 matches legacy data
    freqs = legacy_mdl_s11s["freqs"]
    smooth = gamma_src.smoothed(
        params=sp.input_source_model_params(source), freqs=freqs
    )
    np.testing.assert_allclose(
        smooth.reflection_coefficient, legacy_mdl_s11s[source], rtol=1e-6
    )


def test_hot_load_cable(s11root: Path, legacy_raw_s11s: dict, legacy_mdl_s11s: dict):
    """Test that the B18 hot load cable S11 matches legacy data."""
    d = s11root / "SemiRigidCableHotLoad"

    filespec = io.HotLoadSemiRigidCable(
        osl=io.CalkitFileSpec(
            open=d / "open_sr_0dBm_500av.S1P",
            short=d / "short_sr_0dBm_500av.S1P",
            match=d / "load_sr_REPETITION_2_0dBm_500av.S1P",
        )
    )
    semirigid = sp.get_hot_load_semi_rigid_from_filespec(
        filespec, calkit_overrides=CALKITKW
    )

    # Test that the calibrated unsmoothed S11 matches legacy data
    np.testing.assert_allclose(semirigid.s11, legacy_raw_s11s["s11rig"], atol=1e-14)
    np.testing.assert_allclose(semirigid.s12**2, legacy_raw_s11s["s12rig"], atol=1e-14)
    np.testing.assert_allclose(semirigid.s22, legacy_raw_s11s["s22rig"], atol=1e-14)

    # Test that the smoothed modelled S11 matches legacy data
    freqs = legacy_mdl_s11s["freqs"]
    smooth = semirigid.smoothed(params=sp.hot_load_cable_model_params(), freqs=freqs)
    np.testing.assert_allclose(smooth.s11, legacy_mdl_s11s["semi_rigid s11"], rtol=1e-6)
    np.testing.assert_allclose(
        smooth.s12**2, legacy_mdl_s11s["semi_rigid s12"], rtol=1e-6
    )
    np.testing.assert_allclose(smooth.s22, legacy_mdl_s11s["semi_rigid s22"], rtol=1e-6)
