from pathlib import Path

import numpy as np

from edges.cal import CalibrationObservation, Calibrator


def write_read_roundtrip(identity_calibrator: Calibrator, tmpdir: Path):
    identity_calibrator.write(tmpdir / "tmp.h5")
    new = Calibrator.from_calfile(tmpdir / "tmp.h5")
    assert new == identity_calibrator


def test_cal_uncal_round_trip(calobs: CalibrationObservation, calibrator: Calibrator):
    a, b = calibrator.get_linear_coefficients(
        freqs=calobs.freqs, ant_s11=calobs.open.s11.s11
    )
    q = calobs.open.averaged_q

    np.testing.assert_allclose((q * a + b - b) / a, q, atol=3e-5)

    tcal = calibrator.calibrate_load(calobs.open)
    decal = calibrator.decalibrate(
        tcal,
        ant_s11=calobs.open.s11.s11,
    )
    np.testing.assert_allclose(decal, calobs.open.averaged_q, atol=3e-5)

    new_tcal = calibrator.calibrate_q(
        decal,
        ant_s11=calobs.open.s11.s11,
    )

    np.testing.assert_allclose(new_tcal, tcal, atol=1e-6)
