"""Test the loss module."""

from pathlib import Path

import numpy as np
import pytest
from astropy import units as un

from edges.analysis import loss
from edges.cal.sparams import ReflectionCoefficient
from edges.data import DATA_PATH


def test_no_band():
    with pytest.raises(ValueError, match="you must provide 'band'"):
        loss.ground_loss(filename=":", freq=np.linspace(50, 100, 100))


class TestLow2BalunConnectorLoss:
    @pytest.mark.parametrize("use_approx_eps0", [True, False])
    def test_happy_path(self, use_approx_eps0):
        fq = np.linspace(50, 100, 51) * un.MHz
        ants11 = np.ones(51) / 2

        bcloss = loss.low2_balun_connector_loss(
            ReflectionCoefficient(reflection_coefficient=ants11, freqs=fq),
            use_approx_eps0=use_approx_eps0,
        )

        assert bcloss.shape == fq.shape

    def test_other_s11_inputs(self, tmp_path: Path):
        s11 = ReflectionCoefficient(
            freqs=np.linspace(50, 100, 100) * un.MHz,
            reflection_coefficient=np.zeros(100, dtype=complex),
        )

        # S11Model object
        bcloss = loss.low2_balun_connector_loss(ants11=s11)
        assert bcloss.shape == s11.freqs.shape

        # Hickled file
        s11.write(tmp_path / "tmp-ant.h5")
        bcloss2 = loss.low2_balun_connector_loss(tmp_path / "tmp-ant.h5")
        assert np.allclose(bcloss, bcloss2)


def _read_tabulated_loss(instrument: str, name: str) -> np.ndarray:
    return np.genfromtxt(DATA_PATH / "loss" / instrument / f"{name}.txt")


@pytest.mark.filterwarnings("ignore:Ground loss file .* does not exist")
@pytest.mark.xfail(
    strict=True,
    reason=(
        "Built-in loss files are looked up in edges/analysis/data/loss instead "
        "of edges/data/loss, so a warning is raised and a loss of 1 (no correction) is "
        "returned; fix pending (result-changing)"
    ),
)
@pytest.mark.parametrize(
    ("instrument", "configuration"), [("low", ""), ("low", "45deg"), ("mid", "")]
)
def test_builtin_ground_loss_matches_tabulated(instrument: str, configuration: str):
    name = f"ground_{configuration}" if configuration else "ground"
    table = _read_tabulated_loss(instrument, name)

    gl = loss.ground_loss(
        table[:, 0], ":", instrument=instrument, configuration=configuration
    )
    assert not np.allclose(gl, 1.0)
    # Tabulated losses are ~1e-3; the polynomial fit reproduces them to <3e-5.
    np.testing.assert_allclose(gl, 1 - table[:, 1], atol=5e-5, rtol=0)


@pytest.mark.filterwarnings("ignore:Ground loss file .* does not exist")
@pytest.mark.xfail(
    strict=True,
    reason=(
        "antenna_loss passes loss_type='ground', so it reads the ground-loss "
        "file instead of antenna.txt (and no built-in loss file is found at all); "
        "fix pending (result-changing)"
    ),
)
def test_builtin_antenna_loss_reads_antenna_file():
    table = _read_tabulated_loss("mid", "antenna")
    al = loss.antenna_loss(table[:, 0], ":", instrument="mid")
    np.testing.assert_allclose(al, 1 - table[:, 1], atol=5e-5, rtol=0)


@pytest.mark.xfail(
    strict=True,
    reason=(
        "antenna_loss uses its n_terms=11 as the polynomial degree (12 terms) "
        "rather than the number of terms; fix pending (result-changing)"
    ),
)
def test_antenna_loss_uses_n_terms():
    fname = DATA_PATH / "loss" / "mid" / "antenna.txt"
    table = np.genfromtxt(fname)
    freq = np.linspace(50, 150, 101)

    expected = 1 - np.polyval(np.polyfit(table[:, 0], table[:, 1], 10), freq)
    al = loss.antenna_loss(freq, fname)
    np.testing.assert_allclose(al, expected, rtol=0, atol=1e-12)
