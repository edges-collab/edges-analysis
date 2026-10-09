"""Tests of the high-level xRFI runner."""

import logging
import warnings

import numpy as np
import pytest

from edges import modeling as mdl
from edges.filters import runners, xrfi


def _spectra(nrow: int = 4, nfreq: int = 200):
    freqs = np.linspace(50, 150, nfreq)
    rng = np.random.default_rng(10)
    spec = 1000 * (freqs / 75) ** -2.5 + rng.normal(size=(nrow, nfreq))
    spec[:, 37] += 500
    return freqs, spec


ITER_KWARGS = {
    "data_modeler": xrfi.LinearModeler(model=mdl.Polynomial(n_terms=5)),
    "std_modeler": xrfi.LinearModeler(model=mdl.Polynomial(n_terms=2)),
}


def _xrfi_raise(spectrum, *, freqs, weights, **kwargs):
    raise RuntimeError("boom")


_xrfi_raise.ndim = (1,)


class TestRunXRFI:
    def test_serial(self):
        freqs, spec = _spectra()
        flags = runners.run_xrfi(
            method="iterative", spectrum=spec, freqs=freqs, n_threads=1, **ITER_KWARGS
        )
        assert flags.shape == spec.shape
        assert np.all(flags[:, 37])

    def test_parallel_matches_serial(self):
        freqs, spec = _spectra()
        serial = runners.run_xrfi(
            method="iterative", spectrum=spec, freqs=freqs, n_threads=1, **ITER_KWARGS
        )
        parallel = runners.run_xrfi(
            method="iterative", spectrum=spec, freqs=freqs, n_threads=2, **ITER_KWARGS
        )
        np.testing.assert_array_equal(parallel, serial)

    def test_parallel_fully_flagged_row(self):
        freqs, spec = _spectra()
        weights = np.ones_like(spec)
        weights[1] = 0
        flags = runners.run_xrfi(
            method="iterative",
            spectrum=spec,
            freqs=freqs,
            weights=weights,
            n_threads=2,
            **ITER_KWARGS,
        )
        assert np.all(flags[1])

    @pytest.mark.parametrize("n_threads", [1, 2])
    def test_warnings_counted(self, caplog, n_threads):
        # An 11-term data model on 20 channels is unfittable, so every call of the
        # (real, importable) iterative method warns. Using a real method keeps this
        # valid when pool workers are spawned rather than forked.
        freqs = np.linspace(50, 150, 20)
        rng = np.random.default_rng(10)
        spec = 1000 * (freqs / 75) ** -2.5 + rng.normal(size=(5, 20))
        with caplog.at_level(logging.WARNING, logger=runners.logger.name):
            runners.run_xrfi(
                method="iterative",
                spectrum=spec,
                freqs=freqs,
                n_threads=n_threads,
                data_modeler=xrfi.LinearModeler(model=mdl.Polynomial(n_terms=11)),
                std_modeler=xrfi.LinearModeler(model=mdl.Polynomial(n_terms=2)),
                flag_if_broken=False,
            )
        assert (
            "Received warning 'Termination of iterative loop for model-specific "
            "reasons' 5/5 times" in caplog.text
        )

    def test_showwarning_restored_on_error(self, monkeypatch):
        monkeypatch.setattr(xrfi, "xrfi_raise", _xrfi_raise, raising=False)
        freqs, spec = _spectra()
        before = warnings.showwarning
        with pytest.raises(RuntimeError, match="boom"):
            runners.run_xrfi(method="raise", spectrum=spec, freqs=freqs, n_threads=1)
        assert warnings.showwarning is before

    @pytest.mark.parametrize("shape", [(200,), (3, 200), (2, 3, 200)])
    def test_explicit_method(self, shape):
        freqs = np.linspace(50, 150, 200)
        spec = np.ones(shape)
        flags = runners.run_xrfi(
            method="explicit",
            spectrum=spec,
            freqs=freqs,
            n_threads=1,
            extra_rfi=[(60, 70)],
        )
        assert flags.shape == shape
        inband = (freqs > 60) & (freqs < 70)
        assert np.all(flags[..., inband])
        assert not np.any(flags[..., ~inband])
