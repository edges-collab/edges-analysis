"""Equivalence of the GSDataLinearModel fits with straightforward per-spectrum fits.

Each test compares the library implementation with a simple reference
implementation (the original algorithm) on realistic random inputs.
"""

import numpy as np
import pytest
from astropy import units as un

from edges import modeling as mdl
from edges.analysis.datamodel import (
    GSDataLinearModel,
    _fit_spectra_grouped,
    _group_equal_rows,
)
from edges.averaging import NsamplesStrategy, get_weights_from_strategy
from edges.testing import create_mock_edges_data

STRATEGY = NsamplesStrategy.FLAGGED_NSAMPLES


def _mock(kind: str):
    """Mock data: shared weights, per-spectrum flags, or flags and NaN data."""
    gsd = create_mock_edges_data(ntime=24, fhigh=70 * un.MHz)
    rng = np.random.default_rng(21)
    data = gsd.data * (1 + 1e-3 * rng.normal(size=gsd.data.shape))
    if kind == "nan":
        data[..., 3, 40:60] = np.nan
        data[..., 7, 40:60] = np.nan
    gsd = gsd.update(data=data)
    if kind in ("flags", "nan"):
        flags = np.zeros(gsd.data.shape, dtype=bool)
        flags[..., 2, 100:140] = True
        flags[..., 5, 100:140] = True  # same flags as spectrum 2
        flags[..., 9, 7] = True
        gsd = gsd.add_flags("rfi", flags)
    return gsd


def _ref_params(model, gsd, **fit_kwargs) -> np.ndarray:
    """Fit each spectrum separately (the original algorithm)."""
    d = gsd.data.reshape((-1, gsd.nfreqs))
    w = get_weights_from_strategy(gsd, STRATEGY)[0].reshape(d.shape)
    xmodel = model.at(x=gsd.freqs.to_value("MHz"))
    params = np.zeros((d.shape[0], model.n_terms))
    for i, (dd, ww) in enumerate(zip(d, w, strict=True)):
        params[i] = xmodel.fit(ydata=dd, weights=ww, **fit_kwargs).model_parameters
    return params.reshape((gsd.nloads, gsd.npols, gsd.ntimes, model.n_terms))


def _ref_apply(dm, gsd, d, sign: int) -> np.ndarray:
    """Add/subtract the model spectrum by spectrum (the original algorithm)."""
    d = d.reshape((-1, gsd.nfreqs))
    p = dm.parameters.reshape((-1, dm.nparams))
    model = dm.model.at(x=gsd.freqs.to_value("MHz"))
    out = np.zeros_like(d)
    for i, (dd, pp) in enumerate(zip(d, p, strict=True)):
        out[i] = dd + sign * model(parameters=pp)
    out.shape = gsd.data.shape
    return out


@pytest.mark.parametrize("kind", ["shared", "flags", "nan"])
@pytest.mark.parametrize("method", ["lstsq", "qr", "alan-qrd"])
def test_from_gsdata_is_bit_identical(kind: str, method: str):
    """Grouped fits give exactly the parameters of per-spectrum fits."""
    gsd = _mock(kind)
    model = mdl.LinLog(n_terms=5)
    dm = GSDataLinearModel.from_gsdata(
        model, gsd, nsamples_strategy=STRATEGY, method=method
    )
    np.testing.assert_array_equal(dm.parameters, _ref_params(model, gsd, method=method))


@pytest.mark.parametrize("kind", ["shared", "nan"])
def test_residuals_and_spectra_are_bit_identical(kind: str):
    """get_residuals/get_spectra equal the per-spectrum evaluation exactly."""
    gsd = _mock(kind)
    dm = GSDataLinearModel.from_gsdata(
        mdl.LinLog(n_terms=5), gsd, nsamples_strategy=STRATEGY
    )
    resid = dm.get_residuals(gsd)
    ref = _ref_apply(dm, gsd, gsd.data, -1)
    assert type(resid) is type(ref)
    np.testing.assert_array_equal(resid, ref)

    g2 = gsd.update(residuals=resid)
    np.testing.assert_array_equal(
        dm.get_spectra(g2), _ref_apply(dm, g2, g2.residuals, 1)
    )


def test_plain_array_data_fits_are_bit_identical():
    """Plain (unit-less) arrays of spectra give the same grouped fits."""
    gsd = _mock("flags")
    d = np.asarray(gsd.data).reshape((-1, gsd.nfreqs))
    w = np.asarray(get_weights_from_strategy(gsd, STRATEGY)[0]).reshape(d.shape)
    xmodel = mdl.LinLog(n_terms=4).at(x=gsd.freqs.to_value("MHz"))
    ref = np.array([
        xmodel.fit(ydata=dd, weights=ww).model_parameters
        for dd, ww in zip(d, w, strict=True)
    ])
    np.testing.assert_array_equal(_fit_spectra_grouped(xmodel, d, w), ref)


def test_group_equal_rows():
    """Rows are grouped exactly by equality of all the arrays."""
    rng = np.random.default_rng(2)
    a = rng.normal(size=(3, 50))[[0, 1, 0, 2, 1, 0]]
    b = np.ones((6, 50), dtype=bool)
    b[5, 3] = False
    assert _group_equal_rows(a, b) == [[0, 2], [1, 4], [3], [5]]
    assert _group_equal_rows(a[:0], b[:0]) == []


def test_invalid_weights_raise_as_before():
    """An invalid weight raises the same error as a per-spectrum fit."""
    gsd = _mock("shared")
    nsamples = gsd.nsamples.copy()
    nsamples[..., 4, 10] = -1.0
    gsd = gsd.update(nsamples=nsamples)
    with pytest.raises(ValueError, match="non-negative"):
        GSDataLinearModel.from_gsdata(
            mdl.LinLog(n_terms=3), gsd, nsamples_strategy=STRATEGY
        )
