import numpy as np
import pytest
import yaml

from edges import modeling as mdl


def test_pass_params():
    pl = mdl.PhysicalLin(parameters=[1, 2, 3])

    epl = pl.at(x=np.linspace(0, 1, 10))

    assert epl.n_terms == 3

    with pytest.raises(ValueError):
        mdl.PhysicalLin(parameters=[1, 2, 3], n_terms=4)


def test_bad_get_mdl():
    with pytest.raises(ValueError):
        mdl.get_mdl(3)


@pytest.mark.parametrize(
    "model",
    [mdl.PhysicalLin, mdl.Polynomial, mdl.EdgesPoly, mdl.Fourier, mdl.FourierDay],
)
def test_basis(model: type[mdl.Model]):
    x = np.linspace(1, 2, 10)
    pl = model(parameters=[1, 2, 3]).at(x=x)

    assert pl.basis.shape == (3, 10)
    assert pl().shape == (10,)
    assert pl(x=np.linspace(1, 2, 20)).shape == (20,)

    pl2 = model(n_terms=4).at(x=x)
    with pytest.raises(ValueError):
        pl2()

    assert pl2(parameters=[1, 2, 3]).shape == (10,)

    with pytest.raises(ValueError):
        pl2(parameters=[1, 2, 3, 4, 5])


def test_model_fit():
    pl = mdl.PhysicalLin()
    fit = mdl.ModelFit(
        pl.at(x=np.linspace(0, 1, 10)),
        ydata=np.linspace(0, 1, 10),
    )
    assert isinstance(fit.model.model, mdl.PhysicalLin)


@pytest.mark.parametrize("method", ["lstsq", "qr", "alan-qrd", "qrd-c"])
def test_simple_fit(method: str):
    pl = mdl.PhysicalLin(parameters=[1, 2, 3])
    model = pl.at(x=np.linspace(50, 100, 10))

    data = model()
    fit = mdl.ModelFit(model, ydata=data, method=method)

    assert np.allclose(fit.model_parameters, [1, 2, 3])
    assert np.allclose(fit.residual, 0)
    assert np.allclose(fit.weighted_chi2, 0)
    assert np.allclose(fit.reduced_weighted_chi2, 0)
    assert fit.hessian.shape == (3, 3)


@pytest.mark.parametrize("method", ["lstsq", "qr", "alan-qrd", "qrd-c"])
def test_weighted_fit(method: str):
    rng = np.random.default_rng(1234)
    four = mdl.Fourier(parameters=[1, 2, 3])
    model = four.at(x=np.linspace(50, 100, 10))

    sigmas = np.abs(model() / 100)
    data = model() + rng.normal(scale=sigmas)

    fit = mdl.ModelFit(model, ydata=data, weights=1 / sigmas)

    assert np.allclose(fit.model_parameters, [1, 2, 3], rtol=0.05)


def test_wrong_params():
    with pytest.raises(ValueError):
        mdl.Polynomial(n_terms=5, parameters=(1, 2, 3, 4, 5, 6))


def test_no_nterms():
    with pytest.raises(ValueError):
        mdl.Polynomial()


def test_model_fit_intrinsic():
    m = mdl.Polynomial(n_terms=2).at(x=np.linspace(0, 1, 10))
    fit = m.fit(ydata=np.linspace(0, 1, 10))
    assert np.allclose(fit.evaluate(), fit.ydata)


def test_physical_lin():
    m = mdl.PhysicalLin(n_terms=5, transform=mdl.IdentityTransform()).at(
        x=np.array([1 / np.e, 1, np.e])
    )

    basis = m.basis
    assert np.allclose(basis[0], [np.e**2.5, 1, np.e**-2.5])
    assert np.allclose(basis[1], [-(np.e**2.5), 0, np.e**-2.5])
    assert np.allclose(basis[2], [np.e**2.5, 0, np.e**-2.5])
    assert np.allclose(basis[3], [np.e**4.5, 1, np.e**-4.5])
    assert np.allclose(basis[4], [np.e**2, 1, np.e**-2])


def test_linlog():
    m = mdl.LinLog(n_terms=3).at(x=np.array([0.5, 1, 2]))
    assert m.basis.shape == (3, 3)


def test_yaml_roundtrip():
    p = mdl.Polynomial(n_terms=5)
    s = yaml.dump(p)
    pp = yaml.load(s, Loader=yaml.FullLoader)
    print(p.data_transform.__class__, pp.data_transform.__class__)
    assert p == pp
    assert "!Model" in s


def test_get_mdl_inst():
    assert isinstance(mdl.get_mdl_inst("polynomial", n_terms=5), mdl.Polynomial)
    poly = mdl.Polynomial(n_terms=5)
    assert mdl.get_mdl_inst(poly) == poly
    assert mdl.get_mdl_inst(poly, n_terms=6).n_terms == 6
    assert mdl.get_mdl_inst(mdl.Polynomial, n_terms=5).n_terms == 5
    assert mdl.Model.from_str("polynomial", n_terms=10).n_terms == 10


def test_too_many_nterms():
    with pytest.raises(ValueError):
        mdl.PhysicalLin(n_terms=10)


def test_at_x():
    x1 = np.linspace(0, 1, 20)
    first = mdl.Polynomial(n_terms=10).at(x=x1)
    second = first.at_x(x1 * 2)
    assert not np.allclose(first.basis, second.basis)
    assert first.model == second.model


def test_init_basis():
    x = np.linspace(0, 1, 7)
    with pytest.raises(ValueError):
        mdl.Polynomial(n_terms=10).at(x=x, init_basis=np.zeros((10, 20)))


def test_composite_model():
    poly = mdl.Polynomial(n_terms=5, parameters=[1, 2, 3, 4, 5])
    four = mdl.Fourier(n_terms=10, parameters=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    m = mdl.CompositeModel(models={"poly": poly, "fourier": four})
    x = np.linspace(-1, 1, 10)
    assert np.allclose(m["poly"].parameters, poly.parameters)
    assert np.allclose(m["fourier"].parameters, four.parameters)
    assert m.n_terms == 15
    assert m._index_map[0] == ("poly", 0)
    assert m._index_map[5] == ("fourier", 0)
    assert np.allclose(m.get_model("poly", x=x), poly(x=x))

    mx = m.at(x=x)
    assert np.allclose(mx(), m(x=x))
    assert m == m.with_params([1, 2, 3, 4, 5, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    assert np.allclose(
        m.fit(xdata=x, ydata=np.sin(x)).model_parameters,
        mx.fit(np.sin(x)).model_parameters,
    )


def test_complex_model():
    cmplx = mdl.ComplexMagPhaseModel(
        mag=mdl.Polynomial(n_terms=5), phs=mdl.Polynomial(n_terms=6)
    )

    x = np.linspace(0, 1, 150)
    data = cmplx(x=x, parameters=[0, 1, 2, 3, 4, 0, 1, 2, 3, 4, 5])
    fit = cmplx.fit(ydata=data, xdata=x)
    np.testing.assert_allclose(fit.mag.parameters, np.arange(5), atol=1e-10)
    np.testing.assert_allclose(np.real(fit()), np.real(data), rtol=0, atol=1e-8)


def test_complex_reim_model():
    cmplx = mdl.ComplexRealImagModel(
        real=mdl.Polynomial(n_terms=5), imag=mdl.Polynomial(n_terms=6)
    )

    x = np.linspace(0, 1, 150)
    data = cmplx(x=x, parameters=np.arange(11))
    fit = cmplx.fit(ydata=data, xdata=x)
    np.testing.assert_allclose(fit.real.parameters, np.arange(5), atol=1e-10)
    np.testing.assert_allclose(fit.imag.parameters, np.arange(5, 11), atol=1e-10)


def test_zero_to_one_tfm():
    t = mdl.ZerotooneTransform(range=(5, 10))
    x = t.transform(np.linspace(5, 10, 10))
    assert x.min() == 0
    assert x.max() == 1


def test_model_call():
    m = mdl.Polynomial(n_terms=5)

    with pytest.raises(ValueError, match="You must supply either x or basis"):
        m(parameters=[0, 1, 2, 3, 4])

    x = np.linspace(0, 1, 10)
    assert np.allclose(
        m(x=x, parameters=[1, 2, 3], indices=[0, 1, 2]),
        m(x=x, parameters=[1, 2, 3, 4, 5], indices=[0, 1, 2]),
    )

    assert not np.allclose(
        m(x=x, parameters=[1, 2, 3], indices=[0, 1, 2]),
        m(x=x, parameters=[1, 2, 3, 4, 5], indices=[1, 2, 3]),
    )


def test_physical_lin_too_many_terms():
    pl = mdl.PhysicalLin()
    with pytest.raises(ValueError, match="too many terms"):
        pl.get_basis_term(5, np.linspace(0, 1, 10))


def test_composite_model_getattr():
    mdl1 = mdl.PhysicalLin()
    mdl2 = mdl.Polynomial(n_terms=5)

    cmp = mdl.CompositeModel(models={"lin": mdl1, "pl": mdl2})

    assert cmp["lin"] == mdl1

    with pytest.raises(KeyError):
        cmp["non-existent"]


def test_composite_with_n_terms():
    mdl1 = mdl.PhysicalLin(
        parameters=[0, 1, 2, 3, 4], xtransform=mdl.ScaleTransform(scale=1.5)
    )
    mdl2 = mdl.Polynomial(n_terms=5, parameters=[0, 1, 2, 3, 4])

    cmp = mdl.CompositeModel(models={"lin": mdl1, "pl": mdl2})

    new = cmp.with_nterms("pl", n_terms=6, parameters=[0, 1, 2, 3, 4, 5])
    assert not np.allclose(new(x=np.linspace(1, 2, 10)), cmp(x=np.linspace(1, 2, 10)))


def test_composite_roundtrip(tmpdir):
    mdl1 = mdl.PhysicalLin(
        parameters=[0, 1, 2, 3, 4], xtransform=mdl.ScaleTransform(scale=1.5)
    )
    mdl2 = mdl.Polynomial(n_terms=5, parameters=[0, 1, 2, 3, 4])
    cmp = mdl.CompositeModel(models={"lin": mdl1, "pl": mdl2})

    from edges.io.serialization import converter

    converter.unstructure(mdl1)
    converter.unstructure(mdl2)
    converter.unstructure(cmp.data_transform)

    cmp.write(tmpdir / "cmp.h5")
    new = mdl.CompositeModel.from_file(tmpdir / "cmp.h5")
    assert new == cmp


def test_complex_at():
    rl = mdl.Polynomial(parameters=[1, 2, 3])

    x = np.linspace(0, 1, 10)
    cmplx = mdl.ComplexRealImagModel(real=rl, imag=rl)
    cmplx_fixed = cmplx.at(x=x)

    rng = np.random.default_rng()
    y = rng.random(size=10) + 1j * rng.random(size=10)

    fit1 = cmplx.fit(y, xdata=x)
    fit2 = cmplx_fixed.fit(y)

    assert np.allclose(fit1.real.parameters, fit2.real.parameters)


def test_logpoly():
    m = mdl.LogPoly(n_terms=5, parameters=[1, 0, 0, 0, 0])
    x = np.linspace(1, 2, 10)
    assert m(x).shape == (10,)
    assert np.allclose(m(x), np.ones(10))
    assert np.allclose(m(x, parameters=[0, 1, 0, 0, 0]), np.log10(x))


def test_shift_transform():
    x = np.linspace(0, 1, 10)
    t = mdl.ShiftTransform(shift=1)
    assert np.allclose(t.transform(x), x - 1)


def test_parameter_covariance():
    pl = mdl.Polynomial(parameters=[1, 2, 3])
    model = pl.at(x=np.linspace(0, 1, 10))

    data = model()
    fit = mdl.ModelFit(model, ydata=data)

    cov = fit.parameter_covariance
    assert cov.shape == (3, 3)

    corr = fit.parameter_correlation
    assert corr.shape == (3, 3)
    assert np.allclose(np.diag(corr), 1)


def test_sample_from_posterior():
    pl = mdl.Polynomial(parameters=[1, 2])
    model = pl.at(x=np.linspace(0, 1, 10))

    data = model()
    fit = mdl.ModelFit(model, ydata=data)

    samples = fit.get_sample(1000)
    assert samples.shape == (1000, 2)

    mean_params = np.mean(samples, axis=0)
    assert np.allclose(mean_params, fit.model_parameters, atol=1e-1)
