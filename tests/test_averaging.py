from edges_analysis.analysis import tools
import numpy as np
from edges_cal import modelling as mdl


def test_non_stationary_avg_1d_shape():
    x = np.linspace(0, 1, 20)
    y = x ** 2

    avg = tools.non_stationary_weighted_average(data=y, x=x, n_terms=3)

    assert isinstance(avg, float)
    assert np.isclose(avg, np.mean(y))


def test_non_stationary_avg_2d_shape():
    x = np.linspace(0, 1, 20)
    y = np.array([x ** 2 for _ in range(4)])

    avg = tools.non_stationary_weighted_average(data=y, x=x, n_terms=3)

    assert avg.shape == (4,)
    assert np.allclose(avg, np.mean(y))


def test_non_stationary_avg_1d_weights():
    x = np.linspace(0, 1, 50)
    y = x ** 2

    weights = np.ones(len(x), dtype=bool)
    weights[::4] = 0

    avg = tools.non_stationary_weighted_average(data=y, x=x, n_terms=3, weights=weights)
    assert isinstance(avg, float)
    assert np.isclose(avg, np.mean(y))


def test_non_stationary_avg_1d_nonzero_weights():
    x = np.linspace(0, 1, 50)
    y = x ** 2

    weights = np.linspace(1, 4, len(x))
    #    weights[::4] = 0

    avg = tools.non_stationary_weighted_average(data=y, x=x, n_terms=3, weights=weights)
    assert isinstance(avg, float)
    assert np.isclose(avg, np.mean(y))


def test_non_stationary_avg_1d_with_model_fit():
    x = np.linspace(0, 1, 50)
    y = x ** 2

    def model(x):
        return x ** 2

    weights = np.linspace(1, 4, len(x))

    avg = tools.non_stationary_weighted_average(
        data=y, x=x, n_terms=3, weights=weights, model_fit=model
    )
    assert isinstance(avg, float)
    assert np.isclose(avg, np.mean(y))


def test_non_stationary_avg_1d_with_model():
    x = np.linspace(0, 1, 50)
    y = x ** 2

    model = mdl.Polynomial(default_x=x, n_terms=3)

    weights = np.linspace(1, 4, len(x))

    avg = tools.non_stationary_weighted_average(
        data=y, x=x, n_terms=3, weights=weights, model=model
    )
    assert isinstance(avg, float)
    assert np.isclose(avg, np.mean(y))
