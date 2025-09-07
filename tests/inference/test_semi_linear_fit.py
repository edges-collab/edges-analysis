import numpy as np

from edges.inference import FlattenedGaussian, SemiLinearFit
from edges.modeling import LinLog


def test_semi_linear_fit():
    freqs = np.linspace(50, 100, 100)

    fg = LinLog(parameters=[1750, 1, -1, 2, -2]).at(x=freqs)
    eor = FlattenedGaussian(freqs=freqs, params=["amp", "w"])

    fgx = fg()
    eorx = eor()["eor_spectrum"]
    data = fgx + eorx

    slf = SemiLinearFit(fg=fg, eor=eor, spectrum=data, sigma=0.1)

    best_p = slf()
    print(best_p)
    assert np.allclose(slf.get_eor(best_p.x), eorx)
    assert np.allclose(slf.fg_fit(best_p.x).evaluate(), fgx)
    assert np.allclose(slf.get_resid(best_p.x), 0)
