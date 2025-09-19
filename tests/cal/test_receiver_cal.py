from edges.cal import Calibrator
from edges.cal.receiver_cal import perform_term_sweep


def test_term_sweep(calobs):
    calobs_opt = perform_term_sweep(
        calobs,
        max_cterms=6,
        max_wterms=8,
    )

    assert isinstance(calobs_opt, Calibrator)
