"""Physical and behavioural tests of the antenna beam factor."""

import numpy as np
import pytest
from astropy import units as un
from astropy.coordinates import Longitude
from sim_helpers import make_achromatic_beam

from edges import modeling as mdl
from edges.sim import compute_antenna_beam_factor
from edges.sim.antenna_beam_factor import BeamFactor
from edges.sim.sky_models import ConstantIndex

FAST = {
    "lst_progress": False,
    "freq_progress": False,
    "beam_smoothing": False,
    "interp_kind": "linear",
    "index_model": ConstantIndex(index=2.5),
}


def _achromatic_bf(galaxy_sky, *, sky_at_reference_frequency: bool) -> BeamFactor:
    return compute_antenna_beam_factor(
        beam=make_achromatic_beam(np.arange(50.0, 101.0, 5.0)),
        sky_model=galaxy_sky,
        lsts=Longitude([0.0, 6.0, 12.0, 18.0] * un.hour),
        reference_frequency=75 * un.MHz,
        sky_at_reference_frequency=sky_at_reference_frequency,
        use_astropy_azel=False,
        **FAST,
    )


def test_achromatic_beam_factor_is_unity_eq_a1(galaxy_sky):
    """With sky_at_reference_frequency=False, an achromatic beam gives BF == 1."""
    bf = _achromatic_bf(galaxy_sky, sky_at_reference_frequency=False)
    np.testing.assert_allclose(bf.antenna_temp / bf.antenna_temp_ref, 1.0, rtol=1e-12)
    np.testing.assert_allclose(
        bf.get_beam_factor(mdl.Polynomial(n_terms=4)), 1.0, rtol=1e-8
    )


@pytest.mark.xfail(
    strict=True,
    reason=(
        "SIM-1: with sky_at_reference_frequency=True the numerator uses T_sky(nu) "
        "instead of T_sky(nu_ref) (Sims+23 Eq. 4), so an achromatic beam gives "
        "(nu/nu_ref)^-beta instead of 1; fix pending (result-changing)"
    ),
)
def test_achromatic_beam_factor_is_unity_eq_4(galaxy_sky):
    """With sky_at_reference_frequency=True, an achromatic beam also gives BF == 1."""
    bf = _achromatic_bf(galaxy_sky, sky_at_reference_frequency=True)
    np.testing.assert_allclose(
        bf.get_beam_factor(mdl.Polynomial(n_terms=4)), 1.0, rtol=1e-8
    )
