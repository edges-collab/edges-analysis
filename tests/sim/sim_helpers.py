"""Shared helpers for the simulation tests."""

import astropy_healpix as ahp
import numpy as np
from astropy import units as un
from astropy.coordinates import Galactic

from edges.sim.beams import Beam
from edges.sim.sky_models import SkyModel


def make_galaxy_sky(nside: int = 8, frequency: float = 75.0) -> SkyModel:
    """A cheap, strongly non-uniform sky with a bright Galactic-plane-like band."""
    hp = ahp.HEALPix(nside=nside, order="nested", frame=Galactic())
    coords = hp.healpix_to_skycoord(np.arange(hp.npix))
    temp = 1000 + 3000 * np.exp(-0.5 * (coords.b.deg / 10) ** 2) * (
        1 + np.cos(coords.l.rad)
    )
    return SkyModel(
        frequency=frequency,
        temperature=temp,
        coords=coords,
        healpix=hp,
        name="galaxy-like",
    )


def make_achromatic_beam(
    freqs: np.ndarray = np.arange(50.0, 71.0, 5.0),
    delta_az: float = 2.0,
    delta_el: float = 2.0,
) -> Beam:
    """A FEKO-like (dipole-ish, azimuthally asymmetric) beam with no chromaticity."""
    az = np.arange(0, 360, delta_az)
    el = np.arange(0, 90 + 0.1 * delta_el, delta_el)
    pattern = np.sin(np.radians(el))[:, None] ** 2 * (
        1 + 0.3 * np.cos(2 * np.radians(az))[None, :]
    )
    return Beam(
        frequency=freqs * un.MHz,
        azimuth=az,
        elevation=el,
        beam=np.repeat(pattern[None], len(freqs), axis=0),
        simulator="analytic",
    )
