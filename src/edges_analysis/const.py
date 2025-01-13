"""Useful constants."""

from astropy import coordinates as apc
from astropy import units as apu
from pygsdata import KNOWN_TELESCOPES, Telescope

KNOWN_TELESCOPES["edges-low-alan"] = Telescope(
    name="edges-low-alan",
    location=apc.EarthLocation(lat=-26.7 * apu.deg, lon=116.5 * apu.deg),
    pols=("xx",),
    integration_time=13.0 * apu.s,
    x_orientation=0.0 * apu.deg,
)

KNOWN_TELESCOPES["edges3-devon"] = Telescope(
    name="edges3-devon",
    location=apc.EarthLocation(lat=75.433 * apu.deg, lon=-89.81 * apu.deg),
    pols=("xx",),
    integration_time=13.0 * apu.s,
    x_orientation=0.0 * apu.deg,
)

KNOWN_TELESCOPES["edges3"] = Telescope(
    name="edges3",
    location=apc.EarthLocation(lat=-26.7 * apu.deg, lon=116.5 * apu.deg),
    pols=("xx",),
    integration_time=13.0 * apu.s,
    x_orientation=0.0 * apu.deg,
)


galactic_centre_lst = 17 + (45 / 60) + (40.04 / (60 * 60))
absolute_zero = (0 * apu.deg_C).to(apu.K, equivalencies=apu.temperature()).value
edges_location = KNOWN_TELESCOPES["edges-low"].location
