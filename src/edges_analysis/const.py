"""Useful constants."""
from astropy import coordinates as apc
from astropy import units as apu

edges_lat_deg = -26.714778
edges_lon_deg = 116.605528
devon_island_lat_deg = 75.433
devon_island_lon_deg = -89.81
edges_location = apc.EarthLocation(
    lat=edges_lat_deg * apu.deg, lon=edges_lon_deg * apu.deg
)
galactic_centre_lst = 17 + (45 / 60) + (40.04 / (60 * 60))
absolute_zero = (0 * apu.deg_C).to(apu.K, equivalencies=apu.temperature()).value

KNOWN_LOCATIONS = {
    "edges": edges_location,
    "alan-edges": apc.EarthLocation(lat=-26.7 * apu.deg, lon=116.5 * apu.deg),
    "devon-island": apc.EarthLocation(
        lat=devon_island_lat_deg * apu.deg, lon=devon_island_lon_deg * apu.deg
    ),
}
