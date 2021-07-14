"""Useful constants."""
from astropy import coordinates as apc
from astropy import units as apu

edges_lat_deg = -26.714778
edges_lon_deg = 116.605528
edges_location = apc.EarthLocation(
    lat=edges_lat_deg * apu.deg, lon=edges_lon_deg * apu.deg
)
galactic_centre_lst = 17 + (45 / 60) + (40.04 / (60 * 60))
absolute_zero = (0 * apu.deg_C).to(apu.K, equivalencies=apu.temperature()).value
