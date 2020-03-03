from astropy import coordinates as apc
from astropy import units as apu

edges_lat_deg = -26.714778
edges_lon_deg = 116.605528
edges_location = apc.EarthLocation(
    lat=edges_lat_deg * apu.deg, lon=edges_lon_deg * apu.deg
)
