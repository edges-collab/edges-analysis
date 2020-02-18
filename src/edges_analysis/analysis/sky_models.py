import healpy as hp
import numpy as np
from astropy import coordinates as apc
from astropy import io as fits

edges_folder = ""  # TODO: remove


def haslam_408MHz_map():
    """
    This function will return the Haslam map in NESTED Galactic Coordinates
    """

    # Loading NESTED galactic coordinates
    # -----------------------------------
    coord = fits.open(
        edges_folder
        + "/sky_models/coordinate_maps/pixel_coords_map_nested_galactic_res9.fits"
    )
    coord_array = coord[1].data
    lon = coord_array["LONGITUDE"]
    lat = coord_array["LATITUDE"]
    GALAC_COORD_object = apc.SkyCoord(
        lon, lat, frame="galactic", unit="deg"
    )  # defaults to ICRS frame

    # Loading Haslam map
    # ------------------
    haslam_map = fits.open(
        edges_folder + "/sky_models/haslam_map/lambda_haslam408_dsds.fits"
    )
    haslam408 = (haslam_map[1].data)["temperature"]

    return haslam408, lon, lat, GALAC_COORD_object


def remazeilles_408MHz_map():
    """
    This function will return the Haslam map reprocessd by Remazeilles et al. (2014). It is in
    RING Galactic Coordinates.

    This version is only destriped, not desourced.
    """

    # Loading NESTED galactic coordinates
    # -----------------------------------
    coord = fits.open(
        edges_folder
        + "/sky_models/coordinate_maps/pixel_coords_map_ring_galactic_res9.fits"
    )
    coord_array = coord[1].data
    lon_ring = coord_array["LONGITUDE"]
    lat_ring = coord_array["LATITUDE"]

    lon = hp.reorder(lon_ring, r2n=True)
    lat = hp.reorder(lat_ring, r2n=True)

    GALAC_COORD_object = apc.SkyCoord(
        lon, lat, frame="galactic", unit="deg"
    )  # defaults to ICRS frame

    # Loading Haslam map
    # ------------------
    haslam_map = fits.open(
        edges_folder + "/sky_models/haslam_map/haslam408_ds_Remazeilles2014.fits"
    )
    x = (haslam_map[1].data)["temperature"]
    haslam408_ring = x.flatten()

    haslam408 = hp.reorder(haslam408_ring, r2n=True)

    return haslam408, lon, lat, GALAC_COORD_object


def LW_150MHz_map():
    """
    This function will return the Haslam map in NESTED Galactic Coordinates
    """

    # Loading NESTED galactic coordinates
    # -----------------------------------
    coord = fits.open(
        edges_folder
        + "/sky_models/coordinate_maps/pixel_coords_map_nested_galactic_res8.fits"
    )
    coord_array = coord[1].data
    lon = coord_array["LONGITUDE"]
    lat = coord_array["LATITUDE"]
    GALAC_COORD_object = apc.SkyCoord(
        lon, lat, frame="galactic", unit="deg"
    )  # defaults to ICRS frame

    # Loading LW map
    # --------------
    LW150 = np.genfromtxt(
        edges_folder + "/sky_models/LW_150MHz_map/150_healpix_gal_nested_R8.txt"
    )  # Here we use
    # the raw map, without destriping

    return LW150, lon, lat, GALAC_COORD_object


def guzman_45MHz_map():
    """
    This function will return the Guzman map in NESTED Galactic Coordinates
    """

    # Loading NESTED galactic coordinates
    # -----------------------------------
    coord = fits.open(
        edges_folder
        + "/sky_models/coordinate_maps/pixel_coords_map_nested_galactic_res9.fits"
    )
    coord_array = coord[1].data
    lon_raw = coord_array["LONGITUDE"]
    lat_raw = coord_array["LATITUDE"]
    GALAC_COORD_object = apc.SkyCoord(
        lon_raw, lat_raw, frame="galactic", unit="deg"
    )  # defaults to ICRS frame

    # 45-MHz map. The map is in Plate Caree projection (unprojected,
    # see https://en.wikipedia.org/wiki/Equirectangular_projection)
    map45_fit = fits.open(edges_folder + "/sky_models/map_45MHz/wlb45.fits")
    map45 = map45_fit[0].data
    map45_1D = map45.flatten()

    # Coordinate grid
    lat = np.arange(-90, 90.25, 0.25)
    lon = np.arange(180, -180.25, -0.25)
    lat_2D = np.tile(lat.reshape(-1, 1), len(map45[0, :]))
    lon_2D = np.tile(lon.reshape(-1, 1), len(map45[:, 0]))
    lon_2D = lon_2D.T

    # Converting to Healpix Nside=128
    hp_pix_2D = hp.ang2pix(128, (np.pi / 180) * (90 - lat_2D), (np.pi / 180) * lon_2D)
    hp_pix = hp_pix_2D.flatten()

    # Index for pixels with data, in the format Healpix Nside=128
    unique_hp_pix, unique_hp_pix_index = np.unique(hp_pix, return_index=True)

    # Map in format Healpix Nside=128
    map45_hp = np.zeros(12 * 128 ** 2)
    map45_hp[unique_hp_pix] = map45_1D[unique_hp_pix_index]
    # assigning a characteristic value to the hole at high declinations
    map45_hp[map45_hp < 3300] = hp.UNSEEN

    # Converting map to format Healpix Nside=512 (same as Haslam map)
    map45_512 = hp.pixelfunc.ud_grade(map45_hp, nside_out=512)

    # Converting map from RING to NESTED
    m = hp.reorder(map45_512, r2n=True)

    # Loading celestial coordinates to fill in the temperature hole around the north pole
    coord = fits.open(
        edges_folder
        + "/sky_models/coordinate_maps/pixel_coords_map_nested_celestial_res9.fits"
    )
    coord_array = coord[1].data
    DEC = coord_array["LATITUDE"]

    # Filling the hole
    m[DEC > 68] = np.mean(m[(DEC > 60) & (DEC < 68)])

    return m, lon_raw, lat_raw, GALAC_COORD_object
