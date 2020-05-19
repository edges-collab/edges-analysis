import healpy as hp
import numpy as np
from pathlib import Path
from astropy import coordinates as apc
from astropy import io as fits
from ..config import config
import pygsm
import time
import logging

logger = logging.getLogger(__name__)

MAX_NSIDE = 7  # Corresponds to about 1/2 degree squared.


def get_map_coords(reorder=False, res=9):
    coord = fits.open(
        Path(config["paths"]["sky_models"])
        / "coordinate_maps"
        / "pixel_coords_map_nested_galactic_res{}.fits".format(res)
    )
    coord_array = coord[1].data
    lon = coord_array["LONGITUDE"]
    lat = coord_array["LATITUDE"]

    if reorder:
        lon = hp.reorder(lon, r2n=True)
        lat = hp.reorder(lat, r2n=True)

    # defaults to ICRS frame
    return lon, lat, apc.SkyCoord(lon, lat, frame="galactic", unit="deg")


def haslam_408MHz_map():
    """
    This function will return the Haslam map in NESTED Galactic Coordinates
    """
    # Loading Haslam map
    # ------------------
    haslam_map = fits.open(
        Path(config["paths"]["sky_model"]) / "haslam_map" / "lambda_haslam408_dsds.fits"
    )
    return (haslam_map[1].data)["temperature"], get_map_coords()


def gsm_map():
    """
    Get the 2008 GSM in nested co-ordinates

    Returns
    -------

    """
    gsm = pygsm.GlobalSkyModel().generate(408)
    if len(gsm) > 12 * 128 ** 2:
        gsm = hp.ud_grade(gsm, nside_out=2 ** 7)

    # Get into NESTED order
    gsm = hp.reorder(gsm, r2n=True)

    lon, lat = hp.pix2ang(
        nside=int(np.sqrt(len(gsm) / 12)),
        ipix=np.arange(len(gsm)),
        lonlat=True,
        nest=True,
    )
    return gsm, (lon, lat, apc.SkyCoord(lon, lat, frame="galactic", unit="deg"))


def remazeilles_408MHz_map():
    """
    This function will return the Haslam map reprocessd by Remazeilles et al. (2014). It is in
    RING Galactic Coordinates.

    This version is only destriped, not desourced.
    """
    # Loading Haslam map
    # ------------------
    haslam_map = fits.open(
        Path(config["paths"]["sky_models"])
        / "haslam_map"
        / "haslam408_ds_Remazeilles2014.fits"
    )
    x = (haslam_map[1].data)["temperature"]
    haslam408_ring = x.flatten()

    return hp.reorder(haslam408_ring, r2n=True), get_map_coords(reorder=True)


def LW_150MHz_map():
    """
    This function will return the Haslam map in NESTED Galactic Coordinates
    """
    # Here we use the raw map, without destriping
    return (
        np.genfromtxt(
            Path(config["paths"]["sky_models"])
            / "LW_150MHz_map"
            / "150_healpix_gal_nested_R8.txt"
        ),
        get_map_coords(res=8),
    )


def guzman_45MHz_map():
    """
    This function will return the Guzman map in NESTED Galactic Coordinates
    """
    # 45-MHz map. The map is in Plate Caree projection (unprojected,
    # see https://en.wikipedia.org/wiki/Equirectangular_projection)
    map45_fit = fits.open(Path(config["paths"]["sky_models"]) / "map_45MHz/wlb45.fits")
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
    LON, DEC, apc_coord = get_map_coords()

    # Filling the hole
    m[DEC > 68] = np.mean(m[(DEC > 60) & (DEC < 68)])

    return m, (LON, DEC, apc_coord)


def interpolate_sky_to_freq(
    sky_model,
    freq_array,
    index_model: str = "gaussian",
    sigma_deg: float = 8.5,
    index_center: float = 2.4,
    index_pole: float = 2.65,
    band_deg: float = 10,
    index_inband: float = 2.5,
    index_outband: float = 2.6,
):
    sky_models = {
        "haslam": (haslam_408MHz_map, 408),
        "remazeilles": (remazeilles_408MHz_map, 408),
        "LW": (LW_150MHz_map, 150),
        "guzman": (guzman_45MHz_map, 45),
        "gsm": (gsm_map, 408),
    }
    if sky_model not in sky_models:
        raise ValueError("sky_model must be one of {}".format(sky_models.keys()))

    t = time.time()
    logger.info(f"Obtaining {sky_model} sky model.")
    map_orig, (lon, lat, galac_coord_object) = sky_models[sky_model][0]()
    logger.info(f"Took {time.time() - t} sec.")

    nu0 = sky_models[sky_model][1]

    # Scale sky map (the map contains the CMB, which has to be removed and then added back)
    if index_model == "gaussian":
        index = index_pole - (index_pole - index_center) * np.exp(
            -(1 / 2) * (np.abs(lat) / sigma_deg) ** 2
        )
    elif index_model == "step":
        index = np.zeros(len(lat))
        index[np.abs(lat) <= band_deg] = index_inband
        index[np.abs(lat) > band_deg] = index_outband
    else:
        raise ValueError("index_model must be either 'gaussian' or 'step'")

    Tcmb = 2.725
    sky_map = np.zeros((len(map_orig), len(freq_array)))
    for i in range(len(freq_array)):
        sky_map[:, i] = (map_orig - Tcmb) * (freq_array[i] / nu0) ** (-index) + Tcmb

    return sky_map, lon, lat, galac_coord_object
