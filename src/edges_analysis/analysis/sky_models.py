r"""
A module defining various sky models that can be used to generate beam corrections.

The models here are preferably *not* interpolated over frequency, as that gives more
control to us to interpolate how we wish.
"""

import healpy as hp
import numpy as np
from astropy import coordinates as apc
from astropy.io import fits
import logging
from astropy.utils.data import download_file
from cached_property import cached_property
from typing import Tuple
from contextlib import contextmanager

logger = logging.getLogger(__name__)


class SkyModel:
    """
    A base class representing a sky model at one frequency in healpix.

    Do not use this class -- use a specific subclass that specifies an actual sky model.

    Each sky model's data is gotten from lambda, and the first time it is accessed it
    is downloaded and cached on disk using astropy.

    The primary attribute is `sky_map`, which is always in NESTED galactic co-ordinates.
    Other methods/attributes exist to help with understanding the metadata of the object,
    and interpolating in frequency.

    Examples
    --------
    >>> from edges_analysis.sky_models import Haslam408
    >>> import healpy as hp
    >>> haslam = Haslam408()
    >>> hp.mollview(np.log10(haslam.sky_map), nest=True)

    """

    url = None  # url of the file to download
    header_hdu = 0  # the index of the HDU of the FITS file containing header info
    frequency = None  # frequency at which the sky model is defined.
    data_name = "TEMPERATURE"

    def __init__(self, max_res=None):
        self.max_res = max_res
        self.filename = download_file(self.url, cache=True)

    @contextmanager
    def open(self):
        fl = fits.open(self.filename)
        yield fl
        fl.close()

    @cached_property
    def sky_map(self) -> np.ndarray:
        """The sky map of this SkyModel in NESTED Galactic co-ordinates."""
        sky_map = self._get_sky_model_from_lambda(self.max_res)
        return self._process_map(sky_map, self.lonlat[0], self.lonlat[1])

    @cached_property
    def lonlat(self) -> Tuple[np.ndarray, np.ndarray]:
        """The galactic latitude/longitude co-ordinates of the map."""
        with self.open() as fl:
            ordering = fl[self.header_hdu].header["ORDERING"]
            nside = int(fl[self.header_hdu].header["NSIDE"])

        return self.get_map_coords(nside=nside, nest=ordering.lower() == "nested")

    @classmethod
    def get_map_coords(cls, nside: int, nest: bool = True):
        return hp.pix2ang(
            nside, np.arange(hp.nside2npix(nside)), lonlat=True, nest=nest
        )

    @classmethod
    def _get_sky_model_from_lambda(cls, max_res: [None, int] = None) -> np.ndarray:
        with fits.open(download_file(cls.url, cache=True)) as fl:
            ordering = fl[cls.header_hdu].header["ORDERING"]

            # Need to flatten because sometimes they are over multiple axes... ??
            temp_map = fl[cls.header_hdu].data[cls.data_name].flatten()

        if ordering.lower() == "ring":
            temp_map = hp.reorder(temp_map, r2n=True)

        # Downgrade the data quality to max_res to improve performance, if desired.
        if max_res and hp.get_nside(temp_map) > 2 ** max_res:
            hp.ud_grade(temp_map, nside_out=2 ** max_res)

        return temp_map

    def get_sky_coords(self):
        return apc.SkyCoord(
            self.lonlat[0], self.lonlat[1], frame="galactic", unit="deg"
        )

    def _process_map(self, sky_map, lon, lat):
        """Optional over-writeable method to process the sky map before returning it."""
        return sky_map

    def interpolate_freq(
        self,
        freq_array,
        index_model: str = "gaussian",
        sigma_deg: float = 8.5,
        index_center: float = 2.4,
        index_pole: float = 2.65,
        band_deg: float = 10,
        index_inband: float = 2.5,
        index_outband: float = 2.6,
    ):
        lon, lat = self.lonlat

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
        sky_map = np.zeros((len(self.sky_map), len(freq_array)))
        for i in range(len(freq_array)):
            sky_map[:, i] = (self.sky_map - Tcmb) * (
                freq_array[i] / self.frequency
            ) ** (-index) + Tcmb

        return sky_map


class Haslam408(SkyModel):
    url = "https://lambda.gsfc.nasa.gov/data/foregrounds/haslam/lambda_haslam408_dsds.fits"
    frequency = 408.0
    header_hdu = 1


class Remazeilles408(SkyModel):
    url = (
        "https://lambda.gsfc.nasa.gov/data/foregrounds/haslam_2014/haslam408_dsds_"
        "Remazeilles2014.fits"
    )
    frequency = 408.0
    header_hdu = 1


class LW150(SkyModel):
    url = (
        "https://lambda.gsfc.nasa.gov/data/foregrounds/landecker_150/"
        "lambda_landecker_wielebinski_150MHz_SARAS_recalibrated_hpx_r8.fits"
    )


class Guzman45(SkyModel):
    url = "https://lambda.gsfc.nasa.gov/data/foregrounds/maipu_45/MAIPU_MU_1_64.fits"
    freq = 45.0
    header_hdu = 1
    data_name = "UNKNOWN1"

    def _process_map(self, sky_map, lon, lat):
        # Fill the hole
        sky_map[lat > 68] = np.mean(sky_map[(lat > 60) & (lat < 68)])

        return sky_map


# def gsm_map():
#     """
#     Get the 2008 GSM in nested co-ordinates
#
#     Returns
#     -------
#
#     """
#     gsm = pygsm.GlobalSkyModel().generate(408)
#     if len(gsm) > 12 * 128 ** 2:
#         gsm = hp.ud_grade(gsm, nside_out=2 ** 7)
#
#     # Get into NESTED order
#     gsm = hp.reorder(gsm, r2n=True)
#
#     lon, lat = hp.pix2ang(
#         nside=int(np.sqrt(len(gsm) / 12)),
#         ipix=np.arange(len(gsm)),
#         lonlat=True,
#         nest=True,
#     )
#     return gsm, (lon, lat, apc.SkyCoord(lon, lat, frame="galactic", unit="deg"))
