r"""
A module defining various sky models that can be used to generate beam corrections.

The models here are preferably *not* interpolated over frequency, as that gives more
control to us to interpolate how we wish.
"""
from __future__ import annotations

import healpy as hp
import numpy as np
from astropy import coordinates as apc
from astropy.io import fits
import logging
from astropy.utils.data import download_file
from cached_property import cached_property
from contextlib import contextmanager
import attr
from abc import ABC, abstractmethod

logger = logging.getLogger(__name__)


class IndexModel(ABC):
    """Base model for spectral index variation across the sky."""

    def __init_subclass__(cls, abstract=False, **kwargs):
        """Provide plugin architecture for index models."""
        super().__init_subclass__(**kwargs)
        if not hasattr(cls, "_plugins"):
            cls._plugins = {}
        if not abstract:
            cls._plugins[cls.__name__] = cls

    @abstractmethod
    def get_index(self, lat=None, lon=None, sky_model=None):
        """Overloadable method to compute the index at a specific sky location."""
        pass


@attr.s
class GaussianIndex(IndexModel):
    """Spectral index model that continuously changes from pole to center."""

    index_pole = attr.ib(default=2.65, converter=float)
    index_center = attr.ib(default=2.4, converter=float)
    sigma_deg = attr.ib(default=8.5, converter=float)

    def get_index(
        self,
        lat: [None, np.ndarray] = None,
        lon: [None, np.ndarray] = None,
        sky_model: [None, SkyModel] = None,
    ) -> np.ndarray:
        """Generate the index at a given sky location."""
        if lat is None:
            raise ValueError("GaussianIndex requires passing lat")

        return self.index_pole - (self.index_pole - self.index_center) * np.exp(
            -(1 / 2) * (np.abs(lat) / self.sigma_deg) ** 2
        )


@attr.s
class StepIndex(IndexModel):
    """Spectral index model with two index regions -- one around the pole."""

    index_inband = attr.ib(default=2.5, converter=float)
    index_outband = attr.ib(default=2.6, converter=float)
    band_deg = attr.ib(default=10, converter=float)

    def get_index(
        self,
        lat: [None, np.ndarray] = None,
        lon: [None, np.ndarray] = None,
        sky_model: [None, SkyModel] = None,
    ) -> np.ndarray:
        """Generate the spectral index at a given sky location."""
        if lat is None:
            raise ValueError("StepIndex requires passing lat")

        index = np.zeros(len(lat))
        index[np.abs(lat) <= self.band_deg] = self.index_inband
        index[np.abs(lat) > self.band_deg] = self.index_outband
        return index


@attr.s
class ConstantIndex(IndexModel):
    index = attr.ib(default=2.5, converter=float)

    def get_index(
        self,
        lat: [None, np.ndarray] = None,
        lon: [None, np.ndarray] = None,
        sky_model: [None, SkyModel] = None,
    ) -> np.ndarray:
        """Generate the spectral index at a given sky location."""
        return np.ones_like(lat) * self.index


class SkyModel:
    """
    A base class representing a sky model at one frequency in healpix.

    Do not use this class -- use a specific subclass that specifies an actual sky model.

    Each sky model's data is gotten from lambda, and the first time it is accessed it
    is downloaded and cached on disk using astropy.

    The primary attribute is `sky_map`, which is always in NESTED galactic co-ordinates.
    Other methods/attributes exist to help with understanding the metadata of the
    object, and interpolating in frequency.

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
        """Open the FITS file for reading."""
        fl = fits.open(self.filename)
        yield fl
        fl.close()

    @cached_property
    def sky_map(self) -> np.ndarray:
        """The sky map of this SkyModel in NESTED Galactic co-ordinates."""
        sky_map = self._get_sky_model_from_lambda(self.max_res)
        return self._process_map(sky_map, self.lonlat[0], self.lonlat[1])

    @cached_property
    def lonlat(self) -> tuple[np.ndarray, np.ndarray]:
        """The galactic latitude/longitude co-ordinates of the map."""
        with self.open() as fl:
            ordering = fl[self.header_hdu].header["ORDERING"]
            nside = int(fl[self.header_hdu].header["NSIDE"])

        return self.get_map_coords(nside=nside, nest=ordering.lower() == "nested")

    @classmethod
    def get_map_coords(cls, nside: int, nest: bool = True):
        """Get the angular co-ordinates of the map pixels."""
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
        """Get the sky coords of the map as :class:`astropy.coordinates.SkyCoord`."""
        return apc.SkyCoord(
            self.lonlat[0], self.lonlat[1], frame="galactic", unit="deg"
        )

    def _process_map(self, sky_map, lon, lat):
        """Optional over-writeable method to process the sky map before returning it."""
        return sky_map

    def at_freq(
        self, freq: float | np.ndarray, index_model: IndexModel = GaussianIndex()
    ) -> np.ndarray:
        """
        Generate the sky model at a new set of frequencies.

        Parameters
        ----------
        freq
            The frequencies at which to evaluate the model (can be a single float)
        index_model
            A spectral index model to shift to the new frequencies.

        Returns
        -------
        maps
            The healpix sky maps at the new frequencies, shape (Nsky, Nfreq).
        """
        lon, lat = self.lonlat
        index = index_model.get_index(lon, lat, self)
        f = freq / self.frequency
        Tcmb = 2.725
        scale = np.power.outer(f, -index)
        return ((self.sky_map - Tcmb) * scale + Tcmb).T


class Haslam408(SkyModel):
    url = (
        "https://lambda.gsfc.nasa.gov/data/foregrounds/haslam/"
        "lambda_haslam408_dsds.fits"
    )
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
    header_hdu = 1
    frequency = 150.0


class Guzman45(SkyModel):
    url = "https://lambda.gsfc.nasa.gov/data/foregrounds/maipu_45/MAIPU_MU_1_64.fits"
    frequency = 45.0
    header_hdu = 1
    data_name = "UNKNOWN1"

    def _process_map(self, sky_map, lon, lat):
        # Fill the hole
        sky_map[lat > 68] = np.mean(sky_map[(lat > 60) & (lat < 68)])

        return sky_map
