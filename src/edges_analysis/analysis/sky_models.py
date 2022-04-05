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

    Do not use this class -- use a specific subclass that specifies an actual sky model

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
        if max_res and hp.get_nside(temp_map) > 2**max_res:
            hp.ud_grade(temp_map, nside_out=2**max_res)

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
    """
    408MHz radio continuum all-sky map of Haslam etal combines data from
    4 different surveys. The destriped/desourced (dsds) version was constructed from
    survey data obtained from the archives of the NCSA ADIL in
    galactic coordinates. The data were then transformed to a celestial coordinate
    grid and subsequently processed further in
    both the Fourier and spatial domains to mitigate baseline striping and strong
    point sources. These data were then
    interpolated onto a HEALPix projection.
    """
    frequency = 408.0
    header_hdu = 1


class Remazeilles408(SkyModel):
    url = (
        "https://lambda.gsfc.nasa.gov/data/foregrounds/haslam_2014/haslam408_dsds_"
        "Remazeilles2014.fits"
    )
    """
    Remazeilles et al. 2014 have re-evaluated and re-processed the rawest Haslam
    408 MHz data, available from the Max Planck
    Institute for Radioastronomy Survey Sampler website, to produce an improved
    source-subtracted and destriped 408 MHz all
    sky map. Large-scale striations associated with correlated low-frequency noise in
    the scan direction are reduced using a Fourier-based filtering technique.
    The most important improvement results from the removal of extra-galactic sources.
    An iterative combination of two techniques − two-dimensional Gaussian fitting and
    minimum curvature spline surface inpainting− are used to remove the brightest
    sources (with flux density larger than 2 Jy). Four products are made publicly
    available. They are described below. The sky maps are generated using the HEALPix
    sky pixelization scheme.
    """
    frequency = 408.0
    header_hdu = 1


class LW150(SkyModel):
    url = (
        "https://lambda.gsfc.nasa.gov/data/foregrounds/landecker_150/"
        "lambda_landecker_wielebinski_150MHz_SARAS_recalibrated_hpx_r8.fits"
    )
    """
    Patra et al. 2015, provided a recalibration of the 150 MHz map based on comparison
    with absolutely calibrated sky brightness measurements between 110 and 175 MHz made
    with the SARAS spectrometer, and Monsalve et al. 2021, provided a recalibration
    based on comparison with absolutely calibrated measurements from the EDGES.
    The correction factors that were obtained are
    SARAS: scale 1.048 +- 0.008, offset -22.4 +- 8.0 K
    EDGES: scale 1.112 +- 0.023, offset 0.7 +- 6.0 K
    Monsalve et al. note that the differences may be because EDGES is a southern
    hemisphere instrument, SARAS is a northern hemisphere instrument, and the all-sky
    150 MHz map was constructed from separate southern, northern, and intermediate
    declination surveys
    """
    header_hdu = 1
    frequency = 150.0


class Guzman45(SkyModel):
    url = "https://lambda.gsfc.nasa.gov/data/foregrounds/maipu_45/MAIPU_MU_1_64.fits"
    """
    Guzmán et al. (2011) produced an all-sky map of 45 MHz emission by combining data
    from the 45 MHz survey of Alvarez et al (1997) between declinations -90° and +19.1°
    with data from the 46.5 MHz survey of Maeda et al. (1999) between declinations
    +5° and +65°. The southern survey was made with the Maipu Radio Astronomy
    Observatory 45-MHz array with a beam of 4.6° (RA)x 2.4° (dec) FWHM.
    The northern survey was made with the Japanese Middle and Upper atmosphere
    (MU) radar array with a beam of 3.6° x 3.6°.
    """
    frequency = 45.0
    header_hdu = 1
    data_name = "UNKNOWN1"

    def _process_map(self, sky_map, lon, lat):
        # Fill the hole
        sky_map[lat > 68] = np.mean(sky_map[(lat > 60) & (lat < 68)])

        return sky_map


class WhamHAlpha(SkyModel):
    url = "https://lambda.gsfc.nasa.gov/data/foregrounds/wham/lambda_WHAM_1_256.fits"
    """
    The Wisconsin H-Alpha Mapper (WHAM) made a survey of H alpha intensity over the full
    sky with a 1 degree beam and velocity resolution of 12 km/s. The WHAM public data
    release DR1 includes a data product containing H alpha intensity integrated
    between -80 and +80 km/s LSR velocity for each observed sky position.
    The Centre d'Analyse de Données Etendues group used these data to form an all-sky
    HEALPix format map following method described in Appendix A of Paradis etal 2012
    Their HEALPix map is provided here. LAMBDA made a new version of their FITS file
    with minor changes to the headers to make them compliant with FITS standards.

    """
    frequency = 457108000
    header_hdu = 1
    data_name = "UNKNOWN1"


class PlanckCO(SkyModel):
    url = (
        "https://irsa.ipac.caltech.edu/data/Planck/release_2/all-sky-maps/maps"
        "/component-maps/foregrounds/COM_CompMap_CO-commander_0256_R2.00.fits"
    )
    """
    Planck All Sky Maps are in HEALPix format, with Nside 2048, in Galactic coordinates
    and nested ordering. The signal is given in units of Kcmb for 30-353 GHz.
    Unpolarized maps contain 3 planes, labeled I_Stokes (intensity), Hits (hit count),
    and II_cov (variance of I_Stokes). Polarized maps contain 10 planes, labeled
    I_Stokes (intensity), Q_Stokes and U_Stokes (linear polarization), Hits (hit
    count), II_cov, QQ_cov, and UU_cov (variance of each of the Stokes parameters),
    and IQ_cov, IU_cov, and QU_cov (covariances between the Stokes parameters).
    """
    frequency = 115270
    header_hdu = 1
    data_name = "I_MEAN"


class HI4PI(SkyModel):
    url = (
        "https://lambda.gsfc.nasa.gov/data/foregrounds/ebv_2017/mom0_-90_90_1024"
        ".hpx.fits"
    )
    """
    The HI 4-PI Survey (HI4PI) is a 21-cm all-sky survey of neutral atomic hydrogen.
    It is constructed from the Effelsberg Bonn HI Survey (EBHIS), made with the 100-m
    radio telescope at Effelsberg/Germany, and the Galactic All-Sky Survey (GASS),
    observed with the Parkes 64-m dish in Australia. HI4PI comprises HI line emission
    from the Milky Way. This dataset is the atomic neutral hydrogen (HI) column density
    map derived from HI4PI (|vLSR|< 600 km/s).
    """
    frequency = 1420.40575177
    header_hdu = 1
    data_name = "NHI"
