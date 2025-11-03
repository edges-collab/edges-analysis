r"""
A module defining various sky models that can be used to generate beam corrections.

The models here are preferably *not* interpolated over frequency, as that gives more
control to us to interpolate how we wish.
"""

import logging
import warnings
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Self, Union

import astropy_healpix as ahp
import attrs
import numpy as np
from astropy import coordinates as apc
from astropy.coordinates import Galactic
from astropy.io import fits
from astropy.utils.data import download_file

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


@attrs.define
class GaussianIndex(IndexModel):
    """Spectral index model that continuously changes from pole to center."""

    index_pole = attrs.field(default=2.65, converter=float)
    index_center = attrs.field(default=2.4, converter=float)
    sigma_deg = attrs.field(default=8.5, converter=float)

    def get_index(
        self,
        lat: np.ndarray | None = None,
        lon: np.ndarray | None = None,
        sky_model: Union["SkyModel", None] = None,
    ) -> np.ndarray:
        """Generate the index at a given sky location."""
        if lat is None:
            raise ValueError("GaussianIndex requires passing lat")

        return self.index_pole - (self.index_pole - self.index_center) * np.exp(
            -(1 / 2) * (np.abs(lat) / self.sigma_deg) ** 2
        )


@attrs.define
class StepIndex(IndexModel):
    """Spectral index model with two index regions -- one around the pole."""

    index_inband = attrs.field(default=2.5, converter=float)
    index_outband = attrs.field(default=2.6, converter=float)
    band_deg = attrs.field(default=10, converter=float)

    def get_index(
        self,
        lat: np.ndarray | None = None,
        lon: np.ndarray | None = None,
        sky_model: Union["SkyModel", None] = None,
    ) -> np.ndarray:
        """Generate the spectral index at a given sky location."""
        if lat is None:
            raise ValueError("StepIndex requires passing lat")

        index = np.zeros(len(lat))
        index[np.abs(lat) <= self.band_deg] = self.index_inband
        index[np.abs(lat) > self.band_deg] = self.index_outband
        return index


@attrs.define
class ConstantIndex(IndexModel):
    """A spectral index model with constant spectral-index."""

    index = attrs.field(default=2.5, converter=float)

    def get_index(
        self,
        lat: np.ndarray | None = None,
        lon: np.ndarray | None = None,
        sky_model: Union["SkyModel", None] = None,
    ) -> np.ndarray:
        """Generate the spectral index at a given sky location."""
        return np.ones_like(lat) * self.index


@attrs.define(kw_only=True)
class SkyModel:
    """A simple diffuse unpolarized sky model at a single frequency.

    This is simpler than pyradiosky's SkyModel class in that it doesn't deal with
    polarizations, assumes a single frequency, and doesn't know intrinsically about
    the time of observation etc.

    Parameters
    ----------
    frequency
        The frequency of the sky model in Hz.
    temperature
        The temperature of the sky model in K, shape (Nsky,).
    coords
        The coordinates of the sky model, shape (Nsky,).
    name
        The name of the sky model.
    healpix
        An astropy-healpix HEALPix object describing the pixelization of the sky model,
        if applicable.
    pixel_res
        The solid angle of each pixel in the sky model, in steradians. If a float, this
        is assumed to be the same for all pixels. If an array, it must have the same
        shape as temperature. The mean temperature of the sky is assumed to be the
        sum of the temperature array multiplied by the pixel_res array, divided
        by the sum of the pixel_res array.
    """

    frequency: float = attrs.field(converter=float, validator=attrs.validators.gt(0))
    temperature: np.ndarray = attrs.field(converter=np.asarray)
    coords: apc.SkyCoord = attrs.field()
    name: str = attrs.field(default="SkyModel")
    healpix: ahp.HEALPix | None = attrs.field(default=None, eq=False)
    pixel_res: float | np.ndarray = attrs.field(eq=attrs.cmp_using(eq=np.allclose))

    @coords.validator
    def _check_coords(self, attribute, value):
        if not isinstance(value, apc.SkyCoord):
            raise TypeError(f"coords must be an astropy SkyCoord, not {type(value)}")
        if value.ndim != 1:
            raise ValueError(f"coords must be 1D, not {value.ndim}D")
        if value.size != self.temperature.size:
            raise ValueError("coords must have same size as temperature")

    @pixel_res.default
    def _default_pixel_res(self) -> float:
        return 4 * np.pi / len(self.coords)

    @pixel_res.validator
    def _check_pixel_res(self, attribute, value):
        if isinstance(value, float | int):
            return
        if value.shape != self.temperature.shape:
            raise ValueError("pixel_res must have same shape as temperature")
        if value.dtype != float:
            raise ValueError("pixel_res must be float dtype")

    def at_freq(
        self, freq: float | np.ndarray, index_model: IndexModel = ConstantIndex()
    ) -> np.ndarray:
        """
        Generate the sky model at a new set of frequencies.

        Parameters
        ----------
        freq
            The frequencies at which to evaluate the model (can be a single float or
            an array of floats).
        index_model
            A spectral index model to shift to the new frequencies.

        Returns
        -------
        temperature
            The sky maps as numpy arrays at the new frequencies, shape (Nsky, Nfreq).
        """
        index = index_model.get_index(
            lon=self.coords.l.deg, lat=self.coords.b.deg, sky_model=self
        )
        f = freq / self.frequency
        t_cmb = 2.725
        scale = np.power.outer(f, -index)
        return ((self.temperature - t_cmb) * scale + t_cmb).T

    @classmethod
    def from_lambda(
        cls,
        url: str,
        frequency: float,
        max_nside: int = 2**100,
        min_nside: int = 2**0,
        header_hdu: int = 1,
        data_name="TEMPERATURE",
        name: str = "SkyModel",
    ) -> Self:
        """Load a sky model from the LAMBDA website."""
        fname = download_file(url, cache=True)

        with fits.open(fname) as fl:
            ordering = fl[header_hdu].header["ORDERING"]
            nside = int(fl[header_hdu].header["NSIDE"])
            temp_map = fl[header_hdu].data[data_name].flatten()

        hpix = ahp.HEALPix(nside=nside, order=ordering.lower(), frame=Galactic())
        coords = hpix.healpix_to_skycoord(np.arange(hpix.npix))

        # Downgrade the data quality to max_res to improve performance, if desired.
        if nside > max_nside:
            warnings.warn(
                "Ignoring max_nside because astropy-healpix doesn't support ud_grade",
                stacklevel=2,
            )

        if nside < min_nside:
            warnings.warn(
                "Ignoring min_nside because astropy-healpix doesn't support ud_grade",
                stacklevel=2,
            )

        return cls(
            frequency=frequency,
            temperature=temp_map,
            coords=coords,
            healpix=hpix,
            name=name,
        )

    @classmethod
    def from_latlon_grid_file(
        cls,
        fname: str | Path,
        frequency: float,
        axes=("lat", "lon"),
        frame=Galactic(),
        name: str | None = None,
    ) -> Self:
        """Load a sky model from a lat-lon grid file."""
        temperature = np.genfromtxt(fname)
        if name is None:
            name = Path(fname).stem

        assert temperature.ndim == 2, "Temperature must be 2D"
        if axes == ("lon", "lat"):
            temperature = temperature.T

        nlat, nlon = temperature.shape

        glat = np.linspace(-90, 90, nlat + 1)[:-1]
        glon = np.linspace(180, -180, nlon + 1)[:-1]
        glon, glat = np.meshgrid(glon, glat)

        coords = apc.SkyCoord(glon.flatten(), glat.flatten(), unit="deg", frame=frame)
        return cls(
            frequency=frequency,
            temperature=temperature.flatten(),
            coords=coords,
            name=name,
            pixel_res=(np.pi / 512.0)
            * (2 * np.pi / 1024.0)
            * np.cos(glat.flatten() * np.pi / 180.0),
        )

    @classmethod
    def uniform_healpix(
        cls, frequency: float, temperature: float = 1000.0, nside: int = 32
    ):
        """Create a uniform sky model."""
        hp = ahp.HEALPix(nside=nside, order="nested", frame=Galactic())
        coords = hp.healpix_to_skycoord(np.arange(hp.npix))

        return cls(
            frequency=frequency,
            temperature=temperature * np.ones(len(coords)),
            coords=coords,
            name="uniform",
            healpix=hp,
        )


def Haslam408AllNoh():  # noqa: N802
    """Return the original raw Haslam 408 MHz all-sky map.

    This is the file in Alan's repo.

    It probably came from here: https://www3.mpifr-bonn.mpg.de/cgi-bin/survey ?

    Note that by default, the pixel resolution for each pixel is set to be equivalent
    to what Alan uses in his C-Code (for the Nature Paper). This is a little inaccurate,
    as it gives exactly zero weight to pixels on the poles, instead of acknowledging
    that those pixels have a non-zero size.
    """
    fl = download_file(
        (
            "https://raw.githubusercontent.com/edges-collab/alans-pipeline/main/"
            "scripts/408-all-noh"
        ),
        cache=True,
    )
    return SkyModel.from_latlon_grid_file(fl, frequency=408.0, name="Haslam408AllNoh")


def Haslam408(min_nside=2**0, max_nside=2**100):  # noqa: N802
    """Return he Haslam 408 MHz all-sky map.

    408MHz radio continuum all-sky map of Haslam etal combines data from
    4 different surveys. The destriped/desourced (dsds) version was constructed from
    survey data obtained from the archives of the NCSA ADIL in
    galactic coordinates. The data were then transformed to a celestial coordinate
    grid and subsequently processed further in
    both the Fourier and spatial domains to mitigate baseline striping and strong
    point sources. These data were then
    interpolated onto a HEALPix projection.
    """
    return SkyModel.from_lambda(
        (
            "https://lambda.gsfc.nasa.gov/data/foregrounds/haslam/"
            "lambda_haslam408_dsds.fits"
        ),
        frequency=408.0,
        min_nside=min_nside,
        max_nside=max_nside,
        name="Haslam408",
    )


def Remazeilles408(min_nside=2**0, max_nside=2**100):  # noqa: N802
    """Return the Remazeilles 408 MHz all-sky map.

    Remazeilles et al. 2014 have re-evaluated and re-processed the rawest Haslam
    408 MHz data, available from the Max Planck
    Institute for Radioastronomy Survey Sampler website, to produce an improved
    source-subtracted and destriped 408 MHz all
    sky map. Large-scale striations associated with correlated low-frequency noise in
    the scan direction are reduced using a Fourier-based filtering technique.
    The most important improvement results from the removal of extra-galactic sources.
    An iterative combination of two techniques--two-dimensional Gaussian fitting and
    minimum curvature spline surface inpainting--are used to remove the brightest
    sources (with flux density larger than 2 Jy). Four products are made publicly
    available. They are described below. The sky maps are generated using the HEALPix
    sky pixelization scheme.
    """
    return SkyModel.from_lambda(
        (
            "https://lambda.gsfc.nasa.gov/data/foregrounds/"
            "haslam_2014/lambda_rema408_dsds.fits"
        ),
        frequency=408.0,
        min_nside=min_nside,
        max_nside=max_nside,
        name="Remazeilles408",
    )


def LW150(min_nside=2**0, max_nside=2**100):  # noqa: N802
    """Return the Landecker & Wielebinski 150 MHz all-sky map.

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
    return SkyModel.from_lambda(
        url=(
            "https://lambda.gsfc.nasa.gov/data/foregrounds/landecker_150/"
            "lambda_landecker_wielebinski_150MHz_SARAS_recalibrated_hpx_r8.fits"
        ),
        frequency=150.0,
        min_nside=min_nside,
        max_nside=max_nside,
        name="LandeckerWielebinski150",
    )


def Guzman45(min_nside=2**0, max_nside=2**100):  # noqa: N802
    """Return the Guzman 45 MHz all-sky map.

    Guzmán et al. (2011) produced an all-sky map of 45 MHz emission by combining data
    from the 45 MHz survey of Alvarez et al (1997) between declinations -90° and +19.1°
    with data from the 46.5 MHz survey of Maeda et al. (1999) between declinations
    +5° and +65°. The southern survey was made with the Maipu Radio Astronomy
    Observatory 45-MHz array with a beam of 4.6° (RA)x 2.4° (dec) FWHM.
    The northern survey was made with the Japanese Middle and Upper atmosphere
    (MU) radar array with a beam of 3.6° x 3.6°.
    """
    out = SkyModel.from_lambda(
        url="https://lambda.gsfc.nasa.gov/data/foregrounds/maipu_45/MAIPU_MU_1_64.fits",
        frequency=45.0,
        data_name="UNKNOWN1",
        min_nside=min_nside,
        max_nside=max_nside,
        name="Guzman45",
    )

    # Process the bad values in the map
    new_temp = np.zeros_like(out.temperature)
    new_temp[out.coords.dec.deg > 68.0] = np.mean(
        out.temperature[(out.coords.dec.deg < 68.0) & (out.coords.dec.deg > 60.0)]
    )
    return attrs.evolve(out, temperature=new_temp)


def WhamHAlpha(min_nside=2**0, max_nside=2**100):  # noqa: N802
    """Return the WHAM H-alpha all-sky map.

    The Wisconsin H-Alpha Mapper (WHAM) made a survey of H alpha intensity over the full
    sky with a 1 degree beam and velocity resolution of 12 km/s. The WHAM public data
    release DR1 includes a data product containing H alpha intensity integrated
    between -80 and +80 km/s LSR velocity for each observed sky position.
    The Centre d'Analyse de Données Etendues group used these data to form an all-sky
    HEALPix format map following method described in Appendix A of Paradis etal 2012
    Their HEALPix map is provided here. LAMBDA made a new version of their FITS file
    with minor changes to the headers to make them compliant with FITS standards.
    """
    return SkyModel.from_lambda(
        url="https://lambda.gsfc.nasa.gov/data/foregrounds/wham/lambda_WHAM_1_256.fits",
        frequency=457108000,
        data_name="UNKNOWN1",
        min_nside=min_nside,
        max_nside=max_nside,
        name="WHAMHAlpha",
    )


def PlanckCO(min_nside=2**0, max_nside=2**100):  # noqa: N802
    """Return the Planck CO all-sky map.

    Planck All Sky Maps are in HEALPix format, with Nside 2048, in Galactic coordinates
    and nested ordering. The signal is given in units of Kcmb for 30-353 GHz.
    Unpolarized maps contain 3 planes, labeled I_Stokes (intensity), Hits (hit count),
    and II_cov (variance of I_Stokes). Polarized maps contain 10 planes, labeled
    I_Stokes (intensity), Q_Stokes and U_Stokes (linear polarization), Hits (hit
    count), II_cov, QQ_cov, and UU_cov (variance of each of the Stokes parameters),
    and IQ_cov, IU_cov, and QU_cov (covariances between the Stokes parameters).
    """
    return SkyModel.from_lambda(
        url=(
            "https://irsa.ipac.caltech.edu/data/Planck/release_2/all-sky-maps/maps"
            "/component-maps/foregrounds/COM_CompMap_CO-commander_0256_R2.00.fits"
        ),
        frequency=115270,
        data_name="I_MEAN",
        min_nside=min_nside,
        max_nside=max_nside,
        name="PlanckCO",
    )


def HI4PI(min_nside=2**0, max_nside=2**100):  # noqa: N802
    r"""Return an HI4PI all-sky map.

    The HI 4-PI Survey (HI4PI) is a 21-cm all-sky survey of neutral atomic hydrogen.
    It is constructed from the Effelsberg Bonn HI Survey (EBHIS), made with the 100-m
    radio telescope at Effelsberg/Germany, and the Galactic All-Sky Survey (GASS),
    observed with the Parkes 64-m dish in Australia. HI4PI comprises HI line emission
    from the Milky Way. This dataset is the atomic neutral hydrogen (HI) column density
    map derived from HI4PI (:math:`|vLSR|< 600` km/s).
    """
    return SkyModel.from_lambda(
        url=(
            "https://lambda.gsfc.nasa.gov/data/foregrounds/ebv_2017/mom0_-90_90_1024"
            ".hpx.fits"
        ),
        frequency=1420.40575177,
        data_name="NHI",
        min_nside=min_nside,
        max_nside=max_nside,
        name="HI4PI",
    )
