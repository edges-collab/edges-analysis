"""Functions for excising RFI."""

import logging
import warnings
from abc import abstractmethod
from collections.abc import Callable
from dataclasses import dataclass, field
from functools import cached_property
from typing import Any, Self, TypeVar

import attrs
import numpy as np
import yaml
from astropy.convolution import Box1DKernel, Gaussian1DKernel, convolve
from scipy import ndimage

from .. import modeling as mdl

logger = logging.getLogger(__name__)

T = TypeVar("T")


def xrfi_explicit(
    spectrum: np.ndarray | None = None,
    *,
    freq: np.ndarray,
    flags: np.ndarray | None = None,
    rfi_file=None,
    extra_rfi=None,
) -> np.ndarray[bool]:
    """
    Excise RFI from given data using an explicitly set list of flag ranges.

    Parameters
    ----------
    spectrum
        This parameter is unused in this function.
    freq
        Frequencies, in MHz, of the data.
    flags
        Known flags.
    rfi_file : str, optional
        A YAML file containing the key 'rfi_ranges', which should be a list of 2-tuples
        giving the (min, max) frequency range of known RFI channels (in MHz). By
        default, uses a file included in `edges-analysis` with known RFI channels from
        the MRO.
    extra_rfi : list, optional
        A list of extra RFI channels (in the format of the `rfi_ranges` from the
        `rfi_file`).

    Returns
    -------
    flags : array-like
        Boolean array of the same shape as ``spectrum`` indicated which channels/times
        have flagged RFI.
    """
    if flags is None:
        if spectrum is None:
            flags = np.zeros(freq.shape, dtype=bool)
        else:
            flags = np.zeros(spectrum.shape, dtype=bool)

    rfi_freqs = []
    if rfi_file:
        with open(rfi_file) as fl:
            rfi_freqs += yaml.load(fl, Loader=yaml.FullLoader)["rfi_ranges"]

    if extra_rfi:
        rfi_freqs += extra_rfi

    for low, high in rfi_freqs:
        flags[..., (freq > low) & (freq < high)] = True

    return flags


xrfi_explicit.ndim = (1, 2, 3)


@attrs.define
class Modeler:
    """Class for modeling either data or standard deviation as a function of freq.

    This class is used for RFI excision.
    """

    def set_params(self, iteration: int) -> dict[str, Any]:
        """Set the parameters for the model for a given iteration."""
        return {}

    def init_model(self, params: dict, freqs: np.ndarray, model: T | None = None) -> T:
        """Initialze the model.

        Use this method to initialize any data that doesn't need to be updated on each
        iteration.
        """
        return model

    @abstractmethod
    def get_model(self, model, data: np.ndarray, weights: np.ndarray) -> np.ndarray:
        """Get the model for the given data and weights."""

    def get_std(self, model, resids: np.ndarray, weights: np.ndarray) -> np.ndarray:
        """Get the standard deviation for the given residuals and weights."""
        with np.errstate(divide="ignore", invalid="ignore"):
            rsq = np.log(resids**2)
        rsq[~np.isfinite(rsq)] = np.nan
        smooth_rsq = self.get_model(model, rsq, weights)
        return 1.888 * np.sqrt(np.exp(smooth_rsq))

    def stopping_condition(self, flags: np.ndarray, iteration: int) -> bool:
        """Extra stopping conditions specific to this kind of modeler."""
        return False


@attrs.define
class LinearModeler(Modeler):
    """A :class:`Modeler` that uses a linear model to fit either data or std."""

    model: mdl.Model = attrs.field(validator=attrs.validators.instance_of(mdl.Model))
    min_terms: int = attrs.field(converter=int)
    max_terms: int = attrs.field(converter=int)
    term_increase: int = 0

    @min_terms.default
    def _min_terms_default(self) -> int:
        return self.model.n_terms

    @max_terms.default
    def _max_terms_default(self):
        return self.min_terms

    def set_params(self, iteration: int) -> dict[str, Any]:
        """Set the number of terms for the model for a given iteration."""
        return {
            "nterms": min(
                self.min_terms + iteration * self.term_increase, self.max_terms
            ),
        }

    def init_model(
        self, params: dict, freqs: np.ndarray, model: mdl.FixedLinearModel | None = None
    ) -> mdl.FixedLinearModel:
        """Initialize the model at the known frequencies, and at the default terms."""
        if model is None:
            model = self.model.at(x=freqs)

        return model.with_nterms(params["nterms"])

    def get_model(
        self, model: mdl.FixedLinearModel, data: np.ndarray, weights: np.ndarray
    ) -> np.ndarray:
        """Perform a model fit and evaluate it."""
        fit = model.fit(ydata=data, weights=weights)
        return fit.evaluate()

    def stopping_condition(self, flags: np.ndarray, iteration: int) -> bool:
        """Extra stopping conditions specific to this kind of modeler.

        In this case, stop if the number of unflagged channels is less than twice
        the number of terms in the model. This is unfittable.
        """
        return np.sum(~flags) < self.model.n_terms * 2


@attrs.define
class FilterModeler(Modeler):
    """A :class:`Modeler` that uses a convolutional filter to model = data or std."""

    kernel: np.ndarray = attrs.field()

    def get_model(
        self, model: None, data: np.ndarray, weights: np.ndarray
    ) -> np.ndarray:
        """Convolve the data with the kernel to get a smoother model."""
        return convolve(
            np.where(weights > 0, data, np.nan),
            self.kernel,
            boundary="extend",
            normalize_kernel=True,
        )

    @classmethod
    def gaussian(cls, size: int) -> Self:
        """Create a Gaussian kernel."""
        return cls(kernel=Gaussian1DKernel(size))

    @classmethod
    def mean(cls, size: int) -> Self:
        """Create a mean kernel."""
        return cls(kernel=Box1DKernel(size))


@attrs.define
class MedianFilterModeler(Modeler):
    """A :class:`Modeler` that uses a median filter to model data or std."""

    size: int = attrs.field()

    def get_model(
        self, model: None, data: np.ndarray, weights: np.ndarray
    ) -> np.ndarray:
        """Perform a median filter to get a smoothed model."""
        d = np.where(weights > 0, data, np.nan)
        return ndimage.vectorized_filter(
            d, function=np.nanmedian, size=self.size, mode="reflect"
        )

    def get_std(self, model, resids: np.ndarray, weights: np.ndarray) -> np.ndarray:
        """Calculate the rolling median-absolute-deviation."""
        smooth_rsq = self.get_model(model, resids**2, weights)
        return np.sqrt(smooth_rsq) / 0.456


def xrfi_iterative(
    data: np.ndarray,
    *,
    freqs: np.ndarray,
    flags: np.ndarray | None = None,
    weights: np.ndarray | None = None,
    data_modeler: Modeler = LinearModeler(model=mdl.Fourier(n_terms=37), min_terms=37),
    std_modeler: Modeler = LinearModeler(model=mdl.Fourier(n_terms=15)),
    threshold_setter: Callable = lambda counter: 5.0,
    max_iter: int = 20,
    watershed: dict[float, int] | None = None,
    flag_if_broken: bool = True,
    init_flags: np.ndarray | None = None,
):
    """
    Run a generalized iterative RFI excision algorithm.

    This algorithm works by iteratively modeling the data and its standard deviation,
    flagging data points that are likely affected by RFI based on a z-score threshold.

    Parameters
    ----------
    data : np.ndarray
        The input data to be processed.
    freqs : np.ndarray
        The frequency channels corresponding to the data.
    flags : np.ndarray | None
        Initial flags for the data points. If this is defined, the output flags will
        *at least* include these flags.
    weights : np.ndarray | None
        Weights for the data points. If not given, use uniform weights of one. If given,
        zero weights will be treated the same as flags.
    data_modeler : Modeler
        The modeler to use for the data.
    std_modeler : Modeler
        The modeler to use for the standard deviation.
    threshold_setter : Callable
        A callable that sets the threshold for flagging on each iteration.
        This callable should take the current iteration number as input and return the
        threshold value to use for that iteration.
    max_iter : int
        The maximum number of iterations to run.
    watershed : dict[float, int] | None
        Parameters for watershed flagging. Each key is a threshold value, and each value
        is the number of surrounding channels to include in the flagging. The threshold
        value is multiplied by the threshold of the basic z-score threshold. That is, it
        is not a number of sigma, but is a multiple of a number of sigma.
    flag_if_broken : bool
        Whether to flag an entire integration if the iterative process stops due to
        either hitting the maximum number of iterations, or hitting some model-specific
        condition without convergence of the flags.
    init_flags : np.ndarray | None
        Initial flags for the data points. If given, the initial iteration will use
        these flags, but they will be updated in subsequent iterations. This can be
        used to flag out regions of known likely RFI that might poorly affect the
        first model.

    Returns
    -------
    np.ndarray
        The flags for the data points.
    IterativeXRFIInfo
        Information about the iterative RFI excision process.
    """
    assert data.ndim == 1
    assert freqs.ndim == 1
    if len(freqs) != len(data):
        raise ValueError("freq and spectrum must have the same length")

    n_flags_changed = 1
    counter = 0

    # Set up a few lists that we can update on each iteration to return info to the user
    n_flags_changed_list = []
    total_flags_list = []
    model_params_list = []
    std_params_list = []
    thresholds = []
    std_list = []
    model_list = []
    flag_list = []

    # Initialize some flags, or set them equal to the input
    orig_flags = ~np.isfinite(data)
    if flags is not None:
        orig_flags |= flags

    flags = orig_flags.copy()

    if init_flags is not None:
        flags = flags | init_flags

    orig_weights = np.ones_like(data) if weights is None else weights.copy()

    # Iterate until either no flags are changed between iterations, or we get to the
    # requested maximum iterations, or until we have too few unflagged data to fit
    # appropriately. keep iterating
    n_flags_changed_all = [1]
    modelcls = None
    std_modelcls = None

    while counter < max_iter:
        weights = np.where(flags, 0, orig_weights)
        threshold = threshold_setter(counter)
        model_params = data_modeler.set_params(counter)
        modelcls = data_modeler.init_model(model_params, freqs, model=modelcls)
        model = data_modeler.get_model(modelcls, data=data, weights=weights)

        resid = data - model
        std_params = std_modeler.set_params(counter)
        std_modelcls = std_modeler.init_model(std_params, freqs, model=std_modelcls)
        std_model = std_modeler.get_std(std_modelcls, resids=resid, weights=weights)

        zscore = resid / std_model

        new_flags = orig_flags | (zscore > threshold)

        # Apply a watershed -- assume surrounding channels will succumb to RFI.
        if watershed is not None:
            new_flags |= _apply_watershed(new_flags, watershed, zscore / threshold)

        n_flags_changed_all = [
            np.sum(flags_f ^ new_flags) for flags_f in [*flag_list, flags]
        ]
        n_flags_changed = n_flags_changed_all[-1]

        flags = new_flags.copy()

        counter += 1

        thresholds.append(threshold)

        # Append info to lists for the user's benefit
        n_flags_changed_list.append(n_flags_changed)
        total_flags_list.append(np.sum(flags))
        flag_list.append(flags)
        model_params_list.append(model_params)
        std_params_list.append(std_params)
        std_list.append(std_model)
        model_list.append(model)

        if (
            n_flags_changed == 0
            and model_params == model_params_list[-1]
            and std_params == std_params_list[-1]
        ):
            logger.info(f"Converged after {counter} iterations.")
            break

        if np.sum(~flags) == 0:
            logger.info(f"All data flagged after {counter} iterations.")
            break

        if data_modeler.stopping_condition(
            flags, counter
        ) or std_modeler.stopping_condition(flags, counter):
            logger.info(
                f"Model-specific stopping condition met after {counter} iterations."
            )
            break

    if counter == max_iter and max_iter > 1 and n_flags_changed > 0:
        warnings.warn(
            f"max iterations ({max_iter}) reached, not all RFI might have been caught.",
            stacklevel=2,
        )
        if flag_if_broken:
            flags[:] = True

    elif data_modeler.stopping_condition(
        flags, counter
    ) or std_modeler.stopping_condition(flags, counter):
        warnings.warn(
            "Termination of iterative loop for model-specific reasons",
            stacklevel=2,
        )
        if flag_if_broken:
            flags[:] = True

    return (
        flags,
        IterativeXRFIInfo(
            n_flags_changed=n_flags_changed_list,
            total_flags=total_flags_list,
            data_models=model_list,
            model_params=model_params_list,
            thresholds=thresholds,
            std_params=std_params_list,
            stds=std_list,
            x=freqs,
            data=data,
            flags=flag_list,
        ),
    )


xrfi_iterative.ndim = (1,)


@dataclass
class IterativeXRFIInfo:
    """A simple object representing the information returned by :func:`model_filter`."""

    n_flags_changed: list[int]
    total_flags: list[int]
    model_params: list[dict]
    data_models: list[np.ndarray]
    std_params: list[dict]
    thresholds: list[float]
    stds: list[np.ndarray[float]]
    x: np.ndarray
    data: np.ndarray
    flags: list[np.ndarray[bool]]

    @property
    def n_iters(self) -> int:
        """Get the number of iterations."""
        return len(self.model_params)

    def get_model(self, indx: int = -1):
        """Get the model values."""
        return self.data_models[indx]

    def get_residual(self, indx: int = -1):
        """Get the residuals."""
        return self.data - self.get_model(indx)

    def get_std_model(self, indx: int = -1):
        """Get the *model* of the absolute residuals."""
        return self.stds[indx](self.x)


@dataclass
class ModelFilterInfoContainer:
    """A container of :class:`ModelFilterInfo` objects.

    This is almost a perfect drop-in replacement for a singular :class:`ModelFilterInfo`
    instance, but combines a number of them together seamlessly. This can be useful if
    several sub-models were fit to one long stream of data.
    """

    models: list[IterativeXRFIInfo] = field(default_factory=list)

    def append(self, model: IterativeXRFIInfo) -> Self:
        """Create a new object by appending a set of info to the existing."""
        assert isinstance(model, IterativeXRFIInfo)
        models = [*self.models, model]
        return ModelFilterInfoContainer(models)

    @cached_property
    def x(self):
        """The data coordinates."""
        return np.concatenate(tuple(model.x for model in self.models))

    @cached_property
    def data(self):
        """The raw data that was filtered."""
        return np.concatenate(tuple(model.data for model in self.models))

    @cached_property
    def flags(self):
        """The returned flags on each iteration."""
        return np.concatenate(tuple(model.flags for model in self.models))

    @cached_property
    def n_iters(self):
        """The number of iterations of the filtering."""
        return max(model.n_iters for model in self.models)

    @cached_property
    def n_flags_changed(self):
        """The number of flags changed on each filtering iteration."""
        return [
            sum(
                model.n_flags_changed[min(i, model.n_iters - 1)]
                for model in self.models
            )
            for i in range(self.n_iters)
        ]

    @cached_property
    def total_flags(self):
        """The total number of flags after each iteration."""
        return [
            sum(model.total_flags[min(i, model.n_iters - 1)] for model in self.models)
            for i in range(self.n_iters)
        ]

    def get_model(self, indx: int = -1):
        """Get the model values."""
        assert indx >= -1
        return np.concatenate(
            tuple(
                model.get_model(min(indx, model.n_iters - 1)) for model in self.models
            )
        )

    def get_residual(self, indx: int = -1):
        """Get the residual values."""
        assert indx >= -1
        return np.concatenate(
            tuple(
                model.get_residual(min(indx, model.n_iters - 1))
                for model in self.models
            )
        )

    def get_absres_model(self, indx: int = -1):
        """Get the *model* of the absolute residuals."""
        assert indx >= -1
        return np.concatenate(
            tuple(
                model.get_absres_model(min(indx, model.n_iters - 1))
                for model in self.models
            )
        )

    @cached_property
    def thresholds(self):
        """The threshold at each iteration."""
        for model in self.models:
            if model.n_iters == self.n_iters:
                break

        return model.thresholds

    @cached_property
    def stds(self):
        """The standard deviations at each datum for each iteration."""
        return [
            np.concatenate(
                tuple(model.stds[min(indx, model.n_iters - 1)] for model in self.models)
            )
            for indx in self.n_iters
        ]


def xrfi_iterative_sliding_window(
    spectrum: np.ndarray,
    *,
    freqs: np.ndarray,
    model: mdl.Model,
    flags=None,
    window_frac: int = 16,
    min_window_size: int = 10,
    max_iter: int = 100,
    threshold: float = 2.5,
    watershed: dict | None = None,
    reflag_thresh: float = 1.01,
    fit_kwargs: dict | None = None,
    weights: np.ndarray | None = None,
):
    """
    Flag RFI using a model fit and a sliding RMS window.

    This function is algorithmically the same as that used in Bowman+2018.
    The differences between this and :func:`xrfi_model` (which is the recommended
    function to use) are:

    * This does flagging *inside* the sliding window  -- i.e. once you move the
      window up by one channel, the flags can be different in the previous bins.
      This is a bit strange, since it makes the process more non-linear. If you
      were to start from the top of the band and slide the window down, you'd
      get different results.
    * The watershedding (flagging channels around the "bad" one) only happens
      if the main central channel is far enough away from the edges of the band.
    * It only flags positive outliers.

    Parameters
    ----------
    spectrum : array-like
        The 1D spectrum to flag.
    freq
        The frequencies associated with the spectrum.
    model : :class:`edges_cal.modelling.Model`
        The model to fit to the spectrum to get residuals.
    flags : array-like, optional
        The initial flags to use. If not given, all channels are unflagged.
    window_frac : int, optional
        The size of the sliding window as a fraction of the number of channels (i.e.
        the final window is int(Nchannels / window_frac) in size).
    min_window_size : int, optional
        The minimum size of the sliding window, in number of channels.
    max_iter : int, optional
        The maximum number of iterations to perform.
    threshold : float, optional
        The threshold for flagging a channel. The threshold is the number of standard
        deviations the residuals are from zero.
    watershed : dict, optional
        The parameters for the watershedding algorithm. If not given, no watershedding
        is performed. Each key should be a float that specifies the number of
        threshold*stds away from zero that a channel should be flagged. The value
        should be the number of channels to flag on either side of the flagged channel
        for that threshold. For example, ``{3: 2}`` would flag 2 channels on either
        side of any channel that is 3*threshold standard deviations away from zero.
    reflag_thresh : float, optional
        The basic algorithm has "memory", i.e. if a channel is flagged in one iteration,
        it will remain flagged for all following iterations, even if it is no longer
        an outlier for the updated model. This parameter allows you to re-consider a
        flag on a later iteration, if it was originally flagged at less than
        ``reflag_thresh`` times the threshold. This can improve conformity to the
        results of Bowman+2018, because the model fits are very slightly different
        between the codes used, but it is very difficult to predict exactly how
        the parameter will affect the results.
    fit_kwargs : dict, optional
        Any additional keyword arguments to pass to the model fit. Use the key "method"
        with value "alan-qrd" for the closest match to the Bowman+2018 code.

    Returns
    -------
    flags : array-like
        Boolean array of the same shape as ``spectrum`` indicated which channels/times
        have flagged RFI.
    info : ModelFilterInfo
        A :class:`ModelFilterInfo` object containing information about the fit at
        each iteration.
    """
    fmod = model.at(x=freqs) if not isinstance(model, mdl.FixedLinearModel) else model

    fit_kwargs = fit_kwargs or {}

    if flags is None:
        flags = np.zeros(len(spectrum), dtype=bool)

    weights = (~flags).astype(float) if weights is None else np.where(flags, 0, weights)

    orig_weights = weights.copy()

    n = len(spectrum)
    m = max(n // window_frac, min_window_size)
    prev_n_flags = 0

    n_flags_changed_list = []
    total_flags_list = []
    model_list = []
    std_list = []
    flags_list = []
    potential_reflags = set()

    for it in range(max_iter):
        # TODO: pass through fit_kwargs
        fit = fmod.fit(ydata=spectrum, weights=weights, **fit_kwargs)

        model_list.append(fit.evaluate())

        rms = np.zeros(n)
        avs = np.zeros(n)
        for i in range(n):
            rng = slice(max(i - m, 0), min(n, i + m + 1))
            size = np.sum(weights[rng])
            av = np.sum(fit.residual[rng] * weights[rng]) / size

            rms[i] = np.sqrt(
                np.sum((fit.residual[rng] - av) ** 2 * weights[rng]) / size
            )
            avs[i] = av

            # Now while *INSIDE* the loop over frequencies, apply new flags.
            nsig = fit.residual[i] / (threshold * rms[i])

            # If this channel was previously flagged, but only *just*,
            # give it a chance to get un-flagged. This is useful when
            # trying to reproduce Alan's results, because the model fit
            # on the first iteration is much harder to get the same as Alan.
            if i in potential_reflags and nsig <= 1:
                weights[i] = 1  # unflag

                if watershed:
                    for mult, nbins in watershed.items():
                        if mult < reflag_thresh and i + nbins < n and i - nbins >= 0:
                            weights[i - nbins : i + nbins + 1] = orig_weights[
                                i - nbins : i + nbins + 1
                            ]
                potential_reflags.remove(i)

            if nsig > 1:
                weights[i] = 0

                if nsig < reflag_thresh and it < 2:
                    potential_reflags.add(i)

                if watershed:
                    for mult, nbins in watershed.items():
                        if nsig > mult and i + nbins < n and i - nbins >= 0:
                            weights[i - nbins : i + nbins + 1] = 0
        n_flags = np.sum(weights == 0)
        std_list.append(rms)
        n_flags_changed_list.append(n_flags - prev_n_flags)
        total_flags_list.append(n_flags)
        flags_list.append(~(weights.astype(bool)))

        if n_flags <= prev_n_flags:
            break

        prev_n_flags = n_flags

    return (
        ~(weights.astype(bool)),
        IterativeXRFIInfo(
            n_flags_changed=n_flags_changed_list,
            total_flags=total_flags_list,
            model_params=[],
            data_models=model_list,
            std_params=[],
            stds=std_list,
            thresholds=[threshold] * it,
            x=freqs,
            data=spectrum,
            flags=flags_list,
        ),
    )


xrfi_iterative_sliding_window.ndim = (1,)


def xrfi_watershed(
    spectrum: np.ndarray | None = None,
    *,
    freqs: np.ndarray | None = None,
    flags: np.ndarray | None = None,
    weights: np.ndarray | None = None,
    tol: float | tuple[float] = 0.5,
    inplace=False,
):
    """Apply a watershed over frequencies and times for flags.

    Make sure that times/freqs with many flags are all flagged.

    Parameters
    ----------
    spectrum
        Not used in this routine.
    flags : ndarray of bool
        The existing flags.
    tol : float or tuple
        The tolerance -- i.e. the fraction of entries that must be flagged before
        flagging the whole axis. If a tuple, the first element is for the frequency
        axis, and the second for the time axis.
    inplace : bool, optional
        Whether to update the flags in-place.

    Returns
    -------
    ndarray :
        Boolean array of flags.
    dict :
        Information about the flagging procedure (empty for this function)
    """
    if flags is None:
        if weights is not None:
            flags = ~(weights.astype(bool))
        else:
            raise ValueError("You must provide flags as an ndarray")

    if weights is not None:
        flags |= weights <= 0

    fl = flags if inplace else flags.copy()

    if not hasattr(tol, "__len__"):
        tol = (tol, tol)

    freq_coll = np.sum(flags, axis=-1)
    freq_mask = freq_coll > tol[0] * flags.shape[1]
    fl[freq_mask] = True

    if flags.ndim == 2:
        time_coll = np.sum(fl, axis=0)
        time_mask = time_coll > tol[1] * flags.shape[0]
        fl[:, time_mask] = True

    return fl, {}


xrfi_watershed.ndim = (1, 2)


def _apply_watershed(
    flags: np.ndarray,
    watershed: dict[float, int],
    zscore_thr_ratio: np.ndarray,
):
    watershed_flags = np.zeros_like(flags)

    for thr, nw in sorted(watershed.items()):
        this_flg = zscore_thr_ratio > thr

        for i in range(1, nw + 1):
            watershed_flags[i:] |= this_flg[:-i]
            watershed_flags[:-i] |= this_flg[i:]

    return watershed_flags
