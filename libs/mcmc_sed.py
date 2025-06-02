""" Functions supporting the MCMC analysis of SED curves to estimate dEB stellar masses """
from typing import Callable, Tuple

import numpy as np

from deblib.constants import c, h, k_B
from deblib.vmath import exp

# pylint: disable=invalid-name, too-many-arguments, too-many-positional-arguments

# TODO: need a better way of handling this - static cctor?
from libs.mistisochrones import MistIsochrones
mist_isos = MistIsochrones(metallicities=[0])


def norm_blackbody_model(x: np.ndarray,
                         Teff1: float,
                         Teff2: float,
                         logg1: float=None,
                         logg2: float=None) \
        -> Tuple[np.ndarray, np.ndarray]:
    """
    Model a SED on the Planck blackbody function of two stars of the given effective temperatures.
    The returned model is the min-max normalized sum the the two stars' fluxes.

    :x: the frequencies at which the SED observations are required
    :Teff1: the effective temperature of star 1 in K
    :Teff2: the effective temperature of star 2 in K
    :logg1: the surface gravity of star 1 in log(cgs) (not used)
    :logg2: the surface gravity of star 2 in log(cgs) (not used)
    :returns: the normalized summed fluxes at x for the two stars
    """
    # pylint: disable=unused-argument
    def bb_spec_brightness(teff):
        """
        Calculate the BB spectral brightness at effective temp T and frequencies x with;
        B(x, T) = (2hx^3)/c^2 * 1/(exp(hx/kT)-1)  [W / m^2 / Hz / sr]

        Params teff and x are floats in units of K and Hz, respectively
        """
        pt1 = (2 * h * x**3) / c**2
        pt2 = exp((h * x) / (k_B * teff)) - 1
        return pt1 / pt2
    return min_max_normalize(np.add(bb_spec_brightness(Teff1), bb_spec_brightness(Teff2)))


def ln_like(x: np.ndarray,
            y: np.ndarray,
            y_err: np.ndarray,
            Teff1: float,
            Teff2: float,
            logg1: float,
            logg2: float,
            model_func: Callable=norm_blackbody_model):
    """
    The MCMC likelihood function, which returns a comparison of the x & y (+/-y_err)
    observations with the model produced by the model_func based on the Teffs and loggs.

    Is calculated as 1/2 Σ((y - y_model)/y_err)^2
    
    :x: the frequencies at which the SED observations were made
    :y: the fluxes of the SED observations
    :y_err: the uncertainties of the flux observations
    :Teff1: the effective temperature of star 1 in K
    :Teff2: the effective temperature of star 2 in K
    :logg1: the surface gravity of star 1 in log(cgs)
    :logg2: the surface gravity of star 2 in log(cgs)
    :returns: the calculated likelihood value
    """
    y_model = model_func(x, Teff1, Teff2, logg1, logg2)
    return -0.5 * np.sum(((y - y_model) / y_err)**2)


def ln_prob(theta: Tuple[float, float, float],
            x: np.ndarray,
            y: np.ndarray,
            y_err: np.ndarray,
            k: float,
            ln_prior_func: Callable[[float, float, float, float], Tuple[float, Tuple[any]]],
            model_func: Callable[[np.ndarray, float, float, float, float],
                                 Tuple[np.ndarray]]=norm_blackbody_model):
    """
    The ln_prob function to be called by emcee.EnsembleSample to fit models to a SED.
    Will create the call stack of MCMC ln_prior_func() and ln_like() functions using the
    chosen model_func to support it.

    :theta: the current walker position of the (M1, M2, log_age) parameters
    :x: the frequencies at which the SED observations were made
    :y: the fluxes of the SED observations
    :y_err: the uncertainties of the flux observations
    :k: the ratio of the stellar radii prior
    :ln_prior_func: the mmcmc function to evaluate the priors. This should hace the form
    func(M1, M2, age, k) -> (0 or -np.inf, (Teff1, Teff2, logg1, logg2))
    :model_func: function to produce the model pairs of SEDs to be evaluated. This should have the
    form func(x, Teff1, Teff2, logg1, logg2) -> (sed1+sed2) and defaults to norm_blackbody_model()
    :returns: the log probability of this walker position and the unpacked blob returned from the
    ln_prior_func (which emcee will repack within a numpy array for publication through get_blobs)
    """
    # theta == M1, M2, log_age, blob = Teff1, Teff2, logg1, logg2
    lp, blob = ln_prior_func(*theta, k)
    if np.isfinite(lp):
        return lp + ln_like(x, y, y_err, *blob, model_func), *blob
    return -np.inf, *blob


def min_max_normalize(vals: np.ndarray, val_errs: np.ndarray=None):
    """
    Will min-max normalize the passed vals array and optional val_errs uncertainties array

    :vals: the array to normalize
    :val_errs: the optional uncertainties to normalize with the same scaling
    :returns: the normalized vals, or a tuple with the normalized vals and errs if both given
    """
    val_min = vals.min()
    norm_scale = vals.max() - val_min
    if val_errs is not None:
        return (vals - val_min) / norm_scale, val_errs / norm_scale
    return (vals - val_min) / norm_scale
