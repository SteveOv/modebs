""" Functions supporting the MCMC analysis of SED curves to estimate dEB stellar masses """
from typing import Callable, Tuple

import numpy as np

# pylint: disable=invalid-name, too-many-arguments, too-many-positional-arguments


def ln_like(theta: np.ndarray,
            x: np.ndarray,
            y: np.ndarray,
            y_err: np.ndarray,
            model_func: Callable[[np.ndarray, any], np.ndarray]) -> float:
    """
    The MCMC log likelihood function, which returns a chi-squared based comparison of
    the x & y (+/-y_err) observations with the model produced by the model_func.

    The comparison is calculated as -1/2 Σ((y - model) / y_err)^2

    This returns a negative value as emcee will seek to maximize this value.

    :theta: the current walker position/parameter set to evaluate
    :x: the x values at which the observations are made
    :y: the y values of observations
    :y_err: the y value uncertainties
    :model_func: function to produce the model values to be evaluated.
    This should have the form func(x, *theta_rev) -> np.ndarray.
    :returns: the calculated likelihood value
    """
    chi2 = np.square((y - model_func(x, *theta)) / y_err)
    return -0.5 * np.sum(chi2)


def ln_prob(theta: np.ndarray,
            x: np.ndarray,
            y: np.ndarray,
            y_err: np.ndarray,
            ln_prior_func: Callable[[any], Tuple[float, np.ndarray]],
            model_func: Callable[[np.ndarray, any], np.ndarray]) -> Tuple[float, any]:
    """
    The ln_prob function to be called by emcee.EnsembleSample to fit models to a SED.
    Will create the call stack of MCMC ln_prior_func() and ln_like() functions using the
    chosen model_func to support it.

    :theta: the current walker position/parameter set to evaluate
    :x: the x values at which the observations are made
    :y: the y values of observations
    :y_err: the y value uncertainties
    :ln_prior_func: function to evaluate theta against any priors and to potentially revise
    theta for the model_func. This should have the form func(*theta) -> (0 or -np.inf, theta_rev)
    :model_func: function to produce the model values to be evaluated.
    This should have the form func(x, *theta_rev) -> np.ndarray.
    :returns: log probability at this walker position & the theta_rev values corresponding to theta
    as returned from ln_prior_func (emcee will repack within a numpy array & publish with get_blobs)
    """
    lp, theta_rev = ln_prior_func(*theta)
    if np.isfinite(lp):
        return lp + ln_like(theta_rev, x, y, y_err, model_func), *theta_rev
    return -np.inf, *theta_rev


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
