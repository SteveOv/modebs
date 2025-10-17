""" A module for fitting stellar model fluxes for multiple stars to stellar SEDs """
from typing import Tuple, List, Callable, Union
from numbers import Number
from math import floor as _floor

from warnings import filterwarnings as _filterwarnings, catch_warnings as _catch_warnings

from multiprocessing import Pool as _Pool, cpu_count as _cpu_count
from threading import Lock as _Lock

import numpy as _np

from scipy.optimize import minimize as _minimize
from scipy.optimize import OptimizeResult, OptimizeWarning

from emcee import EnsembleSampler

import astropy.units as _u
from astropy.constants import iau2015 as _iau2015

from uncertainties import UFloat as _UFloat
from uncertainties.unumpy import uarray as _uarray

# pylint: disable=too-many-arguments, too-many-positional-arguments, too-many-locals, no-member
pc = (1 * _u.pc).to(_u.m).value
R_sun = _iau2015.R_sun.to(_u.m).value


# GLOBALS which will be set by (minimize|mcmc)_fit prior to fitting. Hateful things!
# Unfortunately this is how we get fast MCMC, as the way emcee works makes
# using a class or passing these between functions in args way too sloooow!
# The code expects _fixed_theta and _fit_mask to be the same size.
_fixed_theta: _np.ndarray[float]
_fit_mask: _np.ndarray[bool]
_x: _np.ndarray[float]
_y: _np.ndarray[float]
_weights: _np.ndarray[float]
_ln_prior_func: Callable[[_np.ndarray[float]], float]
_flux_func: Callable[[_np.ndarray[float], float, float], _np.ndarray]

# Try to protect them as much as possible by wrapping writes within a critical section
_fit_mutex = _Lock()


def _ln_likelihood_func(y_model: _np.ndarray[float], degrees_of_freedom: int) -> float:
    """
    The fitting likelihoof function used to evaluate the model y values against the observations,
    returning a single negative value indicating the goodness of the fit.
    
    Based on a weighted chi^2: chi^2_w = 1/(N_obs-n_param) * Σ W(y-y_model)^2

    Accesses the following global variables which will be set by call to (minimize|mcmc)_fit()
    - _y: the observed y values
    - _weights: the weights to apply to each observation/model y value

    :y_model: the model y values
    :degrees_of_freedom: the #observations/#params
    :returns: the goodness of the fit
    """
    chisq = _np.sum(_weights * (_y - y_model)**2) / degrees_of_freedom
    return 0.5 * chisq


def model_func(theta: _np.ndarray[float],
               x: _np.ndarray[float]=None,
               flux_func: Callable[[_np.ndarray[float], float, float], _np.ndarray[float]]=None,
               combine: bool=True):
    """
    Generate the model fluxes at points x from the candidate parameters theta.

    flux(star_N) = flux_func(x, teff_N, logg_N) * radius_N^2 / dist^2

    Accesses the following global variables which will be set by call to (minimize|mcmc)_fit()
    - _x: the x points to generate model data for
    - _flux_func: function to call to generate model fluxes, returning floats in same units as SED

    :theta: the full set of parameters from which to generate model fluxes
    :x: optional filter/wavelengths to generate fluxes for - if omitted will use _x
    :flux_func: optional function to call to generate fluxes - if omitted will use _flux_func
    :combine: whether to return a single set of summed fluxes
    :returns: the model fluxes at points x, either per star if combine==False or aggregated
    """
    # These can be taken as args for external calls but fall back in the hateful globals
    if x is None:
        x = _x
    if flux_func is None:
        flux_func = _flux_func

    # The teff, rad and logg for each star is interleaved, so if two stars we expect:
    # [teff0, teff1, rad0, rad1, logg0, logg1, dist]. With 3 params per star + dist the #stars is...
    nstars = (theta.shape[0] - 1) // 3
    params_by_star = theta[:-1].reshape((3, nstars)).transpose()
    y_model = _np.array([
        flux_func(x, teff, logg) * (rad * R_sun)**2 for teff, rad, logg in params_by_star
    ])

    # Finally, divide by the dist^2 (m^2), which is the remaining param not used above
    dist = theta[-1] * pc
    if combine:
        return _np.sum(y_model, axis=0) / dist**2
    return y_model / dist**2


def _objective_func(fit_theta: _np.ndarray[float], minimizable: bool=False) -> float:
    """
    The function to be optimized by adjusting theta so that the return value converges to zero.

    :fit_theta: current set of candidate fitted parameters only
    :minimizable: whether this function is minimizable (returns positive) or not (returns negative)
    :returns: the result of evaluating the fitted model against the observations
    """
    # Combine the fitted and fixed parameters to make a full set.
    theta = _fixed_theta.copy()
    theta[_fit_mask] = fit_theta

    if _np.isfinite(retval := _ln_prior_func(theta)):
        y_model = model_func(theta, combine=True)

        degr_freedom = y_model.shape[0] - fit_theta.shape[0]
        retval += _ln_likelihood_func(y_model, degr_freedom)

        _np.nan_to_num(retval, copy=False, nan=_np.inf)

    if minimizable != (retval >= 0):
        return -retval
    return retval


def _print_theta(theta: _np.ndarray[float],
                 fit_mask: _np.ndarray[bool],
                 prefix: str="",
                 suffix: str=""):
    """ Utility function for pretty printing theta arrays & highlighting which items are fitted. """
    print((prefix if prefix else '') +
          "[" +
          ", ".join(f"{t:.3e}{'*' if f else ''}" for t, f in zip(theta, fit_mask)) +
          "]" +
          (suffix if suffix else ''))


def minimize_fit(x: _np.ndarray[float],
                 y: _np.ndarray[float],
                 y_err: _np.ndarray[float],
                 theta0: _np.ndarray[float],
                 fit_mask: _np.ndarray[float],
                 ln_prior_func: Callable[[_np.ndarray[float]], float],
                 flux_func: Callable[[_np.ndarray[float], float, float], _np.ndarray[float]],
                 methods: List[str]=None,
                 verbose: bool=False) -> Tuple[_np.ndarray[float], OptimizeResult]:
    """
    Quick fit model star(s) to the SED with scipy minimize fit of the model data generated from
    a combination of the fixed params on class iniialization and the fitted ones given here.
    Will choose the best performing fit from the algorithms in methods.

    :x: the wavelength/filter values for the observed SED data
    :y: the flux values, at x, for the observed SED data
    :y_err: the flux error bars, at x, for the observed SED data
    :prior_criteria: the criteria for limits and ratios used in evaluating theta against priors
    :theta0: the initial set of candidate parameters for the model SED
    :fit_mask: a mask on theta0 to pick the parameters that are fitted, the rest being fixed
    :ln_prior_func:
    :flux_func: the function, with form func(x, teff, logg), called to generate fluxes
    :methods: scipy optimize fitting algorithms to try, defaults to [Nelder-Mead, SLSQP, None]
    :returns: the final set of parameters & a scipy OptimizeResult with the details of the outcome
    """
    if verbose:
        _print_theta(theta0, fit_mask, "minimize_fit(theta0=", ")")

    if methods is None:
        methods = ["Nelder-Mead", "SLSQP", None]
    elif isinstance(methods, str):
        methods = [methods]

    max_iters = int(1000 * sum(fit_mask))

    with _fit_mutex, _catch_warnings(category=[RuntimeWarning, OptimizeWarning]):
        _filterwarnings("ignore", "invalid value encountered in ")
        _filterwarnings("ignore", "Desired error not necessarily achieved due to precision loss.")
        _filterwarnings("ignore", "Unknown solver options:")

        # Now we've got exclusive access, we can set the globals required for fitting
        # pylint: disable=global-statement
        global _x, _y, _weights, _fixed_theta, _fit_mask, _ln_prior_func, _flux_func
        _x, _y, _weights = x, y, 1 / y_err**2
        _fixed_theta, _fit_mask = _np.where(fit_mask, None, theta0), fit_mask
        _ln_prior_func, _flux_func = ln_prior_func, flux_func

        the_soln, the_meth = None, None
        for method in methods:
            a_soln = _minimize(_objective_func, x0=theta0[fit_mask], args=(True), # minimizable
                               method=method, options={ "maxiter": max_iters, "maxfev": max_iters })
            if verbose:
                print(f"({method})",
                        "succeeded" if a_soln.success else f"failed [{a_soln.message}]",
                        f"after {a_soln.nit:d} iterations & {a_soln.nfev:d} function evaluation(s)",
                        f"[fun = {a_soln.fun:.6f}]")

            if the_soln is None \
                    or (a_soln.success and not the_soln.success) \
                    or (a_soln.fun < the_soln.fun):
                the_soln, the_meth = a_soln, method

    if the_soln.success:
        theta0[fit_mask] = the_soln.x
        if verbose:
            _print_theta(theta0, fit_mask, f"The best fit with {the_meth} method yielded theta=")
    else:
        _print_theta(theta0, fit_mask, "The fit failed so returning input, theta0=")

    return theta0, the_soln


def mcmc_fit(x: _np.ndarray[float],
             y: _np.ndarray[float],
             y_err: _np.ndarray[float],
             theta0: _np.ndarray[float],
             fit_mask: _np.ndarray[bool],
             ln_prior_func: Callable[[_np.ndarray[float]], float],
             flux_func: Callable[[_np.ndarray[float], float, float], _np.ndarray[float]],
             nwalkers: int=100,
             nsteps: int=100000,
             thin_by: int=10,
             seed: int=42,
             processes: int=1,
             early_stopping: bool=True,
             progress: Union[bool, str]=False,
             verbose: bool=False) -> Tuple[_np.ndarray[_UFloat], EnsembleSampler]:
    """
    Full fit model star(s) to the SED with an MCMC fit of the model data generated from
    a combination of the fixed params on class iniialization and the fitted ones given here.

    Will run up to niters iterations. Every 1000 iterations will check if the fit has
    converged and will stop early if that is the case

    :x: the wavelength/filter values for the observed SED data
    :y: the flux values, at x, for the observed SED data
    :y_err: the flux error bars, at x, for the observed SED data
    :theta0: the initial set of candidate parameters for the model SED
    :fit_mask: a mask on theta0 to pick the parameters that are fitted, the rest being fixed
    :ln_prior_func:
    :flux_func: the function, with form func(x, teff, logg), called to generate fluxes
    :nwalker: the number of mcmc walkers to employ
    :nsteps: the maximium number of mcmc steps to make for each walker
    :thin_by: step interval to inspect fit progress
    :seed: optional seed for random behaviour
    :processes: optional number of parallel processes to use, or None to let code choose
    :progress: whether to show a progress bar (see emcee documentation for other values)
    :early_stopping: stop fitting if solution has converged & further improvements are negligible
    :returns: fitted set of parameters as UFloats and an EnsembleSampler with details of the outcome
    """
    if verbose:
        _print_theta(theta0, fit_mask, "mcmc_fit(theta0=", ")")

    rng = _np.random.default_rng(seed)
    theta_fit = theta0[fit_mask]
    ndim = len(theta_fit)
    autocor_tol = 50 / thin_by
    tau = [_np.inf] * ndim

    # Starting positions for the walkers clustered around theta0
    p0 = [theta_fit + (theta_fit * rng.normal(0, 0.05, ndim)) for _ in _np.arange(int(nwalkers))]

    with _fit_mutex, \
            _Pool(processes=processes) as pool, \
            _catch_warnings(category=[RuntimeWarning, UserWarning]):

        _filterwarnings("ignore", message="invalid value encountered in ")
        _filterwarnings("ignore", message="Using UFloat objects with std_dev==0")

        # Now we've got exclusive access, we can set the globals required for fitting
        # pylint: disable=global-statement
        global _x, _y, _weights, _fixed_theta, _fit_mask, _ln_prior_func, _flux_func
        _x, _y, _weights = x, y, 1 / y_err**2
        _fixed_theta, _fit_mask = _np.where(fit_mask, None, theta0), fit_mask
        _ln_prior_func, _flux_func = ln_prior_func, flux_func

        # Min steps required by Autocorr algo to avoid a warn msg (not a warning so can't filter)
        min_steps_es = int(50 * autocor_tol * thin_by)

        print(f"Running MCMC fit with {nwalkers:d} walkers for {nsteps:d} steps, thinned by",
            f"{thin_by}, on {processes}" if processes else f"up to {_cpu_count()}", "process(es).",
            f"Early stopping is enabled after {min_steps_es:d} steps." if early_stopping else "")
        sampler = EnsembleSampler(int(nwalkers), ndim, _objective_func, pool=pool)

        for _ in sampler.sample(initial_state=p0, iterations=nsteps // thin_by,
                                thin_by=thin_by, tune=True, progress=progress):

            if (step := sampler.iteration * thin_by) > min_steps_es and step % 1000 == 0:
                if early_stopping:
                    # The autocor time (tau) is the steps to effectively forget start position.
                    # As the fit converges the change in tau will tend towards zero.
                    prev_tau = tau
                    tau = sampler.get_autocorr_time(c=5, tol=autocor_tol, quiet=True) * thin_by
                    if not any(_np.isnan(tau)) \
                            and all(tau < step / 100) \
                            and all(abs(prev_tau - tau) / prev_tau < 0.01):
                        print(f"Halting MCMC after {step:,} steps as the walkers are past",
                              "100 times the autocorrelation time & the fit has converged.")
                        break

    tau = sampler.get_autocorr_time(c=5, tol=autocor_tol, quiet=True) * thin_by
    burn_in_steps = int(max(_np.nan_to_num(tau, copy=True, nan=1000)) * 2)
    samples = sampler.get_chain(discard=burn_in_steps, flat=True)

    # Get theta into ufloats with std_dev based on the mean +/- 1-sigma values (where fitted)
    theta_fit = _uarray(theta0, 0)
    fitted_noms = _np.median(samples[burn_in_steps:], axis=0)
    fitted_err_high = _np.quantile(samples[burn_in_steps:], 0.84, axis=0) - fitted_noms
    fitted_err_low = fitted_noms - _np.quantile(samples[burn_in_steps:], 0.16, axis=0)
    theta_fit[fit_mask] = _uarray(fitted_noms, _np.mean([fitted_err_high, fitted_err_low], axis=0))

    if verbose:
        print( "Autocorrelation steps (tau):", ", ".join(f"{t:.3f}" for t in tau))
        print(f"Estimated burn-in steps:     {int(max(_np.nan_to_num(tau, nan=1000)) * 2):,}")
        print(f"Mean Acceptance fraction:    {_np.mean(sampler.acceptance_fraction):.3f}")
        _print_theta(theta_fit, fit_mask, "The MCMC fit yielded theta:  ")

    return theta_fit, sampler


def create_theta(teffs: Union[List[float], float],
                 radii: Union[List[float], float],
                 loggs: Union[List[float], float],
                 dist: float,
                 nstars: int=2,
                 verbose: bool=False) -> _np.ndarray[float]:
    """
    Helper function to validate the teffs, radii, loggs and dist values and create a theta list from
    them. This is the full set of parameters needed to generate a model SED from nstars components.

    The resulting theta array will have the form:
    ```python
    theta = [teff0, ... , teffN, rad0, ... , radN, logg0, ..., loggN, dist]
    ```
    where N is nstars - 1.

    Units: teffs in K, radii in Rsun, logg in dex[cgs] and distance in parsecs

    Note: theta has to be one-dimensional as scipy minimize will not fit multidimensional theta

    :teffs: effective temps [K] as a list of floats nstars long or a single float (same value each)
    :radii: stars' radii [Rsun] as a list of floats nstars long or a single float (same value each)
    :loggs: stars' log(g) as a list of floats nstars long or a single float (same value each)
    :dist: the distance [parsecs] as a single float
    :nstars: the number of stars we're building for 
    :returns: the resulting theta list
    """
    theta = _np.empty((nstars * 3 + 1), dtype=float)
    ix = 0
    for name, val in [("teffs", teffs), ("radii", radii), ("loggs", loggs), ("dist", dist)]:
        exp_count = 1 if name == "dist" else nstars

        # Attempt to interpret the value as a List[Number]
        if isinstance(val, Number):
            theta[ix : ix+exp_count] = [val] * exp_count
        elif isinstance(val, Tuple|List|_np.ndarray) \
                and len(val) == exp_count \
                and all(isinstance(v, Number|None) for v in val):
            theta[ix : ix+exp_count] = [t for t in val]
        else:
            raise ValueError(f"{name}=={val} cannot be interpreted as a List[Number]*{exp_count}")

        ix += exp_count

    if verbose:
        print("theta:\t", ", ".join(f"{t:.3e}" if isinstance(t, Number) else f"{t}" for t in theta))
    return theta
