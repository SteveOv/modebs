""" TODO """
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
from uncertainties.unumpy import nominal_values as _noms, std_devs as _std_devs


# pylint: disable=line-too-long, no-member
pc = (1 * _u.pc).to(_u.m).value
R_sun = _iau2015.R_sun.to(_u.m).value


# GLOBALS which will be set by (minimize|mcmc)_fit prior to fitting. Hateful things!
# Unfortunately this is how we get fast MCMC, as the way emcee works makes
# using a class or passing these between functions in args way too sloooow!
# The code expects _fixed_theta, _fit_mask & each row of _prior_criteria to be the same size, so
# masks & indices apply equally to all three arrays making code within iterations simple & quick.
_fixed_theta: _np.ndarray[float]
_fit_mask: _np.ndarray[bool]
_prior_criteria: _np.ndarray[float]
_x: _np.ndarray[float]
_y: _np.ndarray[float]
_weights: _np.ndarray[float]
_flux_func: Callable[[_np.ndarray[float], float, float], _np.ndarray]

# Try to protect them as much as possible by wrapping writes within a critical section
_fit_mutex = _Lock()


def _ln_prior_func(theta: _np.ndarray[float]) -> float:
    """
    The fitting prior function which evaluate the current set of candidate parameters (theta)
    against the prior criteria and returns a single value indicating the goodness of the parameters.

    Accesses the following global variables which will be set by call to (minimize|mcmc)_fit()
    - _fit_nask: mask to select those members of theta which are fitted
    - _prior_criteria: limits and ratio +/- sigma criteria with rows indices corresponding to theta
        - _prior_criteria[0] are limits, each a tuple of (low, high)
        - _prior_criteria[1] are ratio nominal (float) values
        - _prior_criteria[2] are ratio sigma (float) values

    :theta: the full set of parameters from which to generate model fluxes
    :returns: a single negative value indicating the goodness of this set of parameters 
    """
    # Limit criteria checks - hard pass/fail on these
    if not all(lim[0] < th < lim[1] for th, lim in zip(theta[_fit_mask],
                                                       _prior_criteria[0][_fit_mask])):
        return -_np.inf

    # With 3 params per star + dist the #stars is...
    nstars = (theta.shape[0] - 1) // 3
    if nstars < 2: # no ratios
        return 0

    # Check the ratio wherever a companion value is fitted, or if the primary value is fitted then
    # check all companions for the parameter type (i.e. rad0 is fitted all ratio of radii checked).

    # These indices apply equally across theta, _fit_mask and each type of _prior_criteria.
    comp_ixs = [ix for ix, _ in enumerate(theta) if ix % nstars != 0]       # omit the primaries
    prim_ixs = [int(_np.floor(ix / nstars) * nstars) for ix in comp_ixs]    # primary for each comp

    # Gaussian priors: g(x) = 1/(σ*sqrt(2*pi)) * exp(-1/2 * (x-µ)^2/σ^2)
    # Omitting scaling expressions for now and note the implicit ln cancelling the exp
    return -0.5 * _np.sum([
        ((theta[cix] / theta[pix] - _prior_criteria[1][cix]) / _prior_criteria[2][cix])**2
                    for pix, cix in zip(prim_ixs, comp_ixs) if _fit_mask[pix] or _fit_mask[cix]
    ])


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
    return -0.5 * chisq


def model_func(theta: _np.ndarray[float],
               combine: bool=True):
    """
    Generate the model fluxes at points x from the candidate parameters theta.

    flux(*) = model(x, teff, logg) * radius^2 / dist^2

    Accesses the following global variables which will be set by call to (minimize|mcmc)_fit()
    - _x: the x points to generate model data for
    - _flux_func: the function to call to generate model fluxes

    :theta: the full set of parameters from which to generate model fluxes
    :combine: whether to return a single set of summed fluxes
    :returns: the model fluxes at points x, either per star if combine==False or aggregated
    """
    # The teff, rad and logg for each star is interleaved, so if two stars we expect:
    # [teff0, teff1, rad0, rad1, logg0, logg1, dist]. With 3 params per star + dist the #stars is...
    nstars = (theta.shape[0] - 1) // 3
    params_by_star = theta[:-1].reshape((3, nstars)).transpose()
    y_model = _np.array([
        _flux_func(_x, teff, logg).value * (rad * R_sun)**2 for teff, rad, logg in params_by_star
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

        _np.nan_to_num(retval, copy=False, nan=-_np.inf)

    if minimizable:
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
                 prior_criteria: _np.ndarray[float],
                 theta0: _np.ndarray[float],
                 fit_mask: _np.ndarray[float],
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
        global _x, _y, _weights, _fixed_theta, _fit_mask, _prior_criteria, _flux_func
        _x, _y, _weights = x, y, 1 / y_err**2
        _fixed_theta, _fit_mask = _np.where(fit_mask, None, theta0), fit_mask
        _prior_criteria, _flux_func = prior_criteria, flux_func

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
             prior_criteria: _np.ndarray[float],
             theta0: _np.ndarray[float],
             fit_mask: _np.ndarray[bool],
             flux_func: Callable[[_np.ndarray[float], float, float], _np.ndarray[float]],
             nwalkers: int=100,
             nsteps: int=100000,
             thin_by: int=10,
             seed: int=42,
             processes: int=1,
             early_stopping: bool=True,
             progress: Union[bool, str]=False,
             verbose: bool=False) -> Tuple[_np.ndarray[float], EnsembleSampler]:
    """
    Full fit model star(s) to the SED with an MCMC fit of the model data generated from
    a combination of the fixed params on class iniialization and the fitted ones given here.

    Will run up to niters iterations. Every 1000 iterations will check if the fit has
    converged and will stop early if that is the case

    :x: the wavelength/filter values for the observed SED data
    :y: the flux values, at x, for the observed SED data
    :y_err: the flux error bars, at x, for the observed SED data
    :prior_criteria: the criteria for limits and ratios used in evaluating theta against priors
    :theta0: the initial set of candidate parameters for the model SED
    :fit_mask: a mask on theta0 to pick the parameters that are fitted, the rest being fixed
    :flux_func: the function, with form func(x, teff, logg), called to generate fluxes
    :nwalker: the number of mcmc walkers to employ
    :nsteps: the maximium number of mcmc steps to make for each walker
    :thin_by: step interval to inspect fit progress
    :seed: optional seed for random behaviour
    :processes: optional number of parallel processes to use, or None to let code choose
    :progress: whether to show a progress bar (see emcee documentation for other values)
    :early_stopping: stop fitting if solution has converged & further improvements are negligible
    :returns: the final set of parameters (nominals only) and an emcee EnsembleSampler with the
    details of the outcome
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
        global _x, _y, _weights, _fixed_theta, _fit_mask, _prior_criteria, _flux_func
        _x, _y, _weights = x, y, 1 / y_err**2
        _fixed_theta, _fit_mask = _np.where(fit_mask, None, theta0), fit_mask
        _prior_criteria, _flux_func = prior_criteria, flux_func

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
    theta_fit = _np.median(samples[burn_in_steps:], axis=0)

    theta0[fit_mask] = theta_fit

    if verbose:
        print( "Autocorrelation steps (tau):", ", ".join(f"{t:.3f}" for t in tau))
        print(f"Estimated burn-in steps:     {int(max(_np.nan_to_num(tau, nan=1000)) * 2):,}")
        print(f"Mean Acceptance fraction:    {_np.mean(sampler.acceptance_fraction):.3f}")
        _print_theta(theta0, fit_mask, "Nominals of fitted theta:    ")

    return theta0, sampler


def create_prior_criteria(
        teff_limits: Union[List[Tuple[float, float]], Tuple[float, float]]=(2000, 20000),
        radius_limits: Union[List[Tuple[float, float]], Tuple[float, float]]=(0.1, 100),
        logg_limits: Union[List[Tuple[float, float]], Tuple[float, float]]=(0.0, 5.0),
        dist_limits: Tuple[float, float]=(0.0, _np.inf),
        teff_ratios: Union[List[float], float]=None,
        radius_ratios: Union[list[float], float]=None,
        logg_ratios: Union[list[float], float]=None,
        teff_ratio_sigmas: Union[List[float], float]=None,
        radius_ratio_sigmas: Union[List[float], float]=None,
        logg_ratio_sigmas: Union[List[float], float]=None,
        nstars: int=2,
        verbose: bool=False) -> _np.ndarray:
    """
    Will validate the teffs, radii, loggs and dist criteria and create the prior_criteria array
    from them. This will be in the format of a NDarray of shape (3, #items) where #items is
    nstars*3+1 which breaks dowm as one per the teff, radius and logg for each star + one for the
    whole system distance. With this approach, indices onto fixed_theta are directly applicable to
    the corresponding criteria.

    Note: these criteria are only evaluated on fitted values; they're ignored for fixed values

    The criteria array will have the following general form:
    ```python
    np.ndarray([
        [(teff0_lo,hi),...,(teffN_lo,hi),(rad0_lo,hi),...,(radN_lo,hi),(logg0_lo,hi),...,(loggN_lo,hi),(dist_lo,hi)],
        [1, teff1_rat,...,teffN_rat, 1, rad1_rat,...,radN_rat, 1, logg1_rat,...,loggN_rat, 1],
        [0, teff1_sig,...,teffN_sig, 0, rad1_sig,...,radN_sig, 0, logg1_sig,...,loggN_sig, 0],
    ])
    ```
    where N is nstars - 1.

    The teff|radius|logg _ratios and _ratio_sigmas can be provided through the corresponding args.
    The ratio args will accept UFloats with the sigmas taken from the UFloats' std_dev attribute,
    however these will be overridden by any values in the corresponding _ratio_sigmas args.

    :teff_limits: (low, high) tuples for the T_eff limits for each star, or one tuple for all
    :radius_limits: (low, high) tuples for the radius limits for each star, or one tuple for all
    :logg_limits: (low, high) tuples for the log(g) limits for each star, or one tuple for all
    :dist_limit: (low, high) tuple for the distance limits for all stars
    :teff_ratios: the T_eff ratio criteria for each companion star
    :radius_ratios: the radius ratio criteria for each companion star
    :logg_ratios: the log(g) ratio criteria for each companion star
    :teff_ratio_sigmas: the T_eff ratio sigma criteria for each companion star
    :radius_ratio_sigmas: the radius ratio sigma criteria for each companion star
    :logg_ratio_sigmas: the log(g) ratio sigma criteria for each companion star
    :returns: the fully built up set of criteria usable by the (minimize|mcmc)_fit functions
    """
    criteria = _np.empty((3, nstars*3 + 1), dtype=object)

    def to_limit_tuple(tval, name, i=None) -> Tuple[float, float]:
        if isinstance(tval, Number):
            return (0., tval)
        if isinstance(tval, Tuple) and len(tval) > 0:
            return (0., tval[0]) if len(tval) == 1 else tuple(tval[:2])
        src = f"{name}[{i if i is not None else ''}]"
        raise ValueError(f"{src}=={tval} cannot be interpreted as a Tuple(low, high)")

    # Build the limits array; will have form [(low0, high0), ..., (lown, highn)]
    limit_list = []
    for cix, (name, val) in enumerate([("teff_limits", teff_limits),
                                        ("radius_limits", radius_limits),
                                        ("logg_limits", logg_limits),
                                        ("dist_limits", dist_limits)]):
        exp_ct = nstars if cix < 3 else 1

        # Attempt to interpret the value as a List[(lower, upper)] * exp_ct
        if isinstance(val, Number|Tuple):
            limit_list += [to_limit_tuple(val, name)] * exp_ct
        elif isinstance(val, List|_np.ndarray) \
                    and len(val) == exp_ct and all(isinstance(v, Number|Tuple) for v in val):
            limit_list += [to_limit_tuple(v, name, vix) for vix, v in enumerate(val)]
        else:
            raise ValueError(f"{name}=={val} can't be interpreted as List[(low,high)]*{exp_ct}")

    # Build the ratio arrays into the following form. The 1s & 0s are for the primary component.
    # We expect actual ratios for each companion component, so +1 for a binary, +2 for a triple.
    rat_list = [1] * criteria.shape[1]
    sig_list = [0] * criteria.shape[1]
    cix = 0
    for name, rat_val, sig_val in [("teff", teff_ratios, teff_ratio_sigmas),
                                    ("radius", radius_ratios, radius_ratio_sigmas),
                                    ("logg", logg_ratios, logg_ratio_sigmas)]:
        cix += 1            # Skip the first item for this param - it's the primary component
        exp_ct = nstars - 1 # so the ratio will always be prim/prim == 1 +/- 0

        # Two ways of getting sigmas; either as sd of ufloat ratio or from sigma arg (overrides)
        if rat_val is None:
            pass
        elif isinstance(rat_val, Number):
            rat_list[cix : cix + exp_ct] = [rat_val] * exp_ct
        elif isinstance(rat_val, _UFloat):
            rat_list[cix : cix + exp_ct] = [rat_val.n] * exp_ct
            sig_list[cix : cix + exp_ct] = [rat_val.s] * exp_ct
        elif isinstance(rat_val, List|_np.ndarray) \
                and len(rat_val) == exp_ct and all(isinstance(v, Number) for v in rat_val):
            rat_list[cix : cix + exp_ct] = [v for v in rat_val]
        elif isinstance(rat_val, List|_np.ndarray) \
                and len(rat_val) == exp_ct and all(isinstance(v, _UFloat) for v in rat_val):
            rat_list[cix : cix + exp_ct] = [v for v in _noms(rat_val)]
            sig_list[cix : cix + exp_ct] = [v for v in _std_devs(rat_val)]
        else:
            raise ValueError(f"{name}_ratios=={rat_val} cannot be interpreted as List[float]*{exp_ct}")

        if sig_val is None:
            pass
        elif isinstance(sig_val, Number):
            sig_list[cix : cix + exp_ct] = [sig_val] * exp_ct
        elif isinstance(sig_val, List|_np.ndarray) \
                and len(sig_val) == exp_ct and all(isinstance(v, Number|None) for v in sig_val):
            sig_list[cix : cix + exp_ct] = [v for v in sig_val]
        else:
            raise ValueError(f"{name}_ratio_sigmas=={sig_val} cannot be interpreted as List[float]*{exp_ct}")

        cix += exp_ct

    criteria[:, :] = [limit_list, rat_list, sig_list]
    if verbose:
        for ix, crit in enumerate(criteria):
            print(f"prior_criteria[{ix}]: ",
                  ", ".join(f"{c:.3e}" if isinstance(c, Number) else f"{c}" for c in crit))
    return criteria


def create_theta(teffs: Union[List[float], float],
                 radii: Union[List[float], float],
                 loggs: Union[List[float], float],
                 dist: float,
                 nstars: int=2,
                 verbose: bool=False) -> _np.ndarray[float]:
    """
    Will validate the teffs, radii, loggs and dist values and create a theta list from them.
    This is the full set of parameters needed to generate a model SED from nstars components.

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
            theta[ix : ix+exp_count] += [val] * exp_count
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
