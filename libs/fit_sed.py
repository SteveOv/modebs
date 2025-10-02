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
from uncertainties import UFloat as _UFloat
from uncertainties.unumpy import nominal_values as _noms, std_devs as _std_devs

from deblib import constants as _deblib_const

# GLOBALS which will be set by (minimize|mcmc)_fit prior to fitting. Hateful things!
# Unfortunately this is how we get fast MCMC, as the way emcee works makes
# using a class or passing these between functions in args way too sloooow!
_fixed_theta: _np.ndarray[float]
_prior_criteria: _np.ndarray[float]
_x: _np.ndarray[float]
_y: _np.ndarray[float]
_weights: _np.ndarray[float]
_flux_func: Callable[[_np.ndarray[float], float, float], _np.ndarray]

# Try to protect them as much as possible by wrapping writes within a critical section
_fit_mutex = _Lock()

# pylint: disable=line-too-long

def _ln_prior_func(theta: _np.ndarray[float]) -> float:
    """
    The fitting prior function which evaluate the current set of candidate parameters (theta)
    against the prior criteria and returns a single value indicating the goodness of the parameters.

    Accesses the following global variables which will be set by call to (minimize|mcmc)_fit()
    - _fixed_theta: the set of fixed parameters combined with theta to describe system
    - _prior_criteria: the prior limits and ratio +/- sigma criteria

    :theta: the current set of candidate fitted parameters which "fill the gaps" in fixed theta
    :returns: a single negative value indicating the goodness of this set of parameters 
    """
    # Limit criteria checks - hard pass/fail on these
    ft_mask = _np.isnan(_fixed_theta)
    if not all(lm[0] < t < lm[1] for t, lm in zip(theta, _prior_criteria[0][ft_mask], strict=True)):
        return -_np.inf

    # With 3 params per star + dist the #stars is...
    nstars = (_fixed_theta.shape[0] - 1) // 3
    if nstars == 1: # no ratios
        return 0

    # We check the ratio wherever a companion value is fitted, or if the primary value is fitted
    # it's all compaions for the parameter type (i.e. if rad0 is fitted all ratio of radii checked)
    # Coalesce the fitted and fixed values, so that we can calculate any required ratios.
    full_theta = _fixed_theta.copy()
    _np.putmask(full_theta, ft_mask, theta)

    # Find the components of the ratios are in the coalesced theta. Always 1 less ratio than stars.
    tert_ixs = [i for i, v in enumerate(full_theta) if i % nstars > 0 and v is not None]
    prim_ixs = [int(_np.floor(ix / nstars) * nstars) for ix in tert_ixs]

    # Gaussian priors: g(x) = 1/(σ*sqrt(2*pi)) * exp(-1/2 * (x-µ)^2/σ^2)
    # Omitting scaling expressions for now and note the implicit log cancelling the exp
    inners = []
    for prim_ix, tert_ix in zip(prim_ixs, tert_ixs):
        # Need to evaluate these for the tert, where either the tert or the prim are fitted
        if _np.isnan(_fixed_theta[prim_ix]) or _np.isnan(_fixed_theta[tert_ix]):
            prior_rat = _prior_criteria[1][tert_ix]
            prior_sig = _prior_criteria[2][tert_ix]

            inners += [((full_theta[tert_ix] / full_theta[prim_ix] - prior_rat) / prior_sig)**2]
    return -0.5 * _np.sum(inners)


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
    Generate the model fluxes at points x from the candidate parameters theta & fixed_theta.

    flux(*) = model(x, teff, logg) * radius^2 / dist^2

    Accesses the following global variables which will be set by call to (minimize|mcmc)_fit()
    - _x: the x points to generate model data for
    - _fixed_theta: corresponding set of fixed parameters; combined with theta to describe each star

    :theta: the current set of candidate fitted parameters which "fill the gaps" in fixed theta
    :combine: whether to return a single set of summed fluxes
    :returns: the model fluxes at points x, either per star if combine==False or aggregated
    """
    # Coalesce theta with the underlying fixed params.
    full_theta = _fixed_theta.copy()
    _np.putmask(full_theta, _np.isnan(full_theta), theta)

    # The teff, rad and logg for each star is interleaved, so if two stars we expect:
    # [teff0, teff1, rad0, rad1, logg0, logg1, dist]. With 3 params per star + dist the #stars is...
    nstars = (len(_fixed_theta) - 1) // 3
    params_by_star = full_theta[:-1].reshape((3, nstars)).transpose()
    y_model = _np.array([
        _flux_func(_x, teff, logg).value * (rad * _deblib_const.R_sun.n)**2
                                                    for teff, rad, logg in params_by_star
    ])

    # Finally, divide by the dist^2, which is the remaining param not used above
    if combine:
        return _np.sum(y_model, axis=0) / full_theta[-1]**2
    return y_model / full_theta[-1]**2


def _objective_func(theta: _np.ndarray[float], minimizable: bool=False) -> float:
    """
    The function to be optimized by adjusting theta so that the return value converges to zero.

    :theta: the current set of candidate fitted parameters which "fill the gaps" in fixed theta
    :minimizable: whether this function is minimizable (returns positive) or not (returns negative)
    :returns: the result of evaluating the fitted model against the observations
    """
    if _np.isfinite(retval := _ln_prior_func(theta)):
        y_model = model_func(theta, combine=True)
        degr_freedom = y_model.shape[0] - theta.shape[0]
        retval += _ln_likelihood_func(y_model, degr_freedom)
        _np.nan_to_num(retval, copy=False, nan=-_np.inf)

    if minimizable:
        return -retval
    return retval


def minimize_fit(x: _np.ndarray[float],
                 y: _np.ndarray[float],
                 y_err: _np.ndarray[float],
                 prior_criteria: _np.ndarray[float],
                 theta0: _np.ndarray[float],
                 fixed_theta: _np.ndarray[float],
                 flux_func: Callable[[_np.ndarray[float], float, float], _np.ndarray[float]],
                 methods: List[str]=None,
                 verbose: bool=False) -> OptimizeResult:
    """
    Quick fit model star(s) to the SED with scipy minimize fit of the model data generated from
    a combination of the fixed params on class iniialization and the fitted ones given here.
    Will choose the best performing fit from the algorithms in methods.

    :x: the wavelength/filter values for the observed SED data
    :y: the flux values, at x, for the observed SED data
    :y_err: the flux error bars, at x, for the observed SED data
    :prior_criteria: the criteria for limits and ratios used in evaluating theta against priors
    :theta0: the initial set of candidate fitted parameters which "fill the gaps" in theta_fixed
    :theta_fixed: the fixed, non-fitted parameters required to produced the model the SED data
    :flux_func: the function, with form func(x, teff, logg), called to generate fluxes
    :methods: scipy optimize fitting algorithms to try, defaults to [Nelder-Mead, SLSQP, None]
    :returns: a scipy OptimizeResult with the details of the outcome
    """
    if verbose:
        print("minimize_fit(theta0=[" + ", ".join(f"{t:.6f}" for t in theta0) + "])")

    if methods is None:
        methods = ["Nelder-Mead", "SLSQP", None]
    elif isinstance(methods, str):
        methods = [methods]

    with _fit_mutex, _catch_warnings(category=[RuntimeWarning, OptimizeWarning]):
        _filterwarnings("ignore", "invalid value encountered in subtract")
        _filterwarnings("ignore", "Desired error not necessarily achieved due to precision loss.")
        _filterwarnings("ignore", "Unknown solver options:")

        # Now we've got exclusive access, we can set the globals required for fitting
        # pylint: disable=global-statement
        global _x, _y, _weights, _fixed_theta, _prior_criteria, _flux_func
        _x, _y, _weights = x, y, 1 / y_err**2
        _fixed_theta, _prior_criteria = fixed_theta, prior_criteria
        _flux_func = flux_func

        the_soln, the_method = None, None
        for method in methods:
            a_soln = _minimize(_objective_func, x0=theta0, args=(True), # minimizable
                               method=method, options={ "maxiter": 5000, "maxfev": 5000 })
            if verbose:
                print(f"({method})",
                        "succeeded" if a_soln.success else f"failed [{a_soln.message}]",
                        f"after {a_soln.nit:d} iterations & {a_soln.nfev:d} function evaluation(s)",
                        f"[fun = {a_soln.fun:.6f}]")

            if the_soln is None \
                    or (a_soln.success and not the_soln.success) \
                    or (a_soln.fun < the_soln.fun):
                the_soln, the_method = a_soln, method

    if verbose:
        print(f"The best fit used the '{the_method}' method, yielding final theta = [" +
                ", ".join(f"{t:.6f}" for t in the_soln.x) + "]")
    return the_soln


def mcmc_fit(x: _np.ndarray[float],
             y: _np.ndarray[float],
             y_err: _np.ndarray[float],
             prior_criteria: _np.ndarray[float],
             theta0: _np.ndarray[float],
             fixed_theta: _np.ndarray[float],
             flux_func: Callable[[_np.ndarray[float], float, float], _np.ndarray[float]],
             nwalkers: int=100,
             nsteps: int=100000,
             thin_by: int=10,
             seed: int=42,
             processes: int=1,
             early_stopping: bool=True,
             verbose: bool=False) -> EnsembleSampler:
    """
    Full fit model star(s) to the SED with an MCMC fit of the model data generated from
    a combination of the fixed params on class iniialization and the fitted ones given here.

    Will run up to niters iterations. Every 1000 iterations will check if the fit has
    converged and will stop early if that is the case

    :x: the wavelength/filter values for the observed SED data
    :y: the flux values, at x, for the observed SED data
    :y_err: the flux error bars, at x, for the observed SED data
    :prior_criteria: the criteria for limits and ratios used in evaluating theta against priors
    :theta0: the initial set of candidate fitted parameters which "fill the gaps" in theta_fixed
    :theta_fixed: the fixed, non-fitted parameters required to produced the model the SED data
    :flux_func: the function, with form func(x, teff, logg), called to generate fluxes
    :nwalker: the number of mcmc walkers to employ
    :nsteps: the maximium number of mcmc steps to make for each walker
    :thin_by: step interval to inspect fit progress
    :seed: optional seed for random behaviour
    :processes: optional number of parallel processes to use, or None to let code choose
    :early_stopping: stop fitting if solution has converged & further improvements are negligible
    :returns: a emcee EnsembleSampler with the details of the outcome
    """
    if verbose:
        print("mcmc_fit(theta0=[" + ", ".join(f"{t:.6f}" for t in theta0) + "])")

    rng = _np.random.default_rng(seed)
    ndim = len(theta0)
    autocor_tol = 50 / thin_by
    tau = [_np.inf] * ndim

    # Starting positions for the walkers clustered around theta0
    p0 = [theta0 + (theta0 * rng.normal(0, 0.05, ndim)) for _ in _np.arange(int(nwalkers))]

    with _fit_mutex, \
            _Pool(processes=processes) as pool, \
            _catch_warnings(category=[RuntimeWarning, UserWarning]):

        _filterwarnings("ignore", message="invalid value encountered in scalar subtract")
        _filterwarnings("ignore", message="Using UFloat objects with std_dev==0")

        # Now we've got exclusive access, we can set the globals required for fitting
        # pylint: disable=global-statement
        global _x, _y, _weights, _fixed_theta, _prior_criteria, _flux_func
        _x, _y, _weights = x, y, 1 / y_err**2
        _fixed_theta, _prior_criteria = fixed_theta, prior_criteria
        _flux_func = flux_func

        # Min steps required by Autocorr algo to avoid a warn msg (not a warning so can't filter)
        min_steps_es = int(50 * autocor_tol * thin_by)

        print(f"Running MCMC fit with {nwalkers:d} walkers for {nsteps:d} steps, thinned by",
            f"{thin_by}, on {processes}" if processes else f"up to {_cpu_count()}", "process(es).",
            f"Early stopping is enabled after {min_steps_es:d} steps." if early_stopping else "")
        sampler = EnsembleSampler(int(nwalkers), ndim, _objective_func, pool=pool)

        for _ in sampler.sample(initial_state=p0, iterations=nsteps // thin_by,
                                thin_by=thin_by, tune=True, progress=True):

            if (step := sampler.iteration * thin_by) > min_steps_es and step % 1000 == 0:
                if early_stopping:
                    # The autocor time (tau) is the steps to effectively forget start position.
                    # As the fit converges the change in tau will tend towards zero.
                    prev_tau = tau
                    tau = sampler.get_autocorr_time(c=5, tol=autocor_tol, quiet=True) * thin_by
                    if not any(_np.isnan(tau)) \
                        and all(tau < step / 100) \
                        and all(abs(prev_tau - tau) / prev_tau < 0.01):
                        if verbose:
                            print(f"Halting MCMC after {step:,} steps as we're past",
                                    "100 times the autocorrelation time & the fit has converged.")
                        break

    if verbose:
        tau = sampler.get_autocorr_time(c=5, tol=autocor_tol, quiet=True) * thin_by
        print( "Autocorrelation steps (tau):", ", ".join(f"{t:.3f}" for t in tau))
        print(f"Estimated burn-in steps:     {int(max(_np.nan_to_num(tau, nan=1000)) * 2):,}")
        print(f"Mean Acceptance fraction:    {_np.mean(sampler.acceptance_fraction):.3f}")

    return sampler


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
            rat_list[cix : cix + exp_ct] = [v for v in sig_val]
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


def create_theta(teffs: Union[List[float], float]=None,
                 radii: Union[List[float], float]=None,
                 loggs: Union[List[float], float]=None,
                 dist: float=None,
                 nstars: int=2,
                 build_delta: bool=False,
                 fixed_theta: _np.ndarray[float]=None,
                 verbose: bool=False) -> _np.ndarray[float]:
    """
    Will validate the teffs, radii, loggs and dist values and create a theta list from them.
    This is the set of parameters needed to generate a model SED from nstars components.

    If build_delta is false we get a full theta, which will have the form:
    ```python
    theta = [teff0, ... , teffN, rad0, ... , radN, logg0, ..., loggN, dist]
    ```
    where N is nstars - 1, and values of None are stored wherever they are present
    in the input teffs, radii, loggs and dist parameters.
    
    If build_delta is True and fixed_theta supplied we get a delta(theta) which is just the
    values missing from fixed_theta (with Nones omiited). For example, if:
    ```python
    fixed_theta = [teff0, teff1, rad0, rad1, None, None, dist]
    ```
    with teffs, radii and dist all None, and loggs with values, the resulting delta(theta) will be:
    ```python
    theta = [logg0, logg1]
    ```
    where, in this case, nstars == 2.

    Note: theta has to be one-dimensional as scipy minimize will not fit multidimensional theta

    :teffs: either a list of floats nstars long, a single float (same for each) or None
    :radii: either a list of floats nstars long, a single float (same for each) or None
    :loggs: either a list of floats nstars long, a single float (same for each) or None
    :dist: either a float, Quantity or None
    :nstars: the number of stars we're building for 
    :build_delta: whether to compress out Nones (True) or not (False)
    :fixed_theta: if build_delta==True this is the set of fixed params to validate and build a
    delta for - the resulting theta list must contain values for all missing values in fixed_theta
    :returns: the resulting theta list
    """
    theta = []
    for ix, (name, value) in enumerate([
                        ("teffs", teffs), ("radii", radii), ("loggs", loggs), ("dist", dist)]):
        val_params = None
        exp_count = nstars if ix < 3 else 1

        # Attempt to interpret the value as a List[Number]
        if isinstance(value, Number|None):
            val_params = [value] * exp_count
        elif isinstance(value, Tuple|List) and len(value) == exp_count \
                    and all(isinstance(v, Number|None) for v in value):
            val_params = value
        else:
            raise ValueError(f"{name}=={value} cannot be interpreted as a List[Number]*{exp_count}")

        if build_delta: # Building a delta; expected to supply all the params missing in fixed_theta
            if fixed_theta is None:
                raise ValueError("fixed_theta must be supplied if build_delta == True")

            fix = ix * nstars
            fixed_params = fixed_theta[fix : fix+exp_count]
            if not _np.array_equal(_np.isnan(fixed_params),
                                   _np.array([v is not None for v in val_params])):
                raise ValueError(f"{name}={val_params} does not exactly overlay {fixed_params}")

            theta += [t for t in val_params if t is not None]   # Delta doesn't need the Nones
        else:
            theta += [t for t in val_params]                    # Keep the Nones

    if verbose:
        print("theta:\t",
              ", ".join(f"{t:.3e}" if isinstance(t, Number) else f"{t}" for t in theta))
    return _np.array(theta, dtype=float)
