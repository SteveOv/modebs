""" Prototype for using fitting to derive masses from known sys_mass, radii & teffs """
from typing import List, Tuple, Union
from warnings import filterwarnings, catch_warnings
from multiprocessing import Pool

import numpy as np

from uncertainties import UFloat
from uncertainties.unumpy import uarray, nominal_values, std_devs

from scipy.optimize import minimize, OptimizeResult, OptimizeWarning
from emcee import EnsembleSampler

from .mistisochrones import MistIsochrones, Phase

mist_isos = MistIsochrones(metallicities=[0])
log_ages = mist_isos.list_ages(feh=0, max_phase=Phase.RGB)

def _objective_func(theta: np.ndarray[float],
                    sys_mass: UFloat,
                    obs_radii: np.ndarray[float],
                    obs_teffs: np.ndarray[float],
                    minimizable: bool=False) -> float:
    """
    Optimizable objective function combining a _ln_prior_func, model_func and _ln_likelihood_func
    to evaluates theta values consisting of stellar masses and age via MIST models to observed
    stellar radii and teffs.
    """
    retval = 0
    masses, log_age = theta[:-1], theta[-1]

    # The "prior func"
    if not 5.0 <= log_age <= 1.03e1 or not all(1e-1 <= mass <= 100 for mass in masses):
        retval = np.inf
    else:
        # Gaussian prior on the total masses
        retval = 0.5 * ((sys_mass.n - np.sum(masses)) / sys_mass.s)**2

    # The "model func"
    model_radii = np.zeros_like(masses, float)
    model_teffs = np.zeros_like(masses, float)
    if np.isfinite(retval):
        # TODO: find nearest log_age (need to replace with proper interpolation)
        log_age = log_ages[(np.abs(log_ages - log_age)).argmin()]

        for ix, mass in enumerate(masses):
            try:
                model_vals = mist_isos.stellar_params_for_mass(feh=0, log_age=log_age, mass=mass,
                                                               params=["R", "Teff"])
                model_radii[ix] = model_vals[0]
                model_teffs[ix] = model_vals[1]
            except ValueError:
                retval = np.inf

    # The "likelihood func"
    if np.isfinite(retval):
        degr_free = len(theta)
        chisq = 0
        for obs_vals, model_vals in [(obs_radii, model_radii), (obs_teffs, model_teffs)]:
            weights = 1 / std_devs(obs_vals)**2
            chisq += np.sum(weights * (nominal_values(obs_vals) - model_vals)**2) / degr_free
        retval += 0.5 * chisq

    if minimizable != (retval >= 0):
        return -retval
    return retval

def minimize_fit(theta0: np.ndarray[float],
                 sys_mass: UFloat,
                 radii: np.ndarray[UFloat],
                 teffs: np.ndarray[UFloat],
                 methods: List[str]=None,
                 verbose: bool=False) -> Tuple[np.ndarray[float], OptimizeResult]:
    """
    Quick fit model masses via MIST models to the supplied sys_mass, radii and teffs.

    :theta0: the initial set of candidate parameters for the model SED
    :sys_mass: the observed total mass of the systems
    :radii: the observed stellar radii
    :teffs: the observed stellar effective temperatures    
    :methods: scipy optimize fitting algorithms to try, defaults to [Nelder-Mead, SLSQP, None]
    :returns: the final set of parameters & a scipy OptimizeResult with the details of the outcome
    """
    if methods is None:
        methods = ["Nelder-Mead", "SLSQP", None]
    elif isinstance(methods, str):
        methods = [methods]

    max_iters = int(1000 * len(theta0))

    with catch_warnings(category=[RuntimeWarning, OptimizeWarning]):
        filterwarnings("ignore", "invalid value encountered in ")
        filterwarnings("ignore", "Desired error not necessarily achieved due to precision loss.")
        filterwarnings("ignore", "Unknown solver options:")

        the_soln, the_meth = None, None
        for method in methods:
            a_soln = minimize(_objective_func, x0=theta0, args=(sys_mass, radii, teffs, True),
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

    if the_soln is not None and the_soln.success:
        theta0 = the_soln.x
        if verbose:
            print(f"Taking the best fit from the {the_meth} method")
    else:
        print("The fit failed so returning input")
    return theta0, a_soln

def mcmc_fit(theta0: np.ndarray[float],
             sys_mass: UFloat,
             radii: np.ndarray[UFloat],
             teffs: np.ndarray[UFloat],
             nwalkers: int=100,
             nsteps: int=100000,
             thin_by: int=10,
             seed: int=42,
             processes: int=1,
             early_stopping: bool=True,
             progress: Union[bool, str]=False,
             verbose: bool=False) -> Tuple[np.ndarray[UFloat], EnsembleSampler]:
    """
    Full fit model masses via MIST models to the supplied sys_mass, radii and teffs.

    :theta0: the initial set of candidate parameters for the model SED
    :sys_mass: the observed total mass of the systems
    :radii: the observed stellar radii
    :teffs: the observed stellar effective temperatures
    :nwalkers: the number of mcmc walkers to employ
    :nsteps: the maximium number of mcmc steps to make for each walker
    :thin_by: step interval to inspect fit progress
    :seed: optional seed for random behaviour
    :processes: optional number of parallel processes to use, or None to let code choose
    :early_stopping: stop fitting if solution has converged & further improvements are negligible
    :progress: whether to show a progress bar (see emcee documentation for other values)
    :returns: fitted set of parameters as UFloats and an EnsembleSampler with details of the outcome
    """
    rng = np.random.default_rng(seed)
    ndim = len(theta0)
    autocor_tol = 50 / thin_by
    tau = [np.inf] * ndim

    # Starting positions for the walkers clustered around theta0
    p0 = [theta0 + (theta0 * rng.normal(0, 0.05, ndim)) for _ in np.arange(int(nwalkers))]

    with Pool(processes=processes) as pool, catch_warnings(category=[RuntimeWarning, UserWarning]):
        filterwarnings("ignore", message="invalid value encountered in ")
        filterwarnings("ignore", message="Using UFloat objects with std_dev==0")

        # Min steps required by Autocorr algo to avoid a warn msg (not a warning so can't filter)
        min_steps_es = int(50 * autocor_tol * thin_by)

        sampler = EnsembleSampler(int(nwalkers), ndim,
                                  _objective_func, args=(sys_mass, radii, teffs), pool=pool)
        step = 0
        for _ in sampler.sample(initial_state=p0, iterations=nsteps // thin_by,
                                thin_by=thin_by, tune=True, progress=progress):
            if early_stopping:
                step = sampler.iteration * thin_by
                if step > min_steps_es and step % 1000 == 0:
                    # The autocor time (tau) is the steps to effectively forget start position.
                    # As the fit converges the change in tau will tend towards zero.
                    prev_tau = tau
                    tau = sampler.get_autocorr_time(c=5, tol=autocor_tol, quiet=True) * thin_by
                    if not any(np.isnan(tau)) \
                            and all(tau < step / 100) \
                            and all(abs(prev_tau - tau) / prev_tau < 0.01):
                        break

        if early_stopping and 0 < step < nsteps:
            print(f"Halting MCMC after {step:d} steps as the walkers are past",
                    "100 times the autocorrelation time & the fit has converged.")

        tau = sampler.get_autocorr_time(c=5, tol=autocor_tol, quiet=True) * thin_by
        burn_in_steps = int(max(np.nan_to_num(tau, copy=True, nan=1000)) * 2)
        samples = sampler.get_chain(discard=burn_in_steps, flat=True)

        # Get theta into ufloats with std_dev based on the mean +/- 1-sigma values (where fitted)
        fit_nom = np.median(samples[burn_in_steps:], axis=0)
        fit_err_high = np.quantile(samples[burn_in_steps:], 0.84, axis=0) - fit_nom
        fit_err_low = fit_nom - np.quantile(samples[burn_in_steps:], 0.16, axis=0)
        theta_fit = uarray(fit_nom, np.mean([fit_err_high, fit_err_low], axis=0))

    if verbose:
        print( "Autocorrelation steps (tau):", ", ".join(f"{t:.3f}" for t in tau))
        print(f"Estimated burn-in steps:     {int(max(np.nan_to_num(tau, nan=1000)) * 2):,}")
        print(f"Mean Acceptance fraction:    {np.mean(sampler.acceptance_fraction):.3f}")

    return theta_fit, sampler
