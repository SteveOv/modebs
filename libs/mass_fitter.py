""" Fitting library to derive masses from known sys_mass, radii & teffs """
from typing import List, Tuple, Union, Callable
from warnings import filterwarnings, catch_warnings
from multiprocessing import Pool
from pathlib import Path
from inspect import getsourcefile

import numpy as np

from uncertainties import UFloat
from uncertainties.unumpy import uarray

from scipy.interpolate import RBFInterpolator
from scipy.optimize import minimize, OptimizeResult, OptimizeWarning
from emcee import EnsembleSampler
from emcee.autocorr import AutocorrError
from sed_fit.fitter import samples_from_sampler

from .data.mist.read_mist_models import ISO

MIN_PHASE = 0 # MS
MAX_PHASE = 2 # RGB

_this_dir = Path(getsourcefile(lambda:0)).parent
ISO_FILE = _this_dir / "data/mist/MIST_v1.2_vvcrit0.4_basic_isos" \
                                                / "MIST_v1.2_feh_p0.00_afe_p0.0_vvcrit0.4_basic.iso"
iso = ISO(f"{ISO_FILE}", verbose=True)

# Build up the known datapoints and corresponding radius & teff values.
# Get the linear values for these so that we can perform interpolation in linear space.
ages_list = []
eep_list = []
masses_list = []
radii_list = []
teffs_list = []
logg_list = []
for log_age in sorted(iso.ages):
    iso_block = iso.isos[iso.age_index(log_age)]
    iso_block = iso_block[(iso_block["phase"] >= MIN_PHASE) & (iso_block["phase"] <= MAX_PHASE)]
    if (new_rows := len(iso_block)) > 0:
        mass_sort = np.argsort(iso_block["star_mass"])

        # Points/axes
        ages_list += [10**log_age] * new_rows
        eep_list += list(iso_block[mass_sort]["EEP"])
        masses_list += list(iso_block[mass_sort]["star_mass"])

        # corresponding values
        radii_list += list(10**iso_block[mass_sort]["log_R"])
        teffs_list += list(10**iso_block[mass_sort]["log_Teff"])
        logg_list += list(iso_block[mass_sort]["log_g"])

# Create the interpolators for radius and teff; using RBF interpolation as we have irregular data.
x = np.array(list(zip(ages_list, masses_list)), dtype=float)
radius_interp = RBFInterpolator(x, radii_list, neighbors=2**x.ndim, smoothing=5, kernel="linear")
teff_interp = RBFInterpolator(x, teffs_list, neighbors=2**x.ndim, smoothing=5, kernel="linear")
logg_interp = RBFInterpolator(x, logg_list, neighbors=2**x.ndim, smoothing=5, kernel="linear")

x = np.array(list(zip(eep_list, masses_list)), dtype=float)
age_interp = RBFInterpolator(x, ages_list, neighbors=2**x.ndim, smoothing=5, kernel="linear")

# Priors based on the data
age_limits = (min(ages_list), max(ages_list))
mass_limits = (min(masses_list), max(masses_list))

del x, ages_list, masses_list, radii_list, teffs_list, eep_list, iso

def get_age_limits():
    """ Get the lower and upper bounds of the ages within the model. """
    return age_limits

def get_mass_limits():
    """ Get the lower and upper bounds of the massed within the model. """
    return mass_limits

def _ln_prob_func(fit_theta: np.ndarray[float],
                  ln_prior_func: Callable[[np.ndarray[float]], float],
                  ln_likelihood_func: Callable[[np.ndarray[float]], float]) -> float:
    """
    The MCMC function which returns the log posterior probability; the probability that the
    candidate params (theta) are those responsible for the observations. This is a negative
    value tending towards zero as the probability increases. Think of this as:

    ln(P(posterior)) = ln(P(prior) * P(likelihood)) = _ln_prior_func() + _ln_likelihood_func()

    This takes the current set of fitted params (fit_theta) and merges them with the fixed params.
    The resulting param set (theta) is first evaluated by the prior function then a model is
    generated from them, with model_func, which is then evaluated against the observations with
    the likelihood function. The ln(product) of the two values is returned.

    :fit_theta: current set of candidate fitted parameters only (those given by the _fit_mask)
    :ln_prior_func: a callback function to evaluate the current theta against prior criteria
    :ln_likelihood_func: a callback function to evaluate the goodness of fit of model vs observation
    :returns: the result of evaluating the fitted model against the observations
    """
    retval = ln_prior_func(fit_theta)

    # The "model func": gets the radii & teffs from stars' masses from MIST model via interpolators
    if np.isfinite(retval):
        masses, age = fit_theta[:-1], 10**fit_theta[-1]
        xi = np.array([(age, m) for m in masses])
        model_radii = radius_interp(xi)
        model_teffs = teff_interp(xi)
        model_loggs = logg_interp(xi)
        if any(model_radii <= 0) or any(model_teffs <= 0): # Out of range
            retval = -np.inf

    # The "likelihood func": evaluates the "goodness" of radii & teffs match with the observations
    if np.isfinite(retval):
        retval += ln_likelihood_func(np.concatenate([model_radii, model_teffs, model_loggs]))
    return retval

def minimize_fit(theta0: np.ndarray[float],
                 ln_prior_func: Callable[[np.ndarray[float]], float],
                 ln_likelihood_func: Callable[[np.ndarray[float]], float],
                 methods: List[str]=None,
                 verbose: bool=False) -> Tuple[np.ndarray[float], OptimizeResult]:
    """
    A quick 'minimize' fit to find stars' mass and shared log10(age) values
    by fitting corresponding radii & teffs from MIST models to observed values.

    :theta0: the initial set of candidate mass (M_sun) and log10(age/yr) values
    :ln_prior_func: a callback function to evaluate the current theta against prior criteria
    :ln_likelihood_func: a callback function to evaluate the goodness of fit of model vs observation
    :methods: scipy optimize fitting algorithms to try, defaults to [Nelder-Mead, SLSQP, None]
    :returns: the fitted set of mass & log10(age) values and
    a scipy OptimizeResult with the full details of the fitting outcome
    """
    if methods is None:
        methods = ["Nelder-Mead", "SLSQP", None]
    elif isinstance(methods, str):
        methods = [methods]

    max_iters = int(1000 * len(theta0))
    with catch_warnings(category=[RuntimeWarning, OptimizeWarning]):
        filterwarnings("ignore", "overflow encountered in scalar power")
        filterwarnings("ignore", "invalid value encountered in ")
        filterwarnings("ignore", "Desired error not necessarily achieved due to precision loss.")
        filterwarnings("ignore", "Unknown solver options:")

        best_soln, best_meth = None, None
        for method in methods:
            soln = minimize(lambda *args: -_ln_prob_func(*args),
                            x0=theta0, args=(ln_prior_func, ln_likelihood_func),
                            method=method, options={ "maxiter": max_iters, "maxfev": max_iters })
            if verbose:
                print(f"({method})", "succeeded" if soln.success else f"failed [{soln.message}]",
                      f"after {soln.nit:d} iterations & {soln.nfev:d} function evaluation(s)",
                      f"[fun = {soln.fun:.6f}]")

            if best_soln is None \
                    or (soln.success and not best_soln.success) \
                    or (soln.success == best_soln.success and soln.fun < best_soln.fun):
                best_soln, best_meth = soln, method

        if best_soln is not None and best_soln.success:
            theta0 = best_soln.x
            if verbose:
                print(f"Taking the best fit from the {best_meth} method")
        else:
            print("The fit failed so returning input")
    return theta0, best_soln

def mcmc_fit(theta0: np.ndarray[float],
             ln_prior_func: Callable[[np.ndarray[float]], float],
             ln_likelihood_func: Callable[[np.ndarray[float]], float],
             nwalkers: int=100,
             nsteps: int=100000,
             thin_by: int=1,
             seed: int=42,
             processes: int=1,
             autocor_tol: int=50,
             early_stopping: bool=True,
             early_stopping_from: int=None,
             early_stopping_threshold: float=0.01,
             progress: Union[bool, str]=False,
             verbose: bool=False) -> Tuple[np.ndarray[UFloat], EnsembleSampler]:
    """
    A full fit to find stars' mass and shared log10(age) values from MCMC sampling
    and fitting corresponding radii & teffs from MIST models to observed values.

    :theta0: the initial set of candidate mass (M_sun) and log10(age/yr) values
    :ln_prior_func: a callback function to evaluate the current theta against prior criteria
    :ln_likelihood_func: a callback function to evaluate the goodness of fit of model vs observation
    :nwalkers: the number of mcmc walkers to employ
    :nsteps: the maximium number of mcmc steps to make for each walker
    :thin_by: step interval to inspect fit progress and yield samples
    :seed: optional seed for random behaviour
    :processes: optional number of parallel processes to use, or None to let code choose
    :autocor_tol: the autocorrelation tolerance
    :early_stopping: stop fitting if solution has converged & further improvements are negligible
    :early_stopping_from: override of the number of steps before early stopping is considered
    :early_stopping_threshold: the delta(tau) threshold below which to consider early stopping
    :progress: whether to show a progress bar (see emcee documentation for other values)
    :returns: the fitted set of mass and log(age) values as UFloats with 1-sigma uncertainties
    and an EnsembleSampler with the full details of the fitting outcome
    """

    # Using thin_by is fiddly; the sampler will execute iterations*thin_by steps, so to process the
    # requested nsteps we need to set iterations=(nsteps // thin_by) when creating the sampler and
    # subsequently account for this factor on other calls to the same sampler.
    rng = np.random.default_rng(seed)
    ndim = len(theta0)
    tau = [np.inf] * ndim

    # Min steps before the Autocorr algo becomes useful & unlikely to give a chain too short error
    if early_stopping_from is None or early_stopping_from <= 0:
        early_stopping_from = int(50 * ndim * autocor_tol)

    # Starting positions for the walkers clustered around theta0, via priors to ensure they're valid
    p0, test_theta = [], theta0.copy()
    while len(p0) < int(nwalkers):
        test_theta = theta0 + (theta0 * rng.normal(0, 0.05, ndim))
        if np.isfinite(ln_prior_func(test_theta)):
            p0 += [test_theta]

    with Pool(processes=processes) as pool, catch_warnings(category=[RuntimeWarning, UserWarning]):
        filterwarnings("ignore", message="invalid value encountered in ")
        filterwarnings("ignore", message="Using UFloat objects with std_dev==0")

        if verbose:
            print(f"A MCMC fit on {processes} process(es) with {nwalkers:d} walkers for {nsteps:d}",
                  (f"steps, sampling every {thin_by:d} steps." if thin_by > 1 else "steps."))
            if early_stopping:
                print(f"Early stopping will be considered after {early_stopping_from:d} steps.")


        sampler = EnsembleSampler(int(nwalkers), ndim, _ln_prob_func,
                                  args=(ln_prior_func, ln_likelihood_func), pool=pool)
        step = 0
        for _ in sampler.sample(initial_state=p0, iterations=nsteps // thin_by,
                                thin_by=thin_by, tune=True, progress=progress):
            step = sampler.iteration * thin_by
            if early_stopping and step % 1000 == 0:
                try:
                    # The autocor time (tau) is the #steps to effectively forget start position.
                    # As the fit converges the change in tau will tend towards zero.
                    prev_tau, tau = tau, sampler.get_autocorr_time(c=5, tol=autocor_tol) * thin_by
                    if step >= early_stopping_from \
                            and not any(np.isnan(tau)) \
                            and all(tau < step / 100) \
                            and all(abs(prev_tau - tau) / prev_tau < early_stopping_threshold):
                        break
                except AutocorrError:
                    # The chain is too short. Can set the quiet arg to True in which case a warning
                    # message is output (but not a Python warning). Cleaner to consume the error.
                    pass

        if verbose and early_stopping and 0 < step < nsteps:
            print(f"Halting MCMC after {step:d} steps as the walkers are past",
                   "100 times the autocorrelation time & the fit has converged.")

    samples = samples_from_sampler(sampler, autocor_tol, thin_by, flat=True, verbose=verbose)
    fit_nom = np.median(samples, axis=0)
    fit_err_high = np.quantile(samples, 0.84, axis=0) - fit_nom
    fit_err_low = fit_nom - np.quantile(samples, 0.16, axis=0)
    return uarray(fit_nom, np.mean([fit_err_high, fit_err_low], axis=0)), sampler


def log_age_for_mass_and_eep(mass: float, eep: int=300) -> float:
    """
    An approximate log10(age) for the requested mass and equivalent evolutionary point (EEP).
    Within the same phases range as the interpolators used for radii & masses for the model func.

    :mass: the requested mass (solMass)
    :eep: the equivalent evolutionay point (EEP)
    :returns: the log(age) of the nearest mass
    """
    return np.log10(age_interp([(eep, mass)])[0])
