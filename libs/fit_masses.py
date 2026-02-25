""" Prototype for using fitting to derive masses from known sys_mass, radii & teffs """
from typing import List, Tuple, Union
from warnings import filterwarnings, catch_warnings
from multiprocessing import Pool
from pathlib import Path
from inspect import getsourcefile

import numpy as np

from uncertainties import UFloat
from uncertainties.unumpy import uarray, nominal_values, std_devs

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

# Create the interpolators for radius and teff; using RBF interpolation as we have irregular data.
x = np.array(list(zip(ages_list, masses_list)), dtype=float)
radius_interp = RBFInterpolator(x, radii_list, neighbors=2**x.ndim, smoothing=5, kernel="linear")
teff_interp = RBFInterpolator(x, teffs_list, neighbors=2**x.ndim, smoothing=5, kernel="linear")

x = np.array(list(zip(eep_list, masses_list)), dtype=float)
age_interp = RBFInterpolator(x, ages_list, neighbors=2**x.ndim, smoothing=5, kernel="linear")

# Priors based on the data
age_limits = (min(ages_list), max(ages_list))
mass_limits = (min(masses_list), max(masses_list))

del x, ages_list, masses_list, radii_list, teffs_list, eep_list, iso

def _objective_func(theta: np.ndarray[float],
                    sys_mass: UFloat,
                    obs_radii: np.ndarray[UFloat],
                    obs_teffs: np.ndarray[UFloat],
                    minimizable: bool=False) -> float:
    """
    Optimizable objective function combining a _ln_prior_func, model_func and _ln_likelihood_func
    to evaluate theta values consisting of stellar masses and age by obtaining the corresponding
    stellar radii & teffs from MIST models and fitting these to the observed values.
    """
    retval = 0
    masses, age = theta[:-1], 10**theta[-1]

    # The "prior func": absolute yes/no handling of limits & Gaussian prior on total mass
    if not age_limits[0] <= age <= age_limits[1] \
        or not all(mass_limits[0] <= mass <= mass_limits[1] for mass in masses):
        retval = np.inf
    else:
        retval = 0.5 * ((sys_mass.n - np.sum(masses)) / sys_mass.s)**2

    # The "model func": gets the radii & teffs from stars' masses from MIST model via interpolators
    if np.isfinite(retval):
        xi = np.array([(age, m) for m in masses])
        model_radii = radius_interp(xi)
        model_teffs = teff_interp(xi)
        if any(model_radii <= 0) or any(model_teffs <= 0): # Out of range
            retval = np.inf

    # The "likelihood func": evaluates the "goodness" of radii & teffs match with the observations
    if np.isfinite(retval):
        degr_free = len(theta)
        chisq = 0
        for obs_vals, model_vals in [(obs_radii, model_radii),
                                     (obs_teffs, model_teffs)]:
            weights = 1 / std_devs(obs_vals)**2
            chisq += np.sum(weights * (nominal_values(obs_vals) - model_vals)**2)
        retval += 0.5 * chisq / degr_free

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
    A quick 'minimize' fit to find stars' mass and shared log10(age) values
    by fitting corresponding radii & teffs from MIST models to observed values.

    :theta0: the initial set of candidate mass (M_sun) and log10(age/yr) values
    :sys_mass: the total mass of the system for use as a prior constraint (M_sun)
    :radii: the observed stellar radii to fit against (R_sun)
    :teffs: the observed stellar effective temperatures to fit against (K)
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
            soln = minimize(_objective_func, x0=theta0, args=(sys_mass, radii, teffs, True),
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
             sys_mass: UFloat,
             radii: np.ndarray[UFloat],
             teffs: np.ndarray[UFloat],
             nwalkers: int=100,
             nsteps: int=100000,
             thin_by: int=1,
             seed: int=42,
             processes: int=1,
             autocor_tol: int=50,
             early_stopping: bool=True,
             early_stopping_threshold: float=0.01,
             progress: Union[bool, str]=False,
             verbose: bool=False) -> Tuple[np.ndarray[UFloat], EnsembleSampler]:
    """
    A full MCMC fit to find stars' mass and shared log10(age) values
    by fitting corresponding radii & teffs from MIST models to observed values.

    :theta0: the initial set of candidate mass (M_sun) and log10(age/yr) values
    :sys_mass: the total mass of the system for use as a prior constraint (M_sun)
    :radii: the observed stellar radii to fit against (R_sun)
    :teffs: the observed stellar effective temperatures to fit against (K)
    :nwalkers: the number of mcmc walkers to employ
    :nsteps: the maximium number of mcmc steps to make for each walker
    :thin_by: step interval to inspect fit progress and yield samples
    :seed: optional seed for random behaviour
    :processes: optional number of parallel processes to use, or None to let code choose
    :autocor_tol: the autocorrelation tolerance
    :early_stopping: stop fitting if solution has converged & further improvements are negligible
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
    min_steps_before_es = int(50 * ndim * autocor_tol)

    # Starting positions for the walkers clustered around theta0
    p0 = [theta0 + (theta0 * rng.normal(0, 0.05, ndim)) for _ in np.arange(int(nwalkers))]

    with Pool(processes=processes) as pool, catch_warnings(category=[RuntimeWarning, UserWarning]):
        filterwarnings("ignore", message="invalid value encountered in ")
        filterwarnings("ignore", message="Using UFloat objects with std_dev==0")

        sampler = EnsembleSampler(int(nwalkers), ndim,
                                  _objective_func, args=(sys_mass, radii, teffs), pool=pool)
        step = 0
        for _ in sampler.sample(initial_state=p0, iterations=nsteps // thin_by,
                                thin_by=thin_by, tune=True, progress=progress):
            step = sampler.iteration * thin_by
            if early_stopping and step % 1000 == 0:
                try:
                    # The autocor time (tau) is the #steps to effectively forget start position.
                    # As the fit converges the change in tau will tend towards zero.
                    prev_tau, tau = tau, sampler.get_autocorr_time(c=5, tol=autocor_tol) * thin_by
                    if step >= min_steps_before_es \
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
