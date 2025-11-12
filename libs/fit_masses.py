""" Prototype for using fitting to derive masses from known sys_mass, radii & teffs """
from typing import List, Tuple, Union
from warnings import filterwarnings, catch_warnings
from multiprocessing import Pool
from pathlib import Path
from inspect import getsourcefile

import numpy as np

from uncertainties import UFloat
from uncertainties.unumpy import uarray, nominal_values, std_devs

from scipy.interpolate import RegularGridInterpolator, make_interp_spline
from scipy.optimize import minimize, OptimizeResult, OptimizeWarning
from emcee import EnsembleSampler

from .data.mist.read_mist_models import ISO

MIN_PHASE = 0 # MS
MAX_PHASE = 2 # RGB

_this_dir = Path(getsourcefile(lambda:0)).parent
ISO_FILE = _this_dir / "data/mist/MIST_v1.2_vvcrit0.4_basic_isos" \
                                                / "MIST_v1.2_feh_p0.00_afe_p0.0_vvcrit0.4_basic.iso"
iso = ISO(f"{ISO_FILE}", verbose=True)
log_ages = np.array([ab["log10_isochrone_age_yr"][0] for ab in iso.isos \
                        if ab["phase"][0] >= MIN_PHASE or ab["phase"][-1] <= MAX_PHASE])

grid_ages = 10**log_ages
age_limits = (min(grid_ages), max(grid_ages))

grid_masses = np.geomspace(0.1, 20, 100, endpoint=True)
mass_limits = (min(grid_masses), max(grid_masses))
rad_grid = np.zeros((len(grid_ages), len(grid_masses)), dtype=float)
teff_grid = np.zeros((len(grid_ages), len(grid_masses)), dtype=float)


for age_ix, log_age in enumerate(sorted(log_ages)):
    iso_block = iso.isos[iso.age_index(log_age)]
    mass_order = np.argsort(iso_block["star_mass"])
    x = iso_block[mass_order]["star_mass"]

    # Populate grids with interpolated values for radius and teff at the grid masses
    interp_rads = make_interp_spline(x, y=10**iso_block[mass_order]["log_R"], k=1)(grid_masses)
    rad_grid[age_ix] = interp_rads
    # if any(interp_rads < 0):
    #     print(f"NEG RAD for age {log_age} and masses:",
    #           ",".join(f"{m:.3f}" for m in grid_masses[interp_rads < 0]))

    interp_teffs = make_interp_spline(x, y=10**iso_block[mass_order]["log_Teff"], k=1)(grid_masses)
    teff_grid[age_ix] = interp_teffs

points = (grid_ages, grid_masses)
radius_interp = RegularGridInterpolator(points=points, values=rad_grid, method="slinear")
teff_interp = RegularGridInterpolator(points=points, values=teff_grid, method="slinear")
del iso, rad_grid, teff_grid


def _objective_func(theta: np.ndarray[float],
                    sys_mass: UFloat,
                    obs_radii: np.ndarray[UFloat],
                    obs_teffs: np.ndarray[UFloat],
                    minimizable: bool=False) -> float:
    """
    Optimizable objective function combining a _ln_prior_func, model_func and _ln_likelihood_func
    to evaluates theta values consisting of stellar masses and age via MIST models to observed
    stellar radii and teffs.
    """
    retval = 0
    masses, age = theta[:-1], 10**theta[-1]

    # The "prior func"
    if not grid_ages[0] <= age <= age_limits[1] \
        or not all(mass_limits[0] <= mass <= mass_limits[1] for mass in masses):
        retval = np.inf
    else:
        # Gaussian prior on the total masses
        retval = ((sys_mass.n - np.sum(masses)) / sys_mass.s)**2
        retval *= 0.5

    # The "model func"
    model_radii = np.zeros_like(masses, float)
    model_teffs = np.zeros_like(masses, float)
    if np.isfinite(retval):
        try:
            for ix, mass in enumerate(masses):
                model_radii[ix] = radius_interp(xi=(age, mass))
                model_teffs[ix] = teff_interp(xi=(age, mass))
            if any(model_radii <= 0) or any(model_teffs <= 0):
                # Out of range
                retval = np.inf
        except ValueError:
            retval = np.inf

    # The "likelihood func"
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
        min_steps_es = int(50 * ndim * autocor_tol * thin_by)

        sampler = EnsembleSampler(int(nwalkers), ndim,
                                  _objective_func, args=(sys_mass, radii, teffs), pool=pool)
        step = 0
        for _ in sampler.sample(initial_state=p0, iterations=nsteps // thin_by,
                                thin_by=thin_by, tune=True, progress=progress):
            if early_stopping:
                step = sampler.iteration * thin_by
                if step > min_steps_es and step % 1000 == 0:
                    # The autocor time (tau) is the steps to effectively forget start position.
                    # As the fit converges the change in tau will tend towards zero. We set tol=0
                    # to prevent chain-too-short warning while we're expliciting testing the fit.
                    prev_tau = tau
                    tau = sampler.get_autocorr_time(c=5, tol=0, quiet=True) * thin_by
                    if not any(np.isnan(tau)) \
                            and all(tau < step / 100) \
                            and all(abs(prev_tau - tau) / prev_tau < 0.03):
                        break

        if early_stopping and 0 < step < nsteps:
            print(f"Halting MCMC after {step:d} steps as the walkers are past",
                    "100 times the autocorrelation time & the fit has converged.")

        tau = sampler.get_autocorr_time(c=5, tol=autocor_tol, quiet=True) * thin_by
        burn_in_steps = int(max(np.nan_to_num(tau, copy=True, nan=1000)) * 2)
        samples = sampler.get_chain(discard=burn_in_steps, thin=thin_by, flat=True)

        if verbose:
            print( "Autocorrelation steps (tau):", ", ".join(f"{t:.3f}" for t in tau))
            print(f"Estimated burn-in steps:     {int(max(np.nan_to_num(tau, nan=1000)) * 2):,}")
            print(f"Mean Acceptance fraction:    {np.mean(sampler.acceptance_fraction):.3f}")

        # Get theta into ufloats with std_dev based on the mean +/- 1-sigma values (where fitted)
        fit_nom = np.median(samples[burn_in_steps:], axis=0)
        fit_err_high = np.quantile(samples[burn_in_steps:], 0.84, axis=0) - fit_nom
        fit_err_low = fit_nom - np.quantile(samples[burn_in_steps:], 0.16, axis=0)
        theta_fit = uarray(fit_nom, np.mean([fit_err_high, fit_err_low], axis=0))

    return theta_fit, sampler
