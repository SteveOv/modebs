""" TODO """
from typing import Tuple, List, Callable, Union
from numbers import Number
from math import floor as _floor

from warnings import filterwarnings as _filterwarnings, catch_warnings as _catch_warnings

from multiprocessing import Pool as _Pool, cpu_count as _cpu_count, get_context as _get_context

import numpy as _np

from scipy.optimize import minimize as _minimize
from scipy.optimize import OptimizeResult, OptimizeWarning

from emcee import EnsembleSampler

import astropy.units as _u
from uncertainties import UFloat as _UFloat
from uncertainties.unumpy import nominal_values as _noms, std_devs as _std_devs

from deblib import constants as _deblib_const

# GLOBALS used by the fitting functions which require settng prior to calling minimize/mcmc_fit
# Hateful things but this is how we get fast MCMC.
# The way emcee works makes using a class or passing these around in args sloooow!
fixed_theta: _np.ndarray[float]
prior_criteria: _np.ndarray[float]
x: _np.ndarray[float]
y: _np.ndarray[float]
weights: _np.ndarray[float]
flux_func: Callable[[_np.ndarray[float], float, float], _np.ndarray]


def _ln_prior_func(theta: _np.ndarray[float]) -> float:
    """
    The fitting prior function which evaluate the current set of candidate parameters (theta)
    against the prior criteria and returns a single value indicating the goodness of the parameters.

    Accesses the following module wide variables
    - fixed_theta: the corresponding set of fixed parameters; combined with theta to describe system
    - prior_criteria: the prior limits and ratio+/-sigma criteria

    :theta: the current set of candidate fitted parameters which "fill the gaps" in fixed theta
    :returns: a single negative value indicating the goodness of this set of parameters 
    """
    # Limit criteria checks - hard pass/fail on these
    fit_mask = _np.isnan(fixed_theta)
    if not all(lm[0] < t < lm[1] for t, lm in zip(theta, prior_criteria[0][fit_mask], strict=True)):
        return -_np.inf

    # With 3 params per star + dist the #stars is...
    nstars = (fixed_theta.shape[0] - 1) // 3
    if nstars == 1: # no ratios
        return 0

    # We check the ratio wherever a companion value is fitted, or if the primary value is fitted
    # it's all compaions for the parameter type (i.e. if rad0 is fitted all ratio of radii checked)
    # Coalesce the fitted and fixed values, so that we can calculate any required ratios.
    full_theta = fixed_theta.copy()
    _np.putmask(full_theta, fit_mask, theta)

    # Find the components of the ratios are in the coalesced theta. Always 1 less ratio than stars.
    tert_ixs = [i for i, v in enumerate(full_theta) if i % nstars > 0 and v is not None]
    prim_ixs = [int(_np.floor(ix / nstars) * nstars) for ix in tert_ixs]

    # Gaussian priors: g(x) = 1/(σ*sqrt(2*pi)) * exp(-1/2 * (x-µ)^2/σ^2)
    # Omitting scaling expressions for now and note the implicit log cancelling the exp
    inners = []
    for prim_ix, tert_ix in zip(prim_ixs, tert_ixs):
        # Need to evaluate these for the tert, where either the tert or the prim are fitted
        if _np.isnan(fixed_theta[prim_ix]) or _np.isnan(fixed_theta[tert_ix]):
            prior_rat = prior_criteria[1][tert_ix]
            prior_sig = prior_criteria[2][tert_ix]

            inners += [((full_theta[tert_ix] / full_theta[prim_ix] - prior_rat) / prior_sig)**2]
    return -0.5 * _np.sum(inners)


def _ln_likelihood_func(y_model: _np.ndarray[float], degrees_of_freedom: int) -> float:
    """
    The fitting likelihoof function used to evaluate the model y values against the observations,
    returning a single negative value indicating the goodness of the fit.
    
    Based on a weighted chi^2: chi^2_w = 1/(N_obs-n_param) * Σ W(y-y_model)^2

    Accesses the following module wide variables
    - y: the observed y values
    - weights: the weights to apply to each observation/model y value

    :y_model: the model y values
    :degrees_of_freedom: the #observations/#params
    :returns: the goodness of the fit
    """
    chisq = _np.sum(weights * (y - y_model)**2) / degrees_of_freedom
    return -0.5 * chisq


def model_func(theta: _np.ndarray[float],
               combine: bool=True):
    """
    Generate the model fluxes at points x from the candidate parameters theta & fixed_theta.

    flux(*) = model(x, teff, logg) * radius^2 / dist^2

    Accesses the following module wide variables
    - x: the x points to generate model data for
    - fixed_theta: corresponding set of fixed parameters; combined with theta to describe each stars

    :theta: the current set of candidate fitted parameters which "fill the gaps" in fixed theta
    :combine: whether to return a single set of summed fluxes
    :returns: the model fluxes at points x, either per star if combine==False or aggregated
    """
    # Coalesce theta with the underlying fixed params.
    full_theta = fixed_theta.copy()
    _np.putmask(full_theta, _np.isnan(full_theta), theta)

    # The teff, rad and logg for each star is interleaved, so if two stars we expect:
    # [teff0, teff1, rad0, rad1, logg0, logg1, dist]. With 3 params per star + dist the #stars is...
    nstars = (len(fixed_theta) - 1) // 3
    params_by_star = full_theta[:-1].reshape((3, nstars)).transpose()
    y_model = _np.array([
        flux_func(x, teff, logg).value * (rad * _deblib_const.R_sun.n)**2
                                                    for teff, rad, logg in params_by_star
    ])

    # Finally, divide by the dist^2, which is the remaining param not used above
    if combine:
        return _np.sum(y_model, axis=0) / full_theta[-1]**2
    return y_model / full_theta[-1]**2


def objective_func(theta: _np.ndarray[float], minimizable: bool=False) -> float:
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


def minimize_fit(theta0: _np.ndarray[float],
                 methods: List[str]=None,
                 verbose: bool=False) -> OptimizeResult:
    """
    Quick fit model star(s) to the SED with scipy minimize fit of the model data generated from
    a combination of the fixed params on class iniialization and the fitted ones given here.
    Will choose the best performing fit from the algorithms in methods.

    :theta0: the initial set of candidate fitted parameters which "fill the gaps" in fixed theta
    :methods: scipy optimize fitting algorithms to try, defaults to [Nelder-Mead, SLSQP, None]
    :returns: a scipy OptimizeResult with the details of the outcome
    """
    if verbose:
        print("minimize_fit(theta0=[" + ", ".join(f"{t:.6f}" for t in theta0) + "])")

    if methods is None:
        methods = ["Nelder-Mead", "SLSQP", None]
    elif isinstance(methods, str):
        methods = [methods]

    with _catch_warnings(category=[RuntimeWarning, OptimizeWarning]):
        _filterwarnings("ignore", "invalid value encountered in subtract")
        _filterwarnings("ignore",
                        "Desired error not necessarily achieved due to precision loss.")
        _filterwarnings("ignore", "Unknown solver options:")

        the_soln, the_method = None, None
        for method in methods:
            soln = _minimize(objective_func, x0=theta0, args=(True), # minimizable
                             method=method, options={ "maxiter": 5000, "maxfev": 5000 })
            if verbose:
                print(f"({method})",
                        "succeeded" if soln.success else f"failed [{soln.message}]",
                        f"after {soln.nit:d} iterations & {soln.nfev:d} function evaluation(s)",
                        f"[fun = {soln.fun:.6f}]")

            if the_soln is None \
                    or (soln.success and not the_soln.success) \
                    or (soln.fun < the_soln.fun):
                the_soln, the_method = soln, method

    if verbose:
        print(f"The best fit used the '{the_method}' method, yielding theta = [" +
                ", ".join(f"{t:.6f}" for t in the_soln.x) + "]")
    return the_soln


def mcmc_fit(theta0: _np.ndarray[float],
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

    :theta: the initial set of candidate fitted parameters which "fill the gaps" in fixed theta
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

    with _catch_warnings(category=[RuntimeWarning, UserWarning]), _Pool(processes) as pool:

        _filterwarnings("ignore", message="invalid value encountered in scalar subtract")
        _filterwarnings("ignore", message="Using UFloat objects with std_dev==0")

        # Min steps required by Autocorr algo to avoid a warn msg (not a warning so can't filter)
        min_steps_es = int(50 * autocor_tol * thin_by)

        print(f"Running MCMC fit with {nwalkers:d} walkers for {nsteps:d} steps, thinned by",
            f"{thin_by}, on {processes}" if processes else f"up to {_cpu_count()}", "process(es).",
            f"Early stopping is enabled after {min_steps_es:d} steps." if early_stopping else "")
        sampler = EnsembleSampler(int(nwalkers), ndim, objective_func, pool=pool)

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
        nstars: int=2) -> _np.ndarray[object]:
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
        [(teff0_lo,hi),...,(teffN_lo,hi),(rad0_lo,hi),...,(radN_lo,hi),(logg0_lo,hi),...,(loggN_lo,hi),(dist_lo_hi)],
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
    :returns: the newly created criteria array described above
    """
    # pylint: ignore: line-too-long, too-many-arguments, too-many-positional-arguments, too-many-locals, too-many-branches
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
    for cix, (name, value) in enumerate([("teff_limits", teff_limits),
                                        ("radius_limits", radius_limits),
                                        ("logg_limits", logg_limits),
                                        ("dist_limits", dist_limits)]):
        exp_ct = nstars if cix < 3 else 1

        # Attempt to interpret the value as a List[(lower, upper)] * exp_ct
        if isinstance(value, Number|Tuple):
            limit_list += [to_limit_tuple(value, name)] * exp_ct
        elif isinstance(value, List|_np.ndarray) \
                    and len(value) == exp_ct and all(isinstance(v, Number|Tuple) for v in value):
            limit_list += [to_limit_tuple(v, name, vix) for vix, v in enumerate(value)]
        else:
            raise ValueError(f"{name}=={value} cannot be interpreted as List[(low, high)]*{exp_ct}")

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

        # Two ways of getting sigmas; either as sigma of ufloat ratio or from sigma arg (overrides)
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
            raise ValueError(f"{name}_ratios=={value} cannot be interpreted as List[float]*{exp_ct}")

        if sig_val is None:
            pass
        elif isinstance(sig_val, Number):
            sig_list[cix : cix + exp_ct] = [sig_val] * exp_ct
        elif isinstance(sig_val, List|_np.ndarray) \
                and len(sig_val) == exp_ct and all(isinstance(v, Number|None) for v in sig_val):
            sig_list[cix : cix + exp_ct] = [v for v in sig_val]
        else:
            raise ValueError(f"{name}_ratio_sigmas=={value} cannot be interpreted as List[float]*{exp_ct}")

        cix += exp_ct

    criteria[:, :] = [limit_list, rat_list, sig_list]
    return criteria

def create_theta(teffs: Union[List[float], float]=None,
                 radii: Union[List[float], float]=None,
                 loggs: Union[List[float], float]=None,
                 dist: float=None,
                 nstars: int=2,
                 build_delta: bool=False,
                 fixed_theta: _np.ndarray[float]=None) -> _np.ndarray[float]:
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
    return _np.array(theta, dtype=float)




if __name__ == "__main__":
    from pathlib import Path
    import json
    import re

    from uncertainties import ufloat

    from astropy.coordinates import SkyCoord
    from astroquery.simbad import Simbad
    from astroquery.vizier import Vizier
    from astroquery.gaia import Gaia

    from dust_extinction.parameter_averages import G23

    from deblib.constants import M_sun, R_sun
    from deblib.stellar import log_g

    from libs.pyssed import ModelSed
    from libs.pipeline import get_teff_from_spt
    from libs.sed import get_sed_for_target, create_outliers_mask, group_and_average_fluxes

    
    target = "CM Dra"


    # Read the pre-built bt-settl model file
    model_sed = ModelSed("libs/data/pyssed/model-bt-settl-recast.dat")

    # The G23 (Gordon et al., 2023) Milky Way R(V) filter gives us the broadest coverage
    ext_model = G23(Rv=3.1)
    ext_wl_range = _np.reciprocal(ext_model.x_range) * _u.um # x_range has implicit units of 1/micron

    targets_config_file = Path("./config/fitting-a-sed-targets.json")
    with open(targets_config_file, mode="r", encoding="utf8") as f:
        full_dict = json.load(f)
    targets_cfg = { k: full_dict[k] for k in full_dict if full_dict[k].get("enabled", True) }
    target_config = targets_cfg[target]

    target_config.setdefault("loggA", log_g(target_config["MA"] * M_sun, target_config["RA"] * R_sun).n)
    target_config.setdefault("loggB", log_g(target_config["MB"] * M_sun, target_config["RB"] * R_sun).n)

    # Additional data on the target populated with lookups
    target_data = {
        "label": target_config.get("label", target),
        "search_term": target_config.get("search_term", target)
    }

    simbad = Simbad()
    simbad.add_votable_fields("sp", "ids")
    if _tbl := simbad.query_object(target_data["search_term"]):
        target_data["ids"] = _np.array(
            re.findall(r"(Gaia DR3|V\*|TIC|HD|HIP|2MASS)\s+(.+?(?=\||$))", _tbl["ids"][0]),
            dtype=[("type", object), ("id", object)])
        print("IDs:", ", ".join(f"{i['type']} {i['id']}" for i in target_data["ids"]))
        target_data["spt"] = _tbl["sp_type"][0]
        print("SpT:", target_data["spt"])

    # Let's get the Gaia DR3 data on this here object
    gaia_dr3_id = target_data["ids"][target_data["ids"]["type"] == "Gaia DR3"]["id"][0]
    if _job := Gaia.launch_job(f"SELECT TOP 1 * FROM gaiadr3.gaia_source WHERE source_id = {gaia_dr3_id}"):
        _tbl = _job.get_results()
        target_data["parallax_mas"] = ufloat(_tbl["parallax"][0], _tbl["parallax_error"][0])
        target_data["skycoords"] = _coords = SkyCoord(ra=_tbl["ra"][0] * _u.deg,
                                                      dec=_tbl["dec"][0] * _u.deg,
                                                      distance=1000 / _tbl["parallax"][0] * _u.pc,
                                                      frame="icrs")
        print(f"{target} SkyCoords are {_coords} (or {_coords.to_string('hmsdms')})")

    # Lookup the TESS Input Catalog (8.2) for starting "system" Teff and logg values
    target_data["teff_sys"] = get_teff_from_spt(target_data["spt"]) or ufloat(5700, 0)
    target_data["logg_sys"] = ufloat(4.0, 0)
    if _tbl := Vizier(catalog="IV/39/tic82").query_object(target_data["search_term"], radius=0.1 * _u.arcsec):
        if _row := _tbl[0][_tbl[0]["TIC"] in target_data["ids"][target_data["ids"]["type"] == "TIC"]["id"]]:
            # Teff may not be reliable - only use it if it's consistent with the SpT
            if target_data["teff_sys"].n-target_data["teff_sys"].s < (_row["Teff"] or 0) < target_data["teff_sys"].n+target_data["teff_sys"].s:
                target_data["teff_sys"] = ufloat(_row["Teff"], _row.get("s_Teff", None) or 0)
            if (_row["logg"] or 0) > 0:
                target_data["logg_sys"] = ufloat(_row["logg"], _row.get("s_logg", None) or 0)
    
    target_data["k"] = ufloat(target_config.get("k"), target_config.get("k_err", 0) or 0)
    if "light_ratio" in target_config:
        target_data["light_ratio"] = ufloat(target_config.get("light_ratio"),
                                            target_config.get("light_ratio_err", 0) or 0)
    else:
        # If from LC fit we may also need to consider l3; lA=(1-l3)/(1+(LB/LA)) & lB=(1-l3)/(1+1/(LB/LA))
        target_data["light_ratio"] = ufloat(10**(target_config.get("logLB", 1) - target_config.get("logLA", 1)), 0)
    target_data["teff_ratio"] = (target_data["light_ratio"] / target_data["k"]**2)**0.25

    # Estimate the teffs, based on the published system value and the ratio from fitting
    if target_data["teff_ratio"].n <= 1:
        target_data["teffs0"] = [target_data["teff_sys"].n, (target_data["teff_sys"] * target_data["teff_ratio"]).n]
    else:
        target_data["teffs0"]  = [(target_data["teff_sys"] / target_data["teff_ratio"]).n, target_data["teff_sys"].n]

    print(f"{target} system values from lookup and LC fitting:")
    for p, unit in [("teff_sys", _u.K), ("logg_sys", _u.dex), ("k", None), ("teff_ratio", None)]:
        print(f"{p:>12s} = {target_data[p]:.3f} {unit or _u.dimensionless_unscaled:unicode}")
    print(f"      teffs0 = [{', '.join(f'{t:.3f}' for t in target_data['teffs0'])}]")



    # Read in the SED for this target and de-duplicate (measurements may appear multiple times).
    # Work in Jy rather than W/m^2/Hz as they are a more natural unit, giving values that minimize 
    # potential FP rounding. Plots are agnostic and plot wl [um] and vF(v) [W/m^2] on x and y.
    sed = get_sed_for_target(target, target_data["search_term"], radius=0.1, remove_duplicates=True,
                            freq_unit=_u.GHz, flux_unit=_u.Jy, wl_unit=_u.um, verbose=True)

    sed = group_and_average_fluxes(sed, verbose=True)

    # Filter SED to those covered by our models and also remove any outliers
    model_mask = _np.ones((len(sed)), dtype=bool)
    model_mask &= _np.array([model_sed.has_filter(f) for f in sed["sed_filter"]])
    model_mask &= (sed["sed_wl"] >= min(ext_wl_range)) \
                & (sed["sed_wl"] <= max(ext_wl_range)) \
                & (sed["sed_wl"] >= min(model_sed.wavelength_range)) \
                & (sed["sed_wl"] <= max(model_sed.wavelength_range))
    sed = sed[model_mask]

    out_mask = create_outliers_mask(sed, target_data["teffs0"], min_unmasked=15, verbose=True)
    sed = sed[~out_mask]

    sed.sort(["sed_wl"])
    print(f"{len(sed)} unique SED observation(s) retained after range and outlier filtering",
         "\nwith the units for flux, frequency and wavelength being",
        ", ".join(f"{sed[f].unit:unicode}" for f in ["sed_flux", "sed_freq", "sed_wl"]))
    

    # Deredden
    val = 0.000515 # specific to CM Dra
    sed["sed_der_flux"] = sed["sed_flux"] / ext_model.extinguish(sed["sed_wl"].to(_u.um), Ebv=val)


    # SET UP THE GLOBALS

    # If you set teffs, radii, loggs or dist on class creation they're assumed to be
    # fixed. If you set them in calls to fit_*() then they're assumed to be fitted.
    num_stars = 2
    fixed_theta = create_theta(loggs=[target_data["logg_sys"].n] * 2,
                               dist=target_data["skycoords"].distance.to(_u.m).value,
                               nstars=num_stars)
    print("\n\nfixed_theta:\t", ", ".join(f"{t:.3e}" for t in fixed_theta))

    theta0 = create_theta(teffs=target_data["teffs0"],
                          radii=[1.0, 1.0],
                          build_delta=True,
                          fixed_theta=fixed_theta)
    print("theta0:\t\t", ", ".join(f"{t:.3e}" for t in theta0))

    prior_criteria = create_prior_criteria(
                                teff_limits=(2000, 100000),
                                radius_limits=(0.1, 100),
                                logg_limits=(-0.5, 0.6),
                                dist_limits=(100 * _u.Mpc).to(_u.m).value,
                                teff_ratios=target_data["teff_ratio"].n,
                                radius_ratios=target_data["k"].n,
                                logg_ratios=None,
                                teff_ratio_sigmas=max(target_data["teff_ratio"].n * 0.05, target_data["teff_ratio"].s),
                                radius_ratio_sigmas=max(target_data["k"].n * 0.05, target_data["k"].s))
    print(prior_criteria)

    x = model_sed.get_filter_indices(sed["sed_filter"])
    y = sed["sed_der_flux"].quantity.to(_u.Jy).value
    y_err = sed["sed_eflux"].quantity.to(_u.Jy).value
    weights = 1 / y_err**2
    flux_func = model_sed.get_fluxes

    # Quick initial minimize fit
    print()
    soln = minimize_fit(theta0, verbose=True)
    theta_fit = soln.x

    print(f"Best fit parameters for {target} from minimize fit")
    theta_labels = [("TeffA", model_sed.teff_range.unit), ("TeffB", model_sed.teff_range.unit),
                    ("RA", _u.Rsun), ("RB", _u.Rsun)]
    for ix, (l, unit) in enumerate(theta_labels):
        known_val = ufloat(target_config.get(l, _np.NaN), target_config.get(l + "_err", None) or 0)
        print(f"{l:>12s} = {theta_fit[ix]:.3f} {unit:unicode} (known value {known_val:.3f} {unit:unicode})") 


    # MCMC fit, starting from where the minimize fit finished
    print()
    sampler = mcmc_fit(theta_fit, processes=8, verbose=True)
    tau = sampler.get_autocorr_time(c=5, tol=50, quiet=True)
    burn_in_steps = int(max(_np.nan_to_num(tau, copy=True, nan=1000)) * 2)

    # Gets the median fitted values (currently M1, M2 and log(age))
    samples = sampler.get_chain(discard=burn_in_steps, flat=True)
    theta_fit = _np.median(samples[burn_in_steps:], axis=0)
    theta_err_high = _np.quantile(samples[burn_in_steps:], 0.84, axis=0) - theta_fit
    theta_err_low = theta_fit - _np.quantile(samples[burn_in_steps:], 0.16, axis=0)

    print(f"Best fit parameters for {target} from subsequent MCMC fit")
    for ix, (l, unit) in enumerate(theta_labels):
        known_val = ufloat(target_config.get(l, _np.NaN), target_config.get(l + "_err", None) or 0)
        print(f"{l:>12s} = {theta_fit[ix]:.3f} +/- {theta_err_high[ix]:.3f}/{theta_err_low[ix]:.3f}",
            f"{unit:unicode} (known value {known_val:.3f} {unit:unicode})") 