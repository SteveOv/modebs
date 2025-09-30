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

from deblib import constants as _deblib_const

class SedFitter:
    """
    Class for fitting stellar models to SEDS

    Great in theory as it encapsulates a lot of fiddly code, but unfortunately the way emcee
    multiprocessing works makes this too damn slow. Setting processes beyond 2 tends to make
    things even worse, slowing things down more.
    
    Getting rid of self reference and changing the prior, model, prob and objective func to
    static gives only a marginal improvement. Functional code is usually ~4 times quicker.
    """
    def __init__(
            self,
            x, y, y_err,
            flux_func: Callable[[any], _np.ndarray[float]],
            teffs: Union[float, List[float]]=None,
            radii: Union[float, List[float]]=None,
            loggs: Union[float, List[float]]=None,
            dist: Union[float, _u.Quantity]=None,
            teff_limits: Union[tuple[float, float], Tuple[_u.Quantity, _u.Quantity]]=(2000, 20000),
            radius_limits: Union[tuple[float, float], Tuple[_u.Quantity, _u.Quantity]]=(0.1, 100),
            logg_limits: Union[tuple[float, float], Tuple[_u.Quantity, _u.Quantity]]=(0.0, 5.0),
            dist_limits: Union[tuple[float, float], Tuple[_u.Quantity, _u.Quantity]]=(0.0, _np.inf),
            teff_ratios: Union[float, List[float]]=1,
            radius_ratios: Union[float, List[float]]=1,
            logg_ratios: Union[float, List[float]]=1,
            nstars: int=2,
            verbose: bool=False
        ):
        """
        Do some stuff
        """
        # pylint: disable=line-too-long
        self._x = x
        self._y = y
        self._flux_func = flux_func
        self._nstars = nstars
        self._verbose = verbose

        # These are the chisq weights. By using 1 / sigma^2 we get the reduced chisq value.
        self._weights = 1 / y_err**2

        # The teffs, radii, loggs and dist args represent fixed values for these parameters.
        # If supplied, the teffs, radii & loggs should have 1 or nstars items.
        self._fixed_theta = _np.array(self._validate_and_build_theta(teffs, radii, loggs,  dist),
                                      dtype=float)
        self._fixed_theta_fitted_ixs = _np.where(_np.isnan(self._fixed_theta))[0]

        # TODO: validate and fix up the priors
        teff_ratios = [teff_ratios]*(nstars-1) if isinstance(teff_ratios, Number|None) else teff_ratios
        radius_ratios = [radius_ratios]*(nstars-1) if isinstance(radius_ratios, Number|None) else radius_ratios
        logg_ratios = [logg_ratios]*(nstars-1) if isinstance(logg_ratios, Number|None) else logg_ratios

        # We must have theta as 1-d list/array as scipy minimize will not accept 2-d array and
        # to make lookups easier this means making the corresponding prior arrays the same shape.
        # These are quick to lookup while iterating at the expense of duplication & complexity here.
        self._prior_limits = _np.empty_like(self._fixed_theta, dtype=object)
        self._prior_limits[:] = [ssl for sl in [[teff_limits]*nstars, [radius_limits]*nstars, [logg_limits]*nstars, [dist_limits]] for ssl in sl]
        self._prior_ratios = _np.empty_like(self._fixed_theta, dtype=float)
        self._prior_ratios[:] = [ssl for sl in [[1], teff_ratios, [1], radius_ratios, [1], logg_ratios, [1]] for ssl in sl]
        self._prior_ratio_sigmas = _np.empty_like(self._fixed_theta, dtype=float)
        self._prior_ratio_sigmas[:] = [None if r is None else r * 0.05 for r in self._prior_ratios]     

    def _ln_prior_func(self, theta):
        """
        Evaluate the current set of candidate parameters, theta, against the prior criteria

        :theta: 2-d set of params as [[teffs], [radii], [logg], dist]] - axis 1 for each component &
        non-fitted values None (i.e.: [[1000, 1000], None, [4, 4], None] if radii & dist not fitted)
        """

        if not all(lim[0] < th < lim[1] for th, lim
                    in zip(theta, self._prior_limits[self._fixed_theta_fitted_ixs], strict=True)):
            return -_np.inf

        # Need to check the ratio wherever a tertiary value is fitted, or if the primary value is
        # fitted it's all of the parameter type (i.e. if rad0 is fitted all ratio of radii checked)
        # Coalesce the fitted and fixed values, so that we can calculate any ratios required
        full_theta = self._fixed_theta.copy()
        _np.putmask(full_theta, _np.isnan(full_theta), theta)

        # work out where the components of the ratios are in the coalesced theta
        tert_ixs = [i for i, v in enumerate(full_theta) if i % self._nstars > 0 and v is not None]
        prim_ixs = [int(_np.floor(ix / self._nstars) * self._nstars) for ix in tert_ixs]

        inners = []
        for prim_ix, tert_ix in zip(prim_ixs, tert_ixs):
            # Need to evaluate these for the tert, where either the tert or the prim are fitted
            if _np.isnan(self._fixed_theta[prim_ix]) or _np.isnan(self._fixed_theta[tert_ix]):
                prior_rat = self._prior_ratios[tert_ix]
                prior_sig = self._prior_ratio_sigmas[tert_ix]

                # Gaussian priors: g(x) = 1/(σ*sqrt(2*pi)) * exp(-1/2 * (x-µ)^2/σ^2)
                # Omitting scaling expressions for now and note the implicit log cancelling the exp
                inners += [((full_theta[tert_ix] / full_theta[prim_ix] - prior_rat) / prior_sig)**2]

        return -0.5 * _np.sum(inners)

    def model_func(self, theta, combine: bool=True):
        """
        Generate the model fluxes from the candidate parameters, theta, and any fixed parameters.

        :theta: 2-d set of params - axis 0 for component stars, axis 1 for parameters
        :combine: whether to return a single set of summed fluxes
        """
        # Coalesce theta with the underlying fixed params. The teff, rad and logg for each star
        # is interleaved, so if two stars we expect: [teff0, teff1, rad0, rad1, logg0, logg1, dist]
        full_theta = self._fixed_theta.copy()
        _np.putmask(full_theta, _np.isnan(full_theta), theta)

        params_by_star = full_theta[:-1].reshape((3, self._nstars)).transpose()
        y_model = _np.array([
            self._flux_func(self._x, teff, logg).value * (rad * _deblib_const.R_sun.n)**2
                                                        for teff, rad, logg in params_by_star
        ])

        # Finally, divide by the dist^2, which is the remaining param not used above
        if combine:
            return _np.sum(y_model, axis=0) / full_theta[-1]**2
        return y_model / full_theta[-1]**2

    def _ln_likelihood_func(self, y_model, degrees_of_freedom: int=7):
        """ chi^2_w = 1/(N_obs-n_param) * Σ W(y-y_model)^2 """
        chisq = _np.sum(self._weights * (self._y - y_model)**2) / degrees_of_freedom
        return -0.5 * chisq

    def _objective_func(self, theta, degrees_of_freedom: int=7, minimizable: bool=False) -> float:
        """
        The function to be optimized by adjusting theta so that the return value converges to zero.

        :theta: 2-d set of params - axis 0 for component stars, axis 1 for parameters
        :degrees_of_freedom:
        :returns:
        """
        if _np.isfinite(retval := self._ln_prior_func(theta)):
            retval += self._ln_likelihood_func(self.model_func(theta, True), degrees_of_freedom)
            _np.nan_to_num(retval, copy=False, nan=-_np.inf)

        if minimizable:
            return -retval
        return retval

    def _validate_and_build_theta(self, teffs, radii, loggs, dist, build_delta: bool=False):
        """
        Will validate the teffs, radii, loggs and dist values and build up a theta list from them.
        This is the set of parameters needed to generate a model SED from self._nstars components.

        The resulting theta list will have the form:
            theta = [teff0, ... , teffn, rad0, ... , radn, logg0, ..., loggn, dist]

        where n is the value of self._nstars -1. The build_delta argument controls how the function
        handles None values for teffs, radii, loggs and dist. If nstars==2 and:
        - build_delta=True: Nones will be omitted and the resulting theta will be compressed down
            - i.e.: if loggs=None-> theta = [teff0, teff1, rad0, rad1, dist]
        - build_delta=False: Nones will be included as placeholders for later insertion
            - i.e.: if loggs=None -> theta = [teff0, teff1, rad0, rad1, None, None, dist]

        In addition to the args, this func depends on self._fixed_delta and self._nstars

        :teffs: either a list of floats nstars long, a single float (same for each) or None
        :rads: either a list of floats nstars long, a single float (same for each) or None
        :loggs: either a list of floats nstars long, a single float (same for each) or None
        :dist: either a float, Quantity or None
        :build_delta: whether to compress out Nones (True) or not (False)
        :returns: the resulting theta list
        """
        # pylint: disable=no-member

        # This will grow
        theta = []
        for ix, (name, value) in enumerate([
            ("teffs", teffs), ("radii", radii), ("loggs", loggs), ("dist", dist)
        ]):
            val_params = None
            expected_count = self._nstars if ix < 3 else 1

            # Attempt to interpret the value
            if value is None or isinstance(value, Number):
                val_params = [value] * expected_count
            elif isinstance(value, Tuple|List) \
                    and len(value) == expected_count \
                    and all(isinstance(v, Number|None) for v in value):
                val_params = value
            else:
                raise ValueError(f"{name} cannot be interpreted as List[Number]*{expected_count}")

            if build_delta:
                # If building the delta, then we expect to fill in the bits missing from fixed_theta
                f_ix = ix * self._nstars
                fix_params_needing_value = _np.isnan(self._fixed_theta[f_ix : f_ix+expected_count])
                val_params_with_value = _np.array([v is not None for v in val_params])

                if _np.array_equal(fix_params_needing_value, val_params_with_value):
                    # Squeeze out the Nones
                    theta += [t for t in val_params if t is not None]
                else:
                    raise ValueError(f"{name} does not supply the expected missing values")
            else:
                theta += val_params
        return theta


    def minimize_fit(self,
                     teffs: Union[float, List[float]]=None,
                     radii: Union[float, List[float]]=None,
                     loggs: Union[float, List[float]]=None,
                     dist: Union[float, _u.Quantity]=None,
                     methods: List[str]=None) -> OptimizeResult:
        """
        Quick fit model star(s) to the SED with scipy minimize fit of the model data generated from
        a combination of the fixed params on class iniialization and the fitted ones given here.
        Will choose the best performing fit from the algorithms in methods.

        :teffs: effective temps for fitting, or None if supplied as fixed values on initialization
        :radii: stellar radii for fitting, or None if supplied as fixed values on initialization
        :loggs: setllar log(g)s for fitting, or None if supplied as fixed values on initialization
        :dist: system distance for fitting, or None if supplied as fixed values on initialization
        :methods: scipy optimize fitting algorithms to try, defaults to [Nelder-Mead, SLSQP, None]
        :returns: a scipy OptimizeResult with the details of the outcome
        """
        theta0 = self._validate_and_build_theta(teffs, radii, loggs, dist, build_delta=True)
        if self._verbose:
            print("mcmc_fit(theta0=[" + ", ".join(f"{t:.6f}" for t in theta0) + "])")

        if methods is None:
            methods = ["Nelder-Mead", "SLSQP", None]
        elif isinstance(methods, str):
            methods = [methods]

        with _catch_warnings(category=[RuntimeWarning, OptimizeWarning]):
            _filterwarnings("ignore", "invalid value encountered in subtract")
            _filterwarnings("ignore",
                            "Desired error not necessarily achieved due to precision loss.")
            _filterwarnings("ignore", "Unknown solver options:")

            # Fixed args for objective_func; degrees_of_freedom [#obs - #fitted params], minimizable
            args = (self._x.shape[0] - len(theta0), True)

            the_soln, the_method = None, None
            for method in methods:
                soln = _minimize(self._objective_func, x0=theta0, args=args,
                                 method=method, options={ "maxiter": 5000, "maxfev": 5000 })
                if self._verbose:
                    print(f"({method})",
                          "succeeded" if soln.success else f"failed [{soln.message}]",
                          f"after {soln.nit} iterations & {soln.nfev} function evaluation(s)",
                          f"[fun = {soln.fun:.6f}]")

                if the_soln is None \
                        or (soln.success and not the_soln.success) \
                        or (soln.fun < the_soln.fun):
                    the_soln, the_method = soln, method

        if self._verbose:
            print(f"The best fit used the '{the_method}' method, yielding theta = [" +
                  ", ".join(f"{t:.6f}" for t in the_soln.x) + "]")
        return the_soln


    def mcmc_fit(self,
                 teffs: Union[float, List[float]]=None,
                 radii: Union[float, List[float]]=None,
                 loggs: Union[float, List[float]]=None,
                 dist: Union[float, _u.Quantity]=None,
                 nwalkers: int=100,
                 niters: int=100000,
                 thin_by: int=10,
                 seed: int=42,
                 processes: int=1) -> EnsembleSampler:
        """
        Full fit model star(s) to the SED with an MCMC fit of the model data generated from
        a combination of the fixed params on class iniialization and the fitted ones given here.

        Will run up to niters iterations. Every 1000 iterations will check if the fit has
        converged and will stop early if that is the case

        :teffs: effective temps for fitting, or None if supplied as fixed values on initialization
        :radii: stellar radii for fitting, or None if supplied as fixed values on initialization
        :loggs: setllar log(g)s for fitting, or None if supplied as fixed values on initialization
        :dist: system distance for fitting, or None if supplied as fixed values on initialization
        :nwalker: the number of mcmc walkers to employ
        :niters: the maximium number of mcmc iterations to run
        :thin_by: step interval to inspect fit progress
        :seed: optional seed for random behaviour
        :processes: optional number of parallel processes to use, or None to let code choose
        :returns: a emcee EnsembleSampler with the details of the outcome
        """
        theta0 = self._validate_and_build_theta(teffs, radii, loggs, dist, build_delta=True)
        if self._verbose:
            print("mcmc_fit(theta0=[" + ", ".join(f"{t:.6f}" for t in theta0) + "])")

        rng = _np.random.default_rng(seed)
        ndim = len(theta0)
        autocor_tol = 50 / thin_by
        tau = [_np.inf] * ndim

        # Starting positions for the walkers clustered around theta0
        p0 = [theta0 + (theta0 * rng.normal(0, 0.05, ndim)) for _ in _np.arange(int(nwalkers))]

        # Fixed args for objective_func; degrees_of_freedom [#obs - #fitted params], minimizable
        args = (self._x.shape[0] - len(theta0), False)

        with _catch_warnings(category=[RuntimeWarning, UserWarning]), _Pool(processes) as pool:

            _filterwarnings("ignore", message="invalid value encountered in scalar subtract")
            _filterwarnings("ignore", message="Using UFloat objects with std_dev==0")

            print(f"Running MCMC for up to {niters:,} iterations with {nwalkers} walkers with",
                  f"{processes}" if processes else f"up to {_cpu_count()}", "process(es)")
            sampler = EnsembleSampler(int(nwalkers), ndim, self._objective_func,
                                      args=args, pool=pool)

            # Autocorr algo requires at least this many steps otherwise you
            # get warning text (not a warning so can't control it)
            min_steps_autocor = 50 * autocor_tol * thin_by

            for _ in sampler.sample(initial_state=p0, iterations=niters // thin_by,
                                    thin_by=thin_by, tune=True, progress=True):
                
                if (step := sampler.iteration * thin_by) > min_steps_autocor and step % 1000 == 0:
                    # The autocor time (tau) is the steps to effectively forget start position.
                    # As the fit converges the change in tau will tend towards zero.
                    prev_tau = tau
                    tau = sampler.get_autocorr_time(c=5, tol=autocor_tol, quiet=True) * thin_by
                    if not any(_np.isnan(tau)) \
                        and all(tau < step / 100) \
                        and all(abs(prev_tau - tau) / prev_tau < 0.01):
                        if self._verbose:
                            print(f"Halting MCMC after {step:,} steps as we're past",
                                    "100 times the autocorrelation time & the fit has converged.")
                        break
                    prev_tau = tau

        if self._verbose:
            tau = sampler.get_autocorr_time(c=5, tol=autocor_tol, quiet=True) * thin_by
            print( "Autocorrelation steps (tau):", ", ".join(f"{t:.3f}" for t in tau))
            print(f"Estimated burn-in steps:     {int(max(_np.nan_to_num(tau, nan=1000)) * 2):,}")
            print(f"Mean Acceptance fraction:    {_np.mean(sampler.acceptance_fraction):.3f}")

        return sampler
