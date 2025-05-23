""" Functions supporting the MCMC analysis of SED curves to estimate dEB stellar masses """
from typing import Callable, Tuple

import numpy as np

from deblib.constants import c, h, k_B
from deblib.vmath import exp

# pylint: disable=invalid-name, too-many-arguments, too-many-positional-arguments

# TODO: need a better way of handling this - static cctor?
from libs.mistisochrones import MistIsochrones
mist_isos = MistIsochrones(metallicities=[0])


def blackbody_model(x: np.ndarray,
                    Teff1: float,
                    Teff2: float,
                    logg1: float=None,
                    logg2: float=None) \
        -> Tuple[np.ndarray, np.ndarray]:
    """
    Model a SED on the Planck blackbody function of two stars of the given effective temperatures.

    :x: the frequencies at which the SED observations are required
    :Teff1: the effective temperature of star 1 in K
    :Teff2: the effective temperature of star 2 in K
    :logg1: the surface gravity of star 1 in log(cgs) (not used)
    :logg2: the surface gravity of star 2 in log(cgs) (not used)
    :returns: the sed fluxes at x for each of the two stars as a tuple (sed1, sed2)
    """
    # pylint: disable=unused-argument
    def bb_spec_brightness(teff):
        """
        Calculate the BB spectral brightness at effective temp T and frequencies x with;
        B(x, T) = (2hx^3)/c^2 * 1/(exp(hx/kT)-1)  [W / m^2 / Hz / sr]

        Params teff and x are floats in units of K and Hz, respectively
        """
        pt1 = (2 * h * x**3) / c**2
        pt2 = exp((h * x) / (k_B * teff)) - 1
        return pt1 / pt2
    return (bb_spec_brightness(Teff1), bb_spec_brightness(Teff2))


def ln_like(x: np.ndarray,
            y: np.ndarray,
            y_err: np.ndarray,
            Teff1: float,
            Teff2: float,
            logg1: float,
            logg2: float,
            model_func: Callable=blackbody_model):
    """
    The MCMC likelihood function, which returns a comparison of the x & y (+/-y_err)
    observations with the model produced by the model_func based on the Teffs and loggs.

    Is calculated as 1/2 Σ((y - y_model)/y_err)^2
    
    :x: the frequencies at which the SED observations were made
    :y: the fluxes of the SED observations
    :y_err: the uncertainties of the flux observations
    :Teff1: the effective temperature of star 1 in K
    :Teff2: the effective temperature of star 2 in K
    :logg1: the surface gravity of star 1 in log(cgs)
    :logg2: the surface gravity of star 2 in log(cgs)
    :returns: the calculated likelihood value
    """
    # Compare both in min-max normalized form (y is already normalized)
    y_model_norm = min_max_normalize(np.add(*model_func(x, Teff1, Teff2, logg1, logg2)))
    return -0.5 * np.sum(((y - y_model_norm) / y_err)**2)


def ln_prior(M1: float, M2: float, age: float, k: float):
    """
    The MCMC log prior function which evaluates the properties of the stars defined by
    the current masses and age against known prior constraints; which are k, the previously
    fitted ratio of the stellar radii, and the mass and phase range that constrain the param space.
    """
    # pylint: disable=too-many-locals
    MIN_MASS, MAX_MASS = 0.1, 270.
    MIN_AGE, MAX_AGE = 5.0, 10.3        # log(age) range in MIST data
    MIN_PHASE, MAX_PHASE = 0.0, 2.0     # MIST main sequence to RGB phases
    MIST_PARAMS = ["Teff", "log_g", "R", "phase"]

    Teff1, Teff2, logg1, logg2 = None, None, None, None
    retval = -np.inf # failure

    # Basic validation of priors; lookup won't work if these are out of range of MIST values
    if MIN_MASS <= M1 <= MAX_MASS and MIN_MASS <= M2 <= MAX_MASS and MIN_AGE <= age <= MAX_AGE:
        try:
            # Get the T_eff values which we need to generate a SED and to validate the parameters
            Teff1, logg1, R1, ph1 = mist_isos.stellar_params_for_mass(0, age, M1, MIST_PARAMS)
            Teff2, logg2, R2, ph2 = mist_isos.stellar_params_for_mass(0, age, M2, MIST_PARAMS)

            # Validate the stellar params against the priors
            if np.abs((R2 / R1) - k) < 0.1 \
                and min(ph1, ph2) >= MIN_PHASE and max(ph1, ph2) <= MAX_PHASE:
                retval = 0 # params conform to the priors
        except ValueError:
            pass
    return retval, (Teff1, Teff2, logg1, logg2)


def ln_prob(theta: Tuple[float, float, float],
            x: np.ndarray,
            y: np.ndarray,
            y_err: np.ndarray,
            k: float,
            model_func: Callable[[np.ndarray, float, float, float, float],
                                 Tuple[np.ndarray, np.ndarray]]=blackbody_model):
    """
    The ln_prob function to be called by emcee.EnsembleSample to fit models to a SED.
    Will create the call stack of MCMC ln_prior() and ln_like() functions using the
    chosen model_func to support it.

    :theta: the current walker position of the (M1, M2, log_age) parameters
    :x: the frequencies at which the SED observations were made
    :y: the fluxes of the SED observations
    :y_err: the uncertainties of the flux observations
    :k: the ratio of the stellar radii prior
    :model_func: function to produce the model pairs of SEDs to be evaluated. This should have the
    form func(x, Teff1, Teff2, logg1, logg2) -> (sed1, sed2) and defaults to blackbody_model()
    """
    # theta == M1, M2, log_age, blob = Teff1, Teff2, logg1, logg2
    lp, blob = ln_prior(*theta, k)
    if np.isfinite(lp):
        return lp + ln_like(x, y, y_err, *blob, model_func), blob
    return -np.inf, blob


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
