
"""
Low level utility functions for SED ingest, pre-processing, estimation and fitting.
"""
from typing import Union, Tuple, Callable
from pathlib import Path
import re
from urllib.parse import quote_plus
from numbers import Number

# pylint: disable=no-member
import astropy.units as u
from astropy.table import Table
from uncertainties import UFloat
import numpy as np
from scipy.optimize import minimize

from deblib.constants import c, h, k_B
from deblib.vmath import exp, log10

def get_sed_for_target(target: str,
                       search_term: str=None,
                       radius: float=0.1,
                       missing_uncertainty_ratio: float=0.1,
                       flux_unit=u.W / u.m**2 / u.Hz,
                       freq_unit=u.Hz,
                       wl_unit=u.micron) -> Table:
    """
    Gets spectral energy distribution (SED) observations for the target. These data are found and
    downloaded from the VizieR photometry tool (see http://viz-beta.u-strasbg.fr/vizier/sed/doc/).

    The data are sorted and errorbars based on missing_uncertainty_ratio are set where none given
    (sed_eflux is either zero or NaN). The sed_flux, sed_eflux and sed_freq fields will be converted
    to the requested unit if necessary.

    Calculated fields are added for sed_wl (wavelength), sed_vfv and sed_evfv (freq * flux) to aid
    plotting, where x and y axes of wavelength and nu*F(nu) are often used.

    Tables will be locally cached within the `.cache/.sed/` directory for future requests.

    :target: the name of the target object
    :search_term: optional search term, or leave as None to use the target value
    :radius: the search radius in arcsec
    :missing_uncertainty_rate: uncertainty, as a ratio of the fluxes, to apply where none recorded
    :flux_unit: the unit of the returned sed_flux field (must support conversion from u.Jy)
    :freq_unit: the unit of the returned sed_freq field
    :wl_unit: the unit of the returned sed_wl field
    :returns: an astropy Table containing the chosen data, sorted by descending frequency
    """
    # pylint: disable=too-many-arguments, too-many-positional-arguments
    sed_cache_dir = Path(".cache/.sed/")
    sed_cache_dir.mkdir(parents=True, exist_ok=True)

    # Read in the SED for this target via the cache (filename includes both search criteria)
    sed_fname = sed_cache_dir / (re.sub(r"[^\w\d-]", "-", target.lower()) + f"-{radius}.vot")
    if not sed_fname.exists():
        try:
            targ = quote_plus(search_term or target)
            sed = Table.read(f"https://vizier.cds.unistra.fr/viz-bin/sed?-c={targ}&-c.rs={radius}")
            sed.write(sed_fname, format="votable") # votable matches that published in link above
        except ValueError as err:
            raise ValueError(f"No SED for target={target} and search_term={search_term}") from err

    sed = Table.read(sed_fname)
    sed.sort(["sed_freq"], reverse=True)

    # Set flux uncertainties where none given
    mask_no_err = (sed["sed_eflux"].value == 0) | np.isnan(sed["sed_eflux"])
    sed["sed_eflux"][mask_no_err] = sed["sed_flux"][mask_no_err] * missing_uncertainty_ratio

    # Get the data into desired units
    if sed["sed_flux"].unit != flux_unit:
        sed["sed_flux"] = sed["sed_flux"].to(flux_unit)
        sed["sed_eflux"] = sed["sed_eflux"].to(flux_unit)
    if sed["sed_freq"].unit != freq_unit:
        sed["sed_freq"] = sed["sed_freq"].to(freq_unit)

    # Add wavelength which we may use for plots
    sed["sed_wl"] = np.divide(c * u.m / u.s, sed["sed_freq"]).to(wl_unit)

    # Add vF(v) columns. There's an issue here; we have to explicitly set each new
    # column's unit otherwise the unit from the first source column appears to be copied.
    sed["sed_vfv"] = (sed["sed_freq"].value * sed["sed_flux"].value) \
                        * (sed["sed_freq"].unit * sed["sed_flux"].unit)
    sed["sed_evfv"] = (sed["sed_freq"].value * sed["sed_eflux"].value) \
                        * (sed["sed_freq"].unit * sed["sed_eflux"].unit)
    return sed


def create_outliers_mask(sed: Table, temps0: Tuple=(5000, 5000)) -> np.ndarray[bool]:
    """
    WIP
    """
    mask = np.zeros((len(sed)), dtype=bool)

    # - perform an initial teff fit on target and get the resulting model
    if isinstance(temps0, Number):
        temps0 = (temps0,)
    temp_ratio = temps0[-1] / temps0[0]
    temp_flex = temp_ratio * 0.05
    def priors_func(ts):
        this_tratio = 1 if len(ts) == 1 else ts[-1] / ts[0]
        return all(3000 <= t <= 30000 for t in ts) and abs(this_tratio - temp_ratio) <= temp_flex
    temps, y_model = quick_blackbody_fit(sed["sed_freq"], sed["sed_flux"], sed["sed_eflux"],
                                         temps0, priors_func)
    print(temps)
    print(y_model)

    # - calculate chi^2 of fit
    # - while (chi^2 values exist > threshold and #remaining obs > minimum)
    #   - mask remaining obs with highest chi^2

    return mask

def blackbody_flux(freq: Union[float, UFloat, np.ndarray[float], np.ndarray[UFloat]],
                   temp: Union[float, UFloat],
                   radius: float=1.) -> np.ndarray[float]:
    """
    Calculates the Blackbody / Planck function fluxes of a body of the requested temperature [K]
    at the requested frequencies [Hz] over an area defined by the radius in arcseconds.

    The fluxes are given in units of W / m^2 / Hz. Multiply them 1e26 for the equivalent in Jy.
 
    :freq: the frequency/ies in Hz
    :temp: the temperature in K
    :radius: the area radius in arcseconds
    :returns: the blackbody fluxes at freq, in W / m^2 / Hz
    """
    area = 2 * np.pi * (radius / 206265)**2 # radius in arcsec where 206265 arcsec = 1 rad
    part1 = 2 * h * freq**3 / c**2
    part2 = exp((h * freq) / (k_B * temp)) - 1
    return area * part1 / part2


def quick_blackbody_fit(x: np.ndarray,
                        y: np.ndarray,
                        y_err: np.ndarray,
                        temps0: Tuple,
                        priors_func: Callable[[any], bool]=None,
                        method: str=None) -> Tuple[Tuple, np.ndarray]:
    """
    Perform a quick fit on the passed SED data (x, y and y_err) by minimizing a function which
    which sums one or more sets of blackbody fluxes, as defined by one or more temperatures.
    During fitting, the model fluxes are scaled to the same magnitude as the those of the input.
    The temperatures and scaled combined fluxes of the best fitting model are returned.

    :x: the SED frequencies
    :y: the SED fluxes at the frequencies of x [in units of W / m^2 / Hz or Jy]
    :y_err: the uncertainties of y in the same unit
    :temps0: the initial set of temperatures [in k] from which blackbody spectra will be generated
    at the frequencies, x, and summed to produce the initial candidate model
    :priors_func: a boolean function to evaluate each set of temperatures against known priors,
    returning True or False to indicate whether they conform to known prior conditions or not
    :method: the minimizing method to use - see scipy minimize documentation for options
    :returns: final, best fit set of temperatures and resulting scaled flux values (in input units)
    """
    # pylint: disable=too-many-arguments, too-many-positional-arguments
    flux_unit = y.unit
    x = x.to(u.Hz).value
    y = y.value
    y_log = log10(y)                 # We scale model to sed observations within log space
    y_err = 1 if y_err is None else (y_err.value + 1e-30) # avoid div0 errors
    if isinstance(temps0, Number):
        temps0 = (temps0,)

    def _scaled_combined_bb_flux(temps):
        y_model_log = log10(np.sum([blackbody_flux(x, temp) for temp in temps], axis=0))
        if flux_unit == u.Jy:
            y_model_log += 26
        return 10**(y_model_log + np.median(y_log - y_model_log))

    def _similarity_func(temps):
        if priors_func is None or priors_func(temps):
            y_model = _scaled_combined_bb_flux(temps)
            return 0.5 * np.sum(((y - y_model) / y_err)**2)
        return np.inf

    soln = minimize(_similarity_func, x0=temps0, method=method)
    return tuple(soln.x), _scaled_combined_bb_flux(soln.x) * flux_unit
