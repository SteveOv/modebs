
"""
Low level utility functions for SED ingest, pre-processing, estimation and fitting.
"""
from typing import Union
from pathlib import Path
import re
from urllib.parse import quote_plus

# pylint: disable=no-member
import astropy.units as u
from astropy.table import Table
from uncertainties import UFloat
import numpy as np
from scipy.optimize import minimize

from deblib.constants import c, h, k_B
from deblib.vmath import exp

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


def create_outliers_mask(sed: Table, teff_ratio: float=1.0) -> np.ndarray[bool]:
    """
    WIP
    """
    mask = np.zeros((len(sed)), dtype=bool)

    # - perform a scipy minimize fit on target (incl scaling model to observations)
    freq = sed["sed_freq"].to(u.Hz).value
    flux = sed["sed_flux"].to(u.W / u.m**2 / u.Hz).value
    flux_err = sed["sed_eflux"].to(u.W / u.m**2 / u.Hz).value
    teff1, teff2 = 5000, 5000 * teff_ratio
    teff1, teff2 = _minimize_fit_sed(freq, flux, flux_err, teff1, teff2, teff_ratio)
    y_model = np.add(blackbody_flux(freq, teff1), blackbody_flux(freq, teff2))

    # - calculate chi^2 of fit
    # - while (chi^2 values exist > threshold and #remaining obs > minimum)
    #   - mask remaining obs with highest chi^2

    return mask

def blackbody_flux(freq: Union[float, UFloat, np.ndarray[float], np.ndarray[UFloat]],
                   teff: Union[float, UFloat],
                   radius: float=1.) -> np.ndarray[float]:
    """
    Calculates the Blackbody / Planck function fluxes of a body of the requested temperature [K]
    at the requested frequencies [Hz] over an area defined by the radius in arcseconds.

    The fluxes are given in units of W / m^2 / Hz. Multiply them 1e26 for the equivalent in Jy.
 
    :freq: the frequency/ies in Hz
    :teff: the temperature in K
    :radius: the area radius in arcseconds
    :returns: the blackbody fluxes at freq, in W / m^2 / Hz
    """
    area = 2 * np.pi * (radius / 206265)**2 # radius in arcsec where 206265 arcsec = 1 rad
    part1 = 2 * h * freq**3 / c**2
    part2 = exp((h * freq) / (k_B * teff)) - 1
    return area * part1 / part2


def _minimize_fit_sed(x, y, y_err=None, init_teff1=5000., init_teff2=5000., teff_ratio=1.0):
    """
    WIP
    """
    # Get the x, y and y_err (that latter normalized)
    y_min = y.min()
    y_scale = y.max() - y_min
    y = (y - y_min) / y_scale
    y_err = 1 if y_err is None else y_err / y_scale
    teff_flex = teff_ratio * 0.05

    def _similarity_func(theta):
        # Get the model fluxes and normalize them (y and y_err are already normalized)
        (teff1, teff2) = theta
        y_model = np.add(blackbody_flux(x, teff1), blackbody_flux(x, teff2))
        y_min = y_model.min()
        y_model = (y_model - y_min) / (y_model.max() - y_min)
        if 3000 < teff1 < 30000 and np.abs((teff2 / teff1) - teff_ratio) <= teff_flex:
            return 0.5 * np.sum(((y - y_model) / y_err)**2)
        return np.inf

    soln = minimize(_similarity_func, x0=(init_teff1, init_teff2))
    return soln.x[0], soln.x[1]
