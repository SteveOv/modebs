
"""
Low level utility functions for SED ingest, pre-processing, estimation and fitting.
"""
# pylint: disable=no-member, multiple-statements
from typing import Union, Tuple, List
import warnings
from pathlib import Path
import re
from urllib.parse import quote_plus
from numbers import Number

import astropy.units as u
from astropy.table import Table, Column, unique
from astropy.io.votable import parse_single_table
from uncertainties import UFloat, unumpy
import numpy as np
from scipy.optimize import minimize

from deblib.constants import c, h, k_B
from deblib.vmath import exp, log10

def get_sed_for_target(target: str,
                       search_term: str=None,
                       radius: float=0.1,
                       missing_uncertainty_ratio: float=0.1,
                       remove_duplicates: bool=False,
                       flux_unit=u.W / u.m**2 / u.Hz,
                       freq_unit=u.Hz,
                       wl_unit=u.micron,
                       verbose: bool=False) -> Table:
    """
    Gets spectral energy distribution (SED) observations for the target. These data are found and
    downloaded from the VizieR photometry tool (see http://viz-beta.u-strasbg.fr/vizier/sed/doc/).
    
    The VizieR photometry tool is developed by Anne-Camille Simon and Thomas Boch.

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
    :remove_duplicates: if True, only the first row for each combination of sed_filter, sed_freq,
    sed_flux and sed_eflux will be included in the returned table
    :flux_unit: the unit of the returned sed_flux field (must support conversion from u.Jy)
    :freq_unit: the unit of the returned sed_freq field
    :wl_unit: the unit of the returned sed_wl field
    :verbose: whether to output diagnostics messages
    :returns: an astropy Table containing the chosen data, sorted by descending frequency
    """
    # pylint: disable=too-many-arguments, too-many-positional-arguments, too-many-locals
    sed_cache_dir = Path(".cache/.sed/")
    sed_cache_dir.mkdir(parents=True, exist_ok=True)

    # Read in the SED for this target via the cache (filename includes both search criteria)
    sed_fname = sed_cache_dir / (re.sub(r"[^\w\d-]", "-", target.lower()) + f"-{radius}.vot")
    if not sed_fname.exists():
        if verbose: print(f"Table {sed_fname.name} not cached so will query the VizieR SED service")
        try:
            targ = quote_plus(search_term or target)
            sed = Table.read(f"https://vizier.cds.unistra.fr/viz-bin/sed?-c={targ}&-c.rs={radius}")
            sed.write(sed_fname, format="votable") # votable matches that published in link above
        except ValueError as err:
            raise ValueError(f"No SED for target={target} and search_term={search_term}") from err

    # Read first/only table in votable & parse into a stock astropy Table (more consistent to use)
    sed = parse_single_table(sed_fname).to_table()
    sed.sort(["sed_freq"], reverse=True)
    rcount = len(sed)
    if verbose: print(f"Opened SED table {sed_fname.name} containing {rcount} row(s).")

    # Set flux uncertainties where none given
    mask_no_err = (sed["sed_eflux"].value == 0) | np.isnan(sed["sed_eflux"])
    sed["sed_eflux"][mask_no_err] = sed["sed_flux"][mask_no_err] * missing_uncertainty_ratio

    # Get the data into desired units
    if sed["sed_flux"].unit != flux_unit:
        sed["sed_flux"] = sed["sed_flux"].to(flux_unit)
        sed["sed_eflux"] = sed["sed_eflux"].to(flux_unit)
    if sed["sed_freq"].unit != freq_unit:
        sed["sed_freq"] = sed["sed_freq"].to(freq_unit)

    if remove_duplicates:
        sed = unique(sed, keys=["sed_filter", "sed_freq", "sed_flux", "sed_eflux"], keep="first")
        ucount = len(sed)
        if verbose: print(f"Dropped {rcount-ucount} duplicate(s) leaving {ucount} unique row(s).")
        sed.sort(["sed_freq"], reverse=True)

    # Add wavelength which we may be useful downstream
    sed["sed_wl"] = np.divide(c * u.m / u.s, sed["sed_freq"].to(u.Hz)).to(wl_unit)
    return sed


def calculate_vfv(sed: Table,
                  freq_colname: str="sed_freq",
                  flux_colname: str="sed_flux",
                  flux_err_colname: str="sed_eflux") -> Tuple[Column, Column]:
    """
    Calculate the nu*F(nu) columns which are often plotted in place of raw flux/flux err values.
    The columns are not added directly to the table but may be by client code, if required.
    For example:
    ```python
    sed["sed_vfv"], sed["sed_evfv"] = calculate_vfv(sed)
    ```

    :sed: the table which is the source of the fluxes
    :freq_colname: the name of the frequency column to use
    :flux_colname: the name of the flux column to use
    :flex_err_colname: the name of the flux uncertainty column to use
    :returns: a tuple of new astropy Columns with values (sed_freq * sed_flux, sed_freq * sed_eflux)
    """
    freqs, fluxes, flux_errs = sed.columns[freq_colname, flux_colname, flux_err_colname].values()
    vfv = freqs * fluxes
    vfv.unit = freqs.unit * fluxes.unit    # Fix the unit otherwise it'll only use that of freq
    evfv = freqs * flux_errs
    evfv.unit = freqs.unit * fluxes.unit   # Fix the unit otherwise it'll only use that of freq
    return vfv, evfv


def group_and_average_fluxes(sed: Table,
                             group_by_colnames: List[str] = ["sed_filter", "sed_freq"],
                             verbose: bool=False) -> Table:
    """
    Will group the passed SED table by the requested columns and will then set
    the flux/flux_err columns of each group to the mean values. The resulting
    aggregate rows will be returned as a new table.

    :sed: the source SED table
    :group_by_colnames: the columns to group on
    :verbose: whether to output diagnostics messages
    :returns: a new table of just the aggregate rows
    """
    # pylint: disable=dangerous-default-value
    sed_grps = sed.group_by(group_by_colnames)
    if verbose: print(f"Grouped SED by {group_by_colnames} yielding",
                      f"{len(sed_grps.groups)} group(s) from {len(sed)} row(s).")

    # Find the flux & related uncertainty columns to be aggregated
    flux_colname_pairs = []
    for colname in sed.colnames:
        if colname not in group_by_colnames \
                and not colname.startswith("_") and not colname.startswith("sed_e") \
                and sed[colname].unit is not None and sed[colname].unit.is_equivalent(u.Jy):
            colname_err = colname[:4] + "e" + colname[4:]
            if colname_err in sed.colnames:
                flux_colname_pairs += [(colname, colname_err)]
            else:
                flux_colname_pairs += [(colname, None)]

    # Can't use the default groups.aggregate(np.mean) functionality as we need to
    # be able to work with two columns (noms, errs) to correctly calculate the mean.
    if verbose: print(f"Calculating the group means of the {flux_colname_pairs} columns")
    for _, grp in zip(sed_grps.groups.keys, sed_grps.groups):
        for flux_colname, err_colname in flux_colname_pairs:
            if err_colname is not None:
                # avoid mean() or np.mean() as they may trigger uncertainties' FutureWarning
                mf = unumpy.uarray(grp[flux_colname].value, grp[err_colname].value).sum() / len(grp)
                grp[flux_colname] = mf.nominal_value
                grp[err_colname] = mf.std_dev
            else:
                grp[flux_colname] = np.mean(grp[flux_colname].values)

        # if verbose:
        #     group_col_vals = [key[group_by_colnames][ix] for ix in range(len(group_by_colnames))]
        #     print(f"Aggregated {len(grp)} row(s) for group {group_col_vals}")

    # Return only the grouped table rows (not the original rows)
    return sed_grps[sed_grps.groups.indices[:-1]]


def create_outliers_mask(sed: Table,
                         temps0: Union[Tuple[float], List[float]]=(5000., 5000.),
                         min_unmasked: float=15,
                         min_improvement_ratio: float=0.10,
                         verbose: bool=False) -> np.ndarray[bool]:
    """
    Will create a mask indicating the farthest outliers.

    Carried out by iteratively evaluating test blackbody fits on the observations, and masking
    out the farthest/worst outliers. This continues until the fits no longer improve or further
    masking would drop the number of remaining observations below a defined threshold.

    :sed: the source observations to evaluate
    :temps0: the initial temperatures to use for the test fit
    :min_unmasked: the minimum number of observations to leave unmasked, either as an explicit
    count (if > 1) or as a ratio of the initial number (if within (0, 1])
    :min_improvement_ratio: minimum ratio of test stat improvement required to add to outlier_mask
    :verbose: whether to print progress messages or not
    :returns: a mask indicating those observations selected as outliers
    """
    # pylint: disable=too-many-locals, too-many-branches, too-many-statements
    sed_count = len(sed)
    outlier_mask = np.zeros((sed_count), dtype=bool)
    min_unmasked = int(sed_count * min_unmasked if 0 < min_unmasked <= 1 else max(min_unmasked, 1))
    if sed_count <= min_unmasked:
        if verbose: print(f"No mask created as SED rows already at or below {min_unmasked}")
        return outlier_mask

    # Initial temps & associated priors
    temps0 = [temps0] if isinstance(temps0, Number) else temps0
    temp_ratio = temps0[-1] / temps0[0]
    temp_flex = temp_ratio * 0.05
    temps0_median = np.median(temps0)

    # Prepare the x, y & y_err data for the model & objective funcs which access these data directly
    x = sed["sed_freq"].to(u.Hz).value
    y = sed["sed_flux"].to(u.Jy).value
    y_err = sed["sed_eflux"].to(u.Jy).value
    y_log = log10(y)

    # The model func scaling is in log space, as the range is large, but it returns linear fluxes.
    def scaled_bb_model(temps, mask):
        y_model_log = log10(np.sum([blackbody_flux(x[~mask], t) for t in temps], 0)) + 26 # to Jy
        return 10**(y_model_log + np.median(y_log[~mask] - y_model_log))

    # The minimize target func; checks temps against priors, calls the model func and evaluates like
    def objective_func(temps, mask) -> float:
        if all(2e3 < t < 3e4 for t in temps) and abs(temps[-1] / temps[0] - temp_ratio) < temp_flex:
            return simple_like_func(scaled_bb_model(temps, mask), y[~mask], y_err[~mask])
        return np.inf

    # Iteratively fit the observations, remove the worst fitted points until fit no longer improves
    test_mask = outlier_mask.copy()   # for initial/baseline fit nothing is excluded
    last_test_stat = np.inf
    for _iter in range(sed_count):
        if sed_count - sum(test_mask) < min_unmasked:
            if verbose: print(f"[{_iter:03d}] stopped as the {'next' if _iter > 1 else ''} mask",
                        f"will reduce the number of SED rows below the minimum of {min_unmasked}.")
            break

        # Perform a fit on the target fluxes which are still unmasked
        with warnings.catch_warnings(category=RuntimeWarning):
            warnings.filterwarnings("ignore", message="invalid value encountered in subtract")
            soln = minimize(objective_func, x0=temps0, args=test_mask, method="SLSQP")

        if not soln.success:
            if verbose: print(f"[{_iter:03d}] stopped as this fit failed with '{soln.message}'")
            break

        this_temps = soln.x
        if abs(np.median(this_temps) - temps0_median) > temps0_median * 0.25:
            if verbose: print(f"[{_iter:03d}] stopped as this fit yielded unlikely temps",
                              f"of {this_temps} against input temps of {temps0}")
            break

        # Calculate a comperable summary stat on this fit. TODO: refine test stat & weights
        this_y_model = scaled_bb_model(this_temps, test_mask)
        weights = np.ones_like(this_y_model)
        this_resids_sq = ((y[~test_mask] - this_y_model) / y_err[~test_mask])**2
        this_test_stat = np.sum(this_resids_sq * weights) / (len(this_resids_sq) - 1)

        # After the first iter, which sets the unmasked baseline, evaluate this fit (with mask) vs
        # that of the previous iter. If it's significantly better, we adopt the mask and try again.
        if verbose: print(f"[{_iter:03d}] stat = {this_test_stat:.3e}", end="; " if _iter else "\n")
        if _iter > 0:
            if last_test_stat - this_test_stat > last_test_stat * min_improvement_ratio:
                outlier_mask = test_mask
                if verbose: print(f"{sum(test_mask)}/{sed_count} outliers masked for",
                            f"{', '.join(f'{f}' for f in np.unique(sed['sed_filter'][test_mask]))}")
            else:
                if verbose: print("no significant improvement so stopped further masking")
                break

        # Create the next test mask from the current outlier mask & farthest outliers from this fit.
        (test_mask := outlier_mask.copy())[~outlier_mask] = this_resids_sq == this_resids_sq.max()
        last_test_stat = this_test_stat

    return outlier_mask


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


def simple_like_func(y_model: np.ndarray, y: np.ndarray, y_err: np.ndarray) -> float:
    """
    A very simple like function which compares y_model with y +/- y_err with

    like = 1/2 * Σ((y - y_model) / (y_err + 10^-30))^2

    where the addition of 10^-30 to y_err is to prevent division by zero errors

    :y_model: the model y data points
    :y: the equivalent observation y data points
    :y_err: the equivalent uncertatinties in y
    :returns: the likeness of the model to the data
    """
    return 0.5 * np.sum(((y - y_model) / (y_err + 1e-30))**2)
