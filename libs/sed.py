
"""
Low level utility functions for SED ingest, pre-processing, estimation and fitting.
"""
from typing import Union, Tuple, Callable, List
import warnings
from pathlib import Path
import re
from urllib.parse import quote_plus
from numbers import Number

# pylint: disable=no-member
import astropy.units as u
from astropy.table import Table, Column
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
        if verbose:
            print("Table not cached so we will query the VizieR SED service.")
        try:
            targ = quote_plus(search_term or target)
            sed = Table.read(f"https://vizier.cds.unistra.fr/viz-bin/sed?-c={targ}&-c.rs={radius}")
            sed.write(sed_fname, format="votable") # votable matches that published in link above
        except ValueError as err:
            raise ValueError(f"No SED for target={target} and search_term={search_term}") from err

    sed = Table.read(sed_fname)
    sed.sort(["sed_freq"], reverse=True)
    if verbose:
        print(f"Opened SED table containing {len(sed)} row(s).")

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
        dup_grp = sed.group_by(["sed_filter", "sed_freq", "sed_flux", "sed_eflux"])
        if verbose:
            print(f"Removing {len(sed)-len(dup_grp.groups)} duplicate row(s).")
        sed = sed[dup_grp.groups.indices[:-1]]
        sed.sort(["sed_freq"], reverse=True)

    # Add wavelength which we may be useful downstream
    sed["sed_wl"] = np.divide(c * u.m / u.s, sed["sed_freq"]).to(wl_unit)
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
    if verbose:
        print(f"Grouped SED by {group_by_colnames} yielding {len(sed_grps.groups)} group(s)",
              f"from {len(sed)} row(s).")

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
    if verbose:
        print(f"Calculating the group means of the {flux_colname_pairs} columns")
    for _, grp in zip(sed_grps.groups.keys, sed_grps.groups):
        for flux_colname, flux_err_colname in flux_colname_pairs:
            if flux_err_colname is not None:
                mean_flux = np.mean(unumpy.uarray(grp[flux_colname].value,
                                                  grp[flux_err_colname].value))
                grp[flux_colname] = mean_flux.nominal_value
                grp[flux_err_colname] = mean_flux.std_dev
            else:
                mean_flux = np.mean(grp[flux_colname].values)
                grp[flux_colname] = mean_flux

        # if verbose:
        #     group_col_vals = [key[group_by_colnames][ix] for ix in range(len(group_by_colnames))]
        #     print(f"Aggregated {len(grp)} row(s) for group {group_col_vals}")

    # Return only the grouped table rows (not the original rows)
    return sed_grps[sed_grps.groups.indices[:-1]]


def create_outliers_mask(sed: Table,
                         temps0: Tuple=(5000, 5000),
                         min_unmasked: float=15,
                         min_improvement_ratio: float = 0.10,
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
    # pylint: disable=too-many-locals, multiple-statements
    sed_count = len(sed)
    outlier_mask = np.zeros((sed_count), dtype=bool)
    test_mask = outlier_mask.copy()   # for initial/baseline fit nothing is excluded
    min_unmasked = int(sed_count * min_unmasked if 0 < min_unmasked <= 1 else max(min_unmasked, 1))
    if sed_count <= min_unmasked:
        if verbose: print(f"No mask created as SED rows already at or below {min_unmasked}")
        return outlier_mask

    # Initial temps, associated priors and the prior_func
    temps0 = (temps0,) if isinstance(temps0, Number) else temps0
    temp_ratio = temps0[-1] / temps0[0]
    temp_flex = temp_ratio * 0.05
    def prior_func(temps):
        return all(3e3 < t < 3e4 for t in temps) and abs(temps[-1]/temps[0] -temp_ratio) < temp_flex

    # Prepare the y and y_err data and the model func
    x = sed["sed_freq"].to(u.Hz).value
    y = sed["sed_flux"].to(u.Jy).value
    y_err = sed["sed_eflux"].to(u.Jy).value + 1e-30 # avoid div0 errors
    y_log = log10(y)
    def scaled_summed_bb_model(nu, temps):
        # We scale model to sed observations within log space as the value range is high
        y_model_log = log10(np.sum([blackbody_flux(nu, t) for t in temps], axis=0)) + 26 # to Jy
        return 10**(y_model_log + np.median(y_log[~test_mask] - y_model_log))

    # Iteratively fit the observations, remove the worst fitted points until fit no longer improves
    last_test_stat = np.inf
    for _iter in range(sed_count):
        if sed_count - sum(test_mask) < min_unmasked:
            if verbose: print(f"[{_iter:03d}] stopped as the {'next' if _iter >1 else ''} mask",
                        f"will reduce the number of SED rows below the minimum of {min_unmasked}.")
            break

        # Perform a fit on the unmasked target fluxes and get the resulting model
        target_func = create_minimize_target_func(x[~test_mask], y[~test_mask], y_err[~test_mask],
                                                  scaled_summed_bb_model, prior_func)
        with warnings.catch_warnings(category=RuntimeWarning):
            warnings.filterwarnings("ignore", message="invalid value encountered in subtract")
            soln = minimize(target_func, x0=temps0)

        # TODO: check fitted temps and/or fit against input and warn if bad fit
        this_temps = soln.x
        this_y_model = scaled_summed_bb_model(x[~test_mask], this_temps)

        # Calculate a comperable summary stat on this fit. TODO: refine test stat & weights
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


def create_minimize_target_func(
        x: Tuple[Column, np.ndarray],
        y: Tuple[Column, np.ndarray],
        y_err: Tuple[Column, np.ndarray],
        model_func: Callable[[np.ndarray], Union[Tuple, List]],
        prior_func: Callable[[Union[Tuple, List]], bool]=None,
        sim_func: Callable[[np.ndarray, np.ndarray, np.ndarray], float]=\
                                    lambda ymodel, y, y_err: 0.5*np.sum(((y-ymodel)/y_err)**2),) \
    -> Callable[[Union[Tuple, List]], float]:
    """
    Will create and return a simple similarity function which can be used as the target
    function for scipy's minimize optimization. The resulting similarity function accepts the
    each iterations's set of model arguments (theta) which it first passes to a client supplied
    boolean prior_func, with arguments (theta), for evaluation against some prior criteria.
    
    If the prior_func returns false, the similarity_func immediately returns with value np.inf.

    If the prior_func returns true, the supplied model_func is called with the arguments (x, theta)
    from which the corresponding y_model is expected to be returned. Finally, y_model is evaluated
    against y & y_err with the sim_func(y_model, y, y_err) from which the return value is taken.
    
    :x: the SED frequencies or wavelengths of y and y_err; to be passed to model_func(x, theta)
    :y: the SED fluxes at x; to be passed to sim_func(y_model, y, y_err)
    :y_err: the uncertainties of y in the same unit; to be passed to sim_func(y_model, y, y_err)
    :model_func: a function which creates and returns a candidate model y based on the current theta
    :prior_func: a boolean function to evaluate each iteration's theta against known prior criteria,
    returning True or False to indicate whether theta conforms to these conditions or not
    :sim_func: the function taking arguments (y_model, y, y_err) which evaluates y_model against
    y & y_err and returns a numeric results which is the statistic which is minimized
    :returns: the minimize func which may be passed on to scipy minimize for optimizing
    """
    # pylint: disable=too-many-arguments, too-many-positional-arguments
    def minimize_func(theta):
        if not prior_func or prior_func(theta):
            return sim_func(model_func(x, theta), y, y_err)
        return np.inf
    return minimize_func
