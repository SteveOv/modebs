"""
Low level utility functions for light curve ingest, pre-processing, estimation and fitting.
"""
# pylint: disable=no-member
from typing import Union, Tuple, Dict, List
from pathlib import Path
from io import TextIOBase
from sys import stdout
import warnings
import re
from numbers import Number

import numpy as np
from numpy.typing import ArrayLike
from uncertainties import UFloat, ufloat
from uncertainties.unumpy import nominal_values
import astropy.units as u
from astropy.coordinates import SkyCoord
from astroquery.gaia import Gaia
from astroquery.vizier import Vizier
from lightkurve import LightCurve

from deblib import limb_darkening
from deblib.vmath import arccos, arcsin, degrees

from libs import jktebop

_TRIG_MIN = ufloat(-1, 0)
_TRIG_MAX = ufloat(1, 0)

_spt_to_teff_map = {
    "M": ufloat(3100, 800),
    "K": ufloat(4600, 700),
    "G": ufloat(5650, 350),
    "F": ufloat(6700, 500),
    "A": ufloat(8600, 1300),
    "B": ufloat(20000, 10000),
    "O": ufloat(35000, 10000)
}

def get_teff_from_spt(target_spt):
    """
    Estimates a stellar T_eff [K] from the passed spectral type.

    :target_spt: the spectral type string
    :returns: the estimated teff in K
    """
    teff = None

    # Also add the whole spt in case it's just a single char (i.e.: V889 Aql is set to "A")
    if target_spt is not None \
            and (spts := re.findall(r"([A-Z][0-9])", target_spt) + [target_spt.upper()]):
        for spt in spts:
            if spt and len(spt) and (tp := spt.strip()[0]) in _spt_to_teff_map \
                and _spt_to_teff_map[tp].n > (teff.n if teff is not None else 0):
                teff = _spt_to_teff_map[tp]
    return teff


def nominal_value(value: Union[UFloat, Number]) -> Number:
    """
    Simple helper function to get the nominal value of a number
    whether or not it's a UFloat
    """
    if isinstance(value, UFloat):
        return value.nominal_value
    return value


def get_tess_ebs_data(search_term: str, radius_as: float=5.) -> dict:
    """
    Gets a dictionary of ephemeris and morphology data from the TESS-ebs catalogue.
    """
    ebs_key_patterns = ["{0}-2g", "{0}-pf"] # data derived from 2-Gaussian & polyfit algos
    tess_ebs_catalog = Vizier(catalog="J/ApJS/258/16", row_limit=1)
    if (tbl := tess_ebs_catalog.query_object(search_term, radius=radius_as * u.arcsec)):
        sub_tbl = tbl[0]

        period = ufloat(sub_tbl["Per"][0], sub_tbl["e_Per"][0])
        data = {
            "t0": ufloat(sub_tbl["BJD0"][0], sub_tbl["e_BJD0"][0]),
            "period": period,
        }

        if not sub_tbl.mask["Morph"][0]:
            morph = sub_tbl["Morph"][0]
            if not np.isnan(morph) and isinstance(morph, Number):
                data["morph"] = morph

        # There are two sets of eclipse data; those based on the polyfit algorithm and those on the
        # 2-Gaussian algorithm. For the best chance of consistent values we use one or other set.
        for k_pattern in ebs_key_patterns:
            k_phip, k_phis = k_pattern.format("Phip"), k_pattern.format("Phis")
            k_durp, k_durs = k_pattern.format("Wp"), k_pattern.format("Ws")

            # Each value may be masked, NaN or non-numeric. We try to work with an unmasked set.
            if not any(sub_tbl.mask[k][0] for k in [k_phip, k_phis, k_durp, k_durs]):
                # We want to get the phases so that the primary is zero and the secondary is
                # offset from this. Within TESS-ebs these are usually Phip=1 and Phis=offset
                # which is OK if we wrap 1 to 0. However, some appear shifted so that Phis=1
                # (i.e. TIC 26801525; phip-pf=0.448 & phis-pf=1.000 rather than phip=0 & phis=0.552)
                # In this case we need to undo the shift and switch the widths/durations.
                vals = sub_tbl[[k_phip, k_phis, k_durp, k_durs]][0]
                if all(not np.isnan(v) and isinstance(v, Number) for v in vals):
                    phip, phis, durp, durs = vals

                    # Get both phases into the range [0, 1)
                    phip %= 1
                    phis %= 1
                    if phis < phip: # The phases have been shifted
                        phip, phis = 0, 1 - (phip - phis)
                        durp, durs = durs, durp
                    else:
                        phip, phis = 0, phis - phip

                    data["phiP"] = phip
                    data["phiS"] = phis

                    # The TESS-ebs eclipse widths are in units of phase
                    data["durP"] = durp * period
                    data["durS"] = durs * period
                    break

    else:
        data = None
    return data


def estimate_l3_with_gaia(centre: SkyCoord, radius_as: float=120,
                          target_source_id: int=None, target_g_mag: float=None,
                          max_l3: float=None, verbose: bool=False) -> float:
    """
    Estimates the third-light contribution from any sources near the target found in Gaia DR3.
    The returned L3 value is the sum of the values for each source found. Each target's L3 is its
    flux ratio compared with the target's, multiplied by a factor derived from its angular distance
    from the centre of the search area.

    Either target_source_id or target_g_mag must be supplied.
    
    :centre: the coordinates of the target and the centre of the search cone
    :radius: the radius of the search in arcsec
    :target_source_id: the Gaia DR3 source id of the target, if known
    :target_g_mag: the apparent G-band magnitude of the target, if it is not in Gaia DR3
    :max_l3: optional max allowable L3 value
    :returns: an estimated starting L3 value
    """
    l3 = 0
    Gaia.MAIN_GAIA_TABLE = "gaiadr3.gaia_source"
    if (job := Gaia.cone_search(coordinate=centre, radius=radius_as * u.arcsec,
                                columns=("source_id", "phot_g_mean_mag", "ra", "dec"))):
        tbl = job.get_results()
        if len(tbl) > 0:

            if target_source_id is not None:
                target_mask = tbl["source_id"] == int(target_source_id)
            else:
                # It may not be known, but if there is an object close to the centre of the search
                # cone with a magnitude similar to the target's we will assume it's the target.
                target_mask = (tbl["dist"] * 3600 < 5.0) \
                                & (np.abs(tbl["phot_g_mean_mag"] - target_g_mag) < 0.5)
                if verbose and any(target_mask):
                    print(f"Omitting {target_mask.sum()} object(s) likely to be the target")

            if target_g_mag is None:
                if any(target_mask):
                    target_g_mag = np.max(tbl[target_mask]["phot_g_mean_mag"])
                    if verbose:
                        print(f"Target object has a Gaia magnitude of {target_g_mag:.4f} (4 d.p.)")
                else:
                    raise ValueError("Cannot find target in search cone & no target_g_mag given")

            if any(~target_mask):
                flux_ratios = 10**(0.4 * (target_g_mag - tbl[~target_mask]["phot_g_mean_mag"]))

                # The Gaia query gives us a dist field which appears to be the angular distance
                # from search centre in degrees. Calculate a weighting based on the normalized dist.
                proximity_weights = 1 - (tbl[~target_mask]["dist"] * 3600 / radius_as)

                l3 = np.sum(flux_ratios * proximity_weights)

                if verbose:
                    print(f"Estimated the total third-light ratio (L3) to be {l3:.4f} (4 d.p.)",
                          f"for the {len(tbl[~target_mask])} nearby object(s) found in Gaia DR3.")

                if max_l3 is not None and max_l3 < l3:
                    l3 = max_l3
                    if verbose:
                        print(f"The estimate is reduced to the maximum allowed value of {max_l3}.")
            elif verbose:
                print("No nearby objects found in Gaia DR3 so the estimated L3=0")
    return l3


def calculate_orbital_inclination(sum_r: ArrayLike,
                                  k: ArrayLike,
                                  ecosw: ArrayLike,
                                  esinw: ArrayLike,
                                  bp: ArrayLike) -> ArrayLike:
    """
    Calculate the predictions' inclination value(s) (in degrees).
    
    With the primary impact param:  inc = arccos(b * r1 * (1+esinw)/(1-e^2))

    :sum_r: the sum of the fractional radii
    :k: the ratio of the fractional radii
    :ecosw: the e*cos(omega) Poincare element
    :esinw: the e*sin(omega) Poincare element
    :bp: the primary impact parameters
    :returns: the orbital inclination(s) in units of degree
    """
    with warnings.catch_warnings(category=[FutureWarning]):
        # Deprecation warning caused by the use of np.clip on ufloats
        # We use clip because the predictions may contain unphysical values
        warnings.filterwarnings("ignore", r"AffineScalarFunc.(__le__|__ge__)\(\) is deprecated.")

        r1 = sum_r / (1 + k)
        e_squared = ecosw**2 + esinw**2
        cosi = np.clip(bp * r1 * (1 + esinw) / (1 - e_squared), _TRIG_MIN, _TRIG_MAX)
        return degrees(arccos(cosi))


def predictions_to_mean_dict(preds: np.ndarray[UFloat],
                             calculate_inc: bool=False,
                             inc_key: str="inc") -> dict[str, UFloat]:
    """
    Takes an array of predictions and will produce a corresponding dictionary of mean values
    for use as JKTEBOP input params. Optionally this includes the orbital inclination which
    is calculated from the other values.
    
    :preds: the prediction structured array as produced by the EBOP MAVEN estimator
    :calculate_inc: whether or not to calculate and append an orbital inclination value
    :inc_key: the key to give the orbital inclination value
    :returns: a dictionary with field names and mean values for keys and values
    """
    pd = { k: preds[k].mean() for k in preds.dtype.names }
    if calculate_inc and inc_key is not None:
        pd[inc_key] = calculate_orbital_inclination(pd["rA_plus_rB"], pd["k"],
                                                    pd["ecosw"], pd["esinw"], pd["bP"])
    return pd


def pop_and_complete_ld_config(source_cfg: dict[str, any],
                               teffa: float, teffb: float,
                               logga: float, loggb: float,
                               default_algo: str="quad",
                               verbose: bool=False) -> dict[str, any]:
    """
    Will set up the limb darkening algo and coeffs, first by popping them from the source_cfg
    dictionary then completing the config with missing values. Where missing, the algo defaults
    to the value of the default_algo arg. For the quad, pow2 or h1h1 algos coefficient lookups
    are performed, based on the supplied teff and logg values, to populate any missing values.
    For any other algos, all coefficients must to be given in config.

    NOTE: pops the LD* items from source_cfg (except those ending _fit) into the returned config

    :source_cfg: the config fragment which may contain predefined LD params
    :teffa: effective temp of star A
    :teffb: effective temp of star B
    :logga: log(g) of star A
    :loggb: log(g) of star B
    :default_algo: the LD algo to use if not specified in source_cfg; either quad, pow2 or h1h2
    :verbose: whether or not to output details of the params chosen to stdout
    :return: the LD params only dict
    """
    params = {}
    for ld in [k for k in source_cfg if k.startswith("LD") and not k.endswith("_fit")]:
        params[ld] = source_cfg.pop(ld)

    for star, teff, logg in [("A", teffa, logga), ("B", teffb, loggb)]:
        algo = params.get(f"LD{star}", default_algo)
        if f"LD{star}" not in params \
                or f"LD{star}1" not in params or f"LD{star}2" not in params:
            # If we've not been given overrides for both the algo and coeffs we can look them up
            if algo.lower() == "same":
                coeffs = (0, 0) # JKTEBOP uses the A star params for both
            elif algo.lower() == "quad":
                coeffs = limb_darkening.lookup_quad_coefficients(logg, teff)
            elif algo.lower() in ["pow2", "h1h2"]:
                coeffs = limb_darkening.lookup_pow2_coefficients(logg, teff)
            else:
                raise ValueError(f"Lookup of coeffs not supported for the {algo} LD algorithm")

            # Add any missing algo/coeffs tags to the overrides
            params.setdefault(f"LD{star}", algo)
            if algo.lower() == "h1h2":
                # The h1h2 reparameterisation of the pow2 law addreeses correlation between the
                # coeffs; see Maxted (2018A&A...616A..39M) and Southworth (2023Obs...143...71S)
                params.setdefault(f"LD{star}1", 1 - coeffs[0]*(1 - 2**(-coeffs[1])))
                params.setdefault(f"LD{star}2", coeffs[0] * 2**(-coeffs[1]))
            else:
                params.setdefault(f"LD{star}1", coeffs[0])
                params.setdefault(f"LD{star}2", coeffs[1])
    if verbose:
        print(f"Limb darkening params: StarA={params['LDA']}({params['LDA1']}, {params['LDA2']}),",
              f"StarB={params['LDB']}({params['LDB1']}, {params['LDB2']})")
    return params


def median_fitted_params(fitted_params: ArrayLike,
                         quantiles: Tuple[float, float]=(0.16, 0.84),
                         min_uncertainty_pc: float=0.) -> ArrayLike:
    """
    Produce aggregated values for the passed structured array of fitted params.
    The values are based on the median of the nominal values of the fitted params
    with uncertainties from the corresponding scatter in the fitted values.

    This approach makes the assumption that any inidividual uncertainties in the source
    fitted params are negligible when compared with the scatter in the values.
    
    :fitted_params: a structured array containing the source fitted parameter values
    :quantiles: quantiles used to derive uncertainties with the default value equivalent to 1-sigma
    :min_uncertainty_pc: optional minimum uncertainty as a percentage of the nominal
    :return: a single row of a corresponding structured array
    """
    if sum(quantiles) != 1.:
        warnings.warn("Quantiles are not balanced; they do not total 1.0", UserWarning)

    agg_params = np.empty_like(fitted_params)
    for k in fitted_params.dtype.names:
        noms = nominal_values(fitted_params[k])

        med = np.median(noms)
        qlo = med - np.quantile(noms, min(quantiles))
        qhi = np.quantile(noms, max(quantiles)) - med

        agg_params[k][0] = ufloat(med, max(np.mean([qlo, qhi]), med * min_uncertainty_pc))
    return agg_params[0]


def fit_target_lightcurve(lc: LightCurve,
                          input_params: dict[str, UFloat],
                          read_keys: List[str],
                          file_stem: str,
                          append_lines: List[str]=None,
                          max_attempts: int=1,
                          timeout: int=None,
                          stdout_to: TextIOBase=stdout) -> Tuple[Dict[str, UFloat], Path, Path]:
    """
    Will use JKTEBOP to fit the passed lightcurve. This sets up all of the necessary input files,
    including clearing down any matching files from a previous fit. Retries are supported in
    the case of a fit raising a "## Warning: a good fit was not found after ..." warning and
    are controlled with the max_attempt arg (1 for no retries). Timeouts are supported, with the
    timeout parameter indicating the maximum number of seconds to allow for a fit (None == forever). 

    :lc: the source lightcurve, which must have the time, delta_mag and delta_mag_err columns
    :input_params: the initial input params to the fitting process
    :read_keys: the set of fitted output params to read and return
    :file_stem: the common stem to use in the filenames read and written during fitting - any
    existing files within the jktebop directory matching this stem will be deleted prior to fitting
    :append_lines: any whole lines of fitting instructions to append to the fitting input 'in' file
    :max_attempts: the maximum number of attempts to make
    :timeout: the timeout for any individual fit - will raise a TimeoutExpired if not completed
    :stdout_to: where to send JKTEBOP's stdout text output
    :returns: a tuple containing a dictionary of the read_keys fitted parameters and the paths to
    the jktebop generated fit and out files
    """
    output_params = { }
    best_attempt = 1
    all_keys = list(jktebop._param_file_line_beginswith.keys()) # pylint: disable=protected-access

    # JKTEBOP will fail if it finds files from a previous fitting
    fit_dir = jktebop.get_jktebop_dir()
    for file in fit_dir.glob(file_stem + ".*"):
        file.unlink()

    # Preserve the initial inputs as we'll progressively update the attempt intputs if retries occur
    next_att_in_params = input_params.copy()
    for attempt in range(1, 1 + max(1, int(max_attempts))):

        attempt_fname_stem = file_stem + f".a{attempt:d}"
        in_fname = fit_dir / (attempt_fname_stem + ".in")
        dat_fname = fit_dir / (attempt_fname_stem + ".dat")
        par_fname = fit_dir / (attempt_fname_stem + ".par")
        next_att_in_params["data_file_name"] = dat_fname.name
        next_att_in_params["file_name_stem"] = attempt_fname_stem

        # Warnings are a mess! I haven't found a way to capture a specific type of warning with
        # specified text and leave everything else to behave normally. This is the nearest I can
        # get but it seems to suppress reporting all warnings, which is "sort of" OK as it's a
        # small block of code and I don't expect anythine except JktebopWarnings to be raised here.
        with warnings.catch_warnings(record=True, category=jktebop.JktebopTaskWarning) as warn_list:
            # Context manager will list any JktebopTaskWarning raised in this context. Known issues
            # with warnings context manager & multi(thread|process)ing so we will need to change
            # this if needed. Probably just catch & parse the jktebop output before sending it on.

            jktebop.write_in_file(in_fname, append_lines=append_lines, **next_att_in_params)
            jktebop.write_light_curve_to_dat_file(lc, dat_fname)

            # Blocks on the JKTEBOP task until we can parse the newly written par file contents
            # to read out the revised values for the superset of potentially fitted parameters.
            plines_gen = jktebop.run_jktebop_task(in_fname, par_fname, None, stdout_to, timeout)
            att_out_params = jktebop.read_fitted_params_from_par_lines(plines_gen, all_keys, True)

            if attempt == 1:
                # Set up the fallback position, being the outputs from the first attempt
                best_attempt = 1
                output_params = att_out_params.copy()
                out_fname = fit_dir / (attempt_fname_stem + ".out")
                fit_fname = fit_dir / (attempt_fname_stem + ".fit")

            if max_attempts > 1:
                # Handle retries if we've received warnings which trigger a retry
                if sum(1 for w in warn_list if "good fit was not found after" in str(w.message)):
                    if attempt < max_attempts:
                        next_att_in_params |= att_out_params
                        if stdout_to:
                            stdout_to.write(f"Attempt {attempt} didn't fully converge on a good "
                                            f"fit. Up to {max_attempts} attempt(s) are allowed so "
                                            "will retry from the final position of this attempt.\n")                           
                    else:
                        if stdout_to:
                            stdout_to.write("Failed to fully converge on a good fit after "
                                            f"{max_attempts} attempts. Reverting to the results "
                                            f"from attempt {best_attempt}.\n")
                        break
                elif attempt > 1: # A retry fit worked
                    output_params = att_out_params
                    out_fname = fit_dir / (attempt_fname_stem + ".out")
                    fit_fname = fit_dir / (attempt_fname_stem + ".fit")
                    break
                else: # The initial fit worked - retries not needed
                    break

    return { k: output_params.get(k, None) for k in read_keys }, fit_fname, out_fname
