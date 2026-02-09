"""
Low level utility functions for light curve ingest, pre-processing, estimation and fitting.
"""
# pylint: disable=no-member, too-many-arguments, too-many-positional-arguments
from typing import Union, Tuple, Dict, List
from io import TextIOBase, StringIO
from sys import stdout
import warnings
import re
from numbers import Number
from multiprocessing import Pool

import numpy as np
from numpy.typing import ArrayLike
from uncertainties import UFloat, ufloat
from uncertainties.unumpy import nominal_values
import astropy.units as u
from astropy.coordinates import SkyCoord
from astropy.io import ascii as io_ascii
from astroquery.gaia import Gaia
from astroquery.vizier import Vizier
from lightkurve import LightCurve, LightCurveCollection

from deblib import limb_darkening
from deblib.vmath import arccos, arcsin, degrees

from libs import jktebop
from libs.iohelpers import PassthroughTextWriter

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


def fit_target_lightcurves(lcs: LightCurveCollection,
                           input_params: dict[str],
                           read_keys: List[str],
                           primary_epoch: Union[float, UFloat, np.ndarray]=None,
                           task: int=3,
                           iterations: int=10,
                           max_workers: int=1,
                           max_attempts: int=3,
                           timeout: int=None,
                           file_prefix: str="pipeline-fit"):
    """
    Will use JKTEBOP to fit the passed LightCurves in parallel processing pool up to max_workers
    in size. Retries are supported, controlled by the max_attempt arg, in the case of a fit raising
    a "## Warning: a good fit was not found after ..." warning message. Timeouts are supported, with
    the timeout arg indicating the maximum number of seconds to allow for a fit (None == forever).

    Each lightcurve will be fitted from mostly the same input settings with the exception of;
        - primary_epoch, which may be varied by lightcurve (see primary_epoch parameter)
        - poly fit instructions, which are calculated for the timings of each lightcurve

    **Note:** Unlike the similar fit_target_lightcurve() func, this func does not accept a stdout_to
    arg. This is because the act of running the fits in parallel make it difficult to capture and
    coordinate the jktebop output. Instead JKTEBOP processing output will be written to sys.stdout
    after each attempted fit.

    :lcs: the source lightcurves, which must have the time, delta_mag and delta_mag_err columns
    :input_params: the set initial input params to the fitting process shared by each lightcurve
    :read_keys: the set of fitted output params to read and return for each fit
    :primary_epoch: either a single value for all lcs, individual values for each lc or None when
    values will be taken from each lightcurve's meta dictionary under the t0 or primary_epoch keys
    :task: the jktebop task to be executed
    :iterations: the number of iterations to run if task == 3, otherwise ignored
    :max_workers: the maximum number of concurrent fits to run
    :max_attempts: the maximum number of attempts to make - ignored unless task == 3
    :timeout: the timeout for any individual fit - will raise a TimeoutExpired if not completed
    :returns: a list of dictionaries containing the fitted parameters matching read_keys and the
    paths to the various jktebop files associated with the fitting
    """
    if primary_epoch is None:
        primary_epoch = [lc.meta.get("t0", lc.meta.get("primary_epoch", None)) for lc in lcs]
    elif isinstance(primary_epoch, float|UFloat):
        primary_epoch = [primary_epoch] * len(lcs)

    task_params = { "task": task, "simulations": iterations if task == 8 else "" }
    if task != 3:
        max_attempts = 1
    max_workers = min(len(lcs), max_workers or 1)

    # Set up the sets of fit_target_lightcurve args may differ by lc/sector
    all_in_params = [input_params.copy() | task_params | {"primary_epoch":p} for p in primary_epoch]
    all_fit_stems = [file_prefix + "-" + lc.meta["LABEL"].replace(" ", "-").lower() for lc in lcs]

    # If we're to run in parallel this indicates to _fit_target to write JKTEBOP console output to
    # stdout less frequently, so it is less likely that the fitting narrative will be interleaved.
    hold_stdout = max_workers > 1

    # Create the args for each lc/call to _fit_target.
    # Can't have func take an lc or masked columns as they do not pickle.
    iter_params = ((lc[lc.meta["clip_mask"]]["time"].unmasked.value,
                    lc[lc.meta["clip_mask"]]["delta_mag"].unmasked.value,
                    lc[lc.meta["clip_mask"]]["delta_mag_err"].unmasked.value,
                    in_params,
                    read_keys,
                    fit_stem,
                    _create_lc_std_further_process_cmds(lc, input_params["period"]),
                    max_attempts,
                    timeout,
                    hold_stdout) \
                    for lc, in_params, fit_stem in zip(lcs, all_in_params, all_fit_stems))

    if max_workers <= 1:
        # Could use a pool of 1, but it's useful to keep execution on the interactive proc for debug
        fitted_params = [None] * len(lcs)
        for ix, params in enumerate(iter_params):
            fitted_params[ix] = _fit_target(*params)
    else:
        with Pool(max_workers) as pool:
            # Gives us a list of each return value from _fit_targets in the order of iter_params
            fitted_params = pool.starmap(_fit_target, iter_params)

    return fitted_params


def _create_lc_std_further_process_cmds(lc: LightCurve, period: Union[float, UFloat]) -> List[str]:
    """
    Creates a standard set of JKTEBOP processing instructions for appending to an in file.
    The instructions set up poly fits for scale factor and chi^sq adjustment

    :lc: the source LightCurve
    :period: the known orbital period of the system, for use in deciding how many segments to fit
    :returns: the list of processing instructions
    """
    if nominal_value(period) < 15 and "sector_times" in lc.meta:
        _sf_segs = lc.meta["sector_times"]
    else: # unless long period where there may be too few eclipses per sector
        _sf_segs = [(lc.time.min(), lc.time.max())]

    return [""] + jktebop.build_poly_instructions(_sf_segs, "sf", 1) + ["", "chif", ""]


def _fit_target(time: ArrayLike,
                delta_mag: ArrayLike,
                delta_mag_err: ArrayLike,
                input_params: dict[str, UFloat],
                read_keys: List[str],
                file_stem: str,
                append_lines: List[str]=None,
                max_attempts: int=1,
                timeout: int=None,
                hold_stdout: bool=False,
                stdout_to: TextIOBase=stdout) -> Dict[str, any]:
    """
    Internal function to fit a target's LightCurve data. This function must support pickling so that
    it may be used both singularly and in a multi-processing approach. As such the args tend to be
    more primitive than the public fitting methods.

    :time: the LightCurve time values to be fitted
    :delta_mag: the LightCurve differential magnitude values to be fitted
    :delta_mag_err: the uncertainties in the LightCurve differential magnitude values to be fitted
    :input_params: the input params dictionary used to populate the fitting 'in' file
    :read_keys: the set of fitted output params to read and return for each fit
    :file_stem: the body of the filenames to write/read excluding the attempt counter & extention
    :append_lines: optional processing instructions to append to the fitting 'in' file
    :max_attempts: the maximum number of attempts to make
    :timeout: the timeout for any individual fit - will raise a TimeoutExpired if not completed
    :stdout_to: where to send JKTEBOP's stdout text output
    :hold_stdout: whether or not to hold anything written to stdout until each attempt is completed
    :returns: a dictionary of the read_keys fitted parameters and further items containing the paths
    to the jktebop files used as inputs or the output of the fit (keys having the form ext_fname)
    """
    best_out_params = { }
    best_attempt = 1
    all_keys = list(jktebop._param_file_line_beginswith.keys()) # pylint: disable=protected-access

    # JKTEBOP will fail if it finds files from a previous fitting
    fit_dir = jktebop.get_jktebop_dir()
    for file in fit_dir.glob(file_stem + ".*"):
        file.unlink()

    # The contents of the lightcurve data file are fixed across fitting attempts.
    # jktebop.write_light_curve_to_dat_file(lc, dat_fname)
    dat_fname = fit_dir / (file_stem + ".dat")
    io_ascii.write([time, delta_mag, delta_mag_err],
                   output=dat_fname,
                   format="no_header",
                   names=["time", "delta_mag", "delta_mag_err"],
                   formats={"time": "%.6f", "delta_mag": "%.6f", "delta_mag_err": "%.6f"},
                   comment="#",
                   delimiter=" ")

    def write_out(msg):
        if stdout_to:
            stdout_to.write(msg)

    # Preserve the initial inputs as we'll progressively update the attempt intputs if retries occur
    next_att_in_params = input_params.copy()
    for attempt in range(1, 1 + max(1, int(max_attempts))):

        att_fname_stem = file_stem + f".a{attempt:d}"
        att_file_params = {
            "dat_fname": dat_fname,
            "in_fname": (in_fname := fit_dir / (att_fname_stem + ".in")),
            "par_fname": (par_fname := fit_dir / (att_fname_stem + ".par")),
            "out_fname": fit_dir / (att_fname_stem + ".out"),
            "fit_fname": fit_dir / (att_fname_stem + ".fit"),
        }

        next_att_in_params["data_file_name"] = dat_fname.name
        next_att_in_params["file_name_stem"] = att_fname_stem

        failed_to_converge = False
        with PassthroughTextWriter(stdout_to, hold_output=hold_stdout,
                                inspect_func=lambda ln: "Warning: a good fit was not found" in ln) \
                            as stdout_cap:
            jktebop.write_in_file(in_fname, append_lines=append_lines, **next_att_in_params)

            # Blocks on the JKTEBOP task until we can parse the newly written par file contents
            # to read out the revised values for the superset of potentially fitted parameters.
            plines = jktebop.run_jktebop_task(in_fname, par_fname, None, stdout_cap, timeout)
            att_out_params = jktebop.read_fitted_params_from_par_lines(plines, all_keys, True)

            failed_to_converge = stdout_cap.inspect_flag

        if attempt == 1:
            # The fallback position, being the outputs from the first attempt regardless of success
            best_attempt = 1
            best_out_params = att_out_params.copy()
            best_file_params = att_file_params.copy()

        if failed_to_converge:
            write_out(f"** Attempt {attempt} of {max_attempts} to fit {file_stem} didn't complete.")

            if max_attempts > 1:
                if attempt < max_attempts:
                    next_att_in_params |= att_out_params
                    write_out(" Will retry from the final position of this attempt.\n")
                else:
                    write_out(f" Will revert to the results from attempt {best_attempt}.\n")
            else:
                write_out(" Only 1 attempt allowed so will return these results.\n")
        else:
            write_out(f"** Attempt {attempt} of {max_attempts} to fit {file_stem} completed.\n")

            if attempt > 1: # A retry fit worked, these become the best params
                best_out_params = att_out_params
                best_file_params = att_file_params
            break

    return { k: best_out_params.get(k, None) for k in read_keys } | best_file_params
