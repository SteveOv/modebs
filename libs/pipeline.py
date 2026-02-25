"""
Low level utility functions for light curve ingest, pre-processing, estimation and fitting.
"""
# pylint: disable=no-member, too-many-arguments, too-many-positional-arguments
from typing import Union, Tuple, Dict, List
from io import TextIOBase
from sys import stdout
import warnings
import re
from numbers import Number
from multiprocessing import Pool
from itertools import groupby

import numpy as np
from numpy.typing import ArrayLike
from uncertainties import UFloat, ufloat, nominal_value
from uncertainties.unumpy import nominal_values
import astropy.units as u
from astropy.time import Time
from astropy.coordinates import SkyCoord
from astropy.io import ascii as io_ascii
from astroquery.gaia import Gaia
from lightkurve import LightCurve, LightCurveCollection

from deblib import limb_darkening
from deblib.vmath import arccos, degrees

from libs import jktebop, lightcurves
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

_to_file_safe_sub_pattern = re.compile(r"[^\w\d._-]", re.IGNORECASE)
_spt_find_pattern = re.compile(r"([A-Z]{1}[0-9]*)")

def to_file_safe_str(text: str, replacement: str="-", lower: bool=True) -> str:
    """
    Parse the text and replace any potentially troublesome characters when used as a file name.
    Do no pass in a full path as / and \\ are among the characters which will be replaced.

    :text: the original text
    :replacement: the character to substitute for any troublesome characters
    :lower: whether or not to force the revised text to lower case [True]
    :returns: the revised text
    """
    retval = _to_file_safe_sub_pattern.sub(replacement, text)
    return retval.lower() if lower else retval


def get_teff_from_spt(target_spt):
    """
    Estimates a stellar T_eff [K] from the passed spectral type.

    :target_spt: the spectral type string
    :returns: the estimated teff in K
    """
    teff = None
    if target_spt is not None:
        spts = _spt_find_pattern.findall(target_spt.strip().upper())
        for spt in (s for s in spts if len(s) > 0):
            tp = spt.strip()[0]
            if (tp in _spt_to_teff_map) and (teff is None or _spt_to_teff_map[tp].n > teff.n):
                teff = _spt_to_teff_map[tp]
    return teff


def mask_lightcurves_unusable_fluxes(lcs: LightCurveCollection,
                                     quality_masks: List[Union[List, Tuple]]=None,
                                     min_section_gap_bins: int=60,
                                     min_section_dur: Union[u.Quantity[u.d], float]= 2 * u.d):
    """
    Mask out invalid fluxes, known distorted sections and any short, isolated sections
    of observations found across the passed collection of LightCurves.
    
    :lcs: the LightCurveCollection containing our potential fitting targets
    :quality_masks: an optional list of (from, to) time tuples of sections to be masked
    :min_section_gap_bins: the minimum gap size in time bins, when finding sections of observations
    :min_section_dur: the minimum duration of isolated LightCurve sections to retain
    """
    # pylint: disable=consider-using-enumerate (cannot enumerate lcs as we're changing its content)
    for ix in range(len(lcs)):
        _mask = lightcurves.create_invalid_flux_mask(lcs[ix])

        # The configured quality masks are clipped because they're known to be they're distorted.
        if quality_masks is not None and len(quality_masks) > 0:
            for mask_times in (lightcurves.to_lc_time(t, lcs[ix]) for t in quality_masks):
                _mask &= (lcs[ix].time < np.min(mask_times)) | (np.max(mask_times) < lcs[ix].time)

        # We also look to clip any short isolated regions of the LCs, as they often contain little
        # useful information and often have a detrimental affect on the effectiveness of detrending.
        seg_gap_th = min_section_gap_bins * lcs[ix].meta['FRAMETIM'] * lcs[ix].meta['NUM_FRM'] * u.s
        for seg in lightcurves.find_lightcurve_segments(lcs[ix], seg_gap_th, yield_times=True):
            if max(seg) - min(seg) < min_section_dur:
                _mask &= (lcs[ix].time < min(seg)) | (lcs[ix].time > max(seg))

        lcs[ix] = lcs[ix][_mask]


def add_eclipse_meta_to_lightcurves(lcs: LightCurveCollection,
                                    ref_t0: Union[Time, float, UFloat],
                                    period: Union[u.Quantity, float, UFloat],
                                    widthp: Union[float, UFloat],
                                    widths: Union[float, UFloat],
                                    depthp: Union[float, UFloat]=None,
                                    depths: Union[float, UFloat]=None,
                                    phis: Union[float, UFloat]=0.5,
                                    search_window_phase: float=0.05,
                                    verbose: bool=False):
    """
    Will find the times of all primary and secondary eclipses within the bounds of each
    passed LightCurve sector and the corresponding completeness metrics. These data will
    be added/updated in each LightCurves meta dictionary with the following keys:
    - t0: a sector specific reference primary eclipse time based on the eclipses within sector
    - primary_times: an array of the times of any primary eclipses that fall within the sector
    - primary_completeness: an array of completeness values, each the proportion of fluxes
    found vs number of fluxes expected for a complete eclipse
    - secondary_times: as for primary_times, but for secondary eclipses
    - secondary_completeness: as for primary_completeness, but for secondary eclipses
    
    :lcs: the LightCurveCollection containing our potential fitting targets
    :ref_t0: the known reference primary eclipse time
    :period: the known orbital period
    :widthp: the width of the primary eclipses in units of normalized phase
    :widths: the width of the secondary eclipses in units of normalized phase
    :depthp: the expected depth of the primary eclipse in units of normalized flux
    :depths: the expected depth of the secondary eclipse in units of normalized flux
    :phis: the phase of the secondary eclipses relative to the primary eclipses
    :search_window_phase: size of the window, in units of phase, within which to find each eclipse
    :verbose: whether or not to send messages to stdout with details of the eclipse search
    """
    for lc in lcs:
        ecl_data = lightcurves.find_eclipses_and_completeness(lc, ref_t0, period,
                                                              widthp, widths, depthp, depths,
                                                              phis, search_window_phase, verbose)

        # We will use the revised/refined reference time as a starting position for the next sector
        ref_t0 = lc.meta["t0"] = ecl_data[0] or ref_t0
        lc.meta |= dict(zip(
            ["primary_times", "primary_completeness", "secondary_times", "secondary_completeness"],
            ecl_data[1:]
        ))


def choose_lightcurve_groups_for_fitting(lcs: LightCurveCollection,
                                         completeness_th: float=0.8,
                                         verbose: bool=False) -> List[List[int]]:
    """
    Will work out the most effective arrangement of LightCurves to support JKTEBOP fitting. This
    will need to balance need for sufficient coverage for each fitting to achieve a reliable output,
    while running as many fits as possible across the breadth of time in which we have observations.
    This uses the eclipse timing & completeness metadata added by add_eclipse_meta_to_lightcurves().

    :lcs: the LightCurveCollection containing our potential fitting targets
    :completness_th: threshold percentage of an eclipse we require to consider it complete/usable
    :verbose: whether or not to send messages to stdout with details of the group decisions made
    :returns: the groups to fit, as a list of lists of sector numbers. i.e.: [[13, 14], [15, 16]]
    indicates the LightCurves for sectors 13 & 14 should be grouped for fitting, as should 15 & 16.
    """
    def is_usable_group(seg_ecl_counts) -> bool:
        """ On the eclipse counts, is the corresponding grp of LCs/sectors suitable for fitting? """
        ecl_sums = np.sum(seg_ecl_counts, axis=0)
        return max(ecl_sums) > 2 * completeness_th and min(ecl_sums) > 1 * completeness_th

    # Make sure the sectors are sorted by sector number.
    lcs = LightCurveCollection(sorted(lcs, key=lambda l: l.sector))

    # Isolating the sectors into contiguous blocks, so [1,2,4,5,6,8] becomes [[1,2], [4,5,6], [8]]
    # By using a key of the sector number minus its list index we have a value which we can
    # group by, as it will remain unchanged within a block of contiguously incrementing values.
    sector_groups = []
    for _, block in groupby(enumerate(lcs.sector),
                            key=lambda ix_sec: ix_sec[1] - ix_sec[0]):
        # Now work out how best to use this block of contiguous sectors/LCs for JKTEBOP fitting.
        blk_sectors = list(g for _, g in block)

        # Eclipse counts for each block member as an array([[#prim0, #sec0], ..., [#primN, #secN]])
        blk_ecl_counts = np.array([
            list(sum(l.meta[k][l.meta[k] > completeness_th])
                    for k in ["primary_completeness", "secondary_completeness"])
                        for l in lcs[np.in1d(lcs.sector, blk_sectors)]
        ])

        blk_size = len(blk_sectors)
        max_ecl_count = np.max(blk_ecl_counts) # either primary or secondary
        min_grp_size = max(1, int(np.floor(2*completeness_th / (max_ecl_count+1e-10))))

        if blk_size >= min_grp_size:
            if blk_size == 1:
                if is_usable_group(blk_ecl_counts):
                    sector_groups.append(blk_sectors)
                    if verbose:
                        print(f"Created a group of sector(s) {blk_sectors}.")
                elif verbose:
                    print(f"Dropped the solitary sector {blk_sectors[0]}",
                          "as it has insufficient orbital coverage.")
            else:
                # Multiple sectors/LCs within this block so build combinations.
                grp_start = 0
                while grp_start < blk_size:
                    next_start_inc = 1

                    # Grow group until it has sufficient coverage or we run out of sectors
                    for grp_stop in range(grp_start + min_grp_size, blk_size + 1):
                        grp_slice = slice(grp_start, grp_stop)
                        if is_usable_group(blk_ecl_counts[grp_slice]):
                            # Special case. If we cannot to get another group from the remainder
                            # of the block, we may as well expand this group to make use of it.
                            if grp_stop < blk_size <= grp_stop + min_grp_size \
                                    and not is_usable_group(blk_ecl_counts[grp_stop:]):
                                grp_slice = slice(grp_start, grp_stop := blk_size)

                            sector_groups.append(blk_sectors[grp_slice])
                            next_start_inc = grp_stop - grp_start
                            if verbose:
                                print(f"Created a group of sector(s) {blk_sectors[grp_slice]}.")
                            break

                    grp_start += next_start_inc
        elif verbose:
            print(f"Dropped the block of sectors {blk_sectors} as they have insufficient",
                  "orbital coverage, either singularly or when combined.")

    return sector_groups


def stitch_lightcurve_groups(lcs: LightCurveCollection,
                             sector_groups: List[List[int]],
                             verbose: bool=False) -> LightCurveCollection:
    """
    Will create a new LightCurveCollection containing single and/or stitched LightCurves from the
    source collection passed in, based on the groups listed in the sector_groups argument.

    The sector_groups argument will have the form [[group0], [group1], ... , [groupN]] where each
    group is a list of one or more sector numbers which indicate the source LightCurves which are
    to be stitched to form the group's LightCurve in the output collection. This list can be
    generated with the group_sectors_for_fitting() func, also in this module.
    
    :lcs: the original LightCurveCollection from which to create the stitched copies
    :sector_groups: the list of groups which controls the stitching to be carried out
    :verbose: whether or not to send messages to stdout with details of the actions taken
    :returns: a new LightCurveCollection containing the newly grouped/stitched LightCurves
    """
    grp_lcs = []
    for sector_group in (s for s in sector_groups if len(s) > 0):
        mask = np.in1d(lcs.sector, sector_group)
        if sum(mask) == 0:
            warnings.warn(f"No LCs found for the grouped sector(s) {sector_group}")
        else:
            if sum(mask) != len(sector_group):
                missing = [s for s in sector_group if s not in lcs.sector]
                msg = f"The LC(s) {missing} not found for the grouped sector(s) {sector_group}"
                warnings.warn(msg)

            target = lcs[mask][0].meta.get("target", lcs[mask][0].meta["OBJECT"])
            if verbose and sum(mask) > 1:
                print(f"The {target} LCs for grouped sectors {lcs[mask].sector} will be stitched.")

            # Normalize + combine the sectors in the grouping (also if there's only 1)
            grp_lcs += [lcs[mask].stitch(lambda lc: lc.normalize())]

            # Update/combine metadata where appropriate. From fits tends to be UCASE, ours are lcase
            grp_lcs[-1].meta["LABEL"] = f"{target} S{'+'.join(f'{s:02d}' for s in sector_group)}"
            grp_lcs[-1].meta["sectors"] = lcs[mask].sector
            if sum(mask) > 1:
                # lightkurve's stitch appears smart enough to concat ndarray & list meta values.
                # However, some of the singular values seem to be from the last sector, when it is
                # useful if it were from the first sector (i.e.: t0, TSTART). Fix where necessary.
                for k in ["t0", "TSTART", "DATE-OBS"]:
                    if k in lcs[mask][0].meta:
                        grp_lcs[-1].meta[k] = lcs[mask][0].meta[k]

                for k in ["LIVETIME", "TELAPSE"]:
                    if all(k in lc.meta for lc in lcs[mask]):
                        grp_lcs[-1].meta[k] = sum(lc.meta[k] for lc in lcs[mask])

    return LightCurveCollection(grp_lcs)


def append_mags_to_lightcurves_and_detrend(lcs: LightCurveCollection,
                                           detrend_gap_th: Union[u.Quantity[u.d], float]=2,
                                           detrend_poly_degree: int=2,
                                           detrend_iterations: int=3,
                                           flatten: bool=False,
                                           durp: Union[float, UFloat]=None,
                                           durs: Union[float, UFloat]=None,
                                           verbose: bool=False):
    """
    Append delta_mag and delta_mag_err columns calculated from normalized fluxes to each
    LightCurve, then detrend the delta_mag column by subtracting a fitted polynomial.
    Optionally the fluxes may be flattened outside of any previously detected eclipses,
    prior to calculating the magnitude columns.

    :lcs: the LightCurveCollection containing our potential fitting targets
    :detrend_gap_th: the threshold gap time beyond which a detrending section break is triggered
    :detrend_poly_degree: the degree of the detrending polynomial to fit
    :detrend_iterations: number of iterations to run during detrending to fit a polynomial
    :flatten: whether or not to flatten the LightCurve fluxes prior to calculating the magnitudes
    :durp: the duration of the primary eclipses required for flattening
    :durs: the duration of the secondary eclipses required for flattening
    :verbose: whether or not to send messages to stdout with details of the actions taken
    """
    if flatten and (durp is None or durs is None):
        raise ValueError("durp and durs required if flatten == True")
    if flatten:
        durp = nominal_value(durp) * 1.1
        durs = nominal_value(durs) * 1.1
    if not isinstance(detrend_gap_th, u.Quantity):
        detrend_gap_th = detrend_gap_th * u.d

    # Cannot enumerate lcs as we're changing its content. pylint: disable=consider-using-enumerate
    for ix in range(len(lcs)):
        label = lcs[ix].meta["LABEL"]

        if flatten:
            pri_times = lcs[ix].meta.get("primary_times", [])
            sec_times = lcs[ix].meta.get("secondary_times", [])
            mask = lcs[ix].meta["flat_mask"] = lightcurves.create_eclipse_mask(lcs[ix],
                                                                               pri_times, sec_times,
                                                                               durp, durs)

            if verbose:
                num_ecl = len(np.ma.clump_masked(np.ma.masked_where(mask, mask)))
                print(f"Flattening the {label} LC outside of {num_ecl} masked eclipse(s).")
            lcs[ix] = lcs[ix].flatten(mask=mask)

        # Create detrended & rectified delta_mag and delta_mag err columns
        lightcurves.append_magnitude_columns(lcs[ix], "delta_mag", "delta_mag_err")
        for s in lightcurves.find_lightcurve_segments(lcs[ix], threshold=detrend_gap_th):
            lcs[ix][s]["delta_mag"] -= lightcurves.fit_polynomial(lcs[ix].time[s],
                                                                  lcs[ix][s]["delta_mag"],
                                                                  detrend_poly_degree,
                                                                  detrend_iterations)


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


def median_params(input_params: ArrayLike,
                  quant_size: float=0.5,
                  exclude_outliers: bool=False,
                  min_uncertainty_pc: float=0.) -> ArrayLike:
    """
    Produce aggregated values for the input structured array of param values.
    The returned values are the median of the nominal values of the input with
    uncertainties derived from the mean extent of requested quantile range.

    This approach makes the assumption that any inidividual uncertainties in the
    input params are negligible when compared with the scatter in the values,
    with the scatter approximating a normal distribution.

    :fitted_params: a structured array containing the source fitted parameter values
    :quant_size: the size of the inter-quantile range used to derive uncertainties
    :exclude_outliers: whether to exclude any outliers before calculating the median and
    uncertainties, with outliers being values outside the inter-quartile range (IQR) +/- 1.5*IQR
    :min_uncertainty_pc: optional minimum uncertainty as a percentage of the median
    :return: a single row of a corresponding structured array containing UFloats
    """
    quantiles = (0.5 - quant_size/2, 0.5, 0.5 + quant_size/2)
    agg_params = np.empty((1,), dtype=input_params.dtype)
    for k in input_params.dtype.names:
        noms = nominal_values(input_params[k])

        if exclude_outliers:
            q1, q3 = np.quantile(noms, q=(0.25, 0.75))
            whisker_len = 1.5 * (q3 - q1)
            noms = noms[(q1 - whisker_len <= noms) & (noms <= q3 + whisker_len)]

        lo, med, hi = np.quantile(noms, q=quantiles)
        agg_params[k][0] = ufloat(med, max(np.mean([med-lo, hi-med]), med * min_uncertainty_pc))
    return agg_params[0]


def fit_target_lightcurves(lcs: LightCurveCollection,
                           input_params: dict[str],
                           read_keys: List[str],
                           t0: Union[float, UFloat, np.ndarray]=None,
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

    :lcs: the source lightcurves, which must have the time, delta_mag and delta_mag_err columns
    :input_params: the set initial input params to the fitting process shared by each lightcurve
    :read_keys: the set of fitted output params to read and return for each fit
    :t0: either a single value for all lcs, individual values for each lc or None when
    values will be taken from each lightcurve's meta dictionary under the t0 key
    :task: the jktebop task to be executed
    :iterations: the number of iterations to run if task == 3, otherwise ignored
    :max_workers: the maximum number of concurrent fits to run
    :max_attempts: the maximum number of attempts to make - ignored unless task == 3
    :timeout: the timeout for any individual fit - will raise a TimeoutExpired if not completed
    :returns: a list of dictionaries containing the fitted parameters matching read_keys and the
    paths to the various jktebop files associated with the fitting
    """
    if t0 is None:
        t0 = [lc.meta.get("t0", lc.meta.get("primary_epoch", None)) for lc in lcs]
    elif isinstance(t0, float|UFloat):
        t0 = [t0] * len(lcs)

    task_params = { "task": task, "simulations": iterations if task == 8 else "" }
    if task != 3:
        max_attempts = 1
    max_workers = min(len(lcs), max_workers or 1)

    # Set up the sets of fit_target_lightcurve args may differ by lc/sector
    all_in_params = [input_params.copy() | task_params | { "t0": t } for t in t0]
    all_fit_stems = [file_prefix + "-" + lc.meta["LABEL"].replace(" ", "-").lower() for lc in lcs]

    # If we're to run in parallel this indicates to _fit_target to write JKTEBOP console output to
    # stdout less frequently, so it is less likely that the fitting narrative will be interleaved.
    hold_stdout = max_workers > 1

    # Create the args for each lc/call to _fit_target.
    # Can't have func take an lc or masked columns as they do not pickle.
    iter_params = ((lc["time"].unmasked.value,
                    lc["delta_mag"].unmasked.value,
                    lc["delta_mag_err"].unmasked.value,
                    in_params,
                    read_keys,
                    fit_stem,
                    lc.meta.get("clip_mask", None),
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
                clip_mask: np.ndarray[bool]=None,
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
    :clip_mask: optional mask to apply to time, delta_mag and delta_mag_err data prior to writing
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
    if clip_mask is not None:
        table = [time[clip_mask], delta_mag[clip_mask], delta_mag_err[clip_mask]]
    else:
        table = [time, delta_mag, delta_mag_err]
    io_ascii.write(table,
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
