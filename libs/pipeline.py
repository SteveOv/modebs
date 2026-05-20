"""
Low level utility functions for light curve ingest, pre-processing, estimation and fitting.
"""
# pylint: disable=no-member, too-many-arguments, too-many-positional-arguments
from typing import Union, Tuple, Dict, List, Iterable
from numbers import Number
from io import TextIOBase, StringIO
from sys import stdout
import warnings
import re
from multiprocessing import Pool
from itertools import groupby, zip_longest, chain, combinations

import numpy as np
from numpy.typing import ArrayLike
from uncertainties import UFloat, ufloat, nominal_value
from uncertainties.unumpy import nominal_values
import astropy.units as u
from astropy.time import Time
from astropy.io import ascii as io_ascii
from lightkurve import LightCurve, LightCurveCollection

from keras import layers
from ebop_maven.estimator import Estimator

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


class PipelineError(Exception):
    """ Base class for all custom pipeline runtime exceptions """
    def __init__(self, target_id: str, *args):
        """ Initializes a new PipelineError Exception """
        super().__init__(*args)
        self._target_id = target_id

    @property
    def target_id(self) -> str:
        """ Gets the id of the target """
        return self._target_id

    def __str__(self):
        return f"[{self._target_id}] " + super().__str__()


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


def grouper(iterable: Iterable, size: int, fillvalue=None):
    """
    Iterates over iterable, yielding the contents in groups (tuples) of the requested size.

    From https://docs.python.org/3/library/itertools.html#itertools-recipes

    :iterable: the iterable to iterate in groups
    :size: the size of each group
    :fillvalue: used to fill out the final group when there are insufficient items in the iterable
    :returns: tuples, of the requested size, of values from iterator 
    """
    batchable = [iter(iterable)] * size
    return zip_longest(*batchable, fillvalue=fillvalue)

def powerset(iterable: Iterable, min_len: int=0):
    """
    Yields every possible subset of the source sequence. 

    i.e.: powerset([1,2,3]) -> () (1,) (2,) (3,) (1,2) (1,3) (2,3) (1,2,3)
    
    From https://docs.python.org/3/library/itertools.html#itertools-recipes
    with a modification to specify the minimum length of the subsets.

    :iterable: the iterable to yield from
    :min_size: the minimum length of the subsets to yield
    """
    seq = list(iterable)
    min_len = max(0, min_len)
    return chain.from_iterable(combinations(seq, r) for r in range(min_len, len(seq)+1))

def partitions_slices(sequence_len: int, min_slice_len: int=1, max_slice_len: int=None):
    """
    Yields all possible order-preserving lists of slices onto a sequence of the given length.

    i.e.: partitions_slices(3) -> [[0:3]] [[0:1],[1:3]] [[0:2],[2:3]] [[0:1],[1:2],[2:3]]

    Based on
    https://more-itertools.readthedocs.io/en/stable/_modules/more_itertools/more.html#partitions
    with a modifications to yield slices (so it doesn't need to see the sequence, with only the
    length required) and to restrict the minimum & maximum length of any slices.
    """
    max_slice_len = min(sequence_len, max_slice_len or sequence_len)
    for ix in powerset(range(1, sequence_len)):
        slices = [slice(i, j) for i, j in zip((0,) + ix, ix + (sequence_len,))]
        if all(min_slice_len <= sl.stop - sl.start <= max_slice_len for sl in slices):
            yield slices

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
    # cannot enumerate lcs as we're changing its content, pylint: disable=consider-using-enumerate
    for ix in range(len(lcs)):
        _mask = lightcurves.create_invalid_flux_mask(lcs[ix])

        # The configured quality masks are clipped because they're known to be they're distorted.
        if quality_masks is not None and len(quality_masks) > 0:
            for mask_times in (lightcurves.to_lc_time(t, lcs[ix]) for t in quality_masks):
                _mask &= (lcs[ix].time < np.min(mask_times)) | (np.max(mask_times) < lcs[ix].time)

        # We also look to clip any short isolated sections of the LCs, as they often contain little
        # useful information and yet have a detrimental affect on the effectiveness of detrending.
        sec_gap_th = min_section_gap_bins * lcs[ix].meta['FRAMETIM'] * lcs[ix].meta['NUM_FRM'] * u.s
        for sec in lightcurves.find_lightcurve_sections(lcs[ix], min_gap_duration=sec_gap_th,
                                                        yield_times=True):
            if max(sec) - min(sec) < min_section_dur:
                _mask &= (lcs[ix].time < min(sec)) | (lcs[ix].time > max(sec))

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
    - sector_times: a list containing a tuple of the (min, max) time in the sector
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
        ecl_data = lightcurves.find_and_characterise_eclipses(lc, ref_t0, period,
                                                              widthp, widths, depthp, depths,
                                                              phis, search_window_phase, verbose)

        # Stick the LC's min/max times as a tuple in a list as client code can always expect a list
        # and list values are automatically concatenated when LightCurves are stitched.
        lc.meta["sector_times"] = [(lc.time.min(), lc.time.max())]

        # We will use the revised/refined reference time as a starting position for the next sector
        ref_t0 = lc.meta["t0"] = ecl_data[0] or ref_t0
        lc.meta["sectors"] = [lc.sector]
        lc.meta |= dict(zip(
            ["primary_times", "primary_depths", "primary_completeness",
             "secondary_times", "secondary_depths", "secondary_completeness"],
            ecl_data[1:]
        ))


def join_lightcurves(src_lcs: Union[LightCurveCollection, List[LightCurve]]) -> LightCurve:
    """
    Join the source LightCurves into a single normalized and 'stitched' LightCurve.
    Builds on lightkurve's stitch() with better handling of the meta dictionary.

    :src_lcs: the original LightCurves from which to create the stitched copy
    :returns: a new LightCurve containing the newly grouped/stitched LightCurves
    """
    target = src_lcs[0].meta.get("target", src_lcs[0].meta["OBJECT"])

    # Normalize + combine the sectors in the grouping (also if there's only 1)
    join_lc = src_lcs.stitch(lambda lc: lc.normalize())

    # Update/combine metadata where appropriate. From fits file tend to be UCASE & ours are lcase.
    join_lc.meta["sectors"] = src_lcs.sector
    sec_list = [f"{s:02d}" if int(s) == s else f"{s:02.1f}" for s in src_lcs.sector]
    join_lc.meta["LABEL"] = f"{target} S{sec_list[0]}"
    if len(src_lcs) > 1:
        join_lc.meta["LABEL"] += (" to" if len(src_lcs) > 2 else " and") + f" S{sec_list[-1]}"

        # lightkurve's stitch appears smart enough to concat ndarray & list meta values.
        # However, some of the singular values seem to be from the last sector, when it is
        # useful if it were from the first sector (i.e.: t0, TSTART). Fix where necessary.
        for k in ["TSTART", "DATE-OBS", "SECTOR"]:
            if k in src_lcs[0].meta:
                join_lc.meta[k] = src_lcs[0].meta[k]

        for k in ["TELAPSE"]:
            if all(k in lc.meta for lc in src_lcs):
                join_lc.meta[k] = np.sum([lc.meta[k] for lc in src_lcs])

        for (k, d) in [("CROWDSAP", 1)]:
            join_lc.meta[k] = np.mean([lc.meta.get(k, d) for lc in src_lcs])

        # Revise the t0 time to that of the best/most complete in the combined LC
        if all(k in join_lc.meta for k in ["t0", "primary_times", "primary_completeness"]):
            t0_mask = np.isin(join_lc.meta["primary_times"], [l.meta["t0"] for l in src_lcs])
            if any(t0_mask):
                best_t0_ix = np.argmax(join_lc.meta["primary_completeness"][t0_mask])
                join_lc.meta["t0"] = join_lc.meta["primary_times"][t0_mask][best_t0_ix]
            else:
                join_lc.meta["t0"] = src_lcs[0].meta["t0"]
    return join_lc


def slice_lightcurve(src_lc: LightCurve, slices: List[slice]) -> LightCurveCollection:
    """
    Splits the source LightCurve into a new LightCurveCollection based on the passed slices.
    Handles re-labeling the sector (as .1, .2, ..., .n) and re-assigning the metadata.

    :src_lc: the original LightCurve which will be split
    :slices: the slices, appropriate to src_lc, indicating where the sub LightCurves are taken
    :returns: a new LightCurveCollection containing the newly split LightCurves
    """
    new_lcs = []
    if slices is not None:
        if isinstance(slices, slice):
            slices = [slices]

        target = src_lc.meta.get("target", src_lc.meta["OBJECT"])
        for ix, sec_slice in enumerate(slices, start=1):
            lc = src_lc.copy(True)[sec_slice]
            lc.sector += ix/10
            lc.meta["sectors"] = [lc.sector]
            lc.meta["LABEL"] = f"{target} S{lc.sector}"

            if len(lc) != len(src_lc):
                tstart, tend = min(lc.time), max(lc.time)
                lc.meta["TELAPSE"] = (tend - tstart).to(u.d).value
                if ix > 1:
                    lc.meta["TSTART"] = tstart.value

                # The hard work, splitting the eclipse data
                for ecl_type in ["primary", "secondary"]:
                    key_times = f"{ecl_type}_times"
                    if key_times in lc.meta:
                        mask = (src_lc.meta[key_times] >= tstart.value) \
                            & (src_lc.meta[key_times] <= tend.value)
                        lc.meta[key_times] = src_lc.meta[key_times][mask]
                        for key in [f"{ecl_type}_depths", f"{ecl_type}_completeness"]:
                            if key in lc.meta:
                                lc.meta[key] = src_lc.meta[key][mask]

                if "t0" in src_lc.meta and not tstart.value <= src_lc.meta["t0"] <= tend.value:
                    new_t0_ix = np.argmax(lc.meta["primary_completeness"])
                    lc.meta["t0"] = lc.meta["primary_times"][new_t0_ix]

            new_lcs += [lc]
    return LightCurveCollection(new_lcs)


def arrange_sector_groups(lcs: LightCurveCollection,
                          completeness_th: float=0.8,
                          min_eclipses: Tuple[int, int]=(2, 1),
                          max_group_size: int=None,
                          groups_override: List[List[int]]=None,
                          allow_slice: bool=False,
                          verbose: bool=False) -> List[List[int]]:
    """
    Will make the most effective arrangement of LightCurves to support JKTEBOP fitting. This will
    balance need for sufficient coverage for each fitting to achieve a reliable output, while
    running as many fits as possible across the breadth of time in which we have observations.
    This uses the eclipse timing & completeness metadata added by add_eclipse_meta_to_lightcurves().

    :lcs: the LightCurveCollection containing our potential fitting targets
    :completness_th: threshold percentage of an eclipse we require to consider it complete/usable
    :min_eclipses: the minimum eclipse count criteria for a usable group or subsector
    :max_group_size: the maximum number of sectors to combine for a group, or no max if None
    :groups_override: if set, the grouping logic will be bypassed, and these used (to be deprecated)
    :allow_slice: whether to allow sectors to be sliced into sections, subject to eclipse criteria
    :verbose: whether or not to send messages to stdout with details of the group decisions made
    :returns: a new LightCurveCollection containing the newly grouped/split LightCurves
    """
    if isinstance(min_eclipses, Number):
        min_eclipses = (int(np.ceil(min_eclipses / 2)), int(np.floor(min_eclipses / 2)))
    allow_slice = allow_slice and groups_override is None # override switches off slicing
    min_gap_dur = 0.25 * u.d

    keys_time = ("primary_times", "secondary_times")
    keys_compl = ("primary_completeness", "secondary_completeness")
    def eclipse_counts(lc: LightCurve, section_slice: slice=None) -> Tuple[int, int]:
        if section_slice: # We have to count the eclipses within the times of the slice
            times = lc.time[section_slice].value
            from_time, to_time = min(times), max(times)
            ecl_sums = []
            for key_time, key_compl in zip(keys_time, keys_compl):
                ecl_mask = (from_time <= lc.meta[key_time]) & (lc.meta[key_time] <= to_time)
                ecl_sums += [sum(lc.meta[key_compl][ecl_mask] >= completeness_th)]
            return tuple(ecl_sums)
        return tuple(sum(lc.meta[k] > completeness_th) for k in keys_compl)

    def is_usable_group(grp_ecl_counts: List[Tuple[int, int]]) -> bool:
        ecl_sums = np.sum(grp_ecl_counts, axis=0)
        return max(ecl_sums) >= max(min_eclipses) and min(ecl_sums) >= min(min_eclipses)

    def is_usable_section(from_ix, to_ix, lc) -> bool:
        return is_usable_group([eclipse_counts(lc, slice(from_ix, to_ix+1))])

    # Make sure the sectors are sorted by sector number for grouping to work
    lcs = LightCurveCollection(sorted(lcs, key=lambda l: l.sector))
    out_lcs = []
    if groups_override is None:
        # Isolate the sectors into contiguous blocks; so [1,2,4,5,6,8] becomes [[1,2], [4,5,6], [8]]
        # By making a key of the sector number minus its list index, we have a value which we can
        # group by as it remains unchanged within each block of contiguously incrementing values.
        for _, block in groupby(enumerate(lcs.sector), key=lambda ix_sec: ix_sec[1] - ix_sec[0]):
            blk_mask = np.isin(lcs.sector, list(b for _, b in block))
            blk_sectors = np.array([lc.sector for lc in lcs[blk_mask]])
            blk_ecl_counts = np.array([eclipse_counts(lc) for lc in lcs[blk_mask]])

            if verbose:
                bname = f"{blk_sectors[0]}" + (f"-{blk_sectors[-1]}" if len(blk_sectors)>1 else "")
                print("From the block of sectors (" + bname + ")",
                      "the group(s) with sufficient coverage for fitting are:", end=" ")

            # All combinations of contiguous sectors (slices) where eclipse criteria are met.
            all_sets_slices = [sls for sls in partitions_slices(len(blk_sectors), 1, max_group_size)
                                        if all(is_usable_group(blk_ecl_counts[sl]) for sl in sls)]

            # Choose the best set of groups for this block
            chosen_grp_slices = []
            if len(all_sets_slices) == 1:
                chosen_grp_slices = all_sets_slices[0]
            elif len(all_sets_slices) > 1:
                # Use set with the most groups, with least variance in eclipse counts as tie breaker
                all_sets_sizes = [len(sls) for sls in all_sets_slices]
                largest_sets_ixs = np.argwhere(all_sets_sizes == np.amax(all_sets_sizes)).flatten()
                if len(largest_sets_ixs) == 1:
                    chosen_grp_slices = all_sets_slices[largest_sets_ixs[0]]
                else:
                    ecl_vars = [np.var([sum(blk_ecl_counts[sl]) for sl in all_sets_slices[ix]])
                                                                        for ix in largest_sets_ixs]
                    chosen_grp_slices = all_sets_slices[largest_sets_ixs[np.argmin(ecl_vars)]]

            # Create the LCs for this group. Optionally, where a group is 1 LC try to slice it.
            for group in [blk_sectors[sls] for sls in chosen_grp_slices]:
                grp_lcs = lcs[np.isin(lcs.sector, group)]
                if allow_slice and len(grp_lcs) == 1 and \
                        len(sls := [*lightcurves.find_lightcurve_sections(grp_lcs[0], min_gap_dur,
                                                                          is_usable_section)]) > 1:
                    # Normalize the LC so it's consistent with any that are joined
                    sec_lcs = slice_lightcurve(grp_lcs[0].normalize(), sls).data
                    out_lcs += sec_lcs
                    if verbose:
                        print(" ".join(f"[{lc.sector}]" for lc in sec_lcs), end=" ")
                else:
                    # Call even if only one member as this will also update its meta dict
                    out_lcs += [join_lightcurves(grp_lcs)]
                    if verbose:
                        print(group, end=" ")

            if verbose:
                print()

    return LightCurveCollection(out_lcs)


def append_mags_to_lightcurves_and_detrend(lcs: LightCurveCollection,
                                           detrend_gap_th: Union[u.Quantity[u.d], float]=2,
                                           detrend_poly_degree: int=2,
                                           detrend_iterations: int=3,
                                           flatten: bool=False,
                                           durp: Union[float, UFloat]=None,
                                           durs: Union[float, UFloat]=None,
                                           override_poly_on_flatten: bool=True,
                                           verbose: bool=False):
    """
    Append delta_mag and delta_mag_err columns calculated from normalized fluxes to each
    LightCurve, then detrend the delta_mag column by subtracting a fitted polynomial.
    Optionally the fluxes may be flattened outside of any previously detected eclipses,
    prior to calculating the magnitude columns.

    LightCurves that have been flattened will have a flat_mask array added to their meta dict.

    The detrending poly will be overriden to zero if flattening is requested, unless the
    override_poly_on_flatten argument is set to False.

    :lcs: the LightCurveCollection containing our potential fitting targets
    :detrend_gap_th: the threshold gap time beyond which a detrending section break is triggered
    :detrend_poly_degree: the degree of the detrending polynomial to fit
    :detrend_iterations: number of iterations to run during detrending to fit a polynomial
    :flatten: whether or not to flatten the LightCurve fluxes prior to calculating the magnitudes
    :durp: the duration of the primary eclipses required for flattening
    :durs: the duration of the secondary eclipses required for flattening
    :override_poly_on_flatten: whether to override the poly params if flatten == True
    :verbose: whether or not to send messages to stdout with details of the actions taken
    """
    if flatten and (durp is None or durs is None):
        raise ValueError("durp and durs required if flatten == True")
    if flatten:
        durp = nominal_value(durp) * 1.1
        durs = nominal_value(durs) * 1.1
    if not isinstance(detrend_gap_th, u.Quantity):
        detrend_gap_th = detrend_gap_th * u.d

    if flatten and override_poly_on_flatten:
        # Flattening will address the data trends so don't need the full detrend poly.
        # We'll just fit and subtract a flat line to rectify the delta_mags to zero.
        detrend_poly_degree = 0
        if verbose:
            print(f"Flattening requested so setting detrend_poly_degree to {detrend_poly_degree}.")

    # Cannot enumerate lcs as we're changing its content. pylint: disable=consider-using-enumerate
    for ix in range(len(lcs)):
        label = lcs[ix].meta["LABEL"]

        pri_times = lcs[ix].meta.get("primary_times", [])
        sec_times = lcs[ix].meta.get("secondary_times", [])
        eclipse_mask = lightcurves.create_eclipse_mask(lcs[ix], pri_times, sec_times, durp, durs)

        if flatten:
            if verbose:
                num_ecl = len(np.ma.clump_masked(np.ma.masked_where(eclipse_mask, eclipse_mask)))
                print(f"Flattening the {label} LC fluxes outside of {num_ecl} masked eclipse(s).")
            lcs[ix] = lcs[ix].flatten(mask=eclipse_mask)
            lcs[ix].meta["flat_mask"] = eclipse_mask

        # Create detrended & rectified delta_mag/delta_mag_err columns, by fitting & subtracting a
        # polynomial to delta_mags outside the eclipses.
        lightcurves.append_magnitude_columns(lcs[ix], "delta_mag", "delta_mag_err")
        for sl in lightcurves.find_lightcurve_sections(lcs[ix], min_gap_duration=detrend_gap_th):
            lcs[ix][sl]["delta_mag"] -= lightcurves.fit_polynomial(lcs[ix].time[sl],
                                                                   lcs[ix]["delta_mag"][sl],
                                                                   detrend_poly_degree,
                                                                   detrend_iterations,
                                                                   fit_mask=~eclipse_mask[sl])
    if verbose:
        print(f"Added delta_mag & delta_mag_err columns to {len(lcs)} LC(s),",
               "then detrended & rectified the mags to zero by subtracting polynomials",
              f"(order={detrend_poly_degree}) fitted outside the eclipes.")


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

        if noms is None or len(noms) == 0:
            agg_params[k][0] = None
        else:
            lo, med, hi = np.quantile(noms, q=quantiles)
            agg_params[k][0] = ufloat(med, max(np.mean([med-lo, hi-med]), med * min_uncertainty_pc))
    return agg_params[0]


def force_seed_on_dropout_layers(estimator: Estimator, seed: int=42):
    """
    Forces a seed onto the dropout layers of the model wrapped by the passed Estimator.
    Setting this is a way of making subsequent MC Dropout predictions repeatable.
    Definitely not for "live" but may be useful for testing where repeatability is required.
    
    :estimator: the estimator to modify
    :seed: the new seed value to assign
    """
    # pylint: disable=protected-access
    dropout_layers = (l for l in estimator._model.layers if isinstance(l, layers.Dropout))
    for ix, layer in enumerate(dropout_layers, start=1):
        sg = layer.seed_generator
        new_seed = sg.backend.convert_to_tensor(np.array([0, seed*ix], dtype=sg.state.dtype))
        sg.state.assign(new_seed)


def fit_target_lightcurves(lcs: LightCurveCollection,
                           input_params: Union[dict[str], List[dict[str]]],
                           read_keys: List[str],
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

    The input_params may either be a single dict of values, shared by all LighCurves, or
    a list of dicts, one for each LightCurve. In either case, the following additional fitting
    instructions are set to values specific to each lightcurve;
    - poly fit instructions, which are calculated for the time range of each lightcurve

    If not already present within the input_params the following values, which are likely to change
    with each LightCurve, are set based on values read from each LightCurve's metadata;
    - t0 / primary_epoch time is read from the the t0 meta value
    - L3, which is calculated as 1 - TESS CROWDSAP meta value

    :lcs: the source lightcurves, which must have the time, delta_mag and delta_mag_err columns
    :input_params: the set initial input params to the fitting process for each lightcurve
    :read_keys: the set of fitted output params to read and return for each fit
    :task: the jktebop task to be executed
    :iterations: the number of iterations to run if task == 3, otherwise ignored
    :max_workers: the maximum number of concurrent fits to run
    :max_attempts: the maximum number of attempts to make - ignored unless task == 3
    :timeout: the timeout for any individual fit - will raise a TimeoutExpired if not completed
    :returns: a list of dictionaries containing the fitted parameters matching read_keys and the
    paths to the various jktebop files associated with the fitting
    """
    if isinstance(input_params, dict):
        input_params = [input_params.copy()] * len(lcs)
    elif len(lcs) != len(input_params):
        raise ValueError("Expected either one shared set of input params, or one set per LC. " +
                         f"Got {len(lcs)} LightCurve(s) and {len(input_params)} set(s) of params.")

    # These params are known to vary by LC and have values stored in LC meta dicts,
    # so we can set them as params if they have not already be set in client code.
    for in_params, lc in zip(input_params, lcs):
        t0 = lc.meta.get("t0", lc.meta.get("primary_epoch", in_params.get("t0", None)))
        in_params.setdefault("t0", t0)
        in_params.setdefault("primary_epoch", t0)
        in_params.setdefault("L3", max(0, 1-lc.meta.get("CROWDSAP", 1)))

    task_params = { "task": task, "simulations": iterations if task == 8 else "" }
    if task != 3:
        max_attempts = 1
    max_workers = min(len(lcs), max_workers or 1)

    lcs_gen = (lc[lc.meta.get("clip_mask", np.ones((len(lc),), bool))] for lc in lcs)
    fit_stems_gen = (file_prefix + "-" + lc.meta["LABEL"].replace(" ", "-").lower() for lc in lcs)

    # If we're to run in parallel this indicates to _fit_target to write JKTEBOP console output to
    # stdout less frequently, so it is less likely that the fitting narrative will be interleaved.
    hold_stdout = max_workers > 1

    # Create the args for each lc/call to _fit_target.
    # Can't have func take an lc or masked columns as they do not pickle.
    iter_params = ((lc["time"].unmasked.value,
                    lc["delta_mag"].unmasked.value,
                    lc["delta_mag_err"].unmasked.value,
                    in_params | task_params,
                    read_keys,
                    fit_stem,
                    _create_lc_std_further_process_cmds(lc),
                    max_attempts,
                    timeout,
                    hold_stdout) \
                    for lc, in_params, fit_stem in zip(lcs_gen, input_params, fit_stems_gen))

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


def _create_lc_std_further_process_cmds(lc: LightCurve) -> List[str]:
    """
    Creates a standard set of JKTEBOP processing instructions for appending to an in file.
    The instructions set up poly fits for scale factor and chi^sq adjustment

    :lc: the source LightCurve
    :returns: the list of processing instructions
    """
    # Filter segments to those with observations to prevent jktebop error (no datapoints for poly)
    sf_segs =[s for s in lc.meta.get("sector_times", [(lc.time.min(), lc.time.max())])
                        if any((min(s) <= lc.time) & (lc.time <= max(s)))]
    return [""] + jktebop.build_poly_instructions(sf_segs, "sf", 1) + ["", "chif", ""]


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
    best_file_params = { }
    best_attempt = 1
    msgs = []
    all_keys = list(jktebop._param_file_line_beginswith.keys()) # pylint: disable=protected-access
    stdout_to_log = StringIO()

    # JKTEBOP will fail if it finds files from a previous fitting
    fit_dir = jktebop.get_jktebop_dir()
    for file in fit_dir.glob(file_stem + ".*"):
        file.unlink()

    # Ensure filenames are not going to be > 50 chars or JKTEBOP will truncate them when writing
    # its output, which causes subsequent errors when expected files are not found.
    # Allow a file stem up to 42 chars to leave space for attempt suffixes and file extensions.
    max_stem_len = 42
    if len(file_stem) > max_stem_len:
        # Try truncating after last possible +, leaving the + trailing so it's clear we've truncated
        if (pix := file_stem.rfind("+", 0, max_stem_len-1)) > -1:
            file_stem = file_stem[:pix+1]
        file_stem = file_stem[:max_stem_len] # Fallback/catch-all

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

    # Preserve the initial inputs as we'll progressively update the attempt intputs if retries occur
    next_att_in_params = input_params.copy()
    converged = False
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

        msg = None
        with PassthroughTextWriter(stdout_to, output2=stdout_to_log, hold_output=hold_stdout,
                                inspect_func=lambda ln: "Warning: a good fit was not found" in ln) \
                            as stdout_inspect:
            jktebop.write_in_file(in_fname, append_lines=append_lines, **next_att_in_params)

            # Blocks on the JKTEBOP task until we can parse the newly written par file contents
            # to read out the revised values for the superset of potentially fitted parameters.
            plines = jktebop.run_jktebop_task(in_fname, par_fname, None, stdout_inspect, timeout)
            att_out_params = jktebop.read_fitted_params_from_par_lines(plines, all_keys, True)
            converged = not stdout_inspect.inspect_flag

            att_file_params["warn_msgs"] = jktebop.read_warnings_from_par_file(par_fname)
            stdout_inspect.write_lines(att_file_params["warn_msgs"])

            if attempt == 1:
                # The fallback position being the outputs from the 1st attempt regardless of success
                best_attempt = 1
                best_out_params = att_out_params.copy()
                best_file_params = att_file_params.copy()

            if not converged:
                msg = f"Attempt {attempt} of {max_attempts} of {file_stem} didn't fully converge."
                if max_attempts > 1:
                    if attempt < max_attempts:
                        # Copy the output from this, to the input of the next attempt. Leave ephem
                        # & LD params at original values as they shouldn't be highly perturbed in
                        # fitting and we avoid a problem with unphysical params if prev fit haywire.
                        next_att_in_params |= { k: v for k, v in att_out_params.items()
                                        if k not in ["period", "t0"] and not k.startswith("LD") }
                        msg += " Will retry from the final position of this attempt."
                    else:
                        msg += f" Will revert to the results from attempt {best_attempt}."
                else:
                    msg += " Only 1 attempt allowed so will return these results."
            else:
                msg = f"Attempt {attempt} of {max_attempts} to fit " \
                        + f"{file_stem} completed successfully."
                if attempt > 1: # A retry fit worked, these become the best params
                    best_out_params = att_out_params
                    best_file_params = att_file_params

            if msg is not None:
                stdout_inspect.write("** " + msg + "\n")
                msgs += [msg]

            if converged:
                break

    # Include any progress messages we've generated across all attempts
    best_file_params["msgs"] = msgs
    best_file_params["converged"] = converged
    if stdout_to_log is not None:
        stdout_to_log.seek(0)
        best_file_params["log"] = stdout_to_log.read()
    return { k: best_out_params.get(k, None) for k in read_keys } | best_file_params
