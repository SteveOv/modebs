"""
Low level utility functions for light curve ingest, pre-processing, estimation and fitting.
"""
#pylint: disable=no-member
from typing import Union, List, Iterable, Tuple, Generator
from pathlib import Path

import numpy as np
from scipy import interpolate
import astropy.units as u
from astropy.io import fits
from astropy.time import Time, TimeDelta

import lightkurve as lk
from lightkurve import LightCurve, FoldedLightCurve, LightCurveCollection


def find_lightcurves(search_term: any,
                     download_dir: Path,
                     sectors: Union[int, List[int]]=None,
                     mission: str="TESS",
                     author: str="SPOC",
                     exptime: Union[str, float]=None,
                     flux_column: str="sap_flux",
                     quality_bitmask: Union[str, int]="default",
                     force_mast: bool=False,
                     verbose: bool=True) -> LightCurveCollection:
    """
    This performs both a search and download on Lightkurve assets to the requested download_dir.
    Unless force_mast is True, it will try to service the request from the existing contents the
    of download_dir, where possible avoiding potentially time consuming MAST queries.

    The arguments are the most used arguments associated with the Lightkurve search_lightcurve()
    and results.download_all() functions. Values are required for sectors, mission and author in
    order to be able to service the request from the download_dir. The assumption is made that the
    download dir is exclusive to this target as we cannot tie all potential values of the target
    search criterion directly to the contents of fits.
    
    :search_term: the target search term
    :download_dir: the local directory under which the assets are/will be stored
    :sectors: the sector or sectors search term
    :mission: the mission search term
    :author: the author search term
    :exptime: the exposure time criterion; None, text ("long", "short" or "fast") or numeric seconds
    :flux_column: the flux column to select when loading the assets
    :quality_bitmask: applied to the Quality col; text ("default", "hard" or "hardest") or bit mask
    :force_mast: if True will always bypass local files and search/download from MAST
    :verbose: if True will output some diagnostics text to the console
    """
    # pylint: disable=too-many-locals, too-many-arguments, too-many-branches, unnecessary-lambda-assignment
    lcs = None
    if mission.lower() not in ["tess", "hlsp"]:
        raise ValueError("only missions TESS and HLSP are currently supported")
    if author.lower() not in ["spoc", "tess-spoc"]:
        raise ValueError("only authors SPOC and TESS-SPOC are currently supported")

    if sectors and not isinstance(sectors, Iterable):
        sectors = [sectors]

    if verbose:
        print(f"Searching for light curves based on; search term={search_term}, sectors={sectors},",
              f"mission={mission}, author={author} and exptime={exptime}")

    if not force_mast and sectors and mission and author:
        # We can only reliably shortcut if we have been given a full list criteria.
        if verbose:
            print("Looking for previously downloaded fits within", download_dir)
        # Known filename patterns for author on TESS. The mission doesn't seem to matter:
        # TESS-SPOC: hlsp_tess-spoc_tess_phot_{tic_number:016d}_s{sector:04d}_tess_v1_lc.fits
        # SPOC:      tess{date:13d}_s{sector:04d}_{tic_number:016d}_{sequence??:04d}_lc.fits
        if author.lower() == "spoc":
            glob_pattern = "**/tess*_lc.fits"
        else:
            glob_pattern = "**/hlsp_tess-spoc_*_lc.fits"
        fits_files = [*download_dir.glob(glob_pattern)]

        if fits_files:
            # We have local files that match the file pattern for mission/author
            if verbose:
                print(f"Found {len(fits_files)} existing fits matching mission and author criteria")

            # select only those that match the sector & exptime
            if not exptime:
                chk_exp = None
            else:
                # This is what lambdas are for despite what pylint keeps wittering on about!
                if not isinstance(exptime, str):
                    chk_exp = lambda fits_exptime: fits_exptime == exptime
                elif exptime.lower() == "short":
                    chk_exp = lambda fits_exptime: 60 <= fits_exptime <= 120
                elif exptime.lower() == "fast":
                    chk_exp = lambda fits_exptime: fits_exptime < 60
                else:
                    chk_exp = lambda fits_exptime: fits_exptime > 120

            tic_term = None
            if isinstance(search_term, str) and search_term.upper().startswith("TIC"):
                tic_term = int(''.join(filter(str.isdigit, search_term)))

            hduls = [
                h for h in (fits.open(ff) for ff in fits_files)
                    if h[0].header["SECTOR"] in sectors
                        and (not tic_term or h[1].header["TICID"] == tic_term)
                        and (not chk_exp or chk_exp(h[1].header["FRAMETIM"]*h[1].header["NUM_FRM"]))
            ]

            if len(hduls) == len(sectors):
                # We're on! Load these into a collection and return them.
                if verbose:
                    print(f"Found the required {len(hduls)} fits file(s) meeting the TIC, sector &",
                          "exptime criteria. Will load the requested light curves from these.")
                lcs = LightCurveCollection(
                    lk.read(h.filename(), flux_column=flux_column, quality_bitmask=quality_bitmask)
                        for h in sorted(hduls, key=lambda hdul: hdul[0].header["SECTOR"])
                )

    if not lcs:
        if verbose:
            print("Performing a MAST query based on the criteria.")
        # We've not been able to service the request from already dl assets: usual search & download
        results = lk.search_lightcurve(search_term, sector=sectors, mission=mission,
                                       author=author, exptime=exptime)
        if verbose:
            print("Criteria met by a MAST search yielding a", results)

        # Work around a recent issues with DL assets - suggested by Pierre Maxted
        if "dataUrl" not in results.table.colnames and "dataURI" in results.table.colnames:
            results.table["dataURL"] = results.table["dataURI"]
        download_dir.mkdir(parents=True, exist_ok=True)
        lcs = results.download_all(quality_bitmask, f"{download_dir}",
                                   flux_column=flux_column, cache=True)
    return lcs


def apply_invalid_flux_masks(lc: LightCurve,
                             verbose: bool = False) -> LightCurve:
    """
    Will return the passed LightCurve with any fluxes with value of NaN or less than Zero masked.

    :lc: the LightCurve to mask
    :verbose: whether or not to generate progress messages
    :returns: the masked LightCurve
    """
    if not isinstance(lc, LightCurve):
        raise TypeError(f"lc is {type(lc)}. Expected a LightCurve.")

    mask = (np.isnan(lc.flux)) | (lc.flux < 0)
    count_nan_masked = sum(mask.unmasked)
    if verbose:
        print(f"NaN/negative flux masks match {count_nan_masked} row(s).")
    return lc[~mask]


def apply_time_range_masks(lc: LightCurve,
                           time_ranges: List[Tuple[Time, Time]] = None,
                           verbose: bool = False) -> LightCurve:
    """
    Will return a copy of the passed LightCurve haveing first masked out any indicated time ranges.

    :lc: the LightCurve to mask
    :mask_time_ranges: one or more (from, to) Time ranges to mask out
    :verbose: whether or not to generate progress messages
    """
    if not isinstance(lc, LightCurve):
        raise TypeError(f"lc is {type(lc)}. Expected a LightCurve.")

    mask = [False] * len(lc)
    if time_ranges is not None:
        if not isinstance(time_ranges, Iterable):
            raise TypeError(f"mask_time_ranges is {type(time_ranges)}. Expected an Iterable.")
        for tr in time_ranges:
            mask |= mask_from_time_range(lc, tr)

    if verbose:
        print(f"Time range mask(s) matched {sum(mask)} row(s).")

    return lc[~mask]


def mask_from_time_range(lc: LightCurve,
                         time_range: Union[Tuple[Time, Time], Time]) \
                            -> List[bool]:
    """
    Creates a mask applicable to the LightCurve over the indicated time range,
    with the mask being applied to all datapoints within the range inclusive.

    :lc: the LightCurve to create the mask for
    :time_range: the (from, to) time range to mask
    """
    if not isinstance(time_range, (Iterable, Time)):
        raise TypeError(f"time_range is {type(time_range)}. Expected an Iterable or Time.")
    if not isinstance(lc, LightCurve):
        raise TypeError(f"lc is {type(lc)}. Expected a LightCurve.")

    time_range = to_lc_time(time_range, lc)
    return (lc.time >= time_range.min()) & (lc.time <= time_range.max())


def to_lc_time(value: Union[Time, np.double, Tuple[np.double], List[np.double,]],
            lc: LightCurve) \
                -> Time:
    """
    Converts the passed numeric value to an astropy Time.
    The magnitude of the time will be used to interpret the format to match LC.

    :value: the value to be converted
    :lc: the light-curve to match format with
    """
    if isinstance(value, Time):
        if value.format == lc.time.format and value.scale == lc.time.scale:
            return value
        raise ValueError("Value's time format/scale does not match the Lightcurve's")

    if isinstance(value, Iterable):
        return Time([to_lc_time(v, lc) for v in value])

    # Otherwise try to match the time format and scale to the Lightcurve
    if value < 4e4:
        if lc.time.format == "btjd":
            return Time(value, format="btjd", scale=lc.time.scale)
        raise ValueError(f"Unexpected value/format ({value}/{lc.time.format}) combination.")
    else:
        if value < 2.4e6:
            value += 2.4e6
        return Time(value, format="jd", scale=lc.time.scale)


def bin_lightcurve(lc: LightCurve,
                   time_bin_seconds: Time,
                   verbose: bool = False) -> LightCurve:
    """
    Will return a binned copy of the passed LightCurve

    :lc: the LightCurve to bin
    :time_bin_seconds: the length of the revised bins
    :verbose: whether or not to generate progress messages
    :returns: the binned LightCurve
    """
    if isinstance(time_bin_seconds, (float, np.double, int)):
        time_bin_seconds *= u.s

    # Can only re-bin to longer than the existing bins
    int_time = (lc.meta["INT_TIME"] + lc.meta["READTIME"]) * u.min
    if time_bin_seconds > int_time:
        orig_len = len(lc)
        lc = lc.bin(time_bin_size=time_bin_seconds, aggregate_func=np.nanmean)
        lc = lc[~np.isnan(lc.flux)] # Binning may have re-introduced NaNs
        if verbose:
            print(f"After binning the light curve has reduced from {orig_len} to {len(lc)} rows.")
    elif verbose:
        print(f"Light curve already in bins >= {time_bin_seconds}. Will leave unchanged.")
    return lc


def append_magnitude_columns(lc: LightCurve,
                             name: str = "delta_mag",
                             err_name: str = "delta_mag_err"):
    """
    This will append a relative magnitude and corresponding error column to the passed LightCurve
    based on the values in the flux column.

    :lc: the LightCurve to update
    :name: the name of the new magnitude column
    :err_name: the name of the corresponding magnitude error column
    """
    lc[name] = u.Quantity(-2.5 * np.log10(lc.flux.value) * u.mag)
    lc[err_name] = u.Quantity(
        2.5
        * 0.5
        * np.abs(
            np.subtract(
                np.log10(np.add(lc.flux.value, lc.flux_err.value)),
                np.log10(np.subtract(lc.flux.value, lc.flux_err.value))
            )
        )
        * u.mag)


def fit_polynomial(times: Time,
                   ydata: u.Quantity,
                   degree: int = 2,
                   iterations: int = 2,
                   res_sigma_clip: float = 1.,
                   reset_const_coeff: bool = False,
                   include_coeffs: bool = False,
                   verbose: bool = False) \
                    -> Union[u.Quantity, Tuple[u.Quantity, List]]:
    """
    Will calculate a polynomial fit over the requested time range and y-data
    values. The fit is iterative; after each iteration the residuals are
    evaluated against a threshold defined by the StdDev of the residuals
    multiplied by res_sigma_clip; any datapoints with residuals greater than
    this are excluded from subsequent iterations.  This approach will exclude
    large y-data excursions, such as eclipses, from influencing the final fit.

    :times: the times (x data)
    :ydata: data to fit to
    :degree: degree of polynomial to fit.  Defaults to 2.
    :iterations: number of fit iterations to run.
    :res_sigma_clip: the factor applied to the residual StdDev to calculate
    the clipping threshold for each new iteration.
    :reset_const_coeff: set True to reset the const coeff to 0 before final fit
    :include_coeffs: set True to return the coefficients with the fitted ydata
    :returns: fitted y data and optionally the coefficients used to generate it.
    """
    # pylint: disable=too-many-arguments, too-many-locals
    pivot_ix = int(np.floor(len(times) / 2))
    pivot_jd = times[pivot_ix].jd
    time_values = times.jd - pivot_jd

    fit_mask = [True] * len(ydata)
    for remaining_iterations in np.arange(iterations, 0, -1):
        # Fit a polynomial to the masked data so that we find its coefficients.
        # For the first iteration this will be all the data.
        coeffs = np.polynomial.polynomial.polyfit(time_values[fit_mask],
                                                  ydata.value[fit_mask],
                                                  deg=degree,
                                                  full=False)

        if remaining_iterations > 1:
            # Find and mask out those datapoints where the residual to the
            # above poly lies outside the requested sigma clip. This stops
            # large excursions, such as eclipses, from influencing the poly fit.
            poly_func = np.polynomial.Polynomial(coeffs)
            fit_ydata = poly_func(time_values)
            resids = ydata.value - fit_ydata
            fit_mask &= (np.abs(resids) <= (np.std(resids)*res_sigma_clip))
        else:
            # Last iteration we generate the poly's y-axis datapoints for return
            if reset_const_coeff:
                if verbose:
                    print("\tResetting const/0th coefficient to zero on request.")
                coeffs[0] = 0
            poly_func = np.polynomial.Polynomial(coeffs)
            fit_ydata = poly_func(time_values) * ydata.unit
            if verbose:
                c_list = ", ".join(f'c{ix}={c:.6e}' for ix, c in enumerate(poly_func.coef))
                print(f"\tGenerated polynomial; y = poly(x, {c_list})",
                      f"(sigma(fit_ydata)={np.std(fit_ydata):.6e})")

    return (fit_ydata, coeffs) if include_coeffs else fit_ydata


@u.quantity_input(period=u.d)
def flatten_lightcurve(lc: LightCurve,
                       mask_time_ranges: List[Union[Tuple[Time, Time], Tuple[float, float]]]=None,
                       period: u.Quantity=None,
                       verbose: bool=False,
                       **kwargs) -> LightCurve:
    """
    Will return a flattened copy of the passed Lightcurve. This uses the LightCurve.flatten()
    function and you can pass in a set of kwargs to be passed on to flatten(). As a convenience,
    rather than setting the flatten() function's mask argument directly, you can specify the
    mask_time_ranges List[(from, to)] and period arguments from which the mask is generated.
    You must supply either mask_time_ranges & period values or a mask kwarg.

    :lc: the LightCurve to flatten
    :mask_time_ranges: optional List of time ranges for which a mask is generated
    :period: the orbital period to use with the above time ranges
    :verbose: whether or not to generate progress messages
    :kwargs: to be passed directly to the LightCurve.flatten() function
    :returns: the flattened LightCurve
    """
    if mask_time_ranges and period is not None:
        # This will be used to set the mask kwargs item, so will override it if already present.
        if verbose:
            print(f"Creating a flatten transit mask from {len(mask_time_ranges)}",
                  f"transit time range(s) and the orbital period of {period}.")
        transit_times = [to_lc_time(np.mean(t), lc) for t in mask_time_ranges]
        durations = [max(t)-min(t) for t in mask_time_ranges]
        period = [period] * len(transit_times)
        kwargs["mask"] = lc.create_transit_mask(period, transit_times, durations)
    if "mask" in kwargs:
        pass
    else:
        raise ValueError("Must specify mask_time_ranges and period, or give a mask kwarg")

    if verbose:
        print("Flattening the light curve")
    return lc.flatten(**kwargs)


def get_sampled_phase_mags_data(flc: FoldedLightCurve,
                                num_bins: int = 1024,
                                phase_pivot: Union[u.Quantity, float]=0.75,
                                flc_rollover: int = 200,
                                interp_kind: str="linear") \
                                    -> Tuple[u.Quantity, u.Quantity]:
    """
    A data reduction function which gets a reduced set of phase and delta magnitude data in
    the requested number of equal size bins. 
    
    The data is sourced by sampling the passed FoldedLightCurve.  In case this does not extend over
    a complete phase, rows are copied over from opposite ends of the phase space data to extend the
    coverage.  The number of rows copied is controlled by the flc_rollover argument.

    :flc: the source FoldedLightCurve
    :num_bins: the number of equally spaced rows to return
    :phase_pivot: the pivot point about which the fold phase was wrapped to < 0.
    :flc_rollover: the number of row to extend the ends of the source phases by
    :interp_kind: the 1d interpolation algorithm to use when reducing the fold
    :returns: a tuple with requested number or phases and delta magnitudes
    """
    source_phases = np.concatenate([
        flc.phase[-flc_rollover:] -1.,
        flc.phase,
        flc.phase[:flc_rollover] +1.
    ])

    source_delta_mags = np.concatenate([
        flc["delta_mag"][-flc_rollover:],
        flc["delta_mag"],
        flc["delta_mag"][:flc_rollover]
    ])

    # If there is a phase wrap then phases above the pivot will have been
    # wrapped around to <0. Work out what the expected minimum phase should be.
    # Also, ensure we don't try to interpolate beyond the start of the data!
    min_phase = u.Quantity(0)
    if phase_pivot is not None:
        if isinstance(phase_pivot, u.Quantity) and phase_pivot.value:
            min_phase = phase_pivot.value - 1
        elif phase_pivot:
            min_phase = u.Quantity(phase_pivot - 1)
    min_phase = max(min_phase, source_phases.min())

    interp = interpolate.interp1d(source_phases, source_delta_mags, kind=interp_kind)
    reduced_phases = np.linspace(min_phase, min_phase + 1., num_bins + 1)[:-1]
    reduced_mags = interp(reduced_phases)
    return (reduced_phases, reduced_mags)


def find_lightcurve_segments(lc: LightCurve,
                             threshold: TimeDelta,
                             return_times: bool=False) \
                                -> Generator[Union[Tuple[int, int], Tuple[Time, Time]], any, None]:
    """
    Finds the start and end of contiguous segments in the passed LightCurve. These are subsets of
    where the gaps between bins does not exceed the passed threshold. Gaps greater then the
    threshold are treated as boundaries between segments.

    :lc: the source LightCurve to parse for gaps/segments.
    :threshold: the threshold gap time beyond which a segment break is triggered
    :return_times: if true start/end times will be yielded, otherwise the indices
    :returns: a generator of segment (start, end).
    If no gaps found this will yield a single (start, end) for the whole LightCurve.
    """
    if not isinstance(threshold, TimeDelta):
        threshold = TimeDelta(threshold * u.d)

    # Much quicker if we use primatives - make sure we work in days
    threshold = threshold.to(u.d).value
    times = lc.time.value

    last_ix = len(lc) - 1
    segment_start_ix = 0
    for this_ix, previous_time in enumerate(times, start = 1):
        if this_ix > last_ix or times[this_ix] - previous_time > threshold:
            if return_times:
                yield (lc.time[segment_start_ix], lc.time[this_ix - 1])
            else:
                yield (segment_start_ix, this_ix - 1)
            segment_start_ix = this_ix
