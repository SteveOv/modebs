"""
Low level utility functions for light curve ingest, pre-processing, estimation and fitting.
"""
from typing import Union, List, Tuple, Generator
from pathlib import Path

import numpy as np
import astropy.units as u
from astropy.time import Time, TimeDelta
from lightkurve import LightCurve, LightCurveCollection, SearchResult


def load_lightcurves(results: SearchResult,
                     quality_bitmask: Union[Union[str, int], List[Union[str, int]]]="default",
                     flux_column: Union[str, List[str]]="sap_flux",
                     cache_dir: Path=None) -> LightCurveCollection:
    """
    This is a wrapper for SearchResults.download_all() which allows for the quality_bitmask
    and flux_column to be varied across the lightcurve assets to be opened. Will load the
    lightcurves for the requested SearchResult through the requested local cache. Both
    quality_bitmask and flux_column can be set for all results (single value) or as separate
    values for each result item (as list or array). The cache_dir may be set to a specific
    location within which the MAST assets are cached, or left as None in which case the default
    lightkurve caching configuration is used.

    :results: the SearchResult to load
    :quality_bitmask: either a single value ("hardest", "hard", "default" or bitmask) or a list
    of these values (of the same length as results)
    :flux_column: either a single value ("sap_flux" or "pdcsap_flux") or a list of these values
    (of the same length as results)
    :cache_dir: Path to directory where we want the assets to be locally cached. If None then
    any caching will be under the control of the lightkurve config (see
    https://lightkurve.github.io/lightkurve/reference/config.html)
    """
    # Make sure the results & flags have same dimensions so we can iterate on them
    rcount = len(results)

    if isinstance(quality_bitmask, (str, int)):
        quality_bitmask = [quality_bitmask] * rcount
    elif len(quality_bitmask) != rcount:
        raise ValueError("The len(quality_bitmask) does not match len(results)")

    if isinstance(flux_column, str):
        flux_column = [flux_column] * rcount
    elif len(flux_column) != rcount:
        raise ValueError("The len(flux_column) does not match len(results)")

    # As lk doesn't like taking Path objects directly
    download_dir = f"{cache_dir}" if cache_dir else None

    # Now, load them all individually so we support varying quality_bitmask & flux_column values
    return LightCurveCollection(
        res.download(quality_bitmask=qbm, download_dir=download_dir, flux_column=fcol)
            for res, qbm, fcol in zip(results, quality_bitmask, flux_column)
    )


def create_invalid_flux_mask(lc: LightCurve) -> np.ndarray:
    """
    Will return a mask for the LightCurve indicating any fluxes with value of NaN or less than Zero.

    :lc: the LightCurve to mask
    :returns: the mask
    """
    if not isinstance(lc, LightCurve):
        raise TypeError(f"lc is {type(lc)}. Expected a LightCurve.")

    return ~((np.isnan(lc.flux)) | (lc.flux < 0))


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
    Will calculate a polynomial fit over the requested time range and y-data values. The fit is
    iterative; after each iteration the residuals are evaluated against a threshold defined by the
    StdDev of the residuals multiplied by res_sigma_clip; any datapoints with residuals greater
    than this are excluded from subsequent iterations.  This approach will exclude large y-data
    excursions, such as eclipses, from influencing the final fit.

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
