"""
Low level utility functions for light curve ingest, pre-processing, estimation and fitting.
"""
# pylint: disable=no-member
from typing import Union, List, Iterable, Tuple, Generator
from pathlib import Path
from warnings import warn

import numpy as np
from scipy.signal import find_peaks

from uncertainties import unumpy
import astropy.units as u
from astropy.time import Time, TimeDelta
from lightkurve import LightCurve, LightCurveCollection, FoldedLightCurve, SearchResult
from deblib import orbital


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

    :returns: a LightCurveCollection of the downloaded lightcurves, ordered by sector
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
    lcs = [res.download(quality_bitmask=qbm, download_dir=download_dir, flux_column=fcol)
            for res, qbm, fcol in zip(results, quality_bitmask, flux_column)]
    return LightCurveCollection(sorted(lcs, key=lambda lc: lc.sector))


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
                             yield_times: bool=False) \
                                -> Generator[Union[slice, Tuple[Time, Time]], any, None]:
    """
    Finds the start and end of contiguous segments in the passed LightCurve. These are contiguous
    regions where the gaps between bins does not exceed the passed threshold. Gaps greater then the
    threshold are treated as boundaries between segments. If no gaps found this will yield a single
    segment for the whole LightCurve.

    :lc: the source LightCurve to parse for gaps/segments.
    :threshold: the threshold gap time beyond which a segment break is triggered
    :yield_times: if true start/end times will be yielded, otherwise slices
    :returns: generator of slice(start, end, 1) or tuple(start Time, end Time) if yield_times==True
    """
    if isinstance(threshold, TimeDelta):
        pass
    elif isinstance(threshold, u.Quantity):
        threshold = TimeDelta(threshold)
    else:
        threshold = TimeDelta(threshold * u.d)


    # Much quicker if we use primatives - make sure we work in days
    threshold = threshold.to(u.d).value
    times = lc.time.value

    last_ix = len(lc) - 1
    segment_start_ix = 0
    for this_ix, previous_time in enumerate(times, start = 1):
        if this_ix > last_ix or times[this_ix] - previous_time > threshold:
            if yield_times:
                yield (lc.time[segment_start_ix], lc.time[this_ix - 1])
            else:
                yield slice(segment_start_ix, this_ix, 1)
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


def create_eclipse_mask(lc: LightCurve,
                        t0: Union[Time, float],
                        period: Union[u.Quantity[u.d], float],
                        dur_pri: Union[u.Quantity[u.d], float],
                        dur_sec: Union[u.Quantity[u.d], float],
                        phi_sec: float=0.5,
                        dfactor: float=1.0,
                        verbose: bool=False) -> np.ndarray[bool]:
    """
    Create an eclipse mask for the passed LightCurve based on the accompanying
    extended ephemeris values.

    :lc: the LightCurve to create the mask for
    :t0: the reference time of a primary eclipse
    :period: the orbital period (assumed to be in days if not a Quantity)
    :dur_pri: the duration of the primary eclipses
    :dur_sec: the duration of the secondary eclipses
    :phi_sec: the phase of secondary eclipses (assuming the primary eclipses are at phase 0)
    :dfactor: an expansion factor applied to the eclipse lengths
    :returns: the eclipse mask
    """
    # pylint: disable=too-many-arguments, too-many-positional-arguments
    if verbose:
        print(f"Creating an eclipse mask for t0={t0}, period={period},",
              f"dur_pri={dur_pri:.6f}, dur_sec={dur_sec:.6f} & phi_sec={phi_sec:.6f}")

    eclipse_times = [to_lc_time(t0, lc), to_lc_time(t0 + period * phi_sec, lc)]
    durations = [dur_pri * dfactor, dur_sec * dfactor]
    return lc.create_transit_mask([period] * 2, eclipse_times, durations)


def create_eclipse_mask_from_fitted_params(lc: LightCurve,
                                           t0: Union[Time, float],
                                           period: Union[u.Quantity, float],
                                           sum_r: float,
                                           inc: float,
                                           ecosw: float,
                                           esinw: float,
                                           dfactor: float=1.0,
                                           verbose:bool=False) -> np.ndarray[bool]:
    """
    Create an eclipse mask for the passed LightCurve based on the accompanying
    basic ephemeris information and fitted light curve parameters.

    :lc: the LightCurve to create the mask for
    :t0: reference time of a primary eclipse
    :period: the orbital period (assumed to be in days if not a Quantity)
    :sum_r: the sum of the fractional radii (rA + rB)
    :inc: the inclination in degree
    :ecosw: the e*cos(omega) Poincare element
    :esinw: the e*sin(omega) Poincare element
    :dfactor: an expansion factor applied to the eclipse lengths
    :returns: the eclipse mask
    """
    # pylint: disable=too-many-arguments, too-many-positional-arguments
    period_d = period.to(u.d).value if isinstance(period, u.Quantity) else period
    e = (ecosw**2 + esinw**2)**0.5
    return create_eclipse_mask(lc, t0, period, dfactor=dfactor, verbose=verbose,
        dur_pri=orbital.eclipse_duration(period_d, sum_r, inc, e, esinw, False),
        dur_sec=orbital.eclipse_duration(period_d, sum_r, inc, e, esinw, True),
        phi_sec=orbital.phase_of_secondary_eclipse(ecosw, e))


def get_lightcurve_t0_time(lc: LightCurve,
                           t0: Union[Time, float],
                           period: Union[u.Quantity, float],
                           max_phase_shift: float=0.1) -> float:
    """
    Will find the time of the first eclipse, equivalent to the reference time,
    within the passed LightCurve. This can handle moderate shifts, up to
    max_phase_shift, in eclipse timing compared with the reference ephemeris.

    :lc: the LightCurve to inspect
    :t0: the known reference time of an eclipse
    :period: the known orbital period (assumed to be in days if not a Quantity)
    :max_phase_shift: the maximum allowed positive or negative phase shift - increasing this will
    increase the likelihood of incorrectly selecting an instance of the "other" eclipse type 
    :returns: the time of the first eclipse found, in the frame/scale of the lightcurve
    """
    # Get these into the format/scale of the light-curve
    if isinstance(t0, Time):
        t0 = to_lc_time(t0, lc).value
    if isinstance(period, u.Quantity):
        period = period.to(u.d).value

    found_ecl_time = None
    times = lc.time.value

    # These are the expected t0 timings projected into this lightcurve sector.
    cycles_offset = int(np.ceil((times.min() - t0) / period))
    cycles_in_lc = int(np.ceil((times.max() - times.min()) / period))
    exp_ecl_times = [t0 + p
                        for p in (period * (cycles_offset + c) for c in range(cycles_in_lc))
                            if times.min() <= (t0 + p) <= times.max()]

    # Cannot use a periodogram as there may be too few eclipses to derive a period. Instead
    # we're relying on finding the most prominent dip nearest the expected eclipse timings.
    # As there may be gaps in the LC at these times, we go through until a shifted eclipse is found.
    for exp_ecl_time in exp_ecl_times:
        half_window = period * max_phase_shift
        mask = (exp_ecl_time - half_window < times) & (times < exp_ecl_time + half_window)
        if np.any(mask):
            # Invert the fluxes so we get peaks rather than dips
            window_fluxes = lc.flux[mask].unmasked.value
            window_fluxes = window_fluxes.max() - window_fluxes

            peaks, _ = find_peaks(window_fluxes, width=5, distance=100)
            if len(peaks) > 0:
                highest_peak_ix = peaks[np.argmax(window_fluxes[peaks])]
                found_ecl_time = times[mask][highest_peak_ix]
                break

    if found_ecl_time is None:
        warn(f"No primary eclipses found in {lc.meta['LABEL']}. Estimating t0 from period & cycles")
        found_ecl_time = exp_ecl_times[0]

    return found_ecl_time


def to_lc_time(value: Union[Time, np.double, Tuple[np.double], List[np.double,]], lc: LightCurve) \
                -> Time:
    """
    Converts the passed numeric value to an astropy Time. The magnitude of the time will be used to
    interpret the format and scale to match LC (<4e4: btjd, else modifed jd, except >2.4e6: jd).

    :value: the value or values to be converted
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

    if value < 2.4e6:
        value += 2.4e6
    return Time(value, format="jd", scale=lc.time.scale)


def get_binned_phase_mags_data(flc: FoldedLightCurve,
                               num_bins: int = 1024,
                               phase_pivot: Union[u.Quantity, float]=None) \
                                    -> Tuple[np.ndarray, np.ndarray]:
    """
    Get a binned copy of the phase and delta_mags (nominal values) from the passed FoldedLightCurve.

    :flc: the source FoldedLightCurve
    :num_bins: the number of equally spaced bins to return
    :phase_pivot: the pivot point about which the fold phase was wrapped to < 0; inferred if omitted
    :returns: a tuple with requested number of binned phases and delta magnitudes
    """
    # Can't use lightkurve's bin() on a FoldedLightCurve: unhappy with phase Quantity as time col.
    # By using unumpy/ufloats we're aware of the mags' errors in the mean calculation for each bin.
    src_phase = flc.phase.value
    src_mags = unumpy.uarray(flc["delta_mag"].value, flc["delta_mag_err"].value).unmasked

    # If there is a phase wrap then phases above the pivot will have been
    # wrapped around to <0. Work out what the expected minimum phase should be.
    if phase_pivot is not None:
        max_phase = phase_pivot.value if isinstance(phase_pivot, u.Quantity) else phase_pivot
    else:
        max_phase = src_phase.max()
    max_phase = max(max_phase, src_phase.max())
    min_phase = min(max_phase - 1, src_phase.min())

    # Because we will likely know the exact max phase but the min will be infered we make sure the
    # phases end at the pivot/max_phase but start just "above" the expected min phase (logically
    # equiv to startpoint=False). Working with the searchsorted side="left" arg, which allocates
    # indices where bin_phase[i-1] < src_phase <= bin_phase[i], we map all source data to a bin.
    bin_phase = np.flip(np.linspace(max_phase, min_phase, num_bins, endpoint=False))
    phase_bin_ix = np.searchsorted(bin_phase, src_phase, "left")

    # Perform the "mean" binning
    bin_mags = np.empty_like(bin_phase, dtype=float)
    for bin_ix in range(num_bins):
        phase_ix = np.where(phase_bin_ix == bin_ix)[0] # np.where() indices are quicker than masking
        if len(phase_ix) > 0:
            bin_mags[bin_ix] = src_mags[phase_ix].mean().n
        else:
            bin_mags[bin_ix] = np.nan

    # Fill any gaps by interpolation; we have a np.nan where there were no source data within a bin
    if any(missing := np.isnan(bin_mags)):
        def equiv_ix(ix):
            return ix.nonzero()[0]
        bin_mags[missing] = np.interp(equiv_ix(missing), equiv_ix(~missing), bin_mags[~missing])

    return (bin_phase, bin_mags)
