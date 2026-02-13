"""
Low level utility functions for light curve ingest, pre-processing, estimation and fitting.
"""
# pylint: disable=no-member
from typing import Union, List, Iterable, Tuple, Generator
from pathlib import Path

import numpy as np
from scipy.signal import find_peaks

from uncertainties import unumpy, UFloat
import astropy.units as u
from astropy.time import Time, TimeDelta
from lightkurve import LightCurve, LightCurveCollection, FoldedLightCurve, SearchResult


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


def find_eclipses_and_completeness(lc: LightCurve,
                                   ref_t0: Union[Time, float, UFloat],
                                   period: Union[u.Quantity, float, UFloat],
                                   durp: Union[float, UFloat],
                                   durs: Union[float, UFloat],
                                   depthp: Union[float, UFloat]=None,
                                   depths: Union[float, UFloat]=None,
                                   phis: Union[float, UFloat]=0.5,
                                   search_window_phase: float=0.05,
                                   verbose: bool=False):
    """
    Will find the times of all primary and secondary eclipses within the bounds of the
    passed LightCurve. The eclipse timings will be refined by inspecting the LightCurve fluxes.
    For each eclipse a completeness metric will be calculated, which is the ratio of the number
    of fluxes recorded against the maximum possible based on the supplied eclipse duration.
    Will also return the time of the most complete primary eclipse as a refined reference time (t0).

    The func can use peak prominences to improve discrimination when locating the eclipses, which is
    especially useful if a LightCurve contains strong pulsation. This depends on the optional depthp
    and depths args which should be set to the expected eclipse depths in units of normalized flux.

    :lc: the LightCurve to inspect
    :ref_t0: the known reference primary eclipse time
    :period: the known orbital period
    :durp: the duration of the primary eclipses
    :durs: the duration of the secondary eclipses
    :depthp: the expected depth of the primary eclipse in units of normalized flux
    :depths: the expected depth of the secondary eclipse in units of normalized flux
    :phis: the phase of the secondary eclipses relative to the primary eclipses
    :search_window_phase: size of the window, in units of phase, within which to find each eclipse
    :verbose: whether or not to send messages to stdout with details the search
    :returns: a dict of ```{ "primary_eclipses": ndarray, "secondary_eclipses": ndarray,
    "primary_completeness": ndarray, "secondary_completenes": ndarray, "t0": float }```
    """
    def nominal_value(value):
        return value.nominal_value if isinstance(value, UFloat) else value

    ref_t0 = to_lc_time(ref_t0, lc).value if isinstance(ref_t0, Time) else nominal_value(ref_t0)
    period = period.to(u.d).value if isinstance(period, u.Quantity) else nominal_value(period)
    durp = nominal_value(durp)
    durs = nominal_value(durs)

    # Invert the fluxes so that eclipses are peaks
    times = lc.time.value
    fluxes = lc.normalize().flux.unmasked.value
    fluxes = fluxes.max() - fluxes
    half_window_dur = max(period * search_window_phase / 2, durp, durs)

    def find_eclipses_and_mask(ref_time, ecl_dur, min_ecl_prom) -> Tuple[np.ndarray, np.ndarray]:
        """
        Parse LC for an array of contained eclipse timings and corresponding completeness metrics
        """
        ecl_times, ecl_completeness = [], []
        half_ecl_dur = ecl_dur / 2
        ecl_exp_bins = np.ceil(ecl_dur * 86400 / lc.meta['FRAMETIM'] / lc.meta['NUM_FRM'])
        ecl_min_find_bins = ecl_exp_bins * 0.25     # For potential eclipses with find_peaks

        # Subtract extra period so that, whatever the offset, we're always before the sector start
        ecl_time = ref_time - period + (period * int((times.min()-ref_time) / period))
        while ecl_time < times.max() + half_window_dur:
            is_ecl_found = False
            window_mask = (ecl_time-half_window_dur <= times) & (times <= ecl_time+half_window_dur)

            if np.any(window_mask):
                window_fluxes = fluxes[window_mask]
                peaks, props = find_peaks(window_fluxes, width=ecl_min_find_bins,
                                          prominence=min_ecl_prom, wlen=ecl_exp_bins)
                if is_ecl_found := len(peaks) > 0:
                    # Currently unweighted, but if we find the incorrect peak is being picked in
                    # noisy/pulsator LCs, we can add weighting based on proximity to expected time.
                    peak_ix = np.argmax(props["prominences"])

                    # Found the eclipse, so we can refine its time and decide whether it's complete
                    ecl_time = times[window_mask][peaks[peak_ix]]
                    ecl_times += [ecl_time]
                    ecl_mask = (ecl_time-half_ecl_dur <= times) & (times <= ecl_time+half_ecl_dur)
                    ecl_completeness += [sum(ecl_mask) / ecl_exp_bins]

            if not is_ecl_found and times.min()-half_ecl_dur < ecl_time < times.max()+half_ecl_dur:
                ecl_times += [ecl_time]
                ecl_mask = (ecl_time-half_ecl_dur <= times) & (times <= ecl_time+half_ecl_dur)
                ecl_completeness += [sum(ecl_mask) / ecl_exp_bins]

            ecl_time += period
        return np.array(ecl_times), np.array(ecl_completeness)

    t0 = ref_t0
    min_promp = None if depthp is None else round(nominal_value(depthp) * 0.66, 3)
    min_proms = None if depths is None else round(nominal_value(depths) * 0.66, 3)
    if verbose:
        print(f"Finding (pri, sec) eclipses in sector {lc.sector} with min duration {durp:.3f} &",
              f"{durs:.3f} [d] and min depth {min_promp} & {min_proms} [norm flux]", end="...")
    for p in range(1, 3):
        pri_times, pri_compl = find_eclipses_and_mask(t0, durp, min_promp)

        t0 += period * nominal_value(phis)
        sec_times, sec_compl = find_eclipses_and_mask(t0, durs, min_proms)

        if verbose and p == 2:
            print(f"found {sum(pri_compl>0)} & {sum(sec_compl>0)} eclipse(s) with fluxes.")

        # Refine the reference time (completeness > 0.5 implies enough of a peak to find a time)
        if len(pri_compl) > 0 and any(pri_compl > 0.5):
            t0 = pri_times[np.argmax(pri_compl)]
        elif len(sec_times) > 0 and any(sec_compl > 0.5):
            t0 = sec_times[np.argmax(sec_compl)] - (period * nominal_value(phis))
        else:
            t0 = ref_t0

    return {
        "t0": t0,
        "primary_times": pri_times, "primary_completeness": pri_compl,
        "secondary_times": sec_times, "secondary_completeness": sec_compl
    }


def create_eclipse_mask(lc: LightCurve,
                        primary_times: np.ndarray[Time, float],
                        secondary_times: np.ndarray[Time, float],
                        durp: Union[u.Quantity[u.d], float],
                        durs: Union[u.Quantity[u.d], float],
                        verbose: bool=False) -> np.ndarray[bool]:
    """
    Create an eclipse mask for the passed LightCurve based on the accompanying
    known eclipse times and durations.

    :lc: the LightCurve to create the mask for
    :primary_times: the times of primary eclipses
    :secondary_times: the times of secondary eclipses
    :durp: the duration of the primary eclipses
    :durs: the duration of the secondary eclipses
    :returns: the eclipse mask
    """
    # pylint: disable=too-many-arguments, too-many-positional-arguments
    if verbose:
        print(f"Creating an eclipse mask for {len(primary_times)} primary and",
              f"{len(secondary_times)} secondary eclipse(s).")

    times = lc.time.value
    mask = np.zeros((len(lc)), dtype=bool)
    for ecl_times, ecl_dur in [(primary_times, durp), (secondary_times, durs)]:
        if len(ecl_times) > 0:
            for ecl_time in to_lc_time(ecl_times, lc).value:
                mask |= (ecl_time - ecl_dur/2 <= times) & (times <= ecl_time + ecl_dur/2)
    return mask



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
