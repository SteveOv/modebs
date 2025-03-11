"""
Low level utility functions for light curve ingest, pre-processing, estimation and fitting.
"""
from typing import Union, List
from pathlib import Path

from lightkurve import LightCurveCollection, SearchResult


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
