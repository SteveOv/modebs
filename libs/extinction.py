"""
Low level utility functions for light curve ingest, pre-processing, estimation and fitting.
"""
#pylint: disable=no-member
from typing import Tuple, List, Callable, Generator
import inspect
from functools import lru_cache

from requests.exceptions import HTTPError
import numpy as np
import astropy.units as u
from astropy.coordinates import SkyCoord
from astroquery.vizier import Vizier

from dustmaps import config, bayestar           # Bayestar dustmaps/dereddening map
from pyvo import registry, DALServiceError      # Vergeley at al. extinction catalogue


def get_ebv(target_coords: SkyCoord,
            funcs: List[str]=None,
            rv: float=3.1) -> Generator[Tuple[float, dict], any, any]:
    """
    A convenience function which iterates through the requested extinction lookup functions,
    published on this module, yielding the E(B-V) extinction value and flags returned by each.

    If no funcs specified the following list will be used, in the order shown:
    [get_bayestar_ebv, get_vergely_av, get_gontcharov_ebv]

    :target_coords: the SkyCoords to get the extinction value for
    :funcs: optional list of functions to iterate over, either by name of function object.
    These must be callable as func(coords: SkyCoord) -> (value: float, flags: Dict)
    :rv: the R_V value to use if it is necessary to convert Av values to E(B-V)
    """
    if funcs is None:
        funcs = [get_bayestar_ebv, get_vergely_av, get_gontcharov_ebv]
    if isinstance(funcs, str | Callable):
        funcs = [funcs]

    for ext_func in funcs:
        if isinstance(ext_func, str):
            # Find the matching function in this module
            # TODO: can this be more efficient? Also perhaps better validation of func signature
            for name, func in inspect.getmembers(inspect.getmodule(get_ebv),
                                                 lambda m: isinstance(m, Callable)):
                if ext_func in name:
                    ext_func = func
                    break

        if isinstance(ext_func, Callable):
            val, flags = ext_func(target_coords)
            if val is not None and not np.isnan(val):
                if flags.get("type", "").lower() == "av" \
                    or ext_func.__name__.lower().endswith("_av"):
                    val /= rv
                flags["source"] = ext_func.__name__
                yield val, flags


def get_av(target_coords: SkyCoord,
           funcs: List[str]=None,
           rv: float=3.1) -> Generator[Tuple[float, dict], any, any]:
    """
    A convenience function which iterates through the requested extinction lookup functions,
    published on this module, yielding the A_V extinction value and flags returned by each.

    If no funcs specified the following list will be used, in the order shown:
    [get_bayestar_ebv, get_vergely_av, get_gontcharov_ebv]

    :target_coords: the SkyCoords to get the extinction value for
    :funcs: optional list of functions to iterate over, either by name of function object.
    These must be callable as func(coords: SkyCoord) -> (value: float, flags: Dict)
    :rv: the R_V value to use if it is necessary to convert E(B-V) values to A_V
    """
    if funcs is None:
        funcs = [get_bayestar_ebv, get_vergely_av, get_gontcharov_ebv]
    if isinstance(funcs, str | Callable):
        funcs = [funcs]

    for ext_func in funcs:
        if isinstance(ext_func, str):
            # Find the matching function in this module
            # TODO: can this be more efficient? Also perhaps better validation of func signature
            for name, func in inspect.getmembers(inspect.getmodule(get_av),
                                                 lambda m: isinstance(m, Callable)):
                if ext_func in name:
                    ext_func = func
                    break

        if isinstance(ext_func, Callable):
            val, flags = ext_func(target_coords)
            if val is not None and not np.isnan(val):
                if flags.get("type", "").lower() == "E(B-V)" \
                    or ext_func.__name__.lower().endswith("_ebv"):
                    val *= rv
                flags["source"] = ext_func.__name__
                yield val, flags


def get_bayestar_ebv(target_coords: SkyCoord,
                     version: str="bayestar2019",
                     conversion_factor: float=0.996) -> Tuple[float, dict]:
    """
    Queries the Bayestar dereddening map for the E(B-V) value for the target coordinates.

    Conversion from Bayestar 17 or 19 to E(B-V) documented at http://argonaut.skymaps.info/usage
    as E(B-V) = 0.884 x bayestar or E(B-V) = 0.996 x bayestar
    
    :target_coords: the astropy SkyCoords to query for
    :version: the version of the Bayestar dust maps to use
    :conversion_factor: the factor to apply to bayestar extinction for E(B-V)
    :returns: tuple of the E(B-V) value and a dict of the diagnostic flags associated with the query
    """
    print(f"Querying the Green+ (2019ApJ...887...93G) {version} map for extinction data.")
    query = _get_bayestar_query(version)
    val, flags =  query(target_coords, mode='median', return_flags=True)
    flags_dict = { n: flags[n] for n in flags.dtype.names }
    flags_dict["type"] = "E(B-V)"
    return conversion_factor * val, flags_dict

@lru_cache
def _get_bayestar_query(version: str):
    """ Gets a Bayestar query object. This function is cached as it's an expensive setup. """
    try:
        # Creates/confirms local cache of Bayestar data within the .cache directory
        config.config['data_dir'] = '.cache/.dustmapsrc'
        bayestar.fetch(version=version)
    except HTTPError as exc:
        print(f"Unable to (re)fetch data for {version}. Caught error '{exc}'")
    except ValueError as exc:
        print(f"Unable to parse response for {version}. Caught error '{exc}'")

    # Now we can use the local cache for the lookup - this takes some time to set up
    return bayestar.BayestarQuery(version=version)


def get_gontcharov_ebv(target_coords: SkyCoord,
                       conversion_factor: float=1.7033):
    """
    Queries the Gontcharov (2017) [2017AstL...43..472G] 3-d extinction service
    for the Ebv value of the target coordinates.

    Extends radially to at least 700 pc, in galactic coords at 20 pc distance intervals
    Documented conversion factor for E(B-V) = 1.7033 E(J-K)

    :target_coords: the astropy SkyCoords to query for   
    :conversion_factor: the factor to apply to E(J-Ks) for E(B-V)
    :returns: tuple of the E(B-V) value and a dict of the diagnostic flags associated with the query
    """
    print("Querying the Gontcharov (2017AstL...43..472G) 2MASS photometry map for extinction data.")
    ebv = 0
    flags = { "converged": False } # mimics Bayestar - will set true if we get a good match
    vizier = Vizier(catalog="J/PAZh/43/521/rlbejk", columns=["**"], row_limit=-1)

    # Round up the galactic coords (to nearest deg) and distance (to nearest 20 pc)
    glon, glat = np.ceil(target_coords.galactic.l.deg), np.ceil(target_coords.galactic.b.deg)
    dist = np.ceil(target_coords.distance.to(u.pc).value / 20) * 20

    for r, dflag in [(dist, True), (700, False), (600, False)]:
        if _tbl := vizier.query_constraints(R=r, GLON=glon, GLAT=glat, cache=True):
            if len(_tbl):
                ebv = _tbl[0]["E(J-Ks)"][0] * conversion_factor
                flags["converged"] = dflag
                flags["type"] = "E(B-V)"
                break

    return ebv, flags


def get_vergely_av(target_coords: SkyCoord):
    """
    Queries the Vergely, Lallement & Cox (2022) [2022A&A...664A.174V] 3-d extinction map
    for the Av value of the target coordinates.

    TODO: this needs further work

    :target_coords: the astropy SkyCoords to query for   
    :returns: the Av value
    """
    av = None
    flags = { "converged": False } # mimics Bayestar - will set true if we get a good match
    try:
        # Extinction map of Vergely, Lallement & Cox (2022)
        ivoid = 'ivo://CDS.VizieR/J/A+A/664/A174'
        table = 'J/A+A/664/A174/cube_ext'
        vo_res = registry.search(ivoid=ivoid)[0]
        print(f"Querying Vergely+ ({vo_res.source_value}) {vo_res.res_title} for extinction data.")

        for res in [10, 50]: # central regions at 10 pc resolution and outer at 50 pc
            cart = np.ceil(target_coords.cartesian.xyz.to(u.pc) / res) * res
            rec = vo_res.get_service('tap').search(f'SELECT * FROM "{table}" ' +
                f'WHERE x={cart[0].value:.0f} AND y={cart[1].value:.0f} AND z={cart[2].value:.0f}')

            if len(rec):
                # TODO: anything extra to map these values to Av?
                ext_nmag_per_pc = rec["Exti"][0] # nmag
                av = (ext_nmag_per_pc * target_coords.distance.to(u.pc).value) / 10**9
                flags['converged'] = True
                flags["type"] = "Av"
                break
    except DALServiceError as exc:
        print(f"Failed to query: {exc}")
    return av, flags
