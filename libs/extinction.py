"""
Low level utility functions for light curve ingest, pre-processing, estimation and fitting.
"""
#pylint: disable=no-member
from typing import Tuple, List, Callable, Generator
import inspect
from functools import lru_cache
import traceback

from requests.exceptions import HTTPError
import numpy as np
from scipy.interpolate import RBFInterpolator
import astropy.units as u
from astropy.coordinates import SkyCoord
from astroquery.vizier import Vizier

from dustmaps import config, bayestar           # Bayestar dustmaps/dereddening map
from pyvo import registry, DALServiceError      # Vergeley at al. extinction catalogue


def get_av(target_coords: SkyCoord,
           funcs: List[str]=None,
           rv: float=3.1,
           verbose: bool=False) -> Generator[Tuple[float, dict], any, any]:
    """
    A convenience function which iterates through the requested extinction lookup functions,
    published on this module, yielding the A_V extinction value and flags returned by each.

    If no funcs specified the following list will be used, in the order shown:
    [get_gontcharov_av, get_bayestar_ebv]

    :target_coords: the SkyCoords to get the extinction value for
    :funcs: optional list of functions to iterate over, either by name of function object.
    These must be callable as func(coords: SkyCoord) -> (value: float, flags: Dict)
    :rv: the R_V value to use if it is necessary to convert Av values to E(B-V)
    :verbose: whether or not to print progress/diagnostics to stdout
    """
    for val, flags in _get_extinction(target_coords, funcs, verbose):
        source = flags.get("source", "")
        if flags.get("type", "").lower() == "e(b-v)" or source.lower().endswith("_ebv"):
            val *= rv
            flags["type"] = "A_V"
        if verbose:
            print(f"{source}: A_V = {val:.6f} [converged={flags.get('converged', False)}]")
        yield val, flags


def get_ebv(target_coords: SkyCoord,
            funcs: List[str]=None,
            rv: float=3.1,
            verbose: bool=False) -> Generator[Tuple[float, dict], any, any]:
    """
    A convenience function which iterates through the requested extinction lookup functions,
    published on this module, yielding the E(B-V) extinction value and flags returned by each.

    If no funcs specified the following list will be used, in the order shown:
    [get_gontcharov_av, get_bayestar_ebv]

    :target_coords: the SkyCoords to get the extinction value for
    :funcs: optional list of functions to iterate over, either by name of function object.
    These must be callable as func(coords: SkyCoord) -> (value: float, flags: Dict)
    :rv: the R_V value to use if it is necessary to convert E(B-V) values to A_V
    :verbose: whether or not to print progress/diagnostics to stdout
    """
    for val, flags in _get_extinction(target_coords, funcs, verbose):
        source = flags.get("source", "")
        if flags.get("type", "").lower() == "av" or source.lower().endswith("_av"):
            val /= rv
            flags["type"] = "E(B-V)"
        if verbose:
            print(f"{source}: E(B-V) = {val:.6f} [converged={flags.get('converged', False)}]")
        yield val, flags


def _get_extinction(target_coords: SkyCoord,
                    funcs: List[str]=None,
                    verbose: bool=False) -> Generator[Tuple[float, dict], any, any]:
    """
    Iterates through calls to the requested extinction lookup functions,
    published on this module, yielding the value and flags returned by each.

    If no funcs specified the following list will be used, in the order shown:
    [get_gontcharov_av, get_bayestar_ebv]

    :target_coords: the SkyCoords to get the extinction value for
    :funcs: optional list of functions to iterate over, either by name of function object.
    :verbose: whether or not to print progress/diagnostics to stdout
    """
    if funcs is None:
        funcs = [get_gontcharov_av, get_bayestar_ebv] #, get_vergely_av]
    if isinstance(funcs, str | Callable):
        funcs = [funcs]

    for func in funcs:
        if isinstance(func, str):
            # Find the matching function in this module
            # TODO: can this be more efficient? Also perhaps better validation of func signature
            for name, member_func in inspect.getmembers(inspect.getmodule(get_av),
                                                        lambda m: isinstance(m, Callable)):
                if func in name:
                    func = member_func
                    break

        if isinstance(func, Callable):
            fname = func.__name__
            for attempt in range(2):
                try:
                    val, flags = func(target_coords)
                    if val is not None and not np.isnan(val):
                        flags["source"] = fname
                        yield val, flags
                    elif verbose:
                        print(f"{fname}: None")
                    break

                except Exception as exc: # pylint: disable=broad-exception-caught
                    if not isinstance(exc, HTTPError) or attempt > 0:
                        if verbose:
                            print(f"Caught a {type(exc).__name__} when calling {fname}. Moving on.")
                        traceback.print_exception(exc)
                        break
                    if verbose:
                        print(f"Caught a {type(exc).__name__} when calling {fname}. Trying again.")


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
    query = _get_bayestar_query(version)
    val, flags =  query(target_coords, mode='median', return_flags=True)
    flags_dict = { n: flags[n] for n in flags.dtype.names }
    flags_dict["type"] = "E(B-V)"
    return conversion_factor * val, flags_dict

@lru_cache
def _get_bayestar_query(version: str) -> bayestar.BayestarQuery:
    """ Gets a Bayestar query object. This function is cached as it's an expensive setup. """
    # Creates/confirms local cache of Bayestar data within the .cache directory
    config.config['data_dir'] = '.cache/.dustmapsrc'
    bayestar.fetch(version=version)

    # Now we can use the local cache for the lookup - this takes some time to set up
    return bayestar.BayestarQuery(version=version)


def get_gontcharov_av(target_coords: SkyCoord):
    """
    Queries the Gontcharov (2017) [2017AstL...43..472G] 3-d extinction map
    for the A_V value of the target coordinates.

    Uses a locally cached X, Y, Z table (J/PAZh/43/521/xyzejk), which includes values for Av, E(B-V)
    and Rv unlike the radial table which only has values for E(J-Ks). The X, Y, Z table covers the
    region (-1200 <= X <= 1200, -1200 <= Y <= 1200, -600 <= Z <= 600).

    :target_coords: the astropy SkyCoords to query for   
    :returns: tuple of the A_V value and a dict of the diagnostic flags associated with the query
    """
    ret_val, flags = None, { }
    interp = _get_gontcharov_interp("Av")
    gal_xyz_coords = target_coords.transform_to("galactic").cartesian.xyz.value

    # Check the map covers the target
    points = interp.y
    if all(points[..., d].min() < gal_xyz_coords[d] < points[..., d].max() for d in range(3)):
        ret_val = interp(np.expand_dims(gal_xyz_coords, axis=0))[0]
        flags["converged"] = True
        flags["type"] = "A_V"
    return ret_val, flags

@lru_cache
def _get_gontcharov_interp(interp_field: str="Av"):
    """
    Gets an interpolator for the Gontcharov extinction map (J/PAZh/43/521/xyzejk) with which to
    interp either Av or E(B-V) data from galactic (X, Y, Z) coordinates.
    """
    old_row_limit = Vizier.ROW_LIMIT
    Vizier.ROW_LIMIT = -1
    cat = Vizier.get_catalogs(["J/PAZh/43/521/xyzejk"])[0]
    Vizier.ROW_LIMIT = old_row_limit

    # Can't get this into a RegularGridInterpolator I've been unable to get the data sorted
    # in a way that makes it happy. However RBFs are nice and flexible.
    points = np.array(list(zip(cat["X"].value, cat["Y"].value, cat["Z"].value)), dtype=float)
    return RBFInterpolator(points, cat[interp_field].value, neighbors=3**points.shape[1])


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

        for res in [10, 50]: # central regions at 10 pc resolution and outer at 50 pc
            cart = np.ceil(target_coords.cartesian.xyz.to(u.pc) / res) * res
            rec = vo_res.get_service('tap').search(f'SELECT * FROM "{table}" ' +
                f'WHERE x={cart[0].value:.0f} AND y={cart[1].value:.0f} AND z={cart[2].value:.0f}')

            if len(rec):
                # TODO: anything extra to map these values to Av?
                ext_nmag_per_pc = rec["Exti"][0] # nmag
                av = (ext_nmag_per_pc * target_coords.distance.to(u.pc).value) / 10**9
                flags['converged'] = True
                flags["type"] = "A_V"
                break
    except DALServiceError as exc:
        print(f"Failed to query: {exc}")
    return av, flags
