"""
Low level utility functions for light curve ingest, pre-processing, estimation and fitting.
"""
#pylint: disable=no-member
from typing import Tuple, List, Callable, Generator
from inspect import getmodule, getmembers, isfunction
from functools import lru_cache
import traceback
from warnings import warn

from requests.exceptions import HTTPError
import numpy as np
from scipy.interpolate import RBFInterpolator
import astropy.units as u
from astropy.coordinates import SkyCoord, Galactic
from astroquery.vizier import Vizier

from dustmaps import config, bayestar, decaps, edenhofer2023    # Bayestar and other exinction maps
from pyvo import registry, DALServiceError                      # Vergeley at al. extinction catalogue

# Parent dir for cached dustmaps files
config.config["data_dir"] = ".cache/.dustmapsrc"

def iterate(target_coords: SkyCoord,
            funcs: List[str]=None,
            rv: float=3.1,
            yield_ebv: bool=False,
            verbose: bool=False) -> Generator[Tuple[float, bool], None, None]:
    """
    Iterates through calls to the requested extinction lookup functions, published on this
    module, yielding a coefficient and reliability flag for each where a value available.

    If no funcs specified the following list will be used, in the order shown:
    [get_gontcharov_av, get_edenhofer2023_av, get_bayestar_ebv]

    :target_coords: the SkyCoords to get the extinction value for
    :funcs: optional list of functions to iterate over, either by name of function object.
    These must be callable as func(coords: SkyCoord) -> (value: float, flags: Dict)
    :rv: the R_V value to use if it is necessary to convert between Av and E(B-V) values
    :yield_ebv: whether to yield E(B-V) (True) or A_V (False) values
    :verbose: whether or not to print progress/diagnostics to stdout
    :returns: Generator yielding the chosen value (when found) and a flag indicating its reliability
    """
    if funcs is None:
        funcs = [get_gontcharov_av, get_edenhofer2023_av, get_bayestar_ebv] #, get_vergely_av]
    if isinstance(funcs, str | Callable):
        funcs = [funcs]

    for func in funcs:
        if isinstance(func, str):
            # Find the matching function in this module
            for name, member_func in getmembers(getmodule(iterate), isfunction):
                if name.lower().startswith("get_") and func in name:
                    func = member_func
                    break

        if isinstance(func, Callable):
            fname = func.__name__
            for attempt in range(2):
                try:
                    val, reliable = func(target_coords)
                    if val is not None and not np.isnan(val):
                        if yield_ebv and fname.lower().endswith("_av"):
                            val /= rv
                        elif not yield_ebv and fname.lower().endswith("_ebv"):
                            val *= rv
                        if verbose:
                            print(f"{fname}:", "E(B-V)" if yield_ebv else "A_V", f"= {val:.6f}",
                                  "(reliable)" if reliable else "")
                        yield val, reliable
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
        else:
            warn(f"Ignoring unknown extinction func: {func}", UserWarning)


def get_bayestar_ebv(target_coords: SkyCoord,
                     version: str="bayestar2019",
                     conversion_factor: float=0.884) -> Tuple[float, bool]:
    """
    Queries the Bayestar dereddening map for the E(B-V) value for the target coordinates.

    Conversion from Bayestar 17 or 19 to E(B-V) documented at http://argonaut.skymaps.info/usage
    as E(B-V) = 0.884 x bayestar (from E(g-r)) or E(B-V) = 0.996 x bayestar (from E(r-z)).
    This covers approximately 75% of the sky north of dec -30 deg.
    
    :target_coords: the astropy SkyCoords to query for
    :version: the version of the Bayestar dust maps to use
    :conversion_factor: the factor to apply to bayestar extinction for E(B-V)
    :returns: tuple of the E(B-V) value and a flags indicating whether it is reliable
    """
    val, flags =  _get_bayestar_query(version)(target_coords, mode="best", return_flags=True)
    return conversion_factor * val, flags["reliable_dist"]

@lru_cache
def _get_bayestar_query(version: str) -> bayestar.BayestarQuery:
    """ Gets a Bayestar query object. This function is cached as it's an expensive setup. """
    # Creates/confirms local cache of Bayestar data within the .cache directory
    # Now we can use the local cache for the lookup - this takes some time to set up
    print(f"Setting up query object for the {version.capitalize()} extinction map")
    bayestar.fetch(version=version)
    return bayestar.BayestarQuery(version=version)


def get_decaps_av(target_coords: SkyCoord, rv: float=3.32) -> Tuple[float, bool]:
    """
    Queries the DECaPS dereddening map for the E(B-V) value for the target coordinates.
    This complements the Bayestar maps, being concentrated in the southern hemisphere
    suitable for when the Galactic dec is south of -30 deg.

    Heavy on resources (RAM & disk) and doesn't give good coverage of LOPS2.

    :target_coords: the astropy SkyCoords to query for
    :rv: the R_V value for the map, with the dustmaps docs recommending 3.32
    :returns: tuple of the Av value and a flags indicating whether it is reliable
    """
    # Cannot use both mean_only and contiguous, as doing so appears to always cause an IndexError.
    # mean_only has lower memory usage & doesn't require as large a download (8 instead of 33 GB!).
    query = _get_decaps_query(mean_only=True, contiguous=False)
    val, flags =  query(target_coords, mode="mean", return_flags=True)
    return rv * val, flags["reliable_dist"]

@lru_cache
def _get_decaps_query(mean_only: bool, contiguous: bool) -> decaps.DECaPSQueryLite:
    """ Gets a Bayestar query object. This function is cached as it's an expensive setup. """
    # Creates/confirms local cache of DECaPS data within the .cache directory
    # silence_warnings prevents the fetch from asking user to confirm download
    # Now we can use the local cache for the lookup - this takes some time to set up
    print("Setting up query object for the DECaPS extinction map")
    decaps.fetch(mean_only=mean_only, silence_warnings=True)
    return decaps.DECaPSQueryLite(mean_only=mean_only, contiguous=contiguous)


def get_edenhofer2023_av(target_coords: SkyCoord,
                         flavor: str="main",
                         mode: str="mean") -> Tuple[float, bool]:
    """
    Queries the Edenhofer et al. (2024A&A...685A..82E) 3D dustmap. This is based on the the
    measurements of Zhang, Green & Rix (2023MNRAS.524.1855Z) and publishes extinction in
    their units of E upon which we use the conversion: A_V = 2.8 * E (from 2023MNRAS.524.1855Z).
   
    See https://dustmaps.readthedocs.io/en/latest/modules.html#module-dustmaps.edenhofer2023
    for more detail on options.

    :target_coords: the astropy SkyCoords to query for
    :flavor: the map, either "main" (69-1250 pc) or less_data_but_2kpc (69-2000 pc at lower res)
    :mode: dictates returned values; "mean", "std", "samples" or "random_sample"
    :returns: tuple of the Av value and a flags indicating whether it is reliable
    """
    # We use integrated=True to get extinction density in the ZGR E values, where A_V=2.8*E
    query = _get_edenhofer2023_query(flavor=flavor,
                                     fetch_samples="mean" not in mode,
                                     integrated=True)
    val = query(target_coords, mode=mode)
    in_range = query.distance_bounds.min() < target_coords.distance < query.distance_bounds.max()
    if val is None or np.isnan(val):
        return None, False
    return 2.8 * val, in_range

@lru_cache
def _get_edenhofer2023_query(flavor: str, fetch_samples: bool, integrated: bool) \
                                                            -> edenhofer2023.Edenhofer2023Query:
    """ Gets an Edenhofer2023Query query object. Function cached as may be an expensive setup. """
    # Create/confirm the presence of the dataset in the local cache.
    # Then read the local cache for the lookup. This takes some time to set up if integrated==True.
    print("Setting up query object for the Edenhofer2023 extinction map")
    edenhofer2023.fetch(fetch_samples=fetch_samples, fetch_2kpc="2kpc" in flavor)
    return edenhofer2023.Edenhofer2023Query(load_samples=fetch_samples, integrated=integrated,
                                            flavor=flavor, seed=42)


def get_gontcharov_av(target_coords: SkyCoord) -> Tuple[float, bool]:
    """
    Queries the Gontcharov (2017) [2017AstL...43..472G] 3-d extinction map
    for the A_V value of the target coordinates.

    Uses a locally cached X, Y, Z table (J/PAZh/43/521/xyzejk), which includes values for Av, E(B-V)
    and Rv unlike the radial table which only has values for E(J-Ks). The X, Y, Z table covers the
    region (-1200 <= X <= 1200, -1200 <= Y <= 1200, -600 <= Z <= 600).

    :target_coords: the astropy SkyCoords to query for   
    :returns: tuple of the A_V value and a dict of the diagnostic flags associated with the query
    """
    ret_val, reliable = None, False
    interp = _get_gontcharov_interp("Av")
    gal_xyz_coords = target_coords.transform_to(Galactic).cartesian.xyz.value

    # Check the map covers the target
    points = interp.y
    if all(points[..., d].min() < gal_xyz_coords[d] < points[..., d].max() for d in range(3)):
        ret_val = interp(np.expand_dims(gal_xyz_coords, axis=0))[0]
        reliable = True
    return ret_val, reliable

@lru_cache
def _get_gontcharov_interp(interp_field: str="Av"):
    """
    Gets an interpolator for the Gontcharov extinction map (J/PAZh/43/521/xyzejk) with which to
    interp either Av or E(B-V) data from galactic (X, Y, Z) coordinates.
    """
    print("Acquiring Gontcharov (2017AstL...43..472G) XYZ catalog J/PAZh/43/521/xyzejk", end="...")
    old_row_limit = Vizier.ROW_LIMIT
    Vizier.ROW_LIMIT = -1 # Make sure we download the entire catalogue
    cat = Vizier.get_catalogs(["J/PAZh/43/521/xyzejk"])[0]
    Vizier.ROW_LIMIT = old_row_limit

    # Can't get this into a RegularGridInterpolator I've been unable to get the data sorted
    # in a way that makes it happy. However RBFs are nice and flexible.
    points = np.array(list(zip(cat["X"].value, cat["Y"].value, cat["Z"].value)), dtype=float)
    interp = RBFInterpolator(points, cat[interp_field].value,
                             neighbors=3**points.shape[1], smoothing=5)
    print("done.")
    return interp


def get_vergely_av(target_coords: SkyCoord) -> Tuple[float, bool]:
    """
    Queries the Vergely, Lallement & Cox (2022) [2022A&A...664A.174V] 3-d extinction map
    for the Av value of the target coordinates.

    Source data is in Galactic XYZ (Sun at 0,0,0). X towards galactic centre, Y along direction
    of rotation and Z positive towards galactic North. Distances are in pc.
    Extinction density is in nanomag/pc at a wavelength of 550 nm (aka A0).

    With the dust_exinction G23(RV=3.1) extinction curve the ratio of A0/AV is found to be 0.9985.

    TODO: update to work with a cached copy of the data and an interp, as we do with Gontcharov

    :target_coords: the astropy SkyCoords to query for   
    :returns: tuple of the A_V value and a flags indicating whether it is reliable
    """
    av, reliable = None, False
    try:
        # Extinction map of Vergely, Lallement & Cox (2022)
        ivoid = 'ivo://CDS.VizieR/J/A+A/664/A174'
        table = 'J/A+A/664/A174/cube_ext'
        vo_res = registry.search(ivoid=ivoid)[0]

        for res in [10, 50]: # central regions at 10 pc resolution and outer at 50 pc
            cart = np.ceil(target_coords.transform_to(Galactic).cartesian.xyz.to(u.pc)/res) * res
            rec = vo_res.get_service('tap').search(f'SELECT * FROM "{table}" ' +
                f'WHERE x={cart[0].value:.0f} AND y={cart[1].value:.0f} AND z={cart[2].value:.0f}')

            if len(rec):
                # The Exti field is extinction in nmag/pc @ 550 nm (A0).
                av = rec["Exti"][0] * target_coords.distance.to(u.pc).value / 10**9 / 0.9985
                reliable = True
                break
    except DALServiceError as exc:
        print(f"Failed to query: {exc}")
    return av, reliable
