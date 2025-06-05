"""
Low level utility functions for light curve ingest, pre-processing, estimation and fitting.
"""
#pylint: disable=no-member
from typing import Tuple
from requests.exceptions import HTTPError

import numpy as np
import astropy.units as u
from astropy.coordinates import SkyCoord
from astroquery.vizier import Vizier

from dustmaps import config, bayestar           # Bayestar dustmaps/dereddening map
from pyvo import registry, DALServiceError      # Vergeley at al. extinction catalogue

# TODO: remove the extinction funcs from pipeline and update quick_fit

def get_bayestar_ebv(target_coords: SkyCoord,
                     version: str="bayestar2019",
                     conversion_factor: float=0.996) -> Tuple[float, dict]:
    """
    Queries the Bayestar 2019 dereddening map for the E(B-V) value for the target coordinates.

    Conversion from Bayestar 17 or 19 to E(B-V) documented at http://argonaut.skymaps.info/usage
    as E(B-V) = 0.884 x bayestar or E(B-V) = 0.996 x bayestar
    
    :target_coords: the astropy SkyCoords to query for
    :version: the version of the Bayestar dust maps to use
    :conversion_factor: the factor to apply to bayestar extinction for E(B-V)
    :returns: tuple of the E(B-V) value and a dict of the diagnostic flags associated with the query
    """
    try:
        # Creates/confirms local cache of Bayestar data within the .cache directory
        config.config['data_dir'] = '.cache/.dustmapsrc'
        bayestar.fetch(version=version)
    except HTTPError as exc:
        print(f"Unable to (re)fetch data for {version}. Caught error '{exc}'")
    except ValueError as exc:
        print(f"Unable to parse response for {version}. Caught error '{exc}'")

    # Now we can use the local cache for the lookup
    query = bayestar.BayestarQuery(version=version)
    ebv, flags =  query(target_coords, mode='median', return_flags=True)
    return conversion_factor * ebv, { n: flags[n] for n in flags.dtype.names }


def get_gontcharov_ebv(target_coords: SkyCoord,
                       conversion_factor: float=1.7033):
    """
    Queries the Gontcharov (2017) [2017AstL...43..472G] 3-d extinction map for the Ebv value of the
    target coordinates.

    Extends radially to at least 700 pc, in galactic coords at 20 pc distance intervals
    Conversion: E(B-V) = 1.7033 E(J-K)

    :target_coords: the astropy SkyCoords to query for   
    :conversion_factor: the factor to apply to E(J-Ks) for E(B-V)
    :returns: tuple of the E(B-V) value and a dict of the diagnostic flags associated with the query
    """
    ebv = 0
    flags = { "converged": False } # mimics Bayestar - will set try if we get a good match
    vizier = Vizier(catalog='J/PAZh/43/521/rlbejk', columns=["**"])

    # Round up the galactic coords (to nearest deg) and distance (to nearest 20 pc)
    glon, glat = np.ceil(target_coords.galactic.l.deg), np.ceil(target_coords.galactic.b.deg)
    dist = np.ceil(target_coords.distance.to(u.pc).value / 20) * 20

    for r, dflag in [(dist, True), (700, False), (600, False)]:
        if _tbl := vizier.query_constraints(R=r, GLON=glon, GLAT=glat):
            if len(_tbl):
                ebv = _tbl[0]["E(J-Ks)"][0] * conversion_factor
                flags["converged"] = dflag
                break

    return ebv, flags


def get_vergely_av(target_coords: SkyCoord):
    """
    Queries the Vergely, Lallement & Cox (2022) 3-d extinction map for the Av value of the
    target coordinates.

    TODO: this needs further work

    :target_coords: the astropy SkyCoords to query for   
    :returns: the Av value
    """
    av = None
    flags = { "converged": False } # mimics Bayestar - will set try if we get a good match
    try:
        # Extinction map of Vergely, Lallement & Cox (2022)
        ivoid = 'ivo://CDS.VizieR/J/A+A/664/A174'
        table = 'J/A+A/664/A174/cube_ext'
        vo_res = registry.search(ivoid=ivoid)[0]
        print(f"Querying {vo_res.res_title} ({vo_res.source_value}) for extinction data.")

        for res in [10, 50]: # central regions at 10 pc resolution and outer at 50 pc
            cart = np.ceil(target_coords.cartesian.xyz.to(u.pc) / res) * res
            rec = vo_res.get_service('tap').search(f'SELECT * FROM "{table}" ' +
                f'WHERE x={cart[0].value:.0f} AND y={cart[1].value:.0f} AND z={cart[2].value:.0f}')

            if len(rec):
                # TODO: anything extra to map these values to Av?
                ext_nmag_per_pc = rec["Exti"][0] # nmag
                av = (ext_nmag_per_pc * target_coords.distance.to(u.pc).value) / 10**9
                flags['converged'] = True
                break
    except DALServiceError as exc:
        print(f"Failed to query: {exc}")
    return av, flags
