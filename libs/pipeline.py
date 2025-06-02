"""
Low level utility functions for light curve ingest, pre-processing, estimation and fitting.
"""
#pylint: disable=no-member
from typing import Tuple
from pathlib import Path
import re
from urllib.parse import quote_plus
from requests.exceptions import HTTPError

import numpy as np
from uncertainties import UFloat, ufloat
import astropy.units as u
from astropy.table import Table
from astropy.coordinates import SkyCoord

from dustmaps import config, bayestar           # Bayestar dustmaps/dereddening map
from pyvo import registry, DALServiceError      # Vergeley at al. extinction catalogue

from deblib.vmath import arccos, arcsin, degrees
from deblib.constants import c

_TRIG_MIN = ufloat(-1, 0)
_TRIG_MAX = ufloat(1, 0)

def append_calculated_inc_predictions(preds: np.ndarray[UFloat],
                                      field_name: str="inc") -> np.ndarray[UFloat]:
    """
    Calculate the predictions' inclination value(s) (in degrees) and append/overwrite to the array.

    :predictions: the predictions structured array to which inclination should be appended
    :field_name: the name of the inclination field to write to
    :returns: the revised array
    """
    names = list(preds.dtype.names)
    if "bP" in names:
        # From primary impact param:  i = arccos(bP * r1 * (1+esinw)/(1-e^2))
        r1 = preds["rA_plus_rB"] / (1+preds["k"])
        e_squared = preds["ecosw"]**2 + preds["esinw"]**2
        cosi = np.clip(preds["bP"] * r1 * (1+preds["esinw"]) / (1-e_squared), _TRIG_MIN, _TRIG_MAX)
        inc = degrees(arccos(cosi))
    elif "cosi" in names:
        cosi = np.clip(preds["cosi"], _TRIG_MIN, _TRIG_MAX)
        inc = degrees(arccos(cosi))
    elif "sini" in names:
        sini = np.clip(preds["sini"], _TRIG_MIN, _TRIG_MAX)
        inc = degrees(arcsin(sini))
    else:
        raise KeyError("Missing bP, cosi or sini in predictions required to calc inc.")

    if field_name not in names:
        # It's difficult to append a field to structured array or recarray so copy over to new inst.
        # The numpy recfunctions module has merge and append_field funcs but they're slower.
        new = np.empty_like(preds, np.dtype(preds.dtype.descr + [(field_name, UFloat.dtype)]))
        new[names] = preds[names]
        new[field_name] = inc
    else:
        new = preds
    new[field_name] = inc
    return new


def get_sed_for_target(target: str,
                       search_term: str=None,
                       radius: float=0.1,
                       missing_uncertainty_ratio: float=0.1):
    """
    Gets spectral energy distribution (SED) observations for the target. These data are found and
    downloaded from the VizieR photometry tool (see http://viz-beta.u-strasbg.fr/vizier/sed/doc/).

    Once donwloaded the data are sorted and 10% errorbars are applied where none recorded.
    The sed_flux, sed_eflux and sed_freq fields are converted to SI units; W / m^2 / Hz and Hz.
    Calculated fields are added for sed_wl (wavelength in microns), sed_vfv and sed_evfv
    (freq * flux) to aid plotting, where x and y axes of wavelength and ν*F(ν) are often used.

    Tables will be locally cached within the `.cache/.sed/` directory

    :target: the name of the target object
    :search_term: optional search term, or leave as None to use the target value
    :radius: the search radius in arcsec
    :missing_uncertainty_rate: uncertainty, as a ratio of the fluxes, to apply where none recorded
    :returns: an astropy Table containing the chosen data, sorted by descending frequency
    """
    sed_cache_dir = Path(".cache/.sed/")
    sed_cache_dir.mkdir(parents=True, exist_ok=True)

    # Read in the SED for this target via the cache
    sed_fname = sed_cache_dir / (re.sub(r"[^\w\d-]", "-", target.lower()) + f"-{radius}.vot")
    if not sed_fname.exists():
        targ = quote_plus(search_term or target)
        sed = Table.read(f"https://vizier.cds.unistra.fr/viz-bin/sed?-c={targ}&-c.rs={radius}")
        sed.write(sed_fname, format="votable") # votable matches that published in link above

    sed = Table.read(sed_fname)
    sed.sort(["sed_freq"], reverse=True)

    # Set flux uncertainties where none given
    mask_no_err = (sed["sed_eflux"].value == 0) | np.isnan(sed["sed_eflux"])
    sed["sed_eflux"][mask_no_err] = sed["sed_flux"][mask_no_err] * missing_uncertainty_ratio

    # Get the data into SI units
    sed["sed_flux"] = sed["sed_flux"].to(u.W / u.m**2 / u.Hz)
    sed["sed_eflux"] = sed["sed_eflux"].to(u.W / u.m**2 / u.Hz)
    sed["sed_freq"] = sed["sed_freq"].to(u.Hz)

    # Add wavelength, and vF(v) columns which we may use for plots
    sed["sed_wl"] = np.divide(c * u.m / u.s, sed["sed_freq"]).to(u.um)
    sed["sed_vfv"] = sed["sed_flux"] * sed["sed_freq"] * u.Hz   # Req explicit units to cancel 1/Hz
    sed["sed_evfv"] = sed["sed_eflux"] * sed["sed_freq"] * u.Hz
    return sed


def get_bayestar_ebv_for_coords(target_coords: SkyCoord,
                                version: str="bayestar2019",
                                ebv_factor: float=0.996) -> Tuple[float, dict]:
    """
    Queries the Bayestar 2019 dereddening map for the E(B-V) value for the target coordinates.

    Conversion from Bayestar 17 or 19 to E(B-V) documented at http://argonaut.skymaps.info/usage
    as E(B-V) = 0.884 x bayestar or E(B-V) = 0.996 x bayestar
    
    :target_coords: the astropy SkyCoords to query for
    :version: the version of the Bayestar dust maps to use
    :ebv_factor: the conversion factor from bayestar extinction to E(B-V) to use
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
    return ebv_factor * ebv, { n: flags[n] for n in flags.dtype.names }


def get_vergely_av_for_coords(target_coords: SkyCoord):
    """
    Queries the Vergely, Lallement & Cox (2022) 3-d extinction map for the Av value of the
    target coordinates.

    :target_coords: the astropy SkyCoords to query for   
    :returns: the Av value
    """
    av = None
    try:
        # Extinction map of Vergely, Lallement & Cox (2022)
        ivoid = 'ivo://CDS.VizieR/J/A+A/664/A174'
        table = 'J/A+A/664/A174/cube_ext'
        vo_res = registry.search(ivoid=ivoid)[0]
        print(f"Querying {vo_res.res_title} ({vo_res.source_value}) for extinction data.")

        for res in [10, 50]: # central regions at 10 pc resolution and outer at 50 pc
            cart = np.round(target_coords.cartesian.xyz.to(u.pc) / res) * res
            rec = vo_res.get_service('tap').search(f'SELECT * FROM "{table}" ' +
                f'WHERE x={cart[0].value:.0f} AND y={cart[1].value:.0f} AND z={cart[2].value:.0f}')

            if len(rec):
                # TODO: anything extra to map these values to Av?
                ext_nmag_per_pc = rec["Exti"][0] # nmag
                av = (ext_nmag_per_pc * target_coords.distance.to(u.pc).value) / 10**9
                break
    except DALServiceError as exc:
        print(f"Failed to query: {exc}")
    return av
