"""
Low level utility functions for light curve ingest, pre-processing, estimation and fitting.
"""
# pylint: disable=no-member
from typing import Union
import warnings
import re
from numbers import Number

import numpy as np
from uncertainties import UFloat, ufloat
import astropy.units as u
from astropy.coordinates import SkyCoord
from astroquery.gaia import Gaia
from astroquery.vizier import Vizier

from deblib import limb_darkening
from deblib.vmath import arccos, arcsin, degrees

_TRIG_MIN = ufloat(-1, 0)
_TRIG_MAX = ufloat(1, 0)

_spt_to_teff_map = {
    "M": ufloat(3100, 800),
    "K": ufloat(4600, 700),
    "G": ufloat(5650, 350),
    "F": ufloat(6700, 500),
    "A": ufloat(8600, 1300),
    "B": ufloat(20000, 10000),
    "O": ufloat(35000, 10000)
}

def get_teff_from_spt(target_spt):
    """
    Estimates a stellar T_eff [K] from the passed spectral type.

    :target_spt: the spectral type string
    :returns: the estimated teff in K
    """
    teff = None

    # Also add the whole spt in case it's just a single char (i.e.: V889 Aql is set to "A")
    if target_spt is not None \
            and (spts := re.findall(r"([A-Z][0-9])", target_spt) + [target_spt.upper()]):
        for spt in spts:
            if spt and len(spt) and (tp := spt.strip()[0]) in _spt_to_teff_map \
                and _spt_to_teff_map[tp].n > (teff.n if teff is not None else 0):
                teff = _spt_to_teff_map[tp]
    return teff


def nominal_value(value: Union[UFloat, Number]) -> Number:
    """
    Simple helper function to get the nominal value of a number
    whether or not it's a UFloat
    """
    if isinstance(value, UFloat):
        return value.nominal_value
    return value


def get_tess_ebs_data(search_term: str, radius_as: float=5.) -> dict:
    """
    Gets a dictionary of ephemeris and morphology data from the TESS-ebs catalogue.
    """
    ebs_key_patterns = ["{0}-2g", "{0}-pf"] # data derived from 2-Gaussian & polyfit algos
    tess_ebs_catalog = Vizier(catalog="J/ApJS/258/16", row_limit=1)
    if (tbl := tess_ebs_catalog.query_object(search_term, radius=radius_as * u.arcsec)):
        sub_tbl = tbl[0]

        period = ufloat(sub_tbl["Per"][0], sub_tbl["e_Per"][0])
        data = {
            "t0": ufloat(sub_tbl["BJD0"][0], sub_tbl["e_BJD0"][0]),
            "period": period,
        }

        if not sub_tbl.mask["Morph"][0]:
            morph = sub_tbl["Morph"][0]
            if not np.isnan(morph) and isinstance(morph, Number):
                data["morph"] = morph

        # There are two sets of eclipse data; those based on the polyfit algorithm and those on the
        # 2-Gaussian algorithm. For the best chance of consistent values we use one or other set.
        for k_pattern in ebs_key_patterns:
            k_phip, k_phis = k_pattern.format("Phip"), k_pattern.format("Phis")
            k_durp, k_durs = k_pattern.format("Wp"), k_pattern.format("Ws")

            # Each value may be masked, NaN or non-numeric. We try to work with an unmasked set.
            if not any(sub_tbl.mask[k][0] for k in [k_phip, k_phis, k_durp, k_durs]):
                # We want to get the phases so that the primary is zero and the secondary is
                # offset from this. Within TESS-ebs these are usually Phip=1 and Phis=offset
                # which is OK if we wrap 1 to 0. However, some appear shifted so that Phis=1
                # (i.e. TIC 26801525; phip-pf=0.448 & phis-pf=1.000 rather than phip=0 & phis=0.552)
                # In this case we need to undo the shift and switch the widths/durations.
                vals = sub_tbl[[k_phip, k_phis, k_durp, k_durs]][0]
                if all(not np.isnan(v) and isinstance(v, Number) for v in vals):
                    phip, phis, durp, durs = vals

                    # Get both phases into the range [0, 1)
                    phip %= 1
                    phis %= 1
                    if phis < phip: # The phases have been shifted
                        phip, phis = 0, 1 - (phip - phis)
                        durp, durs = durs, durp
                    else:
                        phip, phis = 0, phis - phip

                    data["phiP"] = phip
                    data["phiS"] = phis

                    # The TESS-ebs eclipse widths are in units of phase
                    data["durP"] = durp * period
                    data["durS"] = durs * period
                    break

    else:
        data = None
    return data


def estimate_l3_with_gaia(centre: SkyCoord, radius_as: float=120,
                          target_source_id: int=None, target_g_mag: float=None,
                          max_l3: float=None, verbose: bool=False) -> float:
    """
    Estimates the third-light contribution from any sources near the target found in Gaia DR3.
    The returned L3 value is the sum of the values for each source found. Each target's L3 is its
    flux ratio compared with the target's, multiplied by a factor derived from its angular distance
    from the centre of the search area.

    Either target_source_id or target_g_mag must be supplied.
    
    :centre: the coordinates of the target and the centre of the search cone
    :radius: the radius of the search in arcsec
    :target_source_id: the Gaia DR3 source id of the target, if known
    :target_g_mag: the apparent G-band magnitude of the target, if it is not in Gaia DR3
    :max_l3: optional max allowable L3 value
    :returns: an estimated starting L3 value
    """
    l3 = 0
    Gaia.MAIN_GAIA_TABLE = "gaiadr3.gaia_source"
    if (job := Gaia.cone_search(coordinate=centre, radius=radius_as * u.arcsec,
                                columns=("source_id", "phot_g_mean_mag", "ra", "dec"))):
        tbl = job.get_results()
        if len(tbl) > 0:

            if target_source_id is not None:
                target_mask = tbl["source_id"] == int(target_source_id)
            else:
                # It may not be known, but if there is an object close to the centre of the search
                # cone with a magnitude similar to the target's we will assume it's the target.
                target_mask = (tbl["dist"] * 3600 < 5.0) \
                                & (np.abs(tbl["phot_g_mean_mag"] - target_g_mag) < 0.5)
                if verbose and any(target_mask):
                    print(f"Omitting {target_mask.sum()} object(s) likely to be the target")

            if target_g_mag is None:
                if any(target_mask):
                    target_g_mag = np.max(tbl[target_mask]["phot_g_mean_mag"])
                    if verbose:
                        print(f"Target object has a Gaia magnitude of {target_g_mag:.4f} (4 d.p.)")
                else:
                    raise ValueError("Cannot find target in search cone & no target_g_mag given")

            if any(~target_mask):
                flux_ratios = 10**(0.4 * (target_g_mag - tbl[~target_mask]["phot_g_mean_mag"]))

                # The Gaia query gives us a dist field which appears to be the angular distance
                # from search centre in degrees. Calculate a weighting based on the normalized dist.
                proximity_weights = 1 - (tbl[~target_mask]["dist"] * 3600 / radius_as)

                l3 = np.sum(flux_ratios * proximity_weights)

                if verbose:
                    print(f"Estimated the total third-light ratio (L3) to be {l3:.4f} (4 d.p.)",
                          f"for the {len(tbl[~target_mask])} nearby object(s) found in Gaia DR3.")

                if max_l3 is not None and max_l3 < l3:
                    l3 = max_l3
                    if verbose:
                        print(f"The estimate is reduced to the maximum allowed value of {max_l3}.")
            elif verbose:
                print("No nearby objects found in Gaia DR3 so the estimated L3=0")
    return l3


def append_calculated_inc_predictions(preds: np.ndarray[UFloat],
                                      field_name: str="inc") -> np.ndarray[UFloat]:
    """
    Calculate the predictions' inclination value(s) (in degrees) and append/overwrite to the array.

    :predictions: the predictions structured array to which inclination should be appended
    :field_name: the name of the inclination field to write to
    :returns: the revised array
    """
    with warnings.catch_warnings(category=[FutureWarning]):
        # Deprecation warning caused by the use of np.clip on ufloats
        warnings.filterwarnings("ignore", r"AffineScalarFunc.(__le__|__ge__)\(\) is deprecated.")
        names = list(preds.dtype.names)
        if "bP" in names:
            # From primary impact param:  i = arccos(bP * r1 * (1+esinw)/(1-e^2))
            r1 = preds["rA_plus_rB"] / (1+preds["k"])
            e_squared = preds["ecosw"]**2 + preds["esinw"]**2
            cosi = np.clip(preds["bP"]*r1*(1+preds["esinw"]) / (1-e_squared), _TRIG_MIN, _TRIG_MAX)
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
            # It's difficult to append a field to structured array or recarray so copy to new inst.
            # The numpy recfunctions module has merge and append_field funcs but they're slower.
            new = np.empty_like(preds, np.dtype(preds.dtype.descr + [(field_name, UFloat.dtype)]))
            new[names] = preds[names]
            new[field_name] = inc
        else:
            new = preds
        new[field_name] = inc
        return new


def pop_and_complete_ld_config(source_cfg: dict[str, any],
                               teffa: float, teffb: float,
                               logga: float, loggb: float,
                               default_algo: str="quad",
                               verbose: bool=False) -> dict[str, any]:
    """
    Will set up the limb darkening algo and coeffs, first by popping them from the source_cfg
    dictionary then completing the config with missing values. Where missing, the algo defaults
    to the value of the default_algo arg. For the quad, pow2 or h1h1 algos coefficient lookups
    are performed, based on the supplied teff and logg values, to populate any missing values.
    For any other algos, all coefficients must to be given in config.

    NOTE: pops the LD* items from source_cfg (except those ending _fit) into the returned config

    :source_cfg: the config fragment which may contain predefined LD params
    :teffa: effective temp of star A
    :teffb: effective temp of star B
    :logga: log(g) of star A
    :loggb: log(g) of star B
    :default_algo: the LD algo to use if not specified in source_cfg; either quad, pow2 or h1h2
    :verbose: whether or not to output details of the params chosen to stdout
    :return: the LD params only dict
    """
    params = {}
    for ld in [k for k in source_cfg if k.startswith("LD") and not k.endswith("_fit")]:
        params[ld] = source_cfg.pop(ld)

    for star, teff, logg in [("A", teffa, logga), ("B", teffb, loggb)]:
        algo = params.get(f"LD{star}", default_algo)
        if f"LD{star}" not in params \
                or f"LD{star}1" not in params or f"LD{star}2" not in params:
            # If we've not been given overrides for both the algo and coeffs we can look them up
            if algo.lower() == "same":
                coeffs = (0, 0) # JKTEBOP uses the A star params for both
            elif algo.lower() == "quad":
                coeffs = limb_darkening.lookup_quad_coefficients(logg, teff)
            elif algo.lower() in ["pow2", "h1h2"]:
                coeffs = limb_darkening.lookup_pow2_coefficients(logg, teff)
            else:
                raise ValueError(f"Lookup of coeffs not supported for the {algo} LD algorithm")

            # Add any missing algo/coeffs tags to the overrides
            params.setdefault(f"LD{star}", algo)
            if algo.lower() == "h1h2":
                # The h1h2 reparameterisation of the pow2 law addreeses correlation between the
                # coeffs; see Maxted (2018A&A...616A..39M) and Southworth (2023Obs...143...71S)
                params.setdefault(f"LD{star}1", 1 - coeffs[0]*(1 - 2**(-coeffs[1])))
                params.setdefault(f"LD{star}2", coeffs[0] * 2**(-coeffs[1]))
            else:
                params.setdefault(f"LD{star}1", coeffs[0])
                params.setdefault(f"LD{star}2", coeffs[1])
    if verbose:
        print(f"Limb darkening params: StarA={params['LDA']}({params['LDA1']}, {params['LDA2']}),",
              f"StarB={params['LDB']}({params['LDB1']}, {params['LDB2']})")
    return params
