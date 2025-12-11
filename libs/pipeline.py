"""
Low level utility functions for light curve ingest, pre-processing, estimation and fitting.
"""
#pylint: disable=no-member
import warnings
import re

import numpy as np
from uncertainties import UFloat, ufloat

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

def pop_and_complete_ld_config(source_cfg: dict[str, any],
                               teffa: float, teffb: float,
                               logga: float, loggb: float,
                               verbose: bool=False) -> dict[str, any]:
    """
    Will set up the limb darkening algo and coeffs, first by popping them from the source_cfg
    dictionary then completing the config with missing values. Where missing, the algo defaults
    to quad unless pow2, h1h2 or same has been specified. Coefficient lookups are performed,
    based on the supplied teff and logg values, to populate any missing values.

    NOTE: pops the LD* items from source_cfg (except those ending _fit) into the returned config

    :source_cfg: the config fragment which may contain predefined LD params
    :teffa: effective temp of star A
    :teffb: effective temp of star B
    :logga: log(g) of star A
    :loggb: log(g) of star B
    :verbose: whether or not to output details of the params chosen to stdout
    :return: the LD params only dict
    """
    params = {}
    for ld in [k for k in source_cfg if k.startswith("LD") and not k.endswith("_fit")]:
        params[ld] = source_cfg.pop(ld)

    for star, teff, logg in [("A", teffa, logga), ("B", teffb, loggb)]:
        algo = params.get(f"LD{star}", "quad") # Only quad, pow2 or h1h2 supported
        if f"LD{star}" not in params \
                or f"LD{star}1" not in params or f"LD{star}2" not in params:
            # If we've not been given overrides for both the algo and coeffs we can look them up
            if algo.lower() == "same":
                coeffs = (0, 0) # JKTEBOP uses the A star params for both
            elif algo.lower() == "quad":
                coeffs = limb_darkening.lookup_quad_coefficients(logg, teff)
            else:
                coeffs = limb_darkening.lookup_pow2_coefficients(logg, teff)

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
