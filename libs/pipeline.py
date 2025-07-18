"""
Low level utility functions for light curve ingest, pre-processing, estimation and fitting.
"""
#pylint: disable=no-member
import warnings
import re

import numpy as np
from uncertainties import UFloat, ufloat

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
    if (spts := re.findall(r"([A-Z][0-9])", target_spt) + [target_spt.upper()]) is not None:
        for spt in spts:
            if (tp := spt.strip()[0]) in _spt_to_teff_map \
                and _spt_to_teff_map[tp].n > (teff.n if teff is not None else 0):
                teff = _spt_to_teff_map[tp]
    return teff
