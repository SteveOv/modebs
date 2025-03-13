"""
Low level utility functions for light curve ingest, pre-processing, estimation and fitting.
"""
#pylint: disable=no-member
import numpy as np
from uncertainties import UFloat, ufloat
from deblib.vmath import arccos, arcsin, degrees

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
