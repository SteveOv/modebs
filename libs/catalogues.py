""" Module for querying various, hopefully offline, catalogues """
from typing import Union, List, Dict, Tuple
from pathlib import Path
from numbers import Number
import re
from functools import lru_cache

from uncertainties import ufloat, UFloat
import numpy as np
from astropy.table import Table

# Used to read the number from TIC ##### format tics
_tic_num_pattern = re.compile(r"TIC\s*(?P<tic>\d+)", re.IGNORECASE)

# Used to estimate eclipse widths from morph value.
# Based on a polyfit() on TESS-ebs morphs vs max(mean(Wp-fp, Ws-fp), mean(Wp-2g, Ws-2g)) which gives
# coefficients of [-1.99218228e+00  3.85365374e+00 -1.76590596e+00  3.76940259e-01 -3.43763161e-04].
# Cannot have negative, so shift up to zero.
_eclipse_width_poly = np.poly1d([-1.99218228, 3.85365374, -1.76590596, 0.37694026, 0.])

def query_tess_ebs_ephemeris(tics: List[Union[int, str]],
                             period_factor: float=1.) -> Dict[str, Union[float, UFloat]]:
    """
    Gets a dict of the ephemeris and morphology data from the TESS-ebs catalogue (J/ApJS/258/16)
    of Prsa+ (2022ApJS..258...16P). From a list of TIC ids this will return data for the first
    match, or None if no match.

    Previously we queried TESS-ebs with astroquery.Vizier which is more flexible at resolving Ids,
    however the questionable availability of the VizieR service makes it safer to take this offline.

    :tics: the potential ids for the target (some may have more than one TIC)
    :duration_factor: multiplier for the period & durations to correct for under/over reporting
    :returns: dict of the requested data
    """
    # pylint: disable=too-many-locals
    table = _read_table(catalogue="J/ApJS/258/16", table_fname="tess-ebs.dat")
    data = None
    for tic in _yield_tic_nums(tics):
        if any(tic_mask := table["TIC"] == tic):
            # Reader will mask any non-numeric/missing vals in the dat file (usually empty text).
            # We want the chosen row, with masked values converted nan, as a single array row.
            row = np.ma.filled(table[tic_mask], fill_value=np.nan).as_array()[0]

            period = ufloat(row["Per"], row["e_Per"]) * period_factor
            data = {
                "t0": ufloat(row["BJD0"], row["e_BJD0"]),
                "period": period,
            }

            morph = row["Morph"]
            if not np.isnan(morph) and isinstance(morph, Number):
                data["morph"] = morph

            # There are two sets of eclipse data; those based on the polyfit algorithm and those on
            # a 2-Gaussian algorithm. For the best chance of consistent values we use one or other.
            vals = np.array([
                [row[k + k_algo] for k in ["Phip", "Phis", "Wp", "Ws", "Dp", "Ds"]]
                    for k_algo in ["-2g", "-pf"]
            ], dtype=float)

            # Prefer the set with the most data, and break a tie in favour of the 2g values (set 0)
            # as these tend to have wider eclipse widths (we find polyfit tends to under value).
            is_num = ~np.isnan(vals)
            row_ix = np.argmax(np.sum(is_num, axis=1))

            # We want to get the phases so that the primary is zero and the secondary is
            # offset from this. Within TESS-ebs Phip is often \sim 1 and Phis < Phip.
            if all(is_num[row_ix, 0:2]):
                while vals[row_ix, 0] > vals[row_ix, 1]:
                    vals[row_ix, 0] -= 1
                data["phiS"] = vals[row_ix, 1] - vals[row_ix, 0]
            else:
                data["phiP"] = 0
                data["phiS"] = None

            # TESS-ebs eclipse widths are in units of phase and depth in units of normalized flux
            data["widthP"] = vals[row_ix, 2] if is_num[row_ix, 2] else None
            data["widthS"] = vals[row_ix, 3] if is_num[row_ix, 3] else None
            data["depthP"] = vals[row_ix, 4] if is_num[row_ix, 4] else None
            data["depthS"] = vals[row_ix, 5] if is_num[row_ix, 5] else None

            # We have a match to a TIC number
            break
    return data


def estimate_eclipse_widths_from_morphology(morph: float,
                                            esinw: Union[float, UFloat]=0.0) \
                                                    -> Tuple[float, float]:
    """
    Will provide an estimate of the eclipse widths from the morph value. The esinw value
    may be supplied, from which the durations will be modified for effects of eccentricity.

    :morph: the morphology value, expected to be in the range [0, 1]
    :esinw: the e*sin(omega) Poincare element, if known
    :returns: the estimated widths of the (primary, secondary) eclipses
    """
    mean_width = max(0.01, _eclipse_width_poly(morph))

    # Works well enough for the effect of eccentricity.
    # From esinw = ds-dp/ds+dp, therefore ds-dp = esinw * (ds+dp) = esinw * 2 * mean
    half_diff = esinw * mean_width
    return (mean_width - half_diff, mean_width + half_diff)


def query_tess_ebs_in_sh(tics: List[Union[int, str]]) -> dict:
    """
    Gets a dictionary of ephemeris and physical data from table 3 of the 'TESS EBs in the
    southern hemisphere catalogue' (J/ApJ/912/123) of Justesen & Albrecht (2021ApJ...912..123J).
    Takes a list of TIC ids and returns the data for the first match, or None if no match.

    :tics: the potential ids for the target (some may have more than one TIC)
    :returns: dict of the requested data
    """
    # pylint: disable=too-many-locals
    table = _read_table(catalogue="J/ApJ/912/123", table_fname="table3.dat")
    data = None
    for tic in _yield_tic_nums(tics):
        if any(tic_mask := table["TIC"] == tic):
            row = table[tic_mask][0]

            data = {
                "t0": row["t1"],
                "period": row["Per"],
                "k": row["rp"],
                "a/R1": row["a/R1"],
                "ecosw": row["ecosw"],
                "esinw": row["esinw"],
                "inc": row["inc"],
                "light_ratio": row["fp"],
                "TeffA": row["Teff1"],
                "TeffB": row["Teff2"]
            }

            # We have a match to a TIC number
            break
    return data


def _yield_tic_nums(tics: List[Union[int, str]]):
    """
    Parse a list of putative TIC identifiers and yield the numeric component of those that are valid
    """
    if isinstance(tics, str|Number):
        tics = [tics]

    for tic in tics:
        if isinstance(tic, Number):
            yield int(tic)
        elif isinstance(tic, str):
            if tic.isnumeric():
                yield int(tic)
            elif (match := _tic_num_pattern.match(tic)) is not None and "tic" in match.groupdict():
                yield int(match.group("tic"))


@lru_cache
def _read_table(catalogue: str, table_fname: str, readme_fname: str="ReadMe") -> Table:
    """
    Will read the requested CDS format ascii table into an astropy Table

    :catalogue: the CDS identifier for the catalogue
    :table_fname: the name of the table file to open
    :readme_fname: the name of the accompanying readme file which contains the table's metadata
    :returns: the requested Table
    """
    cat_dir = Path("./libs/data/catalogues") / catalogue.replace("/", "-")
    return Table.read(cat_dir / table_fname, readme=cat_dir / readme_fname, format="ascii.cds")
