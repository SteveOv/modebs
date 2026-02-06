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

def query_tess_ebs_ephemeris(tics: List[Union[int, str]]) -> Dict[str, Union[float, UFloat]]:
    """
    Gets a dict of the ephemeris and morphology data from the TESS-ebs catalogue (J/ApJS/258/16)
    of Prsa+ (2022ApJS..258...16P). From a list of TIC ids this will return data for the first
    match, or None if no match.

    Previously we queried TESS-ebs with astroquery.Vizier which is more flexible at resolving Ids,
    however the questionable availability of the VizieR service makes it safer to take this offline.

    :tics: the potential ids for the target (some may have more than one TIC)
    :returns: dict of the requested data
    """
    # pylint: disable=too-many-locals
    table = _read_table(catalogue="J/ApJS/258/16", table_fname="tess-ebs.dat")
    data = None
    for tic in _yield_tic_nums(tics):
        if any(tic_mask := table["TIC"] == tic):
            row = table[tic_mask][0]

            period = ufloat(row["Per"], row["e_Per"])
            data = {
                "t0": ufloat(row["BJD0"], row["e_BJD0"]),
                "period": period,
            }

            morph = row["Morph"]
            if not np.isnan(morph) and isinstance(morph, Number):
                data["morph"] = morph

            # There are two sets of eclipse data; those based on the polyfit algorithm and those on
            # a 2-Gaussian algorithm. For the best chance of consistent values we use one or other.
            # We prefer the 2-Gaussian data as the polyfit tends to under value the eclipse widths
            for k_pattern in ["{0}-2g", "{0}-pf"]: # data derived from 2-Gaussian & polyfit algos
                k_phip, k_phis = k_pattern.format("Phip"), k_pattern.format("Phis")
                k_durp, k_durs = k_pattern.format("Wp"), k_pattern.format("Ws")

                # We want to get the phases so that the primary is zero and the secondary is
                # offset from this. Within TESS-ebs these are usually Phip=1 and Phis=offset
                # which is OK if we wrap 1 to 0. However, some appear shifted so that Phis=1
                # (i.e. TIC 26801525; phip-pf=0.448 & phis-pf=1.000 rather than phip=0 & phis=0.552)
                # In this case we need to undo the shift and switch the widths/durations.
                vals = row[[k_phip, k_phis, k_durp, k_durs]]
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

            # We have a match to a TIC number
            break
    return data


def estimate_eclipse_durations_from_morphology(morph: float,
                                               period: Union[float, UFloat]=1.0,
                                               esinw: Union[float, UFloat]=0.0) \
                                                    -> Tuple[float, float]:
    """
    Will provide an estimate of the eclipse durations from the morph value and period. The esinw
    value may be supplied, from which the durations will be modified for effects of eccentricity.

    :morph: the morphology value, expected to be in the range [0, 1]
    :period: the period - the resulting durations will be in the same units
    :esinw: the e*sin(omega) Poincare element, if known
    :returns: the estimated durations of the (primary, secondary) eclipses
    """
    mean_width = max(0.01, _eclipse_width_poly(morph))

    # Works well enough for the effect of eccentricity.
    # From esinw = ds-dp/ds+dp, therefore ds-dp = esinw * (ds+dp) = esinw * 2 * mean
    half_diff = esinw * mean_width
    return ((mean_width - half_diff) * period, (mean_width + half_diff) * period)


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
