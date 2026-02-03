""" Module for querying various, hopefully offline, catalogues """
from typing import Union, List, Dict
from pathlib import Path
from numbers import Number
import re

from uncertainties import ufloat, UFloat
import numpy as np
from astropy.table import Table

# Used to read the number from TIC ##### format tics
_tic_num_pattern = re.compile(r"TIC\s*(?P<tic>\d+)", re.IGNORECASE)

def query_tess_ebs(tics: List[Union[int, str]]) -> Dict[str, Union[float, UFloat]]:
    """
    Gets a dictionary of ephemeris and morphology data from the TESS-ebs catalogue.
    Takes a list of TIC ids and returns the data for the first match, or None if no match.

    Previously we queried TESS-ebs with astroquery.Vizier which is more flexible at resolving Ids,
    however the questionable availability of the VizieR service makes it safer to take this offline.

    :tics: the potential ids for the target (some may have more than one TIC)
    :returns: dict of the requested data
    """
    # pylint: disable=too-many-locals

    # May move to set this up just once (LRU cache?)
    table = Table.read(Path("./libs/data/tess-ebs/tess-ebs.dat"),
                       readme=Path("./libs/data/tess-ebs/ReadMe"),
                       format="ascii.cds")

    if isinstance(tics, Number|str):
        tics = [tics]

    for tic in tics:
        if isinstance(tic, str):
            if tic.isnumeric():
                tic = int(tic)
            elif (match := _tic_num_pattern.match(tic)) is not None and "tic" in match.groupdict():
                tic = int(match.groups("tic"))
            else:
                tic = None

        if tic is not None and any(tic_mask := table["TIC"] == tic):
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
