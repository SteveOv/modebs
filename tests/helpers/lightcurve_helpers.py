""" A module of lightcurve related helper functions for use throughout the tests. """
# pylint: disable=no-member
from typing import List
from inspect import getsourcefile
from pathlib import Path

import numpy as np
import astropy.units as u
from astropy.time import Time
import lightkurve as lk

TEST_FITS_DIR = Path(getsourcefile(lambda:0)).parent / "../data/mast"
TEST_OUTPUT_DIR = Path(getsourcefile(lambda:0)).parent / "../../.cache/.test_data"

""" A dictionary of known lightkurve targets with downloaded fits files. """
KNOWN_TARGETS = {
    "CW Eri": {
        "tic": 98853987,
        "sector": 31,
        # For tests which don't use TESS-ebs
        "t0": Time(2163.056459177, format="btjd", scale="tdb"),
        "period": 2.728370923 * u.d,
        "fits": {
            4: "tess2018292075959-s0004-0000000098853987-0124-s_lc.fits",
            31: "tess2020294194027-s0031-0000000098853987-0198-s_lc.fits",
        }
    },
    "RR Lyn": {
        "tic": 11491822,
        "sector": 20,
        "fits": {
            20: "tess2019357164649-s0020-0000000011491822-0165-s_lc.fits",
            60: "tess2022357055054-s0060-0000000011491822-0249-s_lc.fits",
            73: "tess2023341045131-s0073-0000000011491822-0268-s_lc.fits"
        }
    },
    "IT Cas": {
        "tic": 26801525,
        "sector": 17,
        # Override TESS-ebs
        "phiS": 0.552,
        "fits": {
            17: "tess2019279210107-s0017-0000000026801525-0161-s_lc.fits",
        }
    },
    "AN Cam": {
        "tic": 103098373,
        "sector": 25,
        # Not in TESS-ebs
        "t0": Time(1992.007512423, format="btjd", scale="tdb"),
        "period": 20.99842 * u.d,
        "durP": 0.558,
        "durS": 0.711,
        "phiS": 0.779,
        "fits": {
            25: "hlsp_tess-spoc_tess_phot_0000000103098373-s0025_tess_v1_lc.fits",
            52: "tess2022138205153-s0052-0000000103098373-0224-s_lc.fits",
            53: "tess2022164095748-s0053-0000000103098373-0226-s_lc.fits",
            59: "tess2022330142927-s0059-0000000103098373-0248-s_lc.fits"
        }
    },
    "CM Dra": {
        "tic": 199574208,
        "sector": 24,
        "fits": {
            24: "tess2020106103520-s0024-0000000199574208-0180-s_lc.fits",
            25: "tess2020133194932-s0025-0000000199574208-0182-s_lc.fits",
            26: "tess2020160202036-s0026-0000000199574208-0188-s_lc.fits"
        }
    },
    "V889 Aql": {
        "tic": 300000680,
        "sector": 40,
        # Not in TESS-ebs
        "t0": 2772.124954368,
        "period": 11.120757,
        "durP": 0.271,
        "durS": 0.507,
        "depthP": 0.5,
        "depthS": 0.43,
        "phiS": 0.354,
        "morph": 0.200,
        "fits": {
            40: "hlsp_tess-spoc_tess_phot_0000000300000680-s0040_tess_v1_lc.fits",
        }
    },
    "TIC 30034081": {
        "tic": 30034081,
        "sector": 64,
        # Incorrect period in TESS-ebs (needs doubling)
        "t0": Time(1411.553116, format="btjd", scale="tdb"),
        "period": 2.34461 * 2 * u.d,
        "durP": 0.35,
        "durS": 0.35,
        "phiS": 0.5,
        "fits": {
            64: "tess2023096110322-s0064-0000000030034081-0257-s_lc.fits",
        }
    },
    "TIC 118313102": {
        "tic": 118313102,
        "sector": 8,
        "durS": 0.42,
        "fits": {
            8: "tess2019032160000-s0008-0000000118313102-0136-s_lc.fits",
            9: "tess2019058134432-s0009-0000000118313102-0139-s_lc.fits",
        }
    },
    "TIC 255567460": {
        "tic": 255567460,
        "sector": 66,
        "fits": {
            66: "tess2023153011303-s0066-0000000255567460-0260-s_lc.fits",
        }
    },
    "TIC 350298314": {
        "tic": 350298314,
        "sector": 63,
        "fits": {
            63: "tess2023069172124-s0063-0000000350298314-0255-s_lc.fits",
            65: "tess2023124020739-s0065-0000000350298314-0259-s_lc.fits",
        }
    },
}


def load_default_lightcurve(target: str,
                            with_mag_columns: bool=True) -> lk.LightCurve:
    """
    Loads the sector's default (config: sector) LightCurve from file for the requested target.

    :target: the name of the target, also the key to the KNOWN_TARGETS dict
    :with_mag_columns: whether or not to create delta_mag and delta_mag_err columns
    :returns: the requested LightCurve
    """
    params = KNOWN_TARGETS[target]
    return load_lightcurves(target, [params["sector"]], with_mag_columns)[0]


def load_lightcurves(target: str,
                     sectors: List[int]=None,
                     with_mag_columns: bool=True) -> lk.LightCurveCollection:
    """
    Load lightcurves from file for the configured target
    
    Can only load sectors which have been configured for the target in KNOWN_TARGETS
    and which have corresponding fits files stored under the ../data/mast/[target] directory

    :target: the name of the target, also the key to the KNOWN_TARGETS dict
    :sectors: the known sector(s) to load, or if None all configured sectors
    :with_mag_columns: whether or not to create delta_mag and delta_mag_err columns
    :returns: a LightCurveCollection with the requested LightCurves
    """
    params = KNOWN_TARGETS[target]
    if sectors is None or len(sectors) == 0:
        sectors = list(params["fits"].keys())

    fits_dir = TEST_FITS_DIR / f"{target.lower().replace(' ', '-')}"

    lcs = lk.LightCurveCollection([])
    for sector in sectors:
        lc = lk.read(fits_dir / params["fits"][sector],
                     flux_column=params.get("flux_column", "sap_flux"),
                     quality_bitmask=params.get("quality_bitmask", "hardest"))

        lc = lc[~((np.isnan(lc.flux)) | (lc.flux < 0))].normalize()
        if with_mag_columns:
            append_mag_columns(lc)

        lc.meta["LABEL"] = f"{target} S{lc.meta['SECTOR']:02d}"
        lc.meta["clip_mask"] = np.ones((len(lc)), dtype=bool)

        lcs.append(lc)
    return lcs


def append_mag_columns(lc: lk.LightCurve):
    """
    Calculates and appends the delta_mag and delta_mag_err columns to the passed LightCurve.
    """
    fmax = lc.flux.max().value
    lc["delta_mag"] = u.Quantity(2.5 * np.log10(fmax/lc.flux.value) * u.mag)
    lc["delta_mag_err"] = u.Quantity(
        2.5
        * 0.5
        * np.abs(
            np.subtract(
                np.log10(np.add(lc.flux.value, lc.flux_err.value)),
                np.log10(np.subtract(lc.flux.value, lc.flux_err.value))
            )
        )
        * u.mag)
