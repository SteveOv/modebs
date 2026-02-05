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
    "CW Eri": { # Easy light-curves
        "tic": 98853987,
        "sector": 31,
        "period": 2.728370923 * u.d,
        "epoch_time": Time(2163.056459177, format="btjd", scale="tdb"),
        "ecosw": 0.00502,
        "esinw": -0.0121,
        "ecc": 0.0131,
        "expect_phase2": 0.5032,
        "expect_width2": 0.976,
        "fits": {
            4: "tess2018292075959-s0004-0000000098853987-0124-s_lc.fits",
            31: "tess2020294194027-s0031-0000000098853987-0198-s_lc.fits",
        }
    },
    "RR Lyn": { # Early secondary eclipses
        "tic": 11491822,
        "sector": 20,
        "period": 9.946591113 * u.d,
        "epoch_time": Time(1851.925277662, format="btjd", scale="tdb"),
        "expect_phase2": 0.45,
        "expect_width2": 1.2,
        "fits": {
            20: "tess2019357164649-s0020-0000000011491822-0165-s_lc.fits",
            60: "tess2022357055054-s0060-0000000011491822-0249-s_lc.fits",
            73: "tess2023341045131-s0073-0000000011491822-0268-s_lc.fits"
        }
    },
    "IT Cas": { # Eccentric, late secondary, primary/secondary similar depths
        "tic": 26801525,
        "sector": 17,
        "period": 3.8966513 * u.d,
        "epoch_time": Time(1778.3091293396, format="btjd", scale="tdb"),
        "expect_phase2": 0.55,
        "expect_width": 1.0,
        "fits": {
            17: "tess2019279210107-s0017-0000000026801525-0161-s_lc.fits",
        }
    },
    "AN Cam": { # Very late secondary eclipses, near phase 0.8
        "tic": 103098373,
        "sector": 25,
        "period": 20.99842 * u.d,
        "epoch_time": Time(1992.007512423, format="btjd", scale="tdb"),
        "expect_phase2": 0.78,
        "expect_width2": 1.1,
        "fits": {
            25: "hlsp_tess-spoc_tess_phot_0000000103098373-s0025_tess_v1_lc.fits",
        }
    },
    "V889 Aql": { # Lower (600 s) cadence, highly eccentric and small mid-sector gap
        "tic": 300000680,
        "sector": 40,
        "period": 11.120757 * u.d,
        "epoch_time": Time(2416.259790, format="btjd", scale="tdb"),
        "expect_phase2": 0.35,
        "expect_width2": 1.9,
        "fits": {
            40: "hlsp_tess-spoc_tess_phot_0000000300000680-s0040_tess_v1_lc.fits",
        }
    },
    "TIC 255567460": { # S66 has no primary eclipses
        "tic": 255567460,
        "sector": 66,
        "period": 13.79633 * u.d,
        "epoch_time": Time(1469.208711, format="btjd", scale="tdb"),
        "expect_phase2": 0.5,
        "expect_width2": 0.14,
        "fits": {
            66: "tess2023153011303-s0066-0000000255567460-0260-s_lc.fits",
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
    for sector in sorted(sectors):
        lc = lk.read(fits_dir / params["fits"][sector],
                     flux_column=params.get("flux_column", "sap_flux"),
                     quality_bitmask=params.get("quality_bitmask", "hardest"))

        lc = lc[~((np.isnan(lc.flux)) | (lc.flux < 0))].normalize()
        if with_mag_columns:
            append_mag_columns(lc)

        lc.meta["LABEL"] = f"{target} S{lc.meta['SECTOR']:02d}"
        lc.meta["clip_mask"] = np.ones((len(lc)), dtype=bool)
        lc.meta["t0"] = params["epoch_time"]

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
