""" Pipeline Stage 1 - ingesting targets """
# pylint: disable=no-member
from warnings import catch_warnings, filterwarnings
from pathlib import Path
import re
import warnings
import json
import argparse

import numpy as np
import astropy.units as u
from astropy.table import QTable
from astropy.io.votable import parse_single_table
from astropy.utils.diff import report_diff_values

from astroquery.vizier import Vizier
from astroquery.simbad import Simbad
from astroquery.gaia import Gaia

# pylint: disable=line-too-long, wrong-import-position
warnings.filterwarnings("ignore", "Using UFloat objects with std_dev==0 may give unexpected results.", category=UserWarning)
from uncertainties import ufloat

from libs import pipeline


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Pipeline stage 1: ingest of targets.")
    ap.add_argument("-i", "--input-file", dest="targets_file", type=Path, required=False,
                    help="json file containing the details of the targets to ingest")
    # We use a None in model_files as indication to pull in the default model under ebop_maven/data
    ap.set_defaults(targets_file=Path("./config/plato-lops2-tess-ebs-explicit-targets.json"))
    args = ap.parse_args()


    with open(args.targets_file, mode="r", encoding="utf8") as tf:
        _tfc = json.load(tf)
        _excl_def = _tfc.get("exclude_default", False)
        _tc = _tfc["target_configs"] # want to fail if this isn't found
        target_configs = { k: _tc[k] for k in _tc if not _tc.get(k, {}).get("exclude", _excl_def) }
        print(f"Read in '{tf.name}' which includes {len(target_configs)} target(s).")

    # Used to get the extended information published for each target
    simbad = Simbad()
    simbad.add_votable_fields("parallax", "allfluxes", "sp", "ids")
    tic_catalog = Vizier(catalog="IV/39/tic82", row_limit=10)
    tess_ebs_catalog = Vizier(catalog="J/ApJS/258/16", row_limit=1)
    gaia_tbosb_catalog = Vizier(catalog="I/357/tbosb2", row_limit=1)

    target_dtype = [
        ("target", "<U14"),
        ("search_term", "<U20"),
        ("main_id", "<U20"),
        ("tics", "<U40"),
        ("gaia_dr3_id", int),
        ("spt", "<U20"),
        ("ra", float),
        ("dec", float),
        ("parallax", float),
        ("parallax_err", float),
        ("G_mag", float),
        ("V_mag", float),
        ("BP_mag", float),
        ("RP_mag", float),
        ("ruwe", float),
        ("pe", float),
        ("pe_err", float),
        ("period", float),
        ("period_err", float),
        ("teff_sys", float),
        ("teff_sys_err", float),
        ("logg_sys", float),
        ("logg_sys_err", float),
    ]

    target_units = {
        "ra": u.deg,
        "dec": u.deg,
        "parallax": u.mas,
        "parallax_err": u.mas,
        "G_mag": u.mag,
        "V_mag": u.mag,
        "BP_mag": u.mag,
        "RP_mag": u.mag,
        "period": u.d,
        "period_err": u.d,
        "teff_sys": u.K,
        "teff_sys_err": u.K,
    }

    # Start with the data in a structured array: helpful as we don't need to use units
    # at this point and we can also use Data Wrangler on it for diagnostics.
    target_data = np.empty(shape=(len(target_configs), ), dtype=target_dtype)

    # Where possible, set the initial numerical values to nan to indicate missing data. Any
    # int fields will be masked later, once we have a QTable, as we can only assign nan to floats.
    for col_name in [n for n in target_data.dtype.names if target_data[n].dtype in [float]]:
        target_data[col_name] = np.nan

    # Basic information
    for ix, (target, target_config) in enumerate(target_configs.items()):
        if target.isnumeric():
            target = f"TIC {int(target):d}"
        target_data[ix]["target"] = target
        target_data[ix]["search_term"] = target_config.get("search_term", target)

    # Get the basic published information from SIMBAD
    if (_tbl := simbad.query_objects(target_data["search_term"])) and len(_tbl) == len(target_data):
        _id_pattern = re.compile(r"(Gaia DR3|V\*|TIC|HD|HIP|2MASS)\s+(.+?(?=\||$))", re.IGNORECASE)
        for _trow, _srow in zip(target_data, _tbl):
            ids = np.array(_id_pattern.findall(_srow["ids"]), [("type", "O"), ("id", "O")])
            _trow["tics"] = "|".join(f"{i}" for i in ids[ids["type"]=="TIC"]["id"])
            if "Gaia DR3" in ids["type"]:
                _trow["gaia_dr3_id"] = int(ids[ids["type"]=="Gaia DR3"][0]["id"])
            _trow["main_id"] = _srow["main_id"]

            for (_tfield, _sfield) in [
                ("ra", "ra"), ("dec", "dec"), ("parallax", "plx_value"),
                ("spt", "sp_type"), ("G_mag", "G"), ("V_mag", "V")
            ]:
                if _sfield in _srow.colnames and (_val := _srow[_sfield]):
                    _trow[_tfield] = _val

    # # Augment the basic information from Gaia DR3 (where target is in DR3)
    _AQL = "SELECT source_id, ra, dec, parallax, parallax_error, ruwe, " \
        + "phot_g_mean_mag, phot_bp_mean_mag, phot_rp_mean_mag, teff_gspphot, logg_gspphot " \
        + "FROM gaiadr3.gaia_source_lite " \
        + f"WHERE source_id in ({','.join(f'{i:d}' for i in target_data['gaia_dr3_id'] if i)})"
    if _job := Gaia.launch_job(_AQL):
        for _srow in _job.get_results():
            _trow_ix, = np.where(target_data["gaia_dr3_id"] == _srow["source_id"])
            if len(_trow_ix):
                for (_tfield, _sfield) in [
                    ("ra", "ra"), ("dec", "dec"),
                    ("parallax", "parallax"), ("parallax_err", "parallax_error"),
                    ("G_mag", "phot_g_mean_mag"), ("BP_mag", "phot_bp_mean_mag"),
                    ("RP_mag", "phot_rp_mean_mag"), ("ruwe", "ruwe")
                ]:
                    if _sfield in _srow.colnames and (_val := _srow[_sfield]):
                        target_data[_tfield][_trow_ix] = _val

    # These don't support bulk query by id
    for _trow in target_data:
        # Lookup ephemeris information primarily from TESS-ebs
        if _tbl := tess_ebs_catalog.query_object(_trow["search_term"]):
            _trow["pe"] = _tbl[0]["BJD0"][0]
            _trow["period"] = _tbl[0]["Per"][0]

        with catch_warnings(category=UserWarning):
            filterwarnings("ignore", message="Warning: converting a masked element to nan.")
            # Get published system effective temperature/logg. start with an estimate based on SpT
            _teff_sys = pipeline.get_teff_from_spt(_trow["spt"]) or ufloat(1e4, 7e3)
            _trow["teff_sys"], _trow["teff_sys_err"] = _teff_sys.n, _teff_sys.s
            _trow["logg_sys"], _trow["logg_sys_err"] = 4.0, np.nan
            if _tbl := tic_catalog.query_object(_trow["search_term"], radius=0.1 * u.arcsec):
                if _srow := _tbl[0][f"{_tbl[0]['TIC']}" in _trow["tics"]]:
                    if not np.ma.is_masked(_srow["Teff"]) \
                        and _teff_sys.n - _teff_sys.s < _srow["Teff"] < _teff_sys.n + _teff_sys.s:
                        _trow["teff_sys"] = _srow["Teff"]
                        _trow["teff_sys_err"] = _srow["s_Teff"]
                    if not np.ma.is_masked(_srow["logg"]):
                        _trow["logg_sys"] = _srow["logg"]
                        _trow["logg_sys_err"] = _srow["s_logg"]

    # Finally we convert to an astropy Masked QTable
    targets = QTable(target_data, dtype=target_data.dtype, masked=True, units=target_units)

    # Have to mask values <=0 for gaia_dr3_id as I cannot assign np.nan (for no data/null) to an int
    targets["gaia_dr3_id"] = np.ma.masked_where(targets["gaia_dr3_id"] <= 0, targets["gaia_dr3_id"])
    for col_name in [n for n in targets.columns if targets[n].dtype in [int, float]]:
        targets[col_name] = np.ma.masked_invalid(targets[col_name])

    print(targets)

    save_file = Path(f"./drop/{args.targets_file.stem}.vot")
    targets.write(save_file, format="votable", overwrite=True)

    # Verify the save by reloading
    targets2 = parse_single_table(save_file).to_table()
    with open(save_file.parent / f"{save_file.stem}.diff", mode="w", encoding="utf8") as df:
        if not report_diff_values(targets, targets2, df):
            # We should only see differences around empty/null/nan representation
            print(f"Save verfication indicated differences. Check the diff file: {df.name}")
