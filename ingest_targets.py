""" Pipeline Stage 1 - ingesting targets """
# pylint: disable=no-member
from pathlib import Path
import re
import warnings
import argparse
from contextlib import redirect_stdout

import numpy as np
import astropy.units as u
from astropy.table import QTable
from astropy.utils.diff import report_diff_values

from astroquery.vizier import Vizier
from astroquery.simbad import Simbad
from astroquery.gaia import Gaia

# pylint: disable=line-too-long, wrong-import-position
warnings.filterwarnings("ignore", "Using UFloat objects with std_dev==0 may give unexpected results.", category=UserWarning)
from uncertainties import ufloat

from libs import pipeline, catalogues
from libs.iohelpers import Tee
from libs.targets import Targets


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Pipeline stage 1: ingest of targets.")
    ap.add_argument("-i", "--input-file", dest="targets_file", type=Path, required=False,
                    help="json file containing the details of the targets to ingest")
    # We use a None in model_files as indication to pull in the default model under ebop_maven/data
    ap.set_defaults(targets_file=Path("./config/formal-test-explicit-targets.json"))
    args = ap.parse_args()

    drop_dir = Path.cwd() / "./drop"
    log_file = drop_dir / f"{args.targets_file.stem}.ingest.log"
    with redirect_stdout(Tee(open(log_file, "w", encoding="utf8"))):

        targets_config = Targets("config/formal-test-explicit-targets.json")
        print(f"Read in the configuration for {targets_config.count()} target(s).")

        # Used to get the extended information published for each target
        simbad = Simbad()
        simbad.add_votable_fields("parallax", "allfluxes", "sp", "ids")
        tic_catalog = Vizier(catalog="IV/39/tic82", row_limit=10)
        gaia_tbosb_catalog = Vizier(catalog="I/357/tbosb2", row_limit=1)

        target_dtype = [
            # SIMBAD and IDs
            ("target", "<U14"),
            ("main_id", "<U20"),
            ("tics", "<U40"),
            ("gaia_dr3_id", int),
            ("spt", "<U20"),
            # Gaia DR3 (coords falling back on SIMBAD)
            ("ra", float),
            ("ra_err", float),
            ("dec", float),
            ("dec_err", float),
            ("parallax", float),
            ("parallax_err", float),
            ("G_mag", float),
            ("V_mag", float),
            ("BP_mag", float),
            ("RP_mag", float),
            ("ruwe", float),
            # TESS-ebs
            ("t0", float),
            ("t0_err", float),
            ("period", float),
            ("period_err", float),
            ("morph", float),
            ("widthP", float),
            ("widthS", float),
            ("depthP", float),
            ("depthS", float),
            ("phiS", float),
            # TESS
            ("Teff_sys", float),
            ("Teff_sys_err", float),
            ("logg_sys", float),
            ("logg_sys_err", float),
            # JKTEBOP lightcurve fitting i/o params (initially from EBOP MAVEN preds)
            ("rA_plus_rB", float),
            ("rA_plus_rB_err", float),
            ("k", float),
            ("k_err", float),
            ("J", float),
            ("J_err", float),
            ("ecosw", float),
            ("ecosw_err", float),
            ("esinw", float),
            ("esinw_err", float),
            ("bP", float),
            ("bP_err", float),
            ("inc", float),
            ("inc_err", float),
            # JKTEBOP lightcurve fitting i/o params (from other sources)
            ("L3", float),
            ("L3_err", float),
            # JKTEBOP lightcurve fitting output params
            ("LR", float),
            ("LR_err", float),
            ("TeffR", float),
            ("TeffR_err", float),
            # SED fitting i/o params
            ("Av", float),
            ("TeffA", float),
            ("TeffA_err", float),
            ("TeffB", float),
            ("TeffB_err", float),
            ("loggA", float),
            ("loggA_err", float),
            ("loggB", float),
            ("loggB_err", float),
            ("RA", float),
            ("RA_err", float),
            ("RB", float),
            ("RB_err", float),
            ("dist", float),
            ("dist_err", float),
            # Mass fitting i/o params
            ("M_sys", float),
            ("M_sys_err", float),
            ("a", float),
            ("a_err", float),
            ("MA", float),
            ("MA_err", float),
            ("MB", float),
            ("MB_err", float),
            ("log_age", float),
            ("log_age_err", float),
        ]

        target_units = {
            "ra": u.deg,
            "ra_err": u.deg,
            "dec": u.deg,
            "dec_err": u.deg,
            "parallax": u.mas,
            "parallax_err": u.mas,
            "G_mag": u.mag,
            "V_mag": u.mag,
            "BP_mag": u.mag,
            "RP_mag": u.mag,
            "period": u.d,
            "period_err": u.d,
            "Teff_sys": u.K,
            "Teff_sys_err": u.K,
            "inc": u.deg,
            "inc_err": u.deg,
            "TeffA": u.K,
            "TeffA_err": u.K,
            "TeffB": u.K,
            "TeffB_err": u.K,
            # "loggA": u.dex,
            # "loggA_err": u.dex,
            # "loggB": u.dex,
            # "loggB_err": u.dex,
            "RA": u.solRad,
            "RA_err": u.solRad,
            "RB": u.solRad,
            "RB_err": u.solRad,
            "dist": u.pc,
            "dist_err": u.pc,
            "M_sys": u.solMass,
            "M_sys_err": u.solMass,
            "a": u.AU,
            "a_err": u.AU,
            "MA": u.solMass,
            "MA_err": u.solMass,
            "MB": u.solMass,
            "MB_err": u.solMass,
            # "log_age": u.Dex(u.yr),
            # "log_age_err": u.Dex(u.yr),
        }

        # Start with the data in a structured array: helpful as we don't need to use units
        # at this point and we can also use Data Wrangler on it for diagnostics.
        target_data = np.empty(shape=(targets_config.count(), ), dtype=target_dtype)

        # Where possible, set the initial numerical values to nan to indicate missing data. Any
        # int fields will be masked later, once we have a QTable, as we can only assign nan to floats.
        for col_name in [n for n in target_data.dtype.names if target_data[n].dtype in [float]]:
            target_data[col_name] = np.nan

        print("Getting basic information from config for these target(s).")
        for ix, target_config in enumerate(targets_config.iterate_known_targets()):
            target = target_config.target_id
            if target.isnumeric():
                target = f"TIC {int(target):d}"
            target_data[ix]["target"] = target
            target_data[ix]["main_id"] = target_config.get("search_term", target)

        # Get the basic published information from SIMBAD
        print("Querying SIMBAD for id, SpT, mag & coordinate data.")
        if (tbl := simbad.query_objects(target_data["main_id"])) and len(tbl) == len(target_data):
            id_pattern = re.compile(r"(Gaia DR3|V\*|TIC|HD|HIP|2MASS)\s+(.+?(?=\||$))", re.IGNORECASE)
            for trow, srow in zip(target_data, tbl):
                ids = np.array(id_pattern.findall(srow["ids"]), [("type", "O"), ("id", "O")])
                trow["tics"] = "|".join(f"{i}" for i in ids[ids["type"]=="TIC"]["id"])
                if "Gaia DR3" in ids["type"]:
                    trow["gaia_dr3_id"] = int(ids[ids["type"]=="Gaia DR3"][0]["id"])
                trow["main_id"] = srow["main_id"]

                for (tfield, sfield) in [
                    ("ra", "ra"), ("dec", "dec"), ("parallax", "plx_value"),
                    ("spt", "sp_type"), ("G_mag", "G"), ("V_mag", "V")
                ]:
                    if sfield in srow.colnames and (_val := srow[sfield]):
                        trow[tfield] = _val

        # # Augment the basic information from Gaia DR3 (where target is in DR3)
        print("Querying Gaia DR3 for coordinates, mags and ruwe data.")
        AQL = "SELECT source_id, ra, dec, parallax, parallax_error, ruwe, " \
            + "phot_g_mean_mag, phot_bp_mean_mag, phot_rp_mean_mag, teff_gspphot, logg_gspphot " \
            + "FROM gaiadr3.gaia_source_lite " \
            + f"WHERE source_id in ({','.join(f'{i:d}' for i in target_data['gaia_dr3_id'] if i)})"
        if job := Gaia.launch_job(AQL):
            for srow in job.get_results():
                trow_ix, = np.where(target_data["gaia_dr3_id"] == srow["source_id"])
                if len(trow_ix):
                    for (tfield, sfield) in [
                        ("ra", "ra"), ("ra_err", "ra_error"),
                        ("dec", "dec"), ("dec_err", "dec_error"),
                        ("parallax", "parallax"), ("parallax_err", "parallax_error"),
                        ("G_mag", "phot_g_mean_mag"), ("BP_mag", "phot_bp_mean_mag"),
                        ("RP_mag", "phot_rp_mean_mag"), ("ruwe", "ruwe")
                    ]:
                        if sfield in srow.colnames and (_val := srow[sfield]):
                            target_data[tfield][trow_ix] = _val

        # Lookup ephemeris information primarily from TESS-ebs
        print("Querying TESS-ebs for ephemeris data.")
        for trow in target_data:
            period_factor = targets_config.get(trow["target"]).period_factor
            tebs = catalogues.query_tess_ebs_ephemeris(trow["tics"].split("|"), period_factor)
            if tebs is not None:
                trow["t0"] = tebs["t0"].n
                trow["t0_err"] = tebs["t0"].s
                trow["period"] = tebs["period"].n
                trow["period_err"] = tebs["period"].s
                for k in ["morph", "widthP", "depthP", "widthS", "depthS", "phiS"]:
                    if tebs[k] is not None:
                        trow[k] = tebs[k]
                if tebs["widthP"] is None or tebs["widthS"] is None:
                    durP, _ = catalogues.estimate_eclipse_durations_from_morphology(trow["morph"],
                                                                                    trow["period"])
                    trow["widthP"] = trow["widthS"] = durP / trow["period"]
                    print(f"No eclipse widths for {trow['target']} so estimated a value of",
                        f"{trow['widthP']:.3f} for both, based on the morph metric.")
            else:
                print(f"No TESS-ebs ephmeris for {trow['target']} so relying on config values.")

        # Applying any target config overrides
        print("Applying any target specific overrides from config.")
        for ix, target_config in enumerate(targets_config.iterate_known_targets()):
            for k in ["t0", "period", "morph", "Teff_sys", "logg_sys",
                      "widthP", "widthS", "depthP",  "depthS", "phiP", "phiS"]:
                if val := target_config.get(k, None) is not None:
                    target_data[k] = val

        # Finally we convert to an astropy Masked QTable
        print("Generating a QTable for the data.")
        targets = QTable(target_data, dtype=target_data.dtype, masked=True, units=target_units)

        # Have to mask values <=0 for gaia_dr3_id as I cannot assign np.nan (for no data/null) to an int
        mask_int_mask = targets["gaia_dr3_id"] <= 0
        print(f"Masking data for {sum(mask_int_mask)} rows where Gaia DR3 id is missing.")
        targets["gaia_dr3_id"] = np.ma.masked_where(mask_int_mask, targets["gaia_dr3_id"])
        for col_name in [n for n in targets.columns if targets[n].dtype in [int, float]]:
            targets[col_name] = np.ma.masked_invalid(targets[col_name])

        save_file = drop_dir / f"{args.targets_file.stem}.table"
        SAVE_FORMAT = "votable" # Text based, so editable
        print(f"Saving table to {save_file} in the '{SAVE_FORMAT}' format.")
        save_file.parent.mkdir(parents=True, exist_ok=True)
        targets.write(save_file, format=SAVE_FORMAT, overwrite=True)

        # Verify the save by reloading
        print("Basic verification of newly created data file by re-opening without format prompt.")
        targets2 = QTable.read(save_file)
        # print(targets2)

        # We should only see differences around empty/null/nan representation
        print("Producing a difference report on the newly created data file")
        with open(save_file.parent / f"{save_file.stem}.diff", mode="w", encoding="utf8") as df:
            if not report_diff_values(targets, targets2, df):
                print(f"Save verfication indicated differences. Check the diff file: {df.name}")
