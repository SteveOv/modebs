""" Pipeline Stage 1 - ingesting targets """
# pylint: disable=no-member
from pathlib import Path
import sys
import re
import argparse
from datetime import datetime
from contextlib import redirect_stdout

import numpy as np
import astropy.units as u
from astropy.table import QTable
from astropy.utils.diff import report_diff_values

from astroquery.vizier import Vizier
from astroquery.simbad import Simbad
from astroquery.gaia import Gaia

from libs import catalogues
from libs.iohelpers import Tee
from libs.targets import Targets


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Pipeline stage 1: ingest of targets.")
    ap.add_argument("-tf", "--targets-file", dest="targets_file", type=Path, required=False,
                    help="json file containing the details of the targets to ingest")
    ap.add_argument("-fo", "--force-overwrite", dest="force_overwrite", action="store_true",
                    required=False, help="force the overwritting of any existing ingest found")
    ap.set_defaults(targets_file=Path("./config/plato-lops2-tess-ebs-explicit-targets.json"),
                    force_overwrite=False)
    args = ap.parse_args()
    drop_dir = Path.cwd() / f"drop/{args.targets_file.stem}"
    args.output_file = drop_dir / "targets.table"

    if not args.force_overwrite and args.output_file.exists():
        resp = input(f"** Warning: output data exists in '{drop_dir}'. Continue & overwrite y/N? ")
        if resp.strip().lower() not in ["y", "yes"]:
            sys.exit()
    drop_dir.mkdir(parents=True, exist_ok=True)

    with redirect_stdout(Tee(open(drop_dir / "ingest.log", "w", encoding="utf8"))):
        print(f"Started at {datetime.now():%Y-%m-%d %H:%M:%S%z %Z}")

        targets_config = Targets(args.targets_file)
        print(f"\nRead in the configuration from '{args.targets_file}'",
              f"which contains {targets_config.count()} target(s) not excluded.")

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
            # Progress flags
            ("fit_lcs", bool),
            ("fit_radii", bool),
            ("fit_masses", bool),
            ("warnings", object),
            ("errors", object),
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

        # Where possible, set the initial numerical values to nan to indicate missing data. Any int
        # fields will be masked later, once we have a QTable, as we can only assign nan to floats.
        for col_name in [n for n in target_data.dtype.names if target_data[n].dtype in [float]]:
            target_data[col_name] = np.nan

        print("\nGetting basic information from config for these target(s).")
        for ix, target_config in enumerate(targets_config.iterate_known_targets()):
            target = target_config.target_id
            if target.isnumeric():
                target = f"TIC {int(target):d}"
            row = target_data[ix]
            row["target"] = target
            row["main_id"] = target_config.get("search_term", target)
            row["morph"] = 0.5
            row["phiS"] = 0.5
            row["fit_lcs"] = row["fit_radii"] = row["fit_masses"] = True
            for k in ["Teff_sys", "logg_sys"]:
                row[k] = target_config.get(k, None)

        # Get the basic published information from SIMBAD
        print("\nQuerying SIMBAD for id, SpT, mag & coordinate data.")
        if (tbl := simbad.query_objects(target_data["main_id"])) and len(tbl) == len(target_data):
            id_patt = re.compile(r"(Gaia DR3|V\*|TIC|HD|HIP|2MASS)\s+(.+?(?=\||$))", re.IGNORECASE)
            for trow, srow in zip(target_data, tbl):
                ids = np.array(id_patt.findall(srow["ids"]), [("type", "O"), ("id", "O")])
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
        print("\nQuerying Gaia DR3 for coordinates, mags and ruwe data.")
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
        print("\nQuerying TESS-ebs for ephemeris data.")
        nom_ephem_keys = ["morph", "widthP", "depthP", "widthS", "depthS", "phiS"]
        for trow in target_data:
            target = trow["target"]
            target_config = targets_config.get(target)
            tebs = catalogues.query_tess_ebs_ephemeris(trow["tics"].split("|"),
                                                       target_config.period_factor)
            if tebs is not None:
                trow["t0"] = tebs["t0"].n
                trow["t0_err"] = tebs["t0"].s
                trow["period"] = tebs["period"].n
                trow["period_err"] = tebs["period"].s
                for k in nom_ephem_keys:
                    if tebs[k] is not None:
                        trow[k] = tebs[k]
            else:
                print(f"{target}: no TESS-ebs ephmeris so override values will be required.")

            config_keys = [k for k in ["t0","period"]+nom_ephem_keys if target_config.has_value(k)]
            if len(config_keys) > 0:
                print(f"{target}: copying ephemeris override(s) for {config_keys} from config.")
                for k in config_keys:
                    trow[k] = target_config.get(k)
                    if k in ["t0", "period"]:
                        trow[f"{k}_err"] = target_config.get(f"{k}_err", 0)

            if np.isnan(trow["widthP"]) or np.isnan(trow["widthS"]):
                widthP, _ = catalogues.estimate_eclipse_widths_from_morphology(trow["morph"])
                trow["widthP"] = trow["widthS"] = widthP
                print(f"{target}: no eclipse widths found, so an estimated a value of",
                    f"{trow['widthP']:.3f} is being used for both (based on morph).")

        # Finally we convert to an astropy Masked QTable
        print("\nGenerating a QTable for the data.")
        targets = QTable(target_data, dtype=target_data.dtype, masked=True, units=target_units)

        # Have to mask for gaia_dr3_id <=0 as I cannot assign np.nan (for no data/null) to an int
        mask_int_mask = targets["gaia_dr3_id"] <= 0
        print(f"\nMasking data for {sum(mask_int_mask)} rows where Gaia DR3 id is missing.")
        targets["gaia_dr3_id"] = np.ma.masked_where(mask_int_mask, targets["gaia_dr3_id"])
        for col_name in [n for n in targets.columns if targets[n].dtype in [int, float]]:
            targets[col_name] = np.ma.masked_invalid(targets[col_name])

        SAVE_FORMAT = "votable" # Text based, so editable
        print(f"\nSaving output to '{args.output_file}' in the '{SAVE_FORMAT}' format.")
        targets.write(args.output_file, format=SAVE_FORMAT, overwrite=True)

        # Verify the save by reloading
        print("\nBasic verification of new output file by re-opening without format prompt.")
        targets2 = QTable.read(args.output_file)
        # print(targets2)

        # We should only see differences around empty/null/nan representation
        with open(args.output_file.parent / "ingest.diff", mode="w", encoding="utf8") as df:
            print(f"\nProducing a difference report written to '{df.name}'. This also",
                  "acts as a human readable copy of the contents of the output file.")
            if not report_diff_values(targets, targets2, df):
                print("Ingest verfication indicated differences. Check the diff file.")

        print(f"\nCompleted at {datetime.now():%Y-%m-%d %H:%M:%S%z %Z}")
