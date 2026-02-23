""" Pipeline Stage 1 - ingesting targets """
# pylint: disable=no-member
from inspect import getsourcefile
from pathlib import Path
import warnings
import sys
import re
import argparse
from datetime import datetime
from contextlib import redirect_stdout

# pylint: disable=line-too-long, wrong-import-position
warnings.filterwarnings("ignore", "Using UFloat objects with std_dev==0 may give unexpected results.", category=UserWarning)
from uncertainties import ufloat
from astroquery.vizier import Vizier
from astroquery.simbad import Simbad
from astroquery.gaia import Gaia
import numpy as np

from libs import catalogues
from libs.iohelpers import Tee
from libs.targets import Targets
from libs.pipeline_dal import QTableFileDal

THIS_STEM = Path(getsourcefile(lambda: 0)).stem


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
    args.working_set_file = drop_dir / "working-set.table"


    if not args.force_overwrite and args.working_set_file.exists():
        resp = input(f"** Warning: output data exists in '{drop_dir}'. Continue & overwrite y/N? ")
        if resp.strip().lower() not in ["y", "yes"]:
            sys.exit()

    # Rather nuclear option; replace any existing working set. Maybe more nuanced approach later.
    drop_dir.mkdir(parents=True, exist_ok=True)
    args.working_set_file.unlink(missing_ok=True)


    with redirect_stdout(Tee(open(drop_dir / f"{THIS_STEM}.log", "w", encoding="utf8"))):
        print(f"Started at {datetime.now():%Y-%m-%d %H:%M:%S%z %Z}")

        dal = QTableFileDal(args.working_set_file)
        targets_config = Targets(args.targets_file)
        print(f"\nRead in the configuration from '{args.targets_file}'",
              f"which contains {targets_config.count()} target(s) not excluded.")

        # Used to get the extended information published for each target
        simbad = Simbad()
        simbad.add_votable_fields("parallax", "allfluxes", "sp", "ids")
        tic_catalog = Vizier(catalog="IV/39/tic82", row_limit=10)
        gaia_tbosb_catalog = Vizier(catalog="I/357/tbosb2", row_limit=1)


        print("\nSetting up storage row for these target(s).")
        for ix, config in enumerate(targets_config.iterate_known_targets()):
            if (target_id := config.target_id).isnumeric():
                search_term = config.get("search_term", f"TIC {int(target_id):d}")
            else:
                search_term = config.get("search_term", target_id)
            dal.write_values(target_id, main_id=search_term, morph=0.5, phiS=0.5,
                             fitted_lcs=False, fitted_seds=False, fitted_masses=False)


        # Get the basic published information from SIMBAD. We do a mass query
        # and code is dependent on SIMBAD returning the rows in the requested order.
        print("\nQuerying SIMBAD for id, SpT, mag & coordinate data.")
        id_patt = re.compile(r"(Gaia DR3|V\*|TIC|HD|HIP|2MASS)\s+(.+?(?=\||$))", re.IGNORECASE)
        target_sterms = np.array(list(dal.yield_values(dal.key_name, "main_id"))).T
        if (tbl := simbad.query_objects(target_sterms[1])) and len(tbl) == target_sterms.shape[1]:
            for target_id, srow in zip(target_sterms[0], tbl):
                ids = np.array(id_patt.findall(srow["ids"]), [("type", "O"), ("id", "O")])
                params = {
                    "main_id": srow["main_id"],
                    "tics": "|".join(f"{i}" for i in ids[ids["type"]=="TIC"]["id"]),
                    "gaia_dr3_id": int(ids[ids["type"]=="Gaia DR3"][0]["id"]),
                    ** { col: srow[scol] for col, scol in [("ra", "ra"), ("dec", "dec"),
                                                           ("parallax", "plx_value"),
                                                           ("spt", "sp_type"),
                                                           ("G_mag", "G"), ("V_mag", "V")]
                                                        if scol in srow.colnames }
                }
                dal.write_values(target_id, **params)


        # Augment the basic information from Gaia DR3 (where target is in DR3)
        print("\nQuerying Gaia DR3 for coordinates, mags and ruwe data.")
        target_sterms = np.array(list(dal.yield_values(dal.key_name, "gaia_dr3_id"))).T
        AQL = "SELECT source_id, ra, dec, parallax, parallax_error, ruwe, " \
            + "phot_g_mean_mag, phot_bp_mean_mag, phot_rp_mean_mag, teff_gspphot, logg_gspphot " \
            + "FROM gaiadr3.gaia_source_lite " \
            + f"WHERE source_id in ({','.join(f'{i:d}' for i in target_sterms[1] if i)})"
        if job := Gaia.launch_job(AQL):
            for srow in job.get_results():
                target_id = target_sterms[0][target_sterms[1] == srow["source_id"]][0]
                params = { col: srow[scol] for (col, scol) in [("ra", "ra"),
                                                                ("dec", "dec"),
                                                                ("parallax", "parallax"),
                                                                ("parallax_err", "parallax_error"),
                                                                ("G_mag", "phot_g_mean_mag"),
                                                                ("BP_mag", "phot_bp_mean_mag"),
                                                                ("RP_mag", "phot_rp_mean_mag"),
                                                                ("ruwe", "ruwe")]}
                dal.write_values(target_id, **params)


        # Lookup ephemeris information primarily from TESS-ebs
        # but also config overrides and estimate eclipse widths if missing
        print("\nQuerying TESS-ebs for ephemeris data.")
        ephem_keys = ["t0", "period", "morph", "widthP", "depthP", "widthS", "depthS", "phiS"]
        for target_id, tics in dal.yield_values(dal.key_name, "tics"):
            config = targets_config.get(target_id)
            params = catalogues.query_tess_ebs_ephemeris(tics.split("|"), config.period_factor)
            if params is None:
                print(f"{target_id}: no TESS-ebs ephmeris so override values will be required.")
                params = { }

            # Special case: if no secondary data and we know period is doubled. This is likely to
            # be a system with very similar primary and secondary eclipses, so copy values over.
            if config.period_factor == 2 \
                    and all(params[v] is None for v in ["widthS", "depthS", "phiS"]):
                print(f"{target_id}: No secondary data in TESS-ebs & the period is to be doubled.",
                      "Copying primary data to the secondary (assume similar eclipses & phiS=0.5).")
                params["widthS"] = params["widthP"]
                params["depthS"] = params["depthP"]
                params["phiS"] = 0.5

            ephem_keys_overs = [k for k in ephem_keys if config.has_value(k)]
            if len(ephem_keys_overs) > 0:
                print(f"{target_id}: copying ephemeris overrides of {ephem_keys_overs} from config")
                for k in ephem_keys_overs:
                    if k in ["t0", "period"]:
                        params[k] = ufloat(config.get(k), config.get(f"{k}_err", 0))
                    else:
                        params[k] = config.get(k)

            if params.get("widthP", None) is None or params.get("widthS", None) is None:
                widthP, _ = catalogues.estimate_eclipse_widths_from_morphology(params["morph"])
                params["widthP"] = params["widthS"] = widthP
                print(f"{target_id}: no eclipse widths found, so an estimated a value of",
                    f"{params['widthP']:.3f} is being used for both (based on morph).")

            dal.write_values(target_id, **params)


        print("\nApplying any non-ephemeris overrides from config.")
        over_keys = ["Teff_sys", "logg_sys"]
        for config in targets_config.iterate_known_targets():
            with_overs_keys = list(c for c in over_keys if config.has_value(c))
            if len(with_overs_keys) > 0:
                target_id = config.target_id
                print(f"{target_id}: copying override(s) for {with_overs_keys} from config.")
                params = { }
                for k in with_overs_keys:
                    params[k] = config.get(k)
                    if k in ["t0", "period"]:
                        params[f"{k}_err"] = config.get(f"{k}_err", 0)
                dal.write_values(target_id, **params)


        print(f"\nCompleted at {datetime.now():%Y-%m-%d %H:%M:%S%z %Z}")
