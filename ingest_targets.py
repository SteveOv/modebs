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
from uncertainties import ufloat, nominal_value
from astroquery.simbad import Simbad
from astroquery.gaia import Gaia
import numpy as np

from libs import pipeline
from libs.iohelpers import Tee
from libs.targets import Targets
from libs.pipeline_dal import QTableFileDal

THIS_STEM = Path(getsourcefile(lambda: 0)).stem


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Pipeline stage 1: ingest of targets.")
    ap.add_argument(dest="targets_file", type=Path, metavar="TARGETS_FILE",
                    help="json file containing the details of the targets to ingest")
    ap.add_argument("-fo", "--force-overwrite", dest="force_overwrite", action="store_true",
                    required=False, help="force the overwritting of any existing ingest found")
    ap.set_defaults(force_overwrite=False, batch_size=20)
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
        print("\n\n============================================================")
        print(f"Started {THIS_STEM} at {datetime.now():%Y-%m-%d %H:%M:%S%z %Z}")
        print("============================================================")
        print(f"\nThe targets configuration file:   {args.targets_file}")
        print(f"Directory for data, logs & plots: {drop_dir}")

        dal = QTableFileDal(args.working_set_file)
        targets_config = Targets(args.targets_file)
        print(f"Read in the configuration from '{args.targets_file.name}'",
              f"which contains {targets_config.count()} target(s) not excluded.")


        print("\nSetting up a storage row and search_term for each target.")
        for ix, config in enumerate(targets_config.iterate_known_targets()):
            if (target_id := config.target_id).isnumeric():
                search_term = config.get("search_term", f"TIC {int(target_id):d}")
            else:
                search_term = config.get("search_term", target_id).strip()
                if not search_term.startswith("TIC"):
                    search_term = "V* " + search_term
            dal.write_values(target_id, search_term=search_term, morph=0.5, phiS=0.5,
                             fitted_lcs=False, fitted_sed=False, fitted_masses=False)


        # Get the basic published information from SIMBAD (keyed on search_term). We do batched
        # queries and code is dependent on SIMBAD returning the rows in the requested order.
        print("\nQuerying SIMBAD in batches for id, SpT & coordinate data.")
        simbad = Simbad()
        simbad.add_votable_fields("parallax", "sp", "ids")
        id_patt = re.compile(r"(Gaia DR3|V\*|TIC|HD|HIP|2MASS)\s+(.+?(?=\||$))", re.IGNORECASE)
        st_index = {m: t for m, t in dal.yield_values("search_term", dal.key_name) if m is not None}
        for sterms in pipeline.grouper(st_index.keys(), size=args.batch_size, fillvalue=None):
            # zip strict so we get ValueError if not same len as the sterms
            sterms = [m for m in sterms if m is not None]
            for sterm, srow in zip(sterms, simbad.query_objects(sterms), strict=True):
                target_id = st_index[sterm]
                ids = np.array(id_patt.findall(srow["ids"]), [("type", "O"), ("id", "O")])
                params = {
                    "tics": "|".join(f"{i}" for i in ids[ids["type"]=="TIC"]["id"]),
                    ** { col: srow[scol] for (col, scol) in [("ra", "ra"),
                                                            ("dec", "dec"),
                                                            ("parallax", "plx_value"),
                                                            ("coords_source", "plx_bibcode"),
                                                            ("spt", "sp_type")]
                                                        if scol in srow.colnames }
                }

                if any(dr3_mask := ids["type"]=="Gaia DR3"):
                    params["gaia_dr3_id"] = int(ids[dr3_mask][0]["id"])
                dal.write_values(target_id, **params)


        # Augment the basic information from Gaia DR3 (where target is in DR3).
        # Gaia DR3 queries are keyed on the gaia_dr3_id from above against the source_id field.
        print("\nQuerying Gaia DR3 in batches for coordinates and ruwe data.")
        gt_index = {g: t for g, t in dal.yield_values("gaia_dr3_id", dal.key_name) if g is not None}
        for gids in pipeline.grouper(gt_index.keys(), size=args.batch_size, fillvalue=None):
            AQL = f"SELECT TOP {args.batch_size*2} source_id, ra, dec, parallax, parallax_error, " \
                + "ruwe, teff_gspphot, logg_gspphot FROM gaiadr3.gaia_source_lite " \
                + f"WHERE source_id in ({','.join(f'{i:d}' for i in gids if i is not None)})"
            for srow in Gaia.launch_job(AQL).get_results():
                if (target_id := gt_index.get(srow["source_id"], None)) is not None:
                    params = { "ruwe": srow["ruwe"]}
                    if all((srow[k] or 0) != 0 for k in ["ra", "dec", "parallax"]):
                        params |= {
                            "ra": srow["ra"],
                            "dec": srow["dec"],
                            "parallax": srow["parallax"],
                            "parallax_err": srow["parallax_error"],
                            "coords_source": "2022yCat.1355....0G", # GaiaDR3 Part 1. Main source
                        }
                    dal.write_values(target_id, **params)


        # Highlight missing coords as this will inhibit accounting for extinction when fitting SED
        print()
        for target_id, ra, dec, par, warn_msgs in \
                dal.yield_values(dal.key_name, "ra", "dec", "parallax", "warnings"):
            if any(v is None for v in [ra, dec, par]) or nominal_value(par) == 0:
                warn_msgs = (warn_msgs or "").split(";") + ["coords incomplete"]
                dal.write_values(target_id,
                                 warnings=";".join(w for w in dict.fromkeys(warn_msgs) if len(w)))
                print(f"** Warning {target_id} coords incomplete: ra={ra},dec={dec},parallax={par}")


        print("\nGathering ephemeris, morphology and eclipse data.")
        ephem_keys = ["t0", "period", "morph", "widthP", "depthP", "widthS", "depthS", "phiS"]
        for target_id, tics, warn_msgs in dal.yield_values(dal.key_name, "tics", "warnings"):
            warn_msgs = (dal.read_values(target_id, "warnings") or "").split(";")
            params = {}

            config = targets_config.get(target_id)
            if len(ephem_config_keys := [k for k in ephem_keys if config.has_value(k)]) > 0:
                print(f"{target_id}: copying ephemeris values for {ephem_config_keys} from config")
                for k in ephem_config_keys:
                    if config.has_value(k_err := f"{k}_err"):
                        params[k] = ufloat(config.get(k), config.get(k_err, 0))
                    else:
                        params[k] = config.get(k)

            if missing_ephem_keys := [k for k in ephem_keys if k not in params]:
                print(f"** Warning the following ephemeris values were not in {target_id} config:",
                      ",".join(k for k in missing_ephem_keys))
                warn_msgs += ["incomplete ephemeris"]

            dal.write_values(target_id, **params,
                             warnings=";".join(w for w in dict.fromkeys(warn_msgs) if len(w)))


        print("\nApplying any non-ephemeris overrides from config.")
        over_keys = ["parallax", "ra", "dec", "Teff_sys", "logg_sys"]
        for config in targets_config.iterate_known_targets():
            with_overs_keys = list(c for c in over_keys if config.has_value(c))
            if len(with_overs_keys) > 0:
                target_id = config.target_id
                print(f"{target_id}: copying override(s) for {with_overs_keys} from config.")
                params = { }
                for k in with_overs_keys:
                    if config.has_value(k_err := f"{k}_err"):
                        params[k] = ufloat(config.get(k), config.get(k_err, 0))
                    else:
                        params[k] = config.get(k)
                dal.write_values(target_id, **params)


        print("\n\n============================================================")
        print(f"Completed {THIS_STEM} at {datetime.now():%Y-%m-%d %H:%M:%S%z %Z}")
        print("============================================================")
