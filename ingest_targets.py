#!/usr/bin/env python3
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
from libs.pipeline_dal import create_dal

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

    working_set_file = drop_dir / "working-set.table"
    if not args.force_overwrite and working_set_file.exists():
        resp = input(f"** Warning: output data exists in '{drop_dir}'. Continue & overwrite y/N? ")
        if resp.strip().lower() not in ["y", "yes"]:
            sys.exit()

    # Rather nuclear option; replace any existing working set. Maybe more nuanced approach later.
    drop_dir.mkdir(parents=True, exist_ok=True)
    working_set_file.unlink(missing_ok=True)


    with redirect_stdout(Tee(open(drop_dir / f"{THIS_STEM}.log", "w", encoding="utf8"))):
        print("\n\n============================================================")
        print(f"Started {THIS_STEM} at {datetime.now():%Y-%m-%d %H:%M:%S%z %Z}")
        print("============================================================")
        print(f"\nThe targets configuration file:   {args.targets_file}")
        print(f"Directory for data, logs & plots: {drop_dir}")

        targets_config = Targets(args.targets_file)
        print(f"Read in the configuration from '{args.targets_file.name}'",
              f"which contains {targets_config.count()} target(s) not excluded.")

        dal_kwargs = targets_config.get("dal_kwargs", {})
        dal_kwargs.setdefault("file", working_set_file)
        dal = create_dal(targets_config.get("dal_type", "QTableFileDal"), True, **dal_kwargs)

        print("\nSetting up a storage row and search_term for each target.")
        search_term_index = { }
        for ix, config in enumerate(targets_config.iterate_known_targets()):
            if (target_id := config.target_id).isnumeric():
                search_term = config.get("search_term", f"TIC {int(target_id):d}")
            else:
                search_term = config.get("search_term", target_id).strip()
                if not search_term.startswith("TIC"):
                    search_term = "V* " + search_term
            dal.add_row(key=target_id, search_term=search_term, morph=0.5, phiS=0.5,
                        fitted_lcs=False, fitted_sed=False, fitted_masses=False)
            search_term_index[search_term] = target_id

        # Get the basic published information from SIMBAD (keyed on search_term). We do batched
        # queries and code is dependent on SIMBAD returning the rows in the requested order.
        print("\nQuerying SIMBAD in batches for id, SpT & coordinate data.")
        simbad = Simbad()
        simbad.add_votable_fields("parallax", "sp", "ids")
        id_patt = re.compile(r"(Gaia DR3|V\*|TIC|HD|HIP|2MASS)\s+(.+?(?=\||$))", re.IGNORECASE)
        gaia_id_index = { }
        for sterms in pipeline.grouper(search_term_index.keys(), args.batch_size, fillvalue=None):
            # zip strict so we get ValueError if not same len as the sterms
            sterms = [m for m in sterms if m is not None]
            for sterm, srow in zip(sterms, simbad.query_objects(sterms), strict=True):
                target_id = search_term_index[sterm]
                ids = np.array(id_patt.findall(srow["ids"]), [("type", "O"), ("id", "O")])
                cols_and_values = {
                    "tics": "|".join(f"{i}" for i in ids[ids["type"]=="TIC"]["id"]),
                    ** { col: srow[scol] for (col, scol) in [("ra_coord", "ra"),
                                                            ("dec_coord", "dec"),
                                                            ("parallax", "plx_value"),
                                                            ("parallax_bibcode", "plx_bibcode"),
                                                            ("spt", "sp_type")]
                                                        if scol in srow.colnames }
                }

                if any(dr3_mask := ids["type"]=="Gaia DR3"):
                    gaia_id = int(ids[dr3_mask][0]["id"])
                    cols_and_values["gaia_dr3_id"] = gaia_id
                    gaia_id_index[gaia_id] = target_id

                dal.update_row(key=target_id, **cols_and_values)


        # Augment the basic information from Gaia DR3 (where target is in DR3).
        # Gaia DR3 queries are keyed on the gaia_dr3_id from above against the source_id field.
        print("\nQuerying Gaia DR3 in batches for coordinates and ruwe data.")
        for gids in pipeline.grouper(gaia_id_index.keys(), size=args.batch_size, fillvalue=None):
            AQL = f"SELECT TOP {args.batch_size*2} source_id, ra, dec, parallax, parallax_error, " \
                + "ruwe, teff_gspphot, logg_gspphot FROM gaiadr3.gaia_source_lite " \
                + f"WHERE source_id in ({','.join(f'{i:d}' for i in gids if i is not None)})"
            for srow in Gaia.launch_job(AQL).get_results():
                if (target_id := gaia_id_index.get(srow["source_id"], None)) is not None:
                    cols_and_values = { "ruwe": srow["ruwe"]}
                    if all((srow[k] or 0) != 0 for k in ["ra", "dec", "parallax"]):
                        cols_and_values |= {
                            "ra_coord": srow["ra"],
                            "dec_coord": srow["dec"],
                            "parallax": srow["parallax"],
                            "parallax_err": srow["parallax_error"],
                            "parallax_bibcode": "2022yCat.1355....0G", # GaiaDR3 Part 1. Main source
                        }

                    dal.update_row(key=target_id, **cols_and_values)


        # Highlight missing coords as this will inhibit accounting for extinction when fitting SED
        print()
        for row in dal.acquire_next_row():
            ra, dec, par = row.ra_coord, row.dec_coord, row.parallax
            if any(v is None for v in [ra, dec, par]) or nominal_value(par) == 0:
                row.append_warning("coords incomplete")
                print(f"** Warning {target_id} coords incomplete: ra={ra},dec={dec},parallax={par}")


        print("\nGathering ephemeris, morphology and eclipse data.")
        ephem_keys = ["t0", "period", "morph", "widthP", "depthP", "widthS", "depthS", "phiS"]
        for row  in dal.acquire_next_row():
            target_id = row.key
            cols_and_values = {}

            config = targets_config.get_target_config(target_id)
            if len(ephem_config_keys := [k for k in ephem_keys if config.has_value(k)]) > 0:
                print(f"{target_id}: copying ephemeris values for {ephem_config_keys} from config")
                for k in ephem_config_keys:
                    if config.has_value(k_err := f"{k}_err"):
                        cols_and_values[k] = ufloat(config.get(k), config.get(k_err, 0))
                    else:
                        cols_and_values[k] = config.get(k)

            if missing_ephem_keys := [k for k in ephem_keys if k not in cols_and_values]:
                print(f"** Warning the following ephemeris values were not in {target_id} config:",
                      ",".join(k for k in missing_ephem_keys))
                row.append_warning("incomplete ephemeris")

            row.set_values(**cols_and_values)


        print("\nApplying any non-ephemeris overrides from config.")
        over_keys = ["parallax", "ra", "dec", "Teff_sys", "logg_sys"]
        for config in targets_config.iterate_known_targets():
            with_overs_keys = list(c for c in over_keys if config.has_value(c))
            if len(with_overs_keys) > 0:
                target_id = config.target_id
                print(f"{target_id}: copying override(s) for {with_overs_keys} from config.")
                cols_and_values = { }
                for k in with_overs_keys:
                    if config.has_value(k_err := f"{k}_err"):
                        cols_and_values[k] = ufloat(config.get(k), config.get(k_err, 0))
                    else:
                        cols_and_values[k] = config.get(k)

                dal.update_row(key=target_id, **cols_and_values)


        print("\n\n============================================================")
        print(f"Completed {THIS_STEM} at {datetime.now():%Y-%m-%d %H:%M:%S%z %Z}")
        print("============================================================")
