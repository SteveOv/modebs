""" Pipeline Stage 0 - generating config for suitable LOPS2 targets """
# pylint: disable=no-member
from inspect import getsourcefile
from pathlib import Path
import sys
import json
import argparse

import numpy as np
from mocpy import MOC

from libs import catalogues

THIS_STEM = Path(getsourcefile(lambda: 0)).stem


def tess_ebs_field_to_str(tess_ebs_row, name) -> str:
    """ Context appropriate str representation of the value in a TESS-ebs field """
    if isinstance(tess_ebs_row[name], str):
        return tess_ebs_row[name]
    if name in ["TIC"]:
        return f"{int(tess_ebs_row[name]):.d}"
    if name in ["BJD0", "e_BJD0", "Per", "e_Per"]:
        return f"{tess_ebs_row[name]:.6f}"
    return f"{tess_ebs_row[name]:.3f}"


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Pipeline stage 0: generate config for LOPS2 targets.")
    ap.add_argument(dest="targets_file", type=Path, metavar="TARGETS_FILE",
                    help="json file to write with the details of the targets to ingest")
    ap.add_argument("-fo", "--force-overwrite", dest="force_overwrite", action="store_true",
                    required=False, help="force the overwritting of any existing targets file")
    ap.set_defaults(force_overwrite=False)
    args = ap.parse_args()
    config_dir = Path.cwd() / "config"

    if not args.force_overwrite and args.targets_file.exists():
        resp = input(f"** Warning: output file '{args.targets_file}' exists." \
                     + " Continue & overwrite y/N? ")
        if resp.strip().lower() not in ["y", "yes"]:
            sys.exit()

    # MOC (multi-order coverage map) for querying the full LOPS2 field (regardless of the ncam
    # coverage). Get all targets common to both LOPS2 and and TESS-ebs catalogue
    lops_moc = MOC.load(path="libs/data/lops2-footprints-moc/PLATOfootprint_hpix9_full_v2.fits")
    all_tebs_lops = lops_moc.query_vizier_table("J/ApJS/258/16")
    all_tebs_lops.sort("TIC")


    # Exclude those targets that are not suited to our needs
    # TESS-ebs morphology; we're interested in well-detached so we cut-off Morph at 0.6
    tebs_mask = np.ones(len(all_tebs_lops), dtype=bool)
    tebs_mask &= all_tebs_lops["Morph"] <= 0.6

    # Eclipse depths: need eclipses sufficiently deep to be able to fit with EBOP MAVEN & JKTEBOP.
    # There are 2 algorithms used to characterise the eclipses; a 2-Gaussian fit & the polyfit algo.
    # Prefer the 2g characterisation, values tend to deeper & wider, but coalesce the pf values if
    # have neither 2g values. Keep those without characterisation (masked) to be inspected later.
    min_ecl_depth = np.minimum(np.ma.filled(all_tebs_lops["Dp-2g"], -100),
                               np.ma.filled(all_tebs_lops["Ds-2g"], -100))
    min_ecl_depth_pf = np.minimum(np.ma.filled(all_tebs_lops["Dp-pf"], -100),
                                  np.ma.filled(all_tebs_lops["Ds-pf"], -100))
    missing_2g = min_ecl_depth == -100
    min_ecl_depth[missing_2g] = min_ecl_depth_pf[missing_2g]
    tebs_mask &= (min_ecl_depth >= 0.05) | (min_ecl_depth == -100)

    # Any further criteria/evaluation should go here
    num_matching_rows = sum(tebs_mask)


    # Now build up a dictionary from which we'll generate the target config json
    targets_config = {
        "excplicit": True,   # We are explicitly specifying the target systems
        "target_config_defaults": {
            "quality_bitmask": "hardest",
            "quality_masks": [[1420.0, 1424.0], [1534.0, 1544.0]]
        },
        "target_configs": {}
    }

    # TESS-ebs fields to serialize into a summary "TESS-ebs" config item for each target
    notes_keys = ["BJD0", "e_BJD0", "Per", "e_Per", "Morph",
                  "Phip-2g", "Phis-2g", "Wp-2g", "Ws-2g", "Dp-2g", "Ds-2g",
                  "Phip-pf", "Phis-pf", "Wp-pf", "Ws-pf", "Dp-pf", "Ds-pf"]
    target_configs = targets_config["target_configs"]
    for ix, row in enumerate(all_tebs_lops[tebs_mask], start=0):
        tic = int(row["TIC"])
        print(f"Target {ix+1}/{num_matching_rows}: {tic}")
        target_config = {
            "details": "",
            "notes": "",
            "TESS-ebs": ", ".join(k + ": " + tess_ebs_field_to_str(row, k) for k in notes_keys),
            "why-include": f"morph = {tess_ebs_field_to_str(row, 'Morph')} " \
                            + f"& min(Dp) = {min_ecl_depth[tebs_mask][ix]:.3f}"
        }

        # Capture any known values for these targets
        labels_dict = catalogues.query_tess_ebs_in_sh(tic)
        if labels_dict:
            target_config["labels"] = { "source": "JustesenAlbrecht21apj" }
            target_config["labels"] |= { k: labels_dict[k] for k in labels_dict }

        target_configs[f"TIC {tic:d}"] = target_config


    # Finally, save the dictionary as a formatted JSON file
    with open(args.targets_file, mode="w", encoding="utf8") as jsonf:
        print(f"Saving {num_matching_rows} target config(s) to {jsonf.name}")
        json.dump(targets_config, jsonf, indent=4, default=str)
