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

# These are systems which may be included up by selection criteria but are known to not fit
exclude_tics = [
    # pylint: disable=line-too-long
    "0126446153",   # too close for JKTEBOP
    "0129268651",   # TESS-ebs eclipse depths incorrect - this has very shallow eclipses
    "0140661916",   # too close for JKTEBOP (rA+rB ~ 0.5, morph 0.592)
    "0142105299",   # highly eccentric, long period with shallow eclipses (Ds~0.062) - cannot get a durable fit
    "0150284425",   # cannot get a good fit to "hump" in LC prior to the primary eclipse
    "0150357064",   # very shallow with variability as deep as eclipses - cannot get a good fit
    "0165186801",   # too close/tidally distorted for JKTEBOP (rA+rB ~ 0.55)
    "0167692429",   # eclipses almost non-existent by the latter sectors - needs investigation
    "0220430912",   # too close for JKTEBOP (rA+rB ~ 0.5)
    "0257691369",   # too shallow (more than Ds-2g of 0.081 indicates), with long period - cannot get a durable fit
    "0259543079",   # extremely eccentric and cannot get a reliable fit, even with interventions
    "0260124760",   # too close for JKTEBOP (rA+rB ~ 0.6, morph 0.590), however the fit is plausible
    "0278826996",   # highly eccentric and cannot get a reliable fit even with interventions
    "0299906906",   # low SNR and quite shallow eclipses (although ~0.1) - together make a difficult fit
    "0300654002",   # too close for JKTEBOP (rA+rB ~ 0.5, morph 0.563)
    "0310308203",   # too close for JKTEBOP (rA+rB ~ 0.5, morph 0.593)
    "0349835367",   # too close for JKTEBOP (rA+rB ~ 0.45, morph 0.544)
    "0393344055",   # too close for JKTEBOP (rA+rB ~ 0.5, morph 0.600)
    "0393491149",   # too close for JKTEBOP (rA+rB ~ 0.5, morph 0.595)
]

# Too shallow (review depth criteria)
exclude_tics += ["0064783257", "0150357064"]

# These may be excluded by selection criteria but are included as they're known to fit well
include_tics = [
    # pylint: disable=line-too-long
    "0030034081",   # TESS-ebs misses the secondary, however double the period and it fits well
    "0031810287",   # secondary eclipses are "borderline" (Ds-2g 0.049) but this fits with flattening
    "0037606218",   # secondary eclipses are very shallow (Ds-2g 0.013) but mitigated by being total
    "0220420534",   # secondary eclipses are "borderline" (Ds-2g 0.05) but we easily get good consistent fits
    "0307488184",   # secondary eclipses are very shallow (Ds-2g 0.017) but mitigated by being total
]

# These are systems which are known to need hard-coded overrides to some config settings
known_overrides = {
    # pylint: disable=line-too-long
    # Highly eccentric and needs assistance to fit
    "TIC 7695666": { "jktebop_overrides": { "ecosw": -0.56, "esinw": 0.08, "inc": 88.7 }, },
    # Need to double the TESS-ebs period, copy the primary meta to secondary and halve the widths
    "TIC 30034081": { "period": 4.6892177144299785, "period_err": 0.0002550268060178, "widthP": 0.068, "widthS": 0.068, "depthP": 0.452, "depthS": 0.452, "phiS": 0.500 },
    # Flattening to combat variability
    "TIC 31810287": { "flatten": True, },
    "TIC 53292822": { "t0": 1519.046, "period": 4.93495, "phiS": 0.67 },
    # Gaia DR3 with no parallax; dist from Gaia DR2 ~500 pc so set parallax to 2.0;
    "TIC 55659311": { "parallax": 2.0, },
    "TIC 63579446": { "exclude_sectors": [87], },
    "TIC 80650858": { "Teff_sys": 20000, },
    "TIC 153742549": { "flatten": True, },
    # overriding the TESS-ebs period with value from inspecting S32+33 (left the rest of the ephemeris unchanged)
    "TIC 167756615": { "exptime": [120, 600], "period": 19.179, },
    # overriding the TESS-ebs eclipse data which overstates eclipse widths & depths
    "TIC 173756896": { "widthP": 0.025, "widthS": 0.043, "depthP": 0.100, "depthS": 0.020, },
    # highly eccentric and gives nonsense fit without assistance (esinw)
    "TIC 219362976": { "jktebop_overrides": { "esinw": 0.2 }, },
    "TIC 220397947": { "flatten": True, },
    "TIC 260504147": { "jktebop_overrides": { "inc": 89.3, "L3": 0.5 }, },
    # highly eccentric and needs help
    "TIC 279741942": { "jktebop_overrides": { "ecosw": 0.36, "esinw": 0.06 }, },
    # will not meet 2+1 eclipse criterion, so no fit without the sectors override & fixed period; period override from insepcting S87
    "TIC 299903137": { "sectors": [[6], [87]], "period": 26.3811, "phiS": 0.365, "jktebop_overrides": { "period_fit": 0 }, },
    # TESS-ebs period and phiS corrected and corresponding reduction in eclipse widths
    "TIC 319558164": { "period": 16.596535, "widthP": 0.013, "widthS": 0.012, "phiS": 0.540, },
    # TESS-ebs ephemeris values not usable for this target - ephemeris set by inspection;
    "TIC 319863494": { "t0": 2206.68905, "period": 17.644121, "widthP": 0.035, "widthS": 0.031, "depthP": 0.20, "depthS": 0.15, "phiS": 0.290, },
    # highly eccentric and gives nonsense fits without assistance - particularly sensitive to the Poincare elements
    "TIC 350298314": { "jktebop_overrides": { "ecosw": -0.38, "esinw": 0.11, "period_fit": 0 }, },
    "TIC 355152640": { "flatten": True, },
    "TIC 386166904": { "widthS": 0.050, },
}



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
    include_mask = np.in1d(all_tebs_lops["TIC"], exclude_tics, invert=True)

    # TESS-ebs morphology; we're interested in well-detached so we cut-off Morph at 0.6
    include_mask &= all_tebs_lops["Morph"] < 0.6

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
    include_mask &= (min_ecl_depth >= 0.05) | (min_ecl_depth == -100)

    # Any further criteria/evaluation should go here

    # The final selections, including any hard-coded inclusion overrides
    include_mask |= np.in1d(all_tebs_lops["TIC"], include_tics)
    num_matching_rows = sum(include_mask)


    # Now build up a dictionary from which we'll generate the target config json
    targets_config = {
        "excplicit": True,   # We are explicitly specifying the target systems
        "target_config_defaults": {
            "quality_bitmask": "hardest",
            "quality_masks": [[1420.0, 1424.0], [1534.0, 1544.0]]
        },
        "target_configs": {}
    }

    # Now create a target_config for each target
    target_configs = targets_config["target_configs"]
    for ix, row in enumerate(all_tebs_lops[include_mask], start=0):
        tic = int(row["TIC"])
        target_id = f"TIC {tic:d}"
        print(f"Target {ix+1}/{num_matching_rows}: {target_id}")

        # Start this target's config with some basic info
        # ----------------------------------------------------------------------
        config = {
            "details": "",
            "notes": "",
            "why_include": f"morph={row['Morph']:.3f} " +
                            f"& min(Dp)={min_ecl_depth[include_mask][ix]:.3f}"
        }


        # TESS-ebs fields to the ephemeris config item for each target.
        # These first set of fields are easy as there are direct 1-1 mappings.
        # ----------------------------------------------------------------------
        for kfrom, kto in [("BJD0", "t0"), ("Per", "period"), ("Morph", "morph")]:
            if (val := row.get(kfrom, None)) is not None and not np.isnan(val):
                config[kto] = round(float(val), 6) if kfrom in ["Morph"] else val

        # There are two sets of eclipse data; those based on the 2-Gaussian algorithm and those on
        # a polyfit algorithm. For the best chance of consistent values we use one or other.
        vals = np.array([[row[k+algo] for k in ["Phip", "Phis", "Wp", "Ws", "Dp", "Ds"]]
                                        for algo in ["-2g","-pf"]], dtype=float)

        # Prefer the set with the most data, and break a tie in favour of the 2g values (set 0)
        # as these tend to wider eclipse widths (we find polyfit tends to under value).
        is_num = ~np.isnan(vals)
        vals_row_ix = np.argmax(np.sum(is_num, axis=1))

        # TESS-ebs eclipse widths are in units of phase and depth in units of normalized flux
        for cix, kto in enumerate(["widthP", "widthS", "depthP", "depthS"], start=2):
            config[kto] = round(vals[vals_row_ix, cix], 6) if is_num[vals_row_ix, cix] else None

        # We want to get the phases so that the primary is zero and the secondary is
        # offset from this. Within TESS-ebs Phip is often near 1 and Phis < Phip.
        if all(is_num[vals_row_ix, 0:2]):
            while vals[vals_row_ix, 0] > vals[vals_row_ix, 1]:
                vals[vals_row_ix, 0] -= 1
            config["phiS"] = round(vals[vals_row_ix, 1] - vals[vals_row_ix, 0], 6)
        else:
            config["phiS"] = None


        # Apply any overrides now, before we try to fix any data by inspection
        # ----------------------------------------------------------------------
        config |= known_overrides.get(target_id, { })


        # Now download & inspect any light-curves to confirm/improve the ephemeris
        # For now, we're skipping these targets with insufficient ephemeris data to be fitted
        # ----------------------------------------------------------------------
        if any(config.get(k, None) is None for k in ["phiS", "widthP", "widthS"]):
            continue


        # Capture any known values for these targets as labels
        # ----------------------------------------------------------------------
        label_keys = ["t0", "period", "rA_plus_rB", "k", "J", "ecosw", "esinw", "inc",
                      "LR", "TeffR", "TeffA", "TeffB", "RA", "RB", "MA", "MB", "log_age", "dist"]
        if labels_dict := catalogues.query_tess_ebs_in_sh(tic):
            config["labels"] = { "source": "2021ApJ...912..123J" }    # No errorbars in this
            config["labels"] |= { k: labels_dict[k] for k in label_keys if k in labels_dict }
            if "rA_plus_rB" not in config["labels"] and "rA" in labels_dict and "rB" in labels_dict:
                config["labels"]["rA_plus_rB"] = labels_dict["rA"] + labels_dict["rB"]


        # If we got here, everything is OK with the target so we add it to the config to be written
        # ----------------------------------------------------------------------
        target_configs[target_id] = config


    # Finally, save the dictionary as a formatted JSON file
    with open(args.targets_file, mode="w", encoding="utf8") as jsonf:
        num_saved_rows = len(targets_config["target_configs"].keys())
        print(f"Saving {num_saved_rows} target config(s) to {jsonf.name}")
        json.dump(targets_config, jsonf, indent=4, default=str)
