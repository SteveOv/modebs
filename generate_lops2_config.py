#!/usr/bin/env python3
""" Pipeline Stage 0 - generating config for suitable LOPS2 targets """
# pylint: disable=no-member
from inspect import getsourcefile
from pathlib import Path
import sys
import json
import argparse
from warnings import filterwarnings

filterwarnings("ignore", "Warning: converting a masked element to nan.", category=UserWarning)
filterwarnings("ignore", "Using UFloat objects with std_dev==0 may", category=UserWarning)
filterwarnings("ignore", "Warning: the tpfmodel submodule is not available", category=UserWarning)

# pylint: disable=wrong-import-position
import numpy as np
from mocpy import MOC
import astropy.units as u

from libs import catalogues, lightcurves, pipeline


THIS_STEM = Path(getsourcefile(lambda: 0)).stem

# These are systems which may be included up by selection criteria but are known to not fit
exclude_tics = {
    # pylint: disable=line-too-long
    13062255: "too close for JKTEBOP (rA+rB ~ 0.5)",
    64783257: "low SNR and shallow eclipses make impossible to get a durable fit, with or without flattening",
    126446153: "too close for JKTEBOP",
    129268651: "ESS-ebs eclipse depths incorrect - this has very shallow eclipses",
    140661916: "too close for JKTEBOP (rA+rB ~ 0.5, morph 0.592)",
    142105299: "highly eccentric, long period with shallow eclipses (Ds~0.062) - cannot get a durable fit",
    150284425: "cannot get a good fit to 'hump' in LC prior to the primary eclipse",
    150357064: "very shallow with variability as deep as eclipses - cannot get a good fit",
    165186801: "too close/tidally distorted for JKTEBOP (rA+rB ~ 0.55)",
    167692429: "eclipses almost non-existent by the latter sectors - needs investigation",
    220430912: "too close for JKTEBOP (rA+rB ~ 0.5)",
    257691369: "too shallow (more than Ds-2g of 0.081 indicates), with long period - cannot get a durable fit",
    259543079: "extremely eccentric and cannot get a reliable fit, even with interventions",
    260124760: "too close for JKTEBOP (rA+rB ~ 0.6, morph 0.590), however the fit is plausible",
    278826996: "highly eccentric and cannot get a reliable fit even with interventions",
    299906906: "low SNR and quite shallow eclipses (although ~0.1) - together make a difficult fit",
    300654002: "too close for JKTEBOP (rA+rB ~ 0.5, morph 0.563)",
    310308203: "too close for JKTEBOP (rA+rB ~ 0.5, morph 0.593)",
    349643889: "likely needs TESS-ebs period doubling (not corroborated); even with doubling, a nonsense fit without intervention",
    349835367: "too close for JKTEBOP (rA+rB ~ 0.45, morph 0.544)",
    382069435: "not TESS-ebs depthp, high variability and significant eclipse timing shift in later sectors - too difficult",
    393344055: "too close for JKTEBOP (rA+rB ~ 0.5, morph 0.600)",
    393491149: "too close for JKTEBOP (rA+rB ~ 0.5, morph 0.595)",
}

# Too shallow (review depth criteria)
exclude_tics |= { 4783257: "TBC" }

# These may be excluded by selection criteria but are included as they're known to fit
include_tics = {
    # pylint: disable=line-too-long
    30034081: "TESS-ebs has halved the period (corroborated with J+A) so include with overriden ephemeris",
    31054255: "TESS-ebs has halved the period (not corroborated) so include with overriden ephemeris",
    31273263: "TESS-ebs has halved the period (not corroborated) so include with overriden ephemeris",
    31810287: "secondary eclipses are 'borderline' (Ds-2g 0.049) but this fits with flattening",
    37606218: "secondary eclipses are very shallow (Ds-2g 0.013) but mitigated by being total",
    55369219: "TESS-ebs has halved the period (corroborated with TBOSB) so include with overriden ephemeris",
    147975720: "TESS-ebs has halved the period (not corroborated) so include with overriden ephemeris",
    200440175: "TESS-ebs has halved the period so include with overriden ephemeris. J+A characterisation affected by 1/2 period shift.",
    220420534: "secondary eclipses are 'borderline' (Ds-2g 0.05) but we easily get good consistent fits",
    260659986: "TESS-ebs has halved the period (corroborated with J+A) so include with overriden ephemeris",
    307488184: "secondary eclipses are very shallow (Ds-2g 0.017) but mitigated by being total",
    319863494: "a difficult LC fit as the inclination changes over time - requires inc to be overriden by sector",
    349059354: "TESS-ebs has halved the period (corroborated with GCVS, AAVSO VSX, ShiQian) so include and override ephemeris",
    349480507: "TESS-ebs has halved the period (corroborated with J+A & TBOSB2) so include with overriden ephemeris",
    425064757: "TESS-ebs has no data on the secondary so include with overriden ephemeris"
}

# These are systems which are known to need hard-coded overrides to some config settings
known_overrides = {
    # pylint: disable=line-too-long
    # Highly eccentric and needs assistance to fit
    7695666: { "jktebop_overrides": { "ecosw": -0.56, "esinw": 0.08, "inc": 88.7 }, },
    # Double the TESS-ebs period (corroborated with J+A), copy the primary eclipse data to secondary and halve the widths
    30034081: { "period": 4.6892177144299785, "period_err": 0.0002550268060178, "widthP": 0.068, "widthS": 0.068, "depthP": 0.452, "depthS": 0.452, "phiS": 0.500 },
    # Double the TESS-ebs period (not corroborated), copy the primary eclipse data to secondary and halve the widths
    31054255: { "period": 1.747743979, "period_err": 0.000000513, "widthP": 0.021, "widthS": 0.021, "depthP": 0.021, "depthS": 0.021, "phiS": 0.500 },
    # Double the TESS-ebs period (not corroborated), copy the primary eclipse data to secondary and halve the widths
    31273263: { "period": 45.145916694, "period_err": 0.002004383, "widthP": 0.010, "widthS": 0.010, "depthP": 0.220, "depthS": 0.220, "phiS": 0.500 },
    # Flattening to combat variability
    31810287: { "flatten": True, },
    # Difficult to fit as there is significan variability and flares. More likely to get to the system params with flattening.
    32702481: { "flatten": True, },
    53292822: { "t0": 1519.046, "period": 4.93495, "phiS": 0.67 },
    # Switch t0, double the TESS-ebs period (corroborated with TBOSB), copy the primary eclipse data to secondary and halve the widths
    55369219: { "t0": 1389.775813611, "t0_err": 0.036166900, "period": 3.959191743, "period_err": 0.000061792, "widthP": 0.045, "widthS": 0.045, "depthP": 0.067, "depthS": 0.067, "phiS": 0.500 },
    # Gaia DR3 with no parallax; dist from Gaia DR2 ~500 pc so set parallax to 2.0;
    55659311: { "parallax": 2.0, },
    63579446: { "exclude_sectors": [87], },
    # Some noise and variability which struggles to converge even with retries. With a fairly low morph of 0.224 flattening helps with fit.
    66509654: { "flatten": True, },
    80650858: { "Teff_sys": 20000, },
    # Double the TESS-ebs period (not corroborated), copy the primary eclipse data to secondary and halve the widths
    147975720: { "period": 5.700445194, "period_err": 0.000026526, "widthP": 0.015, "widthS": 0.015, "depthP": 0.264, "depthS": 0.264, "phiS": 0.500 },
    153742549: { "flatten": True, },
    # overriding the TESS-ebs period with value from inspecting S32+33 (left the rest of the ephemeris unchanged)
    167756615: { "exptime": [120, 600], "period": 19.179, },
    # overriding the TESS-ebs eclipse data which overstates eclipse widths & depths
    173756896: { "widthP": 0.025, "widthS": 0.043, "depthP": 0.100, "depthS": 0.020, },
    # Double the TESS-ebs period, copy the primary eclipse data to secondary and halve the widths. J+A characterisation affected by 1/2 period shift.
    200440175: { "period": 3.652007766, "period_err": 0.000010387, "widthP": 0.062, "widthS": 0.062, "depthP": 0.433, "depthS": 0.433, "phiS": 0.500 },
    # highly eccentric and gives nonsense fit without assistance; force the grouping for better coverage and esinw input value
    219362976: { "sectors":[[4, 5, 6], [31, 32]], "jktebop_overrides": { "esinw": 0.25 }, },
    220397947: { "flatten": True, },
    260504147: { "jktebop_overrides": { "inc": 89.3, "L3": 0.5 }, },
    # Double the TESS-ebs period (corroborated with J+A), copy the primary eclipse data to secondary and halve the widths. Fits benefit from flattening, but morph ~ 0.4 so may try better detrending instead.
    260659986: { "period": 7.140179368, "period_err": 0.000016378, "widthP": 0.033, "widthS": 0.033, "depthP": 0.172, "depthS": 0.172, "phiS": 0.500 },
    # Repeated gaussj warnings on S61+62 and a failure to converge after retries unless we start ecosw/esinw at zero
    278826516: { "jktebop_overrides": { "ecosw": 0, "esinw": 0 }, },
    # highly eccentric and needs help
    279741942: { "jktebop_overrides": { "ecosw": 0.36, "esinw": 0.06 }, },
    # will not meet 2+1 eclipse criterion, so no fit without the sectors override & fixed period; period override from insepcting S87
    299903137: { "sectors": [[6], [87]], "period": 26.3811, "phiS": 0.365, "jktebop_overrides": { "period_fit": 0 }, },
    # TESS-ebs period (not corroborated) and phiS corrected and corresponding reduction in eclipse widths
    319558164: { "period": 16.596535, "widthP": 0.013, "widthS": 0.012, "phiS": 0.540, },
    # TESS-ebs ephemeris values not usable for this target - ephemeris set by inspection; also inc changes over time so must override by sector group
    319863494: { "t0": 2206.68905, "period": 17.644121, "widthP": 0.035, "widthS": 0.031, "depthP": 0.20, "depthS": 0.15, "phiS": 0.290,
                "sectors": [[33, 34], [61], [87, 88]], "jktebop_overrides": [{"inc": 88.8}, {"inc": 89.6}, {"inc": 90.0}], },
    # Double the TESS-ebs period (corroborated with GCVS, AAVSO VSX, ShiQian), copy the primary eclipse data to secondary and halve the widths. In AAVSO B/vsx/vsx with per of ~3.3 d
    349059354: { "period": 3.329326238, "period_err": 0.000009050, "widthP": 0.058, "widthS": 0.058, "depthP": 0.174, "depthS": 0.174, "phiS": 0.500 },
    # Need to double the TESS-ebs period (corroborated with J+A & TBOSB2), copy the primary meta to secondary and halve the widths
    349480507: { "period": 3.124038476, "period_err": 0.000000356, "widthP": 0.067, "widthS": 0.067, "depthP": 0.339, "depthS": 0.339, "phiS": 0.500 },
    # Need to double the TESS-ebs period (not corroborated), copy primary meta to secondary & halve widths otherwise there's no sign of a secondary. Only TESS-ebs has ephemeris.
    349643889: { "period": 53.358925782, "period_err": 0.002282463, "widthP": 0.005, "widthS": 0.005, "depthP": 0.165, "depthS": 0.165, "phiS": 0.500 },
    # highly eccentric and gives nonsense fits without assistance - particularly sensitive to the Poincare elements
    350298314: { "jktebop_overrides": { "ecosw": -0.38, "esinw": 0.11, "period_fit": 0 }, },
    355152640: { "flatten": True, },
    386166904: { "widthS": 0.050, },
    # A sub-dwarf period of ~0.25 d. Not included by default as TESS-ebs lack secondary width. Eclipses are shallow but total.
    425064757: { "widthS": 0.080, "depthP": 0.350, "depthS": 0.050, "phiS": 0.500 },
}



if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Pipeline stage 0: generate config for LOPS2 targets.")
    ap.add_argument(dest="targets_file", type=Path, metavar="TARGETS_FILE",
                    help="json file to write with the details of the targets to ingest")
    ap.add_argument("-fo", "--force-overwrite", dest="force_overwrite", action="store_true",
                    required=False, help="force the overwritting of any existing targets file")
    ap.set_defaults(force_overwrite=False,
                    max_morph=0.6, min_ecl_depth=0.05,
                    inspect_on_missing_ephemeris=False)
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
    all_tics = [int(t) for t in all_tebs_lops["TIC"]]


    # Exclude those targets that are not suited to our needs; first explicitly configured exclusions
    include_mask = np.in1d(all_tics, list(exclude_tics.keys()), invert=True)

    # TESS-ebs morphology; we're interested in well-detached so we have a Morph cut-off
    include_mask &= all_tebs_lops["Morph"] <= args.max_morph

    # Eclipse depths: need eclipses sufficiently deep to be able to fit with EBOP MAVEN & JKTEBOP.
    # There are 2 algorithms used to characterise the eclipses; a 2-Gaussian fit & the polyfit algo.
    # Prefer the 2g characterisation, values tend to deeper & wider, but coalesce the pf values if
    # we have neither 2g values. Keep those without characterisation (masked) to be inspected later.
    min_ecl_depths = np.minimum(np.ma.filled(all_tebs_lops["Dp-2g"], np.nan),
                                np.ma.filled(all_tebs_lops["Ds-2g"], np.nan))
    min_ecl_depths_pf = np.minimum(np.ma.filled(all_tebs_lops["Dp-pf"], np.nan),
                                   np.ma.filled(all_tebs_lops["Ds-pf"], np.nan))
    missing_2g_mask = np.isnan(min_ecl_depths)
    min_ecl_depths[missing_2g_mask] = min_ecl_depths_pf[missing_2g_mask]
    include_mask &= (min_ecl_depths >= args.min_ecl_depth) | np.isnan(min_ecl_depths)

    # Any further criteria/evaluation should go here

    # The final selections, including any hard-coded inclusion overrides
    include_mask |= np.in1d(all_tics, list(include_tics.keys()))
    num_matching_rows = sum(include_mask)


    # Now build up a dictionary from which we'll generate the target config json
    targets_config = {
        "excplicit": True,   # We are explicitly specifying the target systems
        "target_config_defaults": {
            "quality_bitmask": None,    # leave choice to fit_lightcurves
            "quality_masks": [[1420.0, 1424.0], [1534.0, 1544.0]],
        },
        "target_configs": {}
    }

    # Now create a target_config for each target
    target_configs = targets_config["target_configs"]
    for ix, row in enumerate(all_tebs_lops[include_mask], start=0):
        tic = int(row["TIC"])
        target_id = f"TIC {tic:d}"
        print(f"Target {ix+1}/{num_matching_rows}: {target_id}", end="...")

        # Start this target's config with some basic info
        # ----------------------------------------------------------------------
        config = {
            "details": "",
            "notes": "",
            "why_include": ""
        }


        # TESS-ebs fields to the ephemeris config item for each target.
        # These first set of fields are easy as there are direct 1-1 mappings.
        # ----------------------------------------------------------------------
        for kfrom, kto in [("BJD0", "t0"), ("e_BJD0", "t0_err"),
                           ("Per", "period"), ("e_Per", "period_err"), ("Morph", "morph")]:
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
        if (config_overrides := known_overrides.get(tic, None)) is not None:
            config |= config_overrides
            config["notes"] += "with overrides to " + ", ".join(k for k in config_overrides) + ";"

        # Now download & inspect any light-curves to confirm/improve the ephemeris
        # For now, we're skipping these targets with insufficient ephemeris data to be fitted
        # ----------------------------------------------------------------------
        if any(config.get(k, None) is None for k in ["phiS", "depthP", "depthS"]):
            if not args.inspect_on_missing_ephemeris:
                print("lacking eclipse phase or depth information...omitted.")
                continue

            print("no TESS-ebs eclipse depths, so inspecting LCs", end="...")
            search_term = config.get("search_term", target_id)
            lcs = lightcurves.load_lightcurves(target_id,
                                               search_term,
                                               sectors=None,
                                               mission=["TESS", "HLSP"],
                                               author=["SPOC", "TESS-SPOC"],
                                               exptime=[120, 600],
                                               quality_bitmask="default",
                                               flux_column="sap_flux",
                                               force_mast=False,
                                               cache_dir=Path() / ".cache/.mast/",
                                               consume_cadence_warnings=True,
                                               verbose=False)

            quality_masks = targets_config.get("target_config_defaults", {}).get("quality_masks",[])
            pipeline.mask_lightcurves_unusable_fluxes(lcs, quality_masks, min_section_dur=2 * u.d)

            widthp = config.get("widthP", None) or 0.01
            widths = config.get("widthS", None) or widthp
            phis = config.get("phiS", None) or 0.5
            pipeline.add_eclipse_meta_to_lightcurves(lcs, config["t0"], config["period"],
                                                     widthp, widths, phis=phis, verbose=False)

            pri_depths = [[d for d in lc.meta["primary_depths"] if not np.isnan(d)] for lc in lcs]
            pri_depths = [d for dd in pri_depths for d in dd]
            sec_depths = [[d for d in lc.meta["secondary_depths"] if not np.isnan(d)] for lc in lcs]
            sec_depths = [d for dd in sec_depths for d in dd]

            avg_pri_depth = np.mean(pri_depths) if len(pri_depths) else 0
            avg_sec_depth = np.mean(sec_depths) if len(sec_depths) else 0
            if min(avg_pri_depth, avg_sec_depth) >= args.min_ecl_depth:
                config["depthP"] = avg_pri_depth
                config["depthS"] = avg_sec_depth
                config["phiS"] = phis
                config["notes"] += "with overrides of eclipse depths from inspection of LCs;"
                print(f"suitable <pri>={avg_pri_depth:.6f} <sec>={avg_sec_depth:.6f}", end="...")
            elif any(d > 0 for d in [avg_pri_depth, avg_sec_depth]):
                print(f"too shallow <pri>={avg_pri_depth:.6f} <sec>={avg_sec_depth:.6f}...omitted.")
                continue
            else:
                print("unable to get sufficient eclipse information from LCs...omitted.")
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
        if tic in include_tics:
            config["why_include"] = include_tics.get(tic, "explicitly included")
        else:
            config["why_include"] = f"morph={config['morph']:.3f} & " \
                                    + f"ecl_depths>={min(config['depthP'],config['depthS']):.3f}"
        target_configs[target_id] = config
        print("added to config.")


    # Finally, save the dictionary as a formatted JSON file
    with open(args.targets_file, mode="w", encoding="utf8") as jsonf:
        num_saved_rows = len(targets_config["target_configs"].keys())
        print(f"Saving {num_saved_rows} target config(s) to {jsonf.name}")
        json.dump(targets_config, jsonf, indent=4, default=str)
