""" Pipeline Stage 2 - fitting target lightcurves """
# pylint: disable=no-member, invalid-name
from pathlib import Path
import warnings
import argparse
from datetime import datetime
from contextlib import redirect_stdout
import copy

import numpy as np
import astropy.units as u
from astropy.table import QTable
from astropy.coordinates import SkyCoord

# pylint: disable=line-too-long, wrong-import-position
warnings.filterwarnings("ignore", "Using UFloat objects with std_dev==0 may give unexpected results.", category=UserWarning)
from uncertainties import ufloat, UFloat, nominal_value
from uncertainties.unumpy import nominal_values

from ebop_maven.estimator import Estimator

from libs import pipeline, lightcurves
from libs.iohelpers import Tee
from libs.targets import Targets

# The eclipse completeness ratio above which eclipses are considered complete
ECLIPSE_COMPLETE_TH = 0.9

# The morph value for systems considered very well detached & from which we invoke flattening & clip
FLATTEN_TH = 0.2

if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Pipeline stage 2: fitting target lightcurves.")
    ap.add_argument("-tf", "--targets-file", dest="targets_file", type=Path, required=False,
                    help="json file containing the details of the targets to fit")
    ap.set_defaults(targets_file=Path("./config/plato-lops2-tess-ebs-explicit-targets.json"))
    args = ap.parse_args()
    drop_dir = Path.cwd() / f"drop/{args.targets_file.stem}"
    args.input_file = args.output_file = drop_dir / "targets.table"

    # EBOP MAVEN estimator for JKTEBOP input params; rA+rB, k, J, ecosw, esinw and bP/inc
    estimator = Estimator()

    with redirect_stdout(Tee(open(drop_dir / "fit_lightcurves.log", "w", encoding="utf8"))):
        print(f"Started at {datetime.now():%Y-%m-%d %H:%M:%S%z %Z}")

        targets_config = Targets(args.targets_file)
        print(f"Read in the configuration from '{args.input_file}'",
              f"which contains {targets_config.count()} target(s).")

        # Open the targets table and the configs
        tdata = QTable.read(args.input_file)
        to_fit_mask = tdata["fit_lcs"] == True # pylint: disable=singleton-comparison

        to_fit_mask = tdata["target"] == "V539 Ara"

        to_fit_count = sum(to_fit_mask)
        print(f"Reading '{args.input_file}' which contains {to_fit_count} target(s) to be fitted."
              f"\nWill write updated data to '{args.output_file}'")

        for row_ix, trow in enumerate(tdata[to_fit_mask], start=0):
            target = trow["target"]
            print("a\n\n============================================================")
            print(f"Processing target {row_ix+1} of {to_fit_count}: {target}")
            print("============================================================")

            target_config = targets_config.get(target)

            # It's quicker to get LCs once and cache the results than to continue to bother MAST.
            lcs = lightcurves.load_lightcurves(target, trow["main_id"], sectors=None,
                                               mission=target_config.mission,
                                               author=target_config.author,
                                               exptime=target_config.exptime,
                                               quality_bitmask=target_config.quality_bitmask,
                                               flux_column=target_config.flux_column,
                                               force_mast=False, cache_dir=Path() / ".cache/",
                                               verbose=True)

            # Then filter out any results that are for a different TIC (unlikely but possible).
            select_mask = np.in1d([l.meta['TARGETID'] for l in lcs],
                                  [int(t) for t in trow["tics"].split("|")])
            lcs = lcs[select_mask]
            print(f"Found {len(lcs)} lightcurves prior to applying any configured selections.")

            # Any configured selections, exclusions first so they're overidden by mention in sectors
            select_mask = np.ones(len(lcs), dtype=bool)
            if len((exclude_sectors := target_config.exclude_sectors) or []) > 0:
                select_mask = np.in1d(lcs.sector, exclude_sectors, invert=True)
            if len((sectors_flat := target_config.sectors_flat) or []) > 0:
                select_mask = np.in1d(lcs.sector, sectors_flat)
            lcs = lcs[select_mask]
            print(f"Retained {len(lcs)} lightcurves after applying any configured selections.")

            print("Clipping the lightcurves' invalid fluxes, known distorted sections",
                  "& any isolated sections < 2 d in length.")
            pipeline.mask_lightcurves_unusable_fluxes(lcs,
                                                      target_config.quality_masks or [],
                                                      min_section_dur=2 * u.d)

            print("Inspecting the lightcurves to find and characterise their eclipses")
            t0 = ufloat(trow["t0"], trow["t0_err"])
            period = ufloat(trow["period"].to(u.d).value, trow["period_err"].to(u.d).value)
            widthP = trow["widthP"] # Should never be empty/None/Masked
            widthS = trow["widthS"]
            depthP = None if np.ma.is_masked(trow["depthP"]) else trow["depthP"]
            depthS = None if np.ma.is_masked(trow["depthS"]) else trow["depthS"]
            pipeline.add_eclipse_meta_to_lightcurves(lcs,
                                                     t0,
                                                     period,
                                                     widthP,
                                                     widthS,
                                                     depthP,
                                                     depthS,
                                                     trow["phiS"],
                                                     verbose=True)

            # Group lightcurves to ensure sufficient coverage for fitting. This will also
            # drop lightcurves with insufficient coverage and which cannot be combined
            print("Grouping lightcurves for orbital coverage required for fitting.")
            if target_config.sectors is not None:
                print("Using groups defined in config.")
                sector_groups = target_config.sectors
            else:
                # Otherwise we use the pipeline logic to choose the best combination of sectors
                print("Groups will chosen by analysis of eclipses, with those having",
                      f">{ECLIPSE_COMPLETE_TH:.0%} fluxes are considered complete.")
                sector_groups = pipeline.choose_lightcurve_groups_for_fitting(lcs,
                                                                              ECLIPSE_COMPLETE_TH,
                                                                              verbose=True)
            lcs = pipeline.stitch_lightcurve_groups(lcs, sector_groups, verbose=True)


            # Flatten (optional depending on morph), append delta_mag & delta_mag_err columns
            # and then detrend & rectify the mags to zero by subtracting a low order polynomial 
            do_flatten = target_config.flatten \
                        or (target_config.flatten is None and trow["morph"] <= FLATTEN_TH)
            print("Appending delta_mags/delta_mags_err colums and detrending.",
                  ("Fluxes will be flattened first." if do_flatten else ""))
            pipeline.append_mags_to_lightcurves_and_detrend(lcs,
                                                            target_config.detrend_gap_threshold,
                                                            target_config.detrend_poly_degree,
                                                            target_config.detrend_iterations,
                                                            do_flatten,
                                                            durp=widthP * period,
                                                            durs=widthS * period,
                                                            verbose=True)


            # EBOP MAVEN estimates of fitting input params. Requires phase folded & binned mags
            print("Preparing phase-folded and binned copies of the lightcurves for EBOP MAVEN.")
            mags_bins = estimator.mags_feature_bins
            wrap_phase = u.Quantity(estimator.mags_feature_wrap_phase or (0.5 + trow["phiS"] / 2))
            binned_fold = np.zeros(shape=(len(lcs), 2, mags_bins), dtype=np.float32)
            for ix, lc in enumerate(lcs):
                flc = lc.fold(period.n * u.d, lc.meta["t0"],
                              wrap_phase=wrap_phase,
                              normalize_phase=True)
                binned_fold[ix] = lightcurves.get_binned_phase_mags_data(flc, mags_bins, wrap_phase)

            print("Estimating fitting input values with EBOP MAVEN.")
            predictions = estimator.predict(binned_fold[:, 1], iterations=1000)
            predictions_dict = pipeline.predictions_to_mean_dict(predictions, calculate_inc=True)
            print(("Mean predicted" if predictions.size > 1 else "Predicted"), "fitting parameters",
                f"from {len(lcs)} sector|group(s), including the value calculated for inc.")
            print("\n".join(f"{p:>14s}: {predictions_dict[p]:12.6f}" for p in predictions_dict))


            # Estimating L3 by looking for nearby flux sources
            print("Estimating fitting input value for L3 with Gaia DR3.")
            coords = SkyCoord(ra=trow["ra"], dec=trow["dec"],
                              distance=(1000 / trow["parallax"]).value * u.pc, frame="icrs")
            if np.ma.is_masked(dr3_id := trow["gaia_dr3_id"]):
                g_mag = lcs[0].meta["TESSMAG"] if np.ma.is_masked(trow["g_mag"]) else trow["g_mag"]
                l3 = pipeline.estimate_l3_with_gaia(coords, target_g_mag=g_mag, max_l3=0.1, verbose=True)
            else:
                l3 = pipeline.estimate_l3_with_gaia(coords, target_source_id=dr3_id, max_l3=0.1, verbose=True)


            # Clip masks retain only obs within 2.5 d of an eclipse for fitting. Can optimise
            # fitting and is especially useful where we have previously flattened the lightcurves/
            do_clip = target_config.create_clip_mask \
                    or (target_config.create_clip_mask is None and do_flatten)
            if do_clip:
                print("Creating clip masks to exclude fluxes beyond 5 eclipse widths from fitting.")
            for lc in lcs:
                if do_clip:
                    # Will be picked up by pipeline.fit_target_lightcurves to exlude rows from dat
                    lc.meta["clip_mask"] = lightcurves.create_eclipse_mask(lc,
                                                                        lc.meta["primary_times"],
                                                                        lc.meta["secondary_times"],
                                                                        widthP * period.n * 5,
                                                                        widthS * period.n * 5)

            # Finally, we get to fit the lightcurves with JKTEBOP
            # Extract any fitting overrides from the target config. May contain LD params which are handled next
            fit_overrides = copy.deepcopy(target_config.get("jktebop_overrides", {}))

            # Generate initial LD params based on the system temperature & log(g) (deferring to any overrides)
            teff_sys = lcs[0].meta["TEFF"] or 5650 if np.ma.is_masked(trow["Teff_sys"]) else trow["Teff_sys"].to(u.K).value
            logg_sys = lcs[0].meta["LOGG"] or 4.0 if np.ma.is_masked(trow["logg_sys"]) else trow["logg_sys"].to(u.dex).value
            teffr = predictions_dict["J"]**0.25 # With J*k^2 as a proxy for LR giving teff_rat ~= ((J*k^2)/k^2)^1/4
            ld_params = pipeline.pop_and_complete_ld_config(fit_overrides,
                *nominal_values((teff_sys, teff_sys*teffr.n) if teffr.n < 1 else (teff_sys/teffr.n, teff_sys)),
                *nominal_values((logg_sys, logg_sys)),
                "pow2", True)

            # The refl flags can be 0 (fixed), 1 (fitted) or -1 (calculated from sys geometry)
            refl_fit = -1 if (trow["morph"] <= FLATTEN_TH) else 1

            in_params = {
                # Mass ratio (qphot), can be -1 (force spherical) or a specified ratio value
                "qphot": -1 if (trow["morph"] <= FLATTEN_TH) else predictions_dict["k"]**1.4,
                "gravA": 0.,                "gravB": 0.,
                "L3": l3,
                "reflA": 0.,                "reflB": 0.,

                "period": nominal_value(period),

                "qphot_fit": 0,
                "ecosw_fit": 1,             "esinw_fit": 1,
                "gravA_fit": 0,             "gravB_fit": 0,
                "L3_fit": 1,
                "LDA1_fit": 1,              "LDB1_fit": 1,
                "LDA2_fit": 0,              "LDB2_fit": 0,
                "reflA_fit": refl_fit,      "reflB_fit": refl_fit,
                "sf_fit": 1,
                "period_fit": 1,            "primary_epoch_fit": 1,

                **predictions_dict,
                **ld_params,
                **fit_overrides,
            }

            # Superset of all of the potentially fitted parameters to be read from fitting
            fitted_param_keys = ["rA_plus_rB", "k", "J", "ecosw", "esinw", "inc", "L3",
                                 "period", "ecc", "bP", "light_ratio", "reflA", "reflB"]

            # If max_workers >1 progress updates will occur after each attempt is complete, but overall elapsed
            # time is reduced. If set to 1, tasks are serialized but more frequent progress updates will occur.
            t0s = [nominal_value(lc.meta.get("t0", t0)) for lc in lcs]
            fitted_param_dicts = pipeline.fit_target_lightcurves(lcs,
                                                                 input_params=in_params,
                                                                 read_keys=fitted_param_keys,
                                                                 primary_epoch=t0s,
                                                                 task=3,
                                                                 max_workers=8,
                                                                 max_attempts=3,
                                                                 timeout=900,
                                                                 file_prefix="quick-fit")

            # Get the results into a structured array format
            fitted_params = np.empty(shape=(len(lcs), ),
                                     dtype=[(k, np.dtype(UFloat.dtype)) for k in fitted_param_keys])
            for ix, lc in enumerate(lcs):
                lc.meta["out_fname"] = fitted_param_dicts[ix]["out_fname"]
                lc.meta["fit_fname"] = fitted_param_dicts[ix]["fit_fname"]
                for k in fitted_param_keys:
                    fitted_params[ix][k] = fitted_param_dicts[ix][k]

            # If there's only 1 sector/group we use the predictions directly, otherwise we use the median and
            # 2-sigma (the sample is relatively small) over the scatter of the predictions for the uncertainty.
            if fitted_params.size > 1:
                summary_params = pipeline.median_params(fitted_params, 0.9545, True)
                print(f"Median fitted values and 2σ error bars from {len(lcs)} fitted lightcurves.")
            else:
                summary_params = fitted_params[0]
                print("Fitted values and formal error bars from 1 fitted lightcurve.")
            print("\n".join(f"{p:>14s}: {summary_params[p]:12.6f}"
                            for p in fitted_param_keys if summary_params[p] is not None))

            # Finally, store the params
            for k in ["rA_plus_rB", "k", "J", "ecosw", "esinw", "bP", "inc", "L3"]:
                unit = u.deg if k == "inc" else 1
                tdata[to_fit_mask][row_ix][k] = summary_params[k].n * unit
                tdata[to_fit_mask][row_ix][f"{k}_err"] = summary_params[k].s * unit

            teffR = (summary_params["light_ratio"] / summary_params["k"]**2)**0.25
            tdata[to_fit_mask][row_ix]["TeffR"] = teffR.n
            tdata[to_fit_mask][row_ix]["TeffR_err"] = teffR.s
            print(f"    Teff_ratio: {teffR:12.6f} (calculated from light_ratio & k)")

        print(f"Writing updated data set to {args.output_file}")
        tdata.write(args.output_file, format="votable", overwrite=True)

        print(f"\nCompleted at {datetime.now():%Y-%m-%d %H:%M:%S%z %Z}")
