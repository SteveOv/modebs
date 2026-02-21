""" Pipeline Stage 2 - fitting target lightcurves """
# pylint: disable=no-member, invalid-name
from inspect import getsourcefile
from pathlib import Path
import warnings
import argparse
from datetime import datetime
from contextlib import redirect_stdout
import copy

import numpy as np
import astropy.units as u
from astropy.coordinates import SkyCoord

# pylint: disable=line-too-long, wrong-import-position
warnings.filterwarnings("ignore", "Using UFloat objects with std_dev==0 may give unexpected results.", category=UserWarning)
from uncertainties import ufloat, UFloat, nominal_value
from uncertainties.unumpy import nominal_values

from ebop_maven.estimator import Estimator

from libs import pipeline, lightcurves
from libs.iohelpers import Tee
from libs.targets import Targets
from libs.pipeline_dal import QTableFileDal


THIS_STEM = Path(getsourcefile(lambda: 0)).stem

# The eclipse completeness ratio above which eclipses are considered complete
ECLIPSE_COMPLETE_TH = 0.9

# The morph value for systems considered very well detached & from which we invoke flattening & clip
FLATTEN_TH = 0.2

if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Pipeline stage 2: fitting target lightcurves.")
    ap.add_argument("-tf", "--targets-file", dest="targets_file", type=Path, required=False,
                    help="json file containing the details of the targets to fit")
    ap.add_argument("-wd", "--write-diags", dest="write_diags", action="store_true", required=False,
                    help="write a second, human readable output file for diagnostics")
    ap.set_defaults(targets_file=Path("./config/plato-lops2-tess-ebs-explicit-targets.json"),
                    write_diags=False)
    args = ap.parse_args()
    drop_dir = Path.cwd() / f"drop/{args.targets_file.stem}"
    args.working_set_file = drop_dir / "working-set.table"

    # EBOP MAVEN estimator for JKTEBOP input params; rA+rB, k, J, ecosw, esinw and bP/inc
    estimator = Estimator()

    with redirect_stdout(Tee(open(drop_dir / f"{THIS_STEM}.log", "w", encoding="utf8"))):
        print(f"Started at {datetime.now():%Y-%m-%d %H:%M:%S%z %Z}")

        targets_config = Targets(args.targets_file)
        print(f"Read in the configuration from '{args.targets_file}'",
              f"which contains {targets_config.count()} target(s) that have not been exluded.")

        # Open the targets table and the configs
        dal = QTableFileDal(args.working_set_file)
        to_fit_target_ids = list(dal.keys_where("fitted_lcs", lambda v: v == False)) # pylint: disable=singleton-comparison
        to_fit_count = len(to_fit_target_ids)
        print(f"The data indicates there are {to_fit_count} target(s) to be fitted.")


        for fit_counter, target_id in enumerate(to_fit_target_ids, start=1):
            config = targets_config.get(target_id)
            print("\n\n============================================================")
            print(f"Processing target {fit_counter} of {to_fit_count}: {target_id}")
            print("============================================================")


            # It's quicker to get LCs once and cache the results than to continue to bother MAST.
            search_term, tics = dal.read_values(target_id, "main_id", "tics")
            lcs = lightcurves.load_lightcurves(target_id,
                                               search_term,
                                               sectors=None,
                                               mission=config.mission,
                                               author=config.author,
                                               exptime=config.exptime,
                                               quality_bitmask=config.quality_bitmask,
                                               flux_column=config.flux_column,
                                               force_mast=False, cache_dir=Path() / ".cache/",
                                               verbose=True)

            # Then filter out any results that are for a different TIC (unlikely but possible).
            select_mask = np.in1d([l.meta['TARGETID'] for l in lcs],
                                  [int(t) for t in tics.split("|")])
            lcs = lcs[select_mask]
            print(f"Found {len(lcs)} lightcurves prior to applying any configured selections.")

            # Any configured selections, exclusions first so they're overidden by mention in sectors
            select_mask = np.ones(len(lcs), dtype=bool)
            if len((exclude_sectors := config.exclude_sectors) or []) > 0:
                select_mask = np.in1d(lcs.sector, exclude_sectors, invert=True)
            if len((sectors_flat := config.sectors_flat) or []) > 0:
                select_mask = np.in1d(lcs.sector, sectors_flat)
            lcs = lcs[select_mask]
            print(f"Retained {len(lcs)} lightcurves after applying any configured selections.")

            print("Clipping the lightcurves' invalid fluxes, known distorted sections",
                  "& any isolated sections < 2 d in length.")
            pipeline.mask_lightcurves_unusable_fluxes(lcs,
                                                      config.quality_masks or [],
                                                      min_section_dur=2 * u.d)


            t0, period, widthP, widthS, depthP, depthS, phiS, morph = dal.read_values(
                target_id, "t0", "period", "widthP", "widthS", "depthP", "depthS", "phiS", "morph")


            print("Inspecting the lightcurves to find and characterise their eclipses")          
            pipeline.add_eclipse_meta_to_lightcurves(lcs, t0, period, widthP, widthS,
                                                     depthP, depthS, phiS, verbose=True)


            # Group lightcurves to ensure sufficient coverage for fitting. This will also
            # drop lightcurves with insufficient coverage and which cannot be combined
            print("Grouping lightcurves for orbital coverage required for fitting.")
            if config.sectors is not None:
                print("Using groups defined in config.")
                sector_groups = config.sectors
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
            do_flatten = config.flatten or (config.flatten is None and morph <= FLATTEN_TH)
            print("Appending delta_mags/delta_mags_err colums and detrending.",
                  ("Fluxes will be flattened first." if do_flatten else ""))
            pipeline.append_mags_to_lightcurves_and_detrend(lcs,
                                                            config.detrend_gap_threshold,
                                                            config.detrend_poly_degree,
                                                            config.detrend_iterations,
                                                            do_flatten,
                                                            durp=widthP * period,
                                                            durs=widthS * period,
                                                            verbose=True)


            # EBOP MAVEN estimates of fitting input params. Requires phase folded & binned mags
            print("Preparing phase-folded and binned copies of the lightcurves for EBOP MAVEN.")
            mags_bins = estimator.mags_feature_bins
            wrap_phase = u.Quantity(estimator.mags_feature_wrap_phase or (0.5 + phiS / 2))
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
            dr3_id, ra, dec, parallax, G_mag = dal.read_values(target_id, "gaia_dr3_id", "ra",
                                                               "dec", "parallax", "G_mag")
            print("Estimating fitting input value for L3 with Gaia DR3.")
            coords = SkyCoord(ra=nominal_value(ra) * u.deg, dec=nominal_value(dec) * u.deg,
                              distance=(1000 / nominal_value(parallax)) * u.pc, frame="icrs")
            G_mag = G_mag or lcs[0].meta["TESSMAG"]
            l3 = pipeline.estimate_l3_with_gaia(coords, 120, dr3_id, G_mag, max_l3=.1, verbose=True)


            # Clip masks retain only obs within 2.5 d of an eclipse for fitting. Can optimise
            # fitting and is especially useful where we have previously flattened the lightcurves/
            do_clip = config.create_clip_mask \
                    or (config.create_clip_mask is None and do_flatten)
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
            # Extract any fitting overrides from the target config.
            # May contain LD params which are handled below.
            fit_overrides = copy.deepcopy(config.get("jktebop_overrides", {}))

            # Generate initial LD params based on the system temperature & log(g)
            # (deferring to any overrides)
            Teff_sys, logg_sys = dal.read_values(target_id, "Teff_sys", "logg_sys")
            Teff_sys = lcs[0].meta["TEFF"] or 5650 if np.isnan(nominal_value(Teff_sys)) else Teff_sys
            logg_sys = lcs[0].meta["LOGG"] or 4.0 if np.isnan(nominal_value(logg_sys)) else logg_sys
            teffr = predictions_dict["J"]**0.25 # J*k^2 a proxy for LR with TeffR~=((J*k^2)/k^2)^1/4
            ld_params = pipeline.pop_and_complete_ld_config(fit_overrides,
                *nominal_values((Teff_sys, Teff_sys*teffr.n) if teffr.n < 1 else (Teff_sys/teffr.n, Teff_sys)),
                *nominal_values((logg_sys, logg_sys)),
                "pow2", True)

            # The refl flags can be 0 (fixed), 1 (fitted) or -1 (calculated from sys geometry)
            refl_fit = -1 if (morph <= FLATTEN_TH) else 1

            in_params = {
                # Mass ratio (qphot), can be -1 (force spherical) or a specified ratio value
                "qphot": -1 if (morph <= FLATTEN_TH) else predictions_dict["k"]**1.4,
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
                                 "period", "ecc", "bP", "LR", "reflA", "reflB"]

            # If max_workers >1 progress updates will occur after each attempt is complete, but overall elapsed
            # time is reduced. If set to 1, tasks are serialized but more frequent progress updates will occur.
            t0s = [nominal_value(lc.meta.get("t0", t0)) for lc in lcs]
            fitted_param_dicts = pipeline.fit_target_lightcurves(lcs,
                                                                 input_params=in_params,
                                                                 read_keys=fitted_param_keys,
                                                                 t0=t0s,
                                                                 task=3,
                                                                 max_workers=8,
                                                                 max_attempts=3,
                                                                 timeout=900,
                                                                 file_prefix="fit-lrs")

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
            TeffR = (summary_params["LR"] / summary_params["k"]**2)**0.25
            print(f"    Teff_ratio: {TeffR:12.6f} (calculated from LR & k)")

            # Finally, store the params and the flag that indicates LC fitting has completed
            params = { k: summary_params[k]
                      for k in ["rA_plus_rB", "k", "J", "ecosw", "esinw", "bP", "inc", "L3", "LR"] }
            params["TeffR"] = TeffR
            params["fitted_lcs"] = True
            dal.write_values(target_id, **params)


        print(f"\nCompleted at {datetime.now():%Y-%m-%d %H:%M:%S%z %Z}")
