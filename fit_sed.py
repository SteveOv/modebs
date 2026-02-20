""" Pipeline Stage 3 - MCMC fitting of target SEDs """
# pylint: disable=no-member, invalid-name
from inspect import getsourcefile
from pathlib import Path
import warnings
import argparse
from datetime import datetime
from contextlib import redirect_stdout

import numpy as np
import astropy.units as u
from astropy.table import QTable
from astropy.coordinates import SkyCoord

# pylint: disable=line-too-long, wrong-import-position
warnings.filterwarnings("ignore", "Using UFloat objects with std_dev==0 may give unexpected results.", category=UserWarning)
from uncertainties import ufloat, nominal_value

# Dereddening of SEDS
from dust_extinction.parameter_averages import G23

from sed_fit.stellar_grids import get_stellar_grid
from sed_fit.fitter import create_theta, minimize_fit, mcmc_fit

from libs import extinction
from libs.sed import get_sed_for_target, group_and_average_fluxes, create_outliers_mask
from libs.iohelpers import Tee
from libs.targets import Targets

THIS_STEM = Path(getsourcefile(lambda: 0)).stem

NUM_STARS = 2

subs = "ABCDEFGHIJKLM"
theta_labels = np.array([(f"Teff{subs[st]}", u.K) for st in range(NUM_STARS)] \
                        +[(f"logg{subs[st]}", u.dimensionless_unscaled) for st in range(NUM_STARS)] \
                        +[(f"R{subs[st]}", u.Rsun) for st in range(NUM_STARS)] \
                        +[("dist", u.pc), ("av", u.dimensionless_unscaled)])


fit_mask = np.array([True] * NUM_STARS      # teff
                    + [False] * NUM_STARS   # logg
                    + [True] * NUM_STARS    # radius
                    + [True]                # dist
                    + [False])              # av (we've handled av by derredening the SED)

if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Pipeline stage 3: fitting target SED.")
    ap.add_argument("-tf", "--targets-file", dest="targets_file", type=Path, required=False,
                    help="json file containing the details of the targets to fit")
    ap.add_argument("-wd", "--write-diags", dest="write_diags", action="store_true", required=False,
                    help="write a second, human readable output file for diagnostics")
    ap.set_defaults(targets_file=Path("./config/plato-lops2-tess-ebs-explicit-targets.json"),
                    write_diags=False)
    args = ap.parse_args()
    drop_dir = Path.cwd() / f"drop/{args.targets_file.stem}"
    args.input_file = args.output_file = drop_dir / "working-set.table"


    with redirect_stdout(Tee(open(drop_dir / f"{THIS_STEM}.log", "w", encoding="utf8"))):
        print(f"Started at {datetime.now():%Y-%m-%d %H:%M:%S%z %Z}")

        targets_config = Targets(args.targets_file)
        print(f"Read in the configuration from '{args.input_file}'",
              f"which contains {targets_config.count()} target(s).")

        # Open the targets table and the configs
        tdata = QTable.read(args.input_file)
        to_fit_row_ixs = np.where(tdata["fitted_radii"] == False)[0] # pylint: disable=singleton-comparison

        to_fit_count = len(to_fit_row_ixs)
        print(f"Reading '{args.input_file}' which contains {to_fit_count} target(s) to be fitted."
              f"\nWill write updated data to '{args.output_file}'")

        # Extinction model: G23 (Gordon et al., 2023) Milky Way R(V) filter gives us broad coverage
        ext_model = G23(Rv=3.1)
        ext_wl_range = np.reciprocal(ext_model.x_range) * u.um # x_range has implicit units of 1/um
        print(f"Using {ext_model.__class__.__name__} extinction model which covers the range from",
            f"{min(ext_wl_range):unicode} to {max(ext_wl_range):unicode}.")


        # Model SED grid based on atmosphere models with known filters pre-applied to non-reddened fluxes
        # Available grids: BtSettlGrid or KuruczGrid
        model_grid = get_stellar_grid("BtSettlGrid", extinction_model=ext_model, verbose=True)
        print("Loaded grid based on synthetic models, covering the ranges:")
        print(f"wavelength {model_grid.wavelength_range * model_grid.wavelength_unit:unicode},",
            f"Teff {model_grid.teff_range * model_grid.teff_unit:unicode},",
            f"logg {model_grid.logg_range * model_grid.logg_unit:unicode}",
            f"\nand metallicity {model_grid.metal_range * u.dimensionless_unscaled:unicode},",
            f"with fluxes returned in units of {model_grid.flux_unit:unicode}")

        # Fixed priors limits for MCMC fit
        teff_limits = model_grid.teff_range
        radius_limits = (0.1, 100)

        for row_ix in to_fit_row_ixs:
            trow = tdata[row_ix]
            target = trow["target"]
            print("\n\n============================================================")
            print(f"Processing target {row_ix+1} of {to_fit_count}: {target}")
            print("============================================================")

            target_config = targets_config.get(target)

            teff_sys = nominal_value(5650 if np.ma.is_masked(trow["Teff_sys"]) else ufloat(trow["Teff_sys"].value, 0))
            logg_sys = nominal_value(4.0 if np.ma.is_masked(trow["logg_sys"]) else ufloat(trow["logg_sys"].value, 0))

            # Read in the SED for this target and de-duplicate (measurements may appear multiple times).
            # Plots are unit agnostic and plot wl [um] and vF(v) [W/m^2] on x and y.
            sed = get_sed_for_target(target, trow["main_id"],
                                     radius=0.1, remove_duplicates=True, verbose=True)
            sed = group_and_average_fluxes(sed, verbose=True)

            # Filter SED to those covered by our models and also remove any outliers
            model_mask = np.ones((len(sed)), dtype=bool)
            model_mask &= model_grid.has_filter(sed["sed_filter"])
            model_mask &= (sed["sed_wl"] >= min(ext_wl_range)) \
                        & (sed["sed_wl"] <= max(ext_wl_range)) \
                        & (sed["sed_wl"] >= min(model_grid.wavelength_range)) \
                        & (sed["sed_wl"] <= max(model_grid.wavelength_range))
            sed = sed[model_mask]

            out_mask = create_outliers_mask(sed, teff_sys, [trow["TeffR"]],
                                            min_unmasked=15, verbose=True)
            sed = sed[~out_mask]
            sed.sort(["sed_wl"])
            print(f"{len(sed)} unique SED observation(s) retained after range and outlier filtering",
                  "with the units for flux, frequency and wavelength being",
                  ", ".join(f"{sed[f].unit:unicode}" for f in ["sed_flux", "sed_freq", "sed_wl"]))            


            # Deredden the SED
            av = target_config.get("av", None)
            if not av:
                coords = SkyCoord(ra=trow["ra"], dec=trow["dec"],
                                 distance=(1000 / trow["parallax"]).value * u.pc, frame="icrs")
                # Get the mean of the various catalogues, prioritising converged results
                extfuncs = ["get_gontcharov_ebv", "get_bayestar_ebv"]
                for conv in [True, False]:
                    avs = [v for v, flags in extinction.get_av(coords, extfuncs, ext_model.Rv, True)
                                if flags.get("converged", False) == conv]
                    if len(avs):
                        av = np.mean(avs)
                        print(f"Found mean extinction of {len(avs)} catalogue(s): A_V = {av:.6f}")
                        break

            if av:
                print(f"Dereddening observations with A_V={av:.3f}")
                sed["sed_der_flux"] = sed["sed_flux"] / \
                                        ext_model.extinguish(sed["sed_wl"].to(u.um), Av=av)
            else:
                sed["sed_der_flux"] = sed["sed_flux"]


            # Set up the MCMC fitting theta and priors. For now, hard coded to 2 stars.
            # The ratios wrt the primary components - the prior_func ignores the 0th item
            teff_ratio = ufloat(trow["TeffR"], trow["TeffR_err"])
            rad_ratio = ufloat(trow["k"], trow["k_err"])
            teff_ratios = [ufloat(teff_ratio.n, max(teff_ratio.s, teff_ratio.n*0.1))] * NUM_STARS
            rad_ratios = [ufloat(rad_ratio.n, max(rad_ratio.s, rad_ratio.n*0.1))] * NUM_STARS
            dist_prior = ufloat(coords.distance.value, coords.distance.value * 0.05)

            def ln_prior_func(theta: np.ndarray[float]) -> float:
                """
                The fitting prior callback function to evaluate the current set of candidate
                parameters (theta), returning a single ln(value) indicating their "goodness".
                """
                # pylint: disable=cell-var-from-loop
                teffs = theta[0:NUM_STARS]
                radii = theta[NUM_STARS*2:NUM_STARS*3]
                dist = theta[-2]

                # Limit criteria checks - hard pass/fail on these
                if not all(teff_limits[0] <= t <= teff_limits[1] for t in teffs) or \
                    not all(radius_limits[0] <= r <= radius_limits[1] for r in radii):
                    return np.inf

                # Gaussian prior criteria: g(x) = 1/(σ*sqrt(2*pi)) * exp(-1/2 * (x-µ)^2/σ^2)
                # Omitting scaling expressions for now and note the implicit ln() cancelling the exp
                rval = 0
                for star_ix in range(1, NUM_STARS):
                    rval += ((teffs[star_ix] / teffs[0]-teff_ratios[star_ix].n) / teff_ratio.s)**2
                    rval += ((radii[star_ix] / radii[0]-rad_ratios[star_ix].n) / teff_ratio.s)**2
                rval += ((dist - dist_prior.n) / dist_prior.s)**2
                return 0.5 * rval


            # Set a reasonable starting position for the fit
            theta0 = create_theta(teffs=nominal_value(teff_sys),
                                  loggs=nominal_value(logg_sys),
                                  radii=1.0,
                                  dist=coords.distance.to(u.pc).unmasked.value,
                                  av=0,
                                  nstars=NUM_STARS,
                                  verbose=True)


            # Prepare the data to be fitted and perform a "quick" minimize fit
            x = model_grid.get_filter_indices(sed["sed_filter"])
            y = (sed["sed_der_flux"].quantity * sed["sed_freq"].quantity)\
                                        .to(model_grid.flux_unit, equivalencies=u.spectral()).value
            y_err = (sed["sed_eflux"].quantity * sed["sed_freq"].quantity)\
                                        .to(model_grid.flux_unit, equivalencies=u.spectral()).value

            theta_min_fit, _ = minimize_fit(x, y, y_err=y_err, theta0=theta0, fit_mask=fit_mask,
                                            ln_prior_func=ln_prior_func, stellar_grid=model_grid,
                                            verbose=True)


            # Then perform a full MCMC fit based on the output from the "quick" fit
            thin_by = 10 # sample every nth step from the chain
            theta_mcmc_fit, _ = mcmc_fit(x, y, y_err, theta0=theta_min_fit, fit_mask=fit_mask,
                                         ln_prior_func=ln_prior_func, stellar_grid=model_grid,
                                         nwalkers=100, nsteps=100000, thin_by=thin_by, seed=42,
                                         early_stopping=True, processes=8, progress=True, verbose=True)

            print(f"Final parameters for {target} with nominals & 1-sigma uncertainties",
                  "from MCMC fit ([known value])")
            for (k, unit), theta_val, mask in zip(theta_labels, theta_mcmc_fit, fit_mask):
                known = ""
                if k == "dist":
                    known = f"({coords.distance.to(u.pc).value:.3f} pc)"
                elif target_config.get("labels", {}).get(k, None) is not None:
                    lvalue = ufloat(target_config.labels.get(k, np.NaN),
                                    target_config.labels.get(k + "_err", 0))
                    known = f"({lvalue:.3f} {unit:unicode})"
                print(f"{k:>12s}{'*' if mask else ' ':s} = {theta_val:.3f} {unit:unicode} \t", known)

                # *** also updates the target data ***
                if k not in ["av"]:
                    tdata[row_ix][k] = theta_val.n * unit
                    tdata[row_ix][f"{k}_err"] = theta_val.s * unit

            tdata[row_ix]["fitted_radii"] = True

            # Update the working set after each target
            print(f"Writing updated data set to {args.output_file}")
            tdata.write(args.output_file, format="votable", overwrite=True)

            if args.write_diags:
                diag_file = drop_dir / f"{THIS_STEM}.diag"
                tdata.write(diag_file, format="ascii.fixed_width_two_line",
                            header_rows=["name", "dtype", "unit"], overwrite=True)

        print(f"\nCompleted at {datetime.now():%Y-%m-%d %H:%M:%S%z %Z}")
