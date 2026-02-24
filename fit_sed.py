""" Pipeline Stage 3 - MCMC fitting of target SEDs """
# pylint: disable=no-member, invalid-name
from inspect import getsourcefile
from pathlib import Path
import warnings
import argparse
from datetime import datetime
from contextlib import redirect_stdout
import traceback

import numpy as np
import astropy.units as u
from astropy.coordinates import SkyCoord

# pylint: disable=line-too-long, wrong-import-position
warnings.filterwarnings("ignore", "Using UFloat objects with std_dev==0 may give unexpected results.", category=UserWarning)
from uncertainties import ufloat, nominal_value

# Dereddening of SEDS
from dust_extinction.parameter_averages import G23

from sed_fit.stellar_grids import get_stellar_grid
from sed_fit.fitter import create_theta, minimize_fit, mcmc_fit

from libs import pipeline, extinction
from libs.sed import get_sed_for_target, group_and_average_fluxes, create_outliers_mask
from libs.iohelpers import Tee
from libs.targets import Targets
from libs.pipeline_dal import QTableFileDal

THIS_STEM = Path(getsourcefile(lambda: 0)).stem

NUM_STARS = 2

subs = "ABCDEFGHIJKLM"
theta_labels = np.array([(f"Teff{subs[st]}", u.K) for st in range(NUM_STARS)] \
                    +[(f"logg{subs[st]}", u.dimensionless_unscaled) for st in range(NUM_STARS)] \
                    +[(f"R{subs[st]}", u.Rsun) for st in range(NUM_STARS)] \
                    +[("dist", u.pc), ("Av", u.dimensionless_unscaled)])


fit_mask = np.array([True] * NUM_STARS      # teff
                    + [False] * NUM_STARS   # logg
                    + [True] * NUM_STARS    # radius
                    + [True]                # dist
                    + [False])              # av (we've handled av by derredening the SED)

if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Pipeline stage 3: fitting target SED.")
    ap.add_argument("-tf", "--targets-file", dest="targets_file", type=Path, required=False,
                    help="json file containing the details of the targets to fit")
    ap.add_argument("-ms", "--max-steps", dest="max_mcmc_steps", type=int, required=False,
                    help="the maximum number of MCMC steps to run for [100 000]")
    ap.set_defaults(targets_file=Path("./config/plato-lops2-tess-ebs-explicit-targets.json"),
                    max_mcmc_steps=100000, mcmc_walkers=100, mcmc_thin_by=10)
    args = ap.parse_args()
    drop_dir = Path.cwd() / f"drop/{args.targets_file.stem}"
    args.working_set_file = drop_dir / "working-set.table"


    with redirect_stdout(Tee(open(drop_dir / f"{THIS_STEM}.log", "a", encoding="utf8"))) as log:
        print(f"Started at {datetime.now():%Y-%m-%d %H:%M:%S%z %Z}")

        targets_config = Targets(args.targets_file)
        print(f"Read in the configuration from '{args.targets_file}'",
              f"which contains {targets_config.count()} target(s) that have not been excluded.")

        wset = QTableFileDal(args.working_set_file)
        to_fit_target_ids = list(wset.yield_keys("fitted_lcs", "fitted_sed",
                                                 where=lambda fl, fs: fl == True and fs == False)) # pylint: disable=singleton-comparison
        to_fit_count = len(to_fit_target_ids)
        print(f"The working-set indicates there are {to_fit_count} targets to be fitted.")

        # Extinction model: G23 (Gordon et al., 2023) Milky Way R(V) filter gives us broad coverage
        ext_model = G23(Rv=3.1)
        ext_wl_range = np.reciprocal(ext_model.x_range) * u.um # x_range has implicit units of 1/um
        print(f"Using {ext_model.__class__.__name__} extinction model which covers the range",
              f"from {min(ext_wl_range):unicode} to {max(ext_wl_range):unicode}.")

        # Model SED grid based on atmosphere models with known filters pre-applied to non-reddened
        # fluxes. Available grids: BtSettlGrid or KuruczGrid
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

        for fit_counter, target_id in enumerate(to_fit_target_ids, start=1):
            try:
                config = targets_config.get(target_id)
                print("\n\n============================================================")
                print(f"Processing target {fit_counter} of {to_fit_count}: {target_id}")
                print("============================================================")


                main_id, k, TeffR, Teff_sys, logg_sys, st = wset.read_values(target_id,
                                        "main_id", "k", "TeffR", "Teff_sys", "logg_sys", "spt")
                Teff_sys = Teff_sys or pipeline.get_teff_from_spt(st) or 5650
                logg_sys = logg_sys or 4.0

                ra, dec, parallax = wset.read_values(target_id, "ra", "dec", "parallax")
                coords = SkyCoord(ra=nominal_value(ra) * u.deg, dec=nominal_value(dec) * u.deg,
                                  distance=(1000 / nominal_value(parallax)) * u.pc, frame="icrs")


                # Get the SED for this target and de-duplicate (obs may appear multiple times).
                print()
                sed = get_sed_for_target(target_id, main_id,
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

                out_mask = create_outliers_mask(sed, Teff_sys, [TeffR], 15, verbose=True)
                sed = sed[~out_mask]
                sed.sort(["sed_wl"])
                print(f"{len(sed)} unique SED observation(s) retained after range & outlier",
                      "filtering with the units for flux, frequency and wavelength being",
                      ", ".join(f"{sed[f].unit:unicode}" for f in ["sed_flux","sed_freq","sed_wl"]))            


                # Deredden the SED
                print()
                print(f"Locating extinction data based on {target_id} {coords}".replace("\n", ""))
                if (Av := config.get("Av", None)) is None:
                    # Get the mean of the various catalogues, prioritising converged results
                    efunc = ["get_gontcharov_ebv", "get_bayestar_ebv"]
                    for conv in [True, False]:
                        avs = [v for v,flags in extinction.get_av(coords, efunc, ext_model.Rv, True)
                                    if flags.get("converged", False) == conv]
                        if len(avs):
                            Av = np.mean(avs)
                            print(f"Found mean extinction of {len(avs)} catalogue(s): A_V={Av:.6f}")
                            break

                if Av:
                    print(f"Dereddening observations with A_V={Av:.3f}")
                    sed["sed_der_flux"] = \
                            sed["sed_flux"] / ext_model.extinguish(sed["sed_wl"].to(u.um), Av=Av)
                else:
                    sed["sed_der_flux"] = sed["sed_flux"]


                # Set up the MCMC fitting theta and priors. For now, hard coded to 2 stars.
                # The ratios are wrt the primary components - the prior_func ignores the 0th item
                print("\nSetting up the fitting priors and the ln_prior_func() callback.")
                Teff_ratios = [ufloat(TeffR.n, max(TeffR.s, TeffR.n*0.1))] * NUM_STARS
                rad_ratios = [ufloat(k.n, max(k.s, k.n*0.1))] * NUM_STARS
                dist_prior = ufloat(coords.distance.value, coords.distance.value * 0.05)
                print(f"Teff ratios={', '.join(f'{r:.3f}' for r in Teff_ratios[1:])}",
                      f"radius_ratios={', '.join(f'{r:.3f}' for r in rad_ratios[1:])},",
                      f"dist={dist_prior:.3f},",
                      f"Teff_limits={teff_limits}, radius_limits={radius_limits}")

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
                    # Omitting scaling expressions and note the implicit ln() cancelling the exp
                    rval = 0
                    for star_ix in range(1, NUM_STARS):
                        rval += ((teffs[star_ix] / teffs[0]-Teff_ratios[star_ix].n) / TeffR.s)**2
                        rval += ((radii[star_ix] / radii[0]-rad_ratios[star_ix].n) / TeffR.s)**2
                    rval += ((dist - dist_prior.n) / dist_prior.s)**2
                    return 0.5 * rval


                print("\nSetting up the starting position (theta0) for fitting.")
                theta0 = create_theta(teffs=nominal_value(Teff_sys),
                                      loggs=nominal_value(logg_sys),
                                      radii=1.0,
                                      dist=coords.distance.to(u.pc).value,
                                      av=0,
                                      nstars=NUM_STARS,
                                      verbose=True)
                x = model_grid.get_filter_indices(sed["sed_filter"])
                y = (sed["sed_der_flux"].quantity * sed["sed_freq"].quantity)\
                                        .to(model_grid.flux_unit, equivalencies=u.spectral()).value
                y_err = (sed["sed_eflux"].quantity * sed["sed_freq"].quantity)\
                                        .to(model_grid.flux_unit, equivalencies=u.spectral()).value


                print("\nPerforming an initial 'quick' minimize fit. Values marked * are fitted.")
                theta_min_fit, _ = minimize_fit(x, y, y_err=y_err, theta0=theta0, fit_mask=fit_mask,
                                                ln_prior_func=ln_prior_func,
                                                stellar_grid=model_grid, verbose=True)


                print("\nPerforming a full MCMC fit from the output from the 'quick' fit.",
                      "Values marked * are fitted.")
                theta_mcmc_fit, _ = mcmc_fit(x, y, y_err,
                                             theta0=theta_min_fit,
                                             fit_mask=fit_mask,
                                             ln_prior_func=ln_prior_func,
                                             stellar_grid=model_grid,
                                             nwalkers=args.mcmc_walkers,
                                             nsteps=args.max_mcmc_steps,
                                             thin_by=args.mcmc_thin_by,
                                             seed=42,
                                             early_stopping=True,
                                             processes=8,
                                             progress=True,
                                             verbose=True)


                print(f"\nFinal parameters for {target_id} with nominals & 1-sigma uncertainties",
                        "from MCMC fit ([known value])")
                write_params = {}
                for (k, unit), val, mask in zip(theta_labels, theta_mcmc_fit, fit_mask):
                    label = ""
                    if k == "dist":
                        label = f"({coords.distance.to(u.pc).value:.3f} pc)"
                    elif config.get("labels", {}).get(k, None) is not None:
                        lval = ufloat(config.labels.get(k, np.NaN), config.labels.get(k+"_err", 0))
                        label = f"({lval:.3f} {unit:unicode})"
                    print(f"{k:>12s}{'*' if mask else ' ':s} = {val:.3f} {unit:unicode} \t", label)

                    # *** also updates the target data ***
                    if mask:
                        write_params[k] = val


                # Finally, store the params and the flag that indicates SED fitting has completed
                print(f"\nWriting fitted params for {list(write_params.keys())} to working-set.")
                wset.write_values(target_id, fitted_sed=True, errors="", **write_params)

            except Exception as exc: # pylint: disable=broad-exception-caught
                print(f"{target_id}: Failed with the following exception. Depending on the nature",
                    "of the failure it may be possible to rerun this module to fit failed targets.")
                traceback.print_exception(exc, file=log)
                wset.write_values(target_id, fitted_sed=False, errors=type(exc).__name__)

        print(f"\nCompleted at {datetime.now():%Y-%m-%d %H:%M:%S%z %Z}")
