#!/usr/bin/env python3
""" Pipeline Stage 3 - MCMC fitting of target SEDs """
# pylint: disable=no-member, invalid-name
from inspect import getsourcefile
from pathlib import Path
import warnings
import argparse
from datetime import datetime
from contextlib import redirect_stdout
import traceback
from time import sleep
from textwrap import fill

import numpy as np
from matplotlib import use as mpl_use
import matplotlib.pyplot as plt
import astropy.units as u
from astropy.coordinates import SkyCoord

# pylint: disable=line-too-long, wrong-import-position
warnings.filterwarnings("ignore", "Using UFloat objects with std_dev==0 may give unexpected results.", category=UserWarning)
from uncertainties import ufloat, UFloat, nominal_value as nom_val, std_dev
from uncertainties.unumpy import nominal_values as nom_vals

# Dereddening of SEDS
from dust_extinction.parameter_averages import G23

import corner
from sed_fit.stellar_grids import get_stellar_grid
from sed_fit.fitter import create_theta, minimize_fit, mcmc_fit, model_func, samples_from_sampler
from sed_fit.fitter import _print_theta # pylint: disable=protected-access

from libs import extinction, plots
from libs.pipeline import PipelineError
from libs.sed import get_sed_for_target
from libs.iohelpers import Tee
from libs.targets import Targets
from libs.pipeline_dal import create_dal
from libs.utils import to_file_safe_str

THIS_STEM = Path(getsourcefile(lambda: 0)).stem

NSTARS = 2
subs = ["ABCDEFGHIJKLM"[n] for n in range(NSTARS)]
theta_labels = np.array([f"$T_{{\\rm eff,{sub}}} / {{\\rm K}}$" for sub in subs]
                      + [f"$\\log{{g}}_{{\\rm {sub}}}$" for sub in subs]
                      + [f"$R_{{\\rm {sub}}} / {{\\rm R_{{\\odot}}}}$" for sub in subs]
                      + ["$D / {\\rm pc}$", "${\\rm A_{V}}$"])

theta_params_and_units = np.array([(f"Teff{sub}", u.K) for sub in subs]
                                + [(f"logg{sub}", u.dimensionless_unscaled) for sub in subs]
                                + [(f"R{sub}", u.Rsun) for sub in subs]
                                + [("dist", u.pc), ("Av", u.dimensionless_unscaled)])

# Dictates which params in theta are fitted (True) and which are held fixed (False)
fit_mask = np.array([True] * NSTARS      # Teff
                  + [False] * NSTARS     # logg
                  + [True] * NSTARS      # radius
                  + [True]                  # dist
                  + [False])                # Av (we handle av by derredening the SED)

# Use a non-interactive matplotlib backend to avoid threading errors (issue #36).
mpl_use("agg")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Pipeline stage 3: fitting target SED.")
    ap.add_argument(dest="targets_file", type=Path, metavar="TARGETS_FILE",
                    help="json file containing the details of the targets to fit")
    ap.add_argument("-pf", "--plot-figs", dest="plot_figs", action="store_true", required=False,
                    help="plot figs for each target as the process progresses")
    ap.add_argument("-ms", "--mcmc-steps", dest="max_mcmc_steps", type=int, required=False,
                    help="the maximum number of MCMC steps to run for [100 000]")
    ap.add_argument("-mw", "--mcmc-walkers", dest="mcmc_walkers", type=int, required=False,
                    help="the number of MCMC walkers to use [100]")
    ap.add_argument("-mp", "--mcmc-processes", dest="mcmc_processes", type=int, required=False,
                    help="the number of concurrent MCMC processes to run [8]")
    ap.add_argument("-mo", "--mcmc-off", dest="do_mcmc_fit", action="store_false", required=False,
                    help="suppress running of MCMC for parameters")
    ap.set_defaults(plot_figs=False, figs_type="png", figs_dpi=100, do_mcmc_fit=True,
                    max_mcmc_steps=100000, mcmc_walkers=100, mcmc_thin_by=10, mcmc_processes=8)
    args = ap.parse_args()
    drop_dir = Path.cwd() / f"drop/{args.targets_file.stem}"

    with redirect_stdout(Tee(open(drop_dir / f"{THIS_STEM}.log", "a", encoding="utf8"))) as log:
        print("\n\n============================================================")
        print(f"Started {THIS_STEM} at {datetime.now():%Y-%m-%d %H:%M:%S%z %Z}")
        print("============================================================")
        print(f"\nThe targets configuration file:   {args.targets_file}")
        print(f"Directory for data, logs & plots: {drop_dir}")

        targets_config = Targets(args.targets_file)
        print(f"Read in the configuration from '{args.targets_file.name}'",
              f"which contains {targets_config.count()} target(s) that have not been excluded.")

        dal_kwargs = targets_config.get("dal_kwargs", {})
        dal_kwargs.setdefault("file", drop_dir / "working-set.table")
        dal = create_dal(targets_config.get("dal_type", "QTableFileDal"), True, **dal_kwargs)
        to_fit_criteria = { "fitted_lcs": True, "fitted_sed": False }
        to_fit_count = dal.count_where(**to_fit_criteria)
        print(f"The working-set indicates there are {to_fit_count} targets to be fitted.")

        # Extinction model: G23 (Gordon et al., 2023) Milky Way R(V) filter gives us broad coverage
        ext_model = G23(Rv=3.1)
        ext_wl_range = np.reciprocal(ext_model.x_range) * u.um # x_range has implicit units of 1/um
        print(f"\nUsing {ext_model.__class__.__name__} extinction model which covers the range",
              f"from {min(ext_wl_range):unicode} to {max(ext_wl_range):unicode}.\n")

        # Model SED grid based on atmosphere models with known filters pre-applied to non-reddened
        # fluxes. Available grids: BtSettlGrid or KuruczGrid
        model_grid = get_stellar_grid(targets_config.get("stellar_grid", "BtSettlGrid"),
                                      extinction_model=ext_model, verbose=True)
        print("Loaded grid based on synthetic models, covering the ranges:")
        print(f"wavelength {model_grid.wavelength_range * model_grid.wavelength_unit:unicode},",
              f"Teff {model_grid.teff_range * model_grid.teff_unit:unicode},",
              f"logg {model_grid.logg_range * model_grid.logg_unit:unicode}",
              f"\nand metallicity {model_grid.metal_range * u.dimensionless_unscaled:unicode},",
              f"with fluxes returned in units of {model_grid.flux_unit:unicode}")

        # Fixed priors limits for MCMC fit
        teff_limits = model_grid.teff_range
        logg_limits = model_grid.logg_range
        radius_limits = (0.05, 100)

        for fit_counter, trow in enumerate(dal.acquire_next_row(**to_fit_criteria), start=1):
            if fit_counter > 1:
                sleep(10) # Give emcee a quick break, in prep for the next target

            try:
                target_id = trow.key
                print("\n\n------------------------------------------------------------")
                print(f"Processing target {fit_counter} of {to_fit_count}: {target_id}")
                print("------------------------------------------------------------")
                config = targets_config.get_target_config(target_id)
                if args.plot_figs:
                    figs_dir = drop_dir / "figs" / to_file_safe_str(target_id)
                    figs_dir.mkdir(parents=True, exist_ok=True)

                coords = SkyCoord(ra=nom_val(trow.ra_coord) * u.deg,
                                  dec=nom_val(trow.dec_coord) * u.deg,
                                  distance=(1000 / nom_val(trow.parallax)) * u.pc,
                                  frame="icrs")

                # Output some known details of the target system
                print()
                print(fill(f"Details:{config.get('details', '')}", subsequent_indent="\t"))
                print(fill(f"Notes:  {config.get('notes', '')}", subsequent_indent="\t"))
                print(f"SpT:\t{trow.spt or config.get('SpT', '')}")
                print(f"morph:\t{trow.morph or -1:.3f}\nDR3 ruwe:\t{trow.ruwe or -1:.3f}")
                print(f"Teff_sys:\t{trow.Teff_sys or -1:.0f}\nlogg_sys:\t{trow.logg_sys or -1:.3f}")

                if trow.ruwe is not None and trow.ruwe > 1.4:
                    trow.append_warning("ruwe>1.4")

                # Get the extinction coefficient, based on the coords
                print()
                if (Av := config.get("A_V", config.get("E(B-V)", 0) * ext_model.Rv)) > 0:
                    print(f"Found extinction override in target config giving A_V={Av:.6f}")
                else:
                    # Get the mean of the various catalogues, prioritising reliable results
                    print(f"Getting extinction data based on {target_id} {coords}".replace("\n",""))
                    avs = np.array([*extinction.iterate(coords, rv=ext_model.Rv, verbose=True)]).T
                    if any(rmask := np.array(avs[1], dtype=bool)):
                        # We have some reliable extinction values, use only these
                        Av = np.mean(avs[0][rmask])
                        print(f"Using the mean of {sum(rmask)} reliable value(s): A_V={Av:.6f}")
                    else:
                        Av = np.mean(avs[0])
                        print(f"Using the mean of {len(rmask)} value(s): A_V={Av:.6f}")
                        trow.append_warning("unreliable A_V")
                Av = Av or 0


                # Get the SED for this target and de-duplicate (obs may appear multiple times).
                print()
                sed = get_sed_for_target(target_id, trow.search_term,
                                         radius=0.1, remove_duplicates=True, verbose=True)
                if sed is None or len(sed) == 0:
                    raise PipelineError(target_id, f"No SED observations for '{trow.search_term}'")

                # Filter SED to those covered by our models and also remove any outliers
                model_mask = np.ones((len(sed)), dtype=bool)
                model_mask &= model_grid.has_filter(sed["sed_filter"])
                model_mask &= ~np.isin(sed["sed_filter"], config.get("sed_filter_exclusions", []))
                model_mask &= ~np.isin(sed["_tabname"], config.get("sed_tabname_exclusions", []))
                model_mask &= (sed["sed_wl"] >= min(ext_wl_range)) \
                            & (sed["sed_wl"] <= max(ext_wl_range)) \
                            & (sed["sed_wl"] >= min(model_grid.wavelength_range)) \
                            & (sed["sed_wl"] <= max(model_grid.wavelength_range))
                sed = sed[model_mask]

                print("Creating de-reddened SED observations")
                sed["sed_der_flux"] = sed["sed_flux"] \
                                    / ext_model.extinguish(sed["sed_wl"].to(u.um), Av=Av)
                sed.sort(["sed_wl"])
                print(f"{len(sed)} unique SED observation(s) retained after range and exclusion",
                      "filtering,\nwith the units for flux, frequency and wavelength being",
                      ", ".join(f"{sed[f].unit:unicode}" for f in ["sed_flux","sed_freq","sed_wl"]))            

                if args.plot_figs:
                    print("\nCreating SED observations plot")
                    _fluxes = [sed["sed_flux"], sed["sed_der_flux"]] if Av else [sed["sed_flux"]]
                    fig = plots.plot_sed(sed["sed_wl"].quantity, _fluxes, [sed["sed_eflux"]]*2,
                                    fmts=["or", ".b"], title=f"{target_id} SED",
                                    labels=["observed",f"dereddened\n($A_{{\\rm V}}={Av:.3f})$"])
                    fig.savefig(figs_dir / f"sed-observations.{args.figs_type}", dpi=args.figs_dpi)
                    plt.close(fig)


                # Set up the MCMC fitting theta and priors. For now, hard coded to 2 stars.
                # The ratios are wrt the primary components - the prior_func ignores the 0th item
                print("\nSetting up the fitting priors and the ln_prior_func() callback.")
                TeffR, radR = trow.TeffR, trow.k
                TeffR_priors = tuple([1]+ [ufloat(TeffR.n, max(TeffR.s, TeffR.n * .05))]*(NSTARS-1))
                radR_priors = tuple([1] + [ufloat(radR.n, max(radR.s, radR.n * .05))]*(NSTARS-1))
                dist_prior = 1000 / trow.parallax
                if not isinstance(dist_prior, UFloat) or not dist_prior.s:
                    dist_prior = ufloat(nom_val(dist_prior), nom_val(dist_prior) * .05)
                print(f"Priors: Teff ratios=({', '.join(f'{r:.3f}' for r in TeffR_priors)}),",
                      f"radius ratios=({', '.join(f'{r:.3f}' for r in radR_priors)}),",
                      f"dist={dist_prior:.3f},",
                      f"Teff_limits={teff_limits}, radius_limits={radius_limits}")

                def ln_prior_func(theta: np.ndarray[float]) -> float:
                    """
                    The fitting prior callback function to evaluate the current set of candidate
                    parameters, returning a single negative ln(value) indicating their "goodness".

                    Return negative value as fitter will maximize the sum of this & the negative val
                    of its ln_prob_func. The fitter knows to flip the sign if running a minimize fit
                    """
                    # pylint: disable=cell-var-from-loop
                    teffs = theta[0:NSTARS]
                    radii = theta[NSTARS*2:NSTARS*3]
                    dist = theta[-2]

                    # Limit criteria checks - hard pass/fail on these
                    if not all(teff_limits[0] <= t <= teff_limits[1] for t in teffs) or \
                        not all(radius_limits[0] <= r <= radius_limits[1] for r in radii):
                        return -np.inf

                    # Gaussian prior criteria: g(x) = 1/(σ*sqrt(2*pi)) * exp(-1/2 * (x-µ)^2/σ^2)
                    # Omitting scaling expressions and note the implicit ln() cancelling the exp()
                    rval = 0
                    for c in range(1, NSTARS):
                        rval += ((teffs[c]/teffs[0] - TeffR_priors[c].n) / TeffR_priors[c].s)**2
                        rval += ((radii[c]/radii[0] - radR_priors[c].n) / radR_priors[c].s)**2
                    rval += ((dist - dist_prior.n) / dist_prior.s)**2
                    return -0.5 * rval


                print("\nSetting up the starting position (theta0) for fitting.")
                init_teff = max(teff_limits[0], min(nom_val(trow.Teff_sys), teff_limits[1]))
                init_logg = max(logg_limits[0], min(nom_val(trow.logg_sys), logg_limits[1]))
                theta0 = create_theta(teffs=[init_teff] * NSTARS,
                                      loggs=[init_logg] * NSTARS,
                                      radii=[1.0] * NSTARS,
                                      dist=coords.distance.to(u.pc).value,
                                      av=0,
                                      nstars=NSTARS,
                                      verbose=True)
                x = model_grid.get_filter_indices(sed["sed_filter"])
                y = (sed["sed_der_flux"].quantity * sed["sed_freq"].quantity)\
                                        .to(model_grid.flux_unit, equivalencies=u.spectral()).value
                y_err = (sed["sed_eflux"].quantity * sed["sed_freq"].quantity)\
                                        .to(model_grid.flux_unit, equivalencies=u.spectral()).value


                # Minimize fits to do outlier filtering and optionally give a starting pos for MCMC
                print("\nPerforming 'quick' minimize fits to mask outliers and set MCMC start.")
                theta_fit = None
                retain_mask = np.ones_like(x, dtype=bool)
                min_to_retain, improve_th = int(np.ceil(len(sed) * 0.75)), 0.8
                print(f"Outliers masked when doing so improves fit stat > {1-improve_th:.0%}")
                cmask, cix, prev_stat = retain_mask.copy(), None, np.inf
                for out_ix in range(len(sed)): # Want this to run at least once so we set theta_fit
                    if prev_stat < 1: # implicitly out_ix > 0
                        print("Stopping outliers search as stat < 1.0")
                        break

                    ctheta, result = minimize_fit(x=x[cmask],
                                                  y=y[cmask],
                                                  y_err=y_err[cmask],
                                                  theta0=theta0,
                                                  fit_mask=fit_mask,
                                                  stellar_grid=model_grid,
                                                  ln_prior_func=ln_prior_func)

                    stat = result.fun
                    print(f"[{out_ix:03d}] stat = {stat:.3e}", end="; ")
                    if out_ix == 0:
                        print("baseline", end="; ")
                        theta_fit = ctheta

                    if (num_retained := sum(cmask)) <= min_to_retain:
                        print(f"stopped as SED rows are at or below the minimum of {min_to_retain}")
                        break

                    if out_ix > 0:
                        print(f"candidate {cix}/{sed['sed_filter'][cix]}", end="; ")
                        if stat < prev_stat * improve_th:
                            print(f"accepted; {sum(cmask)}/{len(cmask)} fluxes remain")
                            theta_fit, retain_mask = ctheta, cmask.copy()
                        else:
                            print("rejected for insufficient improvement and now stopping")
                            break
                    else:
                        print()

                    # The next candidate mask adds the farthest outlier from this fit.
                    y_mdl = model_func(ctheta, x[cmask], model_grid, combine=True)
                    cix = cmask.nonzero()[0][np.argmax(((y_mdl - y[cmask]) / y_err[cmask])**2)]
                    cmask[cix] = False
                    prev_stat = stat

                _print_theta(theta_fit, fit_mask, "Minimize fit yielded theta=")

                if args.plot_figs:
                    print("\nCreating retained SED observations plot")
                    fig = plots.plot_sed(sed["sed_wl"][retain_mask].quantity,
                                    [sed["sed_flux"][retain_mask],sed["sed_der_flux"][retain_mask]],
                                    [sed["sed_eflux"][retain_mask]]*2,
                                    fmts=["or", ".b"], title=f"{target_id} SED",
                                    labels=["observed",f"dereddened\n($A_{{\\rm V}}={Av:.3f})$"])
                    fig.savefig(figs_dir / f"sed-obs-retained.{args.figs_type}", dpi=args.figs_dpi)
                    plt.close(fig)


                if args.do_mcmc_fit:
                    print("\nPerforming a full MCMC from the output from the 'quick' fit.",
                          "Values marked * are free.")
                    theta_fit, sampler = mcmc_fit(x=x[retain_mask],
                                                  y=y[retain_mask],
                                                  y_err=y_err[retain_mask],
                                                  theta0=theta_fit,
                                                  fit_mask=fit_mask,
                                                  stellar_grid=model_grid,
                                                  ln_prior_func=ln_prior_func,
                                                  nwalkers=args.mcmc_walkers,
                                                  nsteps=args.max_mcmc_steps,
                                                  thin_by=args.mcmc_thin_by,
                                                  seed=42,
                                                  early_stopping=True,
                                                  processes=args.mcmc_processes,
                                                  progress=True,
                                                  verbose=True)


                    if args.plot_figs:
                        print("\nCreating MCMC corner and model vs SED observations plots")
                        _data = samples_from_sampler(sampler, thin_by=args.mcmc_thin_by, flat=True)
                        fig = corner.corner(data=_data, show_titles=True, plot_datapoints=True,
                                            quantiles=[0.16, 0.5, 0.84],
                                            labels=theta_labels[fit_mask],
                                            truths=nom_vals(theta_fit[fit_mask]))
                        fig.savefig(figs_dir/f"sed-mcmc-corner.{args.figs_type}", dpi=args.figs_dpi)
                        plt.close(fig)

                        fig = plots.plot_fitted_model_sed(sed[retain_mask], theta_fit, model_grid,
                                                        title=f"{target_id} SED & MCMC model fit")
                        fig.savefig(figs_dir / f"sed-mcmc-fit.{args.figs_type}", dpi=args.figs_dpi)
                        plt.close(fig)

                        _chain = sampler.get_chain(flat=False)
                        _burn_in_samples = _chain.shape[0] - (_data.shape[0] / args.mcmc_walkers)
                        fig, axes = plt.subplots(nrows=sum(fit_mask), figsize=(9, 2*sum(fit_mask)),
                                                 sharex=True)
                        for ix, ax in enumerate(axes.flat):
                            ax.plot(_chain[:, :, ix], "tab:blue", alpha=0.05)
                            ax.axvspan(0, _burn_in_samples, color="silver")
                            ax.set(xlim=(0, len(_chain)), ylabel=theta_labels[fit_mask][ix])
                        axes[-1].set(xlabel=f"step / {args.mcmc_thin_by}")
                        fig.savefig(figs_dir/f"sed-mcmc-trails.{args.figs_type}", dpi=args.figs_dpi)
                        plt.close(fig)
                else:
                    print("\nMCMC disabled. Will use params from minimize fit.")


                print(f"\nFinal parameters for {target_id} ([known value])")
                write_params = {}
                high_uncert_params = []
                for (k, unit), val, mask in zip(theta_params_and_units, theta_fit, fit_mask):
                    label = ""
                    if k == "dist" and trow.parallax:
                        label = f"({1000 / trow.parallax:.3f} pc)"
                    elif config.get("labels", {}).get(k, None) is not None:
                        lval = ufloat(config.labels.get(k, np.NaN), config.labels.get(k+"_err", 0))
                        label = f"({lval:.3f} {unit:unicode})"
                    print(f"{k:>12s}{'*' if mask else ' ':s} = {val:.3f} {unit:unicode} \t", label)

                    # *** also updates the target data ***
                    if mask:
                        write_params[k] = val
                        if std_dev(val) > abs(nom_val(val) * 0.20):
                            high_uncert_params += [k]
                if source := config.get("labels", {}).get("source", None):
                    print(f"Source(s) of known values: {source}")
                if trow.parallax_bibcode:
                    print(f"Source of the known distance (parallax): {trow.parallax_bibcode}")
                if len(high_uncert_params) > 0:
                    trow.append_warning(f"uncert {','.join(k for k in high_uncert_params)}>20%")


                # Finally, store the params and the flag that indicates SED fitting has completed
                print(f"\nWriting fitted params for {list(write_params.keys())} to working-set.")
                trow.set_values(**write_params, fitted_sed=True, errors="")



            except Exception as exc: # pylint: disable=broad-exception-caught
                print("\n*** Failed with the following error. Depending on the nature of the",
                      "error, it may be possible to rerun this module to fit failed targets. ***")
                traceback.print_exception(exc, file=log)
                trow.set_values(fitted_sed=False, errors=type(exc).__name__)

            # Each row's values will be written to the underlying data store as it goes out of scope

        print("\n\n============================================================")
        print(f"Completed {THIS_STEM} at {datetime.now():%Y-%m-%d %H:%M:%S%z %Z}")
        print("============================================================")
