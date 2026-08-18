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
from deblib.constants import G, R_sun, M_sun
from sed_fit.stellar_grids import get_stellar_grid
from sed_fit.fitter import create_theta, minimize_fit, mcmc_fit, model_func
from sed_fit.generic_fitter import samples_from_sampler, print_theta

from libs import extinction, plots
from libs.pipeline import PipelineError
from libs.sed import get_sed_for_target, retain_closest_observations
from libs.iohelpers import Tee
from libs.targets import Targets
from libs.pipeline_dal import create_dal
from libs.utils import to_file_safe_str, format_value

THIS_STEM = Path(getsourcefile(lambda: 0)).stem

NSTARS = 2
subs = ["ABCDEFGHIJKLM"[n] for n in range(NSTARS)]
theta_labels = np.array([f"$T_{{\\rm eff,{sub}}} / {{\\rm K}}$" for sub in subs]
                      + [f"$\\log{{g}}_{{\\rm {sub}}}$" for sub in subs]
                      + [f"$R_{{\\rm {sub}}} / {{\\rm R_{{\\odot}}}}$" for sub in subs]
                      + ["$D / {\\rm pc}$", "${\\rm A_{V}}$"])

theta_params_and_units = np.array([(f"Teff{sub}", u.K) for sub in subs]
                                + [(f"logg{sub}", u.dex) for sub in subs]
                                + [(f"R{sub}", u.Rsun) for sub in subs]
                                + [("dist", u.pc), ("Av", None)])

# Dictates which params in theta are fitted (True) and which are held fixed (False)
fit_av = True
use_quick_mode = True   # If True, grid uses pre-filtered fluxes
fit_mask = np.array([True] * NSTARS      # Teff
                  + [False] * NSTARS     # logg
                  + [True] * NSTARS      # radius
                  + [True]               # dist
                  + [fit_av])            # Av

# Use a non-interactive matplotlib backend to avoid threading errors (issue #36).
mpl_use("agg")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Pipeline stage 3: fitting target SED.")
    ap.add_argument(dest="targets_file", type=Path, metavar="TARGETS_FILE",
                    help="json file containing the details of the targets to fit")
    ap.add_argument("-t", "--target", dest="target", type=str, required=False,
                    help="a single target id from the targets file to be fitted or re-fitted")
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
    ap.add_argument("-uel", "--use-extinction-labels", dest="use_extinction_labels",
                    action="store_true", required=False,
                    help="force override of Av with known Av or E(B-V) values if present in config")
    ap.set_defaults(target=None, plot_figs=False, figs_type="png", figs_dpi=100, do_mcmc_fit=True,
                    max_mcmc_steps=100000, mcmc_walkers=100, mcmc_thin_by=10, mcmc_processes=8,
                    use_extinction_labels=False)
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
        if args.target:
            to_fit_criteria = { "fitted_lcs": True, "target_id": args.target }
        else:
            to_fit_criteria = { "fitted_lcs": True, "fitted_sed": False }
        to_fit_count = dal.count_where(**to_fit_criteria)
        print(f"The working-set and switches indicates {to_fit_count} target(s) to be fitted.")

        # Extinction model: G23 (Gordon et al., 2023) Milky Way R(V) filter gives us broad coverage
        Rv = targets_config.get("Rv", 3.1)
        ext_model = G23(Rv=Rv)
        ext_wl_range = np.reciprocal(ext_model.x_range) * u.um # x_range has implicit units of 1/um
        print(f"\nUsing the {ext_model.__class__.__name__} extinction model with Rv={Rv}, which",
            f"covers the range from {min(ext_wl_range):unicode} to {max(ext_wl_range):unicode}.\n")

        # Model SED grid based on atmosphere models with known filters pre-applied to non-reddened
        # fluxes. Available grids: BtSettlGrid, KuruczGrid or BlackBodyGrid
        model_grid = get_stellar_grid(targets_config.get("stellar_grid", "BtSettlGrid"),
                                      extinction_model=ext_model,
                                      use_quick_mode=use_quick_mode, verbose=True)
        print(f"Loaded the {model_grid.__class__.__name__} which covers the ranges:")
        print(f"wavelength {model_grid.wavelength_range * model_grid.wavelength_unit:unicode},",
              f"Teff {model_grid.teff_range * model_grid.teff_unit:unicode},",
              f"logg {model_grid.logg_range * model_grid.logg_unit:unicode}",
              f"\nand metallicity {model_grid.metal_range * u.dimensionless_unscaled:unicode},",
              f"with fluxes returned in units of {model_grid.flux_unit:unicode}", flush=True)

        # Fixed priors limits for fitting
        teff_limits = model_grid.teff_range
        logg_limits = model_grid.logg_range
        radius_limits = (0.05, 100)

        for fit_counter, trow in enumerate(dal.acquire_next_row(**to_fit_criteria), start=1):

            try:
                target_id = trow.key
                print("\n\n------------------------------------------------------------")
                print(f"Processing target {fit_counter} of {to_fit_count}: {target_id}")
                print("------------------------------------------------------------", flush=True)
                config = targets_config.get_target_config(target_id)
                known_vals = config.get("labels", {})
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


                # Get the extinction, based on the coords, prioritising the mean of reliable results
                print(f"\nGetting extinction based on {target_id}",
                      f"{coords}".replace("\n",""), flush=True)
                Av, is_reliable_av = 0, False
                avs = np.array([*extinction.iterate(coords, rv=Rv, verbose=True)]).T
                if any(reliable_mask := np.array(avs[1], dtype=bool)):
                    Av = ufloat(np.mean(avs[0][reliable_mask]), np.std(avs[0][reliable_mask]))
                    is_reliable_av = True
                    print(f"Found the mean of {sum(reliable_mask)} reliable value(s): Av={Av:.6f}")
                elif avs.shape[1]:
                    Av = ufloat(np.mean(avs[0]), np.std(avs[0]))
                    print(f"Found the mean of {avs.shape[1]} value(s): Av={Av:.6f}")
                # Check for overrides
                for overrides, use_it, msg in [
                    (config, True, "Found an extinction override in target's config"),
                    (known_vals, args.use_extinction_labels, "*** The --use-extinction-labels " \
                                + "switch is set and found an extinction value in target's labels")
                ]:
                    if use_it and ("Av" in overrides or "E(B-V)" in overrides):
                        Av = ufloat(overrides.get("Av", overrides.get("E(B-V)", np.nan)*Rv),
                                    overrides.get("AV_err", overrides.get("E(B-V)_err", 0)*Rv))
                        is_reliable_av = True
                        print(msg, f"giving Av={Av:.6f}")
                if not is_reliable_av:
                    trow.append_warning("unreliable Av")


                # Get the SED for this target and de-duplicate (obs may appear multiple times).
                print(flush=True)
                sed = get_sed_for_target(target_id, trow.search_term,
                                         radius=0.25, remove_duplicates=True, verbose=True)
                if sed is None or len(sed) == 0:
                    raise PipelineError(target_id, f"No SED observations for '{trow.search_term}'")

                # Reduce the SED to 1 obs per Filter, the one closest to the target coords. Then
                # filter it to those Filters covered by our models and remove any known exclusions.
                sed = retain_closest_observations(sed, coords)
                model_mask = np.ones((len(sed)), dtype=bool)
                model_mask &= model_grid.has_filter(sed["sed_filter"])
                model_mask &= ~np.isin(sed["sed_filter"], config.get("sed_filter_exclusions", []))
                model_mask &= ~np.isin(sed["_tabname"], config.get("sed_tabname_exclusions", []))
                model_mask &= (sed["sed_wl"] >= min(ext_wl_range)) \
                            & (sed["sed_wl"] <= max(ext_wl_range)) \
                            & (sed["sed_wl"] >= min(model_grid.wavelength_range)) \
                            & (sed["sed_wl"] <= max(model_grid.wavelength_range))
                sed = sed[model_mask]
                sed.sort(["sed_wl"])
                print(f"{len(sed)} unique SED observation(s) retained after range & exclusion",
                      "filtering, and\nthe retention of only the 'closest' observation to the",
                      "target for each Filter.\nThe units for flux, frequency and wavelength are:",
                      ", ".join(f"{sed[f].unit:unicode}" for f in ["sed_flux","sed_freq","sed_wl"]))            


                if fit_av:
                    print("Will not de-redden SED observations as Av will be fitted")
                    sed["sed_fit_flux"] = sed["sed_flux"]
                else:
                    print("Creating de-reddened SED observations")
                    sed["sed_fit_flux"] = sed["sed_flux"] \
                                    / ext_model.extinguish(sed["sed_wl"].to(u.um), Av=nom_val(Av))

                if args.plot_figs:
                    print("\nCreating SED observations plot", flush=True)
                    if fit_av:
                        fig = plots.plot_sed(sed["sed_wl"].quantity, [sed["sed_fit_flux"]],
                                             [sed["sed_eflux"]], fmts=["or"], labels=["observed"],
                                             title=f"{target_id} SED observations")
                    else:
                        fig = plots.plot_sed(sed["sed_wl"].quantity,
                                             [sed["sed_flux"],sed["sed_fit_flux"]],
                                             [sed["sed_eflux"]]*2,
                                             fmts=["or", ".b"], labels=["observed", "dereddened"],
                                             title=f"{target_id} SED observations")
                    fig.savefig(figs_dir / f"sed-observations.{args.figs_type}", dpi=args.figs_dpi)
                    plt.close(fig)


                # Set up the MCMC fitting theta and priors. For now, hard coded to 2 stars.
                # The ratios are wrt the primary components - the prior_func ignores the 0th item
                print("\nSetting up the fitting priors & the ln_prior_func() callback.", flush=True)
                TeffR, radR = trow.TeffR, trow.k
                TeffR_priors =tuple([1] + [ufloat(TeffR.n, TeffR.s or (TeffR.n * 0.05))]*(NSTARS-1))
                radR_priors = tuple([1] + [ufloat(radR.n, radR.s or (radR.n * 0.05))]*(NSTARS-1))
                av_prior = ufloat(nom_val(Av), std_dev(Av) or (nom_val(Av) * 0.05))
                d_prior = 1000 / trow.parallax
                d_prior = ufloat(nom_val(d_prior), std_dev(d_prior) or (nom_val(d_prior) * 0.05))
                print(f"Priors: TeffR=({', '.join(f'{r:.3f}' for r in TeffR_priors)}),",
                      f"radR=({', '.join(f'{r:.3f}' for r in radR_priors)}),",
                      f"dist={d_prior:.3f}, Av={av_prior:.3f},",
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
                    av = theta[-1]

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
                    rval += ((dist - d_prior.n) / d_prior.s)**2
                    if fit_av:
                        rval += ((av - av_prior.n) / av_prior.s)**2
                    return -0.5 * rval


                print("\nSetting up the starting position (theta0) for fitting.", flush=True)
	            # Ensure initial Teffs, loggs & radii are within prior limits and then apply ratios
                t0_Teffs = [max(teff_limits[0], min(nom_val(trow.Teff_sys), teff_limits[1]))]*NSTARS
                t0_loggs = [max(logg_limits[0], min(nom_val(trow.logg_sys), logg_limits[1]))]*NSTARS
                t0_radii = [max(radius_limits[0], min(t0_Teffs[0] / 5500, radius_limits[1]))]*NSTARS
                for t0, ratio in [(t0_Teffs, nom_val(TeffR)), (t0_radii, nom_val(radR))]:
                    if ratio < 1:
                        t0[1:] = [t * ratio for t in t0[1:]]
                    else:
                        t0[0] /= ratio
                theta0 = create_theta(teffs=t0_Teffs,
                                      loggs=t0_loggs,
                                      radii=t0_radii,
                                      dist=coords.distance.to(u.pc).value,
                                      av=nom_val(Av) if fit_av else 0,
                                      nstars=NSTARS,
                                      verbose=True)


                print("\nPreparing SED data for fitting.",
                      f"Fluxes will be fitted in implied units of {model_grid.flux_unit:unicode}")
                with u.set_enabled_equivalencies(u.spectral_density(sed["sed_freq"].quantity) \
                                                 + u.spectral()):
                    x = model_grid.get_filter_indices(sed["sed_filter"])
                    y = sed["sed_fit_flux"].quantity.to(model_grid.flux_unit).value
                    y_err = sed["sed_eflux"].quantity.to(model_grid.flux_unit).value


                # Minimize fits to do outlier pruning and optionally give a starting pos for MCMC
                print("\nPerforming 'quick' minimize fits to prune outliers.", flush=True)
                theta_fit = None
                retain_mask = np.ones_like(x, dtype=bool)
                min_to_retain, improve_th = max(10, int(np.ceil(len(sed) * 0.75))), 0.8
                print(f"Outliers pruned when doing so improves fit stat > {1-improve_th:.0%}")
                cand_mask, cand_ix, prev_stat = retain_mask.copy(), None, np.inf
                for out_ix in range(len(sed)):
                    cand_theta, result = minimize_fit(x=x[cand_mask],
                                                      y=y[cand_mask],
                                                      y_err=y_err[cand_mask],
                                                      theta0=theta0,
                                                      fit_mask=fit_mask,
                                                      stellar_grid=model_grid,
                                                      ln_prior_func=ln_prior_func)

                    this_stat = result.fun
                    print(f"[{out_ix:03d}] stat={this_stat:.3e}", end="; ")
                    if out_ix == 0:
                        theta_fit = cand_theta
                    else:
                        # Decide outcome on overall stat from test fit with candidate excluded
                        if this_stat >= prev_stat * improve_th:
                            print("rejected for insufficient improvement and now stopping.")
                            break
                        theta_fit, retain_mask, prev_stat = cand_theta, cand_mask.copy(), this_stat
                        print(f"accepted leaving {sum(retain_mask)}/{len(retain_mask)}", end="; ")

                    if sum(retain_mask) <= min_to_retain:
                        print(f"stopping as #rows at or below the minimum of {min_to_retain}.")
                        break

                    if this_stat < 1:
                        print("stopping as stat is < 1.0.")
                        break

                    # Now select the first/next candidate, which is farthest outlier from this fit.
                    y_mdl = model_func(cand_theta, x[cand_mask], model_grid, combine=True)
                    resids = ((y_mdl - y[cand_mask]) / y_err[cand_mask])**2
                    max_resid_ix = np.argmax(resids)
                    cand_ix = cand_mask.nonzero()[0][max_resid_ix]
                    print(f"max outlier [{cand_ix}] {sed['sed_filter'][cand_ix]}", end="; ")
                    if resids[max_resid_ix] < min(100, sum(resids) / 2):
                        print("not significant and now stopping.")
                        break
                    print()
                    cand_mask[cand_ix] = False

                print_theta(theta_fit, fit_mask, "Minimize fit yielded theta=")

                if args.plot_figs:
                    print("\nCreating retained SED observations and minimize fit plot")
                    fig = plots.plot_fitted_model_sed(sed[retain_mask], theta_fit, model_grid,
                                                      sed_flux_colname="sed_fit_flux",
                                                      title=f"{target_id} SED & minimize model fit")
                    fig.savefig(figs_dir / f"sed-retained-fit.{args.figs_type}", dpi=args.figs_dpi)
                    plt.close(fig)


                if args.do_mcmc_fit:
                    print("\nPerforming a full MCMC. Values marked * are free.", flush=True)
                    theta_fit, sampler = mcmc_fit(x=x[retain_mask],
                                                  y=y[retain_mask],
                                                  y_err=y_err[retain_mask],
                                                  theta0=theta0,
                                                  fit_mask=fit_mask,
                                                  stellar_grid=model_grid,
                                                  ln_prior_func=ln_prior_func,
                                                  nwalkers=args.mcmc_walkers,
                                                  nsteps=args.max_mcmc_steps,
                                                  thin_by=args.mcmc_thin_by,
                                                  seed=42,
                                                  processes=args.mcmc_processes,
                                                  early_stopping=True,
                                                  early_stopping_from=10000,
                                                  progress=True,
                                                  verbose=True)


                    if args.plot_figs:
                        print("\nCreating MCMC corner & model v SED observations plots", flush=True)
                        _data = samples_from_sampler(sampler, thin_by=args.mcmc_thin_by, flat=True)
                        fig = corner.corner(data=_data, show_titles=True, plot_datapoints=True,
                                            quantiles=[0.16, 0.5, 0.84],
                                            labels=theta_labels[fit_mask],
                                            truths=nom_vals(theta_fit[fit_mask]))
                        fig.savefig(figs_dir/f"sed-mcmc-corner.{args.figs_type}", dpi=args.figs_dpi)
                        plt.close(fig)

                        fig = plots.plot_fitted_model_sed(sed[retain_mask], theta_fit, model_grid,
                                                          sed_flux_colname="sed_fit_flux",
                                                          title=f"{target_id} SED & MCMC model fit")
                        fig.savefig(figs_dir / f"sed-mcmc-fit.{args.figs_type}", dpi=args.figs_dpi)
                        plt.close(fig)

                        _chain = sampler.get_chain(flat=False)
                        _burn_in_samples = _chain.shape[0] - (_data.shape[0] / args.mcmc_walkers)
                        fig, axes = plt.subplots(nrows=sum(fit_mask), figsize=(8, 1.5*sum(fit_mask)),
                                                 sharex=True, constrained_layout=True)
                        for ix, ax in enumerate(axes.flat):
                            ax.plot(_chain[:, :, ix], "tab:blue", alpha=0.05)
                            ax.axvspan(0, _burn_in_samples, color="silver")
                            ax.set(xlim=(0, len(_chain)), ylabel=theta_labels[fit_mask][ix])
                        axes[-1].set(xlabel="step" + \
                                        (f" / {args.mcmc_thin_by}" if args.mcmc_thin_by>1 else ""))
                        fig.savefig(figs_dir/f"sed-mcmc-trails.{args.figs_type}", dpi=args.figs_dpi)
                        plt.close(fig)
                else:
                    print("\nMCMC disabled. Will use params from minimize fit.")


                print(f"\nFinal parameters for {target_id} ([known value])", flush=True)
                write_params = {}
                for (k, unit), val, mask in zip(theta_params_and_units, theta_fit, fit_mask):
                    kval = None
                    if k == "dist" and trow.parallax:
                        kval = 1000 / trow.parallax
                    elif k in known_vals:
                        kval = ufloat(known_vals[k], known_vals.get(f"{k}_err", 0))
                    elif k == "Av" and "E(B-V)" in known_vals:
                        kval = ufloat(known_vals["E(B-V)"], known_vals.get("E(B-V)_err", 0)) * Rv
                    print(f"{k:>12s}{'*' if mask else ' ':s} =", format_value(val, unit, kval))
                    write_params[k] = val

                print("Calculated system mass from fractional radii, radii and period")
                rA = trow.rA_plus_rB / (trow.k + 1)
                rB = trow.rA_plus_rB / ((1 / trow.k) + 1)
                a = np.mean([write_params["RA"] / rA, write_params["RB"] / rB])
                M_sys = (4 * np.pi**2 * (a * R_sun)**3) / (G * (trow.period * 86400)**2) / M_sun
                kval = None
                if "a" in known_vals:
                    kval = ufloat(known_vals["a"], known_vals.get("a_err", 0))
                print(f"{'a':>12s}  =", format_value(a, u.R_sun, kval))
                kval = None
                if "MA" in known_vals and "MB" in known_vals:
                    kval = ufloat(known_vals["MA"], known_vals.get("MA_err", 0)) \
                            + ufloat(known_vals["MB"], known_vals.get("MB_err", 0))
                print(f"{'M_sys':>12s}  =", format_value(M_sys, u.M_sun, kval))
                write_params |= { "a": a, "M_sys": M_sys }

                if "source" in known_vals:
                    print("Source(s) of known values:", known_vals["source"])
                if trow.parallax_bibcode:
                    print("Source of the known distance (parallax):", trow.parallax_bibcode)

                high_uncert_params = [k for k, val in write_params.items()
                                if std_dev(val) > abs(nom_val(val) * 0.20) and k not in ["Av", "a"]]
                if len(high_uncert_params) > 0:
                    trow.append_warning(f"uncert {','.join(k for k in high_uncert_params)}>20%")


                # Finally, store the params and the flag that indicates SED fitting has completed.
                # Subsequent stages set to incomplete as this fit will have invalidated them.
                print(f"\nWriting fitted params for {list(write_params.keys())} to working-set.",
                      flush=True)
                trow.set_values(**write_params, fitted_sed=True, fitted_masses=False, errors="")



            except Exception as exc: # pylint: disable=broad-exception-caught
                print("\n*** Failed with the following error. Depending on the nature of the",
                      "error, it may be possible to rerun this module to fit failed targets. ***")
                traceback.print_exception(exc, file=log)
                trow.set_values(fitted_sed=False, errors=type(exc).__name__)

            # Each row's values will be written to the underlying data store as it goes out of scope
            log.flush()

        print("\n\n============================================================")
        print(f"Completed {THIS_STEM} at {datetime.now():%Y-%m-%d %H:%M:%S%z %Z}")
        print("============================================================")
