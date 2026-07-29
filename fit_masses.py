#!/usr/bin/env python3
""" Pipeline Stage 4 - MCMC fitting of target masses from MIST models """
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

# pylint: disable=line-too-long, wrong-import-position
warnings.filterwarnings("ignore", "Using UFloat objects with std_dev==0 may give unexpected results.", category=UserWarning)
from uncertainties import ufloat, UFloat, nominal_value as nom_val, std_dev
from uncertainties.unumpy import nominal_values as nom_vals

from deblib.constants import G, R_sun, M_sun
from deblib.vmath import wrap_func_for_uncertainties

import corner
from sed_fit.generic_fitter import minimize_fit, mcmc_fit, samples_from_sampler, print_theta

from libs.mist_models import get_mass_limits, get_eep_limits, log_age_for_mass_and_eep, model_func
from libs.iohelpers import Tee
from libs.targets import Targets
from libs.pipeline_dal import create_dal
from libs.utils import to_file_safe_str

log_age_with_uncertainties = wrap_func_for_uncertainties(log_age_for_mass_and_eep)

THIS_STEM = Path(getsourcefile(lambda: 0)).stem

NUM_STARS = 2
subs = ["ABCDEFGHIJKLM"[n] for n in range(NUM_STARS)]
theta_labels = np.array([f"$M_{{\\rm {sub}}} / {{\\rm M_{{\\odot}}}}$" for sub in subs] \
                      + [f"$EEP_{{\\rm {sub}}}$" for sub in subs])

theta_params_and_units = np.array([(f"M{sub}", u.Msun) for sub in subs] \
                                + [(f"eep{sub}", u.dimensionless_unscaled) for sub in subs])

# Use a non-interactive matplotlib backend to avoid threading errors (issue #36).
mpl_use("agg")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Pipeline stage 4: fitting target masses.")
    ap.add_argument(dest="targets_file", type=Path, metavar="TARGETS_FILE",
                    help="json file containing the details of the targets to fit")
    ap.add_argument("-pf", "--plot-figs", dest="plot_figs", action="store_true", required=False,
                    help="plot figs for each target as the process progresses")
    ap.add_argument("-ms", "--max-steps", dest="max_mcmc_steps", type=int, required=False,
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
        to_fit_criteria = { "fitted_lcs": True, "fitted_sed": True, "fitted_masses": False }
        to_fit_count = dal.count_where(**to_fit_criteria)
        print(f"The working-set indicates there are {to_fit_count} target(s) to be fitted.")


        for fit_counter, trow in enumerate(dal.acquire_next_row(**to_fit_criteria), start=1):

            try:
                target_id = trow.key
                print("\n\n------------------------------------------------------------")
                print(f"Processing target {fit_counter} of {to_fit_count}: {target_id}")
                print("------------------------------------------------------------")
                config = targets_config.get_target_config(target_id)
                if args.plot_figs:
                    figs_dir = drop_dir / "figs" / to_file_safe_str(target_id)
                    figs_dir.mkdir(parents=True, exist_ok=True)

                # Output some known details of the target system
                print()
                print(fill(f"Details:{config.get('details', '')}", subsequent_indent="\t"))
                print(fill(f"Notes:  {config.get('notes', '')}", subsequent_indent="\t"))
                print(f"SpT:\t{trow.spt or config.get('SpT', '')}")
                print(f"morph:\t{trow.morph or -1:.3f}\n")

                print("Getting known values from previous stages to set up fitting priors")
                rA = trow.rA_plus_rB / (trow.k + 1)
                rB = trow.rA_plus_rB / ((1 / trow.k) + 1)
                print("\n".join(f"{p:>20s}: {v:9.3f} {u:unicode}" for p, v, u in [
                                                    ("RA", trow.RA, u.solRad),
                                                    ("RB", trow.RB, u.solRad),
                                                    ("rA", rA, u.dimensionless_unscaled),
                                                    ("rB", rB, u.dimensionless_unscaled),
                                                    ("period", trow.period, u.d)]))

                # Set up the priors and the corresponding function to evaluate them
                # Calculate the system's semi-major axis and system mass (with Kepler's 3rd law)
                a = np.mean([trow.RA / rA, trow.RB / rB])
                print(f" semi-major axis (a): {a:9.3f} {u.Rsun:unicode}",
                      "(calculated from fitted & fractional radii)")
                M_sys = (4 * np.pi**2 * (a * R_sun)**3) / (G * (trow.period * 86400)**2) / M_sun
                print(f" system mass (M_sys): {M_sys:9.3f} {u.Msun:unicode}",
                      "(calculated from semi-major axis & orbital period)")
                eep_limits = get_eep_limits()
                mass_limits = get_mass_limits()
                age_ratio = ufloat(1, 0.02)

                def ln_prior_func(theta: np.ndarray) -> float:
                    """ Evaluate current theta against prior criteria """
                    # pylint: disable=cell-var-from-loop
                    masses, eeps = theta[:NUM_STARS], theta[NUM_STARS:]
                    if not all (eep_limits[0] <= e <= eep_limits[1] for e in eeps) \
                        or not all(mass_limits[0] <= m <= mass_limits[1] for m in masses):
                        return -np.inf

                    # Gaussian priors on the total mass and the closeness of the stars' ages
                    retval = ((np.sum(masses) - M_sys.n) / M_sys.s)**2
                    ages = [log_age_for_mass_and_eep(m, e) for m, e in zip(masses, eeps)]
                    for age_ix in range(1, NUM_STARS):
                        retval += (((ages[age_ix] / ages[0]) - age_ratio.n) / age_ratio.s)**2
                    return -0.5 * retval


                # Estimate fit starting position with masses derived from M_sys & the expected mass
                # ratio and an approximate mid main-sequence EEP for the more massive star.
                print("\nSetting up the starting position/theta0 for fitting",
                      "[" + ", ".join(theta_params_and_units[..., 0]) + "]")
                if (qphot := trow.qphot) is None or nom_val(qphot) <= 0:
                    qphot = 1
                theta_masses = np.array([M_sys / (1+qphot)] + [M_sys / (1 + 1/qphot)]*(NUM_STARS-1))
                theta_masses = nom_vals(theta_masses * (M_sys / sum(theta_masses)))
                theta0 = np.append(theta_masses, [353] * NUM_STARS) # 353 equiv IAMS
                print_theta(theta0, prefix="theta0 = ")

                # Set up the likelihood function to evaluate the result of each theta
                # against known observations from SED fitting
                print("\nGetting known values from previous stages to set up observed values")
                y_obs = np.empty((6, ), dtype=np.dtype(UFloat.dtype))
                for ix, col in enumerate(["RA", "RB", "TeffA", "TeffB", "loggA", "loggB"]):
                    val = trow[col]
                    if not isinstance(val, UFloat) or not val.s:
                        val = ufloat(nom_val(val), 0.02 * nom_val(val))
                    y_obs[ix] = val
                    print(f"{col:>20s}: {val:9.3f}")

                wt = -0.5 / (len(y_obs) - len(theta0)) # likelihood = -0.5 * sum(resids) / deg_free
                def ln_likelihood_func(y_model: np.ndarray) -> float:
                    """ Evaluate current model against observations to give reduced chi^2 """
                    # pylint: disable=cell-var-from-loop
                    return wt * np.sum([((m - o.n) / o.s)**2 for m, o in zip(y_model, y_obs)])


                def ln_prob_func(theta: np.ndarray[float]) -> float:
                    """
                    The function which returns the log posterior probability; the probability that the candidate params
                    (theta) are those responsible for the observations. This is a negative value tending towards zero
                    as the probability increases. Think of this as:

                    ln(P(posterior)) = ln(P(prior) * P(likelihood)) = ln_prior_func() + ln_likelihood_func()
                    """
                    # pylint: disable=cell-var-from-loop
                    retval = ln_prior_func(theta)
                    if np.isfinite(retval):
                        # The "model func": gets the stars' radii, teffs & loggs from MIST models
                        model_y = model_func(masses=theta[:NUM_STARS], eeps=theta[NUM_STARS:])
                        if any(model_y <= 0):
                            return -np.inf

                    if np.isfinite(retval):
                        retval += ln_likelihood_func(model_y)
                    return retval


                print("\nPerforming an initial 'quick' minimize fit for approximate values.")
                theta_fit, _ = minimize_fit(ln_prob_func=ln_prob_func,
                                            theta0=theta0,
                                            verbose=True)


                if args.do_mcmc_fit:
                    print("\nPerforming a full MCMC for masses & eeps with uncertainties.")
                    theta_fit, sampler = mcmc_fit(ln_prob_func=ln_prob_func,
                                                  ln_prior_func=ln_prior_func,
                                                  theta0=theta0,
                                                  nwalkers=args.mcmc_walkers,
                                                  nsteps=args.max_mcmc_steps,
                                                  thin_by=args.mcmc_thin_by,
                                                  seed=42,
                                                  early_stopping=True,
                                                  early_stopping_from=10000,
                                                  processes=args.mcmc_processes,
                                                  progress=True,
                                                  verbose=True)


                    if args.plot_figs:
                        print("\nCreating MCMC corner and trails plots")
                        _data = samples_from_sampler(sampler, thin_by=args.mcmc_thin_by, flat=True)
                        fig = corner.corner(data=_data, show_titles=True, plot_datapoints=True,
                                            quantiles=[0.16, 0.5, 0.84], labels=theta_labels,
                                            truths=nom_vals(theta_fit))
                        fig.savefig(figs_dir / f"masses-mcmc-corner.{args.figs_type}",
                                    dpi=args.figs_dpi)
                        plt.close(fig)

                        _chain = sampler.get_chain(flat=False)
                        _burn_in_samples = _chain.shape[0] - (_data.shape[0] / args.mcmc_walkers)
                        fig, axes = plt.subplots(nrows=theta0.size, figsize=(8, 1.5*theta0.size),
                                                 sharex=True, constrained_layout=True)
                        for ix, ax in enumerate(axes.flat):
                            ax.plot(_chain[:, :, ix], "tab:blue", alpha=0.05)
                            ax.axvspan(0, _burn_in_samples, color="silver")
                            ax.set(xlim=(0, len(_chain)), ylabel=theta_labels[ix])
                        axes[-1].set(xlabel=f"step / {args.mcmc_thin_by}")
                        fig.savefig(figs_dir / f"masses-mcmc-trails.{args.figs_type}",
                                    dpi=args.figs_dpi)
                        plt.close(fig)


                print("\nCalculating the stars' log(age) from masses and eeps")
                log_ages = [log_age_with_uncertainties(m, e)
                                    for m, e in zip(theta_fit[:NUM_STARS], theta_fit[NUM_STARS:])]
                log_age = np.mean(log_ages)


                print(f"\nFinal fitted parameters for {target_id} ([known value])")
                high_uncert_params = []
                write_params = { "M_sys": M_sys, "a": a }
                for (k, unit), val in zip(
                        np.concatenate([theta_params_and_units, [("log_age", u.dex(u.yr))]]),
                        np.concatenate([theta_fit, [log_age]])):
                    label = ""
                    if config.get("labels", {}).get(k, None) is not None:
                        lval = ufloat(config.labels.get(k, np.NaN), config.labels.get(k+"_err", 0))
                        label = f"({lval:.3f} {unit:unicode})"
                    print(f"{k:>12s} = {val:.3f} {unit:unicode} \t", label)

                    # *** also updates the target data ***
                    if not k.startswith("eep"):
                        write_params[k] = val
                        if std_dev(val) > abs(nom_val(val) * 0.20):
                            high_uncert_params += [k]
                if source := config.get("labels", {}).get("source", None):
                    print(f"Source(s) of known values: {source}")
                if len(high_uncert_params) > 0:
                    trow.append_warning(f"uncert {','.join(high_uncert_params)}>20%")


                # Finally, store the params and the flag that indicates fitting has completed
                print(f"\nWriting fitted params for {list(write_params.keys())} to working-set.")
                trow.set_values(**write_params, fitted_masses=True, errors="")


            except Exception as exc: # pylint: disable=broad-exception-caught
                print("\n*** Failed with the following error. Depending on the nature of the",
                      "error, it may be possible to rerun this module to fit failed targets. ***")
                traceback.print_exception(exc, file=log)
                trow.set_values(**write_params, fitted_masses=False, errors=type(exc).__name__)

            # Each row's values will be written to the underlying data store as it goes out of scope

        print("\n\n============================================================")
        print(f"Completed {THIS_STEM} at {datetime.now():%Y-%m-%d %H:%M:%S%z %Z}")
        print("============================================================")
