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
from time import sleep
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

import corner
from sed_fit.fitter import samples_from_sampler

from libs.mass_fitter import minimize_fit, mcmc_fit
from libs.mass_fitter import get_age_limits, get_mass_limits, log_age_for_mass_and_eep
from libs.iohelpers import Tee
from libs.targets import Targets
from libs.pipeline_dal import create_dal
from libs.utils import to_file_safe_str


THIS_STEM = Path(getsourcefile(lambda: 0)).stem

NUM_STARS = 2
subs = ["ABCDEFGHIJKLM"[n] for n in range(NUM_STARS)]
theta_labels = np.array([f"$M_{{\\rm {sub}}} / {{\\rm M_{{\\odot}}}}$" for sub in subs]
                      + ["$\\log{{({{\\rm age}})}} / {{\\rm yr}}$"])

theta_params_and_units = np.array([(f"M{sub}", u.Msun) for sub in subs] \
                                + [("log_age", u.dex(u.yr))])

# Use a non-interactive matplotlib backend to avoid threading errors (issue #36).
mpl_use("agg")


def print_mass_theta(theta, name: str="theta"):
    """ Helper function to print out a mass """
    print(f"{name:s} = [" + ", ".join(f"{t:.3e}" for t in theta) + "]")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Pipeline stage 4: fitting target masses.")
    ap.add_argument(dest="targets_file", type=Path, metavar="TARGETS_FILE",
                    help="json file containing the details of the targets to fit")
    ap.add_argument("-pf", "--plot-figs", dest="plot_figs", action="store_true", required=False,
                    help="plot figs for each target as the process progresses")
    ap.add_argument("-ms", "--max-steps", dest="max_mcmc_steps", type=int, required=False,
                    help="the maximum number of MCMC steps to run for [100 000]")
    ap.set_defaults(plot_figs=False, figs_type="png", figs_dpi=100, do_mcmc_fit=True,
                    max_mcmc_steps=100000, mcmc_walkers=100, mcmc_thin_by=10, mcmc_processes=5)
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
                age_limits = get_age_limits()
                mass_limits = get_mass_limits()

                def ln_prior_func(theta: np.ndarray) -> float:
                    """ Evaluate current theta against prior criteria """
                    # pylint: disable=cell-var-from-loop
                    masses, age = theta[:-1], 10**theta[-1]
                    if not age_limits[0] <= age <= age_limits[1] \
                        or not all(mass_limits[0] <= mass <= mass_limits[1] for mass in masses):
                        return -np.inf
                    # Gaussian prior on the total mass
                    return -0.5 * ((M_sys.n - np.sum(masses)) / M_sys.s)**2


                # Estimate fit starting position with masses derived from M_sys & the expected mass
                # ratio and an approximate age for the more massive star within the main-sequence.
                print("\nSetting up the starting position (theta0) for fitting [MA, MB, log(age)].")
                if (qphot := trow.qphot) is None or nom_val(qphot) <= 0:
                    # The approx single k-q (k=q^0.715) relations of Demircan & Kahraman (1991).
                    qphot = trow.k**1.4
                theta_masses = nom_vals([_MA := M_sys / (qphot + 1), M_sys - _MA])
                theta_age = log_age_for_mass_and_eep(np.max(theta_masses))
                theta0 = np.concatenate([theta_masses, [theta_age]])
                print_mass_theta(theta0, "theta0")

                # Set up the likelihood function to evaluate the result of each theta
                # against known observations from SED fitting
                print("\nGetting known values from previous stages to set up observed values")
                y_obs = np.empty((6,), dtype=object)
                for ix, col in enumerate(["RA", "RB", "TeffA", "TeffB", "loggA", "loggB"]):
                    val = trow[col]
                    if not isinstance(val, UFloat) or not val.s:
                        val = ufloat(nom_val(val), 0.05 * nom_val(val))
                    y_obs[ix] = val
                    print(f"{col:>20s}: {val:9.3f}")
                wt = -0.5 / (len(y_obs) - len(theta0)) # likelihood = -0.5 * sum(resids) / deg_free
                def ln_likelihood_func(y_model: np.ndarray) -> float:
                    """ Evaluate current model against observations to give reduced chi^2 """
                    # pylint: disable=cell-var-from-loop
                    return wt * np.sum([((m - o.n) / o.s)**2 for m, o in zip(y_model, y_obs)])


                print("\nPerforming an initial 'quick' minimize fit for approximate values.")
                theta_fit, _ = minimize_fit(theta0=theta0,
                                            ln_prior_func=ln_prior_func,
                                            ln_likelihood_func=ln_likelihood_func,
                                            verbose=True)
                print_mass_theta(theta_fit, "theta_min")


                if args.do_mcmc_fit:
                    print("\nPerforming a full MCMC fit for masses & log(age) with uncertainties.")
                    theta_fit, sampler = mcmc_fit(theta0=theta0,
                                                  ln_prior_func=ln_prior_func,
                                                  ln_likelihood_func=ln_likelihood_func,
                                                  nwalkers=args.mcmc_walkers,
                                                  nsteps=args.max_mcmc_steps,
                                                  thin_by=args.mcmc_thin_by,
                                                  seed=42,
                                                  early_stopping=True,
                                                  early_stopping_from=25000,
                                                  processes=args.mcmc_processes,
                                                  progress=True,
                                                  verbose=True)
                    print_mass_theta(theta_fit, "theta_mcmc")


                    if args.plot_figs:
                        print("\nCreating MCMC corner plot")
                        _data = samples_from_sampler(sampler, thin_by=args.mcmc_thin_by, flat=True)
                        fig = corner.corner(data=_data, show_titles=True, plot_datapoints=True,
                                            quantiles=[0.16, 0.5, 0.84], labels=theta_labels,
                                            truths=nom_vals(theta_fit))
                        fig.savefig(figs_dir / f"masses-mcmc-corner.{args.figs_type}",
                                    dpi=args.figs_dpi)
                        plt.close(fig)


                print(f"\nFinal fitted parameters for {target_id} ([known value])")
                high_uncert_params = []
                write_params = { "M_sys": M_sys, "a": a }
                for (k, unit), val in zip(theta_params_and_units, theta_fit):
                    label = ""
                    if config.get("labels", {}).get(k, None) is not None:
                        lval = ufloat(config.labels.get(k, np.NaN), config.labels.get(k+"_err", 0))
                        label = f"({lval:.3f} {unit:unicode})"
                    print(f"{k:>12s} = {val:.3f} {unit:unicode} \t", label)

                    # *** also updates the target data ***
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
