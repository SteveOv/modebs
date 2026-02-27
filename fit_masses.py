""" Pipeline Stage 4 - MCMC fitting of target masses from MIST models """
# pylint: disable=no-member, invalid-name
from inspect import getsourcefile
from pathlib import Path
import warnings
import argparse
from datetime import datetime
from contextlib import redirect_stdout
import traceback

import numpy as np
import matplotlib.pyplot as plt
import astropy.units as u

# pylint: disable=line-too-long, wrong-import-position
warnings.filterwarnings("ignore", "Using UFloat objects with std_dev==0 may give unexpected results.", category=UserWarning)
from uncertainties import ufloat, nominal_value
from uncertainties.unumpy import nominal_values

from deblib.constants import G, R_sun, M_sun

import corner
from sed_fit.fitter import samples_from_sampler

from libs.fit_masses import minimize_fit, mcmc_fit, log_age_for_mass_and_eep
from libs import pipeline
from libs.pipeline import PipelineError
from libs.iohelpers import Tee
from libs.targets import Targets
from libs.pipeline_dal import QTableFileDal


THIS_STEM = Path(getsourcefile(lambda: 0)).stem

NUM_STARS = 2
subs = ["ABCDEFGHIJKLM"[n] for n in range(NUM_STARS)]
theta_labels = np.array([f"$M_{{\\rm {sub}}} / {{\\rm R_{{\\odot}}}}$" for sub in subs]
                      + ["$\\log{{({{\\rm age}})}} / {{\\rm yr}}$"])

theta_params_and_units = np.array([(f"M{sub}", u.Msun) for sub in subs] \
                                + [("log_age", u.dex(u.yr))])


def print_mass_theta(theta, name: str="theta"):
    """ Helper function to print out a mass """
    print(f"{name:s} = [" + ", ".join(f"{t:.3e}" for t in theta) + "]")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Pipeline stage 4: fitting target masses.")
    ap.add_argument("-tf", "--targets-file", dest="targets_file", type=Path, required=False,
                    help="json file containing the details of the targets to fit")
    ap.add_argument("-pf", "--plot-figs", dest="plot_figs", action="store_true", required=False,
                    help="plot figs for each target as the process progresses")
    ap.add_argument("-ms", "--max-steps", dest="max_mcmc_steps", type=int, required=False,
                    help="the maximum number of MCMC steps to run for [100 000]")
    ap.set_defaults(targets_file=Path("./config/plato-lops2-tess-ebs-explicit-targets.json"),
                    plot_figs=False, figs_type="png", figs_dpi=100,
                    max_mcmc_steps=100000, mcmc_walkers=100, mcmc_thin_by=10, mcmc_processes=8)
    args = ap.parse_args()
    drop_dir = Path.cwd() / f"drop/{args.targets_file.stem}"
    args.working_set_file = drop_dir / "working-set.table"


    with redirect_stdout(Tee(open(drop_dir / f"{THIS_STEM}.log", "a", encoding="utf8"))) as log:
        print("\n\n============================================================")
        print(f"Started {THIS_STEM} at {datetime.now():%Y-%m-%d %H:%M:%S%z %Z}")
        print("============================================================")

        targets_config = Targets(args.targets_file)
        print(f"Read in the configuration from '{args.targets_file}'",
              f"which contains {targets_config.count()} target(s) that have not been excluded.")

        wset = QTableFileDal(args.working_set_file)
        to_fit_targets = list(wset.yield_keys("fitted_lcs", "fitted_sed", "fitted_masses",
                                              where=lambda fl, fs, fm: fl and fs and not fm))
        to_fit_count = len(to_fit_targets)
        print(f"The data indicates there are {to_fit_count} target(s) to be fitted.")


        for fit_counter, target_id in enumerate(to_fit_targets, start=1):
            try:
                print("\n\n------------------------------------------------------------")
                print(f"Processing target {fit_counter} of {to_fit_count}: {target_id}")
                print("------------------------------------------------------------")
                config = targets_config.get(target_id)
                warn_msg = wset.read_values(target_id, "warnings") or ""
                if args.plot_figs:
                    figs_dir = drop_dir / "figs" / pipeline.to_file_safe_str(target_id)
                    figs_dir.mkdir(parents=True, exist_ok=True)


                print("Getting known values from previous steps to set up fitting priors")
                TeffA, TeffB, RA, RB = wset.read_values(target_id, "TeffA", "TeffB", "RA", "RB")
                rA_plus_rB, k, period, qphot = wset.read_values(target_id, "rA_plus_rB", "k",
                                                                "period", "qphot")
                rA = rA_plus_rB / (k + 1)
                rB = rA_plus_rB / ((1/k) + 1)
                print("\n".join(f"{p:>20s}: {v:12.3f}" for p, v in [("TeffA", TeffA),
                                                                    ("TeffB", TeffB),
                                                                    ("RA", RA), ("RB", RB),
                                                                    ("rA", rA), ("rB", rB),
                                                                    ("period", period)]))


                # Calculate the system's semi-major axis and system mass (with Kepler's 3rd law)
                a = np.mean([RA / rA, RB / rB])
                print(f" semi-major axis (a): {a:12.3f} {u.Rsun:unicode}",
                      "(calculated from fitted & fractional radii)")
                M_sys = (4 * np.pi**2 * (a * R_sun)**3) / (G * (period * 86400)**2) / M_sun
                print(f" system mass (M_sys): {M_sys:12.3f} {u.Msun:unicode}",
                      "(calculated from semi-major axis & orbital period)")


                # Priors: observations from SED fitting
                prior_radii = np.array([RA, RB])
                prior_Teffs = np.array([TeffA, TeffB])


                # Estimate fit starting position with masses derived from M_sys & the expected mass
                # ratio and an approximate age for the more massive star within the main-sequence.
                print("\nSetting up the starting position (theta0) for fitting.")
                if qphot is None or nominal_value(qphot) <= 0:
                    # The approx single k-q (k=q^0.715) relations of Demircan & Kahraman (1991).
                    qphot = k**1.4
                theta_masses = nominal_values([_MA := M_sys / (qphot + 1), M_sys - _MA])
                theta_age = log_age_for_mass_and_eep(np.max(theta_masses))
                theta0 = np.concatenate([theta_masses, [theta_age]])
                print_mass_theta(theta0, "theta0")


                print("\nPerforming an initial 'quick' minimize fit for approximate values.")
                theta_fit, _ = minimize_fit(theta0=theta0,
                                            sys_mass=M_sys,
                                            radii=prior_radii,
                                            teffs=prior_Teffs,
                                            verbose=True)
                print_mass_theta(theta_fit, "theta_min")


                print("\nPerforming a full MCMC fit for masses and log(age) with uncertainties.")
                theta_mcmc_fit, sampler = mcmc_fit(theta0=theta_fit,
                                                   sys_mass=M_sys,
                                                   radii=prior_radii,
                                                   teffs=prior_Teffs,
                                                   nwalkers=args.mcmc_walkers,
                                                   nsteps=args.max_mcmc_steps,
                                                   thin_by=args.mcmc_thin_by,
                                                   seed=42,
                                                   early_stopping=True,
                                                   processes=args.mcmc_processes,
                                                   progress=True,
                                                   verbose=True)
                print_mass_theta(theta_mcmc_fit, "theta_mcmc")


                print(f"Parameters for {target_id} with nominals & 1-sigma uncertainties",
                      "from MCMC fit ([known value])")
                write_params = { "M_sys": M_sys, "a": a }
                for (k, unit), val in zip(theta_params_and_units, theta_mcmc_fit):
                    label = ""
                    if config.get("labels", {}).get(k, None) is not None:
                        lval = ufloat(config.labels.get(k, np.NaN), config.labels.get(k+"_err", 0))
                        label = f"({lval:.3f} {unit:unicode})"
                    print(f"{k:>12s} = {val:.3f} {unit:unicode} \t", label)

                    # *** also updates the target data ***
                    write_params[k] = val


                # Finally, store the params and the flag that indicates fitting has completed
                print(f"\nWriting fitted params for {list(write_params.keys())} to working-set.")
                wset.write_values(target_id, fitted_masses=True,
                                  errors="", warnings=warn_msg, **write_params)


                if args.plot_figs:
                    print("\nCreating MCMC corner plot")
                    _data = samples_from_sampler(sampler, thin_by=args.mcmc_thin_by, flat=True)
                    fig = corner.corner(data=_data, show_titles=True, plot_datapoints=True,
                                        quantiles=[0.16, 0.5, 0.84], labels=theta_labels,
                                        truths=nominal_values(theta_mcmc_fit))
                    fig.savefig(figs_dir/f"masses-mcmc-corner.{args.figs_type}", dpi=args.figs_dpi)
                    plt.close(fig)


            except Exception as exc: # pylint: disable=broad-exception-caught
                print("\n*** Failed with the following error. Depending on the nature of the",
                      "error, it may be possible to rerun this module to fit failed targets. ***")
                traceback.print_exception(exc, file=log)
                wset.write_values(target_id, fitted_masses=False,
                                  errors=type(exc).__name__, warnings=warn_msg)


        print("\n\n============================================================")
        print(f"Completed {THIS_STEM} at {datetime.now():%Y-%m-%d %H:%M:%S%z %Z}")
        print("============================================================")
