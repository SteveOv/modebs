""" Pipeline Stage 4 - MCMC fitting of target masses from MIST models """
# pylint: disable=no-member, invalid-name
from inspect import getsourcefile
from pathlib import Path
import warnings
import argparse
from datetime import datetime
from contextlib import redirect_stdout

import numpy as np
import astropy.units as u

# pylint: disable=line-too-long, wrong-import-position
warnings.filterwarnings("ignore", "Using UFloat objects with std_dev==0 may give unexpected results.", category=UserWarning)
from uncertainties import ufloat
from uncertainties.unumpy import nominal_values

from deblib.constants import G, R_sun, M_sun

from libs.iohelpers import Tee
from libs.targets import Targets
from libs.pipeline_dal import QTableFileDal
from libs.fit_masses import minimize_fit, mcmc_fit


THIS_STEM = Path(getsourcefile(lambda: 0)).stem
NUM_STARS = 2
fit_labels_and_units = { "MA": u.Msun, "MB": u.Msun, "log(age)": u.dex(u.yr) }


def print_mass_theta(theta, label: str="theta"):
    """ Helper function to print out a mass """
    print(f"{label:s} = [" + ", ".join(f"{t:.3e}" for t in theta) + "]")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Pipeline stage 4: fitting target masses.")
    ap.add_argument("-tf", "--targets-file", dest="targets_file", type=Path, required=False,
                    help="json file containing the details of the targets to fit")
    ap.add_argument("-wd", "--write-diags", dest="write_diags", action="store_true", required=False,
                    help="write a second, human readable output file for diagnostics")
    ap.add_argument("-ms", "--max-steps", dest="max_mcmc_steps", type=int, required=False,
                    help="the maximum number of MCMC steps to run for [100 000]")
    ap.set_defaults(targets_file=Path("./config/plato-lops2-tess-ebs-explicit-targets.json"),
                    write_diags=False, max_mcmc_steps=100000, mcmc_walkers=100, mcmc_thin_by=10)
    args = ap.parse_args()
    drop_dir = Path.cwd() / f"drop/{args.targets_file.stem}"
    args.working_set_file = drop_dir / "working-set.table"


    with redirect_stdout(Tee(open(drop_dir / f"{THIS_STEM}.log", "a", encoding="utf8"))):
        print(f"\n\nStarted {THIS_STEM} at {datetime.now():%Y-%m-%d %H:%M:%S%z %Z}")

        targets_config = Targets(args.targets_file)
        print(f"Read in the configuration from '{args.targets_file}'",
              f"which contains {targets_config.count()} target(s).")

        dal = QTableFileDal(args.working_set_file)
        to_fit_targets = list(dal.keys_where("fitted_masses", lambda val: val == False)) # pylint: disable=singleton-comparison
        to_fit_count = len(to_fit_targets)
        print(f"The data indicates there are {to_fit_count} target(s) to be fitted.")


        for fit_counter, target in enumerate(to_fit_targets, start=1):
            target_config = targets_config.get(target)
            print("\n\n============================================================")
            print(f"Processing target {fit_counter} of {to_fit_count}: {target}")
            print("============================================================")


            print("Getting known values from previous steps to set up fitting priors")
            TeffA, TeffB, RA, RB = dal.read_values(target, "TeffA", "TeffB", "RA", "RB")
            rA_plus_rB, k, period = dal.read_values(target, "rA_plus_rB", "k", "period")
            rA = rA_plus_rB / (k + 1)
            rB = rA_plus_rB / ((1/k) + 1)
            print("\n".join(f"{p:>20s}: {v:12.3f}" for p, v in [("TeffA", TeffA), ("TeffB", TeffB),
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


            # Estimate fit starting position with masses based on the observed radii
            # and a reasonable M-S age given the likely mass regime
            theta_masses = np.array([M_sys * r for r in prior_radii])
            theta_masses *= (M_sys / np.sum(theta_masses))
            theta_age = [9.0 if max(nominal_values(theta_masses)) <= 2.0 else 8.0]
            theta0 = np.concatenate([nominal_values(theta_masses), theta_age])


            print()
            print("Performing a quick minimize fit for approximate values for masses and log(age)")
            print_mass_theta(theta0, "theta0")
            theta_fit, _ = minimize_fit(theta0=theta0,
                                        sys_mass=M_sys,
                                        radii=prior_radii,
                                        teffs=prior_Teffs,
                                        verbose=True)
            print_mass_theta(theta_fit, "theta_min")


            print()
            print("Performing a full MCMC fit for masses and log(age) with uncertainties.")
            theta_fit, _ = mcmc_fit(theta0=theta_fit,
                                    sys_mass=M_sys,
                                    radii=prior_radii,
                                    teffs=prior_Teffs,
                                    nwalkers=args.mcmc_walkers,
                                    nsteps=args.max_mcmc_steps,
                                    thin_by=args.mcmc_thin_by,
                                    seed=42,
                                    early_stopping=True,
                                    processes=8,
                                    progress=True,
                                    verbose=True)
            print_mass_theta(theta_fit, "theta_mcmc")


            print(f"Parameters for {target} with nominals & 1-sigma uncertainties",
                  "from MCMC fit ([known value])")
            for (k, unit), theta_val in zip(fit_labels_and_units.items(), theta_fit):
                known = ""
                if target_config.get("labels", {}).get(k, None) is not None:
                    lvalue = ufloat(target_config.labels.get(k, np.NaN),
                                    target_config.labels.get(k + "_err", 0))
                    known = f"({lvalue:.3f} {unit:unicode})"
                print(f"{k:>12s} = {theta_val:.3f} {unit:unicode} \t", known)

            dal.write_values(target, **{
                **{k: v for k, v in zip(fit_labels_and_units.keys(), theta_fit) if k != "log(age)"},
                "fitted_masses": True
            })

        print(f"\nCompleted {THIS_STEM} at {datetime.now():%Y-%m-%d %H:%M:%S%z %Z}")
