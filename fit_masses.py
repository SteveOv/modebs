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
from astropy.table import QTable

# pylint: disable=line-too-long, wrong-import-position
warnings.filterwarnings("ignore", "Using UFloat objects with std_dev==0 may give unexpected results.", category=UserWarning)
from uncertainties import ufloat
from uncertainties.unumpy import nominal_values

from deblib.constants import G, R_sun, M_sun

from libs.iohelpers import Tee
from libs.targets import Targets
from libs.fit_masses import minimize_fit, mcmc_fit


THIS_STEM = Path(getsourcefile(lambda: 0)).stem
NUM_STARS = 2
subs = "ABCDEFGHIJKLM"
mass_fit_labels = np.array([(f"M{subs[st]}", u.Msun) for st in range(NUM_STARS)] \
                          +[("log(age)", u.dex(u.yr))])


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
    args.input_file = args.output_file = drop_dir / "working-set.table"


    with redirect_stdout(Tee(open(drop_dir / f"{THIS_STEM}.log", "a", encoding="utf8"))):
        print(f"\n\nStarted {THIS_STEM} at {datetime.now():%Y-%m-%d %H:%M:%S%z %Z}")
        
        targets_config = Targets(args.targets_file)
        print(f"Read in the configuration from '{args.input_file}'",
              f"which contains {targets_config.count()} target(s).")

        tdata = QTable.read(args.input_file)
        to_fit_row_ixs = np.where(tdata["fitted_masses"] == False)[0] # pylint: disable=singleton-comparison
        to_fit_count = len(to_fit_row_ixs)
        print(f"Reading '{args.input_file}' which contains {to_fit_count} target(s) to be fitted."
              f"\nWill write updated data to '{args.output_file}'")


        for fit_counter, row_ix in enumerate(to_fit_row_ixs, start=1):
            trow = tdata[row_ix]
            target = trow["target"]
            target_config = targets_config.get(target)
            print("\n\n============================================================")
            print(f"Processing target {fit_counter} of {to_fit_count}: {target}")
            print("============================================================")


            print("Getting known values from previous steps to set up fitting priors")
            TeffA = ufloat(trow["TeffA"].value, trow["TeffA_err"].value)
            TeffB = ufloat(trow["TeffB"].value, trow["TeffB_err"].value)
            RA = ufloat(trow["RA"].value, trow["RA_err"].value)
            RB = ufloat(trow["RB"].value, trow["RB_err"].value)
            rA_plus_rB = ufloat(trow["rA_plus_rB"], trow["rA_plus_rB_err"])
            k = ufloat(trow["k"], trow["k_err"])
            rA = rA_plus_rB / (k + 1)
            rB = rA_plus_rB / ((1/k) + 1)
            period = ufloat(trow["period"].value, trow["period_err"].value)
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
            mass_theta0 = np.concatenate([nominal_values(theta_masses), theta_age])


            print()
            print("Performing a quick minimize fit for approximate values for masses and log(age)")
            print_mass_theta(mass_theta0, "theta0")
            theta_min, _ = minimize_fit(theta0=mass_theta0,
                                        sys_mass=M_sys,
                                        radii=prior_radii,
                                        teffs=prior_Teffs,
                                        verbose=True)
            print_mass_theta(theta_min, "theta_min")


            print()
            print("Performing a full MCMC fit for masses and log(age) with uncertainties.")
            theta_mcmc, mass_sampler = mcmc_fit(theta0=theta_min,
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
            print_mass_theta(theta_mcmc, "theta_mcmc")


            print(f"Parameters for {target} with nominals & 1-sigma uncertainties",
                  "from MCMC fit ([known value])")
            for (k, unit), theta_val in zip(mass_fit_labels, theta_mcmc):
                known = ""
                if target_config.get("labels", {}).get(k, None) is not None:
                    lvalue = ufloat(target_config.labels.get(k, np.NaN),
                                    target_config.labels.get(k + "_err", 0))
                    known = f"({lvalue:.3f} {unit:unicode})"
                print(f"{k:>12s} = {theta_val:.3f} {unit:unicode} \t", known)

                # *** also updates the target data ***
                if k not in ["log(age)"]:
                    tdata[row_ix][k] = theta_val.n * unit
                    tdata[row_ix][f"{k}_err"] = theta_val.s * unit

            tdata[row_ix]["fitted_masses"] = True

            print(f"Writing updated data set to {args.output_file}")
            tdata.write(args.output_file, format="votable", overwrite=True)

            if args.write_diags:
                tdata.write(drop_dir / f"{THIS_STEM}.diag", format="ascii.fixed_width_two_line",
                            header_rows=["name", "dtype", "unit"], overwrite=True)

        print(f"\nCompleted {THIS_STEM} at {datetime.now():%Y-%m-%d %H:%M:%S%z %Z}")
