""" Pipeline Stage 5 - plots of fitted targets """
# pylint: disable=no-member, no-name-in-module, singleton-comparison
from inspect import getsourcefile
from pathlib import Path
import argparse
from datetime import datetime
from contextlib import redirect_stdout
import numpy as np
import matplotlib.pyplot as plt
from uncertainties.unumpy import nominal_values

from astropy.constants import sigma_sb
from astropy.constants.iau2015 import L_sun, R_sun

from libs import plots
from libs.iohelpers import Tee
from libs.targets import Targets
from libs.pipeline_dal import QTableFileDal

THIS_STEM = Path(getsourcefile(lambda: 0)).stem


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Pipeline stage 1: ingest of targets.")
    ap.add_argument("-tf", "--targets-file", dest="targets_file", type=Path, required=False,
                    help="json file containing the details of the targets to ingest")
    ap.add_argument("-ft", "--figs-type", dest="figs_type", type=str, required=False,
                    help="the type of fig, as indicated by its file extension [png]")
    ap.add_argument("-fd", "--figs-dpi", dest="figs_dpi", type=int, required=False,
                    help="the dpi of any figs, if they are a raster type [100]")
    ap.set_defaults(targets_file=Path("./config/plato-lops2-tess-ebs-explicit-targets.json"),
                    figs_type="png", figs_dpi=100)
    args = ap.parse_args()
    drop_dir = Path.cwd() / f"drop/{args.targets_file.stem}"
    args.working_set_file = drop_dir / "working-set.table"

    with redirect_stdout(Tee(open(drop_dir / f"{THIS_STEM}.log", "a", encoding="utf8"))) as log:
        print("\n\n============================================================")
        print(f"Started {THIS_STEM} at {datetime.now():%Y-%m-%d %H:%M:%S%z %Z}")
        print("============================================================")

        figs_dir = drop_dir / "figs"
        figs_dir.mkdir(parents=True, exist_ok=True)

        wset = QTableFileDal(args.working_set_file)
        to_plot_target_ids = list(wset.yield_keys("fitted_lcs", "fitted_sed", "fitted_masses",
                                                  where=lambda fl, fs, fm: fl == fs == fm == True))
        to_fit_count = len(to_plot_target_ids)
        print(f"The working-set indicates there are {to_fit_count} targets to be plotted.")


        print("Plotting a Mass-Radius log-log diagram")
        row_gen = wset.yield_values("target_id", "MA", "MB", "RA", "RB")
        masses_and_radii = np.array([row[1:] for row in row_gen if row[0] in to_plot_target_ids]).T
        masses = masses_and_radii[:2]
        radii = masses_and_radii[2:]
        fig = plots.plot_mass_radius_diagram(masses, radii, labels=["star A", "star B"],
                                             plot_zams=True, legend_loc="best", invertx=True)
        fig.savefig(figs_dir / f"mass-radius.{args.figs_type}", dpi=args.figs_dpi)
        plt.close(fig)


        print("Plotting a Hurtzsprung-Russell diagram")
        row_gen = wset.yield_values("target_id", "TeffA", "TeffB", "RA", "RB")
        Teffs_and_radii = np.array([row[1:] for row in row_gen if row[0] in to_plot_target_ids]).T
        Teffs = Teffs_and_radii[:2]
        lums = ((4 * np.pi * (Teffs_and_radii[2:] * R_sun)**2 * sigma_sb * Teffs**4) / L_sun).value

        fig = plots.plot_hr_diagram(Teffs, lums, labels=["star A", "star B"],
                                    plot_zams=True, legend_loc="best", invertx=True)
        fig.savefig(figs_dir / f"hurtzsprung-russell.{args.figs_type}", dpi=args.figs_dpi)
        plt.close(fig)


        print("\n\n============================================================")
        print(f"Completed {THIS_STEM} at {datetime.now():%Y-%m-%d %H:%M:%S%z %Z}")
        print("============================================================")
