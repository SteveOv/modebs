#!/usr/bin/env python3
""" Pipeline Stage 5 - plots of fitted targets """
# pylint: disable=no-member, no-name-in-module, singleton-comparison
from inspect import getsourcefile
from pathlib import Path
import argparse
from datetime import datetime
from contextlib import redirect_stdout
import numpy as np
import matplotlib.pyplot as plt

from astropy.constants import sigma_sb
from astropy.constants.iau2015 import L_sun, R_sun

from libs import plots
from libs.iohelpers import Tee
from libs.targets import Targets
from libs.pipeline_dal3 import create_dal

THIS_STEM = Path(getsourcefile(lambda: 0)).stem


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Pipeline stage 1: ingest of targets.")
    ap.add_argument(dest="targets_file", type=Path, metavar="TARGETS_FILE",
                    help="json file containing the details of the targets to fit")
    ap.add_argument("-ft", "--figs-type", dest="figs_type", type=str, required=False,
                    help="the type of fig, as indicated by its file extension [png]")
    ap.add_argument("-fd", "--figs-dpi", dest="figs_dpi", type=int, required=False,
                    help="the dpi of any figs, if they are a raster type [100]")
    ap.set_defaults(figs_type="png", figs_dpi=100)
    args = ap.parse_args()
    drop_dir = Path.cwd() / f"drop/{args.targets_file.stem}"
    args.working_set_file = drop_dir / "working-set.table"

    with redirect_stdout(Tee(open(drop_dir / f"{THIS_STEM}.log", "a", encoding="utf8"))) as log:
        print("\n\n============================================================")
        print(f"Started {THIS_STEM} at {datetime.now():%Y-%m-%d %H:%M:%S%z %Z}")
        print("============================================================")
        print(f"\nThe targets configuration file:   {args.targets_file}")
        print(f"Directory for data, logs & plots: {drop_dir}")

        figs_dir = drop_dir / "figs"
        figs_dir.mkdir(parents=True, exist_ok=True)

        targets_config = Targets(args.targets_file)
        print(f"\nRead in the configuration from '{args.targets_file.name}'",
              f"which contains {targets_config.count()} target(s) that have not been exluded.")

        # Open the targets table and the configs
        dal_kwargs = targets_config.get("dal_kwargs", {})
        dal_kwargs.setdefault("file", drop_dir / "working-set.table")
        dal = create_dal(targets_config.get("dal_type", "QTableFileDal3"), True, **dal_kwargs)

        # These plots require the pipeline to fitted SEDs for Teffs & Radii
        # ----------------------------------------------------------------------
        where = { "fitted_sed": True }
        to_plot_count = dal.count_where(**where)
        print(f"\nThe working-set has {to_plot_count} targets that have fitted for Teffs & radii.")
        if to_plot_count:

            print("Plotting a Hertzsprung-Russell diagram")
            rows = np.array([(r.TeffA, r.TeffB, r.RA, r.RB) for r in dal.iterate_rows(**where)]).T
            Teffs, radii = rows[:2], rows[2:]
            lums = ((4 * np.pi * (radii * R_sun)**2 * sigma_sb * Teffs**4) / L_sun).value

            fig = plots.plot_hr_diagram(Teffs, lums, labels=["star A", "star B"],
                                        plot_zams=True, legend_loc="best", invertx=True)
            fig.savefig(figs_dir / f"hertzsprung-russell.{args.figs_type}", dpi=args.figs_dpi)
            plt.close(fig)


        # These plots require the pipeline to have fitted for radii (SED) & masses
        # ----------------------------------------------------------------------
        where["fitted_masses"] = True
        to_plot_count = dal.count_where(**where)
        print(f"\nThe working-set has {to_plot_count} targets that have fitted for radii & masses.")
        if to_plot_count:

            print("Plotting a Mass-Radius log-log diagram")
            rows = np.array([(r.MA, r.MB, r.RA, r.RB) for r in dal.iterate_rows(**where)]).T
            masses, radii = rows[:2], rows[2:]
            fig = plots.plot_mass_radius_diagram(masses, radii, labels=["star A", "star B"],
                                                 plot_zams=True, legend_loc="best", invertx=True)
            fig.savefig(figs_dir / f"mass-radius.{args.figs_type}", dpi=args.figs_dpi)
            plt.close(fig)


        print("\n\n============================================================")
        print(f"Completed {THIS_STEM} at {datetime.now():%Y-%m-%d %H:%M:%S%z %Z}")
        print("============================================================")
