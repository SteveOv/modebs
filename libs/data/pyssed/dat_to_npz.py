""" Writes out a npz file with a pre-conv grid based on the pyssed bt-settl dat file """
from pathlib import Path as _Path
from inspect import getsourcefile as _getsourcefile
import json

import numpy as np

# pylint: disable=no-member
import astropy.units as u


# The source for this is the pre-convolved grid of summed fluxes by filter from pyssed.
# Download into this directory from:
# https://github.com/iain-mcdonald/PySSED/blob/v1.1.20250129/src/model-bt-settl-recast.dat
this_dir = _Path(_getsourcefile(lambda:0)).parent
source_file = this_dir / "model-bt-settl-recast.dat"
in_grid = np.genfromtxt(source_file, names=True, delimiter=" ",
                        deletechars=r" ~!@#$%^&*()=+~\|]}[{';: ?>,<")

# This contains mappings from supported VizerSED to pyssed filter names
# Get those which are in the source and for which we have translations.
with open(this_dir / "sed-filter-translation.json", "r", encoding="utf8") as j:
    filter_translator = json.load(j)
    reverse_filter_translator = { v: k for k, v in filter_translator.items() }
in_filters = [f for f in in_grid.dtype.names if f in reverse_filter_translator.keys()]
out_filters = [reverse_filter_translator[f] for f in in_filters]

# Copy the grid and then update it to use the revised/output filter names
key_columns = ["teff", "logg", "metal", "alpha"]
out_grid_conv = in_grid[key_columns + in_filters].copy()
out_grid_conv.dtype.names = tuple(key_columns + out_filters)

# Should already be in this order, but just in case
out_grid_conv.sort(order=key_columns)

# Publish the units in the pre-conv grid as a table of; column, unit
key_col_units = [u.K, u.dex, u.dimensionless_unscaled, u.dimensionless_unscaled]
units = np.array(
    list(zip(key_columns, key_col_units)) + [("lambda", u.um), ("flux", u.Jy)],
    dtype=[("column", "<U6"), ("unit", np.dtype(u.Unit))])

# Publish the ranges for the pre-conv grid in a table of; column, (min, max)
key_ranges = [(n, (out_grid_conv[n].min(), out_grid_conv[n].max()) << u)
                                                for n, u in zip(key_columns, key_col_units)]
ranges = np.array(key_ranges + [("lambda", (0.3, 22) << u.um)],
                  dtype=[("column", "<U6"), ("range", np.dtype(u.Quantity))])

output_file = this_dir / f"{source_file.stem}.npz"
np.savez_compressed(output_file, model_grid_conv=out_grid_conv, units=units, ranges=ranges)

# Test what has been saved
with np.load(output_file, allow_pickle=True) as of:
    loaded_grid = of["model_grid_conv"]
    print(f"Reloaded {output_file.name}; model_grid_conv has {loaded_grid.shape[0]} rows\nover",
          f"the fields {loaded_grid.dtype.names}")

    print("Units:", ", ".join(f"{u['column']}={u['unit']:unicode}" for u in of["units"]))

    print("Ranges:", ", ".join(f"{r['column']}={r['range']:unicode}" for r in of["ranges"]))

    row_mask = (loaded_grid["teff"] == 3000) & (loaded_grid["logg"] == 4.0) \
            & (loaded_grid["metal"] == 0.0) & (loaded_grid["alpha"] == 0)
    print("'Gaia:Gbp' filter flux for teff=3000, logg=4.0, metal==0, alpha==0:",
          loaded_grid["Gaia:Gbp"][row_mask], f"{units[units['column']=='flux']['unit'][0]:unicode}",
          "(expecting 4.019e16)")
