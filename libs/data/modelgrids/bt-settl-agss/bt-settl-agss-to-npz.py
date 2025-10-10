""" Writes out a npz file based on separately downloaded bt-settl agss txt files """
from pathlib import Path as _Path
from inspect import getsourcefile as _getsourcefile
import json
import re
import warnings

from urllib.parse import quote_plus

import numpy as np
from scipy.interpolate import interp1d

from astropy.table import Table
from astropy.io.votable import parse_single_table

# pylint: disable=no-member
import astropy.units as u
from astropy.units import UnitsWarning

# We parse units as text from votables & text files. Stop us getting swamped format with warnings.
warnings.filterwarnings("ignore", category=UnitsWarning)

def get_filter(name) -> Table:
    filter_cache_dir = _Path(_getsourcefile(lambda:0)).parent / "../../../../.cache/.filters/"
    filter_cache_dir.mkdir(parents=True, exist_ok=True)

    filter_fname = (filter_cache_dir / (re.sub(r"[^\w\d.-]", "-", name) + ".xml")).resolve()
    if not filter_fname.exists():
        try:
            fid = quote_plus(name)
            table = Table.read(f"https://svo2.cab.inta-csic.es/theory/fps/fps.php?ID={fid}")
            table.write(filter_fname, format="votable")
        except ValueError as err:
            raise ValueError(f"No filter table for filter={name}") from err

    table = parse_single_table(filter_fname).to_table()
    table.sort("Wavelength")
    return table


# Files and file locations
this_dir = _Path(_getsourcefile(lambda:0)).parent
source_dir = (this_dir / "../../../../.cache/.modelgrids/bt-settl-agss/").resolve()
source_files = sorted(source_dir.glob("lte*.dat.txt"))
out_file_stem = source_dir.name

# Get the filters
with open(this_dir / "sed-filter-translation.json", "r", encoding="utf8") as j:
    filter_translator = json.load(j)
filters = { f: get_filter(n) for f, n in filter_translator.items() }

# Metadata for the output grid
key_columns = ["teff", "logg", "metal", "alpha"]    # These are the same for every model grid
column_names = key_columns + list(filters.keys())
grid_conv_data = []
OUT_LAM_UNIT = u.um
OUT_FLUX_UNIT = u.Jy
filters_short, filters_long = np.inf * OUT_LAM_UNIT, 0 * OUT_LAM_UNIT

# Read in each source file, parse it, calculate the filter fluxes then store as a row
param_re = re.compile(r"^#[\s*](?P<k>\w*)[\s]*=[\s]*(?P<val>[+-]?([0-9]*[.])?[0-9]+)", re.MULTILINE)
lambda_unit_re = re.compile(r"Wavelength in (?P<unit>[\w\/]*)$", re.MULTILINE)
flux_unit_re = re.compile(r"Flux in (?P<unit>[\w\/]*)$", re.MULTILINE)
for file_ix, source_file in enumerate(source_files[:]):

    # First few lines of each file has metadata on file's teff/logg/meta/alpha and the units in use
    with open(source_file, mode="r", encoding="utf8") as sf:
        text = sf.read(500)
        metadata = {
            "teff_unit": u.K,
            "logg_unit": u.dex,
            "metal_unit": u.dimensionless_unscaled,
            "alpha_unit": u.dimensionless_unscaled,
            "lambda_unit": u.Unit(lambda_unit_re.findall(text)[0]),
            "flux_unit": u.Unit(flux_unit_re.findall(text)[0].replace("/A", "/Angstrom")),
            **{ m.group("k"): float(m.group("val")) for m in param_re.finditer(text) },
        }

        if "meta" in metadata and not "metal" in metadata:
            metadata["metal"] = metadata.pop("meta")
        del text

    # Get the data, and then coerce the fluxes into Jy
    lam, flux = np.genfromtxt(source_file, dtype=float, comments="#", unpack=True)
    lam = (lam * metadata["lambda_unit"]).to(OUT_LAM_UNIT, equivalencies=u.spectral())
    lam_short, lam_long = lam.min(), lam.max()
    flux = (flux * metadata["flux_unit"]).to(OUT_FLUX_UNIT, equivalencies=u.spectral_density(lam))

    print(f"{file_ix+1}/{len(source_files)} {source_file.name} [{len(lam):,d} rows]:",
          ", ".join(f"{k}={metadata[k]: .2f}" for k in key_columns), end="...")

    # This is where the magic happens! We need to overlay the filter onto the fluxes to
    # apply its sensitivity, then sum what is transmitted and finally convert to Jy
    filter_fluxes = []
    for ix, (filter_name, filter_grid) in enumerate(filters.items()):
        filter_lam = filter_grid["Wavelength"].quantity
        filter_short = filter_lam.min()
        filter_long = filter_lam.max()

        # Work out the lamda range where the filter and data overlap
        ol_lam_short = max(lam_short, filter_short)
        ol_lam_long = min(lam_long, filter_long)
        if ol_lam_short < ol_lam_long:
            ol_filter = filter_grid[(ol_lam_short <= filter_lam) & (filter_lam <= ol_lam_long)]
            flux_ixs = np.where((ol_lam_short <= lam) & (lam <= ol_lam_long))[0]

            bin_size = int(np.floor(len(flux_ixs) / len(ol_filter)))
            if bin_size > 1:
                # Rebin that model subset to approximately match the filter resolution
                flux_mid_ix = int(np.median(flux_ixs))
                num_flux_points = int(np.floor(len(flux_ixs) / bin_size) * bin_size)
                flux_short_ix = int(np.floor(flux_mid_ix - num_flux_points / 2))
                ol_flux_slice = slice(flux_short_ix, flux_short_ix + num_flux_points)

                # Apply the binning to the region of the model covered by the filter.
                binned_len = num_flux_points // bin_size
                binned_lam = lam[ol_flux_slice].reshape(binned_len, bin_size, -1)\
                                                                    .mean(axis=-2).mean(axis=1)
                binned_flux = flux[ol_flux_slice].reshape(binned_len, bin_size, -1)\
                                                                    .mean(axis=-2).mean(axis=1)
            else:
                # Binning not required
                binned_lam = lam[flux_ixs]
                binned_flux = flux[flux_ixs]

            try:
                # Get the interpolated filter and normalize. Breaks if the units are not same.
                filter_interp = interp1d(
                    ol_filter["Wavelength"].to(lam.unit, equivalencies=u.spectral()),
                    ol_filter["Transmission"])(binned_lam)
                filter_interp[filter_interp < 0] = 0
                filter_interp /= np.sum(filter_interp)

                avg_lam = np.sum(binned_lam * filter_interp)

                # Convolve the filter and model
                flux_energy = binned_flux * filter_interp
                flux_photons = flux_energy * binned_lam / avg_lam
                filter_flux = np.sum(flux_photons)
            except:
                # TODO: dirty temp workaround for outside range of interpolator issues
                filter_flux = 0 * OUT_FLUX_UNIT
                
            filter_fluxes += [filter_flux.value]
            
            # Record the maximum extent of the filter coverage
            if filter_short < filters_short:
                filters_short = filter_short.to(OUT_LAM_UNIT, equivalencies=u.spectral())
            if filter_long > filters_long:
                filters_long = filter_long.to(OUT_LAM_UNIT, equivalencies=u.spectral())
        else:
            # The filter is outside the range of the data
            print(f"Assuming 0 flux for filter {filter_name} as it doesn't overlap with the model")
            filter_fluxes  += [0.]

    # Store this row of pre-filtered measurements
    new_row = [tuple([metadata[k] for k in key_columns] + filter_fluxes)]
    grid_conv_data += new_row
    print(f"adding row of {len(filter_fluxes)} convolved filter fluxes")
    #print(new_row)


# Get the data into a numpy structured array with the expected columns
# Should already be in this order, but just in case
out_grid_conv = np.array(grid_conv_data, dtype=[(cn, float) for cn in column_names])
out_grid_conv.sort(order=key_columns)

# Now we write out the model grid's data and metadata to a compressed npz file
key_columns_and_units = [(k, metadata.get(f"{k}_unit")) for k in key_columns]
key_columns_and_ranges = [(k, (out_grid_conv[k].min(), out_grid_conv[k].max()) << u)
                                                            for k, u in key_columns_and_units]

units = np.array(key_columns_and_units + [("lambda", OUT_LAM_UNIT), ("flux", OUT_FLUX_UNIT)],
                 dtype=[("column", "<U6"), ("unit", np.dtype(u.Unit))])
ranges = np.array(key_columns_and_ranges + [("lambda", (filters_short, filters_long)*OUT_LAM_UNIT)],
                  dtype=[("column", "<U6"), ("range", np.dtype(u.Quantity))])

filter_map = np.array([(v, k) for k, v in filter_translator.items()],
                      dtype=[("vizier", "<U50"), ("column", "<U50")])

out_file = this_dir / f"{out_file_stem}.npz"
print(f"Saving model grid and metadata to {out_file}")
np.savez_compressed(out_file, model_grid_conv=out_grid_conv,
                    units=units, ranges=ranges, filter_map=filter_map)

# Test what has been saved
with np.load(out_file, allow_pickle=True) as of:
    loaded_grid = of["model_grid_conv"]
    print(f"Reloaded {out_file.name}; model_grid has {loaded_grid.shape[0]} rows\nover",
          f"the fields {loaded_grid.dtype.names}")

    print("Units:", ", ".join(f"{u['column']}={u['unit']:unicode}" for u in of["units"]))

    print("Ranges:", ", ".join(f"{r['column']}={r['range']:unicode}" for r in of["ranges"]))

    qfilter = list(filters.keys())[0]
    qmask = (loaded_grid["teff"] == 2000) & (loaded_grid["logg"] == 4.0) \
            & (loaded_grid["metal"] == 0.0) & (loaded_grid["alpha"] == 0)
    print(f"'{qfilter}' filter flux for teff=2000, logg=4.0, metal==0, alpha==0:",
          loaded_grid[qfilter][qmask], f"{units[units['column']=='flux']['unit'][0]:unicode}")
