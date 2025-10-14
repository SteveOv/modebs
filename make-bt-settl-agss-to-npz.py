""" Writes out a npz file based on separately downloaded bt-settl agss txt files """
from pathlib import Path as _Path
from inspect import getsourcefile as _getsourcefile
import json
import re
import warnings

from urllib.parse import quote_plus

import numpy as np

from astropy.table import Table
from astropy.io.votable import parse_single_table

# pylint: disable=no-member
import astropy.units as u
from astropy.units import UnitsWarning

# We parse units as text from votables & text files. Stop us getting swamped format with warnings.
warnings.filterwarnings("ignore", category=UnitsWarning)

def get_filter(svo_name) -> Table:
    filter_cache_dir = _Path(_getsourcefile(lambda:0)).parent / ".cache/.filters/"
    filter_cache_dir.mkdir(parents=True, exist_ok=True)

    filter_fname = (filter_cache_dir / (re.sub(r"[^\w\d.-]", "-", svo_name) + ".xml")).resolve()
    if not filter_fname.exists():
        try:
            fid = quote_plus(svo_name)
            table = Table.read(f"https://svo2.cab.inta-csic.es/theory/fps/fps.php?ID={fid}")
            table.write(filter_fname, format="votable")
        except ValueError as err:
            raise ValueError(f"No filter table in SVO for filter={svo_name}") from err

    table = parse_single_table(filter_fname).to_table()

    # Create a normalized copy of the transmission data
    ftrans = table["Transmission"]
    table["Norm-Transmission"] = (ftrans - ftrans.min()) / (ftrans.max() - ftrans.min())

    # Add metadata on the filter coverage
    flam = table["Wavelength"].to(u.AA, equivalencies=u.spectral())
    table.meta["filter_short"] = flam.min()
    table.meta["filter_long"] = flam.max()
    table.meta["filter_mid"] = np.median(flam)

    table.sort("Wavelength")
    return table


# Files and file locations.
# Download bt-settl-aggs ascii model grids from following url decompress the tgz contents into
# .cache/.modelgrids/ within the bt-settl-agss dir.
# https://svo2.cab.inta-csic.es/theory/newov2/index.php?models=bt-settl-agss
this_dir = _Path(_getsourcefile(lambda:0)).parent
source_dir = (this_dir / ".cache/.modelgrids/bt-settl-agss/").resolve()
source_files = sorted(source_dir.glob("lte*.dat.txt"))
out_file_stem = source_dir.name

# Populate the dict of known/supported filters. Will download a cache these under .cache/.filters/
with open(this_dir / "config/sed-filter-translation.json", "r", encoding="utf8") as j:
    filter_translator = json.load(j)
filters = { f: get_filter(svoname) for f, svoname in filter_translator.items() }

# Metadata for the output grid and pre-allocate it
index_col_names = ["teff", "logg", "metal", "alpha"]    # These are present for every model grid
col_names = index_col_names + list(filters.keys())
OUT_LAM_UNIT = u.um                             # u.AA
OUT_FLUX_DENSITY_UNIT = u.W / u.m**2 / u.Hz     # u.erg / u.s / u.cm**2 / u.AA
OUT_FLUX_UNIT = u.W / u.m**2                    # u.erg / u.s / u.cm**2
filters_short, filters_long = np.inf * OUT_LAM_UNIT, 0 * OUT_LAM_UNIT
out_grid_conv = np.zeros((len(source_files), len(col_names)), dtype=float)

# Read in each source file, parse it, calculate the filter fluxes then store as a row
param_re = re.compile(r"^#[\s*](?P<k>\w*)[\s]*=[\s]*(?P<val>[+-]?([0-9]*[.])?[0-9]+)", re.MULTILINE)
lambda_unit_re = re.compile(r"Wavelength in (?P<unit>[\w\/]*)$", re.MULTILINE)
flux_unit_re = re.compile(r"Flux in (?P<unit>[\w\/]*)$", re.MULTILINE)
for file_ix, source_file in enumerate(source_files[:]):

    # First few lines of each file has metadata on the teff/logg/meta/alpha and units in the file
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

    # "Index" values
    out_grid_conv[file_ix, 0:4] = [metadata[k] for k in index_col_names]

    # Get the data, and then coerce them to CGS units for ease of processing
    lam, flux_density = np.genfromtxt(source_file, dtype=float, comments="#", unpack=True)
    lam = (lam * metadata["lambda_unit"]).to(u.AA, equivalencies=u.spectral())
    flux_density = (flux_density * metadata["flux_unit"]).to(u.erg / u.s / u.cm**2 / u.AA,
                                                         equivalencies=u.spectral_density(lam))

    print(f"{file_ix+1}/{len(source_files)} {source_file.name} [{len(lam):,d} rows]:",
          ", ".join(f"{k}={metadata[k]: .2f}" for k in index_col_names), end="...")

    # Bin the lambdas so we can get the data volumes down to something manageable
    lam_bins = 10**np.arange(np.log10(913), np.log10(320000), 0.003) # Angstrom
    lam_bins_short, lam_bins_long = lam_bins.min() * u.AA, lam_bins.max() * u.AA

    # Bin the fluxes
    lam_to_bin_ixs = np.searchsorted(lam_bins, lam, "left")
    flux_bins = np.zeros_like(lam_bins, dtype=float)
    for lam_bin_ix in range(len(lam_bins)):
        lam_ixs = np.where(lam_to_bin_ixs == lam_bin_ix)[0]
        if len(lam_ixs) > 0:
            flux_bins[lam_bin_ix] = flux_density[lam_ixs].value.mean()

    # Apply the units and multiply flux density by wavelength to give us the binned fluxes
    lam_bins = lam_bins << lam.unit
    flux_bins = (flux_bins << flux_density.unit) * lam_bins
    del lam, flux_density


    # This is where the magic happens! We need to overlay the filter onto the flux densities to
    # apply its sensitivity, then sum what is transmitted and finally convert to fluxes
    for filter_ix, (filter_name, filter_grid) in enumerate(filters.items()):

        # Work out the lambda range where the filter and binned data overlap
        ol_lam_short = max(lam_bins_short, filter_grid.meta["filter_short"])
        ol_lam_long = min(lam_bins_long, filter_grid.meta["filter_long"])

        if ol_lam_short < ol_lam_long:
            lam_filter = filter_grid["Wavelength"].quantity
            ol_filter = filter_grid[(ol_lam_short <= lam_filter) & (lam_filter <= ol_lam_long)]
            trans = ol_filter["Norm-Transmission"].value

            # Apply the filter & calculate overall transmitted flux value
            interp = np.interp(ol_filter["Wavelength"].to(u.AA), lam_bins, flux_bins / lam_bins)
            filter_flux = np.sum((interp * trans / np.sum(trans))) * filter_grid.meta["filter_mid"]

            # # For teff=2000.0, logg=3.5, metal=0.0 and GAIA:Gbp ~ 1.7e-14 erg / s / cm^2
            # test_flux = ( filter_flux * (1.0 * u.Rsun).to(u.cm)**2) \
            #             / (190.91243807532035 * u.pc).to(u.cm)**2

            out_grid_conv[file_ix, 4 + filter_ix] = \
                                    filter_flux.to(OUT_FLUX_UNIT, equivalencies=u.spectral()).value

            # Record the maximum extent of the filter coverage
            filters_short = min(filters_short, filter_grid.meta["filter_short"])
            filters_long = max(filters_long, filter_grid.meta["filter_long"])

    print(f"added row of {len(filters)} total filter fluxes")


# Get the data into a numpy structured array with the expected columns
# Should already be in this order, but just in case
out_grid_conv.dtype = [(cn, float) for cn in col_names]
out_grid_conv.sort(order=index_col_names)

# Now we write out the model grid's data and metadata to a compressed npz file
key_columns_and_units = [(k, metadata.get(f"{k}_unit")) for k in index_col_names]
key_columns_and_ranges = [(k, (out_grid_conv[k].min(), out_grid_conv[k].max()) << u)
                                                            for k, u in key_columns_and_units]

units = np.array(key_columns_and_units
                 + [("lambda", OUT_LAM_UNIT), ("flux", OUT_FLUX_UNIT)],
                 dtype=[("column", "<U6"), ("unit", np.dtype(u.Unit))])
ranges = np.array(key_columns_and_ranges
                  + [("lambda", (filters_short, filters_long) << OUT_LAM_UNIT)],
                  dtype=[("column", "<U6"), ("range", np.dtype(u.Quantity))])

filter_map = np.array([(v, k) for k, v in filter_translator.items()],
                      dtype=[("vizier", "<U50"), ("column", "<U50")])

out_file = this_dir / f"libs/data/modelgrids/{out_file_stem}/{out_file_stem}.npz"
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
