#!/usr/bin/env python3
""" Script to parse previously downloaded full Z=0 spectra files from the NewEra archive. """
from pathlib import Path

import numpy as np
from scipy.interpolate import interp1d
import pickle as pkl

# Specific requirement for these files
import h5py
import f90nml

from deblib.constants import c

cache_dir = Path(".cache/.newera_spectra/")
FILE_PATTERN = "lte?????-?.??-0.0.PHOENIX*.HSR.h5"
spec_files = sorted(cache_dir.glob(FILE_PATTERN))
spec_count = len(spec_files)
print(f"Found {spec_count} NewEra h5 files matching the glob '{FILE_PATTERN}' in '{cache_dir}'")

# Mappings for the columns in source phoenix_nml to named fields in a numpy structured array
# The table which will hold the interps and metadata
fields_index = { "teff": "teff", "logg": "logg", "mass": "m_sun" }
fields_dtype = [("teff", float), ("logg", float), ("mass", float),
                ("lam_from", float), ("lam_to", float), ("lam_steps", int)]
bandpass = (0.25, 20.0) # um
table = np.empty((spec_count, ), dtype=fields_dtype + [("flux_interp", object)])

mask_sl = slice(None, None, 10)

for spec_ix, spec_file in enumerate(spec_files):
    print(f"Parsing file {spec_ix+1} of {spec_count}: {spec_file.name}", end="")
    with h5py.File(spec_file) as sf:

        lams = sf["/PHOENIX_SPECTRUM_LSR/wl"][()] / 1e4     # Angstroem to um
        fluxes = 10.**sf["/PHOENIX_SPECTRUM_LSR/fl"][()]    # log10 to linear

        nml = f90nml.reads(str(sf['/PHOENIX_NAMELIST/phoenix_nml'][()].tobytes())[2:-1]) # pylint: disable=no-member

        # Create the row, with an interp1d for the freq to flux
        mask = (lams > min(bandpass)) & (lams < max(bandpass))
        lams = lams[mask][mask_sl]
        table[spec_ix]["flux_interp"] = interp1d(c*1e6 / lams, fluxes[mask][mask_sl], kind="linear")
        table[spec_ix]["lam_from"] = lams.min()
        table[spec_ix]["lam_to"] = lams.max()
        table[spec_ix]["lam_steps"] = len(lams)
        for field, nml_key in fields_index.items():
            table[spec_ix][field] = nml["phoenix"][nml_key]
        print(f"...stored flux interp at {len(lams)} frequencies")

#
save_file = Path("libs/data/newera/PHOENIX-NewEra-LR.Z-0.0.pkl")
np.save(save_file.parent / (save_file.stem + ".npy"), table, allow_pickle=True)
with open(save_file, "wb") as sf:
    pkl.dump(table, sf)
