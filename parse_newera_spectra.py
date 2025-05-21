#!/usr/bin/env python3
""" Script to parse previously downloaded full Z=0 spectra files from the NewEra archive. """
# pylint: disable=no-member
from pathlib import Path
import pickle as pkl

import numpy as np
from scipy.interpolate import interp1d
import astropy.units as u

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
bandpass = (0.25, 20.0) * u.micron
table = np.empty((spec_count, ), dtype=fields_dtype + [("flux_interp", object)])

mask_sl = slice(None, None, 10)

for spec_ix, spec_file in enumerate(spec_files):
    print(f"Parsing file {spec_ix+1} of {spec_count}: {spec_file.name}", end="")
    with h5py.File(spec_file) as sf:
        # wavelents stored as Angstroem in vacuum, fluxes as log10(F_lambda [erg/cm^2/s/cm])
        wl = sf["/PHOENIX_SPECTRUM_LSR/wl"][()] * u.Angstrom
        fl = 10.**sf["/PHOENIX_SPECTRUM_LSR/fl"][()] / 1e8 * u.erg / (u.cm**2 * u.s * u.Angstrom)

        # Get the flux F(nu) and then into SI units
        fnu = 3.34e-19 * wl.value**2 * fl.value * u.erg / (u.cm**2 * u.s * u.Hz)
        fnu = fnu.to(u.W / (u.m**2 * u.Hz)).value

        # Create the row, with an interp1d for the freq to flux
        mask = (wl > min(bandpass)) & (wl < max(bandpass))
        wl = (wl[mask][mask_sl]).to(u.micron).value
        table[spec_ix]["flux_interp"] = interp1d(c*1e6 / wl, fnu[mask][mask_sl], kind="linear")
        table[spec_ix]["lam_from"] = wl.min()
        table[spec_ix]["lam_to"] = wl.max()
        table[spec_ix]["lam_steps"] = len(wl)

        nml = f90nml.reads(str(sf['/PHOENIX_NAMELIST/phoenix_nml'][()].tobytes())[2:-1])
        for field, nml_key in fields_index.items():
            table[spec_ix][field] = nml["phoenix"][nml_key]
        print(f"...stored flux interp at {len(wl)} frequencies")

# Save with interps serialized
np.save("libs/data/newera/PHOENIX-NewEra-LR.Z-0.0.npy", table, allow_pickle=True)
