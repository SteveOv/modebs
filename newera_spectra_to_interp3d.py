#!/usr/bin/env python3
""" Script to parse previously downloaded full Z=0 spectra files from the NewEra archive. """
# pylint: disable=no-member
from pathlib import Path
import re
import pickle as pkl

import numpy as np
import astropy.units as u

# Specific requirement for these files
import h5py
import f90nml

from deblib.constants import c

cache_dir = Path(".cache/.newera_spectra/")
FILE_PATTERN = "lte?????-?.??-0.0.PHOENIX*.HSR.h5"
spec_files = sorted(cache_dir.glob(FILE_PATTERN)) # will order by teff, logg
print(f"Found {len(spec_files)} NewEra h5 files matching '{FILE_PATTERN}' in '{cache_dir}'")

# We exclude certain combinations of Teff/logg as they are isolated cases not part of the table
valid_re = re.compile(r"lte(?P<teff>\d\d\d00)-(?P<logg>(?:\d.50|[1-9].00))-0.0")
spec_files = [f for f in spec_files if valid_re.match(f.name) is not None]
spec_count = len(spec_files)
print(f"After exclusions, {spec_count} NewEra h5 files will be used")

# Get the distinct set of teffs and loggs so we can pre-allocate storage
teffs = []
loggs = []
for sf in spec_files:
    match = valid_re.match(sf.name)
    if (teff := int(match.group("teff"))) not in teffs:
        teffs += [teff]
    if (logg := float(match.group("logg"))) not in loggs:
        loggs += [logg]

# Mappings for the columns in source phoenix_nml to named fields in a numpy structured array
# The table which will hold the interps and metadata
fields_index = { "teff": "teff", "logg": "logg", "mass": "m_sun" }
fields_dtype = [("teff", float), ("logg", float), ("mass", float),
                ("lam_from", float), ("lam_to", float), ("lam_steps", int)]
bandpass = (0.01, 30.0) * u.micron

points = None       # pylint: disable=invalid-name
values_3d = None    # pylint: disable=invalid-name

for spec_ix, spec_file in enumerate(spec_files):
    print(f"Parsing file {spec_ix+1} of {spec_count}: {spec_file.name}", end="")
    with h5py.File(spec_file) as sf:

        nml = f90nml.reads(str(sf['/PHOENIX_NAMELIST/phoenix_nml'][()].tobytes())[2:-1])
        teff = nml["phoenix"]["teff"]
        logg = nml["phoenix"]["logg"]
        teff_ix = teffs.index(teff)
        logg_ix = loggs.index(logg)

        # wavelengths stored as Angstroem in vacuum, fluxes as log10(F_lambda [erg/cm^2/s/cm])
        wl = sf["/PHOENIX_SPECTRUM_LSR/wl"][()] * u.Angstrom
        fl = 10.**sf["/PHOENIX_SPECTRUM_LSR/fl"][()] / 1e8 * u.erg / u.cm**2 / u.s / u.Angstrom

        mask = (wl >= min(bandpass)) & (wl <= max(bandpass))
        wl = wl[mask]
        fl = fl[mask]

        if values_3d is None:
            freqs = c / wl.to(u.m).value
            values_3d = np.empty((len(teffs), len(loggs), len(freqs)))
        else:
            assert(len(wl) == len(freqs))

        # Get the flux F(nu) and then into SI units and write to the appropriate z axis
        fl_nu = 3.34e-19 * wl.value**2 * fl.value * u.erg / (u.cm**2 * u.s * u.Hz)
        fl_nu = fl_nu.to(u.W / (u.m**2 * u.Hz)).value
        for flux_ix, f in enumerate(fl_nu):
            values_3d[teff_ix, logg_ix, flux_ix] = f
        print(f"...stored {len(fl_nu)} fluxes against Teff={teff}, logg={logg}")

# Save the two items for use with scipy interpn
# To use
# ```Python
# arr = np.load(filename)
# points = (arr['x'], arr['y'], arr['z'])
# V = arr['V']
# xi = [[teff, logg, f] for f in plot_freq]
# interpn(points, V, xi, method="linear")
# ```
save_file = Path("libs/data/newera/PHOENIX-NewEra-for-interp3d.npy.npz")
np.savez_compressed(save_file, allow_pickle=True, overwrite=True,
                    teffs=np.array(teffs), loggs=np.array(loggs), freqs=freqs, V=values_3d)
print(f"Saved interpn data to {save_file} as point arrays teffs, loggs & freqs and values array V")
