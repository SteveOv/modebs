#
# example to read the spectrum 
# and the namelist for a HSR H5 file
# 
# version 2.0, PHH, 22/Jan/2025
# version 1.0, PHH, 22/Apr/2024
#
# requires h5py and f90nml modules
#
from pathlib import Path
import sys
import h5py
import f90nml

filename = "./libs/data/newera/lte05000-5.00-0.0.PHOENIX-NewEra-ACES-COND-2023.HSR.h5"
# try:
#   filename = sys.argv[1]
# except:
#   print(f"Usage: {sys.argv[0]} filename")
#   status = 1
#   raise SystemExit(status)

print('reading file: <'+filename+'>')
try:
  fh5 = h5py.File(filename,'r')
except FileNotFoundError:
  print(f'File {filename} does not exist')
  status = 1
  raise SystemExit(status)

# read the HSR spectrum: 
# wl: wavelength in Angstroem, vacuum
# fl: flux, the file stores log10()
# bb: Planck function at Teff, the file stores log10()

wl = fh5['/PHOENIX_SPECTRUM/wl'][()]
fl = 10.**fh5['/PHOENIX_SPECTRUM/flux'][()]
bb = 10.**fh5['/PHOENIX_SPECTRUM/bb'][()]

# read the LSR spectrum: 
# wl_lsr: wavelength in Angstroem, vacuum
# fl_lsr: flux, the file stores log10()

wl_lsr = fh5['/PHOENIX_SPECTRUM_LSR/wl'][()] / 1e4
fl_lsr = 10.**fh5['/PHOENIX_SPECTRUM_LSR/fl'][()] / 1e4

print(wl_lsr[:10])
print(fl_lsr[:10])

# read namelist:
nml_str = (str(fh5['/PHOENIX_NAMELIST/phoenix_nml'][()].tobytes()))[2:-1]

# convert to a namelist object:
target_nml = f90nml.reads(nml_str)


# now extract and print some values:
teff = target_nml['phoenix']['teff']
r0 = target_nml['phoenix']['r0']
v0 = target_nml['phoenix']['v0']
logg = target_nml['phoenix']['logg']
zscale = target_nml['phoenix']['zscale']
alpha_scale = target_nml['phoenix']['alpha_scale']
m_sun  = target_nml['phoenix']['m_sun']
wltau  = target_nml['phoenix']['wltau']
ngrrad  = target_nml['phoenix']['ngrrad']
ieos  = target_nml['phoenix']['ieos']
mixlng  = target_nml['phoenix']['mixlng']
print(filename, ':', teff, logg, m_sun, wltau, mixlng, zscale, alpha_scale)

# this is the stdout file (as a string):
stdout_str = (str(fh5['/PHOENIX_STDOUT/phx_stdout'][()].tobytes()))[2:-1]

# this is the restart file (as a string):
restart_str = (str(fh5['/PHOENIX_RESTART/phx_restart'][()].tobytes()))[2:-1]

fh5.close()

