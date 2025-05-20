import numpy as np
import io
import sys
import glob


# example to read the very first spectrum from the GAIA format 
# files of the NewEra grid 

#filename = 'PHOENIX-NewEra-JWST-SPECTRA.Z-0.0.txt'
filename = './libs/new_era/PHOENIX-NewEra-LowRes-SPECTRA.Z-0.0.txt'

with open(filename, 'r', encoding='utf8') as f:

    for _ in range(1000 + 1):
        zeile = f.readline()

    # Read row 1 of 2: contains all data bar the fluxes
    helper = np.loadtxt(io.StringIO(zeile),dtype=('S41'),unpack=True)
    res = float(helper[7]) * 1e-3               # resolution in micron
    nwl = int(helper[8])                        # number of wavelength points
    wlstart = float(helper[9]) * 1e-3           # starting wavelength in micron
    wlend = float(helper[10]) * 1e-3            # final  wavelength in micron
    wl_steps = float(helper[11]) * 1e-3         # wavelength steps in micron

    teff = float(helper[12])
    logg  = float(helper[13])
    vturb  = float(helper[14])
    Y  = float(helper[16])
    Z  = float(helper[17])

    feref = float(helper[18])
    alpha_scale = float(helper[19])

    # cAbund = float(helper[20])
    # nAbund = float(helper[21])
    # oAbund = float(helper[22])
    # mgAbund = float(helper[23])
    # siAbund = float(helper[24])
    # caAbund = float(helper[25])
    # feAbund = float(helper[26])

    mass = float(helper[27])


    wl = np.linspace(wlstart, wlend, num=int((wlend-wlstart)/res)+1) * 1e-3 # to micron
    if nwl != wl.shape[0]:
        print(nwl, wl.shape[0])

    #fl = np.array(nwl)
    # Read row 2 of 2: contains the fluxes (lambda is implied by position)
    zeile = f.readline()
    fl = np.loadtxt(io.StringIO(zeile), unpack=True)
