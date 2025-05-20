#
# example to read the spectrum 
# and the namelist for a HSR H5 file
# 
# version 2.0, PHH, 22/Jan/2025
# version 1.0, PHH, 22/Apr/2024
#
# requires h5py and f90nml modules
#
import sys
import io
import re
import numpy as np
import h5py
import f90nml

#######################################################################

elmass = np.array([0.0e0, 1.0079, 4.0026, 6.941, 9.0122, 10.811, 12.0107, 14.0067, 15.9994, 18.9984, 20.1797, 22.9897, 24.305, 26.9815,
  28.0855, 30.9738, 32.065, 35.453, 39.948, 39.0983, 40.078, 44.9559, 47.867, 50.9415, 51.9961, 54.938, 55.845, 58.9332, 58.6934, 63.546,
  65.39,  69.723, 72.64,  74.9216, 78.96,  79.904, 83.8,   85.4678, 87.62,  88.9059, 91.224, 92.9064, 95.94,  98,     101.07,  102.9055,
 106.42,  107.8682, 112.411, 114.818, 118.71,  121.76,  127.6,   126.9045, 131.293, 132.9055, 137.327, 138.9055, 140.116, 140.9077, 144.24,
 145,     150.36,  151.964, 157.25,  158.9253, 162.5,   164.9303, 167.259, 168.9342, 173.04,  174.967 , 178.49,  180.9479, 183.84,  186.207,
 190.23,  192.217, 195.078, 196.9665, 200.59,  204.3833, 207.2,   208.9804, 209,     210,     222,     223,     226,     227,     232.0381,
 231.0359, 238.0289, 237,     244,     243,     247,     247,     251,     252,     257,     258,     259,     262,     261,     262,     266,
 264,     277,     268,     275,     272])


elsymbol = np.array(['H ', 'He', 'Li', 'Be', 'B ', 'C ', 'N ', 'O ', 'F ', 'Ne', 'Na', 'Mg', 'Al', 'Si', 'P ', 'S ', 'Cl', 'Ar',
 'K ', 'Ca', 'Sc', 'Ti', 'V,', 'Cr', 'Mn', 'Fe', 'Co', 'Ni', 'Cu', 'Zn', 'Ga', 'Ge', 'As', 'Se', 'Br', 'Kr', 'Rb', 'Sr', 'Y ', 'Zr', 'Nb',
 'Mo', 'Tc', 'Ru', 'Rh', 'Pd', 'Ag', 'Cd', 'In', 'Sn', 'Sb', 'Te', 'I ', 'Xe', 'Cs', 'Ba', 'La', 'Ce', 'Pr', 'Nd', 'Pm', 'Sm', 'Eu', 'Gd',
 'Tb', 'Dy', 'Ho', 'Er', 'Tm', 'Yb', 'Lu', 'Hf', 'Ta', ' W', 'Re', 'Os', 'Ir', 'Pt', 'Au', 'Hg', 'Tl', 'Pb', 'Bi', 'Po', 'At', 'Rn', 'Fr',
 'Ra', 'Ac', 'Th', 'Pa', ' U', 'Np', 'Pu', 'Am', 'Cm', 'Bk', 'Cf', 'Es', 'Fm', 'Md', 'No', 'Lr', 'Rf', 'Db', 'Sg', 'Bh', 'Hs',
 'Mt', 'Ds', 'Rg'])

#######################################################################

def compute_massfrc(input_nml):
  """ get data from namelist and return number and mass fractions
  input_nml: namelist to process
  """
  
  teff = input_nml['phoenix']['teff']
  logg = input_nml['phoenix']['logg']
  zscale = input_nml['phoenix']['zscale']
  yscale = input_nml['phoenix']['yscale']
  alpha_scale = input_nml['phoenix']['alpha_scale']

  eheu   = np.array(input_nml['phoenix']['eheu'])
  nome   = input_nml['phoenix']['nome']
  nelem = input_nml['phoenix']['nelem']

  numbfrac = np.array(eheu) # working array for normalization etc.

  #print(teff,logg,zscale,yscale,alpha_scale,nelem)
# apply scaling:
  for i in range(nelem):
    if(nome[i] == 200): numbfrac[i] += yscale
    if(nome[i] > 200): numbfrac[i] += zscale
    # apply the alpha-element scale factor:
    if(nome[i] == 800) : numbfrac[i] += alpha_scale
    if(nome[i] == 1000): numbfrac[i] += alpha_scale
    if(nome[i] == 1200): numbfrac[i] += alpha_scale
    if(nome[i] == 1400): numbfrac[i] += alpha_scale
    if(nome[i] == 1600): numbfrac[i] += alpha_scale
    if(nome[i] == 1800): numbfrac[i] += alpha_scale
    if(nome[i] == 2000): numbfrac[i] += alpha_scale
    if(nome[i] == 2200): numbfrac[i] += alpha_scale

  
# compute normalization:
  su = 0.0
  numbfrac[:] = 10.**numbfrac[:]
  su = np.sum(numbfrac)
  numbfrac[:] = numbfrac[:]/su

# compute mass fractions:
  su = 0.0
  for i in range(nelem):
    z = np.int32(nome[i]/100)
    su += elmass[z]*numbfrac[i]
  su = 1./su

  massfrac = np.array(numbfrac)
  for i in range(nelem):
    z = np.int32(nome[i]/100)
    massfrac[i] = elmass[z]*numbfrac[i]*su 
    if(z == 1): X = massfrac[i]
    if(z == 2): Y = massfrac[i]
    # print(i,z,numbfrac[i],massfrac[i])

  Z = 1.0-X-Y
  # print(X,Y,Z)
  return numbfrac,massfrac,X,Y,Z
#######################################################################

def rd20list(zeilen):
   """ read a PHOENIX/1D restart file

   zeilen: string list with the restart file content
   returns structure data as a dictionary!
   """
#
# now we parse the header:
#
   iterations = int((zeilen.pop(0)).split()[4])
   layer = int((zeilen.pop(0)).split()[0])
   zeile = zeilen.pop(0)
   zeile =  zeile.replace("D","E")
   teff,rtau1,vtau1,pout,n,modtyp,identyp,vfold,rout = np.loadtxt(io.StringIO(zeile),unpack=True)
#
# header parsed, compute number of lines per [layer] array:
#
   lines_per_array = layer//3
   if(layer % 3): lines_per_array+=1
#
# read tstd:
#
   zeile = " ".join(zeilen[0:lines_per_array])
   zeile = zeile.replace("D","E")
   zeile = zeile.replace("\n"," ")
   tstd = np.loadtxt(io.StringIO(zeile),ndmin=1)
   zeilen = zeilen[lines_per_array:]
#
# read telec:
#
   zeile = " ".join(zeilen[0:lines_per_array])
   zeile = zeile.replace("D","E")
   zeile = zeile.replace("\n"," ")
   telec = np.loadtxt(io.StringIO(zeile),ndmin=1)
   zeilen = zeilen[lines_per_array:]
#
# create a dictorary with the data so far.
# the string is used as key
#
   structure = {'layer':layer,'iterations':iterations,
   'teff':teff,'rtau1':rtau1,'vtau1':vtau1,'pout':pout,'n':n,'modtyp':modtyp,'identyp':identyp,'vfold':vfold,'rout':rout}
   structure['telec'] = telec
   structure['tstd'] = tstd
#
# now the format is fixed until 'laywind' is found:
# we can loop and fill the dictonary with more data
# this is real easy ...
#
   while True:
       key = (zeilen.pop(0)).split()[0]
       if(key == 'laywind'): break
       zeile = "".join(zeilen[0:lines_per_array])
       zeile = zeile.replace("D","E")
       zeile = zeile.replace("\n"," ")
       zeile = re.sub(r'([0-9])-([0-9])',r'\1 -\2',zeile)  # fixes packed negative values
       zeile = re.sub(r'([0-9])\+([0-9])',r'\1E+\2',zeile) # fixes large exponentials
       data = np.loadtxt(io.StringIO(zeile),ndmin=1)
       zeilen = zeilen[lines_per_array:]
       structure[key] = data
#
# check for "end-of-data"
#
   zeile = zeilen.pop(0)  # a '0'
   zeile = zeilen.pop(0)  # end-of-data
   if(len(zeilen) < 2): return structure         # no departure coeffs data
#
# parse the bi header: check if it is there at all, first
#
   zeile = zeilen.pop(0)  # 'START departure coefficients version 1.0'
   if(zeile.find('PHOENIX') != -1 or zeile.find('phoenix') != -1): return structure         # no departure coeffs data
   layer_check = int((zeilen.pop(0)).split()[0])
   if(layer != layer_check): raise Exception('layer .ne. layer_check!')
   nlevel = int((zeilen.pop(0)).split()[0])
#
# create the arrays with the info for each level and the bi data for each layer and level:
#
   z_ion_code = np.ndarray(nlevel,dtype=np.int)
   level_number = np.zeros(nlevel,dtype=np.int)
   bi = np.zeros((layer,nlevel))
#
# now read and process the data
# this can be done with a for loop
#
   for i in range(nlevel):
       helper = (zeilen.pop(0)).split()
       z_ion_code[i] = helper[0]
       level_number[i] = helper[1]
       zeile = "".join(zeilen[0:lines_per_array])
       zeile = zeile.replace("D","E")
       zeile = zeile.replace("\n"," ")
       zeile = re.sub(r'([0-9])-([0-9])',r'\1 -\2',zeile)  # fixes packed negative values
       zeile = re.sub(r'([0-9])\+([0-9])',r'\1E+\2',zeile) # fixes large exponentials
       bi[:,i] = np.loadtxt(io.StringIO(zeile),ndmin=1)
       zeilen = zeilen[lines_per_array:]
#
# transfer the data to the structure dictonary:
#
   structure['nlevel'] = nlevel
   structure['z_ion_code'] = z_ion_code
   structure['level_number'] = level_number
   structure['bi'] = bi

   return structure
#######################################################################

filename = "libs/data/newera/lte05000-5.00-0.0.PHOENIX-NewEra-ACES-COND-2023.HSR.h5"
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

# this is the restart file (as a string):
restart_str = (str(fh5['/PHOENIX_RESTART/phx_restart'][()].tobytes()))[2:-1]


# read namelist:
nml_str = (str(fh5['/PHOENIX_NAMELIST/phoenix_nml'][()].tobytes()))[2:-1]

fh5.close()

# wrestle the string into something more useful (each line is 1024 byte long):

zeilen = [restart_str[x:x+1024] for x in range(0,len(restart_str),1024)]


# convert to a namelist object:
target_nml = f90nml.reads(nml_str)

# now extract and print some values:
teff = target_nml['phoenix']['teff']
logg = target_nml['phoenix']['logg']
zscale = target_nml['phoenix']['zscale']
alpha_scale = target_nml['phoenix']['alpha_scale']
m_sun  = target_nml['phoenix']['m_sun']
wltau  = target_nml['phoenix']['wltau']
mixlng  = target_nml['phoenix']['mixlng']
print(filename,':',teff,logg,m_sun,wltau,mixlng,zscale,alpha_scale)

#
# get the abundances from the namelist and convert them to number
# and mass fractions
#
numfrac,massfrac,X,Y,Z = compute_massfrc(target_nml)


eheu   = np.array(target_nml['phoenix']['eheu'])
nome   = target_nml['phoenix']['nome']
nelem  = target_nml['phoenix']['nelem']

for i in range(nelem):
 print(nome[i],elsymbol[i],("{:12.2e}"*2).format(numfrac[i],massfrac[i]))

# extract the structure data
model_structure = rd20list(zeilen)

# here are the user-relevant data:
# --------------------------------
n_layer = model_structure['layer']

tau_std = model_structure['tstd']    # this is the (input) tau(standard) grid
Pgas = model_structure['pgas']       # gas pressure (cgs)
rho = model_structure['rho']         # densities (cgs)
Pelec = model_structure['pe']        # electron pressures (cgs)
Telec = model_structure['telec']     # electron temperatures (cgs)
radius = model_structure['radius']   # radii (cgs)

Rout = radius[0]                     # outermost radius (cgs)
Rin = radius[-1]                     # innermost radius (cgs)
