""" Models to derive masses from known sys_mass, radii & teffs and MIST models. """
from pathlib import Path
from inspect import getsourcefile

import numpy as np

from scipy.interpolate import RBFInterpolator

from .data.mist.read_mist_models import ISO

MIN_PHASE = 0 # MS
MAX_PHASE = 2 # RGB

_this_dir = Path(getsourcefile(lambda:0)).parent
ISO_FILE = _this_dir / "data/mist/MIST_v1.2_vvcrit0.4_basic_isos" \
                                                / "MIST_v1.2_feh_p0.00_afe_p0.0_vvcrit0.4_basic.iso"
iso = ISO(f"{ISO_FILE}", verbose=True)

# Build up the known datapoints and corresponding radius & teff values.
# Get the linear values for these so that we can perform interpolation in linear space.
ages_list = []
eep_list = []
masses_list = []
radii_list = []
teffs_list = []
logg_list = []
for log_age in sorted(iso.ages):
    iso_block = iso.isos[iso.age_index(log_age)]
    iso_block = iso_block[(iso_block["phase"] >= MIN_PHASE) & (iso_block["phase"] <= MAX_PHASE)]
    if (new_rows := len(iso_block)) > 0:
        mass_sort = np.argsort(iso_block["star_mass"])

        # Points/axes
        ages_list += [10**log_age] * new_rows
        eep_list += list(iso_block[mass_sort]["EEP"])
        masses_list += list(iso_block[mass_sort]["star_mass"])

        # corresponding values
        radii_list += list(10**iso_block[mass_sort]["log_R"])
        teffs_list += list(10**iso_block[mass_sort]["log_Teff"])
        logg_list += list(iso_block[mass_sort]["log_g"])

# Create the interpolators for radius and teff; using RBF interpolation as we have irregular data.
x = np.array(list(zip(ages_list, masses_list)), dtype=float)
neighbours = 4**x.ndim # limit RBF mem usage; otherwise scales as ~points^2
radius_interp = RBFInterpolator(x, radii_list, neighbours, smoothing=5, kernel="linear")
teff_interp = RBFInterpolator(x, teffs_list, neighbours, smoothing=5, kernel="linear")
logg_interp = RBFInterpolator(x, logg_list, neighbours, smoothing=5, kernel="linear")

x = np.array(list(zip(eep_list, masses_list)), dtype=float)
age_interp = RBFInterpolator(x, ages_list, neighbours, smoothing=5, kernel="linear")

# Priors based on the data
age_limits = (min(ages_list), max(ages_list))
eep_limits = (min(eep_list), max(eep_list))
mass_limits = (min(masses_list), max(masses_list))

del x, ages_list, masses_list, radii_list, teffs_list, eep_list, iso

def get_age_limits():
    """ Get the lower and upper bounds of the ages within the model. """
    return age_limits

def get_mass_limits():
    """ Get the lower and upper bounds of the masses within the model. """
    return mass_limits

def get_eep_limits():
    """ Get the lower and upper bounds of the EEPs within the model. """
    return eep_limits

def log_age_for_mass_and_eep(mass: float, eep: int=353) -> float:
    """
    An approximate log10(age) for the requested mass and Equivalent Evolutionary Point (EEP).
    Within the same phases range as the interpolators used for radii & masses for the model func.

    Known "primary" EEPs are:
    202 - ZAMS (Zero Age M-S)
    353 - IAMS (Intermediate Age M-S)
    454 - TAMS (Terminal Age M-S)

    :mass: the requested mass (solMass)
    :eep: the equivalent evolutionay point (EEP)
    :returns: a log(age) for the requested EEP and star mass
    """
    return np.log10(age_interp([(eep, mass)])[0])

def model_func(masses: np.ndarray[float], log_age: float):
    """
    For each of the passed masses & eeps will model the radius, Teff and logg.

    :masses: the stellar masses
    :log_age: the common log10 age of the stars
    :returns: the model stars' [radii, Teffs, loggs]
    """
    xi = np.array([(10**log_age, m) for m in masses])
    return np.concatenate([radius_interp(xi), teff_interp(xi), logg_interp(xi)])
