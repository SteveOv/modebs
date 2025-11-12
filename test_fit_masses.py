
import numpy as np
from uncertainties import ufloat
from uncertainties.unumpy import uarray

from libs.fit_masses import minimize_fit, mcmc_fit


# TIC 63579446
target = "TIC 63579446"
obs_radii = uarray([1.770, 1.069], [0.065, 0.039])
obs_teffs = uarray([6039.882, 6022.292], [118.545, 118.929])
sys_mass = ufloat(2.005, 0.156)
theta0 = np.array([1.250, 0.755, 9.0])

# # IQ Per (MA=3.516+/-0.050, MB=1.738+/-0.023, log_age=7.9, age=80e6, Z=0.017)
# target = "IQ Per"
# obs_radii = uarray([2.78, 1.50], [0.02, 0.02])
# obs_teffs = uarray([12000, 7700], [200, 150])
# sys_mass = ufloat(5.25, 0.06)
# theta0 = np.array([3.4, 1.8, 8.0])

# # CW Eri (MA=1.568+/-0.016, MB=1.314+/-0.010, log_age=9.23, age=1.7e9, Z=0.017)
# target = "CW Eri"
# obs_radii = uarray([2.105, 1.481], [0.007, 0.005])
# obs_teffs = uarray([6900, 6500], [100, 100])
# sys_mass = ufloat(2.882, 0.019)
# theta0 = np.array([1.6, 1.3, 9.0])

# # ZZ UMa (MA=1.135+/-0.009, MB=0.965+/-0.005, log_age=9.74, age=5.5e9, Z=0.02)
# target = "ZZ UMa"
# obs_radii = uarray([1.44, 1.08], [0.05, 0.05])
# obs_teffs = uarray([6000, 5300], [200, 200])
# sys_mass = ufloat(2.1, 0.1)
# theta0 = np.array([1.2, 0.8, 9.0])

# # AI Phe (MA=1.1938+/-0.0008, MB=1.2438+/-0.0008, age=?)
# target = "AI Phe"
# obs_radii = uarray([1.81, 2.93], [0.07, 0.10])
# obs_teffs = uarray([6250, 5100], [200, 200])
# sys_mass = ufloat(2.44, 0.10)
# theta0 = np.array([1.0, 1.4, 9.0])

def print_mass_theta(theta, label: str="theta"):
    print(f"{label} = [" + ", ".join(f"{t:.3e}" for t in theta) + "]\n")

print_mass_theta(theta0, f"theta0 ({target})")

theta_min_fit, _ = minimize_fit(theta0, sys_mass, obs_radii, obs_teffs, verbose=True)
print_mass_theta(theta_min_fit, f"theta_min_fit ({target})")

thin_by = 10
theta_mcmc_fit, _ = mcmc_fit(theta_min_fit, sys_mass, obs_radii, obs_teffs,
                             nwalkers=100, nsteps=100000, thin_by=thin_by, seed=42,
                             early_stopping=True, early_stopping_threshold=0.05, processes=8,
                             progress=True, verbose=True)
print_mass_theta(theta_mcmc_fit, f"theta_mcmc_fit ({target})")
