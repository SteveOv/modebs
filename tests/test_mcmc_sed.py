""" Unit tests for the MistIsochrones class. """
# pylint: disable=unused-import, too-many-public-methods, line-too-long, invalid-name, no-member
import unittest

import numpy as np
import astropy.units as u

from scipy.optimize import minimize, show_options

from libs.pipeline import get_sed_for_target
from libs.mcmc_sed import norm_blackbody_model, min_max_normalize


class Testmcmcsed(unittest.TestCase):
    """ Unit tests for the mcmc_sed module. """

    @unittest.skip("not really a test as such - more speculative behaviour")
    def test_scipy_optimize_norm_bb_model(self):
        """ Try scipy.minimize to see if it's an option for initial estimates. """
        # pylint: disable=too-many-locals, unused-variable
        # Ratios from LC fitting
        target = "CM Dra"
        k = 0.945                       # Fitted
        J = 0.98133                     # Fitted
        Teff_tess = 3214.00             # TESS metadata

        # Get some SED data (overlook extinction)
        sed = get_sed_for_target(target, f"V* {target}", 0.1)
        mask = (sed["sed_wl"] > 0.4*u.um) & (sed["sed_wl"] < 20*u.um) & (sed["sed_filter"] != "HIP:hp")
        sed_freq = sed[mask]["sed_freq"].to(u.Hz).value
        sed_norm_flux, sed_norm_eflux = min_max_normalize(sed[mask]["sed_flux"].value, sed[mask]["sed_eflux"].value * 5)

        def similarity_func(theta, x, y, y_err):
            (Teff1, logg1, logg2) = theta
            y_model = norm_blackbody_model(x, Teff1, Teff1 * J, logg1, logg2)
            return 0.5 * np.sum(((y - y_model) / y_err)**2)

        x0 = np.array([Teff_tess, 4, 4]) # Teff1|A, logg1|A, logg2|A
        method="Nelder-Mead"
        soln = minimize(similarity_func, x0, args=(sed_freq, sed_norm_flux, sed_norm_eflux),
                        method=method, options={"disp": False})

        (TeffA, loggA, loggB), TeffB = soln.x, soln.x[0] * J
        print()
        #show_options("minimize", method, disp=True)
        print(soln)
        print(f"TeffA={TeffA:.0f} K & loggA={loggA:.3f} dex; TeffB={TeffB:.0f} K & loggB={loggB:.3f}")

if __name__ == "__main__":
    unittest.main()
