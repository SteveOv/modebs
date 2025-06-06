""" Unit tests for the MistIsochrones class. """
# pylint: disable=unused-import, too-many-public-methods, line-too-long, invalid-name, no-member
import unittest

import numpy as np
import astropy.units as u

from scipy.optimize import minimize
from scipy.interpolate import RegularGridInterpolator

from deblib.constants import c, h, k_B
from deblib.vmath import exp

from libs.pipeline import get_sed_for_target
from libs.mcmc_sed import min_max_normalize


class Testmcmcsed(unittest.TestCase):
    """ Unit tests for the mcmc_sed module. """

    #@unittest.skip("not really a test as such - more speculative behaviour")
    def test_scipy_optimize_norm_bb_model(self):
        """ Try scipy.minimize to see if it's an option for initial estimates. """
        # pylint: disable=too-many-locals, unused-variable
        target = "CW Eri"
        J = 0.93            # From LC fit
        Teff_tess = 6900    # From TESS metadata (6861) rounded to 2 s.f.

        # Set up the NewEra model function
        spec = np.load("libs/data/newera/PHOENIX-NewEra-for-interp3d.npy.npz", allow_pickle=True)
        ne_wl_range = c * 1e6 / (spec["freqs"].max(), spec["freqs"].min()) * u.um
        interp = RegularGridInterpolator(points=(spec["teffs"], spec["loggs"], spec["freqs"]), values=spec["V"])

        def newera_model(x, tA, tB, lA, lB):
            return min_max_normalize(np.add(interp(np.array([[tA, lA, freq] for freq in x]), method="linear"),
                                            interp(np.array([[tB, lB, freq] for freq in x]), method="linear")))

        def bb_model(x, tA, tB, lA, lB):
            # pylint: disable=unused-argument
            def bb_spectrum(freqs, teff):
                return ((2 * h * freqs**3) / c**2) / (exp((h * freqs) / (k_B * teff)) - 1)
            return min_max_normalize(np.add(bb_spectrum(x, tA), bb_spectrum(x, tB)))

        # Get target's SED data (overlook extinction). Restrict bandpass to within the NewEra data
        sed = get_sed_for_target(target, f"V* {target}", 0.1)
        mask = (sed["sed_wl"] > min(ne_wl_range)) & (sed["sed_wl"] < max(ne_wl_range)) \
            & ([not f.startswith("HIP:") for f in sed["sed_filter"]]) & (sed["sed_filter"] != "Cousins:I")
        sed_freq = sed[mask]["sed_freq"].to(u.Hz).value
        sed_norm_flux, sed_norm_eflux = min_max_normalize(sed[mask]["sed_flux"].value, sed[mask]["sed_eflux"].value * 5)

        for msg, model_func in [("Blackbody", bb_model), ("NewEra", newera_model)]:
            for method in ["Nelder-Mead", "SLSQP"]:
                soln = minimize(Testmcmcsed._similarity_func,
                                x0=np.array([Teff_tess, 5, 5]), # TeffA, loggA, loggB
                                args=(sed_freq, sed_norm_flux, sed_norm_eflux, J, model_func),
                                method=method, options={"disp": True})
                (TeffA, loggA, loggB), TeffB = soln.x, soln.x[0] * J
                print(f"\nThe {msg} model minimized with the {method} method yielded the following for {target}\n{soln}")
                print(f"TeffA={TeffA:.0f} K & loggA={loggA:.3f} dex; TeffB={TeffB:.0f} K & loggB={loggB:.3f}")


    @classmethod
    def _similarity_func(cls, theta, x, y, y_err, J, model_func):
        """ Simple function for comparing model SEDs with obs when called from scipy.minimize"""
        # pylint: disable=too-many-arguments, too-many-positional-arguments
        (tA, lA, lB), tB = theta, theta[0] * J
        if 2300 < tA < 12000 and 2300 < tB < 12000 and 0.5 < lA < 6.0 and 0.5 < lB < 6.0: # priors
            y_model = model_func(x, tA, tB, lA, lB)
            return 0.5 * np.sum(((y - y_model) / y_err)**2)
        return np.inf

if __name__ == "__main__":
    unittest.main()
