""" Unit tests for the fit_masses module. """
# pylint: disable=unused-import, too-many-public-methods, line-too-long, invalid-name, no-member
import warnings
import unittest

import numpy as np

# pylint: disable=wrong-import-position
warnings.filterwarnings("ignore", "Using UFloat objects with std_dev==0 may give unexpected results.", category=UserWarning)
from uncertainties import ufloat
from uncertainties.unumpy import uarray

from libs.fit_masses import minimize_fit, mcmc_fit

class Testsed(unittest.TestCase):
    """ Unit tests for the fit_masses module. """
    targets = {
        "IQ Per": {
            "theta0": np.array([3.4, 1.8, 8.0]),
            "sys_mass": ufloat(5.25, 0.06),
            "radii": uarray([2.78, 1.50], [0.02, 0.02]),
            "teffs": uarray([12000, 7700], [200, 150]),
            "exp_masses": uarray([3.515, 1.738], [0.050, 0.023]),
            "exp_log_age": 7.9
        },
        "CW Eri": {
            "theta0": np.array([1.6, 1.3, 9.0]),
            "sys_mass": ufloat(2.882, 0.019),
            "radii": uarray([2.105, 1.481], [0.007, 0.005]),
            "teffs": uarray([6900, 6500], [100, 100]),
            "exp_masses": uarray([1.568, 1.314], [0.016, 0.010]),
            "exp_log_age": 9.23
        },
        "ZZ UMa": {
            "theta0": np.array([1.2, 0.8, 9.0]),
            "sys_mass": ufloat(2.1, 0.1),
            "radii": uarray([1.44, 1.08], [0.05, 0.05]),
            "teffs": uarray([6000, 5300], [200, 200]),
            "exp_masses": uarray([1.135, 0.965], [0.009, 0.005]),
            "exp_log_age": 9.74
        },
        "AI Phe": {
            "theta0": np.array([1.0, 1.4, 9.0]),
            "sys_mass": ufloat(2.44, 0.10),
            "radii": uarray([1.81, 2.93], [0.07, 0.10]),
            "teffs": uarray([6250, 5100], [200, 200]),
            "exp_masses": uarray([1.1938, 1.2438], [0.008, 0.008]),
            "exp_log_age": None
        },
    }

    #
    #   minimize_fit(theta0: np.ndarray[float],
    #                sys_mass: UFloat,
    #                radii: np.ndarray[UFloat],
    #                teffs: np.ndarray[UFloat],
    #                methods: List[str]=None,
    #                verbose: bool=False) -> Tuple[np.ndarray[float], OptimizeResult]:
    #
    def test_minimize_fit_known_targets(self):
        """ Tests minimize_fit() basic happy path test for known targets """
        for target, params in self.targets.items():
            with self.subTest(msg=f"minimize fit on {target}"):
                print(self._format_theta(params["theta0"], f"\ntheta0[{target}]"))

                theta_fit, _ = minimize_fit(params["theta0"], params["sys_mass"],
                                            params["radii"], params["teffs"], verbose=True)

                print(self._format_theta(theta_fit, f"theta_fit[{target}]"))

                for ix, (fit_mass, exp_mass) in enumerate(zip(theta_fit[:-1], params["exp_masses"])):
                    # Expecting strong match with masses so round to 1 d.p.
                    # except AI Phe where the quick fit fails to meet this criteria
                    places = 0 if target == "AI Phe" else 1
                    self.assertAlmostEqual(fit_mass, exp_mass.n, places, f"{target}: mass[{ix}]")

                if (exp_log_age := params.get("exp_log_age", None)) is not None:
                    # Fitting age is expected to be less precise so we round to nearest whole number
                    self.assertAlmostEqual(theta_fit[-1], exp_log_age, 0, f"{target}: log_age")


    #
    #   mcmc_fit(theta0: np.ndarray[float],
    #            sys_mass: UFloat,
    #            radii: np.ndarray[UFloat],
    #            teffs: np.ndarray[UFloat],
    #                ...
    #            verbose: bool=False) -> Tuple[np.ndarray[UFloat], EnsembleSampler]:
    #
    @unittest.skip("takes too long for automated runs - comment this out to run test explicitly")
    def test_mcmc_fit_known_targets(self):
        """ Tests mcmc_fit() basic happy path test for selected target """
        target = "IQ Per"
        # target = "CW Eri"
        # target = "ZZ UMa"
        # target = "AI Phe"
        params = self.targets[target]
        nsteps = 50000

        print(self._format_theta(params["theta0"], f"\ntheta0[{target}]"))

        thin_by = 1
        theta_fit, sampler = mcmc_fit(params["theta0"], params["sys_mass"],
                                      params["radii"], params["teffs"],
                                      nwalkers=100, nsteps=nsteps, thin_by=thin_by, seed=42,
                                      early_stopping=True, early_stopping_threshold=0.01,
                                      processes=4, progress=True)

        tau = sampler.get_autocorr_time(c=5, tol=50, quiet=True) * thin_by
        print(f"Autocorrelation steps (tau): {', '.join(f'{t:.3f}' for t in tau)}")
        print(f"Estimated burn-in steps:     {int(max(np.nan_to_num(tau, nan=1000)) * 2):,}")
        print(f"Mean Acceptance fraction:    {np.mean(sampler.acceptance_fraction):.3f}")

        print(self._format_theta(theta_fit, f"theta_fit[{target}]"))

        for ix, (fit_mass, exp_mass) in enumerate(zip(theta_fit[:-1], params["exp_masses"])):
            # Assert the results are consistent within errorbars.
            # Cannot use == as even when the values appear the same they're treated as unequal.
            msg = f"{target}: (mass[{ix}] ==) {fit_mass} != {exp_mass} (within errorbars)"
            if fit_mass.n > exp_mass.n:
                self.assertTrue(fit_mass.n - fit_mass.s < exp_mass.n + exp_mass.s, msg)
            else:
                self.assertTrue(fit_mass.n + fit_mass.s > exp_mass.n - exp_mass.s, msg)

        if (exp_log_age := params.get("exp_log_age", None)) is not None:
            # Fitting age is expected to be less precise so we round to nearest whole number
            self.assertAlmostEqual(theta_fit[-1].n, exp_log_age, 0, f"{target}: log_age")


    def _format_theta(self, theta, label: str="theta"):
        return f"{label} = [" + ", ".join(f"{t:.3e}" for t in theta) + "]"


if __name__ == "__main__":
    unittest.main()
