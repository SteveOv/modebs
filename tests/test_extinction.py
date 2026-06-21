""" Unit tests for the extinction module. """
# pylint: disable=unused-import, line-too-long, invalid-name, no-member
import unittest

import numpy as np
import astropy.units as u
from astropy.coordinates import SkyCoord, CartesianRepresentation

from libs.extinction import iterate, get_bayestar_ebv, get_edenhofer2023_av, get_gontcharov_av

class Testextinction(unittest.TestCase):
    """ Unit tests for the extinction module. """
    targets = {
        # Ooop North! Gontcharov happy but an unreliable extinction from Bayestar
        "UZ Dra": {
            "coords": SkyCoord(291.47947545 * u.deg, 68.93546881 * u.deg, 185.38698966 * u.pc, frame="icrs"),
            # A_V values
            "gontcharov": (0.219, True),
            "bayestar": (0, False),
            "edenhofer": (0.029, True),
        },
        # Covered by both Gontcharov and Bayestar
        "IT Cas": {
            "coords": SkyCoord(355.50569743 * u.deg, 51.74352579 * u.deg, 514.95778083 * u.pc, frame="icrs"),
            # A_V values
            "gontcharov": (0.385, True),
            "bayestar": (0.356, True),
            "edenhofer": (0.167, True),
        },
        # Way down south (in the LOPS2 field). Gontcharov still OK but outside of Bayestar coverage
        "TIC 7695666": {
            "coords": SkyCoord(65.64676092 * u.deg, -41.48319921 * u.deg, 367.56561186 * u.pc, frame="icrs"),
            # A_V values
            "gontcharov": (0.197, True),
            "bayestar": (np.nan, False),
            "edenhofer": (0.046, True), # If flavor is "main" this is 0.046 and for 2k it drops to 0.044
        },
    }

    #
    #   Tests: iterate(coords: SkyCoord, func: List[str|Callable], yield_ebv, verbose) -> Generator[(val, flag)]
    #
    def test_iterate_happy_path(self):
        """ Tests iterate() - basic happy path test """
        print()
        funcs = ["gontcharov", "bayestar", "edenhofer"]
        for target in ["UZ Dra", "IT Cas", "TIC 7695666"]:
            for yield_ebv in [False, True]:
                with self.subTest(f" iterate({target}, yield_ebv={yield_ebv}) "):
                    config = self.targets[target]
                    res = list(iterate(config["coords"], funcs, yield_ebv=yield_ebv, verbose=True))

                    # iterate() expected to filter out nan results
                    exp_results = [e for e in (config[f] for f in funcs) if not np.isnan(e[0])]
                    self.assertEqual(len(exp_results), len(res))

                    for ix, (val, reliable) in enumerate(res):
                        exp_extinction = exp_results[ix][0]
                        if yield_ebv:
                            exp_extinction /= 3.1
                        exp_reliable = exp_results[ix][1]

                        self.assertAlmostEqual(exp_extinction, val, 3)
                        self.assertEqual(exp_reliable, reliable)


    #
    #   Tests: get_bayestar_ebv(coords: SkyCoord, version: str, conversion_factor: float) -> (val, flag)
    #
    def test_get_bayestar_ebv_happy_path(self):
        """ Test get_bayestar_ebv() - simple happy path with known targets """
        for target in ["UZ Dra", "IT Cas", "TIC 7695666"]:
            with self.subTest(f" get_bayestar_ebv({target}) "):
                config = self.targets[target]
                val, reliable = get_bayestar_ebv(config["coords"])

                exp_val, exp_reliable = config["bayestar"]
                if not np.isnan(exp_val):
                    self.assertAlmostEqual(exp_val / 3.1, val, 3)
                else:
                    self.assertTrue(np.isnan(val))
                self.assertEqual(exp_reliable, reliable)



    #
    #   Tests: get_edenhofer2023_av(coords: SkyCoord, version: str) -> (val, flag)
    #
    def test_get_edenhofer2023_av_happy_path(self):
        """ Test get_edenhofer2023_av() - simple happy path with known targets """
        for target in ["UZ Dra", "IT Cas", "TIC 7695666"]:
            with self.subTest(f" get_edenhofer2023_av({target}) "):
                config = self.targets[target]
                val, reliable = get_edenhofer2023_av(config["coords"])

                exp_val, exp_reliable = config["edenhofer"]
                if not np.isnan(exp_val):
                    self.assertAlmostEqual(exp_val, val, 3)
                else:
                    self.assertTrue(np.isnan(val))
                self.assertEqual(exp_reliable, reliable)

    @unittest.skip("Combos other than the default of main/mean incur large donwload & mem use")
    def test_get_edenhofer2023_av_flavor_and_mode(self):
        """ Test get_edenhofer2023_av() flavor and mode options """
        config = self.targets["TIC 7695666"]
        for (flavor,                mode,       exp_val,        exp_reliable) in [
            ("main",                "mean",     0.046,          True),
            ("less_data_but_2kpc",  "mean",     0.044,          True),
            # Excessive memory usage from the combination of the 18 GB samples file and integration
            ("main",                "std" ,     0.000,          True),
        ]:
            with self.subTest(f" get_edenhofer2023_av(TIC 7695666, {flavor}, {mode}) "):
                val, reliable = get_edenhofer2023_av(config["coords"], flavor=flavor, mode=mode)
                self.assertAlmostEqual(exp_val, val, 3)
                self.assertEqual(exp_reliable, reliable)


    #
    #   Tests: get_gontcharov_av(coords: SkyCoord) -> (val, flag)
    #
    def test_get_gontcharov_av_happy_path(self):
        """ Test get_gontcharov_av() - simple happy path with known targets """
        for target in ["UZ Dra", "IT Cas", "TIC 7695666"]:
            with self.subTest(f" get_gontcharov_av({target}) "):
                config = self.targets[target]
                val, reliable = get_gontcharov_av(config["coords"])

                exp_val, exp_reliable = config["gontcharov"]
                if not np.isnan(exp_val):
                    self.assertAlmostEqual(exp_val, val, 3)
                else:
                    self.assertTrue(np.isnan(val))
                self.assertEqual(exp_reliable, reliable)

    def test_get_gontcharov_av_test_interpolation(self):
        """ Test get_gontcharov_av() - test the interpolation against known snippet from table """
        #
        # Table 1 of Gontcharov(2017AstL...43..472G) gives a snippet of the XYZ data
        #
        #   X       Y       Z       E(J-Ks) E(B-V)  RV      AV
        #   -1200   0       0       0.237   0.392   3.10    1.22
        #   -1180   -200    -80     0.168   0.278   3.10    0.86
        #   -1180   -200    -60     0.181   0.300   3.10    0.93
        #   -1180   -200    -40     0.181   0.300   3.10    0.93
        #   -1180   -200    -20     0.186   0.308   3.10    0.95
        #
        for x,      y,      z,      exp_val,    exp_reliable in [
            # These land on table values
            (-1200, 0,      0,      1.22,       True),
            (-1180, -200,   -80,    0.86,       True),
            (-1180, -200,   -20,    0.95,       True),
            # These will require interpolation
            (-1180, -200,   -70,    0.89,       True),
            (-1180, -200,   -30,    0.94,       True),
        ]:
            with self.subTest(f" get_goncharov_av(coords=({x}, {y}, {z}, galactic, cartesian)) "):
                # Doesn't need ICRS but will show if the target func is converting to XYZ correctly
                coords = SkyCoord(CartesianRepresentation(x, y, z, "pc"), frame="galactic").icrs

                val, reliable = get_gontcharov_av(coords)

                self.assertAlmostEqual(exp_val, val, 2)
                self.assertEqual(exp_reliable, reliable)


if __name__ == "__main__":
    unittest.main()
