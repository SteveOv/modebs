""" Unit tests for the extinction module. """
# pylint: disable=unused-import, line-too-long, invalid-name, no-member
import unittest

import astropy.units as u
from astropy.coordinates import SkyCoord

from libs.extinction import iterate, get_bayestar_ebv, get_decaps_av

class Testextinction(unittest.TestCase):
    """ Unit tests for the extinction module. """

    def test_iterate_happy_path(self):
        """ Tests iterate() - basic happy path test """
        print()
        for (target,        yield_ebv, exp_results, coords) in [
            # For A_V coefficients
            ("CM Dra",      False, [(0.02, True), (0., False)], SkyCoord(248.57558061*u.deg, 57.16757382*u.deg, 14.86158398*u.pc, frame="icrs")),
            # In Gontcharov and Bayestar
            ("IT Cas",      False, [(0.38, True), (0.36, True)], SkyCoord(355.50569743*u.deg, 51.74352579*u.deg, 514.95778083*u.pc, frame="icrs")),
            # In Gontcharov and expected to be in DECaPS (but that's throwing errors atm!)
            ("TIC 7695666", False, [(0.20, True)], SkyCoord(65.64676092*u.deg, -41.48319921*u.deg, 367.56561186*u.pc, frame="icrs")),

            # For E(B-V) coefficients
            ("CM Dra",      True, [(0.01, True), (0, False)], SkyCoord(248.57558061*u.deg, 57.16757382*u.deg, 14.86158398*u.pc, frame="icrs")),
            # In Gontcharov and Bayestar
            ("IT Cas",      True, [(0.12, True), (0.11, True)], SkyCoord(355.50569743*u.deg, 51.74352579*u.deg, 514.95778083*u.pc, frame="icrs")),
            # In Gontcharov and expected to be in DECaPS (but that's throwing errors atm!)
            ("TIC 7695666", True, [(0.06, True)], SkyCoord(65.64676092*u.deg, -41.48319921*u.deg, 367.56561186*u.pc, frame="icrs")),
        ]:
            with self.subTest(("E(B-V)" if yield_ebv else "A_V") + f" for {target}"):
                funcs = ["gontcharov_av", get_decaps_av, get_bayestar_ebv]
                for ix, (val, reliable) in enumerate(iterate(coords, funcs, yield_ebv=yield_ebv, verbose=True)):
                    self.assertTrue(ix < len(exp_results))
                    self.assertAlmostEqual(exp_results[ix][0], val, 2)
                    self.assertEqual(exp_results[ix][1], reliable)

if __name__ == "__main__":
    unittest.main()
