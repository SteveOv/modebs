""" Unit tests for the extinction module. """
# pylint: disable=unused-import, line-too-long, invalid-name, no-member
import unittest

import astropy.units as u
from astropy.coordinates import SkyCoord

from libs.extinction import iterate, get_bayestar_ebv

class Testextinction(unittest.TestCase):
    """ Unit tests for the extinction module. """

    def test_iterate_happy_path(self):
        """ Tests iterate() - basic happy path test """
        print()
        for (target, yield_ebv, exp_results, coords) in [
            # For A_V coefficients
            ("CM Dra", False, [(0.02, True), (0., False)], SkyCoord("16h34m20.3302660573", "+57d09m44.368918696", 14.844*u.pc, frame="icrs")),
            ("MU Cas", False, [(1.20, True)], SkyCoord(3.964810786976831*u.deg, 60.43156179196987*u.deg, 1000/0.513272695221339*u.pc, frame="icrs")),

            # For E(B-V) coefficients
            ("CM Dra", True, [(0.01, True), (0, False)], SkyCoord("16h34m20.3302660573", "+57d09m44.368918696", 14.844*u.pc, frame="icrs")),
            ("MU Cas", True, [(0.39, True)], SkyCoord(3.964810786976831*u.deg, 60.43156179196987*u.deg, 1000/0.513272695221339*u.pc, frame="icrs")),
        ]:
            with self.subTest(("E(B-V)" if yield_ebv else "A_V") + f" for {target}"):
                funcs = ["gontcharov_av", get_bayestar_ebv]
                for ix, (val, reliable) in enumerate(iterate(coords, funcs, yield_ebv=yield_ebv, verbose=True)):
                    self.assertTrue(ix < len(exp_results))
                    self.assertAlmostEqual(exp_results[ix][0], val, 2)
                    self.assertEqual(exp_results[ix][1], reliable)

if __name__ == "__main__":
    unittest.main()
