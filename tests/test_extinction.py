""" Unit tests for the extinction module. """
# pylint: disable=unused-import, line-too-long, invalid-name, no-member
import unittest

import astropy.units as u
from astropy.coordinates import SkyCoord

from libs.extinction import get_ebv, get_av, get_bayestar_ebv

class Testextinction(unittest.TestCase):
    """ Unit tests for the extinction module. """

    def test_get_ebv_happy_path(self):
        """ Tests get_ebv() - basic happy path test """
        print()
        for (target, exp_gontcharov_ebv, exp_bayestar_ebv, coords) in [
            ("CM Dra", 0.01, 0., SkyCoord("16h34m20.3302660573", "+57d09m44.368918696", 14.844*u.pc, frame="icrs")),
            ("MU Cas", None, 0.36, SkyCoord(3.964810786976831*u.deg, 60.43156179196987*u.deg, 1000/0.513272695221339*u.pc, frame="icrs")),
        ]:
            with self.subTest(target):
                for val, flags in get_ebv(coords, ["gontcharov_av", get_bayestar_ebv]):
                    if val:
                        print(f"val={val}, flags={flags}")

                        self.assertIn("converged", flags)
                        self.assertIn("source", flags)
                        self.assertIn("type", flags)

                        if "gontcharov" in flags["source"]:
                            self.assertEqual("E(B-V)", flags["type"])
                            self.assertEqual("get_gontcharov_av", flags["source"])
                            self.assertAlmostEqual(val, exp_gontcharov_ebv, 2)
                        elif "bayestar" in flags["source"]:
                            self.assertEqual("E(B-V)", flags["type"])
                            self.assertEqual("get_bayestar_ebv", flags["source"])
                            self.assertAlmostEqual(val, exp_bayestar_ebv, 2)

    def test_get_av_happy_path(self):
        """ Tests get_av() - basic happy path test """
        print()
        for (target, exp_gontcharov_av, exp_bayestar_av, coords) in [
            ("CM Dra", 0.02, 0., SkyCoord("16h34m20.3302660573", "+57d09m44.368918696", 14.844*u.pc, frame="icrs")),
            ("MU Cas", None, 1.13, SkyCoord(3.964810786976831*u.deg, 60.43156179196987*u.deg, 1000/0.513272695221339*u.pc, frame="icrs")),
        ]:
            with self.subTest(target):
                for val, flags in get_av(coords, ["gontcharov_av", get_bayestar_ebv]):
                    if val:
                        print(f"val={val}, flags={flags}")

                        self.assertIn("converged", flags)
                        self.assertIn("source", flags)
                        self.assertIn("type", flags)

                        if "gontcharov" in flags["source"]:
                            self.assertEqual("A_V", flags["type"])
                            self.assertEqual("get_gontcharov_av", flags["source"])
                            self.assertAlmostEqual(val, exp_gontcharov_av, 2)
                        elif "bayestar" in flags["source"]:
                            self.assertEqual("A_V", flags["type"])
                            self.assertEqual("get_bayestar_ebv", flags["source"])
                            self.assertAlmostEqual(val, exp_bayestar_av, 2)

if __name__ == "__main__":
    unittest.main()
