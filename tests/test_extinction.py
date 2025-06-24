""" UUnit tests for the extinction module. """
# pylint: disable=unused-import, too-many-public-methods, line-too-long, invalid-name, no-member
import unittest

import astropy.units as u
from astropy.coordinates import SkyCoord

from libs.extinction import get_ebv, get_bayestar_ebv

class Testextinction(unittest.TestCase):
    """ Unit tests for the extinction module. """


    def test_get_ebv_happy_path(self):
        """ Tests get_extinction() - basic happy path test """
        # CM Dra
        coords = SkyCoord("16h34m20.3302660573", "+57d09m44.368918696", 14.844 * u.pc, frame="icrs")
        for val, flags in get_ebv   (coords, ["gontcharov_ebv", get_bayestar_ebv]):
            print(f"val={val}, flags={flags}")

            self.assertIn("converged", flags)
            self.assertIn("source", flags)
            self.assertIn("type", flags)

            if "gontcharov" in flags["source"]:
                self.assertEqual("E(B-V)", flags["type"])
                self.assertAlmostEqual(val, 0.015, 3)
            elif "bayestar" in flags["source"]:
                self.assertEqual("E(B-V)", flags["type"])
                self.assertEqual(val, 0)

if __name__ == "__main__":
    unittest.main()
