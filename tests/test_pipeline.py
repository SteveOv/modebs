""" Unit tests for the pipeline module. """
# pylint: disable=unused-import, too-many-public-methods, line-too-long, invalid-name, no-member
import unittest

import astropy.units as u
from astropy.coordinates import SkyCoord

from libs.pipeline import estimate_l3_with_gaia


class Testpipeline(unittest.TestCase):
    """ Unit tests for the pipeline module. """

    #
    # estimate_l3_with_gaia(centre: SkyCoord, radius_as: float, target_source_id: int,
    #                       target_g_mag: float, max_l3: float, verbose: bool) -> float:
    #
    def test_estimate_l3_with_gaia_happy_path(self):
        """ Happy path tests of estimate_l3_with_gaia() """

        for targ,       targ_source_id,     targ_g_mag, ra,             dec,            dist,   radius_as,  max_l3, exp_l3 in [
            ("UZ Dra", 2261658485914111744, None,   291.47947545,   68.93546881,    185.38698966,   120,    None,   0.0143),
            ("UZ Dra", 2261658485914111744, None,   291.47947545,   68.93546881,    185.38698966,   120,    0.0100, 0.0100),
            ("UZ Dra", 2261658485914111744, None,   291.47947545,   68.93546881,    185.38698966,   120,    0.0200, 0.0143),

            ("CW Eri", 5152756553745197952, None,   46.00004676,    -17.73777489,   190.91243808,   120,    None,   0.0038),
            ("CW Eri", 5152756553745197952, None,   46.00004676,    -17.73777489,   190.91243808,   30,     None,   0.0),

            ("ZZ Boo", 1450355965609917568, None,   209.03917826,   25.91868226,    106.44379805,   120,    None,   0.0003),

            # Currently not known in Gaia DR3 however it appears in the result with a DR2 source id
            # (5561235358371617152). This tests the logic which, when a target id is not supplied,
            # excludes any object close to the search centre with a magnitude similar to the target.
            ("V362 Pup", None,              7.6741, 107.66503779,   -41.26505267,   304.84087306,   120,    None,   0.0125),
        ]:
            with self.subTest(f"Testing {targ}"):
                centre = SkyCoord(ra=ra*u.deg, dec=dec*u.deg, distance=dist*u.pc, frame="icrs")
                l3 = estimate_l3_with_gaia(centre, radius_as, targ_source_id, targ_g_mag, max_l3, verbose=True)
                self.assertAlmostEqual(exp_l3, l3, 4)

if __name__ == "__main__":
    unittest.main()
