""" Unit tests for the pipeline module. """
# pylint: disable=unused-import, too-many-public-methods, line-too-long, invalid-name, no-member
import unittest

import astropy.units as u
from astropy.coordinates import SkyCoord

from deblib.orbital  import phase_of_secondary_eclipse, eclipse_duration

from libs.pipeline import estimate_l3_with_gaia, get_tess_ebs_data


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


    #
    # get_tess_ebs_data(search_term: str, radius_as: float) -> dict:
    #
    def test_get_tess_ebs_data_happy_path(self):
        """ Happy path tests of get_tess_ebs_data() """
        result = get_tess_ebs_data("V* IT Cas", 5)
        self.assertIsNotNone(result)
        for k in ["t0", "period", "morph"]:
            self.assertIn(k, result)

    def test_get_tess_ebs_data_unknown_search_term(self):
        """ Test get_tess_ebs_data(unknown search term) -> None & no error """
        result = get_tess_ebs_data("V* Unknown", 5)
        self.assertIsNone(result)

    @unittest.skip
    def test_get_tess_ebs_data_specific_target(self):
        """ Tests of get_tess_ebs_data() to see if it returns the expected values """

        # We expect failures on this - it's down to inspection to see whether action is required
        targets = {
            "V* IT Cas": { "tic": 26801525, "period": 3.896637, "sum_r": 0.215, "inc": 89.68, "ecosw": 0.081, "esinw": -0.037 },

            "V* RR Lyn": { "tic": 11491822, "period": 9.945127, "sum_r": 0.142, "inc": 87.46, "ecosw": -0.078, "esinw": -0.0016 },

            "V* HP Dra": { "tic": 48356677, "period": 10.761544, "sum_r": 0.089, "inc": 87.555, "ecosw": 0.027, "esinw": 0.024 },

            "V* MU Cas": { "tic": 83905462, "period": 9.653, "sum_r": 0.194, "inc": 87.110, "ecosw": 0.187, "esinw": 0.042 },

            "V* V530 Ori": { "tic": 11961096, "period": 6.11076, "sum_r": 0.0953, "inc": 89.78, "ecosw": -0.056, "esinw": 0.066 },

            "V* V889 Aql": { "tic": 300000680, "period": 11.120757, "sum_r": 0.109, "inc": 89.06, "ecosw": -0.221, "esinw": 0.303 },
        }

        for search_term, params in targets.items():
            with self.subTest("Testing " + search_term):
                # Get expected phase & eclipse values by calculating them known system parameters
                e = (params["ecosw"]**2 + params["esinw"]**2)**0.5
                exp_phiS = phase_of_secondary_eclipse(params["ecosw"], e)
                exp_durP = eclipse_duration(params["period"], params["sum_r"], params["inc"], e, params["esinw"], False)
                exp_durS = eclipse_duration(params["period"], params["sum_r"], params["inc"], e, params["esinw"], True)

                result = get_tess_ebs_data(search_term)
                self.assertIsNotNone(result, "expected results != None")
                print(f"{search_term}:", "{" , ", ".join(f"{k}: {v:.3f}" for k, v in result.items()), "}")

                self.assertIn("phiS", result)
                self.assertAlmostEqual(result["phiS"], exp_phiS, 2, f"expected phiS ~= {exp_phiS:.3f}")

                self.assertIn("durP", result)
                self.assertAlmostEqual(result["durP"].n, exp_durP, 2, f"expected durP ~= {exp_durP:.3f}")

                self.assertIn("durS", result)
                self.assertAlmostEqual(result["durS"].n, exp_durS, 2, f"exoected durS ~= {exp_durS:.3f}")


if __name__ == "__main__":
    unittest.main()
