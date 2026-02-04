""" Unit tests for the catalogues module. """
# pylint: disable=unused-import, too-many-public-methods, line-too-long, invalid-name, no-member
import unittest

from libs.catalogues import query_tess_ebs_ephemeris, query_tess_ebs_in_sh

class Testcatalogues(unittest.TestCase):
    """ Unit tests for the catalogues module. """

    #
    # query_tess_ebs_ephemeris(tics) -> dict
    #
    def test_query_tess_ebs_ephemeris_happy_path(self):
        """ Happy path tests for query_tess_ebs_ephemeris() """
        # 98853987 and 30313682 are in catalogue, whereas 0000000 is not
        for tics,                       msg in [
            ("TIC 98853987",            "single valid tic as str"),
            (98853987,                  "single valid tic as int"),
            ([98853987],                "list of tics [single valid tic]"),
            ([0000000, 98853987],       "list of tics, first unknown, expect to use second"),
            ([98853987, 30313682],      "list of tics, first known, expect to use first"),
        ]:
            with self.subTest(msg):
                data = query_tess_ebs_ephemeris(tics)

                # Expected values for 98853987
                self.assertAlmostEqual(2.728, data["period"].nominal_value, 3)
                self.assertAlmostEqual(0.511, data["morph"], 3)


    #
    # query_tess_ebs_in_sh(tics) -> dict
    #
    def test_query_tess_ebs_in_sh_happy_path(self):
        """ Happy path tests for query_tess_ebs_in_sh() """

        # 30313682 and 55497281 are in catalogue, whereas 0000000 is not
        for tics,                       msg in [
            ("TIC 30313682",            "single valid tic as str"),
            (30313682,                  "single valid tic as int"),
            ([30313682],                "list of tics [single valid tic]"),
            ([0000000, 30313682],       "list of tics, first unknown, expect to use second"),
            ([30313682, 55497281],      "list of tics, first known, expect to use first"),
        ]:
            with self.subTest(msg):
                data = query_tess_ebs_in_sh(tics)

                # Expected values for 30313682
                self.assertAlmostEqual(5.727, data["period"], 3)
                self.assertAlmostEqual(0.394, data["k"], 3)


if __name__ == "__main__":
    unittest.main()
