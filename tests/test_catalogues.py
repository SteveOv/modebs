""" Unit tests for the catalogues module. """
# pylint: disable=unused-import, too-many-public-methods, line-too-long, invalid-name, no-member
import unittest

from tests.helpers import lightcurve_helpers

import numpy as np
import matplotlib.pyplot as plt

from deblib import orbital

from libs.pipeline import nominal_value

from libs.catalogues import query_tess_ebs_ephemeris, query_tess_ebs_in_sh
from libs.catalogues import estimate_eclipse_durations_from_morphology
from libs.catalogues import _read_table # pylint: disable=protected-access

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


    #
    #   estimate_eclipse_durations_from_morphology(morph, period, esinw)
    #
    def test_estimate_eclipse_widths_from_morphology_happy(self):
        """ Interactive tests estimate_eclipse_widths_from_morphology() calculations """
        print()
        for (target,                esinw,      period) in [
            ("CW Eri",              None,       None),
            ("TIC 0063192395",      None,       None),
            ("TIC 0080650858",      None,       None),
            ("TIC 0118313102",      None,       None),
            ("TIC 0141685465",      None,       None),
            ("TIC 0160328766",      None,       None),
            ("TIC 0167756615",      None,       None),
            ("TIC 0198011271",      None,       None),
            ("TIC 0219362976",      None,       None),
            ("TIC 0259543079",      None,       None),
            ("TIC 0279741942",      None,       None),
        ]:
            with self.subTest(target):
                if target in lightcurve_helpers.KNOWN_TARGETS:
                    config = lightcurve_helpers.KNOWN_TARGETS[target]
                    tess_ebs = query_tess_ebs_ephemeris(config["tic"])
                    esinw = esinw or config.get("esinw", 0)
                    period = period or config["period"]
                else:
                    tess_ebs = query_tess_ebs_ephemeris(target)
                    esinw = esinw or orbital.estimate_esinw(tess_ebs["durP"], tess_ebs["durS"])
                    period = period or tess_ebs["period"]

                morph = tess_ebs["morph"]
                print(f"TESS-ebs[{target}]: durP={tess_ebs['durP']:.3f},",
                      f"durS={tess_ebs['durS']:.3f}, morph={morph:.3f}")

                durations = estimate_eclipse_durations_from_morphology(morph, period, esinw)
                print(f"estimate[{target}]: durP={durations[0]:.3f},",
                      f"durS={durations[1]:.3f} (where esinw={esinw:.6f})")

    #@unittest.skip
    def test_calc_morph_ecl_width_fit(self):
        """ Get coefficients of a fit to the TESS-ebs morph vs mean eclipse width """
        tess_ebs = _read_table(catalogue="J/ApJS/258/16", table_fname="tess-ebs.dat")
        morphs = tess_ebs["Morph"].value
        mean_widths = np.maximum(
            np.mean([tess_ebs["Wp-pf"].value, tess_ebs["Ws-pf"].value], axis=0),
            np.mean([tess_ebs["Wp-2g"].value, tess_ebs["Ws-2g"].value], axis=0)
        )

        mask = mean_widths > 0

        coeffs = np.polyfit(x=morphs[mask], y=mean_widths[mask], deg=2)
        print(f"Fitted coefficients: {coeffs}")

        _, ax = plt.subplots(figsize=(6, 4), constrained_layout=True)
        ax.scatter(morphs[mask], mean_widths[mask], label="TESS-ebs")
        ax.scatter(morphs[mask], np.poly1d(coeffs)(morphs[mask]), label="original fit")
        ax.set(xlabel="morph", ylabel="mean eclipse width [phase]")

        coeffs[-1] = 0.0
        print(f"Shifted coefficients: {coeffs}")
        ax.scatter(morphs[mask], np.poly1d(coeffs)(morphs[mask]), label="shifted fit")

        ax.legend(loc="best")
        plt.show()
        plt.close()


if __name__ == "__main__":
    unittest.main()
