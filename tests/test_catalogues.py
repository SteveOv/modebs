""" Unit tests for the catalogues module. """
# pylint: disable=unused-import, too-many-public-methods, line-too-long, invalid-name, no-member
import unittest

import numpy as np
import matplotlib.pyplot as plt

from deblib import orbital

from tests.helpers import lightcurve_helpers

from libs.pipeline import nominal_value

from libs.catalogues import query_tess_ebs_ephemeris, query_tess_ebs_in_sh
from libs.catalogues import estimate_eclipse_durations_from_morphology
from libs.catalogues import _read_table # pylint: disable=protected-access

class Testcatalogues(unittest.TestCase):
    """ Unit tests for the catalogues module. """

    #
    # query_tess_ebs_ephemeris(tics) -> dict
    #
    def test_query_tess_ebs_ephemeris_argument_handling(self):
        """ Tests for query_tess_ebs_ephemeris() to assert that it decodes TICs correctly """
        # 98853987 and 30313682 are in catalogue, whereas 0000000 is not
        for tics,                       msg in [
            ("TIC 98853987",            "single valid tic as str"),
            (98853987,                  "single valid tic as int"),
            ([98853987],                "list of tics [single valid tic]"),
            ([0000000, 98853987],       "list of tics, first unknown, expect to use second"),
            ([98853987, 30313682],      "list of tics, first known, expect to use first"),
        ]:
            with self.subTest(msg):
                # Expected values for 98853987
                data = query_tess_ebs_ephemeris(tics)
                self.assertAlmostEqual(2.728, data["period"].nominal_value, 3)
                self.assertAlmostEqual(0.511, data["morph"], 3)

    def test_query_tess_ebs_ephemeris_known_target(self):
        """ Tests for query_tess_ebs_ephemeris(known TICs) to assert it returns expected data """
        for (tic,           exp_t0,         exp_per,    exp_morph,  exp_phis,   exp_durp,   exp_durs) in [
            # TIC 26801525 is known that the 2g & pf phip & phis are switched (phis should be 0.551)
            (26801525,      1766.61882,     3.89665,    0.250,      0.449,      0.27,       0.29),
            # TIC 26801525 actual durS is nearer 0.42
            (118313102,     1518.556197,    9.255727,   0.180,      0.400,      0.37,       0.48),
            (350298314,     1358.040315,    47.719114,  0.002,      0.262,      0.28,       0.35),
            # ZZ Boo which is missing eclipse data for the secondary phiS, durS & depthS
            (357358259,     1932.31308,     2.49616,    0.523,      None,      0.406,       None),
        ]:
            with self.subTest(f"Target: {tic}"):
                data = query_tess_ebs_ephemeris(tic)

                self.assertAlmostEqual(data["t0"].n, exp_t0, 5, f"Expected {tic} t0 ~= {exp_t0}")
                self.assertAlmostEqual(data["period"].n, exp_per, 5, f"Expected {tic} period ~= {exp_per}")

                self.assertAlmostEqual(data["morph"], exp_morph, 3, f"Expected {tic} morph ~= {exp_morph}")
                self.assertAlmostEqual(data["phiS"], exp_phis, 3, f"Expected {tic} phiS ~= {exp_phis}")

                # Duration in TESS-ebs are in units of phase whereas these are in days
                if exp_durp is None:
                    self.assertIsNone(data["durP"], f"Expected {tic} durP is None")
                else:
                    self.assertAlmostEqual(data["durP"].n, exp_durp, 2, f"Expected {tic} durP ~= {exp_durp}")
                if exp_durs is None:
                    self.assertIsNone(data["durS"], f"Expected {tic} durS is None")
                else:
                    self.assertAlmostEqual(data["durS"].n, exp_durs, 2, f"Expected {tic} durS ~= {exp_durs}")

    @unittest.skip
    def test_query_tess_ebs_ephemeris_known_orbital_params(self):
        """ Tests of query_tess_ebs_ephemeris() to assert it returns usable orbital values """
        # We expect failures on this - it's down to inspection to see whether action is required
        for (target,        tic,        period,     sum_r,      inc,        ecosw,      esinw) in [
            # IT Cas know the 2g & pf phiS values are incorrect at 0.448 when expected to be 0.552.
            ( "IT Cas",     26801525,   3.896637,   0.215,      89.68,      0.081,      -0.037),
            ( "RR Lyn",     11491822,   9.945127,   0.142,      87.46,      -0.078,     -0.0016),
            ( "HP Dra",     48356677,   10.761544,  0.089,      87.555,     0.027,      0.024),
            ( "MU Cas",     83905462,   9.653,      0.194,      87.110,     0.187,      0.042),
        ]:
            with self.subTest("Testing " + target):
                # Calculate expected approx eclipse phase & duration values from known system params
                e = (ecosw**2 + esinw**2)**0.5
                exp_phiS = orbital.phase_of_secondary_eclipse(ecosw, e)
                exp_durP = orbital.eclipse_duration(period, sum_r, inc, e, esinw, False)
                exp_durS = orbital.eclipse_duration(period, sum_r, inc, e, esinw, True)

                data = query_tess_ebs_ephemeris(tic)
                self.assertIsNotNone(data, f"expected {target} data != None")

                print(f"{target}:", "{" , ", ".join(f"{k}: {v:.3f}" for k, v in data.items()), "}")

                self.assertAlmostEqual(data["phiS"], exp_phiS, 2, f"Expected {target} phiS ~= {exp_phiS:.3f}")
                self.assertAlmostEqual(data["durP"].n, exp_durP, 1, f"Expected {target} durP ~= {exp_durP:.3f}")
                self.assertAlmostEqual(data["durS"].n, exp_durS, 1, f"Expected {target} durS ~= {exp_durS:.3f}")


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
                print()
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

    @unittest.skip
    def test_calc_morph_ecl_width_fit(self):
        """ Get coefficients of a fit to the TESS-ebs morph vs mean eclipse width """
        tess_ebs = _read_table(catalogue="J/ApJS/258/16", table_fname="tess-ebs.dat")
        morphs = tess_ebs["Morph"].value
        mean_widths = np.maximum(
            np.mean([tess_ebs["Wp-pf"].value, tess_ebs["Ws-pf"].value], axis=0),
            np.mean([tess_ebs["Wp-2g"].value, tess_ebs["Ws-2g"].value], axis=0)
        )

        mask = mean_widths > 0

        fit_mask = mask & (morphs <= 0.7)
        coeffs = np.polyfit(x=morphs[fit_mask], y=mean_widths[fit_mask], deg=4)
        print(f"Fitted coefficients: {coeffs}")

        _, ax = plt.subplots(figsize=(6, 4), constrained_layout=True)
        ax.scatter(morphs[mask], mean_widths[mask], marker=".", label="TESS-ebs")
        ax.scatter(morphs[mask], np.poly1d(coeffs)(morphs[mask]), marker=".", label="initial fit")
        ax.set(xlabel="morph", ylabel="mean eclipse width [phase]")

        # Shift the fit up so it's never negative. This follows the data quite
        # nicely up to morph 0.6, which is our region of interest, and beyond.
        coeffs[-1] = 0.0
        print(f"Shifted coefficients: {coeffs}")
        ax.scatter(morphs[mask], np.poly1d(coeffs)(morphs[mask]), marker=".", label="shifted fit")

        ax.legend(loc="best")
        plt.show()
        plt.close()


if __name__ == "__main__":
    unittest.main()
