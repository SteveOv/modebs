""" Unit tests for the pipeline module. """
# pylint: disable=unused-import, too-many-public-methods, line-too-long, invalid-name, no-member
import unittest

import numpy as np
import astropy.units as u
from astropy.coordinates import SkyCoord
from uncertainties import nominal_value, std_dev

from tests.helpers.lightcurve_helpers import load_lightcurves, KNOWN_TARGETS
from libs.catalogues import query_tess_ebs_ephemeris
from libs.lightcurves import find_lightcurve_segments, fit_polynomial

from libs.pipeline import get_teff_from_spt, _spt_to_teff_map # pylint: disable=protected-access
from libs.pipeline import add_eclipse_meta_to_lightcurves
from libs.pipeline import choose_lightcurve_groups_for_fitting
from libs.pipeline import stitch_lightcurve_groups
from libs.pipeline import append_mags_to_lightcurves_and_detrend
from libs.pipeline import fit_target_lightcurves


class Testpipeline(unittest.TestCase):
    """ Unit tests for the pipeline module. """

    def test_get_teff_from_spt_happy_path(self):
        """ Test get_teff_from_spt(str) asserting expected responses """

        for (spt,                       exp_teff) in [
            (None,                      None),
            ("",                        None),
            (" ",                       None),
            ("V",                       None),

            ("M4.5",                    _spt_to_teff_map["M"]),
            ("K4 + M3",                 _spt_to_teff_map["K"]),
            ("G",                       _spt_to_teff_map["G"]),
            ("G1 V + M1 V",             _spt_to_teff_map["G"]),
            ("F5 IV-V + F5 IV-V",       _spt_to_teff_map["F"]),
            ("A2mA5-F2",                _spt_to_teff_map["A"]),
            ("A0 Vp(Si) + Am",          _spt_to_teff_map["A"]),
            ("B1.5 V",                  _spt_to_teff_map["B"]),
            ("O My",                    _spt_to_teff_map["O"]),

            ("M1 V + G1 V",             _spt_to_teff_map["G"]),
            ("g1 V + m1 V",             _spt_to_teff_map["G"]),
        ]:
            with self.subTest("SpT == " + ("None" if spt is None else f"'{spt}'")):
                teff = get_teff_from_spt(spt)
                if exp_teff is None:
                    self.assertIsNone(teff)
                else:
                    self.assertEqual(nominal_value(exp_teff), nominal_value(teff))
                    self.assertEqual(std_dev(exp_teff), std_dev(teff))


    #
    # add_eclipse_meta_to_lightcurves(lcs: LightCurveCollection)
    #
    def test_add_eclipse_meta_to_lightcurves_happy_path(self):
        """ Test add_eclipse_meta_to_lightcurves() assert it adds the expected lc.meta items """        
        target = "CW Eri"
        sectors = [4, 31]

        # Read the ephemeris. We need this to find eclipses and set completeness metrics
        config = KNOWN_TARGETS[target]
        eph = query_tess_ebs_ephemeris(config["tic"]) or {}
        t0 = eph.get("t0", config.get("t0", config.get("t0", None)))
        period = eph.get("period", config.get("period", None))
        widthp = eph.get("widthP", config.get("widthP", None))
        widths = eph.get("widthS", config.get("widthS", None))
        depthp = eph.get("depthP", config.get("depthP", None))
        depths = eph.get("depthS", config.get("depthS", None))
        phis = eph.get("phiS", config.get("phiS", None))
        lcs = load_lightcurves(target, sectors)

        # Test
        add_eclipse_meta_to_lightcurves(lcs, t0, period, widthp, widths, depthp, depths, phis)

        for lc in lcs:
            # See test_lightcurves.test_find_eclipses_and_completeness_known_targets
            # for test which address the values
            self.assertIn("t0", lc.meta)
            self.assertIn("primary_times", lc.meta)
            self.assertIn("primary_depths", lc.meta)
            self.assertIn("primary_completeness", lc.meta)
            self.assertIn("secondary_times", lc.meta)
            self.assertIn("secondary_depths", lc.meta)
            self.assertIn("secondary_completeness", lc.meta)

    #
    # choose_lightcurve_groups_for_fitting(lcs: LightCurveCollection, completeness_th: float) -> List[List[ix]]
    #
    def test_choose_lightcurve_groups_for_fitting_known_targets(self):
        """ Test choose_lightcurve_groups_for_fitting() assert it produces expected arrangement """
        for target,             sectors,                exp_groups in[
            # Sectors not contiguous (so no join) but are fine to use individually do 7+ of each ecl per sector
            ("CW Eri",          [4, 31],                [[4], [31]]),
            # Sector are contiguous, but joining not necessary as there are many of each eclipse per sectors
            ("CM Dra",          [24, 25, 26],           [[24], [25], [26]]),
            # The only usable combo is 52+53 as eclipses too infrequent to fit any sector individually
            ("AN Cam",          [53, 59, 52],           [[52, 53]]),
        ]:
            with self.subTest(f" {target}; {sectors} -> {exp_groups} "):
                config = KNOWN_TARGETS[target]

                # Read the ephemeris. We need this to find eclipses and set completeness metrics
                eph = query_tess_ebs_ephemeris(config["tic"]) or {}
                t0 = eph.get("t0", config.get("t0", config.get("t0", None)))
                period = eph.get("period", config.get("period", None))
                widthp = eph.get("widthP", config.get("widthP", None))
                widths = eph.get("widthS", config.get("widthS", None))
                depthp = eph.get("depthP", config.get("depthP", None))
                depths = eph.get("depthS", config.get("depthS", None))
                phis = eph.get("phiS", config.get("phiS", None))
                lcs = load_lightcurves(target, sectors)

                # Prior steps in the pipeline where we have a dependency
                # Sets primary|secondary _times & _completeness arrays and t0 ('best' primary) to lcs' meta
                add_eclipse_meta_to_lightcurves(lcs, t0, period, widthp, widths, depthp, depths, phis)

                # Test
                sector_groups = choose_lightcurve_groups_for_fitting(lcs, completeness_th=0.8)
                self.assertListEqual(exp_groups, sector_groups)

    #
    # stitch_lightcurve_groups(lcs: LightCurveCollection, completeness_th: float) -> LightCurveCollection:
    #
    def test_stitch_lightcurve_groups_known_targets(self):
        """ Test stitch_lightcurve_groups() assert it produces expected set of output LCs """
        for target,             sectors,                exp_groups in[
            ("CM Dra",          [24, 25, 26],           [[24, 25, 26]]),
            ("CM Dra",          [24, 25, 26],           [[24, 25], [26]]),
            ("CM Dra",          [24, 25, 26],           [[24], [25, 26]]),
            ("CM Dra",          [24, 25, 26],           [[24], [25], [26]]),

            # Unused LCs are an expected scenario and no warnings/errors are expected
            ("CM Dra",          [24, 25, 26],           [[25, 26]]),
            ("CM Dra",          [24, 25, 26],           [[25], [26]]),
        ]:
            with self.subTest(f" {target}; {sectors} grouped as {exp_groups} "):
                lcs = load_lightcurves(target, sectors)

                # stitch depends on the eclipse metadata added by add_eclipse_meta_to_lightcurves
                config = KNOWN_TARGETS[target]
                eph = query_tess_ebs_ephemeris(config["tic"]) or {}
                add_eclipse_meta_to_lightcurves(lcs,
                                                ref_t0=eph.get("t0", config.get("t0", None)),
                                                period=eph.get("period", config.get("period", None)),
                                                widthp=eph.get("widthP", config.get("widthP", None)),
                                                widths=eph.get("widthS", config.get("widthS", None)),
                                                depthp=eph.get("depthP", config.get("depthP", None)),
                                                depths=eph.get("depthS", config.get("depthS", None)),
                                                phis=eph.get("phiS", config.get("phiS", None)))

                # Test
                grp_lcs = stitch_lightcurve_groups(lcs, exp_groups, verbose=True)
                for ix, exp_group_sectors in enumerate(exp_groups):
                    self.assertListEqual(list(grp_lcs[ix].meta["sectors"]), exp_group_sectors)

    def test_stitch_lightcurve_groups_missing_group_members(self):
        """ Test stitch_lightcurve_groups(missing LCs) assert raised appropriate warnings """
        for (target,        sectors,        sector_groups,      exp_sectors,    warn_msg) in[
            ("CM Dra",      [25, 26],       [[25], [26], [27]], [[25], [26]],   "No LCs found"),
            ("CM Dra",      [25, 26],       [[25, 26, 27]],     [[25, 26]],     "The LC(s) [27] not found"),
        ]:
            with self.subTest(f" {target}; {sectors} grouped as {sector_groups} "):
                lcs = load_lightcurves(target, sectors)

                # stitch depends on the eclipse metadata added by add_eclipse_meta_to_lightcurves
                config = KNOWN_TARGETS[target]
                eph = query_tess_ebs_ephemeris(config["tic"]) or {}
                add_eclipse_meta_to_lightcurves(lcs,
                                                ref_t0=eph.get("t0", config.get("t0", None)),
                                                period=eph.get("period", config.get("period", None)),
                                                widthp=eph.get("widthP", config.get("widthP", None)),
                                                widths=eph.get("widthS", config.get("widthS", None)),
                                                depthp=eph.get("depthP", config.get("depthP", None)),
                                                depths=eph.get("depthS", config.get("depthS", None)),
                                                phis=eph.get("phiS", config.get("phiS", None)))

                with self.assertWarns(UserWarning, msg=warn_msg):
                    grp_lcs = stitch_lightcurve_groups(lcs, sector_groups, verbose=True)
                    for ix, exp_group_sectors in enumerate(exp_sectors):
                        self.assertListEqual(list(grp_lcs[ix].meta["sectors"]), exp_group_sectors)

    #
    # append_mags_to_lightcurves_and_detrend(lcs: LightCurveCollection, ...)
    #
    def test_append_mags_to_lightcurves_and_detrend_happy_path(self):
        """ Test append_mags_to_lightcurves_and_detrend() simple happy path test """        
        target = "AN Cam"
        config = KNOWN_TARGETS[target]
        sectors = list(config["fits"].keys())

        # Read the ephemeris. We need this to find eclipses and set completeness metrics
        eph = query_tess_ebs_ephemeris(config["tic"]) or {}
        t0 = eph.get("t0", config.get("t0", config.get("t0", None)))
        period = eph.get("period", config.get("period", None))
        widthp = eph.get("widthP", config.get("widthP", None))
        widths = eph.get("widthS", config.get("widthS", None))
        depthp = eph.get("depthP", config.get("depthP", None))
        depths = eph.get("depthS", config.get("depthS", None))
        phis = eph.get("phiS", config.get("phiS", None))
        lcs = load_lightcurves(target, sectors)

        # Prior step(s) in pipeline that test subject is dependent on.
        # The eclipse meta items will be used if flattening invoked
        add_eclipse_meta_to_lightcurves(lcs, t0, period, widthp, widths, depthp, depths, phis)

        # Test
        append_mags_to_lightcurves_and_detrend(lcs,
                                               detrend_gap_th=2,
                                               detrend_poly_degree=2,
                                               detrend_iterations=3,
                                               flatten=True,
                                               durp=widthp * period,
                                               durs=widths * period,
                                               verbose=True)

        for lc in lcs:
            # See test_lightcurves for tests covering the calculations
            self.assertIn("delta_mag", lc.colnames)
            self.assertIn("delta_mag_err", lc.colnames)


    #
    # fit_target_lightcurves():
    #
    def test_fit_target_lightcurves_happy_path(self):
        """ Debuggable test for fit_target_lightcurves() """
        config = KNOWN_TARGETS["CW Eri"]

        # Get and pre-process some known lightcurves
        lcs = load_lightcurves("CW Eri", [4, 31], with_mag_columns=True)
        for lc in lcs:
            for s in find_lightcurve_segments(lc, threshold=0.5 * u.d):
                lc[s]["delta_mag"] -= fit_polynomial(lc.time[s], lc[s]["delta_mag"], 1, 3)
            lc.meta["clip_mask"] = np.ones((len(lc)), dtype=bool)

        in_params = [{
            "task": 3,
            "rA_plus_rB": 0.3,      "k": 0.7,
            "inc": 86.4,            "qphot": 0.836,
            "ecosw": 0.005,         "esinw": -0.010,
            "gravA": 0,             "gravB": 0,
            "J": 0.9,               "L3": max(0, 1-lc.meta.get("CROWDSAP", 1)),
            "LDA": "pow2",          "LDB": "pow2",
            "LDA1": 0.64,           "LDB1": 0.65,
            "LDA2": 0.47,           "LDB2": 0.50,
            "reflA": 0,             "reflB": 0,

            # Ephemeris
            "t0": config["t0"].value,
            "period": config["period"].to(u.d).value,

                                    "qphot_fit": 0,
            "ecosw_fit": 1,         "esinw_fit": 1,
            "gravA_fit": 0,         "gravB_fit": 0,
                                    "L3_fit": 1,
            "LDA1_fit": 1,          "LDB1_fit": 1,
            "LDA2_fit": 0,          "LDB2_fit": 0,
            "reflA_fit": 0,         "reflB_fit": 0,
                                    "sf_fit": 1,
            "period_fit": 1,        "t0_fit": 1,

        } for lc in lcs]

        read_keys = ["rA_plus_rB", "k", "J", "ecosw", "esinw", "inc", "L3"]
        out_params = fit_target_lightcurves(lcs, in_params, read_keys,
                                            task=3, max_workers=4, file_prefix="test-pipeline")

        for op_dict in out_params:
            print(f"[{op_dict['in_fname'].stem}]",
                  ", ".join(f"{k}={op_dict[k]}" for k in read_keys))


if __name__ == "__main__":
    unittest.main()
