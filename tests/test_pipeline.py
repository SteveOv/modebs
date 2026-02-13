""" Unit tests for the pipeline module. """
# pylint: disable=unused-import, too-many-public-methods, line-too-long, invalid-name, no-member
import unittest

import numpy as np
import astropy.units as u
from astropy.coordinates import SkyCoord

from deblib.orbital  import phase_of_secondary_eclipse, eclipse_duration
from tests.helpers.lightcurve_helpers import load_lightcurves, KNOWN_TARGETS
from libs.catalogues import query_tess_ebs_ephemeris
from libs.lightcurves import find_lightcurve_segments, fit_polynomial
from libs.lightcurves import find_eclipses_and_completeness

from libs.pipeline import group_sectors_for_fitting, stitch_group_lightcurves
from libs.pipeline import estimate_l3_with_gaia
from libs.pipeline import fit_target_lightcurves


class Testpipeline(unittest.TestCase):
    """ Unit tests for the pipeline module. """

    #
    # group_sectors_for_fitting(lcs: LightCurveCollection, completeness_th: float) -> List[List[ix]]
    #
    def test_group_sectors_for_fitting_known_targets(self):
        """ Test group_sectors_for_fitting() assert it produces expected arrangement """
        for target,             sectors,                exp_sector_groups in[
            # Sectors not contiguous (so no join) but are fine to use individually do 7+ of each ecl per sector
            ("CW Eri",          [4, 31],                [[4], [31]]),
            # Sector are contiguous, but joining not necessary as there are many of each eclipse per sectors
            ("CM Dra",          [24, 25, 26],           [[24], [25], [26]]),
            # The only usable combo is 52+53 as eclipses too infrequent to fit any sector individually
            ("AN Cam",          [53, 59, 52],           [[52, 53]]),
        ]:
            with self.subTest(f" {target}; {sectors} -> {exp_sector_groups} "):
                config = KNOWN_TARGETS[target]

                # Read the ephemeris. We need this to find eclipses and set completeness metrics
                eph = query_tess_ebs_ephemeris(config["tic"]) or {}
                t0 = eph.get("t0", config.get("t0", config.get("t0", None)))
                period = eph.get("period", config.get("period", None))
                durp = eph.get("durP", config.get("durP", None))
                durs = eph.get("durS", config.get("durS", None))
                phis = eph.get("phiS", config.get("phiS", None))

                # Sets primary|secondary _times & _completeness arrays and t0 ('best' primary) to lcs' meta
                lcs = load_lightcurves(target, sectors)
                for lc in lcs:
                    lc.meta |= find_eclipses_and_completeness(lc, t0, period, durp, durs, phis)
                    t0 = lc.meta.get("t0", None) or t0

                # Test
                sector_groups = group_sectors_for_fitting(lcs, completeness_th=0.8)
                self.assertListEqual(exp_sector_groups, sector_groups)

    #
    # stitch_group_lightcurves(lcs: LightCurveCollection, completeness_th: float) -> List[List[ix]]
    #
    def test_stitch_group_lightcurves(self):
        """ Test stitch_group_lightcurves() assert it produces expected set of output LCs """
        for target,             sectors,                sector_groups in[
            ("CM Dra",          [24, 25, 26],           [[24, 25, 26]]),
            ("CM Dra",          [24, 25, 26],           [[24, 25], [26]]),
            ("CM Dra",          [24, 25, 26],           [[24], [25, 26]]),
            ("CM Dra",          [24, 25, 26],           [[24], [25], [26]]),

            # Unused LCs are an expected scenario and no warnings/errors are expected
            ("CM Dra",          [24, 25, 26],           [[25, 26]]),
            ("CM Dra",          [24, 25, 26],           [[25], [26]]),
        ]:
            with self.subTest(f" {target}; {sectors} grouped as {sector_groups} "):
                config = KNOWN_TARGETS[target]

                # Read the ephemeris. We need this to find eclipses and set completeness metrics
                eph = query_tess_ebs_ephemeris(config["tic"]) or {}
                t0 = eph.get("t0", config.get("t0", config.get("t0", None)))
                period = eph.get("period", config.get("period", None))
                durp = eph.get("durP", config.get("durP", None))
                durs = eph.get("durS", config.get("durS", None))
                phis = eph.get("phiS", config.get("phiS", None))

                lcs = load_lightcurves(target, sectors)
                for lc in lcs:
                    # These would be present in a real pipeline and we see they've been copied over.
                    # Sets primary|secondary _times & _completeness and t0 ('best' primary) to meta.
                    lc.meta |= find_eclipses_and_completeness(lc, t0, period, durp, durs, phis)
                    t0 = lc.meta.get("t0", None) or t0

                # Test
                grp_lcs = stitch_group_lightcurves(lcs, sector_groups, verbose=True)
                for ix, exp_group_sectors in enumerate(sector_groups):
                    self.assertListEqual(list(grp_lcs[ix].meta["sectors"]), exp_group_sectors)

    def test_stitch_group_lightcurves_missing_group_members(self):
        """ Test stitch_group_lightcurves(missing LCs) assert raised appropriate warnings """
        for (target,        sectors,        sector_groups,      exp_sectors,    warn_msg) in[
            ("CM Dra",      [25, 26],       [[25], [26], [27]], [[25], [26]],   "No LCs found"),
            ("CM Dra",      [25, 26],       [[25, 26, 27]],     [[25, 26]],     "The LC(s) [27] not found"),
        ]:
            with self.subTest(f" {target}; {sectors} grouped as {sector_groups} "):
                lcs = load_lightcurves(target, sectors)
                with self.assertWarns(UserWarning, msg=warn_msg):
                    grp_lcs = stitch_group_lightcurves(lcs, sector_groups, verbose=True)
                    for ix, exp_group_sectors in enumerate(exp_sectors):
                        self.assertListEqual(list(grp_lcs[ix].meta["sectors"]), exp_group_sectors)

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

        in_params = {
            "task": 3,
            "rA_plus_rB": 0.3,      "k": 0.7,
            "inc": 86.4,            "qphot": 0.836,
            "ecosw": 0.005,         "esinw": -0.010,
            "gravA": 0,             "gravB": 0,
            "J": 0.9,               "L3": 0,
            "LDA": "pow2",          "LDB": "pow2",
            "LDA1": 0.64,           "LDB1": 0.65,
            "LDA2": 0.47,           "LDB2": 0.50,
            "reflA": 0,             "reflB": 0,

            # primary_epoch is passed in separately as it may vary by sector
            "period": config["period"].to(u.d).value,

                                    "qphot_fit": 0,
            "ecosw_fit": 1,         "esinw_fit": 1,
            "gravA_fit": 0,         "gravB_fit": 0,
                                    "L3_fit": 1,
            "LDA1_fit": 1,          "LDB1_fit": 1,
            "LDA2_fit": 0,          "LDB2_fit": 0,
            "reflA_fit": 0,         "reflB_fit": 0,
                                    "sf_fit": 1,
            "period_fit": 1,        "primary_epoch_fit": 1,

        }

        read_keys = ["rA_plus_rB", "k", "J", "ecosw", "esinw", "inc", "L3"]
        pe = config["t0"].value
        out_params = fit_target_lightcurves(lcs, in_params, read_keys, pe,
                                            task=3, max_workers=4, file_prefix="test-pipeline")

        for op_dict in out_params:
            print(f"[{op_dict['in_fname'].stem}]",
                  ", ".join(f"{k}={op_dict[k]}" for k in read_keys))


if __name__ == "__main__":
    unittest.main()
