""" Unit tests for the pipeline module. """
from typing import List
from pathlib import Path
import unittest
from contextlib import redirect_stdout
from io import StringIO

# pylint: disable=no-member, wrong-import-position, line-too-long
import numpy as np
import astropy.units as u
import matplotlib.pyplot as plt

from tests.helpers import lightcurve_helpers
from libs import lightcurves, catalogues, plots


# pylint: disable=too-many-public-methods, line-too-long
class Testlightcurves(unittest.TestCase):
    """ Unit tests for the lightcurves module. """
    cache_dir = Path.cwd() / ".cache/.test_data/lightcurves"

    @classmethod
    def setUpClass(cls):
        """ Make sure JKTEBOP_DIR is corrected up as tests may modify it. """
        cls.cache_dir.mkdir(parents=True, exist_ok=True)
        return super().setUpClass()

    @classmethod
    def tearDownClass(cls):
        """ Make sure JKTEBOP_DIR is corrected up as tests may modify it. """
        return super().tearDownClass()


    def test_load_lightcurves_mast_download(self):
        """ Simple happy path test of load_lightcurves() for forces mast queries """
        # Don't push this too hard otherwise we may get throttled by MAST
        for target,     sectors,        mission,            author,                 exptime,        exp_sectors in [
            ("CW Eri",  None,           "TESS",             "SPOC",                 None,           [4, 31]),
            # ("CW Eri",  4,              "TESS",             "SPOC",                 None,           [4]),
            # ("CW Eri",  [31, 4],        "TESS",             "SPOC",                 None,           [4, 31]),
            # ("CW Eri",  4,              ["TESS","HLSP"],    "SPOC",                 None,           [4]),
            # ("CW Eri",  4,              "TESS",             ["SPOC", "TESS-SPOC"],  None,           [4, 4]),

            # S40 & 54 @ 600 s and S80 @ 200 s - will accept up to 2 exptime values (seems to ignore rest)
            ("V889 Aql",  [40, 54, 80],   "TESS",           ["SPOC","TESS-SPOC"],   [200, 600],     [40, 54, 80]),
        ]:

            with self.subTest(f"{target} {sectors}/{mission}/{author}/{exptime}"):
                lcs = lightcurves.load_lightcurves(target,
                                                   sectors=sectors,
                                                   mission=mission,
                                                   author=author,
                                                   exptime=exptime,
                                                   force_mast=True,
                                                   cache_dir=self.cache_dir,
                                                   verbose=True)

                self.assertEqual(len(exp_sectors), len(lcs))
                self.assertListEqual(exp_sectors, list(lcs.sector))


    def test_load_lightcurves_service_locally(self):
        """ Simple happy path test of load_lightcurves() for locally serviced respose """
        for target,     sectors,        mission,            author,                 exptime,        exp_sectors in [
            ("CW Eri",  None,           "TESS",             "SPOC",                 None,           [4, 31]),
        ]:

            with self.subTest(f"{target} {sectors}/{mission}/{author}/{exptime}"):

                # Ensure we have a locally cached search result
                lcs = lightcurves.load_lightcurves(target, target,
                                                   sectors, mission, author, exptime,
                                                   cache_dir=self.cache_dir)

                # Now test - the same query should be serviced from the local cache
                with redirect_stdout(StringIO()) as stdout_cap:
                    lcs = lightcurves.load_lightcurves(target, target,
                                                       sectors, mission, author, exptime,
                                                       force_mast=False, cache_dir=self.cache_dir,
                                                       verbose=True)
                    stdout_text = stdout_cap.getvalue()
                self.assertIn("Loaded previously cached search results", stdout_text)
                self.assertEqual(len(exp_sectors), len(lcs))
                self.assertListEqual(exp_sectors, list(lcs.sector))


    #
    #   append_magnitude_columns(LightCurve, name: str, err_name: str)
    #
    def test_append_magnitude_columns_happy_path(self):
        """ Happy path tests for append_magnitude_columns(LC) """
        exp_name = "delta_mag"
        exp_err_name = "delta_mag_err"
        exp_max_mag = 0.467 * u.mag

        # Test with/without the LC's fluxes having been normalized()
        for normalized in [True, False]:
            lc = lightcurve_helpers.load_default_lightcurve("CW Eri",
                                                            normalized=normalized,
                                                            with_mag_columns=False)

            lightcurves.append_magnitude_columns(lc, exp_name, exp_err_name)

            self.assertIn(exp_name, lc.colnames)
            self.assertIn(exp_err_name, lc.colnames)
            self.assertAlmostEqual(exp_max_mag, lc[exp_name].max(), 3)


    #
    #   get_binned_phase_mags_data(flc, num_bins, phase_pivot) -> (phases, mags)
    #
    def test_get_binned_phase_mags_data_happy_path(self):
        """ Simple happy path test of get_binned_phase_mags_data() for binning """
        lc = lightcurve_helpers.load_lightcurves("CW Eri", [31], with_mag_columns=True)[0]

        t0 = lightcurve_helpers.KNOWN_TARGETS["CW Eri"]["t0"]
        period = lightcurve_helpers.KNOWN_TARGETS["CW Eri"]["period"]
        wrap_phase = u.Quantity(0.75)
        flc = lc.fold(period, t0, wrap_phase=wrap_phase, normalize_phase=True)

        exp_bins = 1024
        phases, mags = lightcurves.get_binned_phase_mags_data(flc, exp_bins, wrap_phase)

        self.assertEqual(exp_bins, len(phases))
        self.assertEqual(wrap_phase.value, phases.max())
        self.assertTrue(wrap_phase.value - 1 <= phases.min())
        self.assertFalse(any(np.isnan(mags)))
        self.assertAlmostEqual(flc["delta_mag"].min().unmasked.value, mags.min(), 1)

    def test_get_binned_phase_mags_data_wrap_phase(self):
        """ Test of get_binned_phase_mags_data() to assert handling of wrapped phase """
        lc = lightcurve_helpers.load_lightcurves("CW Eri", [31], with_mag_columns=True)[0]

        t0 = lightcurve_helpers.KNOWN_TARGETS["CW Eri"]["t0"]
        period = lightcurve_helpers.KNOWN_TARGETS["CW Eri"]["period"]

        for fold_wrap, bin_wrap, msg in [
            (u.Quantity(0.75), u.Quantity(0.75),    "Both equivalent Quantities"),
            (u.Quantity(0.75), 0.75,                "Equivalent Quantity and float"),
            (None, 0.5,                             "No fold, so defaults to 0.5"),
            (u.Quantity(0.75), None,                "Is 0.75 but not supplied so inferred"),
            (u.Quantity(0.90), None,                "Is 0.90 but not supplied so inferred"),
        ]:
            with self.subTest(msg=msg):
                flc = lc.fold(period, t0, wrap_phase=fold_wrap, normalize_phase=True)
                phases, _ = lightcurves.get_binned_phase_mags_data(flc, 2048, bin_wrap)
                self.assertAlmostEqual(flc.phase.max().value, phases.max(), 3)

    def test_get_binned_phase_mags_data_fill_gaps(self):
        """ Test of get_binned_phase_mags_data() to assert handling of gaps in source data """
        lc = lightcurve_helpers.load_lightcurves("CW Eri", [31], with_mag_columns=True)[0]

        t0 = lightcurve_helpers.KNOWN_TARGETS["CW Eri"]["t0"]
        period = lightcurve_helpers.KNOWN_TARGETS["CW Eri"]["period"]
        wrap_phase = u.Quantity(0.5)
        flc = lc.fold(period, t0, wrap_phase=wrap_phase, normalize_phase=True)

        for mask, msg in [
            (flc.phase > u.Quantity(-0.49),                             "missing section at start"),
            (flc.phase < u.Quantity(0.49),                              "missing section at end"),
            ((flc.phase < u.Quantity(0.0)) | (flc.phase > u.Quantity(0.05)), "missing section"),
        ]:
            with self.subTest(msg=msg):
                _, mags = lightcurves.get_binned_phase_mags_data(flc[mask], 2048, wrap_phase)
                self.assertFalse(any(np.isnan(mags)))

    #
    #   find_eclipses_and_completeness(lc, ref_t0, period, widthP, widthS, depthP, depthS, phis, max_phase_shift)
    #
    def test_find_eclipses_and_completeness_known_targets(self):
        """ Tests find_eclipses_and_completeness(known target) correctly finds and identifies eclipses """
        #  in lightcurve_helpers KNOWN_TARGETS & exp_prim, exp_sec are #eclipses >80% complete
        for (target,            sectors,    exp_prim,   exp_sec) in [
            # CW Eri has short period & many good eclipses although S31 ends with an incomplete sec (~70%)
            ("CW Eri",          [4, 31],    [7, 8],     [9, 9]),

            # RR Lyn/20 has incomplete secondary without peak after break; expect it found, with compl <50%
            ("RR Lyn",          [20],       [2],        [2]),

            # RR Lyn/60 has secondary with no fluxes falling within the mid-sector break (0% compl)
            ("RR Lyn",          [60],       [2],        [2]),

            # IT Cas/17; good test of boundary handling with good eclipses v. near start and mid-sector break
            ("IT Cas",          [17],       [5],        [5]),

            # 30034081/64 another boundary test as opens with incomplete sec with peak before start (~35% compl)
            ("TIC 30034081",    [64],       [4],        [4]),

            # 118313102/9 has pulsations which may trick find_peaks(). Without logic to mitigate for pulsations this
            # will find a false "good" primary in sector 9 (a pulsation) which is expected in the mid-sector interval.
            ("TIC 118313102",    [8, 9],     [3, 2],    [2, 2]),

            # 255567460 has no primaries and 2 secondaries
            ("TIC 255567460",   [66],       [0],        [2]),

            # These targets are for checking that pulsations/variation is not being incorrectly selected as eclipses
            # V889 Aql/40 possible false primary nr 2406 which should be in the mid-sector break (~2405) and 0% complete
            ("V889 Aql",        [40],       [2],        [2]),

            # 118313102/63 can find false primary nr 3029 which then leads to another in 65 nr 3077
            ("TIC 350298314",   [63, 65],   [0, 0],     [0, 1]),
        ]:
            completeness_th = 0.8
            target_cfg = lightcurve_helpers.KNOWN_TARGETS[target]
            tess_ebs = catalogues.query_tess_ebs_ephemeris(target_cfg["tic"]) or {}

            lcs = lightcurve_helpers.load_lightcurves(target, sectors)
            ret_vals = [lightcurves.find_eclipses_and_completeness(lc,
                                                                   target_cfg.get("t0", tess_ebs.get("t0", None)),
                                                                   target_cfg.get("period", tess_ebs.get("period", None)),
                                                                   target_cfg.get("widthP", tess_ebs.get("widthP", None)),
                                                                   target_cfg.get("widthS", tess_ebs.get("widthS",None)),
                                                                   target_cfg.get("depthP", tess_ebs.get("depthP", None)),
                                                                   target_cfg.get("depthS", tess_ebs.get("depthS", None)),
                                                                   target_cfg.get("phiS", tess_ebs.get("phiS", None)))
                        for lc in lcs]

            # self._plot_lcs_and_eclipses(lcs, ret_vals, completeness_th)

            for lc, (t0, pri_times, pri_compl, sec_times, sec_compl), exp_num_prim, exp_num_sec in zip(lcs, ret_vals, exp_prim, exp_sec):
                with self.subTest(lc.meta["LABEL"]):
                    if exp_num_prim > 0:
                        self.assertTrue(lc.time.value.min() <= t0 <= lc.time.value.max())
                    self.assertEqual(len(pri_times), len(pri_compl))
                    self.assertEqual(len(sec_times), len(sec_compl))
                    self.assertEqual(sum(pri_compl > completeness_th), exp_num_prim)
                    self.assertEqual(sum(sec_compl > completeness_th), exp_num_sec)


    def _plot_lcs_and_eclipses(self, lcs, data: List[tuple], completeness_th):
        """
        Plots lightcurves and corresponding output from get_eclipse_times_and_masks.
        """
        self.assertEqual(len(lcs), len(data))

        def plot_eclipses(ix, ax, _):
            ed = data[ix]
            ax.plot(ed[0], -0.1, marker="*", markersize=10, color="r")
            for times, compl, ls, c, label in [(ed[1], ed[2], "-.", "r", "primary"), (ed[3], ed[4], "--", "g", "secondary")]:
                alphas = [0.66 if c > completeness_th else 0.20 for c in compl]
                ax.vlines(times, -0.2, 1.0, c, ls, label, alpha=alphas, zorder=-20, transform=ax.get_xaxis_transform())
                for times, compl, a in zip(times, compl, alphas):
                    ax.text(times, 0, f"{compl:.0%}", c=c, rotation=90, alpha=a + 0.3,
                            va="center", ha="center", zorder=-10, backgroundcolor="w")

        plots.plot_lightcurves(lcs, cols=min(len(lcs), 3), column="delta_mag", ax_func=plot_eclipses, legend_loc="best")
        plt.show()
        plt.close()


if __name__ == "__main__":
    unittest.main()
