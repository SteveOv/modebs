""" Unit tests for the pipeline module. """
from typing import List, Dict
from pathlib import Path
import warnings
import re
import unittest

# pylint: disable=no-member, wrong-import-position, line-too-long
warnings.filterwarnings("ignore", "Using UFloat objects with std_dev==0 may give unexpected results.", category=UserWarning)
from uncertainties import ufloat
import numpy as np
import astropy.units as u
import lightkurve as lk
import matplotlib.pyplot as plt

from tests.helpers import lightcurve_helpers
from libs import lightcurves, catalogues, plots


# pylint: disable=too-many-public-methods, line-too-long
class Testlightcurves(unittest.TestCase):
    """ Unit tests for the lightcurves module. """

    def test_happy_path_tess_mission(self):
        """ Simple happy path test of load_lightcurves() while reading through an explicit cache dir """
        target = "CW Eri"
        results = lk.search_lightcurve(target, exptime="short", mission="TESS", author="SPOC")

        download_dir = Path.cwd() / ".cache" / re.sub(r"[^\w\d]", "-", target.lower())

        lcs = lightcurves.load_lightcurves(results, "hardest", ["sap_flux", "pdcsap_flux"], download_dir)
        self.assertEqual(len(lcs), 2)
        self.assertEqual(lcs[0].meta["FLUX_ORIGIN"], "sap_flux")
        self.assertEqual(lcs[1].meta["FLUX_ORIGIN"], "pdcsap_flux")

    def test_happy_path_default_cache(self):
        """ Simple happy path test of load_lightcurves() while reading through the default lk cache """
        target = "CW Eri"
        results = lk.search_lightcurve(target, exptime="short", mission="TESS", author="SPOC")

        # Fits files should be cached under ~/.lightkurve/cache
        lcs = lightcurves.load_lightcurves(results, "hardest", "sap_flux")
        self.assertEqual(len(lcs), 2)


    #
    #   get_binned_phase_mags_data(flc, num_bins, phase_pivot) -> (phases, mags)
    #
    def test_get_binned_phase_mags_data_happy_path(self):
        """ Simple happy path test of get_binned_phase_mags_data() for binning """
        lc = lightcurve_helpers.load_lightcurves("CW Eri", [31], with_mag_columns=True)[0]

        t0 = lightcurve_helpers.KNOWN_TARGETS["CW Eri"]["epoch_time"]
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

        t0 = lightcurve_helpers.KNOWN_TARGETS["CW Eri"]["epoch_time"]
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

        t0 = lightcurve_helpers.KNOWN_TARGETS["CW Eri"]["epoch_time"]
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
    #   get_eclipse_times_and_masks(lc, ref_t0, period, durp, durs, phis, max_phase_shift)
    #
    def test_find_eclipses_and_completeness_happy(self):
        """ Tests find_eclipses_and_completeness() correctly finds and identifies eclipses """
        #  in lightcurve_helpers KNOWN_TARGETS & exp_prim, exp_sec are #ecl >80% complete
        for (target,            sectors,    ref_t0,     phis,   exp_prim,   exp_sec) in [
            ("CW Eri",          [4, 31],    None,       None,   [7, 8],     [9, 9]),# easy although S31 ends with an incomplete sec
            ("RR Lyn",          [20],       None,       None,   [2],        [2]),   # S20 partial sec without peak after break; expect 0.4 < compl < 0.5
            ("RR Lyn",          [60],       3224.555,   None,   [2],        [2]),   # ref_t0 after Sector & a sec within break
            ("IT Cas",          [17],       None,       0.552,  [5],        [5]),   # all sectors complete but some near breaks
            ("TIC 255567460",   [66],       None,       None,   [0],        [2]),   # no primaries and 2 secondaries
        ]:
            target_cfg = lightcurve_helpers.KNOWN_TARGETS[target]
            tess_ebs = catalogues.query_tess_ebs_ephemeris(target_cfg["tic"])

            lcs = lightcurve_helpers.load_lightcurves(target, sectors)
            ecl_dicts = [lightcurves.find_eclipses_and_completeness(lc,
                                                                    ref_t0 or target_cfg["epoch_time"],
                                                                    target_cfg["period"],
                                                                    tess_ebs["durP"],
                                                                    tess_ebs["durS"],
                                                                    phis or tess_ebs["phiS"]) for lc in lcs]

            # self._plot_lcs_and_eclipses(lcs, ecl_dicts)

            for lc, ed, exp_num_prim, exp_num_sec in zip(lcs, ecl_dicts, exp_prim, exp_sec):
                with self.subTest(lc.meta["LABEL"]):
                    self.assertEqual(len(ed["primary_times"]), len(ed["primary_completeness"]))
                    self.assertEqual(len(ed["secondary_times"]), len(ed["secondary_completeness"]))
                    self.assertEqual(sum(ed["primary_completeness"] > 0.8), exp_num_prim)
                    self.assertEqual(sum(ed["secondary_completeness"] > 0.8), exp_num_sec)


    def _plot_lcs_and_eclipses(self, lcs, eclipse_dicts: List[Dict]):
        """
        Plots lightcurves and corresponding output from get_eclipse_times_and_masks.
        """
        self.assertEqual(len(lcs), len(eclipse_dicts))

        def plot_eclipses(ix, ax, _):
            ed = eclipse_dicts[ix]
            for x, completeness, ls, c, label in [
                (ed["primary_times"], ed["primary_completeness"], "-.", "r", "primary"),
                (ed["secondary_times"], ed["secondary_completeness"], "--", "g", "secondary")
            ]:
                ax.vlines(x, -0.2, 1.0, c, ls, label, alpha=0.33, zorder=-20, transform=ax.get_xaxis_transform())
                for x, compl in zip(x, completeness):
                    ax.text(x, 0, f"{compl:.0%}", c=c, rotation=90, va="center", ha="center", zorder=-10, backgroundcolor="w")

        plots.plot_lightcurves(lcs, cols=min(len(lcs), 3), column="delta_mag", ax_func=plot_eclipses, legend_loc="best")
        plt.show()
        plt.close()


if __name__ == "__main__":
    unittest.main()
