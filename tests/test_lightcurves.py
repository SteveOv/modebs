""" Unit tests for the pipeline module. """
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

from tests.helpers import lightcurve_helpers

from libs import lightcurves


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
    #   get_lightcurve_t0_time(lc: LightCurve, t0: Time, period: u.d, max_phase_shift: float) -> float
    #
    def test_get_lightcurve_t0_time_rr_lyn(self):
        """ Tests get_lightcurve_t0_time(RR Lyn) which has a distinct shift on later sectors. """
        # Ephemeris for RR Lyn in TESS-ebs.  OK for S20 but there's significant shift by S60 & 73
        target = "RR Lyn"
        sectors = [20, 60, 73]
        t0 = ufloat(1851.9277371299124, 0.0006761397339687)     # btjd
        period = ufloat(9.946591112938657, 0.0005646471183542)  # d
        exp_revised_t0s = [1851.93, 2945.89, 3293.96]           # btjd

        lcs = lightcurve_helpers.load_lightcurves(target, sectors)
        for lc, exp_revised_t0 in zip(lcs, exp_revised_t0s):
            with self.subTest(f"Testing {target} sector {lc.meta['SECTOR']}"):
                revised_t0 = lightcurves.get_lightcurve_t0_time(lc, t0.n, period.n)
                self.assertAlmostEqual(revised_t0, exp_revised_t0, 2)

    def test_get_lightcurve_t0_time_no_primaries(self):
        """ Tests get_lightcurve_t0_time(LC with no primaries) assert return estimated t0 & warn """
        target = "TIC 255567460"
        sectors = [66]
        t0 = 1469.20871                         # btjd
        period = 13.79633                       # d
        exp_revised_t0 = t0 + (period * 119)    # btjd

        lc = lightcurve_helpers.load_lightcurves(target, sectors)[0]

        with self.assertWarns(UserWarning):
            revised_t0 = lightcurves.get_lightcurve_t0_time(lc, t0, period)
            self.assertEqual(revised_t0, exp_revised_t0)

if __name__ == "__main__":
    unittest.main()
