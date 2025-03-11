""" Unit tests for the pipeline module. """
from pathlib import Path
import re
import unittest

import lightkurve as lk

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

if __name__ == "__main__":
    unittest.main()
