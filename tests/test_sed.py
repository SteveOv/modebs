""" Unit tests for the sed module. """
# pylint: disable=unused-import, too-many-public-methods, line-too-long, invalid-name, no-member
from inspect import getsourcefile
from pathlib import Path
from shutil import copy
import unittest

import numpy as np
from uncertainties import ufloat, UFloat
import astropy.units as u
from astropy.units.errors import UnitConversionError
from astropy.table import Table

from libs.sed import get_sed_for_target, create_outliers_mask
from libs.sed import blackbody_flux

class Testsed(unittest.TestCase):
    """ Unit tests for the sed module. """
    _this_dir = Path(getsourcefile(lambda:0)).parent
    _cache_dir = _this_dir / "../.cache/.sed/"
    _cw_dra_test_target = "testsed CM Dra"
    _cm_dra_test_file = _cache_dir / "testsed-cm-dra-0.1.vot"
    _zz_boo_test_target = "testsed ZZ Boo"
    _zz_boo_test_file = _cache_dir / "testsed-zz-boo-0.1.vot"
    _cw_eri_test_target = "testsed CW Eri"
    _cw_eri_test_file = _cache_dir / "testsed-cw-eri-0.1.vot"

    @classmethod
    def setUpClass(cls):
        # Copy to the cache some SED tables which we used for tests (avoids failed downloads)
        Testsed._cache_dir.mkdir(parents=True, exist_ok=True)
        copy(Testsed._this_dir / "data/sed/cm-dra-0.1.vot", Testsed._cm_dra_test_file)
        copy(Testsed._this_dir / "data/sed/zz-boo-0.1.vot", Testsed._zz_boo_test_file)
        copy(Testsed._this_dir / "data/sed/cw-eri-0.1.vot", Testsed._cw_eri_test_file)

    @classmethod
    def tearDownClass(cls):
        for testsed in Testsed._cache_dir.glob("testsed-*.*"):
            testsed.unlink(missing_ok=True)


    #
    #   get_sed_for_target(target: str,
    #                      search_term: str,
    #                      radius: float=0.1,
    #                      missing_uncertainty_ratio: float=0.1,
    #                      flux_units=u.W / u.m**2 / u.Hz,
    #                      freq_units=u.Hz,
    #                      wavelength_units=u.micron) -> Table:
    #
    def test_get_sed_for_target_simple_happy_path(self):
        """ Tests get_sed_for_target() basic happy path test for known sed """
        sed = get_sed_for_target(Testsed._cw_eri_test_target)
        self.assertIsNotNone(sed)
        self.assertTrue(isinstance(sed, Table))
        self.assertTrue(len(sed) > 0)
        self.assertIn("sed_flux", sed.colnames)     # These from source table
        self.assertIn("sed_eflux", sed.colnames)
        self.assertIn("sed_freq", sed.colnames)
        self.assertIn("sed_filter", sed.colnames)
        self.assertIn("sed_wl", sed.colnames)       # These added once downloaded
        self.assertIn("sed_vfv", sed.colnames)
        self.assertIn("sed_evfv", sed.colnames)

    def test_get_sed_for_target_assert_units(self):
        """ Tests get_sed_for_target() tests requested units are reflected in resulting table """
        for flux_unit, freq_unit, wl_unit in [
            (u.Jy, u.GHz, u.Angstrom),          # Jy & GHz are the default units as downloaded
            (u.W/u.m**2/u.Hz, u.Hz, u.micron),  # Default units expected (requires conversion)
        ]:
            with self.subTest():
                sed = get_sed_for_target(Testsed._cw_eri_test_target,
                                         flux_unit=flux_unit, freq_unit=freq_unit, wl_unit=wl_unit)
                self.assertEqual(sed["sed_flux"].unit, flux_unit)
                self.assertEqual(sed["sed_eflux"].unit, flux_unit)
                self.assertEqual(sed["sed_freq"].unit, freq_unit)
                self.assertEqual(sed["sed_wl"].unit, wl_unit)
                self.assertEqual(sed["sed_vfv"].unit, freq_unit * flux_unit)
                self.assertEqual(sed["sed_evfv"].unit, freq_unit * flux_unit)

    def test_get_sed_for_target_handle_invalid_unit(self):
        """ Tests get_sed_for_target() asserts UnitConversionError when a unit is incompatible """
        for unit_kwargs in [
            { "flux_unit": u.Jy / u.sr },
            { "freq_unit": u.micron },
            { "wl_unit": u.K },
        ]:
            with self.subTest():
                with self.assertRaises(UnitConversionError,
                                    msg=f"Expected **{unit_kwargs} to cause a UnitConversionError"):
                    get_sed_for_target("CW Eri", "V* CW Eri", **unit_kwargs)

    def test_get_sed_for_target_handle_unknown_target(self):
        """ Tests get_sed_for_target() asserts correct ValueError when no match on target """
        target, search_term = "Unknown", "UN Kno"
        with self.assertRaisesRegex(ValueError, f"search_term={search_term}",
                                    msg=f"Expected search_term={search_term} to cause ValueError"):
            get_sed_for_target(target, search_term)


    #
    #   blackbody_flux(freq, teff, radius=1.):
    #
    def test_blackbody_flux_assert_arg_types(self):
        """ Test blackbody_flux() assert the response type is appropriate for the inputs """
        for freq,                           teff,               exp_type,       exp_inner_type in [
            (7e14,                          5000,               float,          None),
            (ufloat(7e14, 0),               5000,               UFloat,         None),
            (7e14,                          ufloat(5000, 0),    UFloat,         None),
            (ufloat(7e14, 0),               ufloat(5000, 0),    UFloat,         None),
            (np.array([7e14]),              5000,               np.ndarray,     float),
            (np.array([ufloat(7e14, 0)]),   5000,               np.ndarray,     UFloat),
            (np.array([7e14]),              ufloat(5000, 0),    np.ndarray,     UFloat),
            (np.array([ufloat(7e14, 0)]),   ufloat(5000, 0),    np.ndarray,     UFloat),
        ]:
            with self.subTest():
                flux = blackbody_flux(freq, teff)
                self.assertIsInstance(flux, exp_type)
                if isinstance(flux, np.ndarray) and exp_inner_type is not None:
                    self.assertIsInstance(flux[0], exp_inner_type)

    def test_blackbody_flux_assert_calculation(self):
        """ Test blackbody_flux() assert the calculation is correct """
        #   Hz              K           arcsec      W / m^2 / Hz
        for freq,           teff,       radius,     exp_flux,       places in [
            (7e14,          5000.,      1.,         9.0322e-19,     23),
            (7e14,          10000.,     1.,         2.6892e-17,     21),
            (1e15,          10000.,     1.,         1.8083e-17,     21),
            (1e12,          3000.,      1.,         1.3503e-22,     26),
            (1e12,          3000.,      10.,        1.3503e-20,     24)
        ]:
            with self.subTest():
                flux = blackbody_flux(freq, teff, radius)
                self.assertAlmostEqual(flux, exp_flux, places)


    #
    #   create_outliers_mask(sed: Table) -> np.ndarray[bool]
    #
    def test_create_outliers_mask_simple_happy_path(self):
        """ Test create_outliers_mask(sed) WIP """
        sed = get_sed_for_target(Testsed._cw_eri_test_target)
        mask = create_outliers_mask(sed, teff_ratio=0.9)
        self.assertTrue(isinstance(mask, np.ndarray))
        self.assertTrue(mask.dtype == np.dtype(bool))
        self.assertEqual(len(sed), len(mask))

if __name__ == "__main__":
    unittest.main()
