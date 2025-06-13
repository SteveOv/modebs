""" Unit tests for the sed module. """
# pylint: disable=unused-import, too-many-public-methods, line-too-long, invalid-name, no-member
import unittest

import astropy.units as u
from astropy.units.errors import UnitConversionError
from astropy.table import Table

from libs.sed import get_sed_for_target

class Testsed(unittest.TestCase):
    """ Unit tests for the sed module. """

    #
    #   get_sed_for_target(target: str,
    #                      search_term: str,
    #                      radius: float=0.1,
    #                      missing_uncertainty_ratio: float=0.1,
    #                      flux_units=u.W / u.m**2 / u.Hz,
    #                      freq_units=u.Hz,
    #                      wavelength_units=u.micron)
    #
    def test_get_sed_for_target_simple_happy_path(self):
        """ Tests get_sed_for_target() basic happy path test for known sed """
        sed = get_sed_for_target("CW Eri", "V* CW Eri")
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
                sed = get_sed_for_target("CW Eri", "V* CW Eri",
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

if __name__ == "__main__":
    unittest.main()
