""" Unit tests for the sed module. """
# pylint: disable=unused-import, too-many-public-methods, line-too-long, invalid-name, no-member
from inspect import getsourcefile
import warnings
from pathlib import Path
from shutil import copy
import unittest

import numpy as np
from scipy.optimize import minimize as scipy_minimize
from uncertainties import ufloat, UFloat, unumpy
import astropy.units as u
from astropy.units.errors import UnitConversionError
from astropy.table import Table, join

from libs.sed import get_sed_for_target, calculate_vfv, group_and_average_fluxes
from libs.sed import create_outliers_mask, blackbody_flux
from libs.sed import create_likelihood_func

class Testsed(unittest.TestCase):
    """ Unit tests for the sed module. """
    _this_dir = Path(getsourcefile(lambda:0)).parent
    _cache_dir = _this_dir / "../.cache/.sed/"
    _cm_dra_test_target = "testsed CM Dra"
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
        sed = get_sed_for_target(Testsed._cw_eri_test_target, verbose=True)
        self.assertIsNotNone(sed)
        self.assertTrue(isinstance(sed, Table))
        self.assertTrue(len(sed) > 0)
        self.assertIn("sed_flux", sed.colnames)     # These from source table
        self.assertIn("sed_eflux", sed.colnames)
        self.assertIn("sed_freq", sed.colnames)
        self.assertIn("sed_filter", sed.colnames)
        self.assertIn("sed_wl", sed.colnames)       # These apended once downloaded

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

    def test_get_sed_for_target_basic_happy_path_for_remove_duplicates(self):
        """ Tests get_sed_for_target() basic test for remove_duplicates functionality """
        sed = get_sed_for_target(Testsed._cm_dra_test_target)
        sed_dedupe = get_sed_for_target(Testsed._cm_dra_test_target,
                                        remove_duplicates=True, verbose=True)

        print(sed_dedupe["sed_filter", "sed_wl", "sed_flux", "sed_eflux"])

        self.assertIsNotNone(sed_dedupe)
        self.assertTrue(isinstance(sed_dedupe, Table))
        self.assertTrue(len(sed) > len(sed_dedupe))

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
    #   calculate_vfv(sed: Table, freq_colname="sed_freq"
    #                 flux_colname="sed_flux", flux_err_colname="sed_eflux") -> (Column, Column):
    #
    def test_calculate_vfv_simple_happy_path(self):
        """ Tests calculate_vfv() basic happy path test of calculations & units """
        sed = get_sed_for_target(Testsed._cw_eri_test_target)
        vfv, evfv = calculate_vfv(sed)

        self.assertListEqual((sed["sed_freq"] * sed["sed_flux"]).value.tolist(), vfv.value.tolist())
        self.assertListEqual((sed["sed_freq"] * sed["sed_eflux"]).value.tolist(), evfv.value.tolist())
        self.assertEqual(sed["sed_flux"].unit * sed["sed_freq"].unit, vfv.unit)
        self.assertEqual(sed["sed_eflux"].unit * sed["sed_freq"].unit, evfv.unit)

    def test_calculate_vfv_assert_can_add_columns(self):
        """ Tests calculate_vfv() ensure returned columns can be added to source sed Table """
        sed = get_sed_for_target(Testsed._cw_eri_test_target)
        sed["sed_vfv"], sed["sed_evfv"] = calculate_vfv(sed)

        self.assertIn("sed_vfv", sed.colnames)
        self.assertIn("sed_evfv", sed.colnames)


    #
    #   group_and_aggregate_sed(sed: Table):
    #
    def test_group_and_average_fluxes_simple_happy_path(self):
        """ Tests group_and_average_fluxes() basic happy path test of calculations & return type """
        sed = get_sed_for_target(Testsed._cw_eri_test_target)
        sed_grps = group_and_average_fluxes(sed, verbose=True)

        self.assertTrue(len(sed_grps) < len(sed))

        # Check the grouped mean(nom, err) calculation
        grp_mask = (sed_grps["sed_filter"] == "Gaia:G") & (sed_grps["sed_freq"] == 4.4546e14 * u.Hz)
        sed_mask = (sed["sed_filter"] == "Gaia:G") & (sed["sed_freq"] == 4.4546e14 * u.Hz)
        exp_mean = np.mean(unumpy.uarray(sed["sed_flux"][sed_mask], sed["sed_eflux"][sed_mask]))
        self.assertAlmostEqual(exp_mean.n, sed_grps["sed_flux"][grp_mask].value[0], 6)
        self.assertAlmostEqual(exp_mean.s, sed_grps["sed_eflux"][grp_mask].value[0], 6)


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
        for target,                         temps0,     min_unmasked,   msg in [
            (Testsed._cw_eri_test_target,   (6800, 6500),   15,     "test stops at no improvement"),
            (Testsed._zz_boo_test_target,   (6700, 6700),   125,    "test stops at min"),
            (Testsed._cm_dra_test_target,   (3200, 3200),   50,     "test stops as already at min"),
            (Testsed._cm_dra_test_target,   (3200, 3200),   0.99,   "test fractional min"),
        ]:
            with self.subTest(msg=msg):
                sed = get_sed_for_target(target)
                mask = create_outliers_mask(sed, temps0, min_unmasked, verbose=True)
                self.assertTrue(isinstance(mask, np.ndarray))
                self.assertTrue(mask.dtype == np.dtype(bool))
                self.assertEqual(len(sed), len(mask))

                if min_unmasked < 1:
                    min_unmasked = np.floor(len(sed) * min_unmasked)
                if min_unmasked < len(sed):
                    self.assertTrue(sum(~mask) >= min_unmasked)
                else:
                    self.assertEqual(sum(~mask), len(sed))

                print(f"{target}: Number of fluxes left: {sum(~mask)} of {len(sed)}")

    #
    #   create_likelihood_func(x: array, y: array, y_err: array,
    #                          model_func: Callable,
    #                          prior_func: Callable=null,
    #                          sim_func: 1/2 sum ((y_model-y)/y_err)**2)) -> func(theta) -> float:
    #
    def test_create_likelihood_func_simple_happy_path(self):
        """ Test create_likelihood_func(...) happy path > does it work & is it minimizable """
        sed = get_sed_for_target(Testsed._cw_eri_test_target)

        x = sed["sed_freq"].to(u.Hz).value
        y = sed["sed_flux"].to(u.Jy).value
        y_err = sed["sed_eflux"].to(u.Jy).value

        y_log = np.log10(y)
        def model_func(x, teffs): # sums blackbody spectra and scales them to y in log10 space
            y_mdl_log = np.log10(np.sum([blackbody_flux(x, t) for t in teffs], axis=0)) + 26 # to Jy
            return 10**(y_mdl_log + np.median(y_log - y_mdl_log))

        def prior_func(teffs):
            return all(6000 <= t <= 7000 for t in teffs) and abs(teffs[1]/teffs[0] - 0.9) <= 0.1

        # Create the target func - leave sim_func() to the default implementation
        target_func = create_likelihood_func(x, y, y_err, model_func, prior_func)

        # Minimize it
        with warnings.catch_warnings(category=RuntimeWarning):
            warnings.filterwarnings("ignore", message="invalid value encountered in subtract")
            soln = scipy_minimize(target_func, (6800, 6500))

        print(f"Final teffs = {soln.x}")
        self.assertTrue(6000 <= soln.x[0] <= 7000)
        self.assertTrue(6000 <= soln.x[1] <= 7000)



if __name__ == "__main__":
    unittest.main()
