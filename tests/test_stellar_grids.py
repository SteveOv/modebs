""" Unit tests for the stellar_grids classes. """
# pylint: disable=line-too-long
from inspect import getsourcefile
from pathlib import Path
import unittest

import numpy as np
import astropy.units as u

from libs.stellar_grids import BtSettlGrid

class TestBtSettlGrid(unittest.TestCase):
    """ Unit tests for the BtSettlGrid class. """
    _this_dir = Path(getsourcefile(lambda:0)).parent
    _test_file = _this_dir / "data/stellar_grids/bt-settl-agss-test.npz"

    @classmethod
    def setUpClass(cls):
        """
        Initialize the class.
        """
        if not cls._test_file.exists():
            # Ensure the .cache/.modelgrids/bt-settl-agss-test/ directory is
            # populated with bt-settl-agss grid files covering the following ranges;
            #   - teff: 4900 to 5100
            #   - logg: 3.5 to 5.0
            #   - metal: -0.5 to 0.5
            #   - alpha: 0.0 to 0.2
            # pylint: disable=protected-access
            in_files = (BtSettlGrid._CACHE_DIR / ".modelgrids/bt-settl-agss-test/").glob("lte*.dat.txt")
            BtSettlGrid.make_grid_file(in_files, cls._test_file)

    #
    #   __init__(data_file):
    #
    def test_init_default_data_file(self):
        """ Tests __init__() will pick up the default model file """
        model_grid = BtSettlGrid()
        self.assertGreaterEqual(model_grid.teff_range[0], 1000)
        self.assertLessEqual(model_grid.teff_range[1], 100000)
        self.assertGreaterEqual(model_grid.logg_range[0], -0.5)
        self.assertLessEqual(model_grid.logg_range[1], 6.0)
        self.assertGreaterEqual(model_grid.metal_range[0], -2.0)
        self.assertLessEqual(model_grid.metal_range[1], 2.0)

    def test_init_specific_data_file(self):
        """ Tests __init__(data_file) will load the specified model filt"""
        model_grid = BtSettlGrid(data_file=self._test_file)
        self.assertEqual(model_grid.teff_range[0], 4900)
        self.assertEqual(model_grid.teff_range[1], 5100)
        self.assertEqual(model_grid.logg_range[0], 3.5)
        self.assertEqual(model_grid.logg_range[1], 5.0)
        self.assertEqual(model_grid.metal_range[0], -0.5)
        self.assertEqual(model_grid.metal_range[1], 0.5)

    #
    #   has_filter(name) -> bool:
    #
    def test_has_filter_various_names(self):
        """ Tests has_filter(name) with various requests """
        model_grid = BtSettlGrid(self._test_file)

        for name,                           exp_response,   msg in [
            ("GAIA/GAIA3:Gbp",              True,           "test filter name is known"),
            ("GAIA/GAIA3:Gesso",            False,          "test filter name is unknown"),
            # Multiple
            (["GAIA/GAIA3:Gbp", "Gaia:G"],  [True, True],   "test filters, both known"),
            (["GAIA/GAIA3:Gbp", "Who?"],    [True, False],  "test filters, one unknown"),
            (["GAIA/GAIA3:Gbp"],            [True],         "test single known filter in list"),
            (np.array(["GAIA/GAIA3:Gbp"]),  [True],         "test single known filter in ndarray"),
            # Edge cases
            ("",                            False,          "test empty name is unknown"),
            (None,                          False,          "test None name is unknown"),
            (12,                            False,          "test non-str filter name is unknown"),
        ]:
            with self.subTest(msg=msg):
                response = model_grid.has_filter(name)
                self.assertIsInstance(response, np.ndarray)
                self.assertEqual(response.dtype, bool)

                # Call always returns a ndarray, but if single then can be treated like a bool
                if response.size == 1:
                    self.assertTrue(exp_response == response)
                else:
                    self.assertListEqual(exp_response, response.tolist())


    #
    #   get_filter_indices(filter_names) -> np.ndarray[int]:
    #
    def test_get_filter_indices_happy_path(self):
        """ Tests get_filter_indices() with simple happy path requests """
        model_grid = BtSettlGrid(self._test_file)

        for request, exp_response, msg in [
            ("GAIA/GAIA3:Gbp", [0], "test filter list is a single str with known SED service filter name"),
            (["GAIA/GAIA3:Gbp"], [0], "test filter list is a list holding a single str with known SED service filter name"),

            (["GAIA/GAIA3:G", "GAIA/GAIA3:Grp", "GAIA/GAIA3:Gbp"], [1, 2, 0], "test filter list with known SED service filter names"),
        ]:
            with self.subTest(msg=msg):
                response = model_grid.get_filter_indices(request)
                self.assertIsInstance(response, np.ndarray)
                self.assertListEqual(exp_response, response.tolist())

    def test_get_filter_indices_unknown_filter(self):
        """ Tests get_filter_indices() with unknown filters -> assert KeyError """
        model_sed = BtSettlGrid(self._test_file)
        for request, msg in [
            ("Unknown:filter", "test single str with unknown filter name"),
            (["Unknown:filter"], "test single item with unknown filter name"),
            (["Gaia:G", "Unknown:filter"], "test list containing known and unknown filter name"),
        ]:
            with self.subTest(msg=msg) and self.assertRaises(ValueError):
                model_sed.get_filter_indices(request)

    #
    #   get_filter_fluxes(filters, teff, logg, metal=0, radius=None, distance=None) -> np.ndarray[float] or u.Quantity:
    #
    def test_get_filter_fluxes_happy_path_no_interp(self):
        """ Tests get_filter_fluxes() with happy path requests with exact row match (no interp fluxes) """
        model_sed = BtSettlGrid(self._test_file)
        model_interps = model_sed._model_interps    # pylint: disable=protected-access

        for filters,                 teff,  logg,   metal,  msg in [
            ("GAIA/GAIA3:Gbp",      5000,   4.0,    0.0,    "test single str filter"),
            (["GAIA/GAIA3:Gbp"],    5000,   4.0,    0.0,    "test single list[str] filter"),
            (["Gaia:G", "GAIA/GAIA3:Grp", "GAIA/GAIA3:Gbp"],
                                    5000,   4.0,    0.0,    "test filters in different order to file cols"),
            ("GAIA/GAIA3:Gbp",      5100,   4.0,    0.0,    "test different teff"),
            ("GAIA/GAIA3:Gbp",      5000,   4.5,    0.0,    "test different logg"),
            ("GAIA/GAIA3:Gbp",      5000,   4.0,    0.3,    "test different metal"), # not currently in grid
        ]:
            with self.subTest(msg=msg):
                fluxes = model_sed.get_filter_fluxes(filters, teff, logg, metal)

                xi = (teff, logg, metal, 0)
                filter_list = model_sed.get_filter_indices([filters] if isinstance(filters, str) else filters)
                exp_fluxes = [model_interps[f]["interp"](xi=xi) for f in filter_list]

                self.assertIsInstance(fluxes, np.ndarray)
                self.assertListEqual(exp_fluxes, fluxes.tolist())

    def test_get_filter_fluxes_happy_path_interp(self):
        """ Tests get_filter_fluxes() with happy path requests which require linear interpolation """
        model_sed = BtSettlGrid(self._test_file)
        model_interps = model_sed._model_interps    # pylint: disable=protected-access

        for filters,                teff,   logg,   metal,  msg in [
            ("GAIA/GAIA3:Gbp",      5050,   4.0,    0.0,    "test linear interpolation on teff"),
            ("GAIA/GAIA3:Gbp",      5000,   4.25,   0.0,    "test linear interpolation on logg"),
            ("GAIA/GAIA3:Gbp",      5000,   4.0,    0.15,   "test linear interpolation on metal"), # not currently in grid
        ]:
            with self.subTest(msg=msg):
                fluxes = model_sed.get_filter_fluxes(filters, teff, logg, metal)

                xi = (teff, logg, metal, 0)
                filter_list = model_sed.get_filter_indices([filters] if isinstance(filters, str) else filters)
                exp_fluxes = [model_interps[f]["interp"](xi=xi) for f in filter_list]

                self.assertIsInstance(fluxes, np.ndarray)
                self.assertListEqual(exp_fluxes, fluxes.tolist())

    def test_get_filter_fluxes_unknown_filter_name(self):
        """ Tests get_filter_fluxes() with unknown filter names -> assert KeyError """
        model_sed = BtSettlGrid(self._test_file)

        kwargs = { "teff": 5000, "logg": 4.0, "metal": 0.0 }
        for filters, msg in [
            ("Unknwon:Fitler", "test single unknown filter name"),
            (["Gaia:G", "Unknwon:Fitler", "GAIA/GAIA3:Gbp"], "test list with one unknown filter"),
        ]:
            with self.subTest(msg=msg):
                with self.assertRaises(ValueError):
                    model_sed.get_filter_fluxes(filters, **kwargs)

    def test_get_filter_fluxes_unknown_filter_index(self):
        """ Tests get_filter_fluxes() with unknown filter indices -> assert IndexError """
        model_sed = BtSettlGrid(self._test_file)

        kwargs = { "teff": 5000, "logg": 4.0, "metal": 0.0 }
        for filters, msg in [
            (500, "test single unknown filter index"),
            ([1, 500, 0], "test list with one unknown filter index"),
        ]:
            with self.subTest(msg=msg):
                with self.assertRaises(IndexError):
                    model_sed.get_filter_fluxes(filters, **kwargs)

    def test_get_filter_fluxes_stellar_params_out_of_range(self):
        """ Tests get_filter_fluxes() with Teff, logg or metal outside the model's range -> ValueError """
        model_sed = BtSettlGrid(self._test_file)

        for teff, logg, metal, msg in [
            (1000, 4.0, 0.0, "test Teff out of range"),
            (5000, 7.0, 0.0, "test logg out of range"),
            (5000, 4.0, 2.0, "test metal out of range"),
        ]:
            with self.subTest(msg=msg):
                with self.assertRaises(ValueError):
                    model_sed.get_filter_fluxes("Gaia:Gbp", teff, logg, metal)


if __name__ == "__main__":
    unittest.main()
