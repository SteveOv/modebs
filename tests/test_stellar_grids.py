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

    #
    #   __init__(data_file):
    #
    def test_init_happy_path(self):
        """ Tests __init__() basic happy path test for ModelSed instance initialization """
        for data_file, msg in [
            (self._this_dir/"../libs/data/stellar_grids/bt-settl-agss/bt-settl-agss.npz", "tests __init__ loads specified data file"),
            (None, "test __init__ falls back on the default npz file under libs/data/stellar_grids/bt-settl-agss/"),
        ]:
            with self.subTest(msg=msg):
                model_grid = BtSettlGrid(data_file=data_file)

                self.assertEqual("bt-settl-agss.npz", model_grid.data_file.name)

                self.assertTrue(model_grid.num_interpolators > 0)
                self.assertGreaterEqual(model_grid.teff_range[0], 1000 << u.K)
                self.assertLessEqual(model_grid.teff_range[1], 70000 << u.K, )
                self.assertGreaterEqual(model_grid.logg_range[0], -0.5 << u.dex)
                self.assertLessEqual(model_grid.logg_range[1], 6.0 << u.dex)
                self.assertGreaterEqual(model_grid.metal_range[0], -2.0 << u.dimensionless_unscaled)
                self.assertLessEqual(model_grid.metal_range[1], 0.5 << u.dimensionless_unscaled)


    #
    #   has_filter(name) -> bool:
    #
    def test_has_filter_various_names(self):
        """ Tests has_filter(name) with various requests """
        model_grid = BtSettlGrid()

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
                if len(response) == 1:
                    self.assertTrue(exp_response == response)
                else:
                    self.assertListEqual(exp_response, response.tolist())


    #
    #   get_filter_indices(filter_names) -> np.ndarray[int]:
    #
    def test_get_filter_indices_happy_path(self):
        """ Tests get_filter_indices() with simple happy path requests """
        model_grid = BtSettlGrid()

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
        model_sed = BtSettlGrid()
        for request, msg in [
            ("Unknown:filter", "test single str with unknown filter name"),
            (["Unknown:filter"], "test single item with unknown filter name"),
            (["Gaia:G", "Unknown:filter"], "test list containing known and unknown filter name"),
        ]:
            with self.subTest(msg=msg) and self.assertRaises(ValueError):
                model_sed.get_filter_indices(request)

    #
    #   get_fluxes(filters, teff, logg, metal=0) -> np.ndarray[float] or u.Quantity:
    #
    def test_get_fluxes_happy_path_no_interp(self):
        """ Tests get_fluxes() with happy path requests with exact row match (no interp fluxes) """
        model_sed = BtSettlGrid()
        model_interps = model_sed._model_interps    # pylint: disable=protected-access

        for filters,                 teff,  logg,   metal,  msg in [
            ("GAIA/GAIA3:Gbp",      5000,   4.0,    0.0,    "test single str filter"),
            (["GAIA/GAIA3:Gbp"],    5000,   4.0,    0.0,    "test single list[str] filter"),
            (["Gaia:G", "GAIA/GAIA3:Grp", "GAIA/GAIA3:Gbp"],
                                    5000,   4.0,    0.0,    "test filters in different order to file cols"),
            ("GAIA/GAIA3:Gbp",      5100,   4.0,    0.0,    "test different teff"),
            ("GAIA/GAIA3:Gbp",      5000,   4.5,    0.0,    "test different logg"),
            # ("GAIA/GAIA3:Gbp",      5000,   4.0,    0.3,    "test different metal"), # not currently in grid
        ]:
            with self.subTest(msg=msg):
                fluxes = model_sed.get_filter_fluxes(filters, teff, logg, metal, as_quantity=False)

                xi = (teff, logg, metal)
                filter_list = model_sed.get_filter_indices([filters] if isinstance(filters, str) else filters)
                exp_fluxes = [model_interps[f]["interp"](xi=xi) for f in filter_list]

                self.assertIsInstance(fluxes, np.ndarray)
                self.assertListEqual(exp_fluxes, fluxes.tolist())

    def test_get_fluxes_happy_path_interp(self):
        """ Tests get_fluxes() with happy path requests which require linear interpolation """
        model_sed = BtSettlGrid()
        model_interps = model_sed._model_interps    # pylint: disable=protected-access

        for filters,                teff,   logg,   metal,  msg in [
            ("GAIA/GAIA3:Gbp",      5050,   4.0,    0.0,    "test linear interpolation on teff"),
            ("GAIA/GAIA3:Gbp",      5000,   4.25,   0.0,    "test linear interpolation on logg"),
            # ("GAIA/GAIA3:Gbp",      5000,   4.0,    0.15,   "test linear interpolation on metal"), # not currently in grid
        ]:
            with self.subTest(msg=msg):
                fluxes = model_sed.get_filter_fluxes(filters, teff, logg, metal, as_quantity=False)

                xi = (teff, logg, metal)
                filter_list = model_sed.get_filter_indices([filters] if isinstance(filters, str) else filters)
                exp_fluxes = [model_interps[f]["interp"](xi=xi) for f in filter_list]

                self.assertIsInstance(fluxes, np.ndarray)
                self.assertListEqual(exp_fluxes, fluxes.tolist())

    def test_get_fluxes_happy_path_as_quantity(self):
        """ Tests get_fluxes() with happy path requests excersing the as_quantity arg """
        model_sed = BtSettlGrid()

        for filters,                                as_quantity, msg in [
            ("GAIA/GAIA3:Gbp",                      True, "single filter / as quantity"),
            (["GAIA/GAIA3:Gbp", "GAIA/GAIA3:Grp"],  True, "multipls filters / as quantity"),
            ("GAIA/GAIA3:Gbp",                      False, "single filter / as value"),
            (["GAIA/GAIA3:Gbp", "GAIA/GAIA3:Grp"],  False, "multipls filters / as value"),
        ]:
            with self.subTest(msg=msg):
                fluxes = model_sed.get_filter_fluxes(filters, 4000, 4.0, 0.0, as_quantity)
                self.assertEqual(as_quantity, isinstance(fluxes, u.Quantity))

    def test_get_fluxes_unknown_filter_name(self):
        """ Tests get_fluxes() with unknown filter names -> assert KeyError """
        model_sed = BtSettlGrid()

        kwargs = { "teff": 5000, "logg": 4.0, "metal": 0.0 }
        for filters, msg in [
            ("Unknwon:Fitler", "test single unknown filter name"),
            (["Gaia:G", "Unknwon:Fitler", "GAIA/GAIA3:Gbp"], "test list with one unknown filter"),
        ]:
            with self.subTest(msg=msg):
                with self.assertRaises(ValueError):
                    model_sed.get_filter_fluxes(filters, **kwargs)

    def test_get_fluxes_unknown_filter_index(self):
        """ Tests get_fluxes() with unknown filter indices -> assert IndexError """
        model_sed = BtSettlGrid()

        kwargs = { "teff": 5000, "logg": 4.0, "metal": 0.0 }
        for filters, msg in [
            (500, "test single unknown filter index"),
            ([1, 500, 0], "test list with one unknown filter index"),
        ]:
            with self.subTest(msg=msg):
                with self.assertRaises(IndexError):
                    model_sed.get_filter_fluxes(filters, **kwargs)

    def test_get_fluxes_stellar_params_out_of_range(self):
        """ Tests get_fluxes() with Teff, logg or metal outside the model's range -> ValueError """
        model_sed = BtSettlGrid()

        for teff, logg, metal, msg in [
            (1000, 4.0, 0.0, "test Teff out of range"),
            (5000, 7.0, 0.0, "test logg out of range"),
            (5000, 4.0, 2.0, "test metal out of range"),
        ]:
            with self.subTest(msg=msg):
                with self.assertRaises(ValueError):
                    model_sed.get_filter_fluxes("Gaia:Gbp", teff, logg, metal)
