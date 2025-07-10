""" Unit tests for the pyssed module. """
# pylint: disable=line-too-long
from inspect import getsourcefile
from pathlib import Path
import unittest

import numpy as np
import astropy.units as u

from libs.pyssed import ModelSed

class TestModelSed(unittest.TestCase):
    """ Unit tests for the pyssed module. """
    _this_dir = Path(getsourcefile(lambda:0)).parent

    #
    #   __init__(data_file):
    #
    def test_init_happy_path(self):
        """ Tests __init__() basic happy path test for ModelSed instance initialization """
        for data_file, msg in [
            (self._this_dir/"../libs/data/pyssed/model-bt-settl-recast.dat", "tests __init__ loafs specified data file"),
            (None, "test __init__ falls back on the default dat file under libs/data/pyssed/"),
        ]:
            with self.subTest(msg=msg):
                model_sed = ModelSed(data_file=data_file)

                self.assertEqual("model-bt-settl-recast.dat", model_sed.data_file.name)

                self.assertTrue(model_sed.num_interpolators > 0)
                self.assertEqual(1900 << u.K, model_sed.teff_range[0])
                self.assertEqual(70000 << u.K, model_sed.teff_range[1])
                self.assertEqual(-0.5 << u.dex, model_sed.logg_range[0])
                self.assertEqual(6.0 << u.dex, model_sed.logg_range[1])
                self.assertEqual(-2.0 << u.dimensionless_unscaled, model_sed.metal_range[0])
                self.assertEqual(0.5 << u.dimensionless_unscaled, model_sed.metal_range[1])


    #
    #   get_filter_indices(filter_names) -> np.ndarray[int]:
    #
    def test_get_filter_indices_happy_path(self):
        """ Tests get_filter_indices() with simple happy path requests """
        model_sed = ModelSed()

        for request, exp_response, msg in [
            ("GAIA/GAIA3.Gbp", [0], "test filter list is a single str with known dat filter name"),
            (["GAIA/GAIA3.Gbp"], [0], "test filter list is a list holding a single str with known dat filter name"),

            ("Gaia:Gbp", [0], "test filter list is a single str with known SED service filter name"),
            (["Gaia:Gbp"], [0], "test filter list is a list holding a single str with known SED service filter name"),

            (["GAIA/GAIA3.G", "GAIA/GAIA3.Grp", "GAIA/GAIA3.Gbp"], [1, 2, 0], "test filter list with known dat filter names"),
            (["Gaia:G", "Gaia:Grp", "Gaia:Gbp"], [1, 2, 0], "test filter list with known SED service filter names"),
        ]:
            with self.subTest(msg=msg):
                response = model_sed.get_filter_indices(request)
                self.assertIsInstance(response, np.ndarray)
                self.assertListEqual(exp_response, response.tolist())

    def test_get_filter_indices_unknown_filter(self):
        """ Tests get_filter_indices() with unknown filters -> assert KeyError """
        model_sed = ModelSed()
        for request, msg in [
            ("Unknown:filter", "test single str with unknown filter name"),
            (["Unknown:filter"], "test single item with unknown filter name"),
            (["Gaia:G", "Unknown:filter"], "test list containing known and unknown filter name"),
        ]:
            with self.subTest(msg=msg) and self.assertRaises(KeyError):
                model_sed.get_filter_indices(request)

    #
    #   get_fluxes(filters, teff, logg, metal=0) -> np.ndarray[float] * Jy:
    #
    # These tests read from model-bt-settl-recast.dat file using the follow lines and cols/filters
    # line: Teff, logg, metal, alpha    Gbp         G           Grp
    # 6212: 5000, 4.0,  0.0,   0.0      3.333e18    4.729e18    6.273e18
    # 6214: 5000, 4.0,  0.3,   0.0      3.270e18    4.705e18    6.294e18
    # 6226: 5000, 4.5,  0.0,   0.0      3.300e18    4.733e18    6.323e18
    # 6408: 5100, 4.0,  0.0,   0.0      3.747e18    5.198e18    6.788e18
    #
    def test_get_fluxes_happy_path_no_interp(self):
        """ Tests get_fluxes() with happy path requests with exact row match (no interp fluxes) """
        model_sed = ModelSed()

        for filters, teff, logg, metal, exp_response, msg in [
            ("Gaia:Gbp", 5000, 4.0, 0.0, [3.333e18], "test single str filter"),
            (["Gaia:Gbp"], 5000, 4.0, 0.0, [3.333e18], "test single list[str] filter"),
            (["Gaia:G", "Gaia:Grp", "Gaia:Gbp"], 5000, 4.0, 0.0, [4.729e18, 6.273e18, 3.333e18],
                                                    "test filters in different order to file cols"),
            ("Gaia:Gbp", 5100, 4.0, 0.0, [3.747e18], "test different teff"),
            ("Gaia:Gbp", 5000, 4.5, 0.0, [3.300e18], "test different logg"),
            ("Gaia:Gbp", 5000, 4.0, 0.3, [3.270e18], "test different metal"),
        ]:
            with self.subTest(msg=msg):
                fluxes = model_sed.get_fluxes(filters, teff, logg, metal)
                self.assertIsInstance(fluxes, np.ndarray)
                self.assertListEqual(exp_response, fluxes.value.tolist())

    def test_get_fluxes_happy_path_interp(self):
        """ Tests get_fluxes() with happy path requests which require linear interpolation """
        model_sed = ModelSed()

        for filters, teff, logg, metal, exp_response, msg in [
            ("Gaia:Gbp", 5050, 4.0, 0.0, [np.median([3.333e18, 3.747e18])],
                                                            "test linear interpolation on teff"),
            ("Gaia:Gbp", 5000, 4.25, 0.0, [np.median([3.333e18, 3.300e18])],
                                                            "test linear interpolation on logg"),
            ("Gaia:Gbp", 5000, 4.0, 0.15, [np.median([3.333e18, 3.270e18])],
                                                            "test linear interpolation on metal"),
        ]:
            with self.subTest(msg=msg):
                fluxes = model_sed.get_fluxes(filters, teff, logg, metal)
                self.assertIsInstance(fluxes, np.ndarray)
                self.assertListEqual(exp_response, fluxes.value.tolist())

    def test_get_fluxes_unknown_filter_name(self):
        """ Tests get_fluxes() with unknown filter names -> assert KeyError """
        model_sed = ModelSed()

        kwargs = { "teff": 5000, "logg": 4.0, "metal": 0.0 }
        for filters, msg in [
            ("Unknwon:Fitler", "test single unknown filter name"),
            (["Gaia:G", "Unknwon:Fitler", "Gaia:Gbp"], "test list with one unknown filter"),
        ]:
            with self.subTest(msg=msg):
                with self.assertRaises(KeyError):
                    model_sed.get_fluxes(filters, **kwargs)

    def test_get_fluxes_unknown_filter_index(self):
        """ Tests get_fluxes() with unknown filter indices -> assert IndexError """
        model_sed = ModelSed()

        kwargs = { "teff": 5000, "logg": 4.0, "metal": 0.0 }
        for filters, msg in [
            (500, "test single unknown filter index"),
            ([1, 500, 0], "test list with one unknown filter index"),
        ]:
            with self.subTest(msg=msg):
                with self.assertRaises(IndexError):
                    model_sed.get_fluxes(filters, **kwargs)

    def test_get_fluxes_stellar_params_out_of_range(self):
        """ Tests get_fluxes() with Teff, logg or metal outside the model's range -> ValueError """
        model_sed = ModelSed()

        for teff, logg, metal, msg in [
            (1000, 4.0, 0.0, "test Teff out of range"),
            (5000, 7.0, 0.0, "test logg out of range"),
            (5000, 4.0, 2.0, "test metal out of range"),
        ]:
            with self.subTest(msg=msg):
                with self.assertRaises(ValueError):
                    model_sed.get_fluxes("Gaia:Gbp", teff, logg, metal)
