""" Unit tests for the jktebop module. """
import os
import io
import unittest
from pathlib import Path
from subprocess import CalledProcessError, TimeoutExpired

from uncertainties import ufloat, UFloat
import numpy as np
from astropy.time import Time

import tests.helpers.lightcurve_helpers as th

from libs import jktebop
from libs.jktebop import _prepare_params_for_task
from libs.jktebop import get_jktebop_dir
from libs.jktebop import run_jktebop_task, generate_model_light_curve
from libs.jktebop import write_in_file, write_light_curve_to_dat_file
from libs.jktebop import JktebopParameterWarning

# pylint: disable=invalid-name, too-many-public-methods, line-too-long, protected-access
class Testjktebop(unittest.TestCase):
    """ Unit tests for the jktebop module. """
    _prefix = "test_jktebop_"
    _task2_params = { # set of valid param/tokens & values for task 2
        "ring": 2,
        "rA_plus_rB": 0.3,  "k": ufloat(0.5, 0.0),
        "inc": 90.,         "qphot": 0.5,
        "ecosw": 0.,        "esinw": 0.,
        "gravA": 0.,        "gravB": 0.,
        "J": 0.8,           "L3": 0.,
        "LDA": "quad",      "LDB": "quad",
        "LDA1": 0.25,       "LDB1": 0.25,
        "LDA2": 0.22,       "LDB2": 0.22,
        "reflA": 0.,        "reflB": 0.,
    }

    _task3_params = { # set of valid param/tokens & values for task 3
        "ring": 3,
        "rA_plus_rB": 0.3,  "k": ufloat(0.5, 0.0),
        "inc": 90.,         "qphot": 0.5,
        "ecosw": 0.,        "esinw": 0.,
        "gravA": 0.,        "gravB": 0.,
        "J": 0.8,           "L3": 0.,
        "LDA": "quad",      "LDB": "quad",
        "LDA1": 0.25,       "LDB1": 0.25,
        "LDA2": 0.22,       "LDB2": 0.22,
        "reflA": 0.,        "reflB": 0.,
        "period": 2.5,
        "primary_epoch": 59876.54321,
        "qphot_fit": 1,
        "ecosw_fit": 1,     "esinw_fit": 1,
        "gravA_fit": 0,     "gravB_fit": 0,
                            "L3_fit": 1,
        "LDA1_fit": 1,      "LDB1_fit": 1,
        "LDA2_fit": 0,      "LDB2_fit": 0,
        "reflA_fit": 1,     "reflB_fit": 1,
                            "sf_fit": 1,
        "period_fit": 1,    "primary_epoch_fit": 1, 

        "data_file_name": "cw_eri_s0004.dat"
    }

    @classmethod
    def setUpClass(cls):
        """ Make sure JKTEBOP_DIR is corrected up as tests may modify it. """
        jktebop._jktebop_directory = Path(os.environ.get("JKTEBOP_DIR", "~/jktebop/")).expanduser().absolute()
        jktebop._jktebop_support_negative_l3 = os.environ.get("JKTEBOP_SUPPORT_NEG_L3", "") == "True"
        th.TEST_DATA_DIR.mkdir(parents=True, exist_ok=True)
        return super().setUpClass()

    @classmethod
    def tearDownClass(cls):
        """ Make sure JKTEBOP_DIR is corrected up as tests may modify it. """
        jktebop._jktebop_directory = Path(os.environ.get("JKTEBOP_DIR", "~/jktebop/")).expanduser().absolute()
        jktebop._jktebop_support_negative_l3 = os.environ.get("JKTEBOP_SUPPORT_NEG_L3", "") == "True"
        return super().tearDownClass()

    #
    # Tests get_jktebop_dir()
    #
    def test_get_jktebop_dir_assert_value(self):
        """ Calls get_jktebop_dir() and assert its value. """
        jdir = get_jktebop_dir()
        exp_value = Path(os.environ.get("JKTEBOP_DIR", "~/jktebop/")).expanduser().absolute()
        self.assertEqual(jdir, exp_value)


    #
    # Tests run_jktebop_task(in_filename, out_filename, delete_files_pattern)
    #
    def test_run_jktebop_task_valid(self):
        """ Test run_jktebop_task(all necessary params) generates model """
        # Need to create the in file first
        in_filename = get_jktebop_dir() / "test_run_jktebop_task_valid.2.in"
        out_filename = get_jktebop_dir() / f"{in_filename.stem}.out"
        params = { **self._task2_params.copy(), "out_filename": f"{out_filename.name}" }
        write_in_file(in_filename, 2, None, **params)

        # Outfile content
        out_content = list(run_jktebop_task(in_filename, out_filename, f"{in_filename.stem}.*"))
        model = np.loadtxt(out_content, usecols=(0, 1), comments="#", dtype=np.double, unpack=True)
        self.assertEqual(model.shape[0], 2) # columns
        self.assertTrue(model.shape[1] > 0) # rows

        # In and out files deleted
        self.assertFalse(in_filename.exists())
        self.assertFalse(out_filename.exists())

    def test_run_jktebop_task_in_file_not_found(self):
        """ Test run_jktebop_task(in_file doesn't exist) raises CalledProcessError """
        # Don't create the in_filename
        in_filename = get_jktebop_dir() / "test_run_jktebop_task_unknown.2.in"
        out_filename = get_jktebop_dir() / f"{in_filename.stem}.out"

        # self.assertRaises() not working for this. Instead assert by catching CalledProcessError
        try:
            next(run_jktebop_task(in_filename, out_filename, f"{in_filename.stem}.*"))
            self.fail("Should not get here")
        except CalledProcessError:
            pass

    def test_run_jktebop_task_stdout_to_TextIO(self):
        """ Test run_jktebop_task(with stdout_to set) assert JKTEBOP console captured """
        # Need to create the in file first
        in_filename = get_jktebop_dir() / "test_run_jktebop_task_ext_valid.2.in"
        out_filename = get_jktebop_dir() / f"{in_filename.stem}.out"
        params = { **self._task2_params.copy(), "out_filename": f"{out_filename.name}" }
        write_in_file(in_filename, 2, None, **params)

        # Run while redirecting JKTEBOP's stdout then assert we have caught its output
        capture = io.StringIO()
        list(jktebop.run_jktebop_task(in_filename, out_filename, f"{in_filename.stem}.*", capture))
        self.assertIn("JKTEBOP", capture.getvalue())

    def test_run_jktebop_task_timeout(self):
        """ Test run_jktebop_task(with timeout set) assert timeout invoked """
        # Need to create the in file first
        in_filename = get_jktebop_dir() / "test_run_jktebop_task_ext_valid.2.in"
        out_filename = get_jktebop_dir() / f"{in_filename.stem}.out"
        params = { **self._task2_params.copy(), "out_filename": f"{out_filename.name}" }
        write_in_file(in_filename, 2, None, **params)

        with self.assertRaises(expected_exception=TimeoutExpired):
            list(jktebop.run_jktebop_task(in_filename, out_filename, f"{in_filename.stem}.*",
                                          timeout=0.001))


    #
    # Tests generate_model_light_curve(file_prefix: str,
    #                                  **params) -> np.ndarray
    #
    def test_generate_model_light_curve_args_none(self):
        """ Test generate_model_light_curve(arguments None) raises TypeError """
        self.assertRaises(TypeError, generate_model_light_curve, None, {"rA":2})
        self.assertRaises(TypeError, generate_model_light_curve, self._prefix, None)

    def test_generate_model_light_curve_missing_params(self):
        """ Test generate_model_light_curve(missing params) raises TypeError """
        params = {"rA": 2 } # No mandatory rA_plus_rB params (among many others)
        self.assertRaises(KeyError, generate_model_light_curve, self._prefix, **params)

    def test_generate_model_light_curve_jktebop_error(self):
        """ Test generate_model_light_curve(jktebop fails) raises CalledProcessError """
        params = self._task2_params.copy()
        params["ecosw"] = None
        self.assertRaises(CalledProcessError, generate_model_light_curve, self._prefix, **params)

    @unittest.skip("skip on full run as can cause parallel tests to fail")
    def test_generate_model_light_curve_env_variable_incorrect(self):
        """ Test generate_model_light_curve(JKTEBOP_DIR points to wrong loc) ignores others """
        params = self._task2_params.copy()
        jktebop._jktebop_directory = Path.home()
        self.assertRaises(FileNotFoundError, generate_model_light_curve, self._prefix, **params)
        jktebop._jktebop_directory = Path(os.environ.get("JKTEBOP_DIR", "~/jktebop")).expanduser().absolute()

    def test_generate_model_light_curve_data_dtype(self):
        """ Test generate_model_light_curve(all necessary params) generates model data """
        params = self._task2_params.copy()
        model = generate_model_light_curve(self._prefix, **params)
        self.assertIn("phase", model.dtype.names)
        self.assertIn("delta_mag", model.dtype.names)

    def test_generate_model_light_curve_valid_params_only(self):
        """ Test generate_model_light_curve(all necessary params) generates model """
        params = self._task2_params.copy()
        model = generate_model_light_curve(self._prefix, **params)
        self.assertIsNotNone(model)
        self.assertTrue(model.shape[0] > 0)                 # rows
        self.assertEqual(len(list(model.dtype.names)), 2)   # columns

    def test_generate_model_light_curve_valid_params_plus_extras(self):
        """ Test generate_model_light_curve(all necessary params + others) ignores others """
        params = self._task2_params.copy()
        params["another_param_to_be_ignores"] = "anything or nothing"
        model = generate_model_light_curve(self._prefix, **params)
        self.assertIsNotNone(model)
        self.assertTrue(model.shape[0] > 0)                 # rows


    #
    # TESTS write_in_file(file_name, task, [append_lines], **params)
    #
    def test_write_in_file_args_none_or_wrong_type(self):
        """ Test write_in_file(wrong file_name type) raises TypeError """
        self.assertRaises(TypeError, write_in_file, None, 2)
        self.assertRaises(TypeError, write_in_file, get_jktebop_dir() / "valid.in", None)
        self.assertRaises(TypeError, write_in_file, "hello", 2)
        self.assertRaises(TypeError, write_in_file, get_jktebop_dir() / "valid.in", "Task2")

    def test_write_in_file_task_is_unknown(self):
        """ Test write_in_file(task is unknown) raises KeyError """
        self.assertRaises(KeyError, write_in_file, get_jktebop_dir() / "valid.in", 1)

    def test_write_in_file_missing_params(self):
        """ Test write_in_file(missing template params) raises KeyError """
        file_name = th.TEST_DATA_DIR / "any_old_file_will_do.dat"
        self.assertRaises(KeyError, write_in_file, file_name, task=3, k=0.5)

    def test_write_in_file_param_validation_warnings(self):
        """ Testwrite_in_file(some out of range LD param values) raises Warning """
        file_name = th.TEST_DATA_DIR / "any_old_file_will_do.dat"
        for param,              value,      coerced_value in [
            ("rA_plus_rB",      0.99,       0.8),
            ("k",               0.0099,     0.01),
            ("k",               100.1,      100),
            ("J",               0.00099,    0.001),
            ("J",               1000.1,     1000),
            ("inc",             49,         50),
            ("inc",             141,        140),
            ("L3",              -2,         -1 if jktebop._jktebop_support_negative_l3 else 0),
            ("L3",              10.01,      10),
        ]:
            with self.subTest(param):
                params = self._task3_params.copy()
                params[param] = value
                match = f"{param}(?:.*)coerced to(?:.*){coerced_value}"
                with self.assertWarnsRegex(JktebopParameterWarning, match,
                                        msg=f"Expected a {JktebopParameterWarning} to be raised"):
                    write_in_file(file_name, 3, None, **params)

    def test_write_in_file_ld_validation_warnings(self):
        """ Testwrite_in_file(some out of range LD param values) raises Warning """
        file_name = th.TEST_DATA_DIR / "any_old_file_will_do.dat"

        for star in ["A", "B"]:
            for algo, test_values in {
                "quad": [(-1.1, -1.0), (2.1, 2.0)],
                "pow2": [(-1.1, -1.0), (2.1, 2.0)],
                "4par": [(-9.1, -9.0), (9.1, 9.0)]
            }.items():
                for value, exp_value in test_values:
                    for coeff_ix in range(1, 5 if algo == "4par" else 3):
                        params = self._task3_params.copy()
                        params[f"LD{star}"] = algo
                        params[f"LD{star}{coeff_ix}"] = value
                        match = f"LD{star}{coeff_ix}(?:.*)coerced to(?:.*){exp_value}"
                        with self.assertWarnsRegex(JktebopParameterWarning, match,
                                            msg=f"Expected {JktebopParameterWarning} to be raised"):
                            write_in_file(file_name, 3, None, **params)

    def test_write_in_file_full_set_of_task2_params(self):
        """ Test write_in_file(full set of task2 template params) asserts file is written """
        file_stem = "test_write_in_file_full_set_of_params.2"
        file_name = th.TEST_DATA_DIR / f"{file_stem}.in"
        params = { **self._task2_params.copy(), "out_filename": f"{file_name.stem}.out" }
        write_in_file(file_name, task=2, **params)

        with open(file_name, "r", encoding="utf8") as inf:
            text = inf.read()
            self.assertIn(params["LDB"], text)
            self.assertIn(params["out_filename"], text)
            self.assertIn(file_stem, text)

    def test_write_in_file_full_set_of_task3_params(self):
        """ Test write_in_file(full set of task3 template params) asserts file is written """
        file_stem = "test_write_in_file_full_set_of_params.3"
        file_name = th.TEST_DATA_DIR / f"{file_stem}.in"
        params = { **self._task3_params.copy(), "data_file_name": f"{file_name.stem}.dat"}
        write_in_file(file_name, task=3, **params)

        with open(file_name, "r", encoding="utf8") as inf:
            text = inf.read()
            self.assertIn(params["LDB"], text)
            self.assertIn(params["data_file_name"], text)
            self.assertIn(file_stem, text)

    def test_write_in_file_append_lines(self):
        """ Test write_in_file(with append_lines) asserts they are written """
        file_name = th.TEST_DATA_DIR / "test_write_task3_in_file_append_lines.3.in"
        append_lines = [ "line 1\n\n", "\n\n\nline 2", "line 3" ]

        write_in_file(file_name, 3, append_lines, **self._task3_params)
        with open(file_name, "r", encoding="utf8") as inf:
            text = inf.read()
            for line in append_lines:
                self.assertIn(line.strip(), text)


    #
    # TESTS build_poly_instructions(file_name, task, [append_lines], **params)
    #
    def test_build_poly_instructions_assert_format(self):
        """ tests build_poly_instructions((from, to), term, degree) -> asserts poly written """
        times = [
            (Time(1000.009, format="btjd", scale="tdb"), Time(1099.991, format="btjd", scale="tdb"))
        ]
        polies = jktebop.build_poly_instructions(time_ranges=times, term="sf", degree=2)
        self.assertEqual(len(polies), len(times))
        self.assertIn("sf", polies[0])
        self.assertIn("1 1 1 0 0 0 ", polies[0])    # from the degree; fit const, 1st deg, 2nd deg
        self.assertIn("1000.0", polies[0])          # start rounded down to 2 d.p.
        self.assertIn("1100.0", polies[0])          # end rounded up to 2 d.p.
        self.assertIn("1050.0", polies[0])          # mean of start and end (after rounding)

    def test_build_poly_instructions_assert_count(self):
        """ tests build_poly_instructions((from, to), term, degree) -> asserts correct number """
        times = [
            (Time(1000, format="btjd", scale="tdb"), Time(1100, format="btjd", scale="tdb")),
            (Time(1200, format="btjd", scale="tdb"), Time(1300, format="btjd", scale="tdb")),
        ]
        polies = jktebop.build_poly_instructions(time_ranges=times, term="sf", degree=2)
        self.assertEqual(len(polies), len(times))


    #
    # TESTS write_light_curve_to_dat_file(lc, file_name, [column_names], [column_formats])
    #
    def test_write_light_curve_to_dat_file_args_none_or_wrong_type(self):
        """ Test write_light_curve_to_dat_file(arguments wrong type) raises TypeError """
        lc = th.load_lightcurve("CW Eri")
        file_name = th.TEST_DATA_DIR / "any_old_file_will_do.dat"
        self.assertRaises(TypeError, write_light_curve_to_dat_file, None,   file_name)
        self.assertRaises(TypeError, write_light_curve_to_dat_file, lc,     None)

        lc_array = [lc["time"].value, lc["delta_mag"].value, lc["delta_mag_err"].value]
        self.assertRaises(TypeError, write_light_curve_to_dat_file, lc_array, file_name)
        self.assertRaises(TypeError, write_light_curve_to_dat_file, lc, f"{file_name}")

    def test_write_light_curve_to_dat_file_column_args_not_matching(self):
        """ Test write_light_curve_to_dat_file(column args numbers mismatch) raises ValueError """
        lc = th.load_lightcurve("CW Eri")
        file_name = th.TEST_DATA_DIR / "any_old_file_will_do.dat"
        self.assertRaises(ValueError, write_light_curve_to_dat_file, lc, file_name, ["time"], None)
        self.assertRaises(ValueError, write_light_curve_to_dat_file, lc, file_name, ["time"], ["%.6f", "%.6f"])
        self.assertRaises(ValueError, write_light_curve_to_dat_file, lc, file_name, None, ["%.6f"])
        self.assertRaises(ValueError, write_light_curve_to_dat_file, lc, file_name, ["time", "delta_mag"], ["%.6f"])

    def test_write_light_curve_to_dat_file_default_columns(self):
        """ Test write_light_curve_to_dat_file(using default column & formats)  writes file"""
        lc = th.load_lightcurve("CW Eri")
        file_name = th.TEST_DATA_DIR / "test_write_light_curve_to_dat_file_default_columns.dat"

        write_light_curve_to_dat_file(lc, file_name)

        data = np.loadtxt(file_name, comments="#", delimiter=" ", unpack=True)
        self.assertEqual(data.shape[0], 3)
        self.assertEqual(data.shape[1], len(lc))
        self.assertAlmostEqual(data[0][750], float(lc.iloc[750]["time"].value), 6)
        self.assertAlmostEqual(data[1][1000], float(lc.iloc[1000]["delta_mag"].value), 6)

    def test_write_light_curve_to_dat_file_explicit_columns(self):
        """ Test write_light_curve_to_dat_file(explicit columns & formats) writes file """
        lc = th.load_lightcurve("CW Eri")
        file_name = th.TEST_DATA_DIR / "test_write_light_curve_to_dat_file_explicit_columns.dat"

        write_light_curve_to_dat_file(lc,
                                      file_name,
                                      ["time", "sap_flux"],
                                      [lambda t: f"{t.jd:.3f}", "%.3f"])

        data = np.loadtxt(file_name, comments="#", delimiter=" ", unpack=True)
        self.assertEqual(data.shape[0], 2)
        self.assertEqual(data.shape[1], len(lc))
        self.assertAlmostEqual(data[0][250], float(lc.iloc[250]["time"].jd), 3)
        self.assertAlmostEqual(data[1][500], float(lc.iloc[500]["sap_flux"].value), 3)


    #
    # Tests (private) _prepare_params_for_task(task: int,
    #                                          params: dict,
    #                                          [fit_rA_and_rB] = False,
    #                                          [fit_e_and_omega] = False,
    #                                          [calc_refl_coeffs] = False,
    #                                          [in_place] = False)
    #                                               -> [None or dict]
    #
    def test__prepare_params_for_task_args_none(self):
        """ Test _prepare_params_for_task(arguments None) raises TypeError """        
        self.assertRaises(TypeError, _prepare_params_for_task, task=None, params={"rA":2})
        self.assertRaises(TypeError, _prepare_params_for_task, task=2, params=None)

    def test__prepare_params_for_task_fit_flags_with_related_params_missing(self):
        """ Test _prepare_params_for_task(fit_*=True and related params missing) raises KeyError """
        # Setting fit_rA_and_rB==True depends on the rA and rB params
        self.assertRaises(KeyError, _prepare_params_for_task, task=2, fit_rA_and_rB=True,  params={"rB":2})
        self.assertRaises(KeyError, _prepare_params_for_task, task=2, fit_rA_and_rB=True,  params={"rA":2})
        # Setting fit_e_and_omega==True depends on the e and omega params
        self.assertRaises(KeyError, _prepare_params_for_task, task=2, fit_e_and_omega=True,  params={"omega":90})
        self.assertRaises(KeyError, _prepare_params_for_task, task=2, fit_e_and_omega=True,  params={"e":0.1})

    def test__prepare_params_for_task_fit_rA_and_rB_true(self):
        """ Test _prepare_params_for_task(fit_rA_and_rB=True) check params correctly updated """
        params = { "rA": 2, "rB": 1, "rA_plus_rB": 3, "k": 0.5 }
        params = _prepare_params_for_task(2, params, fit_rA_and_rB=True)
        self.assertEqual(params["rA_plus_rB"], -params["rA"])
        self.assertEqual(params["k"], params["rB"])

    def test__prepare_params_for_task_fit_e_and_omega_true(self):
        """ Test _prepare_params_for_task(fit_e_and_omega=True) check params correctly updated """
        params = { "ecosw": 0., "esinw": 0., "e": 0.1, "omega": 90 }
        params = _prepare_params_for_task(2, params, fit_e_and_omega=True)
        self.assertEqual(params["ecosw"], 10+params["e"])
        self.assertEqual(params["esinw"], params["omega"])

    def test__prepare_params_for_task_task2_calc_refl_coeffs_true(self):
        """ Test _prepare_params_for_task(task=2, calc_refl_coeffs=True) check params correctly updated """
        params = _prepare_params_for_task(2, { "reflA": 0, "reflB": 0 }, calc_refl_coeffs=True)
        self.assertEqual(params["reflA"], -100)
        self.assertEqual(params["reflB"], -100)
        params = _prepare_params_for_task(2, { }, calc_refl_coeffs=True)
        self.assertTrue("reflA" in params)
        self.assertEqual(params["reflA"], -100)
        self.assertTrue("reflB" in params)
        self.assertEqual(params["reflB"], -100)

    def test__prepare_params_for_task_task3_calc_refl_coeffs_true(self):
        """ Test _prepare_params_for_task(task=3, calc_refl_coeffs=True) check ignored """
        params = _prepare_params_for_task(3, { "reflA": 0, "reflB": 0 }, calc_refl_coeffs=True)
        self.assertEqual(params["reflA"], 0)
        self.assertEqual(params["reflB"], 0)

    def test__prepare_params_for_task_assert_iterations_default(self):
        """ Test _prepare_params_for_task(task=3 | 8 | 9, no simulations param) check correct default """
        for task, exp_sims in [(3, ""), (8, 1), (9, 1)]:
            sims = _prepare_params_for_task(task, { })["simulations"]
            self.assertEqual(sims, exp_sims, f"expected default simulations token for task {task}")

    def test__prepare_params_for_task_assert_iterations(self):
        """ Test _prepare_params_for_task(task=3 | 8 | 9, { "simulations": x }) check not overriden """
        for task, exp_sims in [(3, "# nothing"), (8, 800), (9, 900)]:
            sims = _prepare_params_for_task(task, { "simulations": exp_sims })["simulations"]
            self.assertEqual(sims, exp_sims, f"expected simulations token for task {task}")

    def test__prepare_params_for_task_with_floats(self):
        """ Test _prepare_params_for_task(params as floats (separate sigma values)) values passed on """
        # Requesting a fit on rA & rB causes the rA_plus_rB & k params to be
        # changed which is how we detect if the params arg has been modified.
        params = { "rA_plus_rB": 3, "k": 0.5, "rA_plus_rB_sigma": 0.1, "k_sigma": 0.1 }
        result = _prepare_params_for_task(2, params)
        for key, value in params.items():
            self.assertIn(key, result.keys())
            self.assertEqual(value, result[key])

    def test__prepare_params_for_task_with_ufloats(self):
        """ Test _prepare_params_for_task(params as ufloats) nominal values passed on as floats """
        # Requesting a fit on rA & rB causes the rA_plus_rB & k params to be
        # changed which is how we detect if the params arg has been modified.
        params = { "rA_plus_rB": ufloat(3, 0.1), "k": ufloat(0.5, 0.1), "J": 1.0 }
        result = _prepare_params_for_task(2, params)
        for key, value in params.items():
            self.assertIn(key, result.keys())
            self.assertNotIsInstance(result[key], UFloat)
            if isinstance(value, UFloat):
                self.assertEqual(result[key], value.nominal_value)
            else:
                self.assertEqual(result[key], value)


if __name__ == "__main__":
    unittest.main()
