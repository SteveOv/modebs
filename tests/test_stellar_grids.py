""" Unit tests for the stellar_grids classes. """
# pylint: disable=line-too-long, protected-access, no-member, too-many-locals, too-many-arguments
from inspect import getsourcefile
from pathlib import Path
import unittest

import numpy as np
from dust_extinction.parameter_averages import G23
import astropy.units as u

from libs.stellar_grids import BtSettlGrid

class TestBtSettlGrid(unittest.TestCase):
    """ Unit tests for the BtSettlGrid class. """
    _this_dir = Path(getsourcefile(lambda:0)).parent
    _test_file = _this_dir / "data/stellar_grids/bt-settl-agss-test.npz"
    _flat_file = _this_dir / "data/stellar_grids/bt-settl-aggs-test-flat-file.npz"
    _flat_grid: np.ndarray
    _known_filters: dict

    @classmethod
    def setUpClass(cls):
        """
        Initialize the class.
        """
        # To recreate these files ensure the .cache/.modelgrids/bt-settl-agss-test/ directory
        # is populated with bt-settl-agss grid files covering the following ranges;
        #   - teff: 4900 to 5100
        #   - logg: 3.5 to 5.0
        #   - metal: -0.5 to 0.5
        #   - alpha: 0.0 to 0.2
        in_files = sorted((BtSettlGrid._CACHE_DIR / ".modelgrids/bt-settl-agss-test/").glob("lte*.dat.txt"))

        # This is a BtSettlGrid model file with a limited range of test teff, logg, metal & alpha values
        if not cls._test_file.exists():
            print(f"{cls.__name__}.setUpClass() test data file '{cls._test_file}' not found so will attempt to create it")
            BtSettlGrid.make_grid_file(in_files, cls._test_file)
            print("Done")

        # Equivalent to the above data file but the tests' own simple 2d grid. We get expected vals
        # from this so tests don't depend on, & can detect failures with, the class/file under test.
        if not cls._flat_file.exists():
            print(f"{cls.__name__}.setUpClass() test flat file '{cls._flat_file}' not found so will attempt to create it")
            grid_full_nbins = len(BtSettlGrid( cls._test_file).wavelengths)
            grid_full_bin_lams = np.geomspace(0.05, 50, num=grid_full_nbins, endpoint=True) << u.um
            grid_full_bin_freqs = grid_full_bin_lams.to(u.Hz, equivalencies=u.spectral())

            # Create a simple, flat file from the same data for expected values
            index_names = ["teff", "logg", "metal", "alpha"]
            index_values = BtSettlGrid._get_list_of_index_values(in_files, index_names, False)
            print(f"Ingesting {len(index_values)} files and will save fluxes in {grid_full_nbins} bins")

            flat_grid = np.full((len(index_values), len(index_names) + grid_full_nbins), np.nan, float)
            flat_grid[..., :len(index_names)] = index_values.tolist()

            for in_file in in_files:
                meta = BtSettlGrid._read_metadata_from_ascii_model_file(in_file)

                row_ix = np.where((index_values["teff"] == meta["teff"])\
                                & (index_values["logg"] == meta["logg"])\
                                & (index_values["metal"] == meta["metal"])\
                                & (index_values["alpha"] == meta["alpha"]))

                lams, flux_densities = np.genfromtxt(in_file, float, comments="#", unpack=True)
                lams = (lams * meta["lambda_unit"]).to(BtSettlGrid._LAM_UNIT, equivalencies=u.spectral())
                flux_densities = (flux_densities * meta["flux_unit"]).to(BtSettlGrid._FLUX_DENSITY_UNIT,
                                                                         equivalencies=u.spectral_density(lams))

                # Write the row of binned fluxes to the full grid.
                bin_flux_densities = BtSettlGrid._bin_fluxes(lams, flux_densities, grid_full_bin_lams)
                bin_fluxes = (bin_flux_densities *  grid_full_bin_freqs).value
                flat_grid[row_ix, len(index_names):] = bin_fluxes

            # Save the subset of the filters in the file too
            filter_map = { "GAIA/GAIA3:Gbp": "GAIA/GAIA3.Gbp", "GAIA/GAIA3:Grp": "GAIA/GAIA3.Grp", "Gaia:G": "GAIA/GAIA0.G" }
            filters = { viz: BtSettlGrid.get_filter(svo, BtSettlGrid._LAM_UNIT) for viz, svo in filter_map.items() }

            cls._flat_file.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(cls._flat_file, flat_grid=flat_grid, lams=grid_full_bin_lams.value, filters=filters)
            print("Done")

        with np.load(cls._flat_file, allow_pickle=True) as ff:
            cls._flat_grid = ff["flat_grid"]
            cls._flat_lams = ff["lams"]
            cls._known_filters = ff["filters"].item()

    @classmethod
    def _get_flux_value_from_flat_grid(cls, teff, logg, metal, alpha=0, flux_ix=None):
        """
        Gets the requested flux value matching the teff, logg, metal & alpha from the flat copy of
        the model grid. This only picks up values in the grid; no interpolation. By going direct to
        a flat table of fluxes we avoid any reshaping & other processing in the class being tested.
        """
        row_mask = (cls._flat_grid[..., 0] == teff) \
                    & (cls._flat_grid[..., 1] == logg) \
                    & (cls._flat_grid[..., 2] == metal) \
                    & (cls._flat_grid[..., 3] == alpha)
        if flux_ix is None:
            return cls._flat_grid[row_mask, 4:][0]
        return cls._flat_grid[row_mask, 4 + flux_ix][0]

    @classmethod
    def _calculate_filtered_flux_values(cls, filters, lams, fluxes):
        """
        Calculates the total flux transmitted by the requested filter
        """
        if isinstance(filters, str):
            filters = [filters]
        filter_fluxes = np.zeros((len(filters)), float)
        for ix, filter_name in enumerate(filters):
            ftable = cls._known_filters[filter_name]
            filter_lam = ftable["Wavelength"].value
            filter_trans = ftable["Norm-Transmission"].value
            filter_fluxes[ix] = np.sum(np.interp(filter_lam, lams, fluxes) * filter_trans)
        return filter_fluxes

    @classmethod
    def _get_filter_flux_values_from_flat_grid(cls, filters, teff, logg, metal, alpha=0):
        """
        Gets the total transmitted flux for each filter for the set of fluxes matching requested
        teff, logg, metal & alpha. Again, only picks up flux values in the grid; no interpolation.
        Only really works for filters entirely within lambda range as no overlap logic.
        """
        # Get the whole row of fluxes and lams
        fluxes = cls._get_flux_value_from_flat_grid(teff, logg, metal, alpha)
        lams = cls._flat_lams
        return cls._calculate_filtered_flux_values(filters, lams, fluxes)


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
    def test_get_filter_indices_known_filters(self):
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

    def test_get_filter_indices_unknown_filters(self):
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
    #   get_fluxes(teff, logg, metal=0, radius=None, distance=None, av=None) -> NDArray[float]
    #
    def test_get_fluxes_no_reddening(self):
        """ Tests get_fluxes(teff, logg, metal, radius, dist) happy path tests for combinations of values"""
        model_sed = BtSettlGrid(self._test_file)

        # Known values
        t5000_l40_m00 = self._get_flux_value_from_flat_grid(5000, 4.0, 0.0, flux_ix=1000)
        t5100_l40_m00 = self._get_flux_value_from_flat_grid(5100, 4.0, 0.0, flux_ix=1000)
        t5000_l45_m00 = self._get_flux_value_from_flat_grid(5000, 4.5, 0.0, flux_ix=1000)
        t5000_l40_m03 = self._get_flux_value_from_flat_grid(5000, 4.0, 0.3, flux_ix=1000)

        t5500_l40_m00 = (t5000_l40_m00 + t5100_l40_m00) / 2     # Approx interpolated values
        t5000_l425_m00 = (t5000_l40_m00 + t5000_l45_m00) / 2
        t5000_l40_m015 = (t5000_l40_m00 + t5000_l40_m03) / 2

        r1_d10 = (1.0 * u.R_sun).to(u.m)**2 / (10 * u.pc).to(u.m)**2    # radius & distance modifier

        for teff,   logg,   metal,  radius, dist,   exp_flux_ix_1k,         msg in [
            (5000,  4.0,    0.0,    None,   None,   t5000_l40_m00,          "basic non-interpolated flux"),
            # Interpolation
            (5050,  4.0,    0.0,    None,   None,   t5500_l40_m00,          "teff triggers interpolation"),
            (5000,  4.25,   0.0,    None,   None,   t5000_l425_m00,         "logg triggers interpolation"),
            (5000,  4.0,    0.15,   None,   None,   t5000_l40_m015,         "metal triggers interpolation"),
            # radius & distance
            (5000,  4.0,    0.0,    1.0,    10.0,   t5000_l40_m00 * r1_d10, "exact flux modified by radius & dist"),
            (5050,  4.0,    0.0,    1.0,    10.0,   t5500_l40_m00 * r1_d10, "interp' flux modified by radius & dist"),
        ]:
            with self.subTest(msg=msg):
                flux = model_sed.get_fluxes(teff, logg, metal, radius, dist)[1000]
                self.assertAlmostEqual(exp_flux_ix_1k, flux, 2)

    def test_get_fluxes_test_with_reddening(self):
        """ Tests get_fluxes(teff, logg, metal, radius, dist, av) happy path tests for redenning """
        ext_model = G23(Rv=3.1)
        model_sed = BtSettlGrid(self._test_file, extinction_model=ext_model)

        # Known values
        t5000_l40_m00 = self._get_flux_value_from_flat_grid(5000, 4.0, 0.0, flux_ix=1000)
        r1_d10 = (1.0 * u.R_sun).to(u.m)**2 / (10 * u.pc).to(u.m)**2

        # Index of 1000th flux after masking due to the ext_model restricting the wavelength range
        ix_1k = 1000 - sum(~model_sed._wavelength_mask[:1000])

        for teff,   logg,   metal,  radius, dist,   av,     exp_unred_flux_ix_1k,   msg in [
            (5000,  4.0,    0.0,    None,   None,   None,   t5000_l40_m00,          "basic non-reddened flux"),
            # With Av specified
            (5000,  4.0,    0.0,    None,   None,   0.1,    t5000_l40_m00,          "apply reddening"),
            (5000,  4.0,    0.0,    1.0,    10.0,   0.1,    t5000_l40_m00 * r1_d10, "apply reddening with radius & dist"),
        ]:
            with self.subTest(msg=msg):
                if not av:
                    exp_red_flux_ix_1k = exp_unred_flux_ix_1k
                else:
                    exp_red_flux_ix_1k = exp_unred_flux_ix_1k * ext_model.extinguish(model_sed.wavenumbers[ix_1k], av)

                flux = model_sed.get_fluxes(teff, logg, metal, radius, dist, av=av)[ix_1k]
                self.assertAlmostEqual(exp_red_flux_ix_1k, flux, 2)

    def test_get_fluxes_stellar_params_out_of_range(self):
        """ Tests get_fluxes() with Teff, logg or metal outside the model's range -> ValueError """
        model_sed = BtSettlGrid(self._test_file)

        for teff, logg, metal, msg in [
            (1000, 4.0, 0.0, "test Teff out of range"),
            (5000, 7.0, 0.0, "test logg out of range"),
            (5000, 4.0, 2.0, "test metal out of range"),
        ]:
            with self.subTest(msg=msg):
                with self.assertRaises(ValueError):
                    model_sed.get_fluxes(teff, logg, metal)


    #
    #   get_filter_fluxes(filters, teff, logg, metal=0, radius=None, distance=None) -> np.ndarray[float] or u.Quantity:
    #
    def test_get_filter_fluxes_no_reddening(self):
        """ Tests get_filter_fluxes() with use of pre-filtered grid and interpolated values """
        model_sed = BtSettlGrid(self._test_file)

        # Known fluxes
        t5000_l40_m00 = self._get_filter_flux_values_from_flat_grid("GAIA/GAIA3:Gbp", 5000, 4.0, 0.0)
        t5100_l40_m00 = self._get_filter_flux_values_from_flat_grid("GAIA/GAIA3:Gbp", 5100, 4.0, 0.0)
        t5000_l45_m00 = self._get_filter_flux_values_from_flat_grid("GAIA/GAIA3:Gbp", 5000, 4.5, 0.0)
        t5000_l40_m03 = self._get_filter_flux_values_from_flat_grid("GAIA/GAIA3:Gbp", 5000, 4.0, 0.3)

        multi_gaia_filters = ["Gaia:G", "GAIA/GAIA3:Grp", "GAIA/GAIA3:Gbp"]
        t5000_l40_m00_x3 = self._get_filter_flux_values_from_flat_grid(multi_gaia_filters, 5000, 4.0, 0.0)

        t5050_l40_m00 = (t5000_l40_m00 + t5100_l40_m00) / 2         # Approx interpolated values
        t5000_l425_m00 = (t5000_l40_m00 + t5000_l45_m00) / 2
        t5000_l40_m015 = (t5000_l40_m00 + t5000_l40_m03) / 2

        r1_d10 = (1.0 * u.R_sun).to(u.m)**2 / (10 * u.pc).to(u.m)**2

        for filters,                teff,   logg,   metal,  rad,    dist,   exp_fluxes,                 msg in [
            ("GAIA/GAIA3:Gbp",      5000,   4.0,    0.0,    None,   None,   t5000_l40_m00,              "test single str filter"),
            (["GAIA/GAIA3:Gbp"],    5000,   4.0,    0.0,    None,   None,   t5000_l40_m00,              "test single list[str] filter"),
            (multi_gaia_filters,    5000,   4.0,    0.0,    None,   None,   t5000_l40_m00_x3,           "multiple filters in different order to file cols"),
            # Interpolation
            ("GAIA/GAIA3:Gbp",      5050,   4.0,    0.0,    None,   None,   t5050_l40_m00,              "test interpolation on teff"),
            ("GAIA/GAIA3:Gbp",      5000,   4.25,   0.0,    None,   None,   t5000_l425_m00,             "test interpolation on logg"),
            ("GAIA/GAIA3:Gbp",      5000,   4.0,    0.15,   None,   None,   t5000_l40_m015,             "test interpolation on metal"),
            # radius & distance
            (["GAIA/GAIA3:Gbp"],    5000,   4.0,    0.0,    1.0,    10.0,   t5000_l40_m00 * r1_d10,     "test single list[str] filter"),
            (multi_gaia_filters,    5000,   4.0,    0.0,    1.0,    10.0,   t5000_l40_m00_x3 * r1_d10,  "multiple filters in different order to file cols"),
        ]:
            with self.subTest(msg=msg):
                fluxes = model_sed.get_filter_fluxes(filters, teff, logg, metal, rad, dist)

                self.assertIsInstance(fluxes, np.ndarray)
                for exp_flux, flux in zip(exp_fluxes, fluxes):
                    self.assertAlmostEqual(exp_flux, flux, 2)

    def test_get_filter_fluxes_with_reddening(self):
        """ Tests get_filter_fluxes() with use of pre-filtered grid and interpolated values """
        ext_model = G23(Rv=3.1)
        model_sed = BtSettlGrid(self._test_file, extinction_model=ext_model)

        # Known fluxes; a whole row from which we'll calculate the expected filtered values
        t5000_l40_m00 = self._get_flux_value_from_flat_grid(5000, 4.0, 0.0)
        multi_gaia_filters = ["Gaia:G", "GAIA/GAIA3:Grp", "GAIA/GAIA3:Gbp"]
        r1_d10 = (1.0 * u.R_sun).to(u.m)**2 / (10 * u.pc).to(u.m)**2

        for filters,                teff,   logg,   metal,  rad,    dist,   av,     exp_unred_fluxes,       msg in [
            (["GAIA/GAIA3:Gbp"],    5000,   4.0,    0.0,    None,   None,   None,   t5000_l40_m00,          "unreddened single filter"),
            (multi_gaia_filters,    5000,   4.0,    0.0,    None,   None,   None,   t5000_l40_m00,          "unreddened multiple filters"),
            # with av specified
            (["GAIA/GAIA3:Gbp"],    5000,   4.0,    0.0,    None,   None,   0.1,    t5000_l40_m00,          "reddened single filter no rad/dist"),
            (multi_gaia_filters,    5000,   4.0,    0.0,    None,   None,   0.1,    t5000_l40_m00,          "reddened multiple filters no rad/dist"),
            (["GAIA/GAIA3:Gbp"],    5000,   4.0,    0.0,    1.0,    10.0,   0.1,    t5000_l40_m00 * r1_d10, "reddened single filter no rad/dist"),
            (multi_gaia_filters,    5000,   4.0,    0.0,    1.0,    10.0,   0.1,    t5000_l40_m00 * r1_d10, "reddened multiple filters with rad/dist"),
        ]:
            with self.subTest(msg=msg):
                if not av:
                    exp_red_flux = exp_unred_fluxes[model_sed._wavelength_mask]
                    exp_red_flux = self._calculate_filtered_flux_values(filters,
                                                                        model_sed.wavelengths,
                                                                        exp_red_flux)
                else:
                    wavenumbers = model_sed.wavenumbers
                    exp_red_flux = exp_unred_fluxes[model_sed._wavelength_mask]
                    exp_red_flux = exp_red_flux * ext_model.extinguish(wavenumbers, av)
                    exp_red_flux = self._calculate_filtered_flux_values(filters,
                                                                        model_sed.wavelengths,
                                                                        exp_red_flux)

                fluxes = model_sed.get_filter_fluxes(filters, teff, logg, metal, rad, dist, av)
                self.assertIsInstance(fluxes, np.ndarray)
                for exp_flux, flux in zip(exp_red_flux, fluxes):
                    self.assertAlmostEqual(exp_flux, flux, 2)

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
