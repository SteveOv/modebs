""" Unit tests for the MistIsochrones class. """
import unittest
import numpy as np

from libs.mistisochrones import MistIsochrones

# pylint: disable=too-many-public-methods, line-too-long
class TestMistIsochrones(unittest.TestCase):
    """ Unit tests for the MistIsochrones class. """

    #
    #   __init__(metallicities: list[float]=None)
    #
    def test_init_no_feh(self):
        """ Test __init__() loads all available iso files """
        # pylint: disable=protected-access
        data_dir = MistIsochrones._this_dir / "data/mist/MIST_v1.2_vvcrit0.4_basic_isos"
        num_files = len(list(data_dir.glob("*.iso")))

        misos = MistIsochrones()
        fehs = list(misos._isos.keys())
        self.assertEqual(len(fehs), num_files)

    def test_init_two_fehs(self):
        """ Test __init__(metallicities=[0.0, 0.25]) loads only the two matching iso files """
        # pylint: disable=protected-access
        misos = MistIsochrones(metallicities=[0.0, 0.25]) # both should exist
        fehs = list(misos._isos.keys())
        self.assertEqual(len(fehs), 2)

    def test_init_unknown_feh(self):
        """ Test __init__(metallicities=[0.0, 0.33]) loads only the one known matching iso file """
        # pylint: disable=protected-access
        misos = MistIsochrones(metallicities=[0.0, 0.33]) # there is no 0.33 file
        fehs = list(misos._isos.keys())
        self.assertEqual(len(fehs), 1)

    #
    #   list_metallicities() -> np.ndarray
    #
    def test_list_metallicities(self):
        """ Test list_metallicities() matches those loaded """
        # pylint: disable=protected-access
        misos = MistIsochrones(metallicities=[0.0, 0.25]) # both should exist
        exp_fehs = list(misos._isos.keys())
        fehs = misos.list_metallicities()
        np.testing.assert_array_equal(exp_fehs, fehs)

    #
    #   list_ages(feh: float, min_phase: float=0.0, max_phase: float=9.0) -> np.ndarray
    #
    def test_list_ages_known_feh(self):
        """ Test list_ages(feh=<known feh>) lists expected range """
        ages = MistIsochrones(metallicities=[0.0]).list_ages(feh=0.0)
        self.assertIn(5.0, ages)
        self.assertIn(10.3, ages)

    def test_list_ages_unknown_feh(self):
        """ Test list_ages(feh=<unknown feh>) raises KeyError """
        with self.assertRaises(KeyError):
            MistIsochrones(metallicities=[0.0]).list_ages(feh=0.25)

    def test_list_ages_max_phase(self):
        """ Test list_ages(feh=<known feh>, min_phase=2) lists expected range """
        ages = MistIsochrones(metallicities=[0.0]).list_ages(feh=0.0, max_phase=2.0)
        self.assertEqual(min(ages), 5.0)
        self.assertEqual(max(ages), 10.3)


    #
    #   stellar_params_for_mass(feh: float, log_age: float, mass: float,
    #                           min_phase: float=None, max_phase: float=None,
    #                           params: List[str]=["R", "Teff"]) -> np.ndarray
    #
    def test_stellar_params_for_mass_phase_criteria(self):
        """ Test stellar_params_for_mass(min_phase=0, max_phase=2.0) lists expected range """
        misos = MistIsochrones(metallicities=[0])
        with self.assertRaises(ValueError):
            # Stars of mass 10 M_sun are longer be near the main sequence, so mass value is invalid
            misos.stellar_params_for_mass(feh=0, log_age=9, mass=10, min_phase=0.0, max_phase=2.0)

if __name__ == "__main__":
    unittest.main()
