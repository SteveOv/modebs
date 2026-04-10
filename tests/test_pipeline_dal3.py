""" Unit tests for the pipeline module. """
# pylint: disable=unused-import, too-many-public-methods, line-too-long, invalid-name, no-member
import unittest
from os import getpid
from socket import gethostname
from pathlib import Path

import numpy as np
import astropy.units as u
from uncertainties import nominal_value, std_dev, ufloat

from libs.pipeline_dal3 import QTableDal3, QTableFileDal3



class TestQTableDal3(unittest.TestCase):
    """ Unit tests for the QTableDal3 class. """

    def test_init_happy(self):
        """ Test __init__() asserting expected responses """
        dal = QTableDal3()
        self.assertIsNotNone(dal.lock_id)


class TestQTableFileDal3(unittest.TestCase):
    """ Unit test for the QTableFileDal3 class. """

    def test_QTableFileDal3_general(self):
        """ End to end test """
        test_file = Path.cwd() / ".cache/.test_data/test-qtable-file-dal3.qtable"
        test_file.unlink(missing_ok=True)

        dal = QTableFileDal3(test_file)

        this_lock_id = f"{gethostname()}:{getpid()}"

        # Atomic adds don't require the lock semantics
        dal.add_row("AN Cam", fitted_lcs=True, fitted_sed=False, fitted_masses=False)
        dal.add_row("AN Other", locked_by="AN Other", fitted_lcs=False, fitted_sed=False, fitted_masses=False)
        dal.add_row("CW Eri", locked_by=this_lock_id, fitted_lcs=False, fitted_sed=False, fitted_masses=False)
        dal.add_row("ZZ Boo", fitted_lcs=False, fitted_sed=False, fitted_masses=False)
        dal.add_row("ZZ UMa", fitted_lcs=False, fitted_sed=True, fitted_masses=False)


        for row in dal.acquire_next_row(fitted_lcs=False, fitted_sed=False, fitted_masses=False):
            self.assertNotIn(row.key, ["AN Cam", "ZZ UMa"])
            row.fitted_lcs = True
            row.Teff_sys = ufloat(5750, 50)

        for row in dal.acquire_next_row(fitted_lcs=True, fitted_sed=False, fitted_masses=False):
            print(row.key)
            print(row.Teff_sys)
