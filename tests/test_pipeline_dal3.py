""" Unit tests for the pipeline module. """
# pylint: disable=unused-import, too-many-public-methods, line-too-long, invalid-name, no-member
import unittest
from os import getpid
from socket import gethostname
from pathlib import Path

import numpy as np
import astropy.units as u
from uncertainties import nominal_value, std_dev, ufloat
import mariadb

from libs.pipeline_dal3 import QTableDal3, QTableFileDal3, MariaDbTableDal



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

        # Atomic adds don't require the lock semantics
        dal.add_row("AN Cam", fitted_lcs=True, fitted_sed=False, fitted_masses=False)
        dal.add_row("CW Eri", fitted_lcs=False, fitted_sed=False, fitted_masses=False, locked_by=dal.lock_id)
        dal.add_row("ZZ Boo", fitted_lcs=False, fitted_sed=False, fitted_masses=False)

        # These should not be picked up by acquire_next_row (blow).
        # AN Other is locked by another inst. ZZ UMa doesn't match the where criteria.
        dal.add_row("AN Other", fitted_lcs=False, fitted_sed=False, fitted_masses=False, locked_by="AN Other")
        dal.add_row("ZZ UMa", fitted_lcs=False, fitted_sed=True, fitted_masses=False)

        where = { "fitted_lcs": False, "fitted_sed": False, "fitted_masses": False }
        self.assertEqual(2, dal.count_where(**where))
        for row in dal.acquire_next_row(**where):
            self.assertIn(row.key, ["CW Eri", "ZZ Boo"])
            row.fitted_lcs = True
            row.Teff_sys = ufloat(5750, 50)
            row["logg_sys"] = ufloat(4.0, 0.1)
            row.append_warning("Hello")

        # Check for updates (note the changed where criteria)
        where["fitted_lcs"] = True
        self.assertEqual(3, dal.count_where(**where))
        for row in dal.acquire_next_row(**where):
            self.assertIn(row.key, ["CW Eri", "ZZ Boo", "AN Cam"])
            row.append_warning("Hello again")
            row.append_warning("Hello") # Should not appear more than once
            print(f"{row.key} : Teff_sys={row.Teff_sys}, logg_sys={row.logg_sys}, warnings={row.warnings}")

        # Atomic update on unlocked row
        dal.update_row("CW Eri", search_term="V* CW Eri")

        with self.assertRaises(KeyError):
            # Attempted atomic update on a row locked by another inst (should fail!)
            dal.update_row("AN Other", search_term="V* AN Other")


class TestMariaDbTableDal(unittest.TestCase):
    """ Unit test for the MariaDbTableDal class. """

    def test_MariaDbTableDal_general(self):
        """ End to end test """
        db_config = {
            "host": "localhost",
            "port": 3306,
            "user": "modebs",
            "password": "modebs",
            "database": "modebs"
        }
        table_name = "unit_test"

        dal = MariaDbTableDal(db_config=db_config, table_name=table_name)

        # Reset the table if it was already present
        with mariadb.connect(**db_config) as conn, conn.cursor() as cursor:
            sql = f"DELETE FROM {db_config['database']}.`{table_name}`"
            cursor.execute(sql)
            conn.commit()

        # Atomic adds don't require the lock semantics
        dal.add_row("AN Cam", fitted_lcs=True, fitted_sed=False, fitted_masses=False)
        dal.add_row("CW Eri", fitted_lcs=False, fitted_sed=False, fitted_masses=False, locked_by=dal.lock_id)
        dal.add_row("ZZ Boo", fitted_lcs=False, fitted_sed=False, fitted_masses=False)

        # These should not be picked up by acquire_next_row (blow).
        # AN Other is locked by another inst. ZZ UMa doesn't match the where criteria.
        dal.add_row("AN Other", fitted_lcs=False, fitted_sed=False, fitted_masses=False, locked_by="AN Other")
        dal.add_row("ZZ UMa", fitted_lcs=False, fitted_sed=True, fitted_masses=False)

        where = { "fitted_lcs": False, "fitted_sed": False, "fitted_masses": False }
        self.assertEqual(2, dal.count_where(**where))
        for row in dal.acquire_next_row(**where):
            self.assertIn(row.key, ["CW Eri", "ZZ Boo"])
            row.fitted_lcs = True
            row.Teff_sys = ufloat(5750, 50)
            row["logg_sys"] = ufloat(4.0, 0.1)
            row.append_warning("Hello")
            print(f"{row.key} : Teff_sys={row.Teff_sys}, logg_sys={row.logg_sys}, warnings={row.warnings}")


        # Check for updates (note the changed where criteria)
        where["fitted_lcs"] = True
        self.assertEqual(3, dal.count_where(**where))
        for row in dal.acquire_next_row(**where):
            self.assertIn(row.key, ["CW Eri", "ZZ Boo", "AN Cam"])
            row.append_warning("Hello again")
            row.append_warning("Hello") # Should not appear more than once
            print(f"{row.key} : Teff_sys={row.Teff_sys}, logg_sys={row.logg_sys}, warnings={row.warnings}")

        # Atomic update on unlocked row
        dal.update_row("CW Eri", search_term="V* CW Eri")

        with self.assertRaises(KeyError):
            # Attempted atomic update on a row locked by another inst (should fail!)
            dal.update_row("AN Other", search_term="V* AN Other")
