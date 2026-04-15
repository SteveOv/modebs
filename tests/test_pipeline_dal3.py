""" Unit tests for the pipeline module. """
# pylint: disable=unused-import, too-many-public-methods, line-too-long, invalid-name, no-member
import unittest
from pathlib import Path

from uncertainties import ufloat
import mariadb

from libs.pipeline_dal3 import create_dal

class TestSubclassesOfDal3(unittest.TestCase):
    """ Unit test for the QTableFileDal3 class. """

    def test_consistency_general_happy_path(self):
        """ End to end test """
        for typename,           kwargs in [
            ("QTableDal3",      {}),
            ("QTableFileDal3",  { "file": Path.cwd() / ".cache/.test_data/test-qtable-file-dal3.qtable" }),
            ("MariaDbTableDal", {
                "db_config": { "host": "localhost", "port": 3306, "user": "modebs", "password": "modebs", "database": "modebs" },
                "table_name": "unit_test" }),
        ]:
            with self.subTest(f"Testing {typename} for consistency"):
                if "file" in kwargs:
                    kwargs["file"].unlink(missing_ok=True)
                if "db_config" in kwargs:
                    # Reset the table if it was already present
                    db_config, table_name = kwargs["db_config"], kwargs.get("table_name", "working_set")
                    with mariadb.connect(**db_config) as conn, conn.cursor() as cursor:
                        cursor.execute(f"DELETE FROM {db_config['database']}.`{table_name}`")
                        # cursor.execute(f"DROP TABLE {db_config['database']}.`{table_name}`")
                        conn.commit()

                print(f"\nTesting {typename} for consistency")
                dal = create_dal(typename, **kwargs)

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

                # Check for updates (note the changed where criteria which should cover the above updates + AN Cam)
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
