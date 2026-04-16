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
                        cursor.execute("SELECT * FROM INFORMATION_SCHEMA.TABLES WHERE TABLE_NAME=?;", data=(table_name, ))
                        if cursor.rowcount > 0:
                            cursor.execute(f"DELETE FROM {db_config['database']}.`{table_name}`")
                            # cursor.execute(f"DROP TABLE {db_config['database']}.`{table_name}`")
                        conn.commit()

                print(f"\nTesting {typename} for consistency")
                dal = create_dal(typename, **kwargs)

                # Atomic adds don't require lock semantics.

                # These match the first set of where criteria & should be acquired in this order they're added.
                dal.add_row("HP Dra", fitted_lcs=False, fitted_sed=False, fitted_masses=False)
                dal.add_row("CW Eri", fitted_lcs=False, fitted_sed=False, fitted_masses=False)

                # Should not be picked up by first acquire loop, but matches criteria for the 2nd
                dal.add_row("ZZ Boo", fitted_lcs=True, fitted_sed=False, fitted_masses=False)

                # Should not be picked up by any of the below as it doesn't match either set of where criteria
                dal.add_row("ZZ UMa", fitted_lcs=False, fitted_sed=True, fitted_masses=False)

                # These should not be picked up by any of the below as they have existing locks in place.
                # This includes a row that appears to have already been locked by this instance.
                dal.add_row("AN Cam", fitted_lcs=False, fitted_sed=False, fitted_masses=False, locked_by=dal.lock_id)
                dal.add_row("AN Other", fitted_lcs=False, fitted_sed=False, fitted_masses=False, locked_by="AN Other")

                print("About to iterate all rows (incl those locked).")
                print("\tMatching rows:", ", ".join(row.key for row in dal.iterate_rows()))

                where = { "fitted_lcs": False, "fitted_sed": False, "fitted_masses": False }
                print(f"About to acquire_next_row loop 1 for criteria: {where}")
                self.assertEqual(2, dal.count_where(**where), "Failed count before loop 1")
                for row in dal.acquire_next_row(**where):
                    self.assertNotIn(row.key, ["ZZ Boo", "ZZ UMa", "AN Cam", "AN Other"], "Failed exclude on loop 1")

                    # Use all options for updating the fields of a row
                    row["spt"] = "SpT"
                    row.set_values(Teff_sys=ufloat(5750, 50), logg_sys=ufloat(4.0, 0.1))
                    row.fitted_lcs = True
                    row.append_warning("Hello")

                    print(f"\t{row.key}: SpT={row.spt}, Teff_sys={row.Teff_sys}, logg_sys={row.logg_sys}, warnings={row.warnings}")

                    # Fail atomic update on this row. Cannot update row with any existing lock (even this row/lock).
                    # This prevents update_row from releasing this acquired row's lock before it is updated.
                    with self.assertRaises(KeyError):
                        dal.update_row(row.key, search_term=f"V* {row.key}")

                # Atomic update on, what is once more, an unlocked row
                dal.update_row("HP Dra", search_term="V* HP Dra")

                # Check for updates (note the changed where criteria which should cover the above updates)
                where["fitted_lcs"] = True
                print(f"About to acquire_next_row loop 2 for criteria: {where}")
                self.assertEqual(3, dal.count_where(**where), "Failed count before loop 2")
                for row in dal.acquire_next_row(**where):
                    self.assertNotIn(row.key, ["ZZ UMa", "AN Cam", "AN Other"], "Failed exclude on loop 2")
                    row.append_warning("Hello again")
                    row.append_warning("Hello") # Should not appear more than once
                    print(f"\t{row.key}: search_term={row.search_term}, SpT={row.spt}, Teff_sys={row.Teff_sys}, logg_sys={row.logg_sys}, warnings={row.warnings}")


                with self.assertRaises(KeyError):
                    # Attempted atomic update on a row locked by another inst (should fail!)
                    dal.update_row("AN Other", search_term="V* AN Other")
