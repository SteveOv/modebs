""" Data access components for reading/writing pipeline progress """
# pylint: disable=no-member
from typing import Union as _Union, List as _List, Dict as _Dict
from typing import Callable as _Callable, Generator as _Generator
from pathlib import Path as _Path
from abc import ABC as _ABC, abstractmethod as _abstractmethod
from contextlib import AbstractContextManager as _AbstractContextManager
from numbers import Number as _Number
from warnings import warn as _warn
from threading import RLock as _RLock
from os import getpid as _getpid
from inspect import stack as _stack, getmodule as _getmodule
from inspect import getsourcefile as _getsourcefile, getfullargspec as _getfullargspec
from socket import gethostname as _gethostname
import re as _re
import mariadb as _mariadb

import numpy as _np
from numpy.typing import ArrayLike as _ArrayLike
import astropy.units as _u
from astropy.table import QTable as _QTable
from uncertainties import ufloat as _ufloat, UFloat as _UFloat
from uncertainties import nominal_value as _nom_val, std_dev as _std_dev


class DalDataRow(_AbstractContextManager):
    """
    Generic representation of an updatable row of data from the underlying data store.
    
    Data may be read and modified through the public interface. Changes will be written
    to the underlying data store, via the persist_func, on leaving the current context (__exit__).

    Values are stored in the underlying data store in columns matching the col names used.
    This also supports reading/writing UFloats and expects the nominal and std_dev components
    to be split across pairs of columns named as [col] and [col]_err. Astropy Quantities are
    also supported, and units will be coerced to those expected by the underlying data store.
    """

    _locked_by_len = 50

    # Field definitions. These are in numpy name/dtype format which works directly
    # on astropy tables. Other storage mechanisms will need to interpret these.
    _storage_schema = [
        # Primary Key
        ("target_id", "<U14"),
        # SIMBAD and IDs
        ("search_term", "<U20"),
        ("tics", "<U40"),
        ("gaia_dr3_id", int),
        ("spt", "<U20"),
        # Gaia DR3 (coords in ICRS, falling back on SIMBAD)
        ("ra_coord", float),
        ("ra_coord_err", float),
        ("dec_coord", float),
        ("dec_coord_err", float),
        ("parallax", float),
        ("parallax_err", float),
        ("parallax_bibcode", "<U20"),
        ("ruwe", float),
        # TESS-ebs
        ("t0", float),
        ("t0_err", float),
        ("period", float),
        ("period_err", float),
        ("morph", float),
        ("widthP", float),
        ("widthS", float),
        ("depthP", float),
        ("depthS", float),
        ("phiS", float),
        # TESS
        ("Teff_sys", float),
        ("Teff_sys_err", float),
        ("logg_sys", float),
        ("logg_sys_err", float),
        # JKTEBOP lightcurve fitting i/o params (initially from EBOP MAVEN preds)
        ("rA_plus_rB", float),
        ("rA_plus_rB_err", float),
        ("k", float),
        ("k_err", float),
        ("J", float),
        ("J_err", float),
        ("ecosw", float),
        ("ecosw_err", float),
        ("esinw", float),
        ("esinw_err", float),
        ("bP", float),
        ("bP_err", float),
        ("inc", float),
        ("inc_err", float),
        # JKTEBOP lightcurve fitting i/o params (from other sources)
        ("qphot", float),
        ("qphot_err", float),
        ("L3", float),
        ("L3_err", float),
        # JKTEBOP lightcurve fitting output params
        ("LR", float),
        ("LR_err", float),
        ("TeffR", float),
        ("TeffR_err", float),
        # SED fitting i/o params
        ("TeffA", float),
        ("TeffA_err", float),
        ("TeffB", float),
        ("TeffB_err", float),
        ("loggA", float),
        ("loggA_err", float),
        ("loggB", float),
        ("loggB_err", float),
        ("RA", float),
        ("RA_err", float),
        ("RB", float),
        ("RB_err", float),
        ("dist", float),
        ("dist_err", float),
        # Mass fitting i/o params
        ("M_sys", float),
        ("M_sys_err", float),
        ("a", float),
        ("a_err", float),
        ("MA", float),
        ("MA_err", float),
        ("MB", float),
        ("MB_err", float),
        ("log_age", float),
        ("log_age_err", float),
        # Progress flags
        ("fitted_lcs", bool),
        ("fitted_sed", bool),
        ("fitted_masses", bool),
        ("locked_by", f"<U{_locked_by_len}"),
        ("warnings", object),
        ("errors", object)
    ]

    _col_units = {
        "ra": _u.deg,
        "ra_err": _u.deg,
        "dec": _u.deg,
        "dec_err": _u.deg,
        "parallax": _u.mas,
        "parallax_err": _u.mas,
        "period": _u.d,
        "period_err": _u.d,
        "Teff_sys": _u.K,
        "Teff_sys_err": _u.K,
        "inc": _u.deg,
        "inc_err": _u.deg,
        "TeffA": _u.K,
        "TeffA_err": _u.K,
        "TeffB": _u.K,
        "TeffB_err": _u.K,
        "RA": _u.solRad,
        "RA_err": _u.solRad,
        "RB": _u.solRad,
        "RB_err": _u.solRad,
        "dist": _u.pc,
        "dist_err": _u.pc,
        "M_sys": _u.solMass,
        "M_sys_err": _u.solMass,
        "a": _u.solRad,
        "a_err": _u.solRad,
        "MA": _u.solMass,
        "MA_err": _u.solMass,
        "MB": _u.solMass,
        "MB_err": _u.solMass,
    }

    def __init__(self,
                 key: str,
                 values: _ArrayLike,
                 persist_func: _Callable[[_ArrayLike], None],
                 hidden_cols: _List[str]=None):
        """
        Generic representation of an updatable row of data from the underlying data store with
        support for get/set of values and mass value setter. This is a ContextManager with data
        persisted to its underlying store when an instance leaves its containing context.

        :key: the row's primary key value
        :values: the rows full set of values in an ArrayLike form
        :persist_func: the func called with the update set of values to make the updates permanent
        :hidden_cols: those values cols which are not to be exposed as attr of this inst
        """
        # pylint: disable=too-many-arguments, too-many-positional-arguments
        super().__init__()

        # We do not want __setattr__ to handle these!
        self.__dict__["_key"] = key
        self.__dict__["_values"] = values
        self.__dict__["_persist_func"] = persist_func
        self.__dict__["_hidden_cols"] = hidden_cols or []
        self.__dict__["_dirty_cols"] = []
        self.__dict__["_sep"] = ";"

    @property
    def key(self) -> str:
        """ Gets the value of the row's key col. """
        return self._key

    def has_col(self, col):
        """
        Returns whether or not the underlying row has a col with the requested name
        """
        if self._values is not None:
            return (col in self._values.dtype.names) and (col not in self._hidden_cols or [])
        return False

    def set_values(self, **cols_and_values):
        """ Sets the value of multiple cols in single call. """
        for c, v in cols_and_values.items():
            self[c] = v

    def append_warning(self, new_warn_msg: str):
        """ Appends a unique message to the warnings col. Non-unique messages are discarded. """
        if new_warn_msg and len(new_warn_msg := new_warn_msg.strip()) > 0:
            warn_msgs = (self["warnings"] or "").split(self._sep)
            if new_warn_msg not in warn_msgs:
                self["warnings"] = self._sep.join(w for w in warn_msgs + [new_warn_msg] if len(w))

    def __getattr__(self, col: str) -> any:
        """
        Handles the default behaviour for attributes. Gets the value of the correspondingly
        named column from the row's data and also includes the uncertainty if present.
        """
        if self.has_col(col):
            err_col = col + "_err"
            val = self._read_col_value(self._values, col)
            if isinstance(val, _u.Quantity): # Need to return value sans units
                if self.has_col(err_col):
                    val_err = self._read_col_value(self._values, err_col)
                    return _ufloat(val.value, val_err.value if val_err is not None else None)
                return val.value
            if self.has_col(err_col) and val is not None:
                return _ufloat(val, self._read_col_value(self._values, err_col))
            return val
        raise AttributeError(name=col, obj=self)

    def __getitem__(self, col):
        """ Get a column's value using row['col_name'] syntax """
        return self.__getattr__(col)

    def __setattr__(self, col: str, value):
        """
        Handles the default behaviour for attributes. Sets the value of the similarly named column
        in the row's value array and also the uncertainty if a corresponding _err column exists.
        """
        if self.has_col(col):
            err_col = err_col if self.has_col(err_col := col + "_err") else None
            if isinstance(value, _UFloat):
                self._set_col_value(self._values, col, _nom_val(value))
                if err_col is not None:
                    self._set_col_value(self._values, err_col, _std_dev(value))
                else:
                    _warn(f"Uncertainty for {col} was discarded as there is no {col}_err column")
            else:
                self._set_col_value(self._values, col, value)
                if err_col is not None:
                    # Assume an err column is always going to be numeric
                    self._set_col_value(self._values, err_col, 0)

            for c in (cc for cc in [col, err_col] if cc is not None and cc not in self._dirty_cols):
                self._dirty_cols.append(c)
        else:
            raise AttributeError(name=col, obj=self)

    def __setitem__(self, col, value):
        """ Set a column value with the row['col_name'] = new_value syntax. """
        self.__setattr__(col, value)

    def __exit__(self, exc_type, exc_value, traceback):
        self._persist_func(self._values[self._hidden_cols + self._dirty_cols])
        return super().__exit__(exc_type, exc_value, traceback)

    @classmethod
    def _read_col_value(cls, values: _ArrayLike, col: str):
        """ Low level col read with additional parsing for potentially masked values from QTable """
        value = values[col]
        if _np.ma.is_masked(value):
            # Value is explicitly masked which is expected when no value has been stored
            return None
        if _np.ma.isMaskedArray(value) or hasattr(value, "unmasked"):
            # We also seem to get masked value types, even when we have values
            return value.unmasked
        return value

    @classmethod
    def _set_col_value(cls, values: _ArrayLike, col: str, value: any):
        """ Low level write of the requested value to the requested col while handling units. """
        values[col] = cls._append_column_unit(values[col], value)

    @classmethod
    def _append_column_unit(cls, column: any, value: any):
        """ Applies the indicated column's unit to the passed value """
        if value is None or not isinstance(value, _Number|_u.Quantity):
            return value
        unit = (column.unit or 1) if hasattr(column, "unit") else 1
        if isinstance(value, _u.Quantity):
            return value.to(unit) if isinstance(unit, _u.Unit) else value.value
        return value * unit


class Dal3(_ABC):
    """
    Base data access layer (Dal) for reading/writing simple table data. This supports adding and
    updating rows based on the key values. It also supports 'acquire_next_row' functionality which
    iterates over rows matching the supplied 'where' criteria, yielding each for updates once
    a lock has been applied to prevent other instances from selecting the same row. This is
    intended to manage data while carrying out long running processes with it such as fitting.
    """

    def __init__(self, key_col: str="target_id", lock_col: str="locked_by"):
        """
        Initializes a Dal class which provides a consistent interface to underlying storage.

        :_key_col: the name of the col/param which acts as each row's unique key/index value
        :_lock_col: the name of the col/param which acts as a soft lock on a row
        """
        self._key_col: str = key_col
        self._lock_col: str = lock_col

        client_name = self.__class__.__name__
        this_file = _getsourcefile(lambda:0)
        for frameinfo in _stack():
            if (module := _getmodule(frameinfo.frame)) is not None and module.__file__ != this_file:
                client_name = _Path(module.__file__).stem
                break
        self._lock_id = f"{_gethostname()}/{client_name}/{_getpid()}"[-DalDataRow._locked_by_len:]

        super().__init__()

    @property
    def lock_id(self) -> str:
        """ Return the id with which this instance locks rows. """
        return self._lock_id

    def acquire_next_row(self, **where) -> _Generator[DalDataRow, any, None]:
        """
        Yields the next available row which both matches the criteria and is unlocked. In doing so,
        a lock is placed on the row to prevent other clients acquiring it while processing and
        updates are carried out. Updates made to the yielded DalDataRow will be written back to the
        underlying storage mechanism before the lock is released, when the row leaves its context.

        :where: col/value criteria with which simple, col==value, matches are evaluated for each row
        """
        # We are only interested in rows which are currently unlocked
        for row_data in self._lock_and_yield_data_rows(**where):
            with DalDataRow(key=row_data[self._key_col],
                            values=row_data,
                            persist_func=self._update_and_release_row,
                            hidden_cols=[self._key_col, self._lock_col]) as row:
                # Yield the generic DalDataRow to the client, with which it can read/write values.
                # The write_func will be called to persist any changes when we exit this context.
                yield row

    def update_row(self, key: str, **cols_and_values):
        """
        Acquire, lock, update and release a row in a single call.

        Where long term locks are required, such as while fitting, use acquire_next_row() instead.

        :key: the unique key of the row to update
        :cols_and_values: the col/value pair for each column to update
        """
        # The where criterion is on the primary key so we can only get 0 or 1 rows.
        for row in self.acquire_next_row(**dict([(self._key_col, key)])):
            row.set_values(**cols_and_values)
            # Changes will be persisted as we leave the context here (see acquire_next_row above).
            return
        raise KeyError(f"No unlocked row found for key '{key}'. Updates were not saved.")

    def count_where(self, **where) -> int:
        """ Gets the current number of unlocked rows matching the passed where criteria. """
        # This is a sub-optimal default implementation. Ideally we override this in subclasses.
        return len(list(self.acquire_next_row(**where)))

    @_abstractmethod
    def add_row(self, key: str, **cols_and_values):
        """ Add a new row with the indicated unique key and col/value pairs"""

    @_abstractmethod
    def _lock_and_yield_data_rows(self, **where) -> _Generator[_ArrayLike, any, None]:
        """ Iterate the unlocked storage rows, locking & yield those matching the where criteria """

    @_abstractmethod
    def _update_and_release_row(self, values: _ArrayLike):
        """ Will update any indicated values before removing the lock on the row. """


class QTableDal3(Dal3):
    """ Pipeline Dal for reading/writing of an in memory astropy QTable """

    # Belt and braces: we send writes & reads through a critical section to guard for inconsistency
    _CLIENT_LOCK = _RLock()

    def __init__(self):
        """
        Initializes the QTableDal Dal class which uses an in memory astropy QTable for storage.
        This Dal is not durable and should not be used where permanent storage is required.
        """
        super().__init__()
        self._table = _QTable(masked=True, # Support for None/null values
                              dtype=DalDataRow._storage_schema,
                              units=DalDataRow._col_units,
                              rows=[])
        self._table.add_index(self._key_col, unique=True)
        self._usable_lock_vals = ("", None, "None")

    def count_where(self, **where) -> int:
        count = 0
        with self._CLIENT_LOCK:
            where.setdefault(self._lock_col, None)
            for row in self._table:
                test_vals = [DalDataRow._read_col_value(row, c) for c in where] # pylint: disable=protected-access
                count += all((tv == wv) or (c == self._lock_col and tv in self._usable_lock_vals) \
                                                for tv, (c, wv) in zip(test_vals, where.items()))
        return count

    def add_row(self, key, **cols_and_values):
        with self._CLIENT_LOCK:
            # This will raise a ValueError if a row with the same key already exists.
            self._table.add_row({ self._key_col: key } | cols_and_values)

    def _lock_and_yield_data_rows(self, **where):
        with self._CLIENT_LOCK:
            where[self._lock_col] = None
            for row in self._table:
                test_vals = [DalDataRow._read_col_value(row, c) for c in where] # pylint: disable=protected-access
                if all((tv == wv) or (c == self._lock_col and tv in self._usable_lock_vals) \
                                                for tv, (c, wv) in zip(test_vals, where.items())):
                    row[self._lock_col] = self._lock_id
                    yield row

    def _update_and_release_row(self, values: _ArrayLike):
        with self._CLIENT_LOCK:
            # Will throw a KeyError if key is unknown, although this should not be possible
            key = values[self._key_col]
            row_ix = self._table.loc_indices[key]

            if self._table[row_ix][self._lock_col] == self._lock_id:
                wcols = (c for c in values.dtype.names if c not in [self._key_col, self._lock_col])
                for col in wcols:
                    self._table[row_ix][col] = values[col]

                # release the storage row
                self._table[row_ix][self._lock_col] = ""
            else:
                raise ValueError(f"Cannot write to row with key={key}." +
                                 " It's not locked by this client instance.")


class QTableFileDal3(QTableDal3):
    """ Pipeline data access for reading/writing a file based on an astropy QTable """
    def __init__(self,
                 file: _Union[str, _Path],
                 file_format: str="ascii.fixed_width_two_line",
                 file_format_kwargs: _Dict[str, any] = None):
        """
        Initializes the QTableDal Dal class, associating it with a file in which to permanently
        store its data.

        **NOTE**: This is not robust enough to be used where multiple clients are expected. There
        is no storage level locking mechanism so it will happily overwrite updates from elsewhere.
        If you need multiple clients and/or robust locking, a database Dal is more suitable.

        See https://docs.astropy.org/en/stable/io/unified_table.html for information on
        potential file formats and the kwargs that can be used to customize them. This will
        apply the sensible defaults, of ["name", "dtype", "unit"], to header_rows if the file
        format 'ascii.fixed_width_two_line' is chosen but header_rows is not specified.

        :file: the file name of the storage file
        :file_format: the format of the file
        :file_format_kwargs: optional set of kwargs specific to each file_format
        """
        file_format_kwargs = file_format_kwargs or { }
        if file_format == "ascii.fixed_width_two_line" and "header_rows" not in file_format_kwargs:
            file_format_kwargs["header_rows"] = ["name", "dtype", "unit"]

        self._file = file if isinstance(file, _Path) else _Path(file)
        self._format = file_format
        self._format_kwargs = file_format_kwargs

        with self._CLIENT_LOCK:
            # Inits a new in-memory table to be saved later, once data/changes are written to it
            super().__init__()
            if self._file.exists():
                print(f"Loading from '{self._file.name}' as {self._format}/{self._format_kwargs}")
                self._table = _QTable.read(self._file, format=self._format, **self._format_kwargs)
                self._table.add_index(self._key_col, unique=True)
            else:
                # Cannot save the file immediately as we get a ValueError thrown with the message
                # "max() arg is an empty sequence" from within astropy. Probably because there is no
                # data yet and astropy isn't checking some list contents exist before calling max().
                print(f"New data file '{self._file.name}' as {self._format}/{self._format_kwargs}",
                      "will be created when values are first written to it.")

    def _write_state_to_file(self):
        """ (Over-)write this instance's in memory QTable to the storage file. """
        self._table.write(self._file, overwrite=True, format=self._format, **self._format_kwargs)

    def _lock_and_yield_data_rows(self, **where):
        # Write the table file when locking each row so that the lock is visible while in place.
        for row in super()._lock_and_yield_data_rows(**where):
            self._write_state_to_file()
            yield row

    def _update_and_release_row(self, values: _ArrayLike):
        with self._CLIENT_LOCK:
            # First update the in table as held in memory, then write the whole thing to the file
            super()._update_and_release_row(values)
            self._write_state_to_file()


class MariaDbTableDal(Dal3):
    """ Pipeline Dal for storing data in a MariaDB Table """
    # Used to parse the "<U##" style dtype declarations in the schema for table length
    _u_str_pattern = _re.compile(r"(?:<|)U(?P<len>\d*)", _re.IGNORECASE)

    def __init__(self, db_config: _Dict[str, any], table_name: str="working_set", ):
        """
        Initializes the MariaDBTableDal class, associating it with the database instance
        and table which will be used to store the data. Whiile the data table will be created
        if it does not already exist, the database is expected to exist and the connecting
        user must have suitable permissions to create and interact with the data table.

        See https://mariadb.com/docs/connectors/connectors-quickstart-guides/connector-python-guide
        for information on the contents of a connection configuration dictionary.

        :db_config: a MariaDB database config dictionary
        :table_name: the name of the table
        """
        super().__init__()
        self._db_config = db_config
        self._full_table_name = f"{self._db_config['database']}.`{table_name}`"
        self._prim_key_col = "row_id"
        with _mariadb.connect(**self._db_config) as conn:
            self.create_working_set_table(conn, table_name, self._prim_key_col,
                                          self._key_col, self._lock_col, exists_ok=True)

    @classmethod
    def create_working_set_table(cls,
                                 conn: _mariadb.Connection,
                                 table_name: str,
                                 prim_key_col: str,
                                 key_col: str,
                                 lock_col: str,
                                 exists_ok: bool=True):
        """
        Creates the requested working set table. Assumes the connection is open,
        the named database exists and the user has the necessary permissions to create the table.

        :conn: the connection to use
        :table_name: the name of the table
        :prim_key_col: the name of the primary key column
        :key_col: the name of the unique key/identifier column
        :lock_col: the name of the soft-lock column (which will have an index placed on it)
        :exists_ok: whether to suppress an error if the table exists (it will not be overwritten)
        """
        with conn.cursor() as cursor:
            # If exists_ok is False and the table exists we expect an 1050 same name exists error
            ddl = "CREATE TABLE IF NOT EXISTS" if exists_ok else "CREATE TABLE"
            ddl += f" `{table_name}` (\n\t`{prim_key_col}` BIGINT UNSIGNED AUTO_INCREMENT NOT NULL"

            schema = DalDataRow._storage_schema # pylint: disable=protected-access
            flag_cols = []
            for col_name, dtype in schema:
                field_dbtype = ""
                if isinstance(dtype, str):
                    # These are expected to be in the form "<U##" where ## is the length
                    match = cls._u_str_pattern.match(dtype)
                    if match is not None and "len" in match.groupdict():
                        field_dbtype = f"VARCHAR({match.group('len')}) "
                    field_dbtype += "NOT NULL" if col_name in [key_col] else "NULL"
                elif dtype is int:
                    field_dbtype = "BIGINT NULL"
                elif dtype is float:
                    field_dbtype = "FLOAT NULL"
                elif dtype is bool:
                    field_dbtype = "BOOL NOT NULL DEFAULT False"
                elif dtype is object:
                    field_dbtype = "LONG VARCHAR NULL"

                if col_name.startswith("fitted") and dtype is bool:
                    flag_cols += [col_name]

                if len(field_dbtype) > 0:
                    ddl += f",\n\t`{col_name}` " + field_dbtype
                else:
                    raise ValueError(f"unexpected schema field {col_name} dtype of {dtype}")

            ddl += f",\n\tPRIMARY KEY (`{prim_key_col}`)"
            ddl += f",\n\tUNIQUE INDEX `{key_col}_unique_index` (`{key_col}`)"
            ddl += f",\n\tINDEX `{lock_col}_index` (`{lock_col}`)"
            ddl += ",\n\tINDEX `flags_index` (" + ",".join(f"`{c}`" for c in flag_cols) + ")"
            ddl += "\n)"
            cursor.execute(ddl)
            conn.commit()

    def count_where(self, **where) -> int:
        return len(self._list_lockable_keys_where(**where))

    def add_row(self, key: str, **cols_and_values):
        add_cols = [c for c in cols_and_values if c not in [self._prim_key_col, self._key_col]]
        ncols = len(add_cols)
        with _mariadb.connect(**self._db_config, autocommit=True) as conn, conn.cursor() as cursor:
            sql = f"INSERT INTO {self._full_table_name} (`{self._key_col}`" + \
                        ("," if ncols else "") + ",".join(f"`{c}`" for c in add_cols) + \
                    ") VALUES (" + ",".join(["?"] * (ncols+1)) + ")"
            cursor.execute(sql, data=tuple([key] + list(cols_and_values[c] for c in add_cols)))

    def _lock_and_yield_data_rows(self, **where) -> _Generator[_ArrayLike, any, None]:
        # Snapshot of row keys which are currently suitable
        for key in self._list_lockable_keys_where(**where):
            wcols = [c for c in where if c not in [self._lock_col]]
            wvals = [where[c] for c in wcols]

            with _mariadb.connect(**self._db_config, autocommit=True) as conn,\
                        conn.cursor(buffered=False) as cursor:
                cursor.execute(f"SET @lock_id='{self._lock_id}';")

                # Initial row selection using the the where clause(s) which set the @key variable.
                # With a "for update" lock for duration of this transaction.
                sql = f"SELECT T.`{self._key_col}` INTO @key FROM {self._full_table_name} AS T " + \
                        f"WHERE T.`{self._key_col}`=? AND T.`{self._lock_col}` IS NULL"
                if len(wcols) > 0:
                    sql += " AND " + " AND ".join(f"T.`{c}`=?" for c in wcols)
                sql += " LIMIT 1 FOR UPDATE;"
                cursor.execute(sql, data=tuple([key] + wvals))

                # May not match if the row has changed since the initial list of keys was drawn up.
                if cursor.rowcount > 0:
                    # Put the soft lock the row
                    sql = f"UPDATE {self._full_table_name} AS T " + \
                            f"SET T.`{self._lock_col}`=@lock_id WHERE T.`{self._key_col}`=@key;"
                    cursor.execute(sql)

                    # Select entire row if it's locked
                    sql = f"SELECT T.* FROM {self._full_table_name} AS T" + \
                            f" WHERE T.`{self._key_col}`=@key AND T.`{self._lock_col}`=@lock_id;"
                    cursor.execute(sql)
                    for row in cursor:
                        # A bit of a hack, but a QTable works nicely with DalDataRow & its schema.
                        # pylint: disable=protected-access
                        qtable =_QTable(rows=[], masked=True, dtype=DalDataRow._storage_schema,
                                    units=DalDataRow._col_units)
                        qtable.add_row({ c: DalDataRow._append_column_unit(qtable[c], v)
                                            for c, v in zip(cursor.metadata["field"], row)
                                                if c in qtable.dtype.names and v is not None })
                        yield from qtable
                        break

    def _update_and_release_row(self, values: _ArrayLike):
        key = values[self._key_col]
        ucols = [c for c in values.dtype.names
                        if c not in [self._prim_key_col, self._key_col, self._lock_col]]
        with _mariadb.connect(**self._db_config, autocommit=True) as conn, conn.cursor() as cursor:
            cursor.execute(f"SET @lock_id='{self._lock_id}';")
            cursor.execute(f"SET @key='{key}';")

            sql = f"UPDATE {self._full_table_name} AS T SET T.`{self._lock_col}`=NULL"
            if len(ucols) > 0:
                sql += "," + ",".join(f"T.`{c}`=?" for c in ucols)
            sql += f" WHERE T.`{self._key_col}`=@key AND T.`{self._lock_col}`=@lock_id;"

            # pylint: disable=protected-access
            uvals = tuple(int(v) if isinstance(v, bool|_np.bool_)
                                else v.value if isinstance(v, _u.Quantity) else v
                          for v in (DalDataRow._read_col_value(values, c) for c in ucols))
            cursor.execute(sql, uvals)

    def _list_lockable_keys_where(self, **where) -> _ArrayLike:
        """ Gets a list of row keys which are currently available to lock & match the criteria. """
        with _mariadb.connect(**self._db_config) as conn, conn.cursor() as cursor:
            sql = f"SELECT T.`{self._key_col}` FROM {self._full_table_name} AS T " + \
                    f"WHERE T.`{self._lock_col}` IS NULL"
            if len(wcols := [c for c in where if c not in [self._lock_col]]) > 0:
                sql += " AND " + " AND ".join(f"T.`{c}`=?" for c in wcols)

            cursor.execute(sql, data=tuple(where[c] for c in wcols))
            if cursor.rowcount > 0:
                return [row[0] for row in cursor]
        return []


def create_dal(typename: _Union[str, type[Dal3]], verbose: bool=False, **kwargs):
    """
    A factory method for creating a named Dal instance.

    :typename: the dal type to create
    :verbose: whether or not to write details of the Dal being created to StdOut
    :kwargs: the arguments with which to initialize the dal (specific to the type of dal)
    :returns: the resulting initialized instance
    """
    dal_type = None
    if isinstance(typename, str):
        def get_subclasses(superclass):
            for subclass in superclass.__subclasses__():
                yield subclass
                yield from get_subclasses(subclass)

        possible_names = [typename.casefold(), typename.casefold() + "dal3"]
        for subclass in get_subclasses(Dal3):
            if subclass.__name__.casefold() in possible_names:
                dal_type = subclass
                break
    elif issubclass(typename, Dal3):
        dal_type = typename

    if dal_type is None:
        raise KeyError(f"No Dal type like {dal_type} was found.")
    if _ABC in dal_type.__bases__:
        # Must be careful with this check as we're only ensuring the type is not itself abstract.
        # For this, __bases__ is better than issubclass() as it only looks at direct base types.
        raise ValueError(f"Cannot initialize the abstract class {dal_type.__name__}")

    # Ignore kwargs not used by the type's __init__ otherwise we may get a TypeError. This allows
    # calling code to send a superset of potential kwargs to this func & it will use what is needed.
    argspec = _getfullargspec(dal_type.__init__)
    expected_kwargs = { k: v for k, v in kwargs.items() if k in argspec.args and k not in ["self"] }
    if verbose:
        print(f"Creating a {dal_type.__name__} with kwargs={expected_kwargs}")
    return dal_type(**expected_kwargs)
