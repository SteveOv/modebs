""" Data access components for reading/writing pipeline progress """
# pylint: disable=no-member
from typing import Union as _Union, List as _List
from typing import Callable as _Callable, Generator as _Generator
from pathlib import Path as _Path
from abc import ABC as _ABC, abstractmethod as _abstractmethod
from contextlib import AbstractContextManager as _AbstractContextManager
from warnings import warn as _warn
from threading import RLock as _RLock
from os import getpid as _getpid
from inspect import stack as _stack, getmodule as _getmodule
from inspect import getsourcefile as _getsourcefile, getfullargspec as _getfullargspec
from socket import gethostname as _gethostname

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
    """

    _locked_by_len = 30

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
                 parse_value_func: _Callable[[str, any], any]=lambda col, val: val,
                 hidden_cols: _List[str]=None):
        """
        Generic representation of an updatable row of data from the underlying data store.

        :key: the row's primary key value
        :values: the rows full set of values in an ArrayLike form
        :persist_func: the func called with the update set of values to make the updates permanent
        :parse_value_func: func called whenever a value is read to interpret any special values
        :hidden_cols: those values cols which are not to be exposed as attr of this inst
        """
        # pylint: disable=too-many-arguments, too-many-positional-arguments
        super().__init__()

        # We do not want __setattr__ to handle these!
        self.__dict__["_key"] = key
        self.__dict__["_values"] = values
        self.__dict__["_persist_func"] = persist_func
        self.__dict__["_parse_value_func"] = parse_value_func
        self.__dict__["_hidden_cols"] = hidden_cols or []
        self.__dict__["_dirty_cols"] = []

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

    def set_values(self, **kwargs):
        """ Sets the value of multiple cols in single call. """
        for c, v in kwargs.items():
            self[c] = v

    def __getattr__(self, col: str) -> any:
        """
        Handles the default behaviour for attributes. Gets the value of the correspondingly
        named column from the row's data and also includes the uncertainty if present.
        """
        if self.has_col(col):
            err_col = col + "_err"
            val = self._parse_value_func(col, self._values[col])
            if isinstance(val, _u.Quantity):
                if self.has_col(err_col):
                    val_err = self._parse_value_func(err_col, self._values[err_col])
                    return _ufloat(val.value, val_err.value if val_err is not None else None)
                return val.value
            if self.has_col(err_col) and val is not None:
                return _ufloat(val, self._parse_value_func(err_col, self._values[err_col]))
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

            # Get the units to use if the storage supports/expects them (i.e. astropy QTable row).
            unit = self._values[col].unit if hasattr(self._values[col], "unit") else 1

            if isinstance(value, _UFloat):
                self._values[col] = _nom_val(value) * unit
                if err_col is not None:
                    self._values[err_col] = _std_dev(value) * unit
                else:
                    _warn(f"Uncertainty for {col} was discarded as there is no {col}_err column")
            else:
                if isinstance(value, _u.Quantity):
                    self._values[col] = value.to(unit) if isinstance(unit, _u.Unit) else value.value
                elif value is None or unit is None:
                    # This also covers "no unit expected", which includes non-numeric values.
                    self._values[col] = value
                else:
                    self._values[col] = value * unit

                if err_col is not None:
                    # Assume an err column is always going to be numeric
                    self._values[err_col] = 0 * unit

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


class Dal3(_ABC):
    """
    Base data access layer (Dal) for reading/writing simple table data via a generator
    which yields and locks the next available row matching the selected criteria.
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
        Yields the next available row which both matches the criteria and is unlocked.
        In doing so, a lock is placed on the row to prevent other clients acquiring it.
        Updates made to the values yielded will be written back to the storage row
        before the lock is released when the row goes out of client scope.

        :where: the simple column value criteria with which to make an exact match
        """
        # We are only interested in rows which are currently unlocked
        where[self._lock_col] = None
        for row_data in self._lock_and_yield_data_rows(**where):
            with DalDataRow(key=row_data[self._key_col],
                            values=row_data,
                            persist_func=self._update_and_release_row,
                            parse_value_func=self._parse_col_value,
                            hidden_cols=[self._key_col, self._lock_col]) as row:
                # Yield the generic DalDataRow to the client, with which it can read/write values.
                # The write_func will be called to persist any changes when we exit this context.
                yield row

    def acquire_row_by_key(self, key: str) -> _Generator[DalDataRow, any, None]:
        """
        Yields the row with the requested key if it is currently unlocked. A lock will be placed
        on the row which will be released when the row is persisted as it exits the current context.

        This method is for use when setting up new data rows, otherwise use acquire_next_row().
        Even though this func yields only 0 or 1 rows you should use it in a for loop as it allows
        the row's context manager to do its magic, saving any changes as the row leaves the context.
        Changes to the data may not be persisted to storage if used with next() or similar syntax.
        """
        yield from self.acquire_next_row(**dict([(self._key_col, key)]))

    def count_where(self, **where) -> int:
        """ Gets the current number of rows matching the passed where criteria. """
        # This is a sub-optimal default implementation. Ideally we override this in subclasses.
        return len(list(self.acquire_next_row(**where)))

    def _parse_col_value(self, col, value):
        """ Any additional parsing required to interpret values """
        # pylint: disable=unused-argument
        return value

    @_abstractmethod
    def add_row(self, key: str, **values):
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

    def count_where(self, **where) -> int:
        with self._CLIENT_LOCK:
            return sum(all(self._parse_col_value(c, row[c]) == v for c, v in where.items()) \
                        for row in self._table)

    def add_row(self, key, **values):
        with self._CLIENT_LOCK:
            # This will raise a ValueError if a row with the same key already exists.
            self._table.add_row({ self._key_col: key } | values)

    def _lock_and_yield_data_rows(self, **where):
        with self._CLIENT_LOCK:
            # Yes, it's a table scan but the table is in memory and not expected to be large.
            usable_lock_vals = (None, "", "None", self._lock_id)
            for row in self._table:
                test_vals = [self._parse_col_value(c, row[c]) for c in where]
                if all((tval == wval) or (c == self._lock_col and tval in usable_lock_vals) \
                                        for tval, (c, wval) in zip(test_vals, where.items())):
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

    def _parse_col_value(self, col: str, value):
        """ Handle additional parsing to interpret potentially masked values from QTable """
        if _np.ma.is_masked(value):
            # Value is explicitly masked which is expected when no value has been stored
            return None
        if _np.ma.isMaskedArray(value) or hasattr(value, "unmasked"):
            # We also seem to get masked value types, even when we have values
            return value.unmasked
        return super()._parse_col_value(col, value)


class QTableFileDal3(QTableDal3):
    """ Pipeline data access for reading/writing a file based on an astropy QTable """
    def __init__(self,
                 file: _Union[str, _Path],
                 file_format: str="ascii.fixed_width_two_line",
                 file_format_kwargs: dict[str, any] = None):
        """
        Initializes the QTableDal Dal class, associating it with a file in which to permanently
        store its data.

        Values are stored in the underlying data file in columns named for the param names used.
        This Dal also supports reading/writing UFloats and expects the nominal and std_dev
        components to be split across pairs of columns named as [param] and [param]_err.

        See https://docs.astropy.org/en/stable/io/unified_table.html for information on
        potential file formats and the kwargs that can be used to customize them.
        Will apply sensible defaults, of ["name", "dtype", "unit"], to header_rows if the file
        format 'ascii.fixed_width_two_line' is chosen but header_rows is not specified.

        This is not thread safe nor is it robust enough to be used where multiple clients expected.
        There is no storage locking mechanism so it will happily overwrite updates from elsewhere.
        If you need multiple clients and/or robust locking, other Dal subclasses are more suitable.

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


def create_dal(typename: _Union[str, type[Dal3]], **kwargs):
    """
    A factory method for creating a named Dal instance.

    :typename: the dal type to create
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
        # Must be careful with this check as we only ensuring the type is not itself abstract.
        # For this, __bases__ is better than issubclass() as it only looks at direct base types.
        raise ValueError(f"Cannot initialize the abstract class {dal_type.__name__}")

    # Ignore kwargs not used by the type's __init__ otherwise we may get a TypeError. This allows
    # calling code to send a superset of potential kwargs to this func & it will use what is needed.
    argspec = _getfullargspec(dal_type.__init__)
    expected_kwargs = { k: v for k, v in kwargs.items() if k in argspec.args and k not in ["self"] }
    return dal_type(**expected_kwargs)
