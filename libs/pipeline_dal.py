""" Data access components for reading/writing pipeline progress """
# pylint: disable=no-member
from typing import Union as _Union, Callable as _Callable, Generator as _Generator
from pathlib import Path as _Path
from numbers import Number as _Number
from abc import ABC as _ABC, abstractmethod as _abstractmethod
from warnings import warn as _warn
import threading

import numpy as _np
from numpy.typing import ArrayLike as _ArrayLike
import astropy.units as _u
from astropy.table import QTable as _QTable
from uncertainties import ufloat as _ufloat, UFloat as _UFloat
from uncertainties import nominal_value as _nom_val, std_dev as _std_dev


class Dal(_ABC):
    """ Base data access layer (Dal) for reading/writing simple table data """

    def __init__(self, key_name: str):
        """
        Initializes a Dal class which provides a consistent interface to underlying storage.

        :key_name: the name of the col/param which acts as each row's unique key/index value
        """
        self._key_name = key_name
        super().__init__()

    @property
    def key_name(self) -> str:
        """ Return the name of the primary key column """
        return self._key_name

    def yield_values(self, *params):
        """
        Yields the requested param values for each row in turn.

        :params: the names of the params to read values for
        :returns: a Generator of an array of the requested values for each row
        """
        for key in self.yield_keys():
            yield self.read_values(key, *params)

    @_abstractmethod
    def yield_keys(self,
                   *params: str,
                   where: _Callable[[any], bool]=lambda *vals: True) -> _Generator:
        """
        Yields key values where the where() func evaluates to True when passed the requested values.

        To yield all keys
        ```python
        key_gen = dal.yield_keys()
        ```
        
        To yield only keys where fitted_lcs and fitted_sed flags are True
        ```python
        key_gen = dal.yield_keys("fitted_lcs", "fitted_sed", where=lambda v1, v2: v1 == v2 == True)
        ```
        or, alternatively
        ```python
        key_gen = dal.yield_keys("fitted_lcs", "fitted_sed", where=lambda *vals: np.all(vals))
        ```

        :params: the parameters whose values are to be evaluated for each row
        :where: the bool function to evaluate the parameter values
        :return: yields key values for the rows where the where func evaluates to True
        """

    @_abstractmethod
    def read_values(self, key: any, *params: str) -> _ArrayLike:
        """
        Read the requested param values, for the required key, from the data source.

        :key: the unique key to the item to read params from
        :params: the names of the params to read values for
        :returns: an array of the requested values
        """

    @_abstractmethod
    def write_values(self, key: any, **params: dict[str, any]):
        """
        Writes the requested param values, for the required key, to the data source.

        :key: the unique key to the item to read params from
        :params: the name/value pairs of the params to write
        """


class QTableDal(Dal):
    """
    Pipeline Dal for reading/writing of an in memory astropy QTable
    """
    _col_dtype = [
        # SIMBAD and IDs
        ("target_id", "<U14"),
        ("main_id", "<U20"),
        ("tics", "<U40"),
        ("gaia_dr3_id", int),
        ("spt", "<U20"),
        # Gaia DR3 (coords falling back on SIMBAD)
        ("ra", float),
        ("ra_err", float),
        ("dec", float),
        ("dec_err", float),
        ("parallax", float),
        ("parallax_err", float),
        ("G_mag", float),
        ("V_mag", float),
        ("BP_mag", float),
        ("RP_mag", float),
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
        "G_mag": _u.mag,
        "V_mag": _u.mag,
        "BP_mag": _u.mag,
        "RP_mag": _u.mag,
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

    _WRITE_LOCK = threading.RLock()

    def __init__(self, masked: bool=True):
        """
        Initializes the QTableDal Dal class which uses an atropy QTable for storing the data.

        Values are stored in the underlying QTable in columns named for the param names used.
        This Dal also supports reading/writing UFloats and expects the nominal and std_dev
        components to be split across pairs of columns named as [param] and [param]_err.

        :masked: whether or not the table is masked
        """
        super().__init__(key_name="target_id")
        self._table = _QTable(masked=masked, dtype=self._col_dtype, units=self._col_units, rows=[])
        self._table.add_index(self.key_name, unique=True)

    def yield_keys(self,
                   *params: str,
                   where: _Callable[[any], bool]=lambda *vals: True) -> _Generator:
        if params is not None and len(params) > 0:
            for row in self._table:
                key = self._read_param_value(row, self._key_name)
                if where(*self.read_values(key, *params)):
                    yield key
        else:
            for row in self._table:
                yield self._read_param_value(row, self.key_name)

    def read_values(self, key: any, *params: str):
        if isinstance(params, str):
            params = [params]

        # No lock on a read. Raises an KeyError if the key value is unknown
        row = self._table.loc[key]
        values = _np.empty_like(params, dtype=object)
        for ix, col in enumerate(params):
            if col in row.colnames:
                value = self._read_param_value(row, col)
                if value is not None and (err_col := col + "_err") in row.colnames:
                    values[ix] = _ufloat(value, self._read_param_value(row, err_col))
                else:
                    values[ix] = value
        return values

    def write_values(self, key: any, **params: dict[str, any]):
        with self._WRITE_LOCK:
            # Unlike reading, writes need to be direct to the table (row/col indices) to persist
            try:
                row_ix = self._table.loc_indices[key]
            except KeyError:
                self._table.add_row(vals={ self._key_name: key })
                row_ix = self._table.loc_indices[key]

            for col, value in params.items():
                if col in self._table.colnames:
                    err_col = err_col if (err_col := col + "_err") in self._table.colnames else None
                    unit = self._table[col].unit if hasattr(self._table[col], "unit") else None
                    if isinstance(value, _UFloat):
                        self._table[row_ix][col] = _nom_val(value) * (unit or 1)
                        if err_col is not None:
                            self._table[row_ix][err_col] = _std_dev(value) * (unit or 1)
                        else:
                            _warn(f"Uncertainty for {col} wasn't saved as there is no err column")
                    else:
                        if isinstance(value, _u.Quantity):
                            self._table[row_ix][col] = value.to(unit)
                        elif value is None or unit is None:
                            # Covers no unit expected which includes non-numeric values,
                            # so the (unit or 1) coallescing trick would not work here
                            self._table[row_ix][col] = value
                        else:
                            self._table[row_ix][col] = value * (unit or 1)
                        if err_col is not None:
                            # Assume an err column is always going to be numeric
                            self._table[row_ix][err_col] = 0 * (unit or 1)
                else:
                    _warn(f"No column found named {col}. Value not saved.")

    def _read_param_value(self, row, param: str):
        """
        Read a param value from the row and handle all of the issues around units and masked values
        """
        value = row[param] if (row.columns[param].unit is None) else row[param].value
        if _np.ma.is_masked(value):
            # Value is explicitly masked which is expected when no value has been stored
            return None
        if _np.ma.isMaskedArray(value) or hasattr(value, "unmasked"):
            # We also seem to get masked value types, even when we have values
            return value.unmasked
        return value


class QTableFileDal(QTableDal):
    """ Pipeline data access for reading/writing a file based on an astropy QTable """
    def __init__(self,
                 file: _Union[str, _Path],
                 file_format: str="ascii.fixed_width_two_line",
                 **file_format_kwargs):
        """
        Initializes the QTableDal Dal class, associating it with the file which is storing the data.

        Values are stored in the underlying data file in columns named for the param names used.
        This Dal also supports reading/writing UFloats and expects the nominal and std_dev
        components to be split across pairs of columns named as [param] and [param]_err.

        See https://docs.astropy.org/en/stable/io/unified_table.html for information on
        potential file formats and the kwargs that can be used to customize them.
        Will apply sensible defaults, of ["name", "dtype", "unit"], to header_rows if the file
        format 'ascii.fixed_width_two_line' is chosen but header_rows is not specified.

        This is not thread safe nor is it robust enough to be used where multiple clients expected.
        There is no locking mechanism so it will happily overwrite updates from elsewhere. If you
        need multiple clients, large datasets or more durable storage, use a "real" database Dal.

        :file: the file name of the storage file
        :file_format: the format of the file
        :file_format_kwargs: optional kwargs specific to each file_format
        """
        if file_format == "ascii.fixed_width_two_line" and "header_rows" not in file_format_kwargs:
            file_format_kwargs["header_rows"] = ["name", "dtype", "unit"]

        self._file = file if isinstance(file, _Path) else _Path(file)
        self._format = file_format
        self._format_kwargs = file_format_kwargs

        with self._WRITE_LOCK:
            # Inits a new in-memory table which we can save (no file yet) or overwrite (file exists)
            super().__init__()
            if file.exists():
                print(f"Loading data file '{file.name}' as {file_format}/{file_format_kwargs}")
                self._table = _QTable.read(self._file, format=self._format, **self._format_kwargs)
                self._table.add_index(self.key_name, unique=True)
            else:
                # Cannot save the file immediately as we get a ValueError thrown with the message
                # "max() arg is an empty sequence" from within astropy. Probably because there
                # is no data yet and astropy isn't checking for list contents before calling max().
                print(f"New data file '{file.name}', as {file_format}/{file_format_kwargs},",
                      "will be created when values are first written to it.")
                # self._table.write(self._file, format=self._format, **self._format_kwargs)

    def write_values(self, key, **params):
        with self._WRITE_LOCK:
            super().write_values(key, **params)
            self._table.write(self._file, overwrite=True, format=self._format,**self._format_kwargs)
