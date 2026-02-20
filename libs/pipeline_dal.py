""" Data access components for reading/writing pipeline progress """
from typing import Union as _Union, Callable as _Callable, Generator as _Generator
from pathlib import Path as _Path
from numbers import Number as _Number
from abc import ABC as _ABC, abstractmethod as _abstractmethod
from warnings import warn as _warn

import numpy as _np
from numpy.typing import ArrayLike as _ArrayLike
from astropy.units import Quantity as _Quantity
from uncertainties import ufloat as _ufloat, UFloat as _UFloat
from uncertainties import nominal_value as _nom_val, std_dev as _std_dev

class Dal(_ABC):
    """ Base data access layer (Dal) for reading/writing simple table data """
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

    @_abstractmethod
    def keys_where(self,
                   param: str,
                   where: _Callable[[any], bool]=lambda val: val is True) -> _Generator:
        """
        Yields key values where the where() func evaluate to True

        :param: the parameter to evaluate
        :where: the function to evaluate the parameter
        :return: yields key values for the rows where the where func evaluates to True
        """

class QTableDal(Dal):
    """
    Pipeline data access for reading/writing a file based on an astropy QTable
    """
    # pylint: disable=import-outside-toplevel
    from astropy.table import QTable as _QTable
    import threading

    WRITE_LOCK = threading.Lock()

    def __init__(self,
                 file: _Union[str, _Path],
                 key_name: str="target",
                 file_format: str="votable",
                 **file_format_kwargs):
        """
        Initializes the QTableDal Dal class, associating it with the file which is storing the data.

        Values are stored in the underlying data file in columns named for the param names used.
        This Dal also supports reading/writing UFloats and expects the nominal and std_dev
        components to be split across pairs of columns named as [param] and [param]_err.

        See https://docs.astropy.org/en/stable/io/unified_table.html for information on
        potential file formats and the kwargs that can be used to customize them.

        This is not thread safe nor is it robust enough to be used where multiple clients expected.
        There is no locking mechanism so it will happily overwrite updates from elsewhere. If you
        need multiple clients, large datasets or more durable storage, use a real database.

        :file: the file name of the storage file
        :key_name: the name of the col/param which acts as each row's unique key/index value
        :file_format: the format of the file
        :file_format_kwargs: optional kwargs specific to each file_format
        """
        self._file = file if isinstance(file, _Path) else _Path(file)
        self._key_name = key_name
        self._format = file_format
        self._format_kwargs = file_format_kwargs
        print(f"Opening data file '{file.name}'",
              f"with expected format '{file_format}' and kwargs {file_format_kwargs}")
        super().__init__()

    def keys_where(self,
                   param: str,
                   where: _Callable[[any], bool]=lambda val: val is True) -> _Generator:
        table = self._QTable.read(self._file, format=self._format, **self._format_kwargs)
        for row in table:
            value = self._read_param_value(row, param)
            if where(value):
                yield self._read_param_value(row, self._key_name)

    def read_values(self, key: any, *params: str):
        if isinstance(params, str):
            params = [params]

        # No lock on a read
        table = self._QTable.read(self._file, format=self._format, **self._format_kwargs)
        row_mask = table[self._key_name] == key
        if not _np.any(row_mask):
            raise ValueError(f"No data row found for key {key}")

        row = table[_np.where(row_mask)[0]][0]
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
        # TODO: can we put a lock on the file itself?
        with self.WRITE_LOCK:
            table = self._QTable.read(self._file, format=self._format, **self._format_kwargs)
            row_mask = table[self._key_name] == key
            if not _np.any(row_mask):
                raise IndexError(f"No data row found for key {key}. Values not saved.")

            row_ix = _np.where(row_mask)[0][0]
            for col, value in params.items():
                if col in table.colnames:
                    err_col = err_col if (err_col := col + "_err") in table.colnames else None
                    unit = table[col].unit if hasattr(table[col], "unit") else None
                    if isinstance(value, _UFloat):
                        table[row_ix][col] = _nom_val(value) * (unit or 1)
                        if err_col is not None:
                            table[row_ix][err_col] = _std_dev(value) * (unit or 1)
                        else:
                            _warn(f"Uncertainty for {col} wasn't saved as there is no err column")
                    else:
                        if isinstance(value, _Quantity):
                            table[row_ix][col] = value.to(unit)
                        elif value is None or unit is None:
                            # Covers no unit expected which includes non-numeric values,
                            # so the (unit or 1) coallescing trick would not work here
                            table[row_ix][col] = value
                        else:
                            table[row_ix][col] = value * (unit or 1)
                        if err_col is not None:
                            # Assume an err column is always going to be numeric
                            table[row_ix][err_col] = 0 * (unit or 1)
                else:
                    raise IndexError(f"No column found named {col}. Values not saved.")

            table.write(self._file, format=self._format, overwrite=True, **self._format_kwargs)

    def _read_param_value(self, row, param: str):
        """
        Read a param value from the row and handle all of the issues around units and masked values
        """
        has_unit = row.columns[param].unit is not None
        value = row[param].value if has_unit else row[param]
        if hasattr(value, "unmasked"): # Bit of a hack but np.ma.is_masked doesn't work here
            value = value.unmasked
        return value
