""" A class for handling the generation of model fluxes for filters sourced from Pyssed data """
# pylint: disable=no-member
from typing import Union, Tuple, Iterable
from pathlib import Path as _Path
from inspect import getsourcefile as _getsourcefile
from json import load as _json_load

import numpy as np
import astropy.units as u

from scipy.interpolate import RegularGridInterpolator as _RegularGridInterpolator

class ModelSed():
    """
    Generates model SED fluxes from pyssed model
    """

    def __init__(self, data_file: Union[_Path, str]=None):
        """
        Initializes a new instance of this class.

        :data_file: the source of the model data, in pyssed dat format
        """
        this_dir = _Path(_getsourcefile(lambda:0)).parent
        if data_file is None:
            data_file = this_dir / "data/pyssed/model-bt-settl-recast.dat"
        self._data_file = data_file
        self._flux_unit = u.Jy

        # Read the pre-built model file. The delete chars is to handle some of the
        # column/filter names which contain spaces and grammar chars.
        model_grid = np.genfromtxt(self._data_file, names=True, delimiter=" ",
                                   deletechars=r" ~!@#$%^&*()=+~\|]}[{';: ?>,<")
        model_grid = model_grid[model_grid["alpha"] == 0]

        # Should already be in this order, but just in case as we depend on this order below
        model_grid.sort(order=["teff", "logg", "metal"])

        # The cols 0 to 4 are expected to be teff, logg, metal, alpha and lum.
        # The rest of the cols are the filters and corresponding fluxes.
        teffs, teff_ixs = np.unique(model_grid["teff"], return_inverse=True)
        loggs, logg_ixs = np.unique(model_grid["logg"], return_inverse=True)
        metals, metal_ixs = np.unique(model_grid["metal"], return_inverse=True)
        filter_names = list(model_grid.dtype.names)[5:]

        # Set up a table of interpolators, one per filter. Each interpolator is based on
        # a pivot table with the teffs and loggs as the axes and filter fluxes as the values.
        self._model_interps = np.empty(shape=(len(filter_names), ),
                                       dtype=[("filter", object), ("interp", object)])
        for filter_ix, filter_name in enumerate(filter_names):
            # Need model_grid[filter_name] as teffs*loggs*metals items to write tl_pivot this way
            tl_pivot = np.zeros((teffs.shape[0], loggs.shape[0], metals.shape[0]),
                                dtype=model_grid[filter_name].dtype)
            tl_pivot[teff_ixs, logg_ixs, metal_ixs] = model_grid[filter_name]

            interp = _RegularGridInterpolator((teffs, loggs, metals), tl_pivot, "linear")
            self._model_interps[filter_ix] = (filter_name, interp)
        del model_grid

        self._wavelength_range = (0.3, 22) << u.micron
        self._model_teff_range = (min(teffs), max(teffs)) << u.K
        self._model_logg_range = (min(loggs), max(loggs)) << u.dex
        self._model_metal_range = (min(metals), max(metals)) << u.dimensionless_unscaled

        # Lookup for translating the SED service filter names into those used here & within the dat
        with open(this_dir / "data/pyssed/sed-filter-translation.json", "r", encoding="utf8") as j:
            self._sed_filter_name_map = _json_load(j)

    @property
    def data_file(self) -> _Path:
        """ Gets the Path of the data file being used. """
        return self._data_file

    @property
    def num_interpolators(self) -> int:
        """ Gets the number of interpolators covering this model """
        return self._model_interps.shape[0]

    @property
    def wavelength_range(self) -> u.Quantity["length"]:
        """ Gets the range of wavelength covered by this model """
        return self._wavelength_range

    @property
    def teff_range(self) -> u.Quantity["temperature"]:
        """ Gets the range of effective temperatures covered by this model """
        return self._model_teff_range

    @property
    def logg_range(self) -> u.Dex:
        """ Gets the range of logg covered by this model """
        return self._model_logg_range

    @property
    def metal_range(self) -> u.Quantity:
        """ Gets the range of metallicities covered by this model """
        return self._model_metal_range

    @property
    def flux_unit(self) -> u.Unit:
        """ Gets the unit of the returned fluxes """
        return self._flux_unit

    def map_filter_name(self, name: str) -> str:
        """ Get the name of the equivalent filter within this model """        
        if name in self._model_interps["filter"]:
            return name
        # We want an index error if the mapping does not exist
        return self._sed_filter_name_map[name]

    def has_filter(self, name: str) -> bool:
        """ Gets whether this model knows of the requested filter """
        return name in self._sed_filter_name_map or name in self._sed_filter_name_map.values()

    def get_filter_indices(self, filter_names: Iterable[str]) -> np.ndarray[int]:
        """
        Get the indices of the given filters. Useful in optimizing filter access when iterating
        as the indices can be used in place of the names. Handles mapping filter names.

        Will raise a KeyError if a filter is unknown.

        :filter_names: a list of filters for which we want the indices
        :returns: an array of the equivalent indices
        """
        if isinstance(filter_names, str):
            filter_names = [filter_names]
        ixs = [np.where(self._model_interps['filter'] == self.map_filter_name(f))[0]
                for f in filter_names]
        return np.array(ixs, dtype=int).squeeze(axis=1)

    def get_fluxes(self,
                   filters: Union[np.ndarray[str], np.ndarray[int]],
                   teff: float,
                   logg: float,
                   metal: float=0.) -> u.Quantity:
        """
        Will return a ndarray of flux values calculated for requested filter names at
        the chosen effective temperature, logg and metallicity values.

        Will raise a KeyError if a named filter is unknown.
        Will raise IndexError if an indexed filter is out of range.

        :filters: a list of filter names or indices for which we are generating fluxes
        :teff: the effective temperature for the fluxes
        :logg: the logg for the fluxes
        :metal: the metallicity for the fluxes
        :returns: the resulting flux values (in the units of the underlying data file)
        """
        # Find the unique filters and the map onto the request/response (a filter can appear > once)
        if isinstance(filters, (str|int)):
            unique_filters, flux_mappings = np.array([filters]), np.array([0])
        else:
            unique_filters, flux_mappings = np.unique(filters, return_inverse=True)

        # Get the fluxes once for each of the unique filters
        if unique_filters.dtype not in (np.int64, np.int32):
            unique_filters = self.get_filter_indices(unique_filters)
        xi = (teff, logg, metal)
        fluxes = [self._model_interps[filter]["interp"](xi=xi) for filter in unique_filters]

        # Map these fluxes onto the response, where a filter/flux may appear >1 times
        return np.array([fluxes[m] for m in flux_mappings], dtype=float) << self.flux_unit
