""" A class for handling the generation of model fluxes for filters sourced from Pyssed data """
# pylint: disable=no-member
from typing import Union, Iterable
from pathlib import Path as _Path
from inspect import getsourcefile as _getsourcefile

import numpy as np
import astropy.units as _u

from scipy.interpolate import RegularGridInterpolator as _RegularGridInterpolator

class ModelSed():
    """
    Generates model SED fluxes from pre-built pyssed model_grid.
    This uses a grid of interpolators based on pre-convolved filter fluxes. The pre-convolved
    nature of the fluxes ensures minimal processing is required to produce a model sed, so
    it's fast. However, this model is only suitable for SED observations which are dereddened.
    """

    def __init__(self, data_file: Union[_Path, str]=None):
        """
        Initializes a new instance of this class.

        :data_file: the source of the model data, in numpy npz format
        """
        this_dir = _Path(_getsourcefile(lambda:0)).parent
        if data_file is None:
            data_file = this_dir / "data/pyssed/model-bt-settl-recast.npz"
        self._data_file = data_file

        with np.load(self._data_file, allow_pickle=True) as df:
            # Load the model_grid of pre-convolved filter fluxes. These fluxes are not reddened
            # and are suitable for quickly fitting to a SED which has been dereddened.
            model_grid_conv = df["model_grid_conv"]
            ranges = df["ranges"]
            units = df["units"]

        # Code to populate pivots below depends on fixing to alpha==0 and this ordering of fluxes
        model_grid_conv = model_grid_conv[model_grid_conv["alpha"]==0]
        model_grid_conv.sort(order=["teff", "logg", "metal"])

        # The cols 0 to 3 are expected to be teff, logg, metal and alpha
        # The rest of the cols are the filters and corresponding fluxes.
        teffs, teff_ixs = np.unique(model_grid_conv["teff"], return_inverse=True)
        loggs, logg_ixs = np.unique(model_grid_conv["logg"], return_inverse=True)
        metals, metal_ixs = np.unique(model_grid_conv["metal"], return_inverse=True)
        filter_names = list(model_grid_conv.dtype.names)[4:]

        # Set up a table of interpolators, one per filter. Each interpolator is based on a
        # pivot table with the teffs, loggs and metals as the axes and filter fluxes as the values.
        self._model_interps = np.empty(shape=(len(filter_names), ),
                                       dtype=[("filter", object), ("interp", object)])
        for filter_ix, filter_name in enumerate(filter_names):
            # Need model_grid[filter_name] of teffs*loggs*metals items long and to be sorted by
            # [teff, logg, metal] in order to write to the pivot directly like this. If model_table
            # not limited to alpha==0 then we will have to include the extra dimension.
            pivot_table = np.empty(shape=(teffs.shape[0], loggs.shape[0], metals.shape[0]),
                                   dtype=model_grid_conv[filter_name].dtype)
            pivot_table[teff_ixs, logg_ixs, metal_ixs] = model_grid_conv[filter_name]

            interp = _RegularGridInterpolator((teffs, loggs, metals), pivot_table, "linear")
            self._model_interps[filter_ix] = (filter_name, interp)
        del model_grid_conv

        self._filter_names = filter_names
        self._wavelength_range = ranges[ranges["column"]=="lambda"]["range"][0]
        self._model_teff_range = ranges[ranges["column"]=="teff"]["range"][0]
        self._model_logg_range = ranges[ranges["column"]=="logg"]["range"][0]
        self._model_metal_range = ranges[ranges["column"]=="metal"]["range"][0]
        self._flux_unit = units[units["column"]=="flux"]["unit"][0]

    @property
    def data_file(self) -> _Path:
        """ Gets the Path of the data file being used. """
        return self._data_file

    @property
    def num_interpolators(self) -> int:
        """ Gets the number of interpolators covering this model """
        return self._model_interps.shape[0]

    @property
    def wavelength_range(self) -> _u.Quantity["length"]:
        """ Gets the range of wavelength covered by this model """
        return self._wavelength_range

    @property
    def teff_range(self) -> _u.Quantity["temperature"]:
        """ Gets the range of effective temperatures covered by this model """
        return self._model_teff_range

    @property
    def logg_range(self) -> _u.Quantity:
        """ Gets the range of logg covered by this model """
        return self._model_logg_range

    @property
    def metal_range(self) -> _u.Quantity:
        """ Gets the range of metallicities covered by this model """
        return self._model_metal_range

    @property
    def flux_unit(self) -> _u.Unit:
        """ Gets the unit of the returned fluxes """
        return self._flux_unit

    def has_filter(self, name: str) -> bool:
        """ Gets whether this model knows of the requested filter """
        return name in self._filter_names

    def get_filter_indices(self, filter_names: Iterable[str]) -> np.ndarray[int]:
        """
        Get the indices of the given filters. Useful in optimizing filter access when iterating
        as the indices can be used in place of the names. Handles mapping filter names.

        Will raise a ValueError if a filter is unknown.

        :filter_names: a list of filters for which we want the indices
        :returns: an array of the equivalent indices
        """
        if isinstance(filter_names, str):
            filter_names = [filter_names]
        return np.array([self._filter_names.index(n) for n in filter_names], dtype=int)

    def get_fluxes(self,
                   filters: Union[np.ndarray[str], np.ndarray[int]],
                   teff: float,
                   logg: float,
                   metal: float=0.,
                   as_quantity: bool=False) -> Union[np.ndarray[float], _u.Quantity]:
        """
        Will return a ndarray of flux values calculated for requested filter names at
        the chosen effective temperature, logg and metallicity values.

        Will raise a ValueError if a named filter is unknown.
        Will raise IndexError if an indexed filter is out of range.

        :filters: a list of filter names or indices for which we are generating fluxes
        :teff: the effective temperature for the fluxes
        :logg: the logg for the fluxes
        :metal: the metallicity for the fluxes
        :as_quantity: whether to return the fluxes as a Quantity (True) or a ndarray[float] (False)
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
        values = np.array([fluxes[m] for m in flux_mappings], dtype=float)
        if as_quantity:
            return values << self.flux_unit
        return values
