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
        self._flux_unit = u.Jy

        # Read the pre-built model file. The delete chars is to handle some of the
        # column/filter names which contain spaces and grammar chars.
        model_grid = np.genfromtxt(data_file, names=True, delimiter=" ",
                                   deletechars=r" ~!@#$%^&*()=+~\|]}[{';: ?>,<")

        # For now we're only interested in the solar metallicity model fluxes
        model_grid = model_grid[(model_grid["metal"] == 0) & (model_grid["alpha"] == 0)]

        # Should already be in this order, but just in case as we depend on this order below
        model_grid.sort(order=["teff", "logg"])

        # The cols 0 to 4 are expected to be teff, logg, metal, alpha and lum.
        # The rest of the cols are the filters and corresponding fluxes.
        teffs, teff_ixs = np.unique(model_grid["teff"], return_inverse=True)
        loggs, logg_ixs = np.unique(model_grid["logg"], return_inverse=True)
        filter_names = list(model_grid.dtype.names)[5:]

        # Set up a table of interpolators, one per filter. Each interpolator is based on
        # a pivot table with the teffs and loggs as the axes and filter fluxes as the values.
        self._model_interps = np.empty(shape=(len(filter_names), ),
                                       dtype=[("filter", object), ("interp", object)])
        for filter_ix, filter_name in enumerate(filter_names):
            # Writing tl_pivot this way only works if model_grid[filter_name] has teffs*loggs #items
            tl_pivot = np.zeros((len(teffs), len(loggs)), dtype=model_grid[filter_name].dtype)
            tl_pivot[teff_ixs, logg_ixs] = model_grid[filter_name]

            interp = _RegularGridInterpolator((teffs, loggs), tl_pivot, "linear")
            self._model_interps[filter_ix] = (filter_name, interp)
        del model_grid

        self._wavelength_range = (0.3, 22) * u.micron
        self._model_teff_range = (min(teffs), max(teffs)) * u.K
        self._model_logg_range = (min(loggs), max(loggs)) * u.dex

        # Lookup for translating the SED service filter names into those used here
        with open(this_dir / "data/pyssed/sed-filter-translation.json", "r", encoding="utf8") as j:
            self._sed_filter_name_map = _json_load(j)

    @property
    def wavelength_range(self) -> u.Quantity:
        """ Gets the range of wavelength covered by this model """
        return self._wavelength_range

    @property
    def teff_range(self) -> u.Quantity:
        """ Gets the range of effective temperatures covered by this model """
        return self._model_teff_range

    @property
    def logg_range(self) -> u.Quantity:
        """ Gets the range of logg covered by this model """
        return self._model_teff_range

    @property
    def flux_unit(self) -> u.Unit:
        """ Gets the unit of the returned fluxes """
        return self._flux_unit

    def map_filter_name(self, name: str) -> str:
        """ Get the name of the equivalent filter within this model """        
        # We want an index error if the mapping does not exist
        return self._sed_filter_name_map[name]

    def has_filter(self, name: str) -> bool:
        """ Gets whether this model knows of the requested filter """
        return name in self._sed_filter_name_map or name in self._sed_filter_name_map.values()

    def get_filter_interpolators_and_mappings(self,
                                              filter_names: Iterable[str]) \
            -> Tuple[Iterable[_RegularGridInterpolator], Iterable[int]]:
        """
        This is a rather dangerous convenience/optimization which is coupled to
        get_model_fluxes_from_mappings(). Hopefully I can drop this!

        It takes the passed list of filters and returns a list of interpolators, one for each unique
        filter listed, and a second list of indices which map these back to the original list.

        This allows us to avoid repeating this lookup/mapping within a MCMC run.

        :filter_names: a list of filters to locate and map
        :returns: (array of interpolators, array the mappings from interpolators to the input list)
        """
        # Here np.unique return 2 arrays; one of unique filter names & another of indices mapping
        # them onto input. We use the first to locate & list the interp corresponding to each unique
        # filter and the second to map these onto the >=1 locations within the input they're used.
        unique_names, input_map = np.unique(filter_names, return_inverse=True)
        interps = np.empty(len(unique_names), dtype=object)
        for filter_ix, name in enumerate(unique_names):
            if name not in self._model_interps["filter"]:
                # Will raise IndexError if a filter is unknown
                name = self.map_filter_name(name)

            # TODO: what about if the mapped name is not in the model?
            interp = self._model_interps[self._model_interps["filter"] == name]["interp"][0]
            interps[filter_ix] = interp
        return interps, input_map

    def get_fluxes_from_mappings(self,
                                 filter_interps: Iterable[_RegularGridInterpolator],
                                 flux_mappings: Iterable[int],
                                 teff: float,
                                 logg: float) -> u.Quantity:
        """
        Will return a ndarray of flux values calculated by the filters corresponding to the
        interpolators. The filter_interps and flux_mappings are effectively the return values
        from the get_filter_interpolators_and_mappings() function. This is separated out to avoid
        repeating the same lookup for every attempted fit.

        It's not obvious in the data file, but the scale of the values implies the fluxes are in Jy

        :filter_interps: unique list of interpolators to use to generate flux values
        :flux_mappings: mapping indices from the interpolators onto the output
        :teff: the effective temperature for the fluxes
        :logg: the logg for the fluxes
        :returns: the resulting flux values (in the units of the underlying data file)
        """
        # Generate each unique flux value
        xi = (teff, logg)
        fluxes_by_filter = np.empty((len(filter_interps)), dtype=float)
        for filter_flux_ix, filter_interp in enumerate(filter_interps):
            fluxes_by_filter[filter_flux_ix] = filter_interp(xi=xi)

        # Copy the fluxes to the output via the mappings
        return_fluxes = np.empty((len(flux_mappings)), dtype=float)
        for flux_ix, flux_mapping in enumerate(flux_mappings):
            return_fluxes[flux_ix] = fluxes_by_filter[flux_mapping]
        return return_fluxes * self.flux_unit

    def get_fluxes(self,
                   filter_names: Iterable[str],
                   teff: float,
                   logg: float) -> u.Quantity:
        """
        Will return a ndarray of flux values calculated for requested filter names at
        the chosen effective temperature and logg values.

        :filter_names: a list of filters for which we are generating fluxes
        :teff: the effective temperature for the fluxes
        :logg: the logg for the fluxes
        :returns: the resulting flux values (in the units of the underlying data file)
        """
        filter_iterps, mappings = self.get_filter_interpolators_and_mappings(filter_names)
        return self.get_fluxes_from_mappings(filter_iterps, mappings, teff, logg)
