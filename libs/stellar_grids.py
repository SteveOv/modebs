""" A class for handling the generation of model fluxes for filters sourced from bt-settl data """
# pylint: disable=no-member
from abc import ABC as _AbstractBaseClass, abstractmethod as _abstractmethod
from typing import Union as _Union, Tuple as _Tuple, Iterable as _Iterable, List as _List
from pathlib import Path as _Path
from inspect import getsourcefile as _getsourcefile
from warnings import filterwarnings as _filterwarnings
import re as _re
from json import load as _json_load
from urllib.parse import quote_plus as _quote_plus
from itertools import product as _product

import numpy as _np
from numpy.typing import ArrayLike as _ArrayLike

from scipy.stats import binned_statistic as _binned_statistic
from scipy.interpolate import RegularGridInterpolator as _RegularGridInterpolator

import astropy.units as _u
from astropy.table import Table as _Table
from astropy.io.votable import parse_single_table as _parse_single_table


# We parse units as text from votables & text files. Stop us getting swamped format with warnings.
_filterwarnings("ignore", category=_u.UnitsWarning)

class StellarGrid(_AbstractBaseClass):
    """ Base for classes which expose stellar fluxes """
    # pylint: disable=too-many-arguments, too-many-positional-arguments

    _this_dir = _Path(_getsourcefile(lambda:0)).parent
    _CACHE_DIR = _this_dir / "../.cache"
    _DEF_FILTER_MAP_FILE = _this_dir / "data/stellar_grids/sed-filter-mappings.json"

    # Default output units
    _LAM_UNIT = _u.um
    _FLUX_DENSITY_UNIT = _u.W / _u.m**2 / _u.Hz
    _FLUX_UNIT = _u.W / _u.m**2
    _TEFF_UNIT = _u.K
    _LOGG_UNIT = _u.dex

    def __init__(self,
                 model_grid_full: _ArrayLike,
                 model_grid_filtered: _ArrayLike,
                 index_values: _ArrayLike,
                 wavelengths: _ArrayLike,
                 filter_names: _ArrayLike):
        """
        Initializes a new instance of this class.

        :model_grid_full: full grid of un-filtered fluxes over a range of teff, logg & metal values
        :model_grid_filtered: grid of pre-filtered fluxes over a range of teff, logg & metal values
        :index_values: structured array of the teff, logg and metal row index values for both grids
        :wavelengths: the wavelength column index values of the full grid
        :filter_names: the filter column index values of the pre-filtered grid
        """
        # Below we'll be building interpolators over multi-D arrays of fluxes, with
        # the axes being the teffs, loggs and metals (and in the future alphas).
        # The saved input arrays are currently flat, and for the following code to work
        # they must be sorted by teff, logg & metal with each column being teffs*loggs*metals long.
        sorted_rows = _np.argsort(index_values)

        # Index points common to both types of interpolator.
        self._teffs = _np.unique(index_values["teff"])
        self._loggs = _np.unique(index_values["logg"])
        self._metals = _np.unique(index_values["metal"])
        index_points = (self._teffs, self._loggs, self._metals)

        # Create the single interpolator over the full grid of flux data. Used for the interpolation
        # of the full spectrum of fluxes (over the wavelength range) for given teff, logg and metal.
        full_vals_shape = (len(self._teffs), len(self._loggs), len(self._metals), len(wavelengths))
        interp_fluxes = model_grid_full[sorted_rows].reshape(full_vals_shape)
        self._model_full_interp = _RegularGridInterpolator(index_points, interp_fluxes, "linear")

        # Create a table of interpolators, one per filter, for interpolating filter fluxes
        # for each filter for give teff, logg and metal values.
        self._model_interps = None
        if model_grid_filtered is not None:
            interp_vals_shape = (len(self._teffs), len(self._loggs), len(self._metals))
            self._model_interps = _np.empty(shape=(len(filter_names), ),
                                            dtype=[("filter", "<U50"), ("interp", object)])
            for filter_ix, filter_name in enumerate(filter_names):
                interp_vals = model_grid_filtered[sorted_rows, filter_ix].reshape(interp_vals_shape)
                interp = _RegularGridInterpolator(index_points, interp_vals, "linear")
                self._model_interps[filter_ix] = (filter_name, interp)

        if isinstance(filter_names, _np.ndarray):
            self._filter_names_list = filter_names.tolist() # a list so we can use index()
        else:
            self._filter_names_list = filter_names
        self._wavelengths = wavelengths

    @property
    def wavelengths(self) -> _np.ndarray:
        """ Gets the wavelength values for which unfiltered fluxes are published. """
        return self._wavelengths

    @property
    def wavelength_range(self) -> _Tuple[float]:
        """ Gets the range of wavelength covered by this model (units of wavelength_unit)"""
        return (self.wavelengths.min(), self.wavelengths.max())

    @property
    def teff_range(self) -> _Tuple[float]:
        """ Gets the range of effective temperatures covered by this model (units of teff_unit) """
        return (self._teffs.min(), self._teffs.max())

    @property
    def logg_range(self) -> _Tuple[float]:
        """ Gets the range of logg covered by this model (units of logg_unit) """
        return (self._loggs.min(), self._loggs.max())

    @property
    def metal_range(self) -> _Tuple[float]:
        """ Gets the range of metallicities covered by this model """
        return (self._metals.min(), self._metals.max())

    @property
    def teff_unit(self) -> _u.Unit:
        """ Gets the temperature units """
        return self._TEFF_UNIT

    @property
    def logg_unit(self) -> _u.Unit:
        """ Gets the logg units """
        return self._LOGG_UNIT

    @property
    def wavelength_unit(self) -> _u.Unit:
        """ Gets the unit of the flux wavelengths """
        return self._LAM_UNIT

    @property
    def flux_unit(self) -> _u.Unit:
        """ Gets the unit of the returned fluxes """
        return self._FLUX_UNIT

    def has_filter(self, filter_name: _Union[str, _Iterable]) -> _np.ndarray[bool]:
        """ Gets whether this model knows of the requested filter(s) """
        return _np.isin(filter_name, self._filter_names_list)

    def get_filter_indices(self, filter_names: _Union[str, _Iterable]) -> _np.ndarray[int]:
        """
        Get the indices of the given filters. Useful in optimizing filter access when iterating
        as the indices can be used in place of the names. Handles mapping filter names.

        Will raise a ValueError if a filter is unknown.

        :filter_names: a list of filters for which we want the indices
        :returns: an array of the equivalent indices
        """
        if isinstance(filter_names, str):
            filter_names = [filter_names]
        return _np.array([self._filter_names_list.index(n) for n in filter_names], dtype=int)

    def get_fluxes(self, teff: float, logg: float, metal: float=0) \
                                                        -> _Union[_np.ndarray[float], _u.Quantity]:
        """
        Will return a full spectrum of fluxes, over this model's wavelength range for the
        requested teff, logg and metal values.

        :teff: the effective temperature for the fluxes
        :logg: the logg for the fluxes
        :metal: the metallicity for the fluxes
        :returns: the resulting flux values (in the units of the underlying data file)
        """
        return self._model_full_interp(xi=(teff, logg, metal))

    def get_filter_fluxes(self,
                          filters: _ArrayLike,
                          teff: float,
                          logg: float,
                          metal: float=0.,
                          as_quantity: bool=False) -> _Union[_np.ndarray[float], _u.Quantity]:
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
            unique_filters, flux_mappings = _np.array([filters]), _np.array([0])
        else:
            unique_filters, flux_mappings = _np.unique(filters, return_inverse=True)

        # Get the fluxes once for each of the unique filters
        if unique_filters.dtype not in (_np.int64, _np.int32):
            unique_filters = self.get_filter_indices(unique_filters)
        xi = (teff, logg, metal)
        fluxes = [self._model_interps[filter]["interp"](xi=xi) for filter in unique_filters]

        # Map these fluxes onto the response, where a filter/flux may appear >1 times
        values = _np.array([fluxes[m] for m in flux_mappings], dtype=float)
        if as_quantity:
            return values << self.flux_unit
        return values


    @classmethod
    def get_filter(cls, svo_name: str, lambda_unit: _u.Unit) -> _Table:
        """
        Downloads and caches the requested filter from the SVO. Returns a table of the filter's
        Wavelength and Transmission fields, and adds a Norm-Transmission column.
        Will also add meta entries for filter_short, filter_long and filter_mid to record
        the wavelength range covered by the filter.

        :svo_name: the unique name of the filter given by the SVO
        :lambda_unit: the wavelength unit for the Wavelength column
        :returns: and astropy Table with Wavelength, Transmission and Norm-Transmission columns
        """
        filter_cache_dir = cls._CACHE_DIR / ".filters/"
        filter_cache_dir.mkdir(parents=True, exist_ok=True)

        filter_fname = (filter_cache_dir / (_re.sub(r"[^\w\d.-]", "-", svo_name) + ".xml"))
        if not filter_fname.exists():
            try:
                fid = _quote_plus(svo_name)
                table = _Table.read(f"https://svo2.cab.inta-csic.es/theory/fps/fps.php?ID={fid}")
                table.write(filter_fname, format="votable")
            except ValueError as err:
                raise ValueError(f"No filter table in SVO for filter={svo_name}") from err

        table = _parse_single_table(filter_fname).to_table()

        # Create a normalized copy of the transmission data
        ftrans = table["Transmission"]
        table["Norm-Transmission"] = (ftrans - ftrans.min()) / (ftrans.max() - ftrans.min())

        # Add metadata on the filter coverage
        if table["Wavelength"].unit != lambda_unit:
            table["Wavelength"] = table["Wavelength"].to(lambda_unit, equivalencies=_u.spectral())
        table.meta["filter_short"] = _np.min(table["Wavelength"].quantity)
        table.meta["filter_long"] = _np.max(table["Wavelength"].quantity)
        table.meta["filter_mid"] = _np.median(table["Wavelength"].quantity)

        table.sort("Wavelength")
        return table

    @classmethod
    def _get_filtered_flux_total(cls,
                                 lambdas: _ArrayLike,
                                 fluxes: _ArrayLike,
                                 filter_table: _Table) -> _u.Quantity:
        """
        Calculate the total flux across a filter's bandpass.

        :lambdas: the wavelengths of the model fluxes
        :fluxes: the model fluxes
        :filter_grid: the grid (as returned by get_filter()) which describes the filter
        :returns: the summed flux passed through the filter
        """
        # Work out the lambda range where the filter and binned data overlap
        ol_lam_short = max(lambdas.min(), filter_table.meta["filter_short"])
        ol_lam_long = min(lambdas.max(), filter_table.meta["filter_long"])

        if ol_lam_short > ol_lam_long: # No overlap; no flux
            return 0.0 * fluxes.unit

        # Get the filter's transmission coeffs in the region it overlaps the fluxes
        filter_lam = filter_table["Wavelength"].quantity
        filter_ol_mask = (ol_lam_short <= filter_lam) & (filter_lam <= ol_lam_long)
        filter_lam = filter_lam[filter_ol_mask]
        filter_trans = filter_table["Norm-Transmission"][filter_ol_mask].value

        # Apply the filter & calculate overall transmitted flux value
        interp = _np.interp(filter_lam, lambdas, fluxes)
        return _np.sum((interp * filter_trans / _np.sum(filter_trans)))

    @classmethod
    def _bin_fluxes(cls,
                    lambdas: _ArrayLike,
                    fluxes: _ArrayLike,
                    lam_bin_midpoints: _ArrayLike) -> _u.Quantity:
        """
        Will calculate and return the means of the fluxes within each of the requested bins.

        :lambdas: source flux wavelengths
        :fluxes: source fluxes
        :lam_bin_midpoints: the midpoint lambda of each bin to populate
        :returns: the binned fluxes in the same units as the input
        """
        if lam_bin_midpoints.unit != lambdas.unit:
            lam_bin_midpoints = lam_bin_midpoints.to(lambdas.unit, equivalencies=_u.spectral())

        # Scipy wants bin edges so find midpoints between bins then extend by one at start & end.
        bin_mid_gaps = _np.diff(lam_bin_midpoints) / 2
        bin_edges = _np.concatenate([[lam_bin_midpoints[0] - (bin_mid_gaps[0])],
                                    lam_bin_midpoints[:-1] + (bin_mid_gaps),
                                    [lam_bin_midpoints[-1] + (bin_mid_gaps[-1])]]).value

        result = _binned_statistic(lambdas.value, fluxes.value, statistic=_np.nanmean,
                                   bins=bin_edges, range=(bin_edges.min(), bin_edges.max()))
        return result.statistic << fluxes.unit


class BtSettlGrid(StellarGrid):
    """
    Generates model SED fluxes from pre-built grids of bt-settl-agss model fluxes.
    """
    # pylint: disable=too-many-arguments, too-many-positional-arguments, too-many-locals

    _DEF_MODEL_FILE = StellarGrid._this_dir / "data/stellar_grids/bt-settl-agss/bt-settl-agss.npz"

    # Regexes for reading metadata from bt-settl ascii files
    _PARAM_RE = \
        _re.compile(r"^#[\s*](?P<k>\w*)[\s]*=[\s]*(?P<val>[+-]?([0-9]*[.])?[0-9]+)", _re.MULTILINE)
    _LAMBDA_UNIT_RE = _re.compile(r"Wavelength in (?P<unit>[\w\/]*)$", _re.MULTILINE)
    _FLUX_UNIT_RE = _re.compile(r"Flux in (?P<unit>[\w\/]*)$", _re.MULTILINE)

    def __init__(self, data_file: _Union[_Path, str]=None):
        """
        Initializes a new instance of this class.

        :data_file: the source of the model data, in numpy npz format
        """
        if data_file is None:
            data_file = self._DEF_MODEL_FILE

        with _np.load(data_file, allow_pickle=True) as df:
            meta = df["meta"].item()
            super().__init__(df["grid_full"], df["grid_filtered"],
                             meta["index_values"], meta["wavelengths"], meta["filter_names"])

    @classmethod
    def make_grid_file(cls,
                       source_files: _Iterable,
                       out_file: _Path=_DEF_MODEL_FILE,
                       filter_map_file: _Path=StellarGrid._DEF_FILTER_MAP_FILE,
                       dense_grids: bool=False):
        """
        Will ingest the chosen bt-settl-agss ascii grid files to produce a grid file containing
        the grids of fluxes and associated metadata to act as a source for instances of this class.

        Download bt-settl-aggs ascii model grids from following url
        https://svo2.cab.inta-csic.es/theory/newov2/index.php?models=bt-settl-agss
        
        :source_files: an iterator/list of the source bt-settle ascii files to read
        :out_file: the model file to write (overwriting any existing file)
        :filter_map_file: the json file with the filter name mappings between the
        Vizier SED service and the SVO filter library
        :dense_grids: whether to make a grid with all possible combinations of teff, logg, meta &
        alpha from source files (True) or only those combinations found (False)
        """
        grid_full_nbins = 1000
        grid_full_bin_lams = _np.geomspace(0.05, 50, num=grid_full_nbins, endpoint=True) << _u.um
        grid_full_bin_freqs = grid_full_bin_lams.to(_u.Hz, equivalencies=_u.spectral())
        filters_short, filters_long = _np.inf * cls._LAM_UNIT, 0 * cls._LAM_UNIT
        index_names = ["teff", "logg", "metal", "alpha"]

        # Need the files in sorted list as we go through them twice and the order may set indices
        source_files = sorted(source_files)
        print(f"{cls.__name__}.make_grid_file(): importing {len(source_files)} bt-settl-agss ascii",
              f"grid files into a new compressed model file written to:\n\t{out_file}\n")

        # The json has names of supported Vizier SED filters & maps to the corresponding SVO name.
        with open(filter_map_file, "r", encoding="utf8") as j:
            filter_map = _json_load(j)
        filters = { viz: cls.get_filter(svo, cls._LAM_UNIT) for viz, svo in filter_map.items() }

        # Set up the output grids
        index_values = cls._get_list_of_index_values(source_files, index_names, dense_grids)
        grid_len = len(index_values)
        grid_filtered = _np.full((grid_len, len(filters)), _np.nan, float)
        grid_full = _np.full((grid_len, grid_full_nbins), _np.nan, float)

        # Read in each source file, parse it, calculate the fluxes then store a row in output grids
        for file_ix, source_file in enumerate(source_files):
            meta = cls._read_metadata_from_ascii_model_file(source_file)
            lams, flux_densities = _np.genfromtxt(source_file, float, comments="#", unpack=True)
            print(f"{file_ix+1}/{len(source_files)} {source_file.name} [{len(lams):,d} rows]:",
                    ", ".join(f"{k}={meta[k]: .2f}" for k in index_names), end="...")

            # Find the row to be populated from the indexed values (same for both grids)
            row_ix = _np.where((index_values["teff"] == meta["teff"])\
                               & (index_values["logg"] == meta["logg"])\
                               & (index_values["metal"] == meta["metal"])\
                               & (index_values["alpha"] == meta["alpha"]))

            lams = (lams * meta["lambda_unit"]).to(cls._LAM_UNIT, equivalencies=_u.spectral())
            flux_densities = (flux_densities * meta["flux_unit"])\
                                .to(cls._FLUX_DENSITY_UNIT, equivalencies=_u.spectral_density(lams))
            fluxes = flux_densities * lams.to(_u.Hz, equivalencies=_u.spectral())

            # Write the row of binned fluxes to the full grid.
            bin_flux_densities = cls._bin_fluxes(lams, flux_densities, grid_full_bin_lams)
            grid_full[row_ix] = (bin_flux_densities *  grid_full_bin_freqs).value

            # Write the row of pre-filtered fluxes to the filtered grid.
            for filter_ix, (filter_name, filter_table) in enumerate(filters.items()):    # pylint: disable=unused-variable
                filter_flux = cls._get_filtered_flux_total(lams, fluxes, filter_table)
                grid_filtered[row_ix, filter_ix] = \
                                filter_flux.to(cls._FLUX_UNIT, equivalencies=_u.spectral()).value

                # # For teff=2000.0, logg=3.5, metal=0.0 and GAIA:Gbp ~ 1.7e-14 erg / s / cm^2
                # test_flux = (( filter_flux * (1.0 * u.Rsun).to(u.cm)**2) \
                #         / (190.91243807532035 * u.pc).to(u.cm)**2).to(u.erg / u.s / u.cm**2)

                if filter_flux.value: # Note the full extent of the applied filters
                    filters_short = min(filters_short, filter_table.meta["filter_short"])
                    filters_long = max(filters_long, filter_table.meta["filter_long"])
            print(f"added rows of {len(filters)} pre-filtered and {grid_full_nbins} binned fluxes")

        # Make sure there are no NaN values in the output grids with simple linear interpolation.
        for grid in [grid_filtered, grid_full]:
            x = _np.arange(grid.shape[0])
            for col_ix in range(grid.shape[1]):
                not_nan = ~_np.isnan(grid[..., col_ix])
                grid[..., col_ix] = _np.interp(x, x[not_nan], grid[not_nan, col_ix])

        # For now we're only dealing with alpha==zero
        alpha_zero_row_mask = index_values["alpha"] == 0
        index_names = index_names[:3]
        grid_full = grid_full[alpha_zero_row_mask]
        grid_filtered = grid_filtered[alpha_zero_row_mask]
        index_values = index_values[alpha_zero_row_mask][index_names]

        # Complete the metadata; row indices and col indices (filters & wavelengths)
        grid_meta = {
            "index_names": index_names,
            "index_values": index_values,
            "filter_names": list(filters.keys()),
            "wavelengths": grid_full_bin_lams
        }

        # Now we write out the model grids and metadata to a compressed npz file
        print(f"Saving model grids and metadata to {out_file}, overwriting any existing file.")
        _np.savez_compressed(out_file, meta=grid_meta,
                             grid_full=grid_full, grid_filtered=grid_filtered)
        return out_file

    @classmethod
    def _read_metadata_from_ascii_model_file(cls, source_file: _Path) -> dict[str, any]:
        """
        Reads the metadata for teff/logg/metal/alpha values used to generate this model file
        and the units associated with them and the grid of wavelengths and flux densities.
        """
        # First few lines of each file has metadata on it teff/logg/meta/alpha and units
        with open(source_file, mode="r", encoding="utf8") as sf:
            text = sf.read(1000)
        metadata = {
            **{ m.group("k"): float(m.group("val")) for m in cls._PARAM_RE.finditer(text) },
            "teff_unit": _u.K,
            "logg_unit": _u.dex,
            "metal_unit": _u.dimensionless_unscaled,
            "alpha_unit": _u.dimensionless_unscaled,
            "lambda_unit": _u.Unit(cls._LAMBDA_UNIT_RE.findall(text)[0]),
            "flux_unit": _u.Unit(cls._FLUX_UNIT_RE.findall(text)[0].replace("/A", "/Angstrom")),
        }

        if "meta" in metadata and not "metal" in metadata:
            metadata["metal"] = metadata.pop("meta")
        return metadata

    @classmethod
    def _get_list_of_index_values(cls, source_files: _ArrayLike, index_names: _List[str],
                                  dense: bool=False) -> _np.ndarray[float]:
        """
        Gets a sorted structured NDArray of the index values across the source files.

        :source_files: the list of files to parse
        :index_names: the values to read from the files and to index on
        :dense: if True, the resulting list will be the Cartesian product of the unique values
        """
        if dense:
            index_lists = { }
            for source_file in source_files:
                metadata = cls._read_metadata_from_ascii_model_file(source_file)
                if all(n in metadata.keys() for n in index_names):
                    for k in index_names:
                        if k in index_lists:
                            index_lists[k] += [metadata[k]]
                        else:
                            index_lists[k] = [metadata[k]]
            index_list = list(_product(*(_np.unique(index_lists[k]) for k in index_names)))
        else:
            index_list = []
            for source_file in source_files:
                metadata = cls._read_metadata_from_ascii_model_file(source_file)
                if all(n in metadata.keys() for n in index_names):
                    index_list += [tuple(metadata[k] for k in index_names)]
        return _np.array(sorted(index_list), dtype=[(k, float) for k in index_names])


if __name__ == "__main__":
    # Download bt-settl-aggs ascii model grids from following url
    # https://svo2.cab.inta-csic.es/theory/newov2/index.php?models=bt-settl-agss
    # then decompress the tgz contents into the ../.cache/.modelgrids/bt-settl-agss dir

    # pylint: disable=protected-access
    in_files = (StellarGrid._CACHE_DIR / ".modelgrids/bt-settl-agss/").glob("lte*.dat.txt")
    new_file = BtSettlGrid.make_grid_file(sorted(in_files), dense_grids=True)

    # Test what has been saved
    bgrid = BtSettlGrid(data_file=new_file)
    print(f"\nLoaded newly created model grid from {new_file}")
    print( "Filters:", ", ".join(bgrid._filter_names_list))
    print(f"Ranges: teff={bgrid.teff_range} {bgrid.teff_unit:unicode},",
          f"logg={bgrid.logg_range} {bgrid.logg_unit:unicode}, metal = {bgrid.metal_range}")
    print("Test flux for 'GAIA/GAIA3:Gbp' filter, teff=2000, logg=4.0, metal=0, alpha=0:",
          ", ".join(f"{f:.3f}" for f in bgrid.get_filter_fluxes(["GAIA/GAIA3:Gbp"], 2000, 4, 0)),
          f"[{bgrid.flux_unit:unicode}]")
