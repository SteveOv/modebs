""" A class for handling the generation of model fluxes for filters sourced from bt-settl data """
# pylint: disable=no-member
from abc import ABC as _AbstractBaseClass, abstractmethod as _abstractmethod
from typing import Union as _Union, Iterable as _Iterable
from pathlib import Path as _Path
from inspect import getsourcefile as _getsourcefile
from warnings import filterwarnings as _filterwarnings
import re as _re
from json import load as _json_load
from urllib.parse import quote_plus as _quote_plus

import numpy as _np
from numpy.typing import ArrayLike as _ArrayLike

# from scipy.stats import binned_statistic as _binned_statistic
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

    def __init__(self, data_file: _Union[_Path, str],
                 wavelength_range: _u.Quantity["length"],
                 teff_range: _u.Quantity["temperature"],
                 logg_range: _u.Quantity,
                 metal_range: _u.Quantity,
                 flux_unit: _u.Unit,
                 filter_names: _ArrayLike):
        """
        Initializes a new instance of this class.

        :data_file: the source of the model data, in numpy npz format
        """
        self._data_file = data_file
        self._wavelength_range = wavelength_range
        self._teff_range = teff_range
        self._logg_range = logg_range
        self._metal_range = metal_range
        self._flux_unit = flux_unit
        self._filter_names = filter_names

    @property
    def data_file(self) -> _Path:
        """ Gets the Path of the data file being used. """
        return self._data_file

    @property
    def wavelength_range(self) -> _u.Quantity["length"]:
        """ Gets the range of wavelength covered by this model """
        return self._wavelength_range

    @property
    def teff_range(self) -> _u.Quantity["temperature"]:
        """ Gets the range of effective temperatures covered by this model """
        return self._teff_range

    @property
    def logg_range(self) -> _u.Quantity:
        """ Gets the range of logg covered by this model """
        return self._logg_range

    @property
    def metal_range(self) -> _u.Quantity:
        """ Gets the range of metallicities covered by this model """
        return self._metal_range

    @property
    def flux_unit(self) -> _u.Unit:
        """ Gets the unit of the returned fluxes """
        return self._flux_unit

    def has_filter(self, filter_name: _Union[str, _Iterable]) -> _Union[bool, _np.ndarray[bool]]:
        """ Gets whether this model knows of the requested filter(s) """
        return _np.in1d(filter_name, self._filter_names)

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
        return _np.array([self._filter_names.index(n) for n in filter_names], dtype=int)

    @_abstractmethod
    def get_filter_fluxes(self,
                          filters: _ArrayLike,
                          teff: float,
                          logg: float,
                          metal: float=0.,
                          as_quantity: bool=False) -> _Union[_np.ndarray[float], _u.Quantity]:
        """
        Will return a ndarray of flux values calculated for requested filter names at
        the chosen effective temperature, logg and metallicity values.

        :filters: a list of filter identifiers for which we are generating fluxes
        :teff: the effective temperature for the fluxes
        :logg: the logg for the fluxes
        :metal: the metallicity for the fluxes
        :as_quantity: whether to return the fluxes as a Quantity (True) in units of flux_unit
        or a ndarray[float] (False)
        :returns: the resulting flux values (in the units of the underlying data file)
        """

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



class BtSettlGrid(StellarGrid):
    """
    Generates model SED fluxes from pre-built grid of filtered fluxes.
    For now, the grid file is created from bt-settl-agss data files with make-bt-settl-agss.py
    This uses a grid of interpolators based on pre-convolved filter fluxes. The pre-filtered
    nature of the fluxes ensures minimal processing is required to produce a model sed, so
    it's fast. However, this model is only suitable for SED observations which are dereddened.
    """
    # pylint: disable=too-many-arguments, too-many-positional-arguments, too-many-locals

    _DEF_MODEL_FILE = StellarGrid._this_dir / "data/stellar_grids/bt-settl-agss/bt-settl-agss.npz"

    _LAM_UNIT = _u.um
    _FLUX_DENSITY_UNIT = _u.W / _u.m**2 / _u.Hz
    _FLUX_UNIT = _u.W / _u.m**2
    _PRE_BIN_MODEL = False

    def __init__(self, data_file: _Union[_Path, str]=None):
        """
        Initializes a new instance of this class.

        :data_file: the source of the model data, in numpy npz format
        """
        if data_file is None:
            data_file = self._DEF_MODEL_FILE

        with _np.load(data_file, allow_pickle=True) as df:
            # Load the model_grid of individual pre-filtered fluxes. These fluxes are not reddened
            # and are suitable for quickly fitting to a SED which has been dereddened.
            model_grid_filtered = df["model_grid_filtered"]
            ranges = df["ranges"]
            units = df["units"]

        # Code to populate pivots below depends on fixing to alpha==0 and this ordering of fluxes
        model_grid_filtered = model_grid_filtered[model_grid_filtered["alpha"]==0]
        model_grid_filtered.sort(order=["teff", "logg", "metal"])

        # The cols 0 to 3 are expected to be teff, logg, metal and alpha
        # The rest of the cols are the filters and corresponding fluxes.
        teffs, teff_ixs = _np.unique(model_grid_filtered["teff"], return_inverse=True)
        loggs, logg_ixs = _np.unique(model_grid_filtered["logg"], return_inverse=True)
        metals, metal_ixs = _np.unique(model_grid_filtered["metal"], return_inverse=True)
        filter_names = list(model_grid_filtered.dtype.names)[4:]

        # Set up a table of interpolators, one per filter. Each interpolator is based on a
        # pivot table with the teffs, loggs and metals as the axes and filter fluxes as the values.
        self._model_interps = _np.empty(shape=(len(filter_names), ),
                                        dtype=[("filter", object), ("interp", object)])
        for filter_ix, filter_name in enumerate(filter_names):
            # Need model_grid[filter_name] of teffs*loggs*metals items long and to be sorted by
            # [teff, logg, metal] in order to write to the pivot directly like this. If model_table
            # not limited to alpha==0 then we will have to include the extra dimension.
            pivot_table = _np.empty(shape=(teffs.shape[0], loggs.shape[0], metals.shape[0]),
                                    dtype=model_grid_filtered[filter_name].dtype)
            pivot_table[teff_ixs, logg_ixs, metal_ixs] = model_grid_filtered[filter_name]

            interp = _RegularGridInterpolator((teffs, loggs, metals), pivot_table, "linear")
            self._model_interps[filter_ix] = (filter_name, interp)
        del model_grid_filtered

        super().__init__(data_file,
                         wavelength_range=ranges[ranges["column"]=="lambda"]["range"][0],
                         teff_range=ranges[ranges["column"]=="teff"]["range"][0],
                         logg_range=ranges[ranges["column"]=="logg"]["range"][0],
                         metal_range=ranges[ranges["column"]=="metal"]["range"][0],
                         flux_unit=units[units["column"]=="flux"]["unit"][0],
                         filter_names=filter_names)

    @property
    def num_interpolators(self) -> int:
        """ Gets the number of interpolators covering this model """
        return self._model_interps.shape[0]

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
    def make_grid_file(cls,
                       source_files: _Iterable,
                       out_file: _Path=_DEF_MODEL_FILE,
                       filter_map_file: _Path=StellarGrid._DEF_FILTER_MAP_FILE):
        """
        Will process the chosen bt-settl-agss ascii grid files to produce a grid file containing
        the grids of fluxes and associated metadata to act as a source for instances of this class.

        Download bt-settl-aggs ascii model grids from following url
        https://svo2.cab.inta-csic.es/theory/newov2/index.php?models=bt-settl-agss
        
        :source_files: an iterator/list of the source bt-settle ascii files to read
        :out_file: the model file to write (overwriting any existing file)
        :filter_map_file: the json file with the filter name mappings between the
        Vizier SED service and the SVO filter library
        """
        source_files = sorted(source_files)
        print(f"{cls.__name__}.make_grid_file():",
              f"will import {len(source_files)} bt-settl-agss ascii model grid files",
              f"into a new compressed model file written to:\n\t{out_file}\n")

        # Populate the dict of known/supported filters.
        # The translator holds an index mapping Vizier SED service filter name to SVO filter name.
        with open(filter_map_file, "r", encoding="utf8") as j:
            filter_translator = _json_load(j)
        filters = { k: cls.get_filter(v, cls._LAM_UNIT) for k, v in filter_translator.items() }

        # Metadata for the output grid and pre-allocate it
        index_col_names = ["teff", "logg", "metal", "alpha"]# These are present for every model grid
        col_names = index_col_names + list(filters.keys())
        filters_short, filters_long = _np.inf * cls._LAM_UNIT, 0 * cls._LAM_UNIT
        model_grid_filtered = _np.zeros((len(source_files), len(col_names)), dtype=float)

        # Read in each source file, parse it, calculate the filter fluxes then store as a row
        param_re = _re.compile(r"^#[\s*](?P<k>\w*)[\s]*=[\s]*(?P<val>[+-]?([0-9]*[.])?[0-9]+)",
                               _re.MULTILINE)
        lambda_unit_re = _re.compile(r"Wavelength in (?P<unit>[\w\/]*)$", _re.MULTILINE)
        flux_unit_re = _re.compile(r"Flux in (?P<unit>[\w\/]*)$", _re.MULTILINE)
        for file_ix, source_file in enumerate(sorted(source_files)[:]):

            # First few lines of each file has metadata on it teff/logg/meta/alpha and units
            with open(source_file, mode="r", encoding="utf8") as sf:
                text = sf.read(500)
                metadata = {
                    "teff_unit": _u.K,
                    "logg_unit": _u.dex,
                    "metal_unit": _u.dimensionless_unscaled,
                    "alpha_unit": _u.dimensionless_unscaled,
                    "lambda_unit": _u.Unit(lambda_unit_re.findall(text)[0]),
                    "flux_unit": _u.Unit(flux_unit_re.findall(text)[0].replace("/A", "/Angstrom")),
                    **{ m.group("k"): float(m.group("val")) for m in param_re.finditer(text) },
                }

                if "meta" in metadata and not "metal" in metadata:
                    metadata["metal"] = metadata.pop("meta")
                del text

            # "Index" values
            model_grid_filtered[file_ix, 0:4] = [metadata[k] for k in index_col_names]

            # Get the data, and then coerce them to output units for ease of processing
            lam, flux_density = _np.genfromtxt(source_file, dtype=float, comments="#", unpack=True)
            lam = (lam * metadata["lambda_unit"]).to(cls._LAM_UNIT, equivalencies=_u.spectral())
            flux_density = (flux_density * metadata["flux_unit"])\
                                .to(cls._FLUX_DENSITY_UNIT, equivalencies=_u.spectral_density(lam))

            print(f"{file_ix+1}/{len(source_files)} {source_file.name} [{len(lam):,d} rows]:",
                    ", ".join(f"{k}={metadata[k]: .2f}" for k in index_col_names), end="...")

            # TODO: sample code for binning fluxes to be used when building full (non-filtered grid)
            # # Bin & window the data so we can get the volumes down to something manageable.
            # lam_bin = _np.logspace(_np.log10(0.1), _np.log10(300), num=5000, base=10) # um
            # lam_bin_half_gap = _np.diff(lam_bin) / 2
            # lam_bin_edge = _np.concatenate([[lam_bin[0] - (lam_bin_half_gap[0])],
            #                                 lam_bin[:-1] + (lam_bin_half_gap),
            #                                 [lam_bin[-1] + (lam_bin_half_gap[-1])]])
            # binres = _binned_statistic(lam.value, flux_density.value, statistic="mean",
            #                             bins=lam_bin_edge,
            #                             range=(lam_bin_edge.min(), lam_bin_edge.max()))
            # lam = lam_bin << lam.unit
            # del lam_bin, lam_bin_edge, lam_bin_half_gap
            # # Finally calculate the binned fluxes, based on its nominal wavelength
            # fluxes = (binres.statistic << flux_density.unit) \
            #             * lam.to(_u.Hz, equivalencies=_u.spectral())

            fluxes = (flux_density * lam.to(_u.Hz, equivalencies=_u.spectral()))

            # This is where the magic happens! We need to overlay the filter onto the flux densities
            # to apply its sensitivity, then sum what is transmitted and finally convert to fluxes
            lam_short, lam_long = lam.min(), lam.max()
            for filter_ix, (filter_name, filter_grid) in enumerate(filters.items()):

                # Work out the lambda range where the filter and binned data overlap
                ol_lam_short = max(lam_short, filter_grid.meta["filter_short"])
                ol_lam_long = min(lam_long, filter_grid.meta["filter_long"])

                if ol_lam_short < ol_lam_long:
                    filter_lam = filter_grid["Wavelength"].quantity
                    filter_ol_mask = (ol_lam_short <= filter_lam) & (filter_lam <= ol_lam_long)
                    filter_lam = filter_lam[filter_ol_mask]
                    filter_trans = filter_grid["Norm-Transmission"][filter_ol_mask].value

                    # Apply the filter & calculate overall transmitted flux value
                    interp = _np.interp(filter_lam, lam, fluxes)
                    filter_flux = _np.sum((interp * filter_trans / _np.sum(filter_trans)))

                    # # For teff=2000.0, logg=3.5, metal=0.0 and GAIA:Gbp ~ 1.7e-14 erg / s / cm^2
                    # test_flux = (( filter_flux * (1.0 * u.Rsun).to(u.cm)**2) \
                    #         / (190.91243807532035 * u.pc).to(u.cm)**2).to(u.erg / u.s / u.cm**2)

                    model_grid_filtered[file_ix, 4 + filter_ix] = \
                                filter_flux.to(cls._FLUX_UNIT, equivalencies=_u.spectral()).value

                    # Record the maximum extent of the filter coverage
                    filters_short = min(filters_short, filter_grid.meta["filter_short"])
                    filters_long = max(filters_long, filter_grid.meta["filter_long"])

            print(f"added row of {len(filters)} total filter fluxes")


        # Get the data into a numpy structured array with the expected columns
        # Should already be in this order, but just in case
        model_grid_filtered.dtype = [(cn, float) for cn in col_names]
        model_grid_filtered.sort(order=index_col_names)

        # Now we write out the model grid's data and metadata to a compressed npz file
        key_columns_and_units = [(k, metadata.get(f"{k}_unit")) for k in index_col_names]
        key_columns_and_ranges = \
                [(k, (model_grid_filtered[k].min(), model_grid_filtered[k].max()) << u)
                                                                for k, u in key_columns_and_units]

        units = _np.array(key_columns_and_units
                        + [("lambda", cls._LAM_UNIT), ("flux", cls._FLUX_UNIT)],
                        dtype=[("column", "<U6"), ("unit", _np.dtype(_u.Unit))])
        ranges = _np.array(key_columns_and_ranges
                        + [("lambda", (filters_short, filters_long) << cls._LAM_UNIT)],
                        dtype=[("column", "<U6"), ("range", _np.dtype(_u.Quantity))])

        filter_map = _np.array([(v, k) for k, v in filter_translator.items()],
                            dtype=[("svo", "<U50"), ("vizier", "<U50")])

        print(f"Saving model grid and metadata to {out_file}, overwriting any existing file.")
        _np.savez_compressed(out_file, model_grid_filtered=model_grid_filtered,
                             units=units, ranges=ranges, filter_map=filter_map)
        return out_file


if __name__ == "__main__":
    # Download bt-settl-aggs ascii model grids from following url
    # https://svo2.cab.inta-csic.es/theory/newov2/index.php?models=bt-settl-agss
    # then decompress the tgz contents into the ../.cache/.modelgrids/bt-settl-agss dir

    # pylint: disable=protected-access
    in_files = (StellarGrid._CACHE_DIR / ".modelgrids/bt-settl-agss/").glob("lte*.dat.txt")
    new_file = BtSettlGrid.make_grid_file(in_files)

    # Test what has been saved
    grid = BtSettlGrid(data_file=new_file)
    print(f"\nLoaded newly created model grid from {new_file}")
    print( "Filters:", ", ".join(grid._filter_names))
    print(f"Ranges: teff={grid.teff_range}, logg={grid.logg_range}, metal = {grid.metal_range}")
    print("Test flux for 'GAIA/GAIA3:Gbp' filter, teff=2000, logg=4.0, metal=0, alpha=0:",
          ", ".join(f"{f:.3f}" for f in grid.get_filter_fluxes(["GAIA/GAIA3:Gbp"], 2000, 4, 0)),
          f"[{grid.flux_unit:unicode}]")
