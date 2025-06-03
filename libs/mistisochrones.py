""" A simple class for querying MIST Isochrones """
import errno
from typing import List, Iterable
from pathlib import Path
from inspect import getsourcefile
import re

import numpy as np

from .data.mist.read_mist_models import ISO

# Index onto the columns underlying commonly requested linear values
COLS_FOR_PARAMS = {
    "L":    "log_L",
    "Teff": "log_Teff",
    "R":    "log_R",
    "g":    "log_g",
}

class MistIsochrones():
    """
    This class wraps one or more MIST isochrone files and exposes functions to query them.
    """
    _this_dir = Path(getsourcefile(lambda:0)).parent

    def __init__(self, metallicities: list[float]=None) -> None:
        """
        Initializes a MistIsochrones class.

        :metallicities: optional list of metallicities to load by feh value, or any found if not set
        """
        isos_dir = self._this_dir / "data/mist/MIST_v1.2_vvcrit0.4_basic_isos"
        iso_files = sorted(isos_dir.glob("*.iso"))

        if not iso_files or len(iso_files) == 0:
            raise FileNotFoundError(errno.ENOENT,
                                    f"No iso files found in {isos_dir}. The readme.txt file " +
                                    "in this directory has information on how to populate it.")

        # Index the iso files on their [Fe/H]
        self._isos: dict[float, List] = {}
        feh_file_pattern = re.compile(r"feh_(?P<pm>m|p)(?P<feh>[0-9\.]*)_", re.IGNORECASE)
        for iso_file in iso_files:
            match = feh_file_pattern.search(iso_file.stem)
            if match and "feh" in match.groupdict():
                pm = -1 if (match.group("pm") or "p") == "m" else 1
                feh = float(match.group("feh")) * pm
                if metallicities is None or feh in metallicities:
                    self._isos[feh] = ISO(f"{iso_file.resolve()}")
            else:
                raise Warning(f"Unexpected iso file name format: {iso_file}")


    def list_metallicities(self) -> np.ndarray:
        """
        List the distinct isochrone metallicity values available.
        """
        return np.array(list(self._isos.keys()))


    def list_ages(self, feh: float, min_phase: float=0.0, max_phase: float=9.0) -> np.ndarray:
        """
        List the distinct log10(age yr)s within the isochrones matching the passed metallicity.
        Only ages with records for stars within the min and max phases are returned.

        The supported phase values are (from the MIST documentation):
        -1=PMS, 0=MS, 2=RGB, 3=CHeB, 4=EAGB, 5=TPAGB, 6=postAGB, 9=WR

        :feh: the metallicity key value - used to select the appropiate isochrone
        :min_phase: only ages with at least one star at this or a later phase are returned
        :max_phase: only ages with at least one star at this or an earlier phase are returned
        :returns: the list of ages with at least some stars within the min and max phases
        """
        # We only want age blocks which contain stars within our chosen phase criteria
        iso = self._isos[feh]
        ages = [ab["log10_isochrone_age_yr"][0] for ab in iso.isos if ab["phase"][-1] >= min_phase
                                                                    or ab["phase"][0] <= max_phase]
        return np.array(ages)


    def stellar_params_for_mass(self,
                                feh: float,
                                log_age: float,
                                mass: float,
                                params: Iterable[str],
                                min_phase: float=None,
                                max_phase: float=None) -> np.ndarray:
        """
        Get the requested stellar param values for the metallicity, age, mass and phase criteria.
        Will allow a log field (i.e.: log_R) to be requested as either the log value (as log_R) or
        the linear value (as R).

        Return values will be interpolated from the rows with masses nearest the requested value
        within the selected age block (subject to optional restrictions on phase)

        :feh: the chosen metallicity
        :log_age: the chosen age; the nearest matching age block will be used
        :mass: the chosen current stellar mass in M_sun
        :min_phase: when set, this will restrict the match to rows of at least this phase
        :max_phase: when set, this will restrict the match to rows up to this phase
        :params: the list of names param values to return
        """
        # pylint: disable=too-many-arguments, too-many-positional-arguments, too-many-locals
        # Find the age block nearest the requested value
        iso = self._isos[feh]
        age_block = iso.isos[iso.age_index(log_age)]

        if min_phase is not None or max_phase is not None:
            # Mask out any rows which do not match any phase criteria
            mask = np.array([True] * len(age_block))
            if min_phase is not None:
                mask &= age_block["phase"] >= min_phase
            if max_phase is not None:
                mask &= age_block["phase"] <= max_phase
            all_masses = age_block[mask]["star_mass"]
        else:
            all_masses = age_block["star_mass"]

        # Mass gradient can change direction so we only use the nearest 1 or 2 rows to the
        # requested mass and order them by increasing mass.
        row_ixs = self._find_sample_rows_indices(mass, all_masses)
        rmasses, rows = all_masses[row_ixs], age_block[row_ixs]

        # Read the columns, undoing any "log" so we can do linear interpolation/extrapolation
        if isinstance(params, str):
            params = [params]
        cols = [COLS_FOR_PARAMS.get(p, p) for p in params]
        raw_vals = np.array([
            np.interp(mass, rmasses, 10**rows[c] if c.startswith("log") else rows[c]) for c in cols
        ])

        # Now re-apply log10 to the interpolated values where this is the requested param
        ret_vals = [np.log10(rv) if p.startswith("log") else rv for p, rv in zip(params, raw_vals)]
        return np.array(ret_vals)


    def _find_sample_rows_indices(self, value, values, error_if_out_of_range=ValueError):
        """
        Find the rows to sample based on where the lookup value best matches the values.

        This handles the fact that the values may not just steadily increase. For some fields
        they may go down, or even change direction part way though the range. 
        """
        if not min(values) < value < max(values):
            if error_if_out_of_range:
                raise error_if_out_of_range(f"The input value of {value} is outside the range " +
                                            f"[{min(values)}, {max(values)}]")
            return []

        difs = values - value
        ix_closest = np.argmin(np.abs(difs))
        closest_dif = difs[ix_closest]

        if closest_dif == 0: # Exact match
            return ix_closest

        # Not an exact match. We need the closest row and the adjascent row beyond the value.
        if ix_closest == 0:
            return [ix_closest, ix_closest + 1]
        if ix_closest == len(values) - 1:
            return [ix_closest - 1, ix_closest]

        if closest_dif < 0:
            return [ix_closest, ix_closest + (1 if difs[ix_closest+1] > 0 else -1)]
        return [ix_closest + (1 if difs[ix_closest+1] < 0 else -1), ix_closest]
