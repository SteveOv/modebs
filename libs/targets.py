""" Helper for interacting with targets json """
from typing import List, Dict, Tuple, Generator
from pathlib import Path as _Path
import json as _json

_default_target_config_defaults = {
    "exclude": False,
    "details": "",
    "notes": "",
    "exptime": "short",
    "author": ["SPOC", "TESS-SPOC"],
    "mission": ["TESS", "HLSP"],
    "flux_column": "sap_flux",
    "quality_bitmask": "default",
    "sectors": None,
    "exclude_sectors": None,
    "quality_masks": [],
    "detrend_gap_threshold": 0.5,
    "detrend_poly_degree": 1,
    "detrend_iterations": 2,
    "phiS": 0.5,
    "t0": None,
    "period": None,
    "double_period": False
}

class TargetConfig():
    """ Wrapper class for the configuration of a target system """
    def __init__(self,
                 target_id: any,
                 config: Dict[str, any],
                 defaults: Dict[str, any]):
        """
        TODO
        """
        self._target_id = target_id
        self._config = { **defaults, **config }

    @property
    def target_id(self) -> any:
        """ Gets the primary id of the target """
        return self._target_id

    @property
    def quality_masks(self) -> List[Tuple[float, float]]:
        """ Gets the quality masks """
        return [tuple(mask) for mask in self._config["quality_masks"]]

    @property
    def combine_sectors(self) -> bool:
        """ Temp - to be replaced by groupings in sectors """
        return False

    @property
    def sectors_flat(self) -> List[int]:
        """ Gets a flattened list of the sectors, without the grouping """
        if (sectors := self.sectors) is not None:
            return list(TargetConfig._yield_recursive(sectors))
        return sectors

    def get(self, key, default: None=None):
        """ Return the value if the key is in this config, else return the default"""
        return self._config.get(key, default)

    @classmethod
    def _yield_recursive(cls, values: List[any]):
        """ Recursively yield values from a List which may contain Lists """
        for val in values:
            if isinstance(val, List):
                yield from cls._yield_recursive(val)
            else:
                yield val

    def __getattr__(self, item):
        """
        Handles the default behaviour for attributes, which is to get the value of the corresponding
        config dictionary item. Saves having to write property funcs for each attr.
        """
        if item in self._config:
            return self._config[item]
        raise AttributeError

class Targets():
    """ Helper for interacting with targets json """
    def __init__(self,
                 target_file: _Path):
        """
        TODO
        """
        with open(target_file, mode="r", encoding="utf8") as cf:
            targets_config = _json.load(cf)

        self._explicit = targets_config.get("explicit", False)
        self._target_configs = targets_config.get("target_configs", {})
        self._target_config_defaults = {
            **_default_target_config_defaults,
            **targets_config.get("target_config_defaults", {})
        }

    @property
    def explicit(self) -> bool:
        """ Whether these targets are an explicit list of targets, or a set of filter criteria """
        return self._explicit

    def iterate_known_targets(self, omit_excluded: bool=True) -> Generator[TargetConfig, any, any]:
        """
        Iterates over the known TargetConfigs.

        :omit_excluded: if true, will omit targets with the excluded flag set to True
        """
        exclude_default = self._target_config_defaults["exclude"]
        for target_id, target_config in self._target_configs.items():
            if not omit_excluded or not target_config.get("exclude", exclude_default):
                yield TargetConfig(target_id,
                                   target_config,
                                   self._target_config_defaults)

    def get_known_target_ids(self, omit_excluded: bool=True) -> List[any]:
        """
        Gets a list of configured target_ids.

        :omit_excluded: if true, will omit targets with the excluded flag set to True
        """
        return [cfg.target_id for cfg in self.iterate_known_targets(omit_excluded)]

    def get(self, target_id, fallback_to_default: bool=False) -> TargetConfig:
        """
        Get the target_config for a specific target
        
        :target_id: unique id of the target
        :fallback_to_default: return a default config if no config exists for target_id
        :return: the requested TargetConfig
        """
        if target_id in self._target_configs:
            return TargetConfig(target_id,
                                self._target_configs[target_id],
                                self._target_config_defaults)
        if fallback_to_default:
            return TargetConfig(target_id, {}, self._target_config_defaults)
        raise KeyError(f"target {target_id} is unknown")
