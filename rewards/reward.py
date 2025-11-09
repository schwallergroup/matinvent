import os
import numpy as np
from typing import Union, Tuple, List, Dict
from omegaconf import DictConfig, OmegaConf
from pymatgen.core.structure import Structure


def linear_scaling(values, minv=0.0, maxv=6.0):
    ss = (values - minv) / (maxv - minv)
    ss[ss > 1.0] = 1.0
    ss[ss < 0.0] = 0.0
    return ss


def average_props(prop_dict):
    prop_num = len(prop_dict)
    prop_list = list(prop_dict.values())
    prop_sum = prop_list[0]
    for _prop in prop_list[1:]:
        prop_sum += _prop
    prop_mean = prop_sum / prop_num

    return prop_mean


def min_props(prop_dict):
    prop_list = list(prop_dict.values())
    prop_arr = np.array(prop_list)
    prop_min = prop_arr.min(axis=0)
    return prop_min


class Reward():
    def __init__(
        self,
        root_dir: str,
        prop_cfg: DictConfig,
        reward_threshold: float,
        reduce: str = 'mean',
        **kwargs,
    ) -> None:
        assert reduce in ['mean', 'min', 'weight']
        self.root_dir = root_dir
        self.prop_cfg = prop_cfg
        self.threshold = reward_threshold
        self.cfg = OmegaConf.create(kwargs)
        self.reduce = reduce
        if not os.path.exists(self.root_dir):
            os.makedirs(self.root_dir)

    def calc_props(
        self,
        samples: Tuple[List[Structure], str],
        label: str = 'tmp'
    ):
        prop_dict, prop_list = {}, []
        for _cfg in self.prop_cfg:
            _prop1 = _cfg.calculator.calc(samples, label)
            prop_list.append(_prop1)
            _prop2 = np.nan_to_num(_prop1, nan=0.0)
            prop_dict[_cfg.name] = _prop2.astype(float)

        prop_list = np.array(prop_list)
        none_ids = np.isnan(prop_list).any(axis=0)

        return prop_dict, none_ids

    def scoring(
        self,
        samples: Tuple[List[Structure], str],
        label: str = 'tmp'
    ):

        prop_dict, failed_mask = self.calc_props(samples, label)

        scaled_prop_dict = {}
        for _cfg in self.prop_cfg:
            if _cfg.target == 'ascending':
                _sprop = linear_scaling(
                    values = prop_dict[_cfg.name],
                    minv = _cfg.minv,
                    maxv = _cfg.maxv,
                )
            elif _cfg.target == 'descending':
                _sprop = linear_scaling(
                    values = -prop_dict[_cfg.name],
                    minv = -_cfg.maxv,
                    maxv = -_cfg.minv,
                )
            elif isinstance(_cfg.target, float):
                diff = np.abs(prop_dict[_cfg.name] - _cfg.target)
                _sprop = linear_scaling(
                    values = -diff,
                    minv = -_cfg.maxv,
                    maxv = -_cfg.minv,
                )
            else:
                raise TypeError("prop cfg.target must be a float or descending or ascending")

            scaled_prop_dict[_cfg.name] = _sprop

        if self.reduce == 'mean':
            rewards = average_props(scaled_prop_dict)
        elif self.reduce == 'min':
            rewards = min_props(scaled_prop_dict)
        elif self.reduce == 'weight':
            for _cfg in self.prop_cfg:
                w = _cfg.weight
                scaled_prop_dict[_cfg.name] = scaled_prop_dict[_cfg.name] * w
            sprop_list = list(scaled_prop_dict.values())
            sprop_arr = np.array(sprop_list)
            rewards = sprop_arr.sum(axis=0)

        rewards[failed_mask] = 0.0
        return rewards, prop_dict, failed_mask
