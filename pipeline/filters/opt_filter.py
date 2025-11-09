import io
import contextlib
import multiprocessing as mp
from typing import List, Literal
import numpy as np
from numpy.typing import NDArray
from pymatgen.core.structure import Structure
from omegaconf import OmegaConf
from huggingface_hub import hf_hub_download
import torch


from mattergen.evaluation.reference.reference_dataset_serializer import LMDBGZSerializer
from mattergen.evaluation.metrics.evaluator import MetricsEvaluator
from mattergen.evaluation.metrics.structure import structure_validity, is_smact_valid
from mattergen.evaluation.utils.relaxation import relax_structures
from mattergen.evaluation.utils.structure_matcher import (
    DefaultDisorderedStructureMatcher,
    DefaultOrderedStructureMatcher,
)

METRIC_LIST = [
    "validity", "novel", "unique", "stable", "synthesizable",
]


def get_device(device: str | None = None):
    if device is None:
        if torch.cuda.is_available():
            device = 'cuda'
        elif torch.backends.mps.is_available():
            device = 'mps'
        else:
            device = 'cpu'
    return torch.device(device)


def parallel_run(func, args_list, num_workers: int = None):
    if num_workers is None:
        num_workers = mp.cpu_count()
    with mp.Pool(processes=num_workers) as pool:
        if isinstance(args_list[0], tuple):
            results = pool.starmap(func, args_list)
        else:
            results = pool.map(func, args_list)

    return results


def invalid_filter(sample_data, sample_struc, return_mask=False):
    valid_comp = parallel_run(structure_validity, sample_struc)
    valid_struc = parallel_run(is_smact_valid, sample_struc)
    valid_cell = [
        max(list(s.lattice.abc)) < 25 for s in sample_struc
    ]
    mask = np.array(valid_comp) & np.array(valid_struc) & np.array(valid_cell)
    filtered_data = [x for x, m in zip(sample_data, mask) if m]
    filtered_struc = [x for x, m in zip(sample_struc, mask) if m]

    if return_mask:
        return mask

    return filtered_data, filtered_struc


class OptFilter:
    def __init__(
        self,
        metrics: List[str],
        relax: bool = True,
        silent: bool = True,
        device: str | None = None,
        reference_path: str | None = None,
        structure_matcher: Literal["ordered", "disordered"] = "disordered",
        **kwargs,
    ) -> None:
        assert all(m in METRIC_LIST for m in metrics)
        self.metrics = metrics
        self.relax = relax
        self.silent = silent
        self.device = get_device(device)
        self.structure_matcher = (
            DefaultDisorderedStructureMatcher()
            if structure_matcher == "disordered"
            else DefaultOrderedStructureMatcher()
        )
        self.cfg = OmegaConf.create(kwargs)
        if reference_path is None:
            _path = hf_hub_download(
                repo_id="jwchen25/MatInvent",
                filename="reference_MP2020correction.gz",
            )
            self.reference = LMDBGZSerializer().deserialize(_path)
        else:
            self.reference = LMDBGZSerializer().deserialize(reference_path)

    def __call__(
        self,
        data_list: list,
        structures: list[Structure],
        energies: list[float] | None = None,
        potential_load_path: str = "MatterSim-v1.0.0-5M.pth",
    ):
        data_list, structures = self.pre_filter_ehull(data_list, structures)

        if self.silent:
            buffer = io.StringIO()
            with contextlib.redirect_stdout(buffer), contextlib.redirect_stderr(buffer):
                mask, relaxed_struc, metrics = self.relax_and_filter(
                    structures, energies, potential_load_path
                )
        else:
            mask, relaxed_struc, metrics = self.relax_and_filter(
                structures, energies, potential_load_path
            )

        filtered_data = [x for x, m in zip(data_list, mask) if m]
        filtered_struc = [x for x, m in zip(relaxed_struc, mask) if m]
        return filtered_data, filtered_struc, metrics

    def pre_filter_ehull(
        self,
        data_list: list,
        structures: list[Structure],
    ):
        uni_element_set = set()
        for s in structures:
            uni_element_set.update([str(el) for el in s.composition.elements])
        terminal_systems = uni_element_set
        ref_set = set(self.reference.entries_by_chemsys.keys())
        missing_terminals = list(terminal_systems - ref_set)
        terminals_in_reference = terminal_systems & ref_set
        missing_energy = [
            chemsys
            for chemsys in terminals_in_reference
            if all([np.isnan(e.energy) for e in self.reference.entries_by_chemsys[chemsys]])
        ]

        mask = []
        bad_elements = set(missing_terminals + missing_energy)
        for struc in structures:
            struc_element_set = set([str(e) for e in struc.composition.elements])
            if len(struc_element_set & bad_elements) > 0:
                mask.append(False)
            else:
                mask.append(True)

        filtered_data = [x for x, m in zip(data_list, mask) if m]
        filtered_struc = [x for x, m in zip(structures, mask) if m]
        return filtered_data, filtered_struc

    def relax_and_filter(
        self,
        structures: list[Structure],
        energies: list[float] | None = None,
        potential_load_path: str = "MatterSim-v1.0.0-5M.pth",
    ):
        if self.relax and energies is None:
            relaxed_structures, energies = relax_structures(
                structures, device=self.device, potential_load_path=potential_load_path,
            )
        else:
            relaxed_structures = structures

        evaluator = MetricsEvaluator.from_structures_and_energies(
            structures=relaxed_structures,
            energies=energies,
            original_structures=structures,
            reference=self.reference,
            structure_matcher=self.structure_matcher,
        )

        metric_dict = evaluator.compute_metrics(
            metrics=evaluator.available_metrics,
        )

        mask_list = []
        if "validity" in self.metrics:
            valid_struc = parallel_run(structure_validity, relaxed_structures)
            valid_comp = parallel_run(is_smact_valid, relaxed_structures)
            valid_mask = np.array(valid_comp) & np.array(valid_struc)
            mask_list.append(valid_mask)

        if "novel" in self.metrics:
            mask_list.append(evaluator.is_novel)

        if "unique" in self.metrics:
            mask_list.append(evaluator.is_unique)

        if "stable" in self.metrics:
            mask_list.append(evaluator.is_stable)

        if "synthesizable" in self.metrics:
            pass

        mask_all = np.logical_and.reduce(mask_list)
        return mask_all, relaxed_structures, metric_dict

    @staticmethod
    def get_slice(data: list, mask: NDArray) -> list:
        """Filters a list of data points based on a boolean mask."""
        assert len(data) == len(mask), "Data and mask must have the same length."
        return [x for x, m in zip(data, mask) if m]


class OptEval:
    def __init__(
        self,
        relax: bool = True,
        silent: bool = True,
        device: str | None = None,
        structure_matcher: Literal["ordered", "disordered"] = "disordered",
        **kwargs,
    ) -> None:
        self.relax = relax
        self.silent = silent
        self.device = get_device(device)
        self.structure_matcher = (
            DefaultDisorderedStructureMatcher()
            if structure_matcher == "disordered"
            else DefaultOrderedStructureMatcher()
        )
        self.cfg = OmegaConf.create(kwargs)

        _path = hf_hub_download(
            repo_id="jwchen25/MatInvent",
            filename="reference_MP2020correction.gz",
        )
        self.reference = LMDBGZSerializer().deserialize(_path)

    def __call__(
        self,
        structures: list[Structure],
        energies: list[float] | None = None,
        potential_load_path: str = "MatterSim-v1.0.0-5M.pth",
    ):
        structures = self.pre_filter_ehull(structures)

        if self.silent:
            buffer = io.StringIO()
            with contextlib.redirect_stdout(buffer), contextlib.redirect_stderr(buffer):
                relaxed_struc, metrics = self.relax_and_filter(
                    structures, energies, potential_load_path
                )
        else:
            relaxed_struc, metrics = self.relax_and_filter(
                structures, energies, potential_load_path
            )

        return metrics, relaxed_struc

    def pre_filter_ehull(
        self,
        structures: list[Structure],
    ):
        uni_element_set = set()
        for s in structures:
            uni_element_set.update([str(el) for el in s.composition.elements])
        terminal_systems = uni_element_set
        ref_set = set(self.reference.entries_by_chemsys.keys())
        missing_terminals = list(terminal_systems - ref_set)
        terminals_in_reference = terminal_systems & ref_set
        missing_energy = [
            chemsys
            for chemsys in terminals_in_reference
            if all([np.isnan(e.energy) for e in self.reference.entries_by_chemsys[chemsys]])
        ]

        mask = []
        bad_elements = set(missing_terminals + missing_energy)
        for struc in structures:
            struc_element_set = set([str(e) for e in struc.composition.elements])
            if len(struc_element_set & bad_elements) > 0:
                mask.append(False)
            else:
                mask.append(True)

        filtered_struc = [x for x, m in zip(structures, mask) if m]
        return filtered_struc

    def relax_and_filter(
        self,
        structures: list[Structure],
        energies: list[float] | None = None,
        potential_load_path: str = "MatterSim-v1.0.0-5M.pth",
    ):
        if self.relax and energies is None:
            relaxed_structures, energies = relax_structures(
                structures, device=self.device, potential_load_path=potential_load_path,
            )
        else:
            relaxed_structures = structures

        evaluator = MetricsEvaluator.from_structures_and_energies(
            structures=relaxed_structures,
            energies=energies,
            original_structures=structures,
            reference=self.reference,
            structure_matcher=self.structure_matcher,
        )

        metric_dict = evaluator.compute_metrics(
            metrics=evaluator.available_metrics,
        )

        return relaxed_structures, metric_dict
