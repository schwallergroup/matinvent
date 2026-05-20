import logging
from collections import defaultdict
from typing import Iterable, List, Mapping

import numpy as np
from pymatgen.analysis.structure_matcher import StructureMatcher
from pymatgen.core.structure import Structure
from pymatgen.entries.computed_entries import ComputedStructureEntry
from tqdm import tqdm

from utils.evaluation.reference_dataset import ReferenceDataset
from utils.evaluation.structure_matcher import (
    DefaultDisorderedStructureMatcher,
    DisorderedStructureMatcher,
    OrderedStructureMatcher,
)

logger = logging.getLogger(__name__)


def get_matches(matcher: StructureMatcher, d1: List[Structure], d2: List[Structure]) -> dict[int, list[int]]:
    matches: dict[int, list[int]] = defaultdict(list)
    for i in range(len(d1)):
        for j in range(len(d2)):
            if matcher.fit(d1[i], d2[j]):
                matches[i].append(j)
    return matches


def get_unique(matcher: StructureMatcher, structures: List[Structure]) -> List[int]:
    if len(structures) == 1:
        return [0]
    unique_strucs: list[Structure] = []
    unique_idx: list[int] = []
    for idx, s in enumerate(structures):
        if all(not matcher.fit(s, u) for u in unique_strucs):
            unique_strucs.append(s)
            unique_idx.append(idx)
    return unique_idx


def matches_to_mask(match_idx: Iterable[int], num_samples: int) -> np.ndarray:
    mask = np.zeros(num_samples, dtype=bool)
    mask[list(match_idx)] = True
    return mask


def get_global_index_from_local_index(
    entries_mapping: Mapping[str, list[ComputedStructureEntry]],
    local_index: Mapping[str, list[int]],
) -> list[int]:
    return [entries_mapping[k][v].entry_id for k, vs in local_index.items() for v in vs]


def get_mask_from_local_index(
    entries_mapping: Mapping[str, list[ComputedStructureEntry]],
    local_index: Mapping[str, List[int]],
) -> np.ndarray:
    idxs = get_global_index_from_local_index(entries_mapping, local_index)
    total = sum(len(v) for v in entries_mapping.values())
    mask = np.zeros(total, dtype=bool)
    mask[idxs] = True
    return mask


def get_global_match_dict(
    data_mapping: Mapping[str, list[ComputedStructureEntry]],
    ref_mapping: Mapping[str, list[ComputedStructureEntry]],
    local_index: Mapping[str, dict[int, list[int]]],
) -> dict[int, list[str]]:
    result = {}
    for k, match_dict in local_index.items():
        if not match_dict or max((len(v) for v in match_dict.values()), default=0) == 0:
            continue
        data_ents = data_mapping[k]
        ref_ents = ref_mapping[k]
        for d_ix, r_ixs in match_dict.items():
            result[data_ents[d_ix].entry_id] = [ref_ents[r].data["material_id"] for r in r_ixs]
    return result


def get_dataset_matcher(all_ordered: bool, matcher: StructureMatcher) -> "DatasetMatcher":
    return OrderedDatasetMatcher(matcher) if all_ordered else DisorderedDatasetMatcher(matcher)


class OrderedDatasetUniquenessComputer:
    def __init__(self, matcher: StructureMatcher = DefaultDisorderedStructureMatcher()):
        self.matcher = matcher

    def __call__(self, dataset: ReferenceDataset) -> np.ndarray:
        local_index: dict[str, List[int]] = {}
        for rf, entries in tqdm(dataset.entries_by_reduced_formula.items(), desc="Finding unique structures"):
            local_index[rf] = get_unique(self.matcher, [e.structure for e in entries])
        return get_mask_from_local_index(dataset.entries_by_reduced_formula, local_index)


class DisorderedDatasetUniquenessComputer:
    def __init__(self, matcher: StructureMatcher = DefaultDisorderedStructureMatcher()):
        self.matcher = matcher

    def __call__(self, dataset: ReferenceDataset) -> np.ndarray:
        local_index: dict[str, List[int]] = {}
        for chemsys, entries in tqdm(dataset.entries_by_chemsys.items(), desc="Finding unique structures"):
            local_index[chemsys] = get_unique(self.matcher, [e.structure for e in entries])
        return get_mask_from_local_index(dataset.entries_by_chemsys, local_index)


class DatasetMatcher:
    def __init__(self, matcher: OrderedStructureMatcher | DisorderedStructureMatcher):
        self.matcher = matcher

    def grouped_entries(self, dataset: ReferenceDataset) -> Mapping[str, list[ComputedStructureEntry]]:
        raise NotImplementedError

    def __call__(self, dataset: ReferenceDataset, reference: ReferenceDataset) -> dict[int, list[str]]:
        local: dict[str, dict[int, list[int]]] = {}
        grouped_data = self.grouped_entries(dataset)
        grouped_ref = self.grouped_entries(reference)
        for key, data_ents in tqdm(grouped_data.items(), desc="Finding novel structures"):
            local[key] = get_matches(
                self.matcher,
                [e.structure for e in data_ents],
                [e.structure for e in grouped_ref.get(key, [])],
            )
        return get_global_match_dict(grouped_data, grouped_ref, local)


class OrderedDatasetMatcher(DatasetMatcher):
    def grouped_entries(self, dataset: ReferenceDataset):
        return dataset.entries_by_reduced_formula


class DisorderedDatasetMatcher(DatasetMatcher):
    def grouped_entries(self, dataset: ReferenceDataset):
        return dataset.entries_by_chemsys
