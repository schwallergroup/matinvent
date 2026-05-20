import itertools
import logging
from collections import Counter
from copy import deepcopy
from dataclasses import dataclass
from functools import cached_property
from typing import Literal, Sequence

import cachetools
import numpy as np
import numpy.typing
import smact
from pandas import DataFrame
from pymatgen.core.composition import Element
from pymatgen.core.structure import Structure
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
from smact.screening import pauling_test
from tqdm import tqdm

from utils.evaluation.metrics_core import BaseAggregateMetric, BaseMetric, BaseMetricsCapability
from utils.evaluation.reference_dataset import ReferenceDataset
from utils.evaluation.dataset_matcher import (
    DisorderedDatasetUniquenessComputer,
    OrderedDatasetUniquenessComputer,
    get_dataset_matcher,
    matches_to_mask,
)
from utils.evaluation.structure_matcher import DisorderedStructureMatcher, OrderedStructureMatcher
from utils.evaluation.symmetry_analysis import DefaultSpaceGroupAnalyzer, DisorderedSpaceGroupAnalyzer

logger = logging.getLogger(__name__)


def structure_validity(structure: Structure, cutoff: float = 0.5) -> bool:
    dist_mat = structure.distance_matrix
    dist_mat = dist_mat + np.diag(np.ones(dist_mat.shape[0]) * (cutoff + 10.0))
    return not (dist_mat.min() < cutoff or structure.volume < 0.1)


def smact_validity(comp, count, use_pauling_test=True, include_alloys=True, use_element_symbol=False) -> bool:
    assert len(comp) == len(count)
    elem_symbols = comp if use_element_symbol else tuple([str(Element.from_Z(Z=e)) for e in comp])
    space = smact.element_dictionary(elem_symbols)
    smact_elems = [e[1] for e in space.items()]
    electronegs = [e.pauling_eneg for e in smact_elems]
    ox_combos = [e.oxidation_states for e in smact_elems]
    if len(set(elem_symbols)) == 1:
        return True
    if include_alloys and all(e in smact.metals for e in elem_symbols):
        return True
    threshold = np.max(count)
    compositions = []
    for ox_states in itertools.product(*ox_combos):
        stoichs = [(c,) for c in count]
        cn_e, cn_r = smact.neutral_ratios(ox_states, stoichs=stoichs, threshold=threshold)
        if cn_e:
            try:
                en_ok = pauling_test(ox_states, electronegs) if use_pauling_test else True
            except TypeError:
                en_ok = True
            if en_ok:
                for ratio in cn_r:
                    compositions.append(tuple([elem_symbols, ox_states, ratio]))
    compositions = list(set([(i[0], i[2]) for i in compositions]))
    return len(compositions) > 0


def is_smact_valid(structure: Structure) -> bool:
    elem_counter = Counter(structure.atomic_numbers)
    composition = [(elem, elem_counter[elem]) for elem in sorted(elem_counter.keys())]
    elems, counts = list(zip(*composition))
    counts = np.array(counts)
    counts = counts / np.gcd.reduce(counts)
    comps = tuple(np.array(counts).astype("int"))
    try:
        return smact_validity(comp=elems, count=comps, use_pauling_test=True, include_alloys=True)
    except (TypeError, UnicodeDecodeError):
        return smact_validity(comp=elems, count=comps, use_pauling_test=True, include_alloys=True)


def get_space_group(structure: Structure, analyzer_cls=DefaultSpaceGroupAnalyzer) -> str:
    try:
        return analyzer_cls(structure=structure).get_space_group_symbol()
    except TypeError:
        return "P1"


class StructureMetricsCapability(BaseMetricsCapability):
    name: str = "structure_capability"

    def __init__(
        self,
        structure_summaries: list,
        reference_dataset: ReferenceDataset,
        structure_matcher: OrderedStructureMatcher | DisorderedStructureMatcher,
        n_failed_jobs: int = 0,
    ) -> None:
        super().__init__(structure_summaries=structure_summaries, n_failed_jobs=n_failed_jobs)
        _structures = [s.structure for s in structure_summaries]
        all_ordered = all(s.is_ordered for s in _structures) and reference_dataset.is_ordered
        self.reference_dataset = reference_dataset
        self.structure_matcher = structure_matcher
        self._ensure_material_ids()
        self.uniqueness_computer = (
            OrderedDatasetUniquenessComputer(structure_matcher)
            if all_ordered
            else DisorderedDatasetUniquenessComputer(structure_matcher)
        )
        self.dataset_matcher = get_dataset_matcher(all_ordered, structure_matcher)

    def _ensure_material_ids(self):
        if len(self.reference_dataset) > 0:
            first = next(iter(self.reference_dataset))
            if first.data.get("material_id") is None:
                for i, entry in enumerate(self.reference_dataset):
                    entry.data["material_id"] = i

    @property
    def structures(self) -> list[Structure]:
        return [s.structure for s in self._structure_summaries]

    @cached_property
    def chemistry_agnostic_structures(self) -> list[Structure]:
        strucs = [deepcopy(s) for s in self.structures]
        for s in strucs:
            s.replace_species({Element(k.name): Element("Cs") for k in set(s.species)})
        return strucs

    @cached_property
    def is_unique(self) -> numpy.typing.NDArray[np.bool_]:
        return self.uniqueness_computer(self.dataset)

    @cached_property
    def is_in_reference(self) -> numpy.typing.NDArray[np.bool_]:
        return matches_to_mask(self.matches_in_reference.keys(), len(self.dataset))

    @cached_property
    def is_novel(self) -> numpy.typing.NDArray[np.bool_]:
        return ~self.is_in_reference

    @cached_property
    def matches_in_reference(self) -> dict[int, list[str]]:
        return self.dataset_matcher(self.dataset, self.reference_dataset)

    def as_dataframe(self) -> DataFrame:
        return DataFrame(
            data={"is_unique": self.is_unique, "is_novel": self.is_novel},
            index=[e.entry_id for e in self.dataset],
        )


# ---- Aggregate metrics ----

@dataclass(frozen=True)
class BaseStructureMetric(BaseMetric):
    required_capabilities = (StructureMetricsCapability,)

    @property
    def name(self) -> str:
        return "base_structure_metric"

    def __init__(self, structure_capability: StructureMetricsCapability, **kwargs):
        self.structure_capability = structure_capability
        self.reference_dataset = structure_capability.reference_dataset
        self.dataset = structure_capability.dataset


class FracUniqueStructures(BaseStructureMetric, BaseAggregateMetric):
    aggregation_method: Literal["mean"] = "mean"
    name = "frac_unique_structures"
    pre_aggregation_name = "unique"

    @property
    def description(self) -> str:
        return "Fraction of unique structures in sampled data."

    def compute_pre_aggregation_values(self) -> numpy.typing.NDArray:
        return self.structure_capability.is_unique


class FracNovelStructures(BaseStructureMetric, BaseAggregateMetric):
    aggregation_method: Literal["mean"] = "mean"
    name = "frac_novel_structures"
    pre_aggregation_name = "novel"

    @property
    def description(self) -> str:
        return "Fraction of novel structures in sampled data."

    def compute_pre_aggregation_values(self) -> numpy.typing.NDArray:
        return self.structure_capability.is_novel


class AvgStructureValidity(BaseStructureMetric, BaseAggregateMetric):
    aggregation_method: Literal["mean"] = "mean"
    name = "avg_structure_validity"
    pre_aggregation_name = "structure_validity"

    @property
    def description(self) -> str:
        return "Average structural validity."

    def compute_pre_aggregation_values(self) -> numpy.typing.NDArray:
        return np.array([structure_validity(s) for s in self.structure_capability.structures])


class AvgCompValidity(BaseStructureMetric, BaseAggregateMetric):
    aggregation_method: Literal["mean"] = "mean"
    name = "avg_comp_validity"
    pre_aggregation_name = "comp_validity"

    @property
    def description(self) -> str:
        return "Average composition validity (SMACT)."

    def compute_pre_aggregation_values(self) -> numpy.typing.NDArray:
        return np.array([is_smact_valid(s) for s in self.structure_capability.structures])
