from functools import cached_property
from typing import Iterable, Iterator, Mapping

import numpy as np
from pymatgen.entries.computed_entries import ComputedStructureEntry

from utils.evaluation.utils import generate_chemsys_dict, generate_reduced_formula_dict


class ReferenceDataset(Iterable[ComputedStructureEntry]):
    def __init__(self, name: str, impl: "ReferenceDatasetImpl"):
        self.name = name
        self.impl = impl

    @staticmethod
    def from_entries(name: str, entries: Iterable[ComputedStructureEntry]) -> "ReferenceDataset":
        return ReferenceDataset(name, ReferenceDatasetImpl(entries))

    def __iter__(self) -> Iterator[ComputedStructureEntry]:
        yield from self.impl

    def __len__(self) -> int:
        return len(self.impl)

    @property
    def entries_by_reduced_formula(self) -> Mapping[str, list[ComputedStructureEntry]]:
        return self.impl.entries_by_reduced_formula

    @property
    def entries_by_chemsys(self) -> Mapping[str, list[ComputedStructureEntry]]:
        return self.impl.entries_by_chemsys

    @cached_property
    def is_ordered(self) -> bool:
        return all(e.structure.is_ordered for e in self)


class ReferenceDatasetImpl(Iterable[ComputedStructureEntry]):
    def __init__(self, entries: Iterable[ComputedStructureEntry]):
        self._entries = tuple(entries)

    def __iter__(self) -> Iterator[ComputedStructureEntry]:
        return iter(self._entries)

    def __len__(self) -> int:
        return len(self._entries)

    @cached_property
    def entries_by_reduced_formula(self) -> Mapping[str, list[ComputedStructureEntry]]:
        return generate_reduced_formula_dict(self._entries)

    @cached_property
    def entries_by_chemsys(self) -> Mapping[str, list[ComputedStructureEntry]]:
        return generate_chemsys_dict(self._entries)
