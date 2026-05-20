import warnings
from dataclasses import dataclass, field
from functools import cached_property

import numpy as np
from pymatgen.core import Structure
from pymatgen.entries.compatibility import Compatibility, MaterialsProject2020Compatibility
from pymatgen.entries.computed_entries import ComputedStructureEntry

from utils.evaluation.utils import compute_rmsd_angstrom, preprocess_structure
from utils.evaluation.vasprunlike import VasprunLike


@dataclass
class MetricsStructureSummary:
    entry: ComputedStructureEntry
    properties: dict[str, float] = field(default_factory=dict)
    original_structure: Structure | None = None

    @staticmethod
    def from_structure_and_energy(
        structure: Structure,
        energy: float,
        properties: dict[str, float] | None = None,
        original_structure: Structure | None = None,
        energy_correction_scheme: Compatibility = MaterialsProject2020Compatibility(),
    ) -> "MetricsStructureSummary":
        vasprun_like = VasprunLike(structure=structure, energy=energy)
        entry = vasprun_like.get_computed_entry(
            inc_structure=True, energy_correction_scheme=energy_correction_scheme,
        )
        if original_structure is None:
            warnings.warn("No original structure found, cannot compute RMSD metric.")
        return MetricsStructureSummary(
            entry=entry, properties=properties or {}, original_structure=original_structure,
        )

    @staticmethod
    def from_structure(
        structure: Structure,
        properties: dict[str, float] | None = None,
    ) -> "MetricsStructureSummary":
        return MetricsStructureSummary(
            entry=ComputedStructureEntry(structure=structure, energy=np.nan),
            properties=properties or {},
        )

    @cached_property
    def rmsd_from_relaxation(self) -> float:
        if self.original_structure is None:
            return np.nan
        return compute_rmsd_angstrom(self.entry.structure, preprocess_structure(self.original_structure))

    @property
    def structure(self) -> Structure:
        return self.entry.structure

    @property
    def chemical_system(self) -> str:
        return self.entry.composition.chemical_system


def get_metrics_structure_summaries(
    structures: list[Structure],
    energies: list[float],
    properties: dict[str, list[float]] | None = None,
    original_structures: list[Structure] | None = None,
    energy_correction_scheme: Compatibility = MaterialsProject2020Compatibility(),
) -> list[MetricsStructureSummary]:
    if properties is None:
        properties = {}
    return [
        MetricsStructureSummary.from_structure_and_energy(
            structure=structures[i],
            energy=energies[i],
            properties={k: v[i] for k, v in properties.items()} if properties else None,
            original_structure=original_structures[i] if original_structures else None,
            energy_correction_scheme=energy_correction_scheme,
        )
        for i in range(len(structures))
    ]
