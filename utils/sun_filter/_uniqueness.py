"""Step 1: filter a list of structures down to unique ones.

Groups by (reduced_formula, spacegroup_number) then applies StructureMatcher
within each group, keeping the first occurrence of every equivalence class.
Parallelism is applied across groups.
"""

from __future__ import annotations

from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import List, Tuple

from pymatgen.analysis.structure_matcher import StructureMatcher
from pymatgen.core import Structure
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer


def _group_key(structure: Structure) -> Tuple[str, int]:
    try:
        sg = SpacegroupAnalyzer(structure, symprec=0.1).get_space_group_number()
    except Exception:
        sg = -1
    return (structure.composition.reduced_formula, sg)


def _deduplicate_group(
    indexed_structures: List[Tuple[int, Structure]],
    ltol: float,
    stol: float,
    angle_tol: float,
) -> List[int]:
    """Return original indices of unique structures within one group."""
    sm = StructureMatcher(ltol=ltol, stol=stol, angle_tol=angle_tol)
    unique_indices: List[int] = []
    seen: List[Structure] = []
    for orig_idx, s in indexed_structures:
        if not any(sm.fit(s, prev) for prev in seen):
            seen.append(s)
            unique_indices.append(orig_idx)
    return unique_indices


def filter_unique(
    structures: List[Structure],
    n_jobs: int = 4,
    ltol: float = 0.2,
    stol: float = 0.3,
    angle_tol: float = 5.0,
) -> List[int]:
    """Return indices (into *structures*) of the unique structures.

    Parameters
    ----------
    structures:
        Input list of pymatgen Structures.
    n_jobs:
        Number of parallel workers for group deduplication.
    ltol, stol, angle_tol:
        StructureMatcher tolerances.

    Returns
    -------
    Sorted list of indices into *structures* that are unique.
    """
    if not structures:
        return []

    # Group by (formula, spacegroup)
    groups: dict[Tuple[str, int], List[Tuple[int, Structure]]] = defaultdict(list)
    for i, s in enumerate(structures):
        groups[_group_key(s)].append((i, s))

    group_list = list(groups.values())

    if n_jobs <= 1 or len(group_list) == 1:
        unique_indices: List[int] = []
        for group in group_list:
            unique_indices.extend(_deduplicate_group(group, ltol, stol, angle_tol))
    else:
        unique_indices = []
        with ProcessPoolExecutor(max_workers=min(n_jobs, len(group_list))) as ex:
            futures = {
                ex.submit(_deduplicate_group, grp, ltol, stol, angle_tol): grp
                for grp in group_list
            }
            for fut in as_completed(futures):
                unique_indices.extend(fut.result())

    return sorted(unique_indices)
