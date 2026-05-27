"""Step 3: filter structures down to novel ones (not in LeMat-Bulk).

Uses the bundled BAWL fingerprinting implementation (_bawl.py) for O(1)
lookup per query structure against pre-computed fingerprints in LeMat-Bulk.

Fallback: composition prefilter + StructureMatcher when the BAWL fingerprint
column is not present in the reference dataset.
"""

from __future__ import annotations

import warnings
from concurrent.futures import ProcessPoolExecutor
from functools import lru_cache
from typing import List, Optional, Set, Tuple

import numpy as np
from pymatgen.core import Lattice, Structure

from utils.sun_filter._bawl import get_bawl_hash

warnings.filterwarnings("ignore")


# --------------------------------------------------------------------------- #
# Fast path: BAWL fingerprinting                                              #
# --------------------------------------------------------------------------- #

@lru_cache(maxsize=None)
def _load_reference_fingerprints(
    reference_dataset: str,
    reference_config: str,
) -> Set[str]:
    """Load the BAWL fingerprint set from the reference dataset (cached)."""
    from datasets import load_dataset

    ds = load_dataset(
        reference_dataset,
        reference_config,
        split="train",
    )

    fp_col = "entalpic_fingerprint"
    if fp_col not in ds.column_names:
        raise KeyError(
            f"Column '{fp_col}' not found in {reference_dataset}. "
            f"Available: {ds.column_names}"
        )

    fps: Set[str] = set()
    for fp in ds[fp_col]:
        if fp is not None:
            fps.add(fp)
    return fps


def _get_bawl_fingerprint(structure: Structure) -> Optional[str]:
    try:
        return get_bawl_hash(structure)
    except Exception:
        return None


def _bawl_filter_novel(
    structures: List[Structure],
    reference_fps: Set[str],
    n_jobs: int,
) -> List[int]:
    """Return novel indices using BAWL fingerprint lookup."""
    if n_jobs <= 1:
        query_fps = [_get_bawl_fingerprint(s) for s in structures]
    else:
        with ProcessPoolExecutor(max_workers=n_jobs) as ex:
            query_fps = list(ex.map(_get_bawl_fingerprint, structures))

    novel = []
    for i, fp in enumerate(query_fps):
        if fp is None:
            # Fingerprint failed (unusual structure) → treat as novel
            novel.append(i)
            continue
        if fp not in reference_fps:
            novel.append(i)
    return novel


# --------------------------------------------------------------------------- #
# Fallback path: composition prefilter + StructureMatcher                    #
# --------------------------------------------------------------------------- #

def _row_to_structure(row: dict) -> Optional[Structure]:
    try:
        lattice = Lattice(np.array(row["lattice_vectors"]))
        species = row["species_at_sites"]
        coords = np.array(row["cartesian_site_positions"])
        return Structure(lattice, species, coords, coords_are_cartesian=True)
    except Exception:
        return None


def _sm_filter_novel(
    structures: List[Structure],
    reference_dataset: str,
    reference_config: str,
    n_jobs: int,
    ltol: float,
    stol: float,
    angle_tol: float,
) -> List[int]:
    """Fallback novelty check: composition prefilter + StructureMatcher."""
    from collections import Counter, defaultdict

    from datasets import load_dataset
    from pymatgen.analysis.structure_matcher import StructureMatcher
    from pymatgen.core import Composition

    ds = load_dataset(reference_dataset, reference_config, split="train")

    query_formulas = []
    for s in structures:
        try:
            query_formulas.append(s.composition.reduced_formula)
        except Exception:
            query_formulas.append(None)

    ref_formula_set: Set[str] = set()
    for row in ds:
        try:
            comp = Composition(Counter(row["species_at_sites"]))
            ref_formula_set.add(comp.reduced_formula)
        except Exception:
            continue

    need_sm = {f for f in set(query_formulas) if f is not None and f in ref_formula_set}

    ref_groups: dict[str, list] = defaultdict(list)
    if need_sm:
        for row in ds:
            try:
                comp = Composition(Counter(row["species_at_sites"]))
                formula = comp.reduced_formula
                if formula not in need_sm:
                    continue
                s = _row_to_structure(row)
                if s is not None:
                    ref_groups[formula].append(s)
            except Exception:
                continue

    sm = StructureMatcher(ltol=ltol, stol=stol, angle_tol=angle_tol)
    novel = []
    for i, (s, formula) in enumerate(zip(structures, query_formulas)):
        if formula is None:
            continue
        if formula not in ref_formula_set:
            novel.append(i)
            continue
        refs = ref_groups.get(formula, [])
        if not any(sm.fit(s, ref) for ref in refs):
            novel.append(i)
    return novel


# --------------------------------------------------------------------------- #
# Public filter                                                                #
# --------------------------------------------------------------------------- #

def filter_novel(
    structures: List[Structure],
    reference_dataset: str = "LeMaterial/LeMat-Bulk",
    reference_config: str = "compatible_pbe",
    n_jobs: int = 4,
    ltol: float = 0.2,
    stol: float = 0.3,
    angle_tol: float = 5.0,
) -> List[int]:
    """Return indices (into *structures*) of the novel structures.

    A structure is novel if no equivalent structure exists in the reference dataset.

    Uses BAWL fingerprinting for O(1) lookup per structure when the
    `entalpic_fingerprint` column is present in the dataset. Falls back to
    composition prefilter + StructureMatcher otherwise.

    Parameters
    ----------
    structures:
        List of pymatgen Structures to evaluate.
    reference_dataset:
        HuggingFace dataset name for the reference.
    reference_config:
        Config/subset of the dataset.
    n_jobs:
        Number of parallel workers for BAWL fingerprint computation.
    ltol, stol, angle_tol:
        StructureMatcher tolerances (fallback path only).

    Returns
    -------
    Sorted list of indices that are novel.
    """
    if not structures:
        return []

    try:
        ref_fps = _load_reference_fingerprints(reference_dataset, reference_config)
        return _bawl_filter_novel(structures, ref_fps, n_jobs)
    except KeyError:
        # entalpic_fingerprint column not present → use StructureMatcher fallback
        return _sm_filter_novel(
            structures, reference_dataset, reference_config,
            n_jobs, ltol, stol, angle_tol,
        )
