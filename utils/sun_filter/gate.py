"""Main SUN/MSUN filtering gate.

Pipeline order:
  1. Uniqueness  — keep deduplicated structures (StructureMatcher)
  2. Validity    — keep physically plausible structures
  3. Novelty     — keep structures not in reference dataset
  4. Ehull       — relax with specified MLIP(s), compute Ehull, drop failures
"""

from __future__ import annotations

import math
import warnings
from typing import List, Optional

from pymatgen.core import Structure

from utils.sun_filter._batch_relax import batch_relax, _resolve_device
from utils.sun_filter._ehull import get_energy_above_hull
from utils.sun_filter._novelty import filter_novel
from utils.sun_filter._uniqueness import filter_unique
from utils.sun_filter._validity import filter_valid

warnings.filterwarnings("ignore")

_SUPPORTED_MLIPS = {"orb", "uma", "mace_mp"}


def sun_gate(
    structures: List[Structure],
    candidate_indices: list[int],
    stability_threshold: float = 0.0,
    metastability_threshold: float = 0.1,
    device: Optional[str] = None,
    fmax: float = 0.02,
    max_steps: int = 100,
    n_jobs: int = 4,
    reference_dataset: str = "LeMaterial/LeMat-Bulk",
    reference_config: str = "compatible_pbe",
    mlip_names: List[str] = ["orb", "uma", "mace_mp"],
) -> tuple[list[bool | None], list[bool | None], list[float | None]]:
    """Filter a list of structures through the SUN/MSUN pipeline.

    Steps
    -----
    1. Uniqueness  : keep one representative from each group of equivalent structures.
    2. Validity    : keep structures that pass physical plausibility checks.
    3. Novelty     : keep structures not already in the reference dataset.
    4. Ehull       : relax with MLIP(s), compute Ehull, remove failures.

    The final sets are:
    - **sun**  : Ehull ≤ stability_threshold
    - **msun** : stability_threshold < Ehull ≤ metastability_threshold

    Parameters
    ----------
    structures:
        Input pymatgen Structures (may contain duplicates/invalid/known structures).
    mlip_names:
        One or more of ["orb", "uma", "mace_mp"].
    stability_threshold:
        Ehull threshold (eV/atom) for classifying a structure as stable (default 0.0).
    metastability_threshold:
        Ehull threshold (eV/atom) for classifying a structure as metastable (default 0.1).
    device:
        PyTorch device string, e.g. "cpu" or "cuda".
        Defaults to "cuda" if a GPU is available, otherwise "cpu".
    fmax:
        Force convergence criterion for geometry relaxation (eV/Å).
    max_steps:
        Maximum number of optimisation steps per structure.
    n_jobs:
        Number of parallel workers for validity/novelty checks and ASE fallback relaxation.
    reference_dataset:
        HuggingFace dataset identifier for the novelty reference.
    reference_config:
        Dataset split/config for the novelty reference.

    Returns
    -------
    three lists of length ``len(structures)``.  Non-candidate
    positions carry ``None``; candidates that failed a stage carry ``None``
    as well (caller converts to ``False`` when the gate is active).
    """

    # Validate MLIP names
    unknown = set(mlip_names) - _SUPPORTED_MLIPS
    if unknown:
        raise ValueError(
            f"Unknown MLIP(s): {unknown}. Supported: {_SUPPORTED_MLIPS}"
        )
    if not mlip_names:
        raise ValueError("mlip_names must not be empty.")

    n = len(structures)
    device = _resolve_device(device)
    sun_flags: list[bool | None] = [None] * n
    msun_flags: list[bool | None] = [None] * n
    ehull_vals: list[float | None] = [None] * n

    if not candidate_indices:
        return sun_flags, msun_flags, ehull_vals

    cand_strucs = [structures[i] for i in candidate_indices]  # type: ignore[misc]

    # Stage 1: uniqueness within this batch
    uniq_local = filter_unique(cand_strucs, n_jobs=1)
    uniq_strucs = [cand_strucs[i] for i in uniq_local]

    # Stage 2: physical validity (distance + density)
    valid_in_uniq = filter_valid(uniq_strucs, n_jobs=n_jobs)
    valid_strucs = [uniq_strucs[j] for j in valid_in_uniq]

    # Stage 3: novelty against reference dataset
    novel_in_valid = filter_novel(
        valid_strucs,
        reference_dataset=reference_dataset,
        reference_config=reference_config,
        n_jobs=n_jobs,
    )
    novel_strucs = [valid_strucs[k] for k in novel_in_valid]

    if not novel_strucs:
        return sun_flags, msun_flags, ehull_vals

    # Stage 4: relax with each MLIP and accumulate per-structure E_hull values
    ehull_per_novel: list[list[float]] = [[] for _ in novel_strucs]
    for mlip_name in mlip_names:
        try:
            relax_results = batch_relax(
                novel_strucs, mlip_name=mlip_name, device=device,
                fmax=fmax, max_steps=max_steps, n_jobs=n_jobs,
            )
            for local_i, (relaxed, energy) in enumerate(relax_results):
                if relaxed is not None and energy is not None:
                    ehull = get_energy_above_hull(energy, relaxed.composition, hull_type=mlip_name)
                    if ehull is not None and math.isfinite(ehull):
                        ehull_per_novel[local_i].append(ehull)
        except Exception:
            print("SUN gate MLIP %r failed for this batch", mlip_name, exc_info=True)

    # Map novel-structure results back to original batch indices
    for local_k in range(len(novel_strucs)):
        in_uniq = valid_in_uniq[novel_in_valid[local_k]]  # pos in uniq_strucs
        in_cand = uniq_local[in_uniq]                     # pos in candidate_strucs
        orig_idx = candidate_indices[in_cand]             # pos in full batch

        ehulls = ehull_per_novel[local_k]
        if ehulls:
            mean_ehull = sum(ehulls) / len(ehulls)
            ehull_vals[orig_idx] = mean_ehull
            sun_flags[orig_idx] = mean_ehull <= stability_threshold
            msun_flags[orig_idx] = (
                stability_threshold < mean_ehull <= metastability_threshold
            )

    return sun_flags, msun_flags, ehull_vals
