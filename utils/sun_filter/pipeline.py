"""Main SUN/MSUN filtering pipeline.

Pipeline order:
  1. Uniqueness  — keep deduplicated structures (StructureMatcher)
  2. Validity    — keep physically plausible structures
  3. Novelty     — keep structures not in reference dataset
  4. Ehull       — relax with specified MLIP(s), compute Ehull, drop failures

Outputs:
  sun_strucs / sun_ehull     — stable   (Ehull ≤ stability_threshold)
  msun_strucs / msun_ehull   — metastable (stability_threshold < Ehull ≤ metastability_threshold)
  un_strucs / un_ehull       — all successful after step 4, sorted by Ehull
  sun_rate / msun_rate       — counts relative to the original input length
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
from pymatgen.core import Structure

from utils.sun_filter._batch_relax import batch_relax, _resolve_device
from utils.sun_filter._ehull import HULL_TYPE_MAP, get_energy_above_hull
from utils.sun_filter._novelty import filter_novel
from utils.sun_filter._uniqueness import filter_unique
from utils.sun_filter._validity import filter_valid

warnings.filterwarnings("ignore")

# --------------------------------------------------------------------------- #
# MLIP configuration                                                           #
# --------------------------------------------------------------------------- #

_SUPPORTED_MLIPS = {"orb", "uma", "mace_mp"}


# --------------------------------------------------------------------------- #
# Result container                                                             #
# --------------------------------------------------------------------------- #

@dataclass
class SUNFilterResult:
    """Result of the sun_filter pipeline.

    Attributes
    ----------
    sun_strucs:
        Stable structures (Ehull ≤ stability_threshold), sorted by Ehull ascending.
    sun_ehull:
        Corresponding Ehull values for sun_strucs (eV/atom).
    msun_strucs:
        Metastable structures, sorted by Ehull ascending.
    msun_ehull:
        Corresponding Ehull values for msun_strucs (eV/atom).
    un_strucs:
        All unique+valid+novel structures with a successful Ehull,
        sorted by Ehull ascending (superset of sun_strucs + msun_strucs).
    un_ehull:
        Corresponding Ehull values for un_strucs (eV/atom).
    sun_rate:
        len(sun_strucs) / len(input_structures).
    msun_rate:
        len(msun_strucs) / len(input_structures).
    """

    sun_strucs: List[Structure] = field(default_factory=list)
    sun_ehull: List[float] = field(default_factory=list)
    msun_strucs: List[Structure] = field(default_factory=list)
    msun_ehull: List[float] = field(default_factory=list)
    un_strucs: List[Structure] = field(default_factory=list)
    un_ehull: List[float] = field(default_factory=list)
    sun_rate: float = 0.0
    msun_rate: float = 0.0


# --------------------------------------------------------------------------- #
# Ehull aggregation across MLIPs                                               #
# --------------------------------------------------------------------------- #

def _compute_ehull_all_mlips(
    structures: List[Structure],
    mlip_names: List[str],
    device: str,
    fmax: float,
    max_steps: int,
    batch_size: int,
    n_jobs: int,
) -> List[Optional[float]]:
    """Relax + compute Ehull for each structure using each MLIP; return averaged values.

    Returns a list of length len(structures). Each element is the mean Ehull
    across successful MLIPs, or None if ALL MLIPs failed for that structure.
    """
    # Per-MLIP Ehull arrays: shape (n_mlips, n_structures)
    per_mlip: Dict[str, List[Optional[float]]] = {}

    for mlip_name in mlip_names:
        hull_type = HULL_TYPE_MAP.get(mlip_name, mlip_name)
        relax_results = batch_relax(
            structures,
            mlip_name=mlip_name,
            device=device,
            fmax=fmax,
            max_steps=max_steps,
            batch_size=batch_size,
            n_jobs=n_jobs,
        )

        ehull_values: List[Optional[float]] = []
        for (relaxed_struct, energy), original_struct in zip(relax_results, structures):
            if relaxed_struct is None or energy is None:
                ehull_values.append(None)
                continue
            ehull = get_energy_above_hull(
                total_energy=energy,
                composition=relaxed_struct.composition,
                hull_type=hull_type,
            )
            ehull_values.append(ehull)

        per_mlip[mlip_name] = ehull_values

    # Average across successful MLIPs per structure
    averaged: List[Optional[float]] = []
    for i in range(len(structures)):
        vals = [
            per_mlip[m][i]
            for m in mlip_names
            if per_mlip[m][i] is not None
        ]
        averaged.append(float(np.mean(vals)) if vals else None)

    return averaged


# --------------------------------------------------------------------------- #
# Main pipeline                                                                #
# --------------------------------------------------------------------------- #

def vsun_filter(
    data: list,
    structures: List[Structure],
    stability_threshold: float = 0.0,
    metastability_threshold: float = 0.125,
    device: Optional[str] = None,
    fmax: float = 0.02,
    max_steps: int = 50,
    n_jobs: int = 4,
    batch_size: int = 16,
    reference_dataset: str = "LeMaterial/LeMat-Bulk",
    reference_config: str = "compatible_pbe",
    mlip_names: List[str] = ["orb", "uma", "mace_mp"],
) -> SUNFilterResult:
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
    - **un**   : all structures with a successful Ehull computation (sun ∪ msun ∪ rest)

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
    batch_size:
        Number of structures per TorchSim GPU batch.
    reference_dataset:
        HuggingFace dataset identifier for the novelty reference.
    reference_config:
        Dataset split/config for the novelty reference.

    Returns
    -------
    SUNFilterResult
        Contains sun_strucs, sun_ehull, msun_strucs, msun_ehull,
        un_strucs, un_ehull, sun_rate, msun_rate.
    """
    n_input = len(structures)
    if n_input == 0:
        return [], []

    device = _resolve_device(device)

    # Validate MLIP names
    unknown = set(mlip_names) - _SUPPORTED_MLIPS
    if unknown:
        raise ValueError(
            f"Unknown MLIP(s): {unknown}. Supported: {_SUPPORTED_MLIPS}"
        )
    if not mlip_names:
        raise ValueError("mlip_names must not be empty.")

    # ------------------------------------------------------------------ #
    # Step 1: Uniqueness                                                   #
    # ------------------------------------------------------------------ #
    unique_indices = filter_unique(structures, n_jobs=n_jobs)
    unique_strucs = [structures[i] for i in unique_indices]
    unique_data = [data[i] for i in unique_indices]

    if not unique_strucs:
        return [], []

    # ------------------------------------------------------------------ #
    # Step 2: Validity                                                     #
    # ------------------------------------------------------------------ #
    valid_rel_indices = filter_valid(unique_strucs, n_jobs=n_jobs)
    valid_strucs = [unique_strucs[i] for i in valid_rel_indices]
    valid_data = [unique_data[i] for i in valid_rel_indices]

    if not valid_strucs:
        return [], []

    # ------------------------------------------------------------------ #
    # Step 3: Novelty                                                      #
    # ------------------------------------------------------------------ #
    novel_rel_indices = filter_novel(
        valid_strucs,
        reference_dataset=reference_dataset,
        reference_config=reference_config,
        n_jobs=n_jobs,
    )
    novel_strucs = [valid_strucs[i] for i in novel_rel_indices]
    novel_data = [valid_data[i] for i in novel_rel_indices]

    if not novel_strucs:
        return [], []

    # ------------------------------------------------------------------ #
    # Step 4: Ehull computation                                            #
    # ------------------------------------------------------------------ #
    ehull_values = _compute_ehull_all_mlips(
        novel_strucs,
        mlip_names=mlip_names,
        device=device,
        fmax=fmax,
        max_steps=max_steps,
        batch_size=batch_size,
        n_jobs=n_jobs,
    )

    # Keep only structures with successful Ehull
    successful_pairs = [
        (d, s, e)
        for d, s, e in zip(novel_data, novel_strucs, ehull_values)
        if e is not None
    ]

    if not successful_pairs:
        return [], []

    # ------------------------------------------------------------------ #
    # Classify and sort                                                    #
    # ------------------------------------------------------------------ #
    # sun_pairs: List[Tuple[Structure, float]] = []
    msun_pairs = []
    # un_pairs: List[Tuple[Structure, float]] = list(successful_pairs)

    for d, s, e in successful_pairs:
        if e <= metastability_threshold:
            s.properties["ehull"] = e
            msun_pairs.append((d, s, e))

    msun_data = [p[0] for p in msun_pairs]
    msun_strucs = [p[1] for p in msun_pairs]
    msun_ehull = [p[2] for p in msun_pairs]
    msun_ratio = len(msun_strucs) / n_input

    return msun_data, msun_strucs, msun_ratio
