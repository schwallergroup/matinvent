"""Standalone energy-above-hull calculation using HuggingFace reference data.

Replicates the logic from lemat_genbench/preprocess/reference_energies.py
without importing from that package.

Reference hull data is loaded from:
  HuggingFace dataset: LeMaterial/LeMat-Bulk-MLIP-Hull
  Files: threshold_0_001/{hull_type}_above_hull_dataset.parquet
         threshold_0_001/{hull_type}_above_hull_composition_matrix.npz
"""

from __future__ import annotations

import os
from collections import Counter
from functools import lru_cache
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from pymatgen.analysis.phase_diagram import PDEntry, PhaseDiagram
from pymatgen.core import Composition
from scipy import sparse

_CACHE_DIR = Path.home() / ".cache" / "sun_filter" / "hull_data"

# Map from user-facing MLIP names to hull dataset split names
HULL_TYPE_MAP = {
    "orb": "orb",
    "uma": "uma",
    "mace_mp": "mace_mp",
}


# --------------------------------------------------------------------------- #
# One-hot composition encoding (118 elements, index = atomic number)          #
# --------------------------------------------------------------------------- #

def _one_hot_encode_elements(elements) -> np.ndarray:
    vec = np.zeros(119)
    for el in elements:
        try:
            if hasattr(el, "symbol"):
                sym = el.symbol
            elif hasattr(el, "element"):
                sym = str(el.element)
            else:
                sym = str(el).rstrip("+-0123456789")
            from pymatgen.core.periodic_table import Element as _El
            z = _El(sym).Z
            vec[z] = 1
        except Exception:
            continue
    return vec


# --------------------------------------------------------------------------- #
# Data loading (cached per (hull_type, threshold) pair)                       #
# --------------------------------------------------------------------------- #

@lru_cache(maxsize=None)
def _load_hull_df(hull_type: str, threshold: float = 0.001) -> pd.DataFrame:
    threshold_str = f"{threshold:.3f}".replace(".", "_")
    parquet_name = f"threshold_{threshold_str}/{hull_type}_above_hull_dataset.parquet"

    # Try HuggingFace datasets first (most reliable)
    try:
        from datasets import load_dataset
        ds_dict = load_dataset("LeMaterial/LeMat-Bulk-MLIP-Hull")
        if hull_type in ds_dict:
            df = ds_dict[hull_type].to_pandas()
            if "species_at_sites" in df.columns:
                df["species_at_sites"] = df["species_at_sites"].apply(
                    lambda x: x.tolist() if hasattr(x, "tolist") else x
                )
            return df
    except Exception:
        pass

    # Fall back to huggingface_hub file download
    try:
        from huggingface_hub import hf_hub_download
        _CACHE_DIR.mkdir(parents=True, exist_ok=True)
        file_path = hf_hub_download(
            repo_id="LeMaterial/LeMat-Bulk-MLIP-Hull",
            filename=parquet_name,
            repo_type="dataset",
            cache_dir=str(_CACHE_DIR),
        )
        df = pd.read_parquet(file_path)
        if "species_at_sites" in df.columns:
            df["species_at_sites"] = df["species_at_sites"].apply(
                lambda x: x.tolist() if hasattr(x, "tolist") else x
            )
        return df
    except Exception as e:
        raise RuntimeError(
            f"Cannot load hull dataset for hull_type='{hull_type}'. "
            f"Error: {e}"
        ) from e


@lru_cache(maxsize=None)
def _load_hull_matrix(hull_type: str, threshold: float = 0.001) -> np.ndarray:
    threshold_str = f"{threshold:.3f}".replace(".", "_")
    npz_name = f"threshold_{threshold_str}/{hull_type}_above_hull_composition_matrix.npz"

    try:
        from huggingface_hub import hf_hub_download
        _CACHE_DIR.mkdir(parents=True, exist_ok=True)
        file_path = hf_hub_download(
            repo_id="LeMaterial/LeMat-Bulk-MLIP-Hull",
            filename=npz_name,
            repo_type="dataset",
            cache_dir=str(_CACHE_DIR),
        )
        return sparse.load_npz(file_path).toarray()
    except Exception as e:
        raise RuntimeError(
            f"Cannot load composition matrix for hull_type='{hull_type}'. "
            f"Error: {e}"
        ) from e


# --------------------------------------------------------------------------- #
# Composition filtering                                                        #
# --------------------------------------------------------------------------- #

def _filter_df_by_composition(
    df: pd.DataFrame,
    matrix: np.ndarray,
    composition: Composition,
) -> pd.DataFrame:
    structure_vector = _one_hot_encode_elements(composition.elements).reshape(-1, 1)
    forbidden = 1 - structure_vector
    mask = (matrix @ forbidden).flatten() == 0
    return df.loc[mask]


# --------------------------------------------------------------------------- #
# Neutral composition helper                                                   #
# --------------------------------------------------------------------------- #

def _neutral_composition(composition: Composition) -> Composition:
    """Strip charges from a composition (e.g. Cs+ → Cs)."""
    neutral: dict[str, float] = {}
    for element, count in composition.as_dict().items():
        if isinstance(element, str):
            base = element.rstrip("+-0123456789")
        elif hasattr(element, "element"):
            base = str(element.element)
        else:
            base = str(element)
        neutral[base] = neutral.get(base, 0) + count
    return Composition(neutral)


# --------------------------------------------------------------------------- #
# Public interface                                                             #
# --------------------------------------------------------------------------- #

def get_energy_above_hull(
    total_energy: float,
    composition: Composition,
    hull_type: str,
    threshold: float = 0.001,
) -> Optional[float]:
    """Calculate energy above hull in eV/atom.

    Parameters
    ----------
    total_energy:
        Total DFT/MLIP energy in eV.
    composition:
        pymatgen Composition of the structure.
    hull_type:
        One of "orb", "uma", "mace_mp".
    threshold:
        Ehull threshold used to select reference structures (default 0.001 eV/atom).

    Returns
    -------
    Energy above hull in eV/atom, or None if calculation fails.
    """
    try:
        ht = HULL_TYPE_MAP.get(hull_type, hull_type)
        df = _load_hull_df(ht, threshold)
        matrix = _load_hull_matrix(ht, threshold)

        subset = _filter_df_by_composition(df, matrix, composition)
        if subset.empty:
            return None

        pd_entries = [
            PDEntry(Composition(Counter(row["species_at_sites"])), row["energy"])
            for _, row in subset.iterrows()
        ]

        phase_diag = PhaseDiagram(pd_entries)
        neutral_comp = _neutral_composition(composition)
        entry = PDEntry(neutral_comp, total_energy)
        e_above_hull = phase_diag.get_decomp_and_e_above_hull(
            entry, allow_negative=True
        )[1]
        return float(e_above_hull)

    except Exception:
        return None
