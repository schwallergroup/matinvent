"""Step 2: filter structures down to physically valid ones.

Three checks, all must pass:
  A. Physical plausibility (density, volume, lattice)
  B. Minimum interatomic distance
  C. Charge neutrality

Parallelised over structures with ProcessPoolExecutor.
"""

from __future__ import annotations

import warnings
from concurrent.futures import ProcessPoolExecutor
from typing import List

import numpy as np
from pymatgen.analysis.bond_valence import BVAnalyzer
from pymatgen.core import Structure
from pymatgen.core.periodic_table import Element

warnings.filterwarnings("ignore")

# --------------------------------------------------------------------------- #
# Individual checks (module-level so they are picklable)                      #
# --------------------------------------------------------------------------- #

_MIN_ATOMIC_DENSITY = 1e-5   # Å⁻³
_MAX_ATOMIC_DENSITY = 0.5    # Å⁻³
_MIN_MASS_DENSITY = 0.01     # g/cm³
_MAX_MASS_DENSITY = 25.0     # g/cm³
_DISTANCE_SCALE = 0.5
_CHARGE_TOL = 0.1


def _check_plausibility(s: Structure) -> bool:
    try:
        vol = s.volume
        if vol <= 1.0:
            return False

        lat = s.lattice
        if not all(1.0 <= a <= 100.0 for a in lat.abc):
            return False
        if not all(0.0 < ang < 180.0 for ang in lat.angles):
            return False

        atomic_density = len(s) / vol
        if not (_MIN_ATOMIC_DENSITY <= atomic_density <= _MAX_ATOMIC_DENSITY):
            return False

        if not (_MIN_MASS_DENSITY <= s.density <= _MAX_MASS_DENSITY):
            return False

        return True
    except Exception:
        return False


def _check_distance(s: Structure) -> bool:
    try:
        dm = s.distance_matrix
        species = s.species
        n = len(species)
        for i in range(n):
            try:
                ri = float(Element(species[i].symbol).atomic_radius or 0.5)
            except Exception:
                ri = 0.5
            for j in range(i + 1, n):
                try:
                    rj = float(Element(species[j].symbol).atomic_radius or 0.5)
                except Exception:
                    rj = 0.5
                min_dist = _DISTANCE_SCALE * (0.7 + ri + rj)
                if dm[i, j] < min_dist:
                    return False
        return True
    except Exception:
        return False


def _check_charge(s: Structure) -> bool:
    try:
        decorated = BVAnalyzer().get_oxi_state_decorated_structure(s)
        total_charge = sum(site.specie.oxi_state for site in decorated)
        return abs(total_charge) <= _CHARGE_TOL
    except Exception:
        pass

    # Fallback: if structure is likely metallic (low electronegativity spread),
    # treat as charge-neutral.
    try:
        elecs = [Element(sp.symbol).X for sp in s.species if Element(sp.symbol).X]
        if elecs and (max(elecs) - min(elecs)) < 1.0:
            return True
    except Exception:
        pass

    return False


def _is_valid(structure: Structure) -> bool:
    """Return True iff structure passes all three validity checks."""
    return _check_plausibility(structure) and _check_distance(structure) and _check_charge(structure)


# Picklable top-level wrapper for ProcessPoolExecutor
def _is_valid_worker(structure: Structure) -> bool:
    return _is_valid(structure)


# --------------------------------------------------------------------------- #
# Public filter                                                                #
# --------------------------------------------------------------------------- #

def filter_valid(
    structures: List[Structure],
    n_jobs: int = 4,
) -> List[int]:
    """Return indices (into *structures*) of the valid structures.

    Parameters
    ----------
    structures:
        Input list of pymatgen Structures.
    n_jobs:
        Number of parallel workers.

    Returns
    -------
    Sorted list of indices that passed all validity checks.
    """
    if not structures:
        return []

    if n_jobs <= 1:
        results = [_is_valid(s) for s in structures]
    else:
        with ProcessPoolExecutor(max_workers=n_jobs) as ex:
            results = list(ex.map(_is_valid_worker, structures))

    return [i for i, ok in enumerate(results) if ok]
