"""MLIP batch geometry relaxation.

Both lattice vectors and atomic coordinates are relaxed (FrechetCellFilter).

Primary path  : TorchSim — FIRE + FrechetCellFilter, batched on GPU.
                Model classes come from torch_sim.models (MaceModel, OrbModel,
                FairChemModel); each requires the corresponding MLIP package to
                provide its native torch-sim integration:
                  mace-torch   ≥ 0.3.14  (mace.calculators.mace_torchsim)
                  orb-models   with orb_models.forcefield.inference.orb_torchsim
                  fairchem-core ≥ 2.10.0 (fairchem.core.calculate.torchsim_interface)
                Falls back to the ASE path if torch_sim or the model class is
                unavailable.
Fallback path : ASE FIRE + FrechetCellFilter, dispatched via ProcessPoolExecutor.

Supported MLIP names: "orb", "uma", "mace_mp".
"""

from __future__ import annotations

import warnings
from concurrent.futures import ProcessPoolExecutor
from typing import List, Optional, Tuple

import numpy as np
from pymatgen.core import Structure
from pymatgen.io.ase import AseAtomsAdaptor

warnings.filterwarnings("ignore")


def _resolve_device(device: Optional[str]) -> str:
    """Return the device string to use: 'cuda' when available, 'cpu' otherwise.

    Passing an explicit string bypasses auto-detection.
    """
    if device is not None:
        return device
    import torch
    return "cuda" if torch.cuda.is_available() else "cpu"


# --------------------------------------------------------------------------- #
# ASE calculator factories                                                     #
# --------------------------------------------------------------------------- #

def _make_mace_calc(device: str = "cpu"):
    from mace.calculators import mace_mp
    return mace_mp(device=device)


def _make_orb_calc(device: str = "cpu"):
    from orb_models.forcefield import pretrained
    from orb_models.forcefield.calculator import ORBCalculator
    model = pretrained.orb_v3_conservative_inf_omat(device=device)
    return ORBCalculator(model, device=device)


def _make_uma_calc(device: str = "cpu"):
    from fairchem.core import FAIRChemCalculator, pretrained_mlip
    predict_unit = pretrained_mlip.get_predict_unit("uma-s-1p1", device=device)
    return FAIRChemCalculator(predict_unit, task_name="omat")


_MLIP_BUILDERS = {
    "mace_mp": _make_mace_calc,
    "orb": _make_orb_calc,
    "uma": _make_uma_calc,
}

# --------------------------------------------------------------------------- #
# TorchSim model factories                                                     #
# FrechetCellFilter requires compute_stress=True for all models.              #
# --------------------------------------------------------------------------- #

def _build_ts_mace(device: str):
    """Build MaceModel for TorchSim (requires mace-torch >= 0.3.14)."""
    import torch
    from mace.calculators.foundations_models import mace_mp as _mace_mp_fn
    from torch_sim.models.mace import MaceModel
    raw = _mace_mp_fn(return_raw_model=True, device=device)
    return MaceModel(
        model=raw,
        device=torch.device(device),
        dtype=torch.float32,
        compute_forces=True,
        compute_stress=True,
    )


def _build_ts_orb(device: str):
    """Build OrbModel for TorchSim (requires orb-models with orb_torchsim integration)."""
    from orb_models.forcefield import pretrained
    from torch_sim.models.orb import OrbModel
    result = pretrained.orb_v3_conservative_inf_omat(device=device, precision="float32-high")
    if isinstance(result, tuple):
        orb_ff, atoms_adapter = result
        return OrbModel(orb_ff, atoms_adapter, device=device)
    # older orb-models API returns the model directly
    return OrbModel(result, device=device)


def _build_ts_uma(device: str):
    """Build FairChemModel for TorchSim (requires fairchem-core >= 2.10.0)."""
    import torch
    from torch_sim.models.fairchem import FairChemModel
    return FairChemModel(
        "uma-s-1p1",
        task_name="omat",
        device=torch.device(device),
        compute_stress=True,
    )


_TS_MODEL_BUILDERS = {
    "mace_mp": _build_ts_mace,
    "orb": _build_ts_orb,
    "uma": _build_ts_uma,
}

# --------------------------------------------------------------------------- #
# TorchSim batch relaxation                                                    #
# --------------------------------------------------------------------------- #

# Module-level model cache so we don't reload on every batch_relax call.
_ts_model_cache: dict = {}


def _torchsim_batch_relax(
    structures: List[Structure],
    mlip_name: str,
    device: str,
    fmax: float,
    max_steps: int,
) -> Optional[List[Tuple[Optional[Structure], Optional[float]]]]:
    """TorchSim FIRE + FrechetCellFilter relaxation.

    Returns None on any failure, signalling the caller to use the ASE fallback.
    """
    try:
        import torch_sim as _ts
        from torch_sim.io import state_to_structures
        from torch_sim.optimizers import Optimizer
        from torch_sim.runners import generate_force_convergence_fn, optimize as _ts_optimize
        CellFilter = _ts.CellFilter
    except ImportError:
        return None

    cache_key = f"{mlip_name}::{device}"
    if cache_key not in _ts_model_cache:
        builder = _TS_MODEL_BUILDERS.get(mlip_name)
        if builder is None:
            return None
        try:
            _ts_model_cache[cache_key] = builder(device)
        except Exception:
            return None

    model = _ts_model_cache[cache_key]

    try:
        final_state = _ts_optimize(
            system=structures,
            model=model,
            optimizer=Optimizer.fire,
            convergence_fn=generate_force_convergence_fn(fmax, include_cell_forces=True),
            max_steps=max_steps,
            autobatcher=False,
            init_kwargs={"cell_filter": CellFilter.frechet},
        )
        relaxed = state_to_structures(final_state)
        energies = final_state.energy.detach().cpu().numpy().tolist()
        return list(zip(relaxed, energies))
    except Exception:
        _ts_model_cache.pop(cache_key, None)
        return None


# --------------------------------------------------------------------------- #
# ASE fallback: FIRE + FrechetCellFilter (module-level for pickling)          #
# --------------------------------------------------------------------------- #

def _ase_relax_one(
    args: Tuple[Structure, str, float, int, str],
) -> Tuple[Optional[Structure], Optional[float]]:
    structure, mlip_name, fmax, max_steps, device = args
    try:
        from ase.filters import FrechetCellFilter
        from ase.optimize import FIRE

        calc = _MLIP_BUILDERS[mlip_name](device)
        adaptor = AseAtomsAdaptor()
        atoms = adaptor.get_atoms(structure)
        atoms.calc = calc

        filtered = FrechetCellFilter(atoms)
        opt = FIRE(filtered, logfile=None)
        opt.run(fmax=fmax, steps=max_steps)

        energy = float(atoms.get_potential_energy())
        relaxed = AseAtomsAdaptor.get_structure(atoms)
        return (relaxed, energy)
    except Exception:
        return (None, None)


# --------------------------------------------------------------------------- #
# Public entry point                                                           #
# --------------------------------------------------------------------------- #

def batch_relax(
    structures: List[Structure],
    mlip_name: str,
    device: Optional[str] = None,
    fmax: float = 0.02,
    max_steps: int = 100,
    batch_size: int = 16,
    n_jobs: int = 4,
) -> List[Tuple[Optional[Structure], Optional[float]]]:
    """Relax a list of structures using FIRE + FrechetCellFilter.

    Both lattice vectors and atomic positions are optimised.

    Parameters
    ----------
    structures:
        Structures to relax.
    mlip_name:
        One of "orb", "uma", "mace_mp".
    device:
        Torch device string ("cpu" or "cuda").
        Defaults to "cuda" if a GPU is available, otherwise "cpu".
    fmax:
        Force convergence criterion in eV/Å (applied to both atomic and cell forces).
    max_steps:
        Maximum number of FIRE steps.
    batch_size:
        Batch size hint (kept for API compatibility; not used when autobatcher=False).
    n_jobs:
        Workers for the ASE fallback path.

    Returns
    -------
    List of (relaxed_structure, total_energy_eV) tuples.
    None entries indicate relaxation failure for that structure.
    """
    if not structures:
        return []

    device = _resolve_device(device)

    if mlip_name not in _MLIP_BUILDERS:
        raise ValueError(
            f"Unknown mlip_name '{mlip_name}'. "
            f"Supported: {list(_MLIP_BUILDERS.keys())}"
        )

    # Try TorchSim first (FIRE + FrechetCellFilter, batched GPU)
    torchsim_result = _torchsim_batch_relax(
        structures, mlip_name, device, fmax, max_steps
    )
    if torchsim_result is not None:
        return torchsim_result

    # Fallback: ASE FIRE + FrechetCellFilter via ProcessPoolExecutor
    args_list = [(s, mlip_name, fmax, max_steps, device) for s in structures]
    if n_jobs <= 1:
        return [_ase_relax_one(a) for a in args_list]

    with ProcessPoolExecutor(max_workers=n_jobs) as ex:
        return list(ex.map(_ase_relax_one, args_list))
