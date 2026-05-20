import numpy as np
from ase import Atoms
from mattersim.applications.batch_relax import BatchRelaxer
from mattersim.forcefield.potential import Potential
from pymatgen.core import Structure
from pymatgen.io.ase import AseAtomsAdaptor


def _get_device(device: str | None = None) -> str:
    import torch
    if device is None:
        if torch.cuda.is_available():
            return 'cuda'
        elif torch.backends.mps.is_available():
            return 'mps'
        return 'cpu'
    return device


def relax_structures(
    structures: Structure | list[Structure],
    device: str = None,
    potential_load_path: str = None,
    output_path: str | None = None,
    **kwargs,
) -> tuple[list[Structure], np.ndarray]:
    if isinstance(structures, Structure):
        structures = [structures]
    device = _get_device(device)
    atoms = [AseAtomsAdaptor.get_atoms(s) for s in structures]
    potential = Potential.from_checkpoint(
        device=device, load_path=potential_load_path, load_training_state=False,
    )
    batch_relaxer = BatchRelaxer(potential=potential, filter="EXPCELLFILTER", **kwargs)
    trajectories = batch_relaxer.relax(atoms)
    relaxed_atoms = [t[-1] for t in trajectories.values()]
    total_energies = np.array([a.info["total_energy"] for a in relaxed_atoms])
    if output_path:
        from ase.io import write
        write(output_path, relaxed_atoms, format="extxyz")
    relaxed_structures = [AseAtomsAdaptor.get_structure(a) for a in relaxed_atoms]
    return relaxed_structures, total_energies
