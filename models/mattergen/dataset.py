from dataclasses import dataclass
import numpy as np
from numpy.typing import NDArray

from mattergen.common.data.collate import collate
from mattergen.common.data.dataset import CrystalDataset
from mattergen.common.data.chemgraph import ChemGraph
from mattergen.common.data.transform import (
    Transform,
    symmetrize_lattice,
    set_chemical_system_string,
)

TF_LIST = [
    symmetrize_lattice, set_chemical_system_string,
]


@dataclass(frozen=True, kw_only=True)
class MatterGenDataset(CrystalDataset):
    """
    Dataset for crystal structures. Takes as input numpy arrays for positions, cell, atomic numbers,
    number of atoms and structure id. Optionally, properties can be added as well, as a dictionary
    of numpy arrays. The dataset can also be transformed using a list of transforms.
    The recommended way of creating a CrystalDataset is to use the class method
    CrystalDataset.from_preset with a preset name, which will use the CrystalDatasetBuilder class to
    fetch the dataset from cache if it exists, and otherwise cache it.
    """

    pos: NDArray
    cell: NDArray
    atomic_numbers: NDArray
    num_atoms: NDArray
    structure_id: NDArray
    properties: dict[NDArray]
    transforms: list[Transform] | None = None

    def __post_init__(self):
        pass

    @classmethod
    def from_samples(
        cls,
        samples: ChemGraph | list[ChemGraph],
        rewards: NDArray | None = None,
        transforms: list[Transform] | None = TF_LIST,
    ):
        if rewards is None:
            properties = {}
        else:
            properties = {'reward': rewards}
        if isinstance(samples, list):
            samples = collate(samples)

        structure_id = np.arange(len(samples.num_atoms), dtype=int)
        dataset = cls(
            pos=samples.pos.cpu().numpy(),
            cell=samples.cell.cpu().numpy(),
            atomic_numbers=samples.atomic_numbers.cpu().numpy(),
            num_atoms=samples.num_atoms.cpu().numpy(),
            structure_id=structure_id,
            properties=properties,
            transforms=transforms,
        )
        return dataset
