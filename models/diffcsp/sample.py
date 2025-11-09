from dataclasses import dataclass
from typing import Tuple, List
import numpy as np
from pymatgen.core.lattice import Lattice
from pymatgen.core.structure import Structure
import torch
from torch.utils.data import Dataset
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader

from models.diffcsp.diffusion import DiffCSPModule
from pipeline.filters import invalid_filter


ATOM_DIST = {
    'perov_5' : [0, 0, 0, 0, 0, 1],
    'carbon_24' : [0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.3250697750779839,
                0.0,
                0.27795107535708424,
                0.0,
                0.15383352487276308,
                0.0,
                0.11246100804465604,
                0.0,
                0.04958134953209654,
                0.0,
                0.038745690362830404,
                0.0,
                0.019044491873255624,
                0.0,
                0.010178952552946971,
                0.0,
                0.007059596125430964,
                0.0,
                0.006074536200952225],
    'mp_20' : [0.0,
            0.0021742334905660377,
            0.021079009433962265,
            0.019826061320754717,
            0.15271226415094338,
            0.047132959905660375,
            0.08464770047169812,
            0.021079009433962265,
            0.07808814858490566,
            0.03434551886792453,
            0.0972877358490566,
            0.013303360849056603,
            0.09669811320754718,
            0.02155807783018868,
            0.06522700471698113,
            0.014372051886792452,
            0.06703272405660378,
            0.00972877358490566,
            0.053176591981132074,
            0.010576356132075472,
            0.08995430424528301]
}


DEFAULT_STEP_LR = {
    'csp':{
        "perov_5": 5e-7,
        "carbon_24": 5e-6,
        "mp_20": 1e-5,
        "mpts_52": 1e-5
    },
    'csp_multi':{
        "perov_5": 5e-7,
        "carbon_24": 5e-7,
        "mp_20": 1e-5,
        "mpts_52": 1e-5
    },
    'gen':{
        "perov_5": 1e-6,
        "carbon_24": 1e-5,
        "mp_20": 5e-6
    },
}


def data2struc(data):
    frac_coords = data.frac_coords.numpy()
    atom_types = data.atom_types.numpy()
    lengths = data.lengths[0].tolist()
    angles = data.angles[0].tolist()
    lattice = Lattice.from_parameters(*(lengths + angles))
    struc = Structure(
        lattice=lattice,
        species=atom_types,
        coords=frac_coords,
        coords_are_cartesian=False,
    )

    return struc


def lattices_to_params_shape(lattices):

    lengths = torch.sqrt(torch.sum(lattices ** 2, dim=-1))
    angles = torch.zeros_like(lengths)
    for i in range(3):
        j = (i + 1) % 3
        k = (i + 2) % 3
        angles[...,i] = torch.clamp(torch.sum(lattices[...,j,:] * lattices[...,k,:], dim = -1) /
                            (lengths[...,j] * lengths[...,k]), -1., 1.)
    angles = torch.arccos(angles) * 180.0 / np.pi

    return lengths, angles


class SampleDataset(Dataset):

    def __init__(self, total_num, dataset='mp_20'):
        super().__init__()
        self.total_num = total_num
        self.distribution = ATOM_DIST[dataset]
        self.num_atoms = np.random.choice(len(self.distribution), total_num, p = self.distribution)
        self.is_carbon = dataset == 'carbon_24'

    def __len__(self) -> int:
        return self.total_num

    def __getitem__(self, index):

        num_atom = self.num_atoms[index]
        data = Data(
            num_atoms=torch.LongTensor([num_atom]),
            num_nodes=num_atom,
        )
        if self.is_carbon:
            data.atom_types = torch.LongTensor([6] * num_atom)
        return data


@dataclass
class DiffCSPSampler:
    batch_size: int | None = None
    num_batches: int | None = None
    target_compositions_dict: list[dict[str, float]] | None = None
    num_atoms_distribution: str = "mp_20"

    def generate(
        self,
        model: DiffCSPModule,
        batch_size: int | None = None,
        num_batches: int | None = None,
        **kwargs,
    ) -> Tuple[List[Data], List[Structure]]:
        # Prioritize the runtime provided batch_size, num_batches
        batch_size = batch_size or self.batch_size
        num_batches = num_batches or self.num_batches
        assert batch_size is not None
        assert num_batches is not None

        model.eval()
        dataset = SampleDataset(total_num=batch_size * num_batches)
        loader = DataLoader(dataset, batch_size=batch_size)
        step_lr = DEFAULT_STEP_LR['gen']['mp_20']

        for batch in loader:
            batch = batch.to(model.device)
            outputs, traj = model.sample(batch, step_lr=step_lr)
            # frac_coords.append(outputs['frac_coords'].detach().cpu())
            # num_atoms.append(outputs['num_atoms'].detach().cpu())
            # atom_types.append(outputs['atom_types'].detach().cpu())
            # lattices.append(outputs['lattices'].detach().cpu())

        frac_coords = outputs['frac_coords'].detach().cpu()
        num_atoms = outputs['num_atoms'].detach().cpu()
        atom_types = outputs['atom_types'].detach().cpu()
        lattices = outputs['lattices'].detach().cpu()
        lengths, angles = lattices_to_params_shape(lattices)

        data_list = []
        struc_list = []
        atom_types = torch.argmax(atom_types, dim=-1) + 1
        offset = torch.cumsum(num_atoms, dim=0).tolist()
        offset = torch.LongTensor([0] + offset)
        for i in range(len(num_atoms)):
            _atom_types = atom_types[offset[i]: offset[i + 1]]
            _angles = angles[i].view(1, -1)
            _lengths = lengths[i].view(1, -1)
            _frac_coords = frac_coords[offset[i]: offset[i + 1]]
            data = Data(
                frac_coords=_frac_coords,
                atom_types=_atom_types,
                lengths=_lengths,
                angles=_angles,
                num_atoms=num_atoms[i],
                num_nodes=num_atoms[i],
            )
            data_list.append(data)
            struc_list.append(data2struc(data))

        return data_list, struc_list


def sample_loop(sample_size, model, device, step_lr=-1):

    # model = model.to(args.device)
    # device = model.dummy_param.device
    model.eval()
    dataset = SampleDataset(total_num=sample_size)
    loader = DataLoader(dataset, batch_size=sample_size)
    step_lr = step_lr if step_lr >= 0 else DEFAULT_STEP_LR['gen']['mp_20']

    for batch in loader:
        batch = batch.to(device)
        outputs, traj = model.sample(batch, step_lr=step_lr)
        # frac_coords.append(outputs['frac_coords'].detach().cpu())
        # num_atoms.append(outputs['num_atoms'].detach().cpu())
        # atom_types.append(outputs['atom_types'].detach().cpu())
        # lattices.append(outputs['lattices'].detach().cpu())

    frac_coords = outputs['frac_coords'].detach().cpu()
    num_atoms = outputs['num_atoms'].detach().cpu()
    atom_types = outputs['atom_types'].detach().cpu()
    lattices = outputs['lattices'].detach().cpu()
    lengths, angles = lattices_to_params_shape(lattices)

    data_list = []
    atom_types = torch.argmax(atom_types, dim=-1) + 1
    offset = torch.cumsum(num_atoms, dim=0).tolist()
    offset = torch.LongTensor([0] + offset)
    for i in range(len(num_atoms)):
        _atom_types = atom_types[offset[i]: offset[i + 1]]
        _angles = angles[i].view(1, -1)
        _lengths = lengths[i].view(1, -1)
        _frac_coords = frac_coords[offset[i]: offset[i + 1]]
        data = Data(
            frac_coords=_frac_coords,
            atom_types=_atom_types,
            lengths=_lengths,
            angles=_angles,
            num_atoms=num_atoms[i],
            num_nodes=num_atoms[i],
        )
        data_list.append(data)

    return data_list


def sample_mdp(sample_size, model, device, step_lr=-1):

    # model = model.to(args.device)
    # device = model.dummy_param.device
    model.eval()
    dataset = SampleDataset(total_num=sample_size)
    loader = DataLoader(dataset, batch_size=sample_size)
    step_lr = step_lr if step_lr >= 0 else DEFAULT_STEP_LR['gen']['mp_20']

    for batch in loader:
        batch = batch.to(device)
        outputs, traj = model.sample(batch, step_lr=step_lr)
        # frac_coords.append(outputs['frac_coords'].detach().cpu())
        # num_atoms.append(outputs['num_atoms'].detach().cpu())
        # atom_types.append(outputs['atom_types'].detach().cpu())
        # lattices.append(outputs['lattices'].detach().cpu())

    frac_coords = outputs['frac_coords'].detach().cpu()
    num_atoms = outputs['num_atoms'].detach().cpu()
    atom_types = outputs['atom_types'].detach().cpu()
    lattices = outputs['lattices'].detach().cpu()
    lengths, angles = lattices_to_params_shape(lattices)

    data_list = []
    atom_types = torch.argmax(atom_types, dim=-1) + 1
    offset = torch.cumsum(num_atoms, dim=0).tolist()
    offset = torch.LongTensor([0] + offset)
    for i in range(len(num_atoms)):
        _atom_types = atom_types[offset[i]: offset[i + 1]]
        _angles = angles[i].view(1, -1)
        _lengths = lengths[i].view(1, -1)
        _frac_coords = frac_coords[offset[i]: offset[i + 1]]
        data = Data(
            frac_coords=_frac_coords,
            atom_types=_atom_types,
            lengths=_lengths,
            angles=_angles,
            num_atoms=num_atoms[i],
            num_nodes=num_atoms[i],
        )
        data_list.append(data)

    sample_list, valid_bool = invalid_filter(data_list)
    valid_idx = np.where(valid_bool)[0].tolist()
    sample_traj = []
    atom_bool = [idx in valid_idx for idx in traj[0]['batch_idx']]
    for i in range(model.beta_scheduler.timesteps, 1, -1):
        _traj = {
            'atom_types': traj[i]['atom_types'][atom_bool].detach().cpu(),
            'lattices': traj[i]['lattices'][valid_idx].detach().cpu(),
            'frac_coords': traj[i]['frac_coords'][atom_bool].detach().cpu(),
            'frac_coords_mid': traj[i]['frac_coords_mid'][atom_bool].detach().cpu(),
            'num_atoms': traj[i]['num_atoms'][valid_idx].detach().cpu(),
            'timesteps': torch.tensor([i] * len(valid_idx), dtype=torch.long),
            'log_prob_t': traj[i]['log_prob_t'][valid_idx].detach().cpu(),
            'log_prob_x': traj[i]['log_prob_x'][valid_idx].detach().cpu(),
            'log_prob_l': traj[i]['log_prob_l'][valid_idx].detach().cpu(),
        }
        sample_traj.append(_traj)

    return sample_list, sample_traj
