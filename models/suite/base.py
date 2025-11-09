from typing import Literal
from omegaconf import DictConfig, OmegaConf
import torch


AVA_MODEL_NAME = Literal[
    "diffcsp",
    "mattergen_base",
    "mattergen_chemical_system",
    "mattergen_space_group",
    "mattergen_dft_mag_density",
    "mattergen_dft_band_gap",
    "mattergen_ml_bulk_modulus",
    "mattergen_dft_mag_density_hhi_score",
    "mattergen_chemical_system_energy_above_hull",
]


def get_device(device: str | None = None):
    if device is None:
        if torch.cuda.is_available():
            device = 'cuda'
        elif torch.backends.mps.is_available():
            device = 'mps'
        else:
            device = 'cpu'
    return torch.device(device)


class ModelSuite:
    def __init__(
        self,
        model_name: str,
        sample_cfg: DictConfig,
        finetune_cfg: DictConfig,
        model_path: str | None = None,
        config_overrides: list[str] = [],
        device: str | None = None,
        **kwargs,
    ) -> None:
        self.model_name = model_name
        self.sample_cfg = sample_cfg
        self.finetune_cfg = finetune_cfg
        self.model_path = model_path
        self.config_overrides = config_overrides
        self.device = get_device(device)
        self.cfg = OmegaConf.create(kwargs)

    def load_model(self):
        raise NotImplementedError

    def get_sampler(self):
        raise NotImplementedError

    def get_dataloader(self):
        raise NotImplementedError

    def save_model(self):
        raise NotImplementedError
