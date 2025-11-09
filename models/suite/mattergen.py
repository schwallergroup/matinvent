import os
from pathlib import Path
from typing import List, Literal
import hydra
from omegaconf import DictConfig, OmegaConf
from numpy.typing import NDArray
import torch
from torch.utils.data import DataLoader

from mattergen.common.utils.eval_utils import MatterGenCheckpointInfo
from mattergen.common.data.chemgraph import ChemGraph
from mattergen.common.data.collate import collate
from mattergen.common.data.datamodule import worker_init_fn

from models.suite.base import ModelSuite
from models.mattergen.pl_module import MatterGenModule
from models.mattergen.sample import MatterGenSampler
from models.mattergen.dataset import MatterGenDataset


AVA_MODEL_NAME = Literal[
    "mattergen_base",
    "mattergen_chemical_system",
    "mattergen_space_group",
    "mattergen_dft_mag_density",
    "mattergen_dft_band_gap",
    "mattergen_ml_bulk_modulus",
    "mattergen_dft_mag_density_hhi_score",
    "mattergen_chemical_system_energy_above_hull",
]


class MatterGenSuite(ModelSuite):
    def __init__(
        self,
        model_name: AVA_MODEL_NAME,
        sample_cfg: DictConfig,
        finetune_cfg: DictConfig,
        model_path: str | None = None,
        config_overrides: list[str] = [],
        device: str | None = None,
        **kwargs,
    ) -> None:
        super().__init__(
            model_name=model_name,
            sample_cfg=sample_cfg,
            finetune_cfg=finetune_cfg,
            model_path=model_path,
            config_overrides=config_overrides,
            device=device,
            **kwargs,
        )

    def load_model(self):
        if self.model_name != "mattergen_base":
            model_name = self.model_name[10:]
        else:
            model_name = self.model_name

        if self.model_path is None:
            ckpt_info = MatterGenCheckpointInfo.from_hf_hub(
                model_name,
                config_overrides=self.config_overrides,
            )
        else:
            ckpt_info = MatterGenCheckpointInfo(
                model_path=Path(self.model_path).resolve(),
                load_epoch='last',
                config_overrides=self.config_overrides,
                strict_checkpoint_loading=True,
            )

        assert ckpt_info.load_epoch is not None
        all_cfg = OmegaConf.create(
            OmegaConf.to_container(ckpt_info.config, resolve=False)
        )  # deepcopy
        try:
            model, incompatible_keys = MatterGenModule.load_from_checkpoint_and_config(
                ckpt_info.checkpoint_path,
                config=ckpt_info.config.lightning_module,
                map_location=self.device,
                strict=ckpt_info.strict_checkpoint_loading,
            )
            model.all_cfg = all_cfg
        except hydra.errors.HydraException as e:
            raise
        if len(incompatible_keys.unexpected_keys) > 0:
            raise ValueError(f"Unexpected keys in checkpoint: {incompatible_keys.unexpected_keys}.")
        if len(incompatible_keys.missing_keys) > 0:
            raise ValueError(f"Missing keys in checkpoint: {incompatible_keys.missing_keys}.")

        return model

    def get_sampler(self):
        sampler = MatterGenSampler(
            batch_size=self.sample_cfg.batch_size,
            num_batches=self.sample_cfg.num_batches,
        )
        return sampler

    def get_dataloader(
        self,
        samples: List[ChemGraph],
        rewards: NDArray | None,
        batch_size: int | None = None,
        shuffle: bool = True,
    ):
        if batch_size is None:
            batch_size = self.finetune_cfg.batch_size
        dataset = MatterGenDataset.from_samples(samples, rewards)
        dataloader = DataLoader(
            dataset,
            shuffle=shuffle,
            batch_size=batch_size,
            worker_init_fn=worker_init_fn,
            collate_fn=collate,
        )
        return dataloader

    def save_model(
        self,
        model: MatterGenModule,
        save_dir: str,
    ):
        os.makedirs(save_dir, exist_ok=True)
        ckpt_dict = {
            "state_dict": model.state_dict(),
            "config": OmegaConf.to_container(model.config, resolve=True)
        }
        torch.save(ckpt_dict, os.path.join(save_dir, "last.ckpt"))
        OmegaConf.save(model.all_cfg, os.path.join(save_dir, "config.yaml"))
