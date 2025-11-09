import os
import logging
from typing import List, Literal, Tuple, Dict
from omegaconf import DictConfig, OmegaConf
from pymatgen.core.structure import Structure
import torch

from pipeline.utils.logger import Logger
from models.suite.base import ModelSuite
from memory.replay_buffer import ReplayBuffer
from memory.ltm import LongTimeMem
from rewards.reward import Reward


def get_device(device: str | None = None):
    if device is None:
        if torch.cuda.is_available():
            device = 'cuda'
        elif torch.backends.mps.is_available():
            device = 'mps'
        else:
            device = 'cpu'
    return torch.device(device)


class ReinL:
    def __init__(
        self,
        rl_epoch: int,
        model_suite: ModelSuite,
        reward: Reward,
        sample_cfg: DictConfig,
        finetune_cfg: DictConfig,
        save_dir: str,
        save_freq: int,
        device: str = None,
        logger: Logger = None,
        replay: bool = False,
        replay_args: Dict = None,
        **kwargs,
    ) -> None:
        self.rl_epoch = rl_epoch
        self.model_suite = model_suite
        self.reward = reward
        self.save_dir = save_dir
        self.save_freq = save_freq
        self.logger = logger
        self.device = get_device(device)
        self.cfg = OmegaConf.create(kwargs)
        self.step = 0
        self.cost = 0

        self.sample_cfg = OmegaConf.merge(
            model_suite.sample_cfg, sample_cfg
        )

        self.finetune_cfg = OmegaConf.merge(
            model_suite.finetune_cfg, finetune_cfg
        )

        self.sampler = model_suite.get_sampler()

        # long-term memory
        self.ltm = LongTimeMem()

        self.models_dir = os.path.join(save_dir, 'models')
        self.sample_dir = os.path.join(save_dir, 'samples')
        # self.reward_dir = os.path.join(save_dir, 'rewards')
        if not os.path.exists(self.models_dir):
            os.makedirs(self.models_dir)
        if not os.path.exists(self.sample_dir):
            os.makedirs(self.sample_dir)
        # if not os.path.exists(self.reward_dir):
        #     os.makedirs(self.reward_dir)

        if replay:
            self.replay = ReplayBuffer(**replay_args)
        else:
            self.replay = None

    def init_optimizer(self, lr=5e-4, optimizer=torch.optim.Adam):
        self.optimizer = optimizer(self.agent.parameters(), lr=lr)

    def init_scheduler(self, start_factor=0.1, total_iters=10):
        self.scheduler = torch.optim.lr_scheduler.LinearLR(
            self.optimizer,
            start_factor=start_factor,
            total_iters=total_iters,
            verbose=False,
        )

    def freeze_model(self, freeze):
        n_freeze = freeze * 4 + 1
        for i, param in enumerate(self.agent.network.parameters()):
            if i < n_freeze:  # Freeze parameter
                param.requires_grad = False

    def reward_step(
        self,
        sample_data: list,
        sample_struc: List[Structure],
        xyz_path: str,
        label: str = 'tmp'
    ):
        rewards, prop_dict, failed_mask = self.reward.scoring(
            (sample_struc, xyz_path), label,
        )
        self.cost += len(sample_struc)

        # remove failed samples/jobs
        success_rewards = rewards[~failed_mask].astype(float)
        success_prop_dict = {
            k: v[~failed_mask] for k, v in prop_dict.items()
        }
        success_data, success_struc = [], []
        for i, failed in enumerate(failed_mask):
            if not failed:
                success_data.append(sample_data[i])
                success_struc.append(sample_struc[i])

        logging.info(f'Evaluation costs to date: {self.cost}')
        logging.info(f'Number of samples that successfully obtained rewards: {len(success_struc)}')
        logging.info(f'reward mean={success_rewards.mean():.4f} std={success_rewards.std():.4f}')
        prop_str = [f'{k} mean={v.mean():.4f} std={v.std():.4f}' for k, v in success_prop_dict.items()]
        logging.info(' | '.join(prop_str))

        return success_data, success_struc, success_rewards, success_prop_dict

    def load_model(self):
        raise NotImplementedError

    def sample_step(self):
        raise NotImplementedError

    def ft_step(self, data_list):
        raise NotImplementedError

    def rl_step(self):
        raise NotImplementedError

    def run_rl(self):
        raise NotImplementedError
