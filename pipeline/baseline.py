import time
import logging
from typing import Dict
from omegaconf import DictConfig

from pipeline.base import ReinL
from pipeline.utils.save import save_structures
from pipeline.utils.logger import Logger
from rewards.reward import Reward
from models.suite.base import ModelSuite


class Baseline(ReinL):
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
        super().__init__(
            rl_epoch=rl_epoch,
            model_suite=model_suite,
            reward=reward,
            sample_cfg=sample_cfg,
            finetune_cfg=finetune_cfg,
            save_dir=save_dir,
            save_freq=save_freq,
            device=device,
            logger=logger,
            replay=replay,
            replay_args=replay_args,
            **kwargs,
        )
        self.load_model()

    def load_model(self):
        self.agent = self.model_suite.load_model()
        for param in self.agent.parameters():
            param.requires_grad = False
        self.agent.to(self.device)

    def sample_step(self):
        sample_data, sample_struc = self.sampler.generate(
            model=self.agent, **self.sample_cfg,
        )

        # Filter bad samples
        # sample_list, _ = invalid_filter(sample_list)
        logging.info(f'Number of filtered samples: {len(sample_struc)}')
        return sample_data, sample_struc

    def ft_step(self):
        pass

    def rl_step(self):
        logging.info(f'*****   LOOP {self.step} START   *****')
        # print(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))

        logging.info('SAMPLE:')
        sample_list, sample_strucs = self.sample_step()
        xyz_path = save_structures(
            structures=sample_strucs,
            save_dir=self.sample_dir,
            filename=f'step_{self.step:0>4d}.extxyz',
        )

        # sample scoring, ranking and get top k samples
        logging.info('SCORE:')
        rewards, prop_dict = self.reward.scoring(
            (sample_strucs, xyz_path),
            f'step_{self.step:0>4d}'
        )
        logging.info(f'reward mean={rewards.mean():.4f} std={rewards.std():.4f}')
        prop_str = [f'{k} mean={v.mean():.4f} std={v.std():.4f}' for k, v in prop_dict.items()]
        logging.info(' | '.join(prop_str))

        log_dict = {f'{k} mean': v.mean() for k, v in prop_dict.items()}
        log_dict.update({f'{k} std': v.std() for k, v in prop_dict.items()})
        log_dict.update({'reward mean': rewards.mean(), 'reward std': rewards.std()})

        # long-term memory
        self.ltm.extend(sample_strucs, rewards, self.step)
        metrics = self.ltm.calc_metrics(self.reward.threshold)
        logging.info(
            f'{len(self.ltm)} crystals generated so far, ' +
            f'{len(self.ltm.unique_comps)} unique components.' +
            f'  Burden: {metrics[0]}, Div. Ratio: {metrics[1]}.'
        )
        log_dict.update(
            {
                'crystal_num': len(self.ltm),
                'unique_comps': len(self.ltm.unique_comps),
                'burden': metrics[0],
                'div_ratio': metrics[1],
            }
        )
        if self.logger is not None:
            self.logger.log(log_dict, step=self.step)

        logging.info(f'*****   LOOP {self.step} FINISH   *****\n\n')

    def run_rl(self):
        logging.info('*****   RL START   *****')
        start_time = time.time()

        for step in range(self.rl_epoch):
            self.step = step
            self.rl_step()

        logging.info('*****   RL END   *****')
        end_time = time.time()
        logging.info('Total time taken: {} s.'.format(int(end_time - start_time)))
