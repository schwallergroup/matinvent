import os
import time
import logging
from typing import Dict
import numpy as np
import torch
from omegaconf import DictConfig

from pipeline.base import ReinL
from pipeline.filters import OptEval, invalid_filter
from pipeline.utils.save import save_structures
from pipeline.utils.logger import Logger
from rewards.reward import Reward
from models.suite.base import ModelSuite


class MatInvent(ReinL):
    def __init__(
        self,
        rl_epoch: int,
        model_suite: ModelSuite,
        reward: Reward,
        sample_cfg: DictConfig,
        finetune_cfg: DictConfig,
        topk_ratio: float,
        save_dir: str,
        save_freq: int = 50,
        device: str = None,
        logger: Logger = None,
        replay: bool = False,
        replay_args: Dict = None,
        div_filter: bool = False,
        df_args: Dict = None,
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
        assert topk_ratio > 0.0 and topk_ratio <= 1.0
        self.topk_ratio = topk_ratio

        # diversity filter
        self.div_filter = div_filter
        self.df_args = df_args

        if 'filter' not in self.sample_cfg:
            self.opt_eval = OptEval()

        self.load_model()

    def load_model(self):
        self.agent = self.model_suite.load_model()
        self.prior = self.model_suite.load_model()

        for param in self.agent.parameters():
            param.requires_grad = True
        # Freeze the parameter of prior (pretrained) model
        for param in self.prior.parameters():
            param.requires_grad = False
        self.agent.to(self.device)
        self.prior.to(self.device)

    def sample_step(self):
        sample_data, sample_struc = self.sampler.generate(
            model=self.agent, **self.sample_cfg,
        )
        # Filter invalid samples
        sample_data, sample_struc = invalid_filter(sample_data, sample_struc)

        # save all generated valid structures
        valid_xyz_path = save_structures(
            structures=sample_struc,
            save_dir=self.sample_dir,
            filename=f'step_{self.step:0>4d}_valid.extxyz',
        )

        # MLIP relaxation
        if self.sample_cfg.get('mlip_opt'):
            mlip_opt = self.sample_cfg.mlip_opt
            sample_struc, energies = mlip_opt(sample_struc, valid_xyz_path)
        else:
            energies = None

        # Filter bad samples by selected metrics
        if self.sample_cfg.get('filter'):
            filter = self.sample_cfg.filter
            sample_data, sample_struc, metrics = filter(
                sample_data, sample_struc, energies,
            )
            logging.info(f'Number of filtered samples: {len(sample_struc)}')
        else:
            # metrics, _ = self.opt_eval(sample_struc, energies)
            metrics = {}

        log_str = [f'{k}: {v:.6f}' for k, v in metrics.items()]
        logging.info(', '.join(log_str))

        # max sample size to score/reward
        if self.sample_cfg.get('max_num'):
            max_num = self.sample_cfg.max_num
            if len(sample_struc) > max_num:
                sample_data = sample_data[:max_num]
                sample_struc = sample_struc[:max_num]

        # save structures for evaluation
        eval_xyz_path = save_structures(
            structures=sample_struc,
            save_dir=self.sample_dir,
            filename=f'step_{self.step:0>4d}_eval.extxyz',
        )

        return sample_data, sample_struc, eval_xyz_path, metrics

    def ft_step(self, data_list, rewards, baseline):
        # Tensor Core acceleration for new GPUs (Ampere, Hopper, etc)
        torch.set_float32_matmul_precision("high")
        cfg = self.finetune_cfg
        loader = self.model_suite.get_dataloader(
            samples=data_list,
            rewards=rewards,
            batch_size=len(data_list),
        )

        # model = model.to(args.device)
        optimizer = torch.optim.Adam(self.agent.parameters(), lr=cfg.lr)
        # rewards = torch.tensor(rewards, dtype=torch.float, device=self.device)
        accum_steps = cfg.accum_steps  # accumulation_steps

        for epoch in range(cfg.epochs):
            # logging.info(f"Epoch {epoch} starts:")
            self.agent.train()

            loss_all, loss_diff_all, loss_kl_all = 0., 0., 0.
            for batch in loader:
                batch = batch.to(self.device)
                optimizer.zero_grad()
                loss, loss_diff, loss_kl = 0., 0., 0.

                for t in range(cfg.timesteps):

                    noised_input = self.agent.add_noise(batch, t)
                    sample_loss, agent_pred = self.agent.calc_sample_loss(noised_input)
                    _, prior_pred = self.prior.calc_sample_loss(noised_input)
                    adv = batch.reward
                    # adv = (batch.score - batch.score.mean()) / batch.score.std()
                    # adv = (batch.reward - baseline) / (batch.reward.max() - baseline)
                    _loss_diff = adv * sample_loss

                    kl_term = self.agent.calc_kl_reg(agent_pred, prior_pred, batch)
                    _loss_kl = kl_term * (1.1 - batch.reward)

                    _loss = (_loss_diff + _loss_kl * cfg.sigma).mean() / accum_steps
                    _loss.backward()
                    if (t + 1) % accum_steps == 0:
                        optimizer.step()
                        optimizer.zero_grad()
                    loss += _loss.item() * accum_steps
                    loss_diff += _loss_diff.sum().item()
                    loss_kl += _loss_kl.sum().item()

                loss_diff = loss_diff / cfg.timesteps
                loss_kl = loss_kl / cfg.timesteps
                loss = loss / cfg.timesteps

                if (t + 1) % accum_steps != 0:
                    optimizer.step()

                loss_all += loss * batch.num_graphs
                loss_diff_all += loss_diff
                loss_kl_all += loss_kl

            loss_dict = {
                'loss': loss_all / len(data_list),
                'loss_diff': loss_diff_all / len(data_list),
                'loss_kl': loss_kl_all / len(data_list),
            }
            log_str = [f'{k}: {v:.4f}' for k, v in loss_dict.items()]
            logging.info(f'Epoch {epoch}: ' + ', '.join(log_str))

    def rl_step(self):
        logging.info(f'*****   LOOP {self.step} START   *****')
        start_time = time.time()

        logging.info('SAMPLE:')
        sample_list, sample_struc, xyz_path, sample_metrics = self.sample_step()

        # sample scoring, remove failed samples, ranking and get top k samples
        logging.info('SCORE:')
        sample_list, sample_struc, rewards, prop_dict = self.reward_step(
            sample_list, sample_struc, xyz_path, f'step_{self.step:0>4d}',
        )

        log_dict = {f'{k} mean': v.mean() for k, v in prop_dict.items()}
        log_dict.update({f'{k} std': v.std() for k, v in prop_dict.items()})
        log_dict.update({'reward mean': rewards.mean(), 'reward std': rewards.std()})
        log_dict.update(sample_metrics)

        # long-term memory
        self.ltm.extend(sample_struc, rewards, self.step)
        metrics = self.ltm.calc_metrics(self.reward.threshold)
        self.ltm.save(os.path.join(self.sample_dir, 'long_term_memory.csv'))
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
                'cost': self.cost,
            }
        )
        if self.logger is not None:
            self.logger.log(log_dict, step=self.step)

        # diversity filter
        if self.div_filter:
            rewards, penalty_idx, tol_n, buff_n = self.ltm.div_filter(
                sample_struc, rewards, **self.df_args
            )
            penalty_sample = [sample_list[p] for p in penalty_idx]
            penalty_strucs = [sample_struc[p] for p in penalty_idx]
            logging.info(f'Diversity filter: tol_n={tol_n}, buff_n={buff_n}')

        # topk data points
        sort_idx = np.argsort(rewards)[::-1]
        topk_idx = sort_idx[: int(self.finetune_cfg.batch_size * self.topk_ratio)]
        sample_topk = [sample_list[_i] for _i in topk_idx]
        strucs_topk = [sample_struc[_i] for _i in topk_idx]
        reward_topk = rewards[topk_idx]

        # experience replay
        if self.replay is not None:
            if self.div_filter and len(penalty_strucs) > 0:
                self.replay.memory_purge(penalty_strucs)
            data_replay, reward_replay = self.replay.sample()
            ft_data = sample_topk + data_replay
            ft_reward = np.concatenate((reward_topk, reward_replay))
            self.replay.extend(sample_topk, strucs_topk, reward_topk)
            logging.info(f'replay buffer size={len(self.replay)}')
            # print(f'replay rewards={reward_replay}')
            logging.info(f'buffer reward mean={self.replay.buffer["reward"].values.mean()}')
            # print(f'buffer rewards={replay.buffer["reward"].values}')
        else:
            ft_data = sample_topk
            ft_reward = reward_topk

        # finetuning
        logging.info('FINETUNE:')
        baseline = self.ltm.get_baseline(self.step)
        baseline = min(baseline, ft_reward.min())
        self.ft_step(ft_data, ft_reward, baseline)

        end_time = time.time()
        total_time = (end_time - start_time) / 60
        logging.info(f'*****   LOOP {self.step} FINISH   *****')
        logging.info(f'Total time taken: {total_time:.2f} min.\n\n')

    def run_rl(self):
        logging.info('*****   RL START   *****')
        start_time = time.time()

        for step in range(self.rl_epoch):
            self.step = step
            self.rl_step()
            # Save the agent weights every few iterations
            if (step + 1) % self.save_freq == 0:
                ckpt_dir = os.path.join(self.models_dir, f'loop_{step:0>4d}')
                self.model_suite.save_model(self.agent, ckpt_dir)
        # If the entire training finishes, clean up
        ckpt_dir = os.path.join(self.models_dir, 'final')
        self.model_suite.save_model(self.agent, ckpt_dir)

        logging.info('*****   RL END   *****')
        end_time = time.time()
        logging.info('Total time taken: {} s.'.format(int(end_time - start_time)))
