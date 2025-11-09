import math
from typing import Any
import numpy as np
import hydra
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal
from torch_scatter import scatter
import pytorch_lightning as pl

from models.diffcsp.utils import lattice_params_to_matrix_torch
from models.diffcsp.scheduler import d_log_p_wrapped_normal

MAX_ATOMIC_NUM=100


def p_wrapped_normal(x, sigma, N=10, T=1.0):
    p_ = 0
    for i in range(-N, N + 1):
        p_ += torch.exp(-(x + T * i) ** 2 / 2 / sigma ** 2)
    return p_


def log_prob_wn(x, mu, sigma, N=10, T=1.0):
    p_ = 0
    for i in range(-N, N + 1):
        p_ += torch.exp(-(x - mu + T * i) ** 2 / 2 / sigma ** 2)
    return torch.log(p_)


class BaseModule(pl.LightningModule):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__()
        # populate self.hparams with args and kwargs automagically!
        self.save_hyperparameters()
        if hasattr(self.hparams, "model"):
            self._hparams = self.hparams.model


    def configure_optimizers(self):
        opt = hydra.utils.instantiate(
            self.hparams.optim.optimizer, params=self.parameters(), _convert_="partial"
        )
        if not self.hparams.optim.use_lr_scheduler:
            return [opt]
        scheduler = hydra.utils.instantiate(
            self.hparams.optim.lr_scheduler, optimizer=opt
        )
        return {"optimizer": opt, "lr_scheduler": scheduler, "monitor": "val_loss"}


class SinusoidalTimeEmbeddings(nn.Module):
    """ Attention is all you need. """
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, time):
        device = time.device
        half_dim = self.dim // 2
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = time[:, None] * embeddings[None, :]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        return embeddings


class DiffCSPModule(BaseModule):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        
        self.decoder = hydra.utils.instantiate(self.hparams.decoder, latent_dim = self.hparams.latent_dim + self.hparams.time_dim, pred_type = True, smooth = True)
        self.beta_scheduler = hydra.utils.instantiate(self.hparams.beta_scheduler)
        self.sigma_scheduler = hydra.utils.instantiate(self.hparams.sigma_scheduler)
        self.time_dim = self.hparams.time_dim
        self.time_embedding = SinusoidalTimeEmbeddings(self.time_dim)
        self.keep_lattice = self.hparams.cost_lattice < 1e-5
        self.keep_coords = self.hparams.cost_coord < 1e-5

    def add_noise(self, batch, time=None):
        batch_size = batch.num_graphs
        if time is None:
            times = self.beta_scheduler.uniform_sample_t(batch_size, self.device)
        else:
            time_arr = np.arange(self.beta_scheduler.timesteps, 0, -1)
            times = torch.full((batch_size, ), time_arr[time], device=self.device)
        time_emb = self.time_embedding(times)

        alphas_cumprod = self.beta_scheduler.alphas_cumprod[times]
        beta = self.beta_scheduler.betas[times]

        c0 = torch.sqrt(alphas_cumprod)
        c1 = torch.sqrt(1. - alphas_cumprod)

        sigmas = self.sigma_scheduler.sigmas[times]
        sigmas_norm = self.sigma_scheduler.sigmas_norm[times]

        lattices = lattice_params_to_matrix_torch(batch.lengths, batch.angles)
        frac_coords = batch.frac_coords

        rand_l, rand_x = torch.randn_like(lattices), torch.randn_like(frac_coords)

        input_lattice = c0[:, None, None] * lattices + c1[:, None, None] * rand_l
        sigmas_per_atom = sigmas.repeat_interleave(batch.num_atoms)[:, None]
        sigmas_norm_per_atom = sigmas_norm.repeat_interleave(batch.num_atoms)[:, None]
        input_frac_coords = (frac_coords + sigmas_per_atom * rand_x) % 1.

        gt_atom_types_onehot = F.one_hot(batch.atom_types - 1, num_classes=MAX_ATOMIC_NUM).float()

        rand_t = torch.randn_like(gt_atom_types_onehot)

        atom_type_probs = (c0.repeat_interleave(batch.num_atoms)[:, None] * gt_atom_types_onehot + c1.repeat_interleave(batch.num_atoms)[:, None] * rand_t)
        tar_x = d_log_p_wrapped_normal(sigmas_per_atom * rand_x, sigmas_per_atom) / torch.sqrt(sigmas_norm_per_atom)

        noised_input = (time_emb, atom_type_probs, input_frac_coords, input_lattice, batch.num_atoms, batch.batch)
        noises = (rand_l, tar_x, rand_t)

        return noised_input, noises, batch.batch

    def calc_sample_loss(self, input_all):
        noised_input, noises, batch = input_all
        rand_l, tar_x, rand_t = noises
        pred_l, pred_x, pred_t = self.decoder(*noised_input)

        loss_lattice = torch.pow((pred_l - rand_l), 2).mean(dim=(1,2))
        _loss_coord = torch.pow((pred_x - tar_x), 2).mean(dim=1)
        loss_coord = scatter(_loss_coord, batch, dim=0, reduce='mean')
        _loss_type = torch.pow((pred_t - rand_t), 2).mean(dim=1)
        loss_type = scatter(_loss_type, batch, dim=0, reduce='mean')

        loss = (
            self.hparams.cost_lattice * loss_lattice +
            self.hparams.cost_coord * loss_coord + 
            self.hparams.cost_type * loss_type
        )

        return loss, (pred_l, pred_x, pred_t)

    def calc_kl_reg(self, agent_pred, prior_pred, batch):
        pred_l, pred_x, pred_t = agent_pred
        pred_l_p, pred_x_p, pred_t_p = prior_pred
        kl_term0 = torch.pow((pred_l - pred_l_p.detach()), 2).mean(dim=(1,2))
        x_ap = torch.pow((pred_x - pred_x_p.detach()), 2).mean(dim=1)
        kl_term1 = scatter(x_ap, batch.batch, dim=0, reduce='mean')
        t_ap = torch.pow((pred_t - pred_t_p.detach()), 2).mean(dim=1)
        kl_term2 = scatter(t_ap, batch.batch, dim=0, reduce='mean')
        kl_term = kl_term0 + kl_term1 + kl_term2
        return kl_term

    def forward(self, noised_input):

        # time_emb, atom_type_probs, input_frac_coords, input_lattice, num_atoms, batch = noised_input
        pred_l, pred_x, pred_t = self.decoder(*noised_input)

        return pred_l, pred_x, pred_t

    def forward_logprb(self, state, step_lr=1e-5):
        for k, v in state.items():
            state[k] = v.to(self.device)
        t = state['timesteps'][0].item()
        time_emb = self.time_embedding(state['timesteps'])

        alphas = self.beta_scheduler.alphas[t]
        alphas_cumprod = self.beta_scheduler.alphas_cumprod[t]
        c0 = 1.0 / torch.sqrt(alphas)
        c1 = (1 - alphas) / torch.sqrt(1 - alphas_cumprod)

        sigmas = self.beta_scheduler.sigmas[t]
        sigma_x = self.sigma_scheduler.sigmas[t]
        sigma_norm = self.sigma_scheduler.sigmas_norm[t]

        batch = torch.arange(len(state['num_atoms']), device=self.device)
        batch = batch.repeat_interleave(state['num_atoms'])

        # Corrector
        step_size = step_lr * (sigma_x / self.sigma_scheduler.sigma_begin) ** 2
        std_x = torch.sqrt(2 * step_size)
        pred_l_corr, pred_x_corr, pred_t_corr = self.decoder(
            time_emb,
            state['atom_types'],
            state['frac_coords'],
            state['lattices'],
            state['num_atoms'],
            batch,
        )
        pred_x_s = pred_x_corr * torch.sqrt(sigma_norm)
        x_mu_corr = (state['frac_coords'] - step_size * pred_x_s) % 1.
        x_sigma_corr = std_x
        log_prob_x_corr = log_prob_wn(
            state['frac_coords_mid'],
            x_mu_corr,
            x_sigma_corr,
        ).mean(dim=-1)
        log_prob_x_corr = scatter(log_prob_x_corr, batch, reduce='mean')

        # Predictor
        adjacent_sigma_x = self.sigma_scheduler.sigmas[t-1]
        step_size = (sigma_x ** 2 - adjacent_sigma_x ** 2)
        std_x = torch.sqrt((adjacent_sigma_x ** 2 * (sigma_x ** 2 - adjacent_sigma_x ** 2)) / (sigma_x ** 2))
        pred_l_pred, pred_x_pred, pred_t_pred = self.decoder(
            time_emb,
            state['atom_types'],
            state['frac_coords_mid'],
            state['lattices'],
            state['num_atoms'],
            batch,
        )
        pred_x_s = pred_x_pred * torch.sqrt(sigma_norm)
        x_mu_pred = (state['frac_coords_mid'] - step_size * pred_x_s) % 1.
        x_sigma_pred = std_x
        log_prob_x_pred = log_prob_wn(
            state['next_frac_coords'],
            x_mu_pred,
            x_sigma_pred,
        ).mean(dim=-1)
        log_prob_x_pred = scatter(log_prob_x_pred, batch, reduce='mean')

        log_prob_x = log_prob_x_corr + log_prob_x_pred

        dist_l = Normal(c0 * (state['lattices'] - c1 * pred_l_pred), sigmas)
        dist_t = Normal(c0 * (state['atom_types'] - c1 * pred_t_pred), sigmas)
        log_prob_l = dist_l.log_prob(state['next_lattices']).mean(dim=-1).mean(dim=-1)
        log_prob_t = dist_t.log_prob(state['next_atom_types']).mean(dim=-1)
        log_prob_t = scatter(log_prob_t, batch, reduce='mean')

        return log_prob_l, log_prob_t, log_prob_x, (pred_l_corr, pred_x_corr, pred_t_corr)

    def forward_logprb_old(self, state, step_lr=1e-5):
        for k, v in state.items():
            state[k] = v.to(self.device)
        t = state['timesteps'][0].item()
        time_emb = self.time_embedding(state['timesteps'])

        alphas = self.beta_scheduler.alphas[t]
        alphas_cumprod = self.beta_scheduler.alphas_cumprod[t]
        c0 = 1.0 / torch.sqrt(alphas)
        c1 = (1 - alphas) / torch.sqrt(1 - alphas_cumprod)

        sigmas = self.beta_scheduler.sigmas[t]
        sigma_x = self.sigma_scheduler.sigmas[t]
        sigma_norm = self.sigma_scheduler.sigmas_norm[t]
        step_size = step_lr * (sigma_x / self.sigma_scheduler.sigma_begin) ** 2
        std_x = torch.sqrt(2 * step_size)

        batch = torch.arange(len(state['num_atoms']), device=self.device)
        batch = batch.repeat_interleave(state['num_atoms'])
        pred_l, pred_x, pred_t = self.decoder(
            time_emb,
            state['atom_types'],
            state['frac_coords'],
            state['lattices'],
            state['num_atoms'],
            batch,
        )
        pred_x_s = pred_x * torch.sqrt(sigma_norm)

        dist_l = Normal(c0 * (state['lattices'] - c1 * pred_l), sigmas)
        dist_t = Normal(c0 * (state['atom_types'] - c1 * pred_t), sigmas)

        log_prob_l = dist_l.log_prob(state['next_lattices']).mean(dim=-1).mean(dim=-1)
        log_prob_t = dist_t.log_prob(state['next_atom_types']).mean(dim=-1)
        log_prob_t = scatter(log_prob_t, batch, reduce='mean')
        log_prob_x = log_prob_wn(
            state['next_frac_coords'],
            (state['frac_coords'] - step_size * pred_x_s) % 1.,
            std_x,
        ).mean(dim=-1)
        log_prob_x = scatter(log_prob_x, batch, reduce='mean')

        return log_prob_l, log_prob_t, log_prob_x, (pred_l, pred_x, pred_t)

    @torch.no_grad()
    def sample(self, batch, diff_ratio=1.0, step_lr=1e-5):

        batch_size = batch.num_graphs
        x_T = torch.rand([batch.num_nodes, 3]).to(self.device)
        l_T = torch.randn([batch_size, 3, 3]).to(self.device)
        t_T = torch.randn([batch.num_nodes, MAX_ATOMIC_NUM]).to(self.device)

        if self.keep_coords:
            x_T = batch.frac_coords

        if self.keep_lattice:
            l_T = lattice_params_to_matrix_torch(batch.lengths, batch.angles)

        traj = {self.beta_scheduler.timesteps: {
            'atom_types': t_T,
            'frac_coords': x_T % 1.,
            'lattices': l_T,
            'num_atoms': batch.num_atoms,
            'batch_idx': batch.batch,
        }}

        for t in range(self.beta_scheduler.timesteps, 0, -1):

            times = torch.full((batch_size, ), t, device = self.device)
            time_emb = self.time_embedding(times)

            alphas = self.beta_scheduler.alphas[t]
            alphas_cumprod = self.beta_scheduler.alphas_cumprod[t]
            c0 = 1.0 / torch.sqrt(alphas)
            c1 = (1 - alphas) / torch.sqrt(1 - alphas_cumprod)

            sigmas = self.beta_scheduler.sigmas[t]
            sigma_x = self.sigma_scheduler.sigmas[t]
            sigma_norm = self.sigma_scheduler.sigmas_norm[t]

            x_t = traj[t]['frac_coords']
            l_t = traj[t]['lattices']
            t_t = traj[t]['atom_types']

            if self.keep_coords:
                x_t = x_T

            if self.keep_lattice:
                l_t = l_T

            # Corrector
            rand_l = torch.randn_like(l_T) if t > 1 else torch.zeros_like(l_T)
            rand_t = torch.randn_like(t_T) if t > 1 else torch.zeros_like(t_T)
            rand_x = torch.randn_like(x_T) if t > 1 else torch.zeros_like(x_T)

            step_size = step_lr * (sigma_x / self.sigma_scheduler.sigma_begin) ** 2
            std_x = torch.sqrt(2 * step_size)

            pred_l, pred_x, pred_t = self.decoder(time_emb, t_t, x_t, l_t, batch.num_atoms, batch.batch)
            pred_x = pred_x * torch.sqrt(sigma_norm)

            x_t_minus_05 = x_t - step_size * pred_x + std_x * rand_x if not self.keep_coords else x_t
            l_t_minus_05 = l_t
            t_t_minus_05 = t_t
            x_mu_corr = (x_t - step_size * pred_x) % 1.
            x_sigma_corr = std_x

            # Predictor
            rand_l = torch.randn_like(l_T) if t > 1 else torch.zeros_like(l_T)
            rand_t = torch.randn_like(t_T) if t > 1 else torch.zeros_like(t_T)
            rand_x = torch.randn_like(x_T) if t > 1 else torch.zeros_like(x_T)

            adjacent_sigma_x = self.sigma_scheduler.sigmas[t-1]
            step_size = (sigma_x ** 2 - adjacent_sigma_x ** 2)
            std_x = torch.sqrt((adjacent_sigma_x ** 2 * (sigma_x ** 2 - adjacent_sigma_x ** 2)) / (sigma_x ** 2)) 

            pred_l, pred_x, pred_t = self.decoder(time_emb, t_t_minus_05, x_t_minus_05, l_t_minus_05, batch.num_atoms, batch.batch)
            pred_x = pred_x * torch.sqrt(sigma_norm)

            x_t_minus_1 = x_t_minus_05 - step_size * pred_x + std_x * rand_x if not self.keep_coords else x_t
            l_t_minus_1 = c0 * (l_t_minus_05 - c1 * pred_l) + sigmas * rand_l if not self.keep_lattice else l_t
            t_t_minus_1 = c0 * (t_t_minus_05 - c1 * pred_t) + sigmas * rand_t
            x_t_minus_1 = x_t_minus_1 % 1.

            if t > 1:
                dist_l = Normal(c0 * (l_t_minus_05 - c1 * pred_l), sigmas)
                dist_t = Normal(c0 * (t_t_minus_05 - c1 * pred_t), sigmas)

                log_prob_l = dist_l.log_prob(l_t_minus_1.detach().clone()).mean(dim=-1).mean(dim=-1)
                log_prob_t = dist_t.log_prob(t_t_minus_1.detach().clone()).mean(dim=-1)
                log_prob_t = scatter(log_prob_t, batch.batch, reduce='mean')

                log_prob_x_corr = log_prob_wn(
                    x_t_minus_05.detach().clone() % 1.,
                    x_mu_corr,
                    x_sigma_corr,
                ).mean(dim=-1)
                log_prob_x_corr = scatter(log_prob_x_corr, batch.batch, reduce='mean')

                x_mu_pred = (x_t_minus_05 - step_size * pred_x) % 1.
                x_sigma_pred = std_x
                log_prob_x_pred = log_prob_wn(
                    x_t_minus_1.detach().clone(),
                    x_mu_pred,
                    x_sigma_pred,
                ).mean(dim=-1)
                log_prob_x_pred = scatter(log_prob_x_pred, batch.batch, reduce='mean')

                log_prob_x = log_prob_x_corr + log_prob_x_pred

                traj[t]['log_prob_l'] = log_prob_l
                traj[t]['log_prob_t'] = log_prob_t
                traj[t]['log_prob_x'] = log_prob_x
                traj[t]['frac_coords_mid'] = x_t_minus_05 % 1.

            traj[t - 1] = {
                'atom_types': t_t_minus_1,
                'frac_coords': x_t_minus_1 % 1.,
                'lattices': l_t_minus_1,
                'num_atoms': batch.num_atoms,
                'batch_idx': batch.batch,
            }

        # traj_stack = {
        #     'num_atoms' : batch.num_atoms,
        #     'atom_types' : torch.stack([traj[i]['atom_types'] for i in range(self.beta_scheduler.timesteps, -1, -1)]).argmax(dim=-1) + 1,
        #     'all_frac_coords' : torch.stack([traj[i]['frac_coords'] for i in range(self.beta_scheduler.timesteps, -1, -1)]),
        #     'all_lattices' : torch.stack([traj[i]['lattices'] for i in range(self.beta_scheduler.timesteps, -1, -1)])
        # }

        return traj[0], traj

    def ft_step(self, batch):

        noised_input, noises = self.add_noise(batch)
        rand_l, tar_x, rand_t = noises
        pred_l, pred_x, pred_t = self(noised_input)

        loss_lattice = F.mse_loss(pred_l, rand_l)
        loss_coord = F.mse_loss(pred_x, tar_x)
        loss_type = F.mse_loss(pred_t, rand_t)

        loss = (
            self.hparams.cost_lattice * loss_lattice +
            self.hparams.cost_coord * loss_coord + 
            self.hparams.cost_type * loss_type
        )

        loss_dict = {
            'loss' : loss,
            'loss_lattice' : loss_lattice,
            'loss_coord' : loss_coord,
            'loss_type' : loss_type
        }

        return loss, loss_dict

    def loss_time_items(self, batch, time):

        noised_input, noises = self.add_noise(batch, time)
        rand_l, tar_x, rand_t = noises
        pred_l, pred_x, pred_t = self(noised_input)

        loss_lattice = torch.pow((pred_l - rand_l), 2).mean(dim=(1,2))
        _loss_coord = torch.pow((pred_x - tar_x), 2).mean(dim=1)
        loss_coord = scatter(_loss_coord, batch.batch, dim=0, reduce='mean')
        _loss_type = torch.pow((pred_t - rand_t), 2).mean(dim=1)
        loss_type = scatter(_loss_type, batch.batch, dim=0, reduce='mean')

        # loss_lattice = F.mse_loss(pred_l, rand_l)
        # loss_coord = F.mse_loss(pred_x, tar_x)
        # loss_type = F.mse_loss(pred_t, rand_t)

        loss = (
            self.hparams.cost_lattice * loss_lattice +
            self.hparams.cost_coord * loss_coord + 
            self.hparams.cost_type * loss_type
        )

        # loss_dict = {
        #     'loss' : loss,
        #     'loss_lattice' : loss_lattice,
        #     'loss_coord' : loss_coord,
        #     'loss_type' : loss_type
        # }

        return loss

    def training_step(self, batch: Any, batch_idx: int) -> torch.Tensor:

        noised_input, noises = self.add_noise(batch)
        rand_l, tar_x, rand_t = noises
        pred_l, pred_x, pred_t = self(noised_input)

        loss_lattice = F.mse_loss(pred_l, rand_l)
        loss_coord = F.mse_loss(pred_x, tar_x)
        loss_type = F.mse_loss(pred_t, rand_t)

        loss = (
            self.hparams.cost_lattice * loss_lattice +
            self.hparams.cost_coord * loss_coord + 
            self.hparams.cost_type * loss_type
        )

        # self.log_dict(
        #     {'train_loss': loss,
        #     'lattice_loss': loss_lattice,
        #     'coord_loss': loss_coord,
        #     'type_loss': loss_type},
        #     on_step=True,
        #     on_epoch=True,
        #     prog_bar=True,
        # )

        if loss.isnan():
            return None

        return loss

    def validation_step(self, batch: Any, batch_idx: int) -> torch.Tensor:

        output_dict = self(batch)

        log_dict, loss = self.compute_stats(output_dict, prefix='val')

        self.log_dict(
            log_dict,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
        )
        return loss

    def test_step(self, batch: Any, batch_idx: int) -> torch.Tensor:

        output_dict = self(batch)

        log_dict, loss = self.compute_stats(output_dict, prefix='test')

        self.log_dict(
            log_dict,
        )
        return loss

    def compute_stats(self, output_dict, prefix):

        loss_lattice = output_dict['loss_lattice']
        loss_coord = output_dict['loss_coord']
        loss_type = output_dict['loss_type']
        loss = output_dict['loss']

        log_dict = {
            f'{prefix}_loss': loss,
            f'{prefix}_lattice_loss': loss_lattice,
            f'{prefix}_coord_loss': loss_coord,
            f'{prefix}_type_loss': loss_type,
        }

        return log_dict, loss
