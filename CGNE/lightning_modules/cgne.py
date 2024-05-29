import random
from typing import Tuple, Dict

import torch
from lightning import LightningModule
from torch import Tensor
from torch.distributions import kl_divergence, Normal
from torch.nn import functional as F

from CGNE.models.cgne import ApproximatePosterior, UNet, ConditionalPrior, SpatialZDecoder
from CGNE.models.layers import linear
from CGNE.utils.misc import get_reduce_fn, prefix_dict, samplewise_dropout


class CGNE(LightningModule):
    def __init__(self, lr=1e-3,
                 unroll_length=30,
                 max_t=float('inf'),
                 enc_hidden_dim=64,
                 spatial_z=True,
                 z_dim=64,
                 z_emb_dim=64,
                 z_emb_hidden_dim=256,
                 z_emb_n_layers=4,
                 y_dim=7,
                 y_emb_dim=16,
                 y_emb_hidden_dim=64,
                 y_emb_n_layers=2,
                 y_recon_hidden_dim=128,
                 y_recon_n_layers=4,
                 unet_hidden_dim=64,
                 unet_z_inject_method='cat',
                 unet_out_loss_weight=0.1,
                 boundary_loss_weight=1.0,
                 prior_loss_weight=0.0,
                 y_recon_loss_weight=1.0,
                 beta=1.0,
                 beta_annealing_epochs=16,
                 free_bits=0.0,
                 decoder_dropout=0.0):
        super().__init__()
        self.save_hyperparameters()
        channels = 1
        self.example_input_array = [torch.randn(4, channels, 64, 64), torch.randn(4, y_dim)]

        self.decoder = UNet(c_in=channels, c_out=channels, z_dim=z_dim,
                            hidden_dim=unet_hidden_dim,
                            spatial_z=spatial_z, z_emb_dim=z_emb_dim,
                            z_emb_hidden_dim=z_emb_hidden_dim, z_emb_n_layers=z_emb_n_layers,
                            z_inject_method=unet_z_inject_method)

        self.mask = lambda x: ~x.type(torch.bool)

        self.qz_xt1xty = ApproximatePosterior(x_channels=channels, hidden_dim=enc_hidden_dim, z_dim=z_dim,
                                              spatial_z=spatial_z, y_dim=y_dim, y_embed_dim=y_emb_dim,
                                              y_hidden_dim=y_emb_hidden_dim, y_n_layers=y_emb_n_layers)

        self.pz_xty = ConditionalPrior(x_channels=channels, hidden_dim=enc_hidden_dim, z_dim=z_dim,
                                       spatial_z=spatial_z, y_dim=y_dim, y_embed_dim=y_emb_dim,
                                       y_hidden_dim=y_emb_hidden_dim, y_n_layers=y_emb_n_layers)

        if spatial_z:
            self.py_z = SpatialZDecoder(z_dim=z_dim, out_dim=y_dim, fc_hidden_dim=y_recon_hidden_dim,
                                        fc_n_layers=y_recon_n_layers)
        else:
            self.py_z = linear(z_dim, y_dim, hidden_dim=y_recon_hidden_dim, n_layers=y_recon_n_layers)

        self.automatic_optimization = False
        self.unroll_length = 1
        self.decoder_dropout = 0.0
        self.max_t = float('inf')
        self.beta_annealing_factor = 1.0

    def forward(self, x_t: Tensor, y: Tensor) -> Tuple[Tensor, Tensor]:
        """Inference pass"""
        x_t1_hat, unet_out, _, _ = self.prior_sample(x_t, y)
        return x_t1_hat, unet_out

    def sample_rollout(self, x_t, y, n_steps=16, stop_margin=None) -> Tensor:
        """Sample a rollout of length n_steps"""
        x_t_hat = x_t
        x_hat = []
        for _ in range(n_steps):
            x_t1_hat, _, _, _ = self.prior_sample(x_t_hat, y)
            if stop_margin:
                done_1 = torch.sum(x_t_hat[:, :, -stop_margin:, :], dim=[1, 2, 3]) > 0
                done_2 = torch.sum(x_t_hat[:, :, :, -stop_margin:], dim=[1, 2, 3]) > 0
                done = done_1 | done_2
                x_t1_hat_filtered = x_t_hat.clone()
                x_t1_hat_filtered[~done] = x_t1_hat[~done]
                if done.all():
                    break
            else:
                x_t1_hat_filtered = x_t1_hat
            x_hat.append(x_t1_hat_filtered)
            x_t_hat = x_t1_hat_filtered
        return torch.stack(x_hat)

    def guided_rollout(self, x, y) -> Tensor:
        """Sample a rollout of length len(x) - 1, using the
        variational posterior Q(z|x_t1, x_t, y) at each step."""
        x_t_hat = x[0]
        x_hat = []
        for i in range(x.shape[0] - 1):
            x_t1_hat, _, _, _ = self.posterior_sample(x[i + 1], x_t_hat, y)
            x_hat.append(x_t1_hat)
            x_t_hat = x_t1_hat
        return torch.stack(x_hat)

    def full_forward(self, x_t: Tensor, y: Tensor, x_t1: Tensor):
        """Training pass"""
        p_x_t1_hat, p_unet_out, pz_xty, pz = self.prior_sample(x_t, y)
        q_x_t1_hat, q_unet_out, qz_xt1xty, qz = self.posterior_sample(x_t1, x_t, y)
        p_y_hat = self.py_z(pz)
        q_y_hat = self.py_z(qz)

        return p_x_t1_hat, p_unet_out, q_x_t1_hat, q_unet_out, p_y_hat, q_y_hat, pz_xty, qz_xt1xty

    def prior_sample(self, x_t: Tensor, y: Tensor) -> Tuple[Tensor, Tensor, Normal, Tensor]:
        """Sample from the prior"""
        pz_loc, pz_scale = self.pz_xty(x_t, y)
        x_t1_hat, unet_out, pz_xty, z = self._make_dist_and_sample(x_t, pz_loc, pz_scale, y)
        return x_t1_hat, unet_out, pz_xty, z

    def posterior_sample(self, x_t1: Tensor, x_t: Tensor, y: Tensor) -> Tuple[Tensor, Tensor, Normal, Tensor]:
        """Sample from the posterior"""
        qz_loc, qz_scale = self.qz_xt1xty(x_t1, x_t, y)
        x_t1_hat, unet_out, qz_xt1xty, z = self._make_dist_and_sample(x_t, qz_loc, qz_scale, y)
        return x_t1_hat, unet_out, qz_xt1xty, z

    def _make_dist_and_sample(self, x_t: Tensor, z_loc: Tensor, z_scale: Tensor, y: Tensor) -> Tuple[
        Tensor, Tensor, Normal, Tensor]:
        z_dist = torch.distributions.Normal(z_loc, z_scale)
        z = z_dist.rsample()
        x_t_dropout = samplewise_dropout(x_t, self.decoder_dropout, self.training)
        unet_out = self.decoder(x_t_dropout, z)
        x_t1_hat = self._mask_and_mix_output(x_t, unet_out)
        return x_t1_hat, unet_out, z_dist, z

    def _mask_and_mix_output(self, x_t, unet_out):
        unet_out_a_sig = F.sigmoid(unet_out[:, :1])
        frozen_state_mask = self.mask(x_t)
        unet_out_binary = (unet_out_a_sig > 0.5).type(torch.float32)
        x_t1_hat = x_t * ~frozen_state_mask + unet_out_binary * frozen_state_mask
        return x_t1_hat

    def recon_losses(self, x_t, x_t1, unet_out, reduce_fn=torch.mean) -> Dict[str, Tensor]:
        losses = {}
        mask = self.mask(x_t)
        # Reconstruction loss
        bce = F.binary_cross_entropy_with_logits(unet_out[:, :1], x_t1[:, :1], reduction='none')
        losses["boundary_loss"] = reduce_fn(bce * mask)
        losses["unet_out_loss"] = reduce_fn(bce * ~mask)

        return losses

    def loss_func(self, x_t, y, x_t1, p_unet_out, q_unet_out, pz_xty, qz_xt1xty, p_y_hat, q_y_hat,
                  reduce_fn=torch.mean):
        losses = prefix_dict(self.recon_losses(x_t, x_t1, p_unet_out, reduce_fn=reduce_fn), 'prior_')
        losses.update(prefix_dict(self.recon_losses(x_t, x_t1, q_unet_out, reduce_fn=reduce_fn), 'posterior_'))
        # KL loss
        losses["kl_loss"] = reduce_fn(kl_divergence(qz_xt1xty, pz_xty), dim=0).clamp_min(
            self.hparams.free_bits).mean()

        # Auxiliary loss
        losses["prior_y_recon_loss"] = reduce_fn(F.mse_loss(p_y_hat, y, reduction='none'))
        losses["posterior_y_recon_loss"] = reduce_fn(F.mse_loss(q_y_hat, y, reduction='none'))

        loss = losses["posterior_boundary_loss"] * self.hparams.boundary_loss_weight + \
               losses["posterior_unet_out_loss"] * self.hparams.unet_out_loss_weight

        if self.hparams.prior_loss_weight > 0.0:
            loss += (losses["prior_boundary_loss"] * self.hparams.boundary_loss_weight +
                     losses["prior_unet_out_loss"] * self.hparams.unet_out_loss_weight) * self.hparams.prior_loss_weight

        if self.hparams.boundary_reinforcement_loss_weight > 0.0:
            loss += losses["posterior_boundary_reinforcement_loss"] * self.hparams.boundary_reinforcement_loss_weight
        if self.hparams.boundary_count_loss_weight > 0.0:
            loss += losses["posterior_boundary_count_loss"] * self.hparams.boundary_count_loss_weight

        loss += (losses["kl_loss"] - self.hparams.free_bits) * self.hparams.beta * self.beta_annealing_factor
        loss += losses["posterior_y_recon_loss"] * self.hparams.y_recon_loss_weight

        losses["loss"] = loss

        return losses

    def _schedule_hparam(self, attr_name: str, value_type, min_value=None, max_value=None):
        if isinstance(getattr(self.hparams, attr_name), value_type):
            setattr(self, attr_name, getattr(self.hparams, attr_name))
        elif isinstance(getattr(self.hparams, attr_name), dict):
            if self.current_epoch in getattr(self.hparams, attr_name):
                setattr(self, attr_name, getattr(self.hparams, attr_name)[self.current_epoch])
        elif isinstance(getattr(self.hparams, attr_name), str):
            expr = getattr(self.hparams, attr_name).replace("epoch", str(self.current_epoch))
            try:
                value = value_type(eval(expr))
                if min_value is not None:
                    value = max(min_value, value)
                if max_value is not None:
                    value = min(max_value, value)
                setattr(self, attr_name, value)
            except Exception as _:
                raise ValueError(f"Invalid expression for {attr_name}: {expr}")

        self.log(attr_name, getattr(self, attr_name), on_step=False, on_epoch=True)

    def on_train_epoch_start(self) -> None:
        self._schedule_hparam("unroll_length", int, min_value=1)
        self._schedule_hparam("decoder_dropout", float, min_value=0.0, max_value=1.0)
        self._schedule_hparam("max_t", float, min_value=1)

        # Diva beta annealing
        if self.hparams.beta_annealing_epochs > 0:
            self.beta_annealing_factor = min(1.0, self.current_epoch / self.hparams.beta_annealing_epochs)

    def training_step(self, batch, batch_idx):
        self.shared_step(batch, training=True)

    def validation_step(self, batch, batch_idx):
        self.shared_step(batch, training=False)

    def shared_step(self, batch, training: bool):
        x, pad_masks, y, t = batch
        batch_size = y.shape[0]
        opt = self.optimizers()

        max_len = (pad_masks & (t <= self.max_t)).sum(dim=0).max().item()

        # get t0
        x_t_hat = x[0]
        unroll_length = unroll_steps = min(random.randint(1, self.unroll_length), max_len)

        for i in range(x.shape[0] - 1):
            loss_mask = pad_masks[i + 1] & (t[i + 1] <= self.max_t)
            if not torch.any(loss_mask):
                # Stop batch if all sequences have ended
                if training:
                    opt.step()
                    opt.zero_grad()
                break

            p_x_t1_hat, p_unet_out, q_x_t1_hat, q_unet_out, p_y_hat, q_y_hat, pz_xty, qz_xt1xty \
                = self.full_forward(x_t_hat, y, x[i + 1])

            losses = self.loss_func(x[i], y, x[i + 1], p_unet_out, q_unet_out, pz_xty, qz_xt1xty, p_y_hat, q_y_hat,
                                    reduce_fn=get_reduce_fn('mean', loss_mask))

            x_t_hat = q_x_t1_hat
            unroll_steps -= 1

            if training:
                self.log_dict(prefix_dict(losses, 'train/'), on_step=True, on_epoch=False)
                loss = losses['loss']
                self.manual_backward(loss * loss_mask.float().mean() / unroll_length)
                x_t_hat = x_t_hat.detach()
            else:
                self.log_dict(prefix_dict(losses, 'val/'), on_step=False, on_epoch=True, batch_size=batch_size)
            if unroll_steps == 0 or i == x.shape[0] - 2:
                if training:
                    opt.step()
                    opt.zero_grad()
                x_t_hat = x[i + 1]
                unroll_length = unroll_steps = min(random.randint(1, self.unroll_length), max_len - i - 1)

    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=self.hparams.lr)
