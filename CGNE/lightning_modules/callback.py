import random
from io import BytesIO
from typing import Union, Optional

import numpy as np
import ot
import torch
import wandb
from PIL import Image
from einops import rearrange
from lightning import LightningModule, Trainer
from lightning.pytorch import Callback
from torch import Tensor
from torch.nn import functional as F
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Subset

from CGNE.dataset import reverse_transform_params, FirstAndLastHDF5Dataset
from CGNE.lightning_modules.cgne import CGNE
from CGNE.models.layers import BorderMask
from CGNE.utils import param_list


class ImageLogCallback(Callback):
    """Logs images of the input, output, and delta of the model, as well as the ground truth labels."""

    def __init__(self, log_every_n_epochs: int = 0, num_samples: int = 16, sample_from_first_n_samples: int = 128,
                 **kwargs):
        super().__init__()
        self.example_batch = None
        self.log_every_n_epochs = log_every_n_epochs
        self.first_log_done = False
        self.num_samples = num_samples
        self.sample_from_first_n_samples = sample_from_first_n_samples
        if len(kwargs) > 0:
            print(f"Other kwargs: {kwargs}")

    def on_train_epoch_end(self, trainer, pl_module: CGNE):
        if self.example_batch is None:
            dl = pl_module.trainer.train_dataloader
            dataset = dl.dataset
            collate_fn = dl.collate_fn
            indices = list(range(min(len(dataset), self.sample_from_first_n_samples)))
            random.shuffle(indices)
            samples = [dataset[i] for i in indices[:self.num_samples]]
            x, padding_masks, y, t = collate_fn(samples)
            example_batch = x[:2], padding_masks[:2], y, t[:2]
            self.example_batch = pl_module.transfer_batch_to_device(example_batch, device=pl_module.device,
                                                                    dataloader_idx=0)

        if trainer.current_epoch % self.log_every_n_epochs == 0 or not self.first_log_done:
            x, padding_masks, y, t = self.example_batch
            with torch.no_grad():
                p_x_t1_hat, p_unet_out, q_x_t1_hat, q_unet_out, p_y_hat, q_y_hat, pz_xty, qz_xt1xty \
                    = pl_module.full_forward(x[0], y, x[1])

            if not self.first_log_done:
                pl_module.logger.log_image(key="x_t", images=[img for img in x[0]])
                pl_module.logger.log_image(key="x_t1", images=[img for img in x[1]])
                pl_module.logger.log_image(key="delta", images=[img for img in x[1] - x[0]])

                # Convert labels to list of lists for table data and create a table for labels
                trainer.logger.experiment.log({
                    "raw_labels": wandb.Table(
                        data=[label.tolist() for label in y],
                        columns=param_list
                    )
                })
                trainer.logger.experiment.log({
                    "labels": wandb.Table(
                        data=[reverse_transform_params(label).tolist() for label in y],
                        columns=param_list
                    )
                })
                self.first_log_done = True

            pl_module.logger.log_image(key="unet_out", images=[img for img in F.sigmoid(q_unet_out)])
            pl_module.logger.log_image(key="x_t1_hat", images=[img for img in q_x_t1_hat])
            pl_module.logger.log_image(key="delta_x_hat", images=[img for img in q_x_t1_hat - x[0]])
            pl_module.logger.log_image(key="prior_unet_out", images=[img for img in F.sigmoid(p_unet_out)])

            y_mae: Tensor = torch.abs(q_y_hat - y).mean(dim=0)
            bins = range(len(y_mae) + 1)
            np_hist = y_mae.detach().cpu().numpy(), np.array(bins)
            pl_module.logger.experiment.log({"y_mean_abs_err": wandb.Histogram(np_histogram=np_hist)})


class RolloutSamplerCallback(Callback):
    """Visualizes and logs crystal growth trajectories sampled from the model."""

    def __init__(self, log_every_n_epochs: int = 0, num_steps: int = 64, start_step_indices: Union[list, int] = 0,
                 num_samples: int = 4, fps: int = 4):
        super().__init__()
        self.log_every_n_epochs = log_every_n_epochs
        self.num_steps = num_steps
        if isinstance(start_step_indices, int):
            start_step_indices = [start_step_indices]
        self.start_step_indices = start_step_indices
        self.num_samples = num_samples
        self.fps = fps

    def on_train_epoch_end(self, trainer, pl_module: CGNE):
        if trainer.current_epoch % self.log_every_n_epochs == 0:
            pl_module.eval()
            with torch.no_grad():
                self.log_rollout(trainer, pl_module, pl_module.trainer.train_dataloader, log_key="rollout/train")
                self.log_rollout(trainer, pl_module, pl_module.trainer.train_dataloader, log_key="rollout/train_guided",
                                 guided=True)
            pl_module.train()

    def on_validation_epoch_end(self, trainer, pl_module: CGNE):
        if trainer.sanity_checking:
            return
        if trainer.current_epoch % self.log_every_n_epochs == 0:
            self.log_rollout(pl_module, pl_module.trainer.val_dataloaders, log_key="rollout/val")
            self.log_rollout(pl_module, pl_module.trainer.val_dataloaders, log_key="rollout/val_guided",
                             guided=True)

    def log_rollout(self, pl_module: CGNE, dl, log_key, guided=False):
        dataset = dl.dataset
        collate_fn = dl.collate_fn
        indices = list(range(len(dataset.files_in_class)))
        if len(indices) < self.num_samples:
            indices = indices * self.num_samples
        random.shuffle(indices)
        samples = [
            dataset.get_sequence_from_class(class_idx, self.start_step_indices[i % len(self.start_step_indices)],
                                            sequence_length=self.num_steps)
            for i, class_idx in enumerate(indices[:self.num_samples])]
        x, _, y, _ = pl_module.transfer_batch_to_device(collate_fn(samples), device=pl_module.device,
                                                        dataloader_idx=0)

        with torch.no_grad():
            if guided:
                x_hat = pl_module.guided_rollout(x, y)
            else:
                x_hat = pl_module.sample_rollout(x[0], y, n_steps=min(self.num_steps, len(x) - 1))
        x_3c = x.repeat(1, 1, 3, 1, 1)
        x_3c[1:, :, [0, 2]] = x_3c[:-1, :, [0, 2]].clone()  # highlight delta in green

        x_hat = torch.cat([x[:1], x_hat], dim=0)
        x_hat_3c = x_hat.repeat(1, 1, 3, 1, 1)
        x_hat_3c[1:, :, 1:] = x_hat_3c[:-1, :, 1:].clone()  # highlight delta in red

        # highlight excess in red, missing in teal
        err_3c = rearrange(pad_sequence([x_hat, x, x]), 't n b c h w -> t b (n c) h w')
        delta_x = torch.cat([x[:1], x[1:] - x[:-1]], dim=0)
        delta_x_hat = torch.cat([x_hat[:1], x_hat[1:] - x_hat[:-1]], dim=0)
        # highlight ground truth delta in green, predicted delta in red
        zero = torch.zeros_like(x_hat)
        delta_compared = rearrange(pad_sequence([delta_x_hat, delta_x, zero]), 't n b c h w -> t b (n c) h w')

        t = pad_sequence([x_3c, x_hat_3c, delta_compared, err_3c])
        t = rearrange(t, 't (l1 l2) b c h w -> b t (l1 h) (l2 w) c', l1=2, l2=2).cpu().numpy().astype('uint8') * 255
        image_sequences = [[Image.fromarray(img) for img in images] for images in t]
        gif_buffers = []
        for seq in image_sequences:
            buffer = BytesIO()
            seq[0].save(buffer, format='GIF', save_all=True, append_images=seq[1:], duration=1000 / self.fps, loop=0)
            buffer.seek(0)
            gif_buffers.append(buffer)

        wandb.log({log_key: [wandb.Video(gif, format="gif") for gif in gif_buffers]})


class GradientClippingCallback(Callback):
    def __init__(self, clip_value: float = 1.0):
        super().__init__()
        self.clip_value = clip_value

    def on_after_backward(self, trainer, pl_module: LightningModule):
        grad_norm = torch.nn.utils.clip_grad_norm_(pl_module.parameters(), self.clip_value)
        pl_module.log("grad_norm", grad_norm, on_step=True, on_epoch=False)


class WassersteinDistanceMetricCallback(Callback):
    """Computes and logs the expected Wasserstein distance between the ground truth and the model's distributions
    for the area and boundary length of the simulated crystals."""

    def __init__(self, train_log_every_n_epochs: int = 1, val_log_every_n_epochs: int = 1,
                 rollout_stop_margin: Optional[int] = None, max_rollout_length: int = 64, num_slices: int = 16,
                 train_subset_fraction: float = 1.0, val_subset_fraction: float = 1.0):
        super().__init__()
        self.mask = BorderMask()
        self.train_log_every_n_epochs = train_log_every_n_epochs
        self.val_log_every_n_epochs = val_log_every_n_epochs
        self.rollout_stop_margin = rollout_stop_margin
        self.max_rollout_length = max_rollout_length
        self.num_slices = num_slices
        self.train_subset_fraction = train_subset_fraction
        self.val_subset_fraction = val_subset_fraction

    def on_train_epoch_end(self, trainer: Trainer, pl_module: CGNE):
        if trainer.sanity_checking or trainer.current_epoch % self.train_log_every_n_epochs != 0:
            return
        self.log_metrics(pl_module, pl_module.trainer.train_dataloader, "train", self.train_subset_fraction)

    def on_validation_epoch_end(self, trainer: Trainer, pl_module: CGNE) -> None:
        if trainer.sanity_checking or trainer.current_epoch % self.val_log_every_n_epochs != 0:
            return
        self.log_metrics(pl_module, pl_module.trainer.val_dataloaders, "val", self.val_subset_fraction)

    def log_metrics(self, pl_module: CGNE, dl, split, subset_fraction):
        was_training = pl_module.training
        pl_module.eval()
        with torch.no_grad():
            dataset = FirstAndLastHDF5Dataset(dl.dataset.data_dir, specific_classes=dl.dataset.classes)
            # Subset the dataset if subset_fraction is less than 1
            if subset_fraction < 1.0:
                total_samples = len(dataset)
                subset_size = int(total_samples * subset_fraction)
                indices = np.random.choice(total_samples, subset_size, replace=False)
                subset = Subset(dataset, indices)
            else:
                subset = dataset

            dataloader = DataLoader(subset, batch_size=dl.batch_size, num_workers=dl.num_workers, shuffle=False,
                                    collate_fn=dl.collate_fn)
            self.mask.to(pl_module.device)
            rhos = []
            gt_volumes = []
            gt_boundaries = []
            rollout_volumes = []
            rollout_boundaries = []
            seed_batch = torch.zeros_like(next(iter(dataloader))[0]).to(pl_module.device)

            seed_batch[:, :, 0, 0] = 1

            # Collect datapoints
            for batch in dataloader:
                x, pad_masks, y, t = pl_module.transfer_batch_to_device(batch, device=pl_module.device,
                                                                        dataloader_idx=0)
                rollout = pl_module.sample_rollout(x[0], y, n_steps=self.max_rollout_length,
                                                   stop_margin=self.rollout_stop_margin)

                rho_idx = param_list.index('rho')

                rhos.extend(y[:, rho_idx].tolist())

                gt_volumes.extend(torch.sum(x[-1], dim=(1, 2, 3)).tolist())
                gt_boundaries.extend(torch.sum(self.mask(x[-1]), dim=(1, 2, 3)).tolist())
                rollout_volumes.extend(torch.sum(rollout[-1], dim=(1, 2, 3)).tolist())
                rollout_boundaries.extend(torch.sum(self.mask(rollout[-1]), dim=(1, 2, 3)).tolist())

            rhos = np.array(rhos)
            gt_volumes = np.array(gt_volumes)
            gt_boundaries = np.array(gt_boundaries)
            rollout_volumes = np.array(rollout_volumes)
            rollout_boundaries = np.array(rollout_boundaries)

            # Compute Wasserstein distances
            rho_min = np.min(rhos)
            rho_max = np.max(rhos)

            # Create equal-width bins for 'rho' between its min and max
            rho_bins = np.linspace(rho_min, rho_max, num=self.num_slices + 1)

            # Bin the 'rho' values
            rho_bin_indices = np.digitize(rhos, rho_bins) - 1  # Get bin indices for each 'rho'

            # Initialize a list to store Wasserstein distances
            wasserstein_distances = []

            # Compute Wasserstein distances for each bin
            for bin_index in range(self.num_slices):
                # Select the rows for the current bin
                bin_mask = (rho_bin_indices == bin_index)

                if np.any(bin_mask):
                    # Extract GT and Rollout distributions for the current bin
                    gt_distribution = np.stack((gt_volumes[bin_mask], gt_boundaries[bin_mask]), axis=-1)
                    rollout_distribution = np.stack((rollout_volumes[bin_mask], rollout_boundaries[bin_mask]), axis=-1)

                    # Uniform distribution weights for Wasserstein distance calculation
                    gt_weights = np.ones((gt_distribution.shape[0],)) / gt_distribution.shape[0]
                    rollout_weights = np.ones((rollout_distribution.shape[0],)) / rollout_distribution.shape[0]

                    # Compute the cost matrix between the two distributions
                    cost_matrix = ot.dist(gt_distribution, rollout_distribution, metric='euclidean')

                    # Compute the Wasserstein distance using the EMD solver
                    wasserstein_dist = ot.emd2(gt_weights, rollout_weights, cost_matrix)
                    wasserstein_distances.append(wasserstein_dist)
                else:
                    print(f"No data in bin {bin_index} for the set")

            # Log the mean Wasserstein distance for the current epoch
            mean_wasserstein_distance = np.mean(wasserstein_distances)
            pl_module.log(f"{split}/mean_wasserstein_distance", mean_wasserstein_distance, on_step=False, on_epoch=True)
        if was_training:
            pl_module.train()
