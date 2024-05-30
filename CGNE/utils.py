import math
from argparse import Namespace
from copy import deepcopy
from pathlib import Path
from typing import Union, Dict, Any, Optional, Tuple, List

import numpy as np
import ot
import torch
import wandb
from lightning import LightningModule, Trainer
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.loggers.utilities import _scan_checkpoints
from scipy.ndimage import distance_transform_cdt
from torch import Tensor


def ceildiv(a, b):
    return -(-a // b)


def samplewise_dropout(x, p=0.5, training=True):
    if p == 0 or not training:
        return x
    elif p == 1:
        return torch.zeros_like(x)
    dropout_mask = torch.rand(x.size(0), device=x.device) > p
    return x * dropout_mask.view(-1, 1, 1, 1)


def round_hidden_dim(val):
    """Rounds a hidden number to a nice value for a hidden dimension size."""
    if val < 32:
        return max(1, 2 ** math.ceil(math.log2(val)))
    elif val < 256:
        return 32 * round(val / 32)
    else:
        return 64 * round(val / 64)


def update_nested_dict(defaults, changes):
    for key, value in changes.items():
        if isinstance(value, dict) and key in defaults and isinstance(defaults[key], dict):
            # Get the default dictionary for this key (create a new empty dict if it doesn't exist)
            defaults[key] = update_nested_dict(defaults.get(key, {}), value)
        else:
            defaults[key] = value
    return defaults


def load_model_state_dict_from_artifact(model: LightningModule, model_weights_artifact: str, wandb_logger: WandbLogger):
    model_weights_artifact = deepcopy(model_weights_artifact)
    diff_keys_total = None
    if isinstance(model_weights_artifact, str):
        model_weights_artifact = [model_weights_artifact]
    for artifact_name in model_weights_artifact:
        artifact = wandb_logger.use_artifact(artifact_name)
        artifact_dir = artifact.download()
        diff_keys = model.load_state_dict(
            torch.load(Path(artifact_dir) / "model.ckpt", map_location="cpu")["state_dict"], strict=False)
        if diff_keys_total is None:
            diff_keys_total = diff_keys
        else:
            diff_keys_total.missing_keys = list(
                set(diff_keys_total.missing_keys).union(set(diff_keys.missing_keys)))
            diff_keys_total.unexpected_keys = list(
                set(diff_keys_total.unexpected_keys).union(set(diff_keys.unexpected_keys)))
    print(f"Missing keys: {diff_keys_total.missing_keys}")
    print(f"Unexpected keys: {diff_keys_total.unexpected_keys}")


def get_reduce_fn(reduce: str = 'mean', pad_masks: Optional[torch.Tensor] = None):
    """Creates a reduction function that takes padding into account."""

    def reduce_fn(x, dim: Optional[Union[int, Tuple[int]]] = None):
        if pad_masks is not None:
            shape = [-1] + [1] * (x.dim() - 1)
            pad_masks_resized = pad_masks.reshape(*shape)
            x = x * pad_masks_resized
        if reduce == 'mean':
            if pad_masks is not None and (dim is None or dim == 0 or 0 in dim):
                # batch reduction with padding
                return torch.mean(x, dim=dim) / pad_masks.float().mean()
            return torch.mean(x, dim=dim)
        elif reduce == 'sum':
            return torch.sum(x, dim=dim)
        elif reduce == 'none':
            return x
        else:
            raise ValueError(f"Unknown reduce function {reduce}")

    return reduce_fn


def prefix_dict(d: dict, prefix: str):
    return {prefix + key: value for key, value in d.items()}


class CustomModelCheckpoint(ModelCheckpoint):
    def __init__(self, *args, alias='main', **kwargs):
        super().__init__(*args, **kwargs)
        self._alias = alias

    def _monitor_candidates(self, trainer: "Trainer") -> Dict[str, Tensor]:
        monitor_candidates = deepcopy(trainer.logged_metrics)  # use logged_metrics instead of callback_metrics
        # cast to int if necessary because `self.log("epoch", 123)` will convert it to float. if it's not a tensor
        # or does not exist we overwrite it as it's likely an error
        epoch = monitor_candidates.get("epoch")
        monitor_candidates["epoch"] = epoch.int() if isinstance(epoch, Tensor) else torch.tensor(trainer.current_epoch)
        step = monitor_candidates.get("step")
        monitor_candidates["step"] = step.int() if isinstance(step, Tensor) else torch.tensor(trainer.global_step)
        return monitor_candidates


class CustomWandbLogger(WandbLogger):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._checkpoint_callbacks: List[CustomModelCheckpoint] = []

    def log_hyperparams(self, params: Union[Dict[str, Any], Namespace]) -> None:
        params = {'model': params}
        super().log_hyperparams(params)

    def after_save_checkpoint(self, checkpoint_callback: CustomModelCheckpoint) -> None:
        # log checkpoints as artifacts
        if self._log_model == "all" or self._log_model is True and checkpoint_callback.save_top_k == -1:
            self._scan_and_log_checkpoints(checkpoint_callback)
        elif self._log_model is True:
            self._checkpoint_callbacks.append(checkpoint_callback)

    def finalize(self, status: str) -> None:
        # log checkpoints as artifacts
        if self._checkpoint_callbacks and self._experiment is not None:
            for checkpoint_callback in self._checkpoint_callbacks:
                self._scan_and_log_checkpoints(checkpoint_callback)

    def _scan_and_log_checkpoints(self, checkpoint_callback: CustomModelCheckpoint) -> None:
        # get checkpoints to be saved with associated score
        checkpoints = _scan_checkpoints(checkpoint_callback, self._logged_model_time)

        # log iteratively all new checkpoints
        for t, p, s, tag in checkpoints:
            metadata = (
                {
                    "score": s.item() if isinstance(s, Tensor) else s,
                    "original_filename": Path(p).name,
                    checkpoint_callback.__class__.__name__: {
                        k: getattr(checkpoint_callback, k)
                        for k in [
                            "monitor",
                            "mode",
                            "save_last",
                            "save_top_k",
                            "save_weights_only",
                            "_every_n_train_steps",
                        ]
                        # ensure it does not break if `ModelCheckpoint` args change
                        if hasattr(checkpoint_callback, k)
                    },
                }
            )
            if not self._checkpoint_name:
                self._checkpoint_name = f"model-{self.experiment.id}"
            artifact = wandb.Artifact(name=self._checkpoint_name, type="model", metadata=metadata)
            artifact.add_file(p, name="model.ckpt")
            aliases = [f"{checkpoint_callback._alias}-latest",
                       f"{checkpoint_callback._alias}-best"] if p == checkpoint_callback.best_model_path else \
                [f"{checkpoint_callback._alias}-latest"]
            self.experiment.log_artifact(artifact, aliases=aliases)
            # remember logged models - timestamp needed in case filename didn't change (lastkckpt or custom name)
            self._logged_model_time[p] = t


hex_dist_metric = np.array([[1, 1, 2],
                            [1, 1, 1],
                            [2, 1, 1]])


def batch_distance_transform_cdt(batch):
    # Assuming batch is a PyTorch tensor of shape (batch_size, 1, height, width)
    batch_np = batch.cpu().numpy()  # Convert to numpy array
    # Apply distance transform to each item in the batch
    # This uses a comprehension list to apply the operation individually
    # and then stacks the results back into a single numpy array
    transformed = np.stack(
        [np.expand_dims(distance_transform_cdt(im[0], metric=hex_dist_metric), axis=0) for im in batch_np])
    return torch.from_numpy(transformed).to(batch.device)

def get_expected_wasserstein_distance(rho, gt_area, gt_boundary_length, area, boundary_length, num_slices=16):
    """Given arrays for the conditioning variable, ground truth and rollout area and boundary length,
    compute the expected Wasserstein distance between the ground truth and rollout distributions."""

    # Compute Wasserstein distances
    var_min = np.min(rho)
    var_max = np.max(rho)

    # Create equal-width bins for the variable between its min and max
    var_bins = np.linspace(var_min, var_max, num=num_slices + 1)

    # Bin the variable
    var_bin_indices = np.digitize(rho, var_bins) - 1  # Get bin indices for each 'rho'

    # Initialize a list to store Wasserstein distances
    wasserstein_distances = []

    # Compute Wasserstein distances for each bin
    for bin_index in range(num_slices):
        # Select the rows for the current bin
        bin_mask = (var_bin_indices == bin_index)

        if np.any(bin_mask):
            # Extract GT and Rollout distributions for the current bin
            gt_distribution = np.stack((gt_area[bin_mask], gt_boundary_length[bin_mask]), axis=-1)
            rollout_distribution = np.stack((area[bin_mask], boundary_length[bin_mask]), axis=-1)

            # Uniform distribution weights for Wasserstein distance calculation
            gt_weights = np.ones((gt_distribution.shape[0],)) / gt_distribution.shape[0]
            rollout_weights = np.ones((rollout_distribution.shape[0],)) / rollout_distribution.shape[0]

            # Compute the cost matrix between the two distributions
            cost_matrix = ot.dist(gt_distribution, rollout_distribution, metric='euclidean')

            # Compute the Wasserstein distance using the EMD solver
            wasserstein_dist = ot.emd2(gt_weights, rollout_weights, cost_matrix)
            wasserstein_distances.append(wasserstein_dist)
        else:
            print(f"No data in bin {bin_index}.")

    return np.mean(wasserstein_distances)


param_list = ['alpha', 'beta', 'gamma', 'theta', 'kappa', 'mu', 'rho']
