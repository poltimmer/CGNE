from argparse import ArgumentParser
from datetime import timedelta
from pathlib import Path

import lightning as L
import wandb
import yaml
from lightning.pytorch.callbacks import RichModelSummary, RichProgressBar, StochasticWeightAveraging
from lightning_fabric import seed_everything

from lightning_modules.callback import ImageLogCallback, RolloutSamplerCallback, GradientClippingCallback, \
    WassersteinDistanceMetricCallback
from lightning_modules.datamodule import AutoRegressiveHDF5DataModule
from lightning_modules.cgne import CGNE
from utils.misc import update_nested_dict, load_model_state_dict_from_artifact, CustomWandbLogger, CustomModelCheckpoint


def run(configs: list, wandb_resume_run_id: str, debug=False):
    with open("./configs/pns_base_config.yaml", "r") as f:
        config = yaml.safe_load(f)

    ckpt_path = None

    project_name = "debug" if debug else 'CGNE'
    if wandb_resume_run_id is not None:
        wandb_logger = CustomWandbLogger(project=project_name,
                                         log_model=True,
                                         id=wandb_resume_run_id,
                                         resume="must",
                                         mode="online")
        artifact = wandb_logger.use_artifact(
            f"poltimmer/{wandb_logger.experiment.project}/model-{wandb_logger.experiment.id}:latest")
        artifact_dir = artifact.download()
        ckpt_path = Path(artifact_dir) / "model.ckpt"
    else:
        wandb_logger = CustomWandbLogger(project=project_name,
                                         log_model=True,
                                         mode="online")

    config = update_nested_dict(config, wandb_logger.experiment.config.as_dict())
    for conf_name in configs:
        with open(f"./configs/{conf_name}.yaml", "r") as f:
            config = update_nested_dict(config, yaml.safe_load(f))
    wandb_logger.experiment.config.update(config, allow_val_change=True)

    seed_everything(config["general"]["seed"])

    data_module = AutoRegressiveHDF5DataModule(**config["data"])
    model = CGNE(**config["model"])
    imglog_callback = ImageLogCallback(**config["callback"]["imglog_callback"])

    wandb_logger.watch(model, log="gradients")

    if config["general"]["model_weights_artifact"] is not None:
        load_model_state_dict_from_artifact(model, config["general"]["model_weights_artifact"], wandb_logger)

    checkpoint_callback = CustomModelCheckpoint(
        alias='val_metric',
        filename='val_metric-{epoch}-{step}',
        monitor="val/mean_wasserstein_distance",
        mode="min",
    )

    interval_checkpoint_callback = CustomModelCheckpoint(
        alias='interval',
        filename='interval-{epoch}-{step}',
        train_time_interval=timedelta(hours=12),
        save_top_k=-1,
    )

    save_last_callback = CustomModelCheckpoint(
        alias='last',
        filename='last-{epoch}-{step}',
        save_last=True,
    )

    callbacks = [RichModelSummary(max_depth=3), RichProgressBar(), checkpoint_callback, interval_checkpoint_callback,
                 save_last_callback, imglog_callback]

    if "rollout_callback" in config["callback"] \
            and config["callback"]["rollout_callback"] is not None:
        callbacks.append(RolloutSamplerCallback(**config["callback"]["rollout_callback"]))

    if "grad_clip_callback" in config["callback"] and config["callback"]["grad_clip_callback"] is not None:
        callbacks.append(GradientClippingCallback(**config["callback"]["grad_clip_callback"]))

    if "swa_callback" in config["callback"] and config["callback"]["swa_callback"] is not None:
        callbacks.append(StochasticWeightAveraging(**config["callback"]["swa_callback"]))

    if "metric_callback" in config["callback"] and config["callback"]["metric_callback"] is not None:
        callbacks.append(WassersteinDistanceMetricCallback(**config["callback"]["metric_callback"]))

    trainer = L.Trainer(
        callbacks=callbacks,
        logger=wandb_logger,
        **config["trainer"])

    data_module.setup()

    trainer.fit(model, datamodule=data_module, ckpt_path=ckpt_path if ckpt_path is not None else None)


if __name__ == "__main__":
    argparse = ArgumentParser()
    argparse.add_argument("-c", "--config", type=str, action='append', default=[])
    argparse.add_argument("-id", "--wandb_resume_run_id", type=str, default=None)
    argparse.add_argument("--debug", action="store_true")
    args = argparse.parse_args()
    run(args.config, args.wandb_resume_run_id, args.debug)
