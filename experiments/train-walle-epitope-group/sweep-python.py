import importlib
import os
import sys
from pathlib import Path
from pprint import pprint
from typing import Callable, Dict, Optional

import hydra
import wandb
import yaml
from loguru import logger
from omegaconf import DictConfig, OmegaConf, open_dict
from torch.utils.tensorboard import SummaryWriter  # Import SummaryWriter

from asep.model.utils import generate_random_seed, seed_everything
from asep.train_model import train_model
from asep.utils.utils import time_stamp

# ==================== Sweep Configuration ====================
# add your configuration here


# ==================== Function ====================
def log_config_to_wandb(config: Dict, wandb_run: Optional[wandb.sdk.wandb_run.Run]):
    pprint(config)
    print("wandb run info:")
    print(f"- name: {wandb_run.name}")
    print(f"- project: {wandb_run.project}")
    print(f"- entity: {wandb_run.entity}")
    # upload config as artifact
    artifact = wandb.Artifact("config", type="config")
    with artifact.new_file("config.yaml", mode="w") as f:
        f.write(yaml.dump(config))
    wandb_run.log_artifact(artifact)


def process_config(cfg: DictConfig) -> Dict:
    if OmegaConf.is_missing(cfg, "seed"):
        cfg.seed = generate_random_seed()

    # model_checkpoint save dir
    if cfg.mode == "dev":
        cfg.callbacks.model_checkpoint.save_dir = Path("/tmp/ckpts").joinpath(
            time_stamp()
        )
        cfg.hparams.max_epochs = 5
    cfg.callbacks.model_checkpoint.save_dir = os.path.join(
        cfg.callbacks.model_checkpoint.save_dir, time_stamp()
    )
    os.makedirs(cfg.callbacks.model_checkpoint.save_dir, exist_ok=True)

    # add any post-processing here
    # e.g. add full path to data_dir

    # print out config
    config = OmegaConf.to_container(cfg, resolve=True)
    pprint(config)

    return config


def init_logger(cfg: DictConfig):
    """
    Init wandb or/and TensorBoard logger

    Args:
        cfg (DictConfig): config object

    Returns:
        Tuple: wandb_run, tb_writer
    """
    wandb_run, tb_writer = None, None
    if cfg.logging_method == "wandb":
        wandb_run = wandb.init(
            project=cfg.wandb_init.project,
            entity=cfg.wandb_init.entity,
            group=cfg.wandb_init.group,
            job_type=cfg.wandb_init.job_type,
            tags=cfg.wandb_init.tags,
            notes=cfg.wandb_init.notes,
            config=OmegaConf.to_container(cfg, resolve=True),
        )
    elif cfg.logging_method == "tensorboard":
        tensorboard_dir = os.path.join(
            cfg.callbacks.model_checkpoint.save_dir, "tensorboard"
        )
        os.makedirs(tensorboard_dir, exist_ok=True)
        tb_writer = SummaryWriter(log_dir=tensorboard_dir)
    return wandb_run, tb_writer



def preprocess_wandb_config(wandb_config: Dict[str, any]) -> Dict[str, any]:
    """
    Adjusts the wandb configuration dictionary to match the expected nested structure.
    In Sweep config parameters, use ":" to separate nested keys.

    Args:
        wandb_config (Dict[str, any]): The flat dictionary configuration from wandb.

    Returns:
        Dict[str, any]: The nested dictionary configuration.
    """
    new_config = {}
    for key, value in wandb_config.items():
        if ":" in key:
            parts = key.split(":")
            d = new_config  # Current level of the dictionary
            for k in parts[:-1]:
                if k not in d:
                    d[k] = {}
                d = d[k]
            d[parts[-1]] = value
    return new_config


@hydra.main(version_base=None, config_path="./conf", config_name="config.yaml")
def main(cfg: DictConfig) -> None:
    # Initialize wandb
    with wandb.init(config=OmegaConf.to_container(cfg, resolve=True)) as run:
        # Convert wandb config to a standard dictionary and then to OmegaConf
        wandb_cfg = OmegaConf.create(
            preprocess_wandb_config(run.config.as_dict())
        )

        # log the config
        logger.info("wandb Config:")
        logger.info(OmegaConf.to_container(wandb_cfg, resolve=True))

        # test if the config is updated
        logger.info("Updated config:")
        logger.info(OmegaConf.to_container(cfg, resolve=True))

        # merge the configurations
        logger.debug(f"{cfg.hparams.train_batch_size=}")
        logger.debug(f"{wandb_cfg.hparams.train_batch_size=}")

        cfg = OmegaConf.merge(cfg, wandb_cfg)

        # -------------------- config template --------------------
        config = process_config(cfg)

        # -------------------- init run logger --------------------
        wandb_run, tb_writer = init_logger(cfg)

        # -------------------- run training --------------------
        seed_everything(cfg.seed)
        if wandb_run:
            log_config_to_wandb(config=config, wandb_run=wandb_run)

        train_model(
            config=config,  # config
            wandb_run=wandb_run,  # logger wandb
            tb_writer=tb_writer,  # logger tensorboard
        )

        # ---------- close the SummaryWriter after training ----------
        if tb_writer:
            tb_writer.close()


# integrate with wandb sweep
def run_sweep():
    sweep_config = {
        "method": "bayes",  # or 'grid', 'random'
        "metric": {
            "name": "valEpoch/avg_epi_node_mcc",  # Define the metric for optimization
            "goal": "maximize",  # `minimize` 或 `maximize`
        },
        "parameters": {"hparams:train_batch_size": {"values": [64, 128]}},
    }
    # Initialize a sweep
    sweep_id = wandb.sweep(sweep_config, project=os.environ["WANDB_PROJECT"])
    wandb.agent(sweep_id, function=main)


if __name__ == "__main__":
    run_sweep()
