import os
from pathlib import Path
from pprint import pformat, pprint
from typing import Dict, Optional, Tuple

import hydra
import wandb
import yaml
from loguru import logger
from omegaconf import DictConfig, OmegaConf
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


def process_config(cfg: DictConfig) -> Tuple[DictConfig, Dict]:
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

    return cfg, config


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
    # -------------------- sweep config --------------------
    # assert the hparams.train_batch_size is updated by sweep
    logger.info(f"raw hydra config:\n{pformat(OmegaConf.to_container(cfg, resolve=True))}")

    # init wandb to get the sweep config
    wandb_run = wandb.init(
        project=cfg.wandb_init.project,
        entity=cfg.wandb_init.entity,
        group=cfg.wandb_init.group,
        job_type=cfg.wandb_init.job_type,
        tags=cfg.wandb_init.tags,
        notes=cfg.wandb_init.notes,
    )
    logger.info(f"wandb config:\n{pformat(dict(wandb.config))}")
    # update cfg with wandb config
    cfg = OmegaConf.merge(cfg, OmegaConf.create(dict(wandb.config)) )
    # make wandb to track the updated config
    wandb.config.update(OmegaConf.to_container(cfg, resolve=True))
    logger.info(f"updated hydra config:\n{pformat(OmegaConf.to_container(cfg, resolve=True))}")
    logger.info(f"updated wandb config:\n{pformat(dict(wandb.config))}")

    # -------------------- wandb or TensorBoard init --------------------
    # wandb_run, tb_writer = init_logger(cfg)
    tb_writer = None

    # -------------------- process config --------------------
    # final config
    cfg, config = process_config(cfg)
    logger.info(f"final config:\n{pformat(config)}")

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


if __name__ == "__main__":
    main()
