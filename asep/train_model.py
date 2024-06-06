import os
import os.path as osp
from pathlib import Path
from pprint import pformat, pprint
from typing import Any, Callable, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import wandb
import yaml
from loguru import logger
from torch import Tensor
from torch.optim import lr_scheduler
from torch.utils.tensorboard import SummaryWriter
from torch_geometric.data import Batch as PygBatch
from torch_geometric.loader import DataLoader as PygDataLoader
from tqdm import tqdm

# custom
from asep.data.asepv1_dataset import AsEPv1Dataset
from asep.data.embedding.handle import EmbeddingHandler
from asep.data.embedding_config import EmbeddingConfig
from asep.model import loss as loss_module
from asep.model.asepv1_model import LinearAbAgIntGAE, PyGAbAgIntGAE
from asep.model.callbacks import EarlyStopper, ModelCheckpoint
from asep.model.metric import (cal_edge_index_bg_metrics,
                               cal_epitope_node_metrics)
from asep.model.utils import generate_random_seed, seed_everything
from asep.utils import time_stamp

# ==================== Configuration ====================
# set precision
torch.set_float32_matmul_precision("high")

ESM2DIM = {
    "esm2_t6_8M_UR50D": 320,
    "esm2_t12_35M_UR50D": 480,
    "esm2_t30_150M_UR50D": 640,
    "esm2_t33_650M_UR50D": 1280,
}

DataRoot = Path.cwd().joinpath("data")


# ==================== Function ====================
# PREPARE: EmbeddingConfig
def create_embedding_config(dataset_config: Dict[str, Any]) -> EmbeddingConfig:
    """
    Create embedding config from config dict

    Args:
        dataset_config (Dict[str, Any]): dataset config

    Returns:
        EmbeddingConfig: embedding config
    """
    # assert dataset_config is a primitive dict
    try:
        assert isinstance(dataset_config, dict)
    except AssertionError as e:
        raise TypeError(f"dataset_config must be a dict, instead got {type(dataset_config)}") from e

    if dataset_config["node_feat_type"] in ("pre_cal", "one_hot"):
        # parse the embedding model for ab and ag
        d = dict(
            node_feat_type=dataset_config["node_feat_type"],
            ab=dataset_config["ab"].copy(),
            ag=dataset_config["ag"].copy(),
        )
        return EmbeddingConfig(**d)

    # otherwise, node_feat_type is custom, need to load function from user specified script
    try:
        d = dataset_config["ab"]["custom_embedding_method_src"]
        ab_func = EmbeddingHandler(
            script_path=d["script_path"], function_name=d["method_name"]
        ).embed
    except Exception as e:
        raise RuntimeError(
            "Error loading custom embedding method for Ab. Please check the script."
        ) from e
    try:
        d = dataset_config["ag"]["custom_embedding_method_src"]
        ag_func = EmbeddingHandler(
            script_path=d["script_path"], function_name=d["method_name"]
        ).embed
    except Exception as e:
        raise RuntimeError(
            "Error loading custom embedding method for Ag. Please check the script."
        ) from e
    updated_dataset_config = dataset_config.copy()
    updated_dataset_config["ab"]["custom_embedding_method"] = ab_func
    updated_dataset_config["ag"]["custom_embedding_method"] = ag_func
    return EmbeddingConfig(**updated_dataset_config)


# PREPARE: dataset
def create_asepv1_dataset(
    root: str = None,
    name: str = None,
    embedding_config: EmbeddingConfig = None,
):
    """
    Create AsEPv1 dataset

    Args:
        root (str, optional): root directory for dataset. Defaults to None.
            if None, set to './data'
        name (str, optional): dataset name. Defaults to None.
            if None, set to 'asep'
        embedding_config (EmbeddingConfig, optional): embedding config. Defaults to None.
            if None, use default embedding config
            {
                'node_feat_type': 'pre_cal',
                'ab': {'embedding_model': 'igfold'},
                'ag': {'embedding_model': 'esm2'},
            }

    Returns:
        AsEPv1Dataset: AsEPv1 dataset
    """
    root = root if root is not None else "./data"
    embedding_config = embedding_config or EmbeddingConfig()
    asepv1_dataset = AsEPv1Dataset(
        root=root, name=name, embedding_config=embedding_config
    )
    return asepv1_dataset


# PREPARE: dataloaders
def create_asepv1_dataloaders(
    asepv1_dataset: AsEPv1Dataset,
    wandb_run: wandb.sdk.wandb_run.Run = None,
    config: Dict[str, Any] = None,
    split_method: str = None,
    split_idx: Dict[str, Tensor] = None,
    return_dataset: bool = False,
    dev: bool = False,
) -> Tuple[PygDataLoader, PygDataLoader, PygDataLoader]:
    """
    Create dataloaders for AsEPv1 dataset

    Args:
        wandb_run (wandb.sdk.wandb_run.Run, optional): wandb run object. Defaults to None.
        config (Dict[str, Any], optional): config dict. Defaults to None.
        return_dataset (bool, optional): return dataset instead of dataloaders. Defaults to False.
        dev (bool, optional): use dev mode. Defaults to False.
        split_idx (Dict[str, Tensor], optional): split index. Defaults to None.
    AsEPv1Dataset kwargs:
        embedding_config (EmbeddingConfig, optional): embedding config. Defaults to None.
            If None, use default EmbeddingConfig, for details, see asep.data.embedding_config.EmbeddingConfig.
        split_method (str, optional): split method. Defaults to None. Either 'epitope_ratio' or 'epitope_group'

    Returns:
        Tuple[PygDataLoader, PygDataLoader, PygDataLoader]: _description_
    """
    # split dataset
    split_idx = split_idx or asepv1_dataset.get_idx_split(split_method=split_method)
    train_set = asepv1_dataset[split_idx["train"]]
    val_set = asepv1_dataset[split_idx["val"]]
    test_set = asepv1_dataset[split_idx["test"]]

    # if dev, only use 100 samples
    if dev:
        train_set = train_set[:170]
        val_set = val_set  # [:100]
        test_set = test_set  # [:100]

    # patch: if test_batch_size is not specified, use val_batch_size, otherwise use test_batch_size
    if ("test_batch_size" not in config["hparams"].keys()) or (
        config["hparams"]["test_batch_size"] is None
    ):
        config["hparams"]["test_batch_size"] = config["hparams"]["val_batch_size"]
        print(
            f"WARNING: test_batch_size is not specified, use val_batch_size instead: {config['hparams']['test_batch_size']}"
        )

    _default_kwargs = dict(follow_batch=["x_b", "x_g"], shuffle=False)
    _default_kwargs_train = dict(
        batch_size=config["hparams"]["train_batch_size"], **_default_kwargs
    )
    _default_kwargs_val = dict(
        batch_size=config["hparams"]["val_batch_size"], **_default_kwargs
    )
    _default_kwargs_test = dict(
        batch_size=config["hparams"]["test_batch_size"], **_default_kwargs
    )
    train_loader = PygDataLoader(train_set, **_default_kwargs_train)
    val_loader = PygDataLoader(val_set, **_default_kwargs_val)
    test_loader = PygDataLoader(test_set, **_default_kwargs_test)

    # save a train-set example to wandb
    if wandb_run is not None:
        artifact = wandb.Artifact(
            name="train_set_example", type="dataset", description="train set example"
        )
        with artifact.new_file("train_set_example.pt", "wb") as f:
            torch.save(train_set[0], f)
        wandb_run.log_artifact(artifact)

    if not return_dataset:
        return train_loader, val_loader, test_loader
    return train_set, val_set, test_set, train_loader, val_loader, test_loader


# PREPARE: model
def create_model(
    config: Dict[str, Any], wandb_run: wandb.sdk.wandb_run.Run = None
) -> nn.Module:
    if config["hparams"]["model_type"] == "linear":
        model_architecture = LinearAbAgIntGAE
    elif config["hparams"]["model_type"] == "graph":
        model_architecture = PyGAbAgIntGAE
    else:
        raise ValueError("model must be either 'linear' or 'graph'")
    # create the model
    model = model_architecture(
        input_ab_dim=config["hparams"]["input_ab_dim"],
        input_ag_dim=config["hparams"]["input_ag_dim"],
        input_ab_act=config["hparams"]["input_ab_act"],
        input_ag_act=config["hparams"]["input_ag_act"],
        dim_list=config["hparams"]["dim_list"],
        act_list=config["hparams"]["act_list"],
        decoder=config["hparams"]["decoder"],
        try_gpu=config["try_gpu"],
    )
    if wandb_run is not None:
        wandb_run.watch(model)
    return model


# PREPARE: loss callables
def generate_loss_callables_from_config(
    loss_config: Dict[str, Any],
) -> List[Tuple[str, Callable, Tensor, Dict[str, Any]]]:
    for loss_name, kwargs in loss_config.items():
        try:
            assert "name" in kwargs.keys() and "w" in kwargs.keys()
        except AssertionError as e:
            raise KeyError("each loss term must contain keys 'name' and 'w'") from e

    loss_callables: List[Tuple[str, Callable, Tensor, Dict[str, Any]]] = [
        (
            name := kwargs.get("name"),  # loss name
            getattr(loss_module, name),  # loss function callable
            torch.tensor(kwargs["w"]),  # loss weight
            kwargs.get("kwargs", {}),  # other kwargs
        )
        for loss_name, kwargs in loss_config.items()
    ]
    return loss_callables


# RUN: feed forward step
def feed_forward_step(
    model: nn.Module,
    batch: PygBatch,
    loss_callables: List[Tuple[str, Callable, Tensor, Dict]],
    is_train: bool,
    edge_cutoff: Optional[int] = None,
    num_edge_cutoff: Optional[int] = None,
) -> Tuple[Tensor, Dict[str, Tensor], Dict[str, Tensor]]:
    """
    Feed forward and calculate loss & metrics for a batch of AbAg graph pairs

    Args:
        batch (Dict): a batch of AbAg graph pairs
        model (nn.Module): model to be trained
        loss_callables (List[Tuple[str, Callable, Tensor, Dict, Dict]]):
            loss_name: (str)        => used as key in outputs
            loss_fn: (Callable)     => the loss function callable
            loss_wt: (Tensor)       => the weight of the loss function for calculating total loss
            loss_fn_kwargs: (Dict)  => kwargs that are constant values

    Returns:
        Dict: outputs from model and loss
    """
    if is_train:
        model.train()
    else:
        model.eval()

    # feed forward
    batch_result = model(batch)
    edge_index_bg_pred = batch_result["edge_index_bg_pred"]
    edge_index_bg_true = batch_result["edge_index_bg_true"]

    # unpack loss_callables
    # loss_items = {}
    batch_loss = None
    for loss_name, loss_fn, loss_w, loss_kwargs in loss_callables:
        if loss_name == "edge_index_bg_rec_loss":
            loss_v = [
                loss_fn(x, y, **loss_kwargs)
                for x, y in zip(edge_index_bg_pred, edge_index_bg_true)
            ]
        elif loss_name == "edge_index_bg_sum_loss":
            loss_v = [loss_fn(x, **loss_kwargs) for x in edge_index_bg_pred]
        # loss_items |= {loss_name: {"v": loss_v, "w": loss_w,}}
        batch_loss = (
            torch.stack(loss_v) * loss_w
            if batch_loss is None
            else batch_loss + torch.stack(loss_v) * loss_w
        )
        # # log loss values
        # print(f"loss name : {loss_name}")
        # print(f"loss weight: {loss_w}")
        # print(f"raw      loss value: mean {torch.stack(loss_v).mean()}, std {torch.stack(loss_v).std()}")
        # print(f"weighted loss value: mean {batch_loss.mean()}, std {batch_loss.std()}")

    # metrics
    batch_edge_index_bg_metrics: List[Dict[str, Tensor]] = [
        cal_edge_index_bg_metrics(x, y, edge_cutoff)
        for x, y in zip(edge_index_bg_pred, edge_index_bg_true)
    ]
    batch_edge_epi_node_metrics: List[Dict[str, Tensor]] = [
        cal_epitope_node_metrics(x, y, num_edge_cutoff)
        for x, y in zip(edge_index_bg_pred, edge_index_bg_true)
    ]

    # average loss and metrics
    avg_loss: Tensor = batch_loss.mean()
    avg_edge_index_bg_metrics: Dict[str, Tensor] = {
        k: torch.stack([d[k] for d in batch_edge_index_bg_metrics]).mean()
        for k in batch_edge_index_bg_metrics[0].keys()
    }
    avg_epi_node_metrics: Dict[str, Tensor] = {
        k: torch.stack([d[k] for d in batch_edge_epi_node_metrics]).mean()
        for k in batch_edge_epi_node_metrics[0].keys()
    }

    return avg_loss, avg_edge_index_bg_metrics, avg_epi_node_metrics


# CALLBACK: on after backward
def on_after_backward(model: nn.Module):
    """Log gradients and model parameters norm after each backward pass"""
    for name, param in model.named_parameters():
        wandb.log(
            {f"gradients/{name}": param.grad.norm(), f"params/{name}": param.norm()}
        )


# CALLBACK: on epoch end
def epoch_end(
    step_outputs: List[Tuple[Tensor, Dict[str, Tensor], Dict[str, Tensor]]]
) -> Tuple[Tensor, Dict[str, Tensor], Dict[str, Tensor]]:
    """
    Args:
        step_outputs (List[Dict[str, Tensor]]):
            shape (n x m)
            `n` element list of outputs from each step (batch)
            each element is a tuple of `m` elements - loss or metrics
    """
    with torch.no_grad():
        # calculate average loss
        avg_epoch_loss = torch.stack([x[0] for x in step_outputs]).mean()
        # calculate average metrics
        avg_epoch_edge_index_bg_metrics = {
            k: torch.stack([x[1][k] for x in step_outputs]).mean()
            for k in step_outputs[0][1].keys()
        }
        avg_epoch_epi_node_metrics = {
            k: torch.stack([x[2][k] for x in step_outputs]).mean()
            for k in step_outputs[0][2].keys()
        }
        return (
            avg_epoch_loss,
            avg_epoch_edge_index_bg_metrics,
            avg_epoch_epi_node_metrics,
        )


# TRAIN helper - learning rate scheduler
def exec_lr_scheduler(
    ck_lr_scheduler: Optional[torch.optim.lr_scheduler._LRScheduler],
    config: Dict[str, Any],
    val_epoch_metrics: Dict[str, Tensor],
) -> None:
    if ck_lr_scheduler is not None:
        if config["callbacks"]["lr_scheduler"]["step"] is not None:
            # reduce learning rate on plateau
            if config["callbacks"]["lr_scheduler"]["name"] == "ReduceLROnPlateau":
                ck_lr_scheduler.step(
                    metrics=val_epoch_metrics[
                        config["callbacks"]["lr_scheduler"]["step"]["metrics"]
                    ]
                )
        else:
            ck_lr_scheduler.step()


# MAIN function
def train_model(
    config: Dict,
    wandb_run: Optional[wandb.sdk.wandb_run.Run] = None,
    tb_writer: Optional[SummaryWriter] = None,
):
    """
    Args:
        config: (Dict) config dict, contains all hyperparameters
        wandb_run: (wandb.sdk.wandb_run.Run) wandb run object
    """
    # for debugging purpose
    logger.debug(f"config:\n{pformat(config)}")
    # set num threads
    torch.set_num_threads(config["num_threads"])
    # --------------------
    # Datasets
    # --------------------
    dev = config.get("mode") == "dev"
    embedding_config = create_embedding_config(dataset_config=config["dataset"])
    asepv1_dataset = create_asepv1_dataset(
        root=config["dataset"]["root"],
        name=config["dataset"]["name"],
        embedding_config=embedding_config,
    )
    train_loader, val_loader, test_loader = create_asepv1_dataloaders(
        asepv1_dataset=asepv1_dataset,
        wandb_run=wandb_run,
        config=config,
        split_idx=config["dataset"]["split_idx"],
        split_method=config["dataset"]["split_method"],
        dev=dev,
    )
    print(f"{len(train_loader.dataset)=}")
    print(f"{len(val_loader.dataset)=}")
    print(f"{len(test_loader.dataset)=}")

    # --------------------
    # Model,
    # Loss, Optimizer
    # Callbacks
    # --------------------
    model = create_model(config=config, wandb_run=wandb_run)

    # log the model architecture
    if wandb_run is not None:
        wandb_run.watch(model)
    # print out the model architecture
    print(model)

    # loss and optimizer
    loss_callables = generate_loss_callables_from_config(config["loss"])
    optimizer = getattr(torch.optim, config["optimizer"]["name"])(
        params=model.parameters(),
        **config["optimizer"]["params"],
    )

    # callback objects
    ck_early_stop = (
        EarlyStopper(**config["callbacks"]["early_stopping"])
        if config["callbacks"]["early_stopping"] is not None
        else None
    )
    # add a model ckpt to record best k models for node prediction
    ck_model_ckpt = (
        ModelCheckpoint(**config["callbacks"]["model_checkpoint"])
        if config["callbacks"]["model_checkpoint"] is not None
        else None
    )
    # add a model ckpt to record best k models for edge prediction
    ck_model_ckpt_edge = (
        ModelCheckpoint(**config["callbacks"]["model_checkpoint_edge"])
        if config["callbacks"]["model_checkpoint_edge"] is not None
        else None
    )
    # add a learning rate scheduler
    ck_lr_scheduler = (
        getattr(lr_scheduler, config["callbacks"]["lr_scheduler"]["name"])(
            optimizer=optimizer, **config["callbacks"]["lr_scheduler"]["kwargs"]
        )
        if config["callbacks"]["lr_scheduler"] is not None
        else None
    )

    # --------------------
    # Train Val Test Loop
    # --------------------
    train_step_outputs, val_step_outputs, test_step_outputs = [], [], []
    current_epoch_idx, current_val_metric = None, None
    for epoch_idx in range(config["hparams"]["max_epochs"]):
        current_epoch_idx = epoch_idx
        print(f"Epoch {epoch_idx + 1}/{config['hparams']['max_epochs']}")
        # --------------------
        # training
        # --------------------
        model.train()
        _default_kwargs = dict(unit="GraphPairBatch", ncols=100)
        for batch_idx, batch in tqdm(
            enumerate(train_loader),
            total=len(train_loader),
            desc=f"{'train':<5}",
            **_default_kwargs,
        ):
            optimizer.zero_grad()
            # feed forward (batch)
            avg_loss, avg_edge_index_bg_metrics, avg_epi_node_metrics = (
                feed_forward_step(
                    model=model,
                    batch=batch,
                    loss_callables=loss_callables,
                    is_train=True,
                    edge_cutoff=config["hparams"]["edge_cutoff"],
                    num_edge_cutoff=config["hparams"]["num_edge_cutoff"],
                )
            )
            d = {
                "trainStep/avg_loss": avg_loss,
                "trainStep/avg_edge_index_bg_auprc": avg_edge_index_bg_metrics["auprc"],
                "trainStep/avg_edge_index_bg_mcc": avg_edge_index_bg_metrics["mcc"],
                "trainStep/avg_edge_index_bg_tn": avg_edge_index_bg_metrics["tn"],
                "trainStep/avg_edge_index_bg_fp": avg_edge_index_bg_metrics["fp"],
                "trainStep/avg_edge_index_bg_fn": avg_edge_index_bg_metrics["fn"],
                "trainStep/avg_edge_index_bg_tp": avg_edge_index_bg_metrics["tp"],
                "trainStep/avg_epi_node_auprc": avg_epi_node_metrics["auprc"],
                "trainStep/avg_epi_node_mcc": avg_epi_node_metrics["mcc"],
                "trainStep/avg_epi_node_tn": avg_epi_node_metrics["tn"],
                "trainStep/avg_epi_node_fp": avg_epi_node_metrics["fp"],
                "trainStep/avg_epi_node_fn": avg_epi_node_metrics["fn"],
                "trainStep/avg_epi_node_tp": avg_epi_node_metrics["tp"],
            }
            if wandb_run is not None:
                wandb_run.log(d)
            elif tb_writer is not None:
                tb_writer.add_scalars(
                    main_tag="train",
                    tag_scalar_dict=d,
                    global_step=epoch_idx * len(train_loader) + batch_idx,
                )
            # append to step outputs
            train_step_outputs.append(
                (avg_loss, avg_edge_index_bg_metrics, avg_epi_node_metrics)
            )
            # calculate gradients
            avg_loss.backward()
            # update parameters
            optimizer.step()

        # epoch end: calculate epoch average loss and metric
        avg_epoch_loss, avg_epoch_edge_index_bg_metrics, avg_epoch_epi_node_metrics = (
            epoch_end(step_outputs=train_step_outputs)
        )
        train_epoch_metrics = {
            "trainEpoch/avg_loss": avg_epoch_loss,
            "trainEpoch/avg_edge_index_bg_auprc": avg_epoch_edge_index_bg_metrics[
                "auprc"
            ],
            "trainEpoch/avg_edge_index_bg_mcc": avg_epoch_edge_index_bg_metrics["mcc"],
            "trainEpoch/avg_edge_index_bg_tn": avg_epoch_edge_index_bg_metrics["tn"],
            "trainEpoch/avg_edge_index_bg_fp": avg_epoch_edge_index_bg_metrics["fp"],
            "trainEpoch/avg_edge_index_bg_fn": avg_epoch_edge_index_bg_metrics["fn"],
            "trainEpoch/avg_edge_index_bg_tp": avg_epoch_edge_index_bg_metrics["tp"],
            "trainEpoch/avg_epi_node_auprc": avg_epoch_epi_node_metrics["auprc"],
            "trainEpoch/avg_epi_node_mcc": avg_epoch_epi_node_metrics["mcc"],
            "trainEpoch/avg_epi_node_tn": avg_epoch_epi_node_metrics["tn"],
            "trainEpoch/avg_epi_node_fp": avg_epoch_epi_node_metrics["fp"],
            "trainEpoch/avg_epi_node_fn": avg_epoch_epi_node_metrics["fn"],
            "trainEpoch/avg_epi_node_tp": avg_epoch_epi_node_metrics["tp"],
            "epoch": epoch_idx + 1,
        }
        if wandb_run is not None:
            wandb_run.log(train_epoch_metrics)
        elif tb_writer is not None:
            tb_writer.add_scalars(
                main_tag="train",
                tag_scalar_dict=train_epoch_metrics,
                global_step=epoch_idx,
            )
        pprint(train_epoch_metrics)
        # free memory
        train_step_outputs.clear()

        # --------------------
        # validation
        # --------------------
        model.eval()
        for batch_idx, batch in tqdm(
            enumerate(val_loader),
            total=len(val_loader),
            desc=f"{'val':<5}",
            unit="GraphPairBatch",
            ncols=100,
        ):
            # feed forward (batch)
            avg_loss, avg_edge_index_bg_metrics, avg_epi_node_metrics = (
                feed_forward_step(
                    model=model,
                    batch=batch,
                    loss_callables=loss_callables,
                    is_train=False,
                    edge_cutoff=config["hparams"]["edge_cutoff"],
                    num_edge_cutoff=config["hparams"]["num_edge_cutoff"],
                )
            )
            d = {
                "valStep/avg_loss": avg_loss,
                "valStep/avg_edge_index_bg_auprc": avg_edge_index_bg_metrics["auprc"],
                "valStep/avg_edge_index_bg_mcc": avg_edge_index_bg_metrics["mcc"],
                "valStep/avg_edge_index_bg_tn": avg_edge_index_bg_metrics["tn"],
                "valStep/avg_edge_index_bg_fp": avg_edge_index_bg_metrics["fp"],
                "valStep/avg_edge_index_bg_fn": avg_edge_index_bg_metrics["fn"],
                "valStep/avg_edge_index_bg_tp": avg_edge_index_bg_metrics["tp"],
                "valStep/avg_epi_node_auprc": avg_epi_node_metrics["auprc"],
                "valStep/avg_epi_node_mcc": avg_epi_node_metrics["mcc"],
                "valStep/avg_epi_node_tn": avg_epi_node_metrics["tn"],
                "valStep/avg_epi_node_fp": avg_epi_node_metrics["fp"],
                "valStep/avg_epi_node_fn": avg_epi_node_metrics["fn"],
                "valStep/avg_epi_node_tp": avg_epi_node_metrics["tp"],
            }
            if wandb_run is not None:
                wandb_run.log(d)
            elif tb_writer is not None:
                tb_writer.add_scalars(
                    main_tag="val",
                    tag_scalar_dict=d,
                    global_step=epoch_idx * len(val_loader) + batch_idx,
                )
            # append to step outputs
            val_step_outputs.append(
                (avg_loss, avg_edge_index_bg_metrics, avg_epi_node_metrics)
            )

        # epoch end: calculate epoch average loss and metric
        avg_epoch_loss, avg_epoch_edge_index_bg_metrics, avg_epoch_epi_node_metrics = (
            epoch_end(step_outputs=val_step_outputs)
        )
        val_epoch_metrics = {
            "valEpoch/avg_loss": avg_epoch_loss,
            "valEpoch/avg_edge_index_bg_auprc": avg_epoch_edge_index_bg_metrics[
                "auprc"
            ],
            "valEpoch/avg_edge_index_bg_mcc": avg_epoch_edge_index_bg_metrics["mcc"],
            "valEpoch/avg_edge_index_bg_tn": avg_epoch_edge_index_bg_metrics["tn"],
            "valEpoch/avg_edge_index_bg_fp": avg_epoch_edge_index_bg_metrics["fp"],
            "valEpoch/avg_edge_index_bg_fn": avg_epoch_edge_index_bg_metrics["fn"],
            "valEpoch/avg_edge_index_bg_tp": avg_epoch_edge_index_bg_metrics["tp"],
            "valEpoch/avg_epi_node_auprc": avg_epoch_epi_node_metrics["auprc"],
            "valEpoch/avg_epi_node_mcc": avg_epoch_epi_node_metrics["mcc"],
            "valEpoch/avg_epi_node_tn": avg_epoch_epi_node_metrics["tn"],
            "valEpoch/avg_epi_node_fp": avg_epoch_epi_node_metrics["fp"],
            "valEpoch/avg_epi_node_fn": avg_epoch_epi_node_metrics["fn"],
            "valEpoch/avg_epi_node_tp": avg_epoch_epi_node_metrics["tp"],
            "epoch": epoch_idx + 1,
        }
        if wandb_run is not None:
            wandb_run.log(val_epoch_metrics)
        elif tb_writer is not None:
            tb_writer.add_scalars(
                main_tag="val",
                tag_scalar_dict=val_epoch_metrics,
                global_step=epoch_idx,
            )
        pprint(val_epoch_metrics)
        # free memory
        val_step_outputs.clear()

        # --------------------
        # testing
        # --------------------
        model.eval()
        for batch_idx, batch in tqdm(
            enumerate(test_loader),
            total=len(test_loader),
            desc=f"{'test':<5}",
            unit="GraphPairBatch",
            ncols=100,
        ):
            # feed forward (batch)
            avg_loss, avg_edge_index_bg_metrics, avg_epi_node_metrics = (
                feed_forward_step(
                    model=model,
                    batch=batch,
                    loss_callables=loss_callables,
                    is_train=False,
                    edge_cutoff=config["hparams"]["edge_cutoff"],
                    num_edge_cutoff=config["hparams"]["num_edge_cutoff"],
                )
            )
            d = {
                "testStep/avg_loss": avg_loss,
                "testStep/avg_edge_index_bg_auprc": avg_edge_index_bg_metrics["auprc"],
                "testStep/avg_edge_index_bg_mcc": avg_edge_index_bg_metrics["mcc"],
                "testStep/avg_edge_index_bg_tn": avg_edge_index_bg_metrics["tn"],
                "testStep/avg_edge_index_bg_fp": avg_edge_index_bg_metrics["fp"],
                "testStep/avg_edge_index_bg_fn": avg_edge_index_bg_metrics["fn"],
                "testStep/avg_edge_index_bg_tp": avg_edge_index_bg_metrics["tp"],
                "testStep/avg_epi_node_auprc": avg_epi_node_metrics["auprc"],
                "testStep/avg_epi_node_mcc": avg_epi_node_metrics["mcc"],
                "testStep/avg_epi_node_tn": avg_epi_node_metrics["tn"],
                "testStep/avg_epi_node_fp": avg_epi_node_metrics["fp"],
                "testStep/avg_epi_node_fn": avg_epi_node_metrics["fn"],
                "testStep/avg_epi_node_tp": avg_epi_node_metrics["tp"],
            }
            if wandb_run is not None:
                wandb_run.log(d)
            elif tb_writer is not None:
                tb_writer.add_scalars(
                    main_tag="test",
                    tag_scalar_dict=d,
                    global_step=epoch_idx * len(test_loader) + batch_idx,
                )
            # append to step outputs
            test_step_outputs.append(
                (avg_loss, avg_edge_index_bg_metrics, avg_epi_node_metrics)
            )

        # epoch end: calculate epoch average loss and metric
        avg_epoch_loss, avg_epoch_edge_index_bg_metrics, avg_epoch_epi_node_metrics = (
            epoch_end(step_outputs=test_step_outputs)
        )
        test_epoch_metrics = {
            "testEpoch/avg_loss": avg_epoch_loss,
            "testEpoch/avg_edge_index_bg_auprc": avg_epoch_edge_index_bg_metrics[
                "auprc"
            ],
            "testEpoch/avg_edge_index_bg_mcc": avg_epoch_edge_index_bg_metrics["mcc"],
            "testEpoch/avg_edge_index_bg_tn": avg_epoch_edge_index_bg_metrics["tn"],
            "testEpoch/avg_edge_index_bg_fp": avg_epoch_edge_index_bg_metrics["fp"],
            "testEpoch/avg_edge_index_bg_fn": avg_epoch_edge_index_bg_metrics["fn"],
            "testEpoch/avg_edge_index_bg_tp": avg_epoch_edge_index_bg_metrics["tp"],
            "testEpoch/avg_epi_node_auprc": avg_epoch_epi_node_metrics["auprc"],
            "testEpoch/avg_epi_node_mcc": avg_epoch_epi_node_metrics["mcc"],
            "testEpoch/avg_epi_node_tn": avg_epoch_epi_node_metrics["tn"],
            "testEpoch/avg_epi_node_fp": avg_epoch_epi_node_metrics["fp"],
            "testEpoch/avg_epi_node_fn": avg_epoch_epi_node_metrics["fn"],
            "testEpoch/avg_epi_node_tp": avg_epoch_epi_node_metrics["tp"],
            "epoch": epoch_idx + 1,
        }
        if wandb_run is not None:
            wandb_run.log(test_epoch_metrics)
        elif tb_writer is not None:
            tb_writer.add_scalars(
                main_tag="test",
                tag_scalar_dict=test_epoch_metrics,
                global_step=epoch_idx,
            )
        pprint(test_epoch_metrics)
        # free memory
        test_step_outputs.clear()

        # --------------------
        # callbacks
        # --------------------
        # model checkpoint
        if ck_model_ckpt is not None:
            ck_model_ckpt.step(
                metrics=val_epoch_metrics,
                model=model,
                epoch=epoch_idx,
                optimizer=optimizer,
            )
        # model checkpoint edge level
        if ck_model_ckpt_edge is not None:
            ck_model_ckpt_edge.step(
                metrics=val_epoch_metrics,
                model=model,
                epoch=epoch_idx,
                optimizer=optimizer,
            )
        # early stopping
        if (ck_early_stop is not None) and (
            ck_early_stop.early_stop(epoch=epoch_idx, metrics=val_epoch_metrics)
        ):
            print(f"Early stopping at epoch {epoch_idx}")
            break

        exec_lr_scheduler(ck_lr_scheduler, config, val_epoch_metrics=val_epoch_metrics)
        for param_group in optimizer.param_groups:
            print(f"Epoch {epoch_idx+1}, Learning Rate: {param_group['lr']:.6f}")

    # --------------------
    # save models (train
    # finished or early
    # stopped)
    # --------------------
    # save the last model
    ck_model_ckpt.save_last(
        epoch=current_epoch_idx,
        model=model,
        optimizer=optimizer,
        metric_value=val_epoch_metrics[
            config["callbacks"]["model_checkpoint"]["metric_name"]
        ],
        upload=True,  # upload to wandb
        wandb_run=wandb_run,  # wandb run object
    )

    # save best k models based on provided metric_name
    ck_model_ckpt.save_best_k(keep_interim=config["keep_interim_ckpts"])
    ck_model_ckpt_edge.save_best_k(keep_interim=config["keep_interim_ckpts"])
    # upload models to wandb artifacts
    if wandb_run is not None:
        ck_model_ckpt.upload_best_k_to_wandb(wandb_run=wandb_run)
        ck_model_ckpt_edge.upload_best_k_to_wandb(wandb_run=wandb_run, suffix='-edge')
    # --------------------
    # testing
    # --------------------
    # load the best model
    ckpt_data = ck_model_ckpt.load_best()
    model.load_state_dict(ckpt_data["model_state_dict"])

    # test time
    model.eval()
    test_step_outputs = []
    with torch.no_grad():
        for batch_idx, batch in tqdm(
            enumerate(test_loader),
            total=len(test_loader),
            desc=f"{'testF':<5}",
            unit="graph",
            ncols=100,
        ):
            # feed forward (batch)
            avg_loss, avg_edge_index_bg_metrics, avg_epi_node_metrics = (
                feed_forward_step(
                    model=model,
                    batch=batch,
                    loss_callables=loss_callables,
                    is_train=False,
                    edge_cutoff=config["hparams"]["edge_cutoff"],
                    num_edge_cutoff=config["hparams"]["num_edge_cutoff"],
                )
            )
            d = {
                "testStepFinal/avg_loss": avg_loss,
                "testStepFinal/avg_edge_index_bg_auprc": avg_edge_index_bg_metrics[
                    "auprc"
                ],
                "testStepFinal/avg_edge_index_bg_mcc": avg_edge_index_bg_metrics["mcc"],
                "testStepFinal/avg_edge_index_bg_tn": avg_edge_index_bg_metrics["tn"],
                "testStepFinal/avg_edge_index_bg_fp": avg_edge_index_bg_metrics["fp"],
                "testStepFinal/avg_edge_index_bg_fn": avg_edge_index_bg_metrics["fn"],
                "testStepFinal/avg_edge_index_bg_tp": avg_edge_index_bg_metrics["tp"],
                "testStepFinal/avg_epi_node_auprc": avg_epi_node_metrics["auprc"],
                "testStepFinal/avg_epi_node_mcc": avg_epi_node_metrics["mcc"],
                "testStepFinal/avg_epi_node_tn": avg_epi_node_metrics["tn"],
                "testStepFinal/avg_epi_node_fp": avg_epi_node_metrics["fp"],
                "testStepFinal/avg_epi_node_fn": avg_epi_node_metrics["fn"],
                "testStepFinal/avg_epi_node_tp": avg_epi_node_metrics["tp"],
            }
            if wandb_run is not None:
                wandb_run.log(d)
            elif tb_writer is not None:
                tb_writer.add_scalars(
                    main_tag="test",
                    tag_scalar_dict=d,
                    global_step=epoch_idx * len(test_loader) + batch_idx,
                )
            # append to step outputs
            test_step_outputs.append(
                (avg_loss, avg_edge_index_bg_metrics, avg_epi_node_metrics)
            )

        # epoch end: calculate epoch average loss and metric
        avg_epoch_loss, avg_epoch_edge_index_bg_metrics, avg_epoch_epi_node_metrics = (
            epoch_end(step_outputs=test_step_outputs)
        )
        test_epoch_metrics = {
            "testEpochFinal/avg_loss": avg_epoch_loss,
            "testEpochFinal/avg_edge_index_bg_auprc": avg_epoch_edge_index_bg_metrics[
                "auprc"
            ],
            "testEpochFinal/avg_edge_index_bg_mcc": avg_epoch_edge_index_bg_metrics[
                "mcc"
            ],
            "testEpochFinal/avg_edge_index_bg_tn": avg_epoch_edge_index_bg_metrics[
                "tn"
            ],
            "testEpochFinal/avg_edge_index_bg_fp": avg_epoch_edge_index_bg_metrics[
                "fp"
            ],
            "testEpochFinal/avg_edge_index_bg_fn": avg_epoch_edge_index_bg_metrics[
                "fn"
            ],
            "testEpochFinal/avg_edge_index_bg_tp": avg_epoch_edge_index_bg_metrics[
                "tp"
            ],
            "testEpochFinal/avg_epi_node_auprc": avg_epoch_epi_node_metrics["auprc"],
            "testEpochFinal/avg_epi_node_mcc": avg_epoch_epi_node_metrics["mcc"],
            "testEpochFinal/avg_epi_node_tn": avg_epoch_epi_node_metrics["tn"],
            "testEpochFinal/avg_epi_node_fp": avg_epoch_epi_node_metrics["fp"],
            "testEpochFinal/avg_epi_node_fn": avg_epoch_epi_node_metrics["fn"],
            "testEpochFinal/avg_epi_node_tp": avg_epoch_epi_node_metrics["tp"],
        }
        if wandb_run is not None:
            wandb_run.log(test_epoch_metrics)
        elif tb_writer is not None:
            tb_writer.add_scalars(
                main_tag="test",
                tag_scalar_dict=test_epoch_metrics,
                global_step=epoch_idx,
            )
        # free memory
        test_step_outputs.clear()
