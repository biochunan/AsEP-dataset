"""
Evaluate the performance of pretrained model on the WALLE test set.
"""

"""
Running inference code using the isolated preprocess module
"""
import argparse
# cli
import json
import textwrap
from argparse import Namespace
from pathlib import Path
from typing import Any, Callable, Dict, Optional, Tuple, Union

import numpy as np
# torch tools
import torch
import torch.nn as nn
from loguru import logger
from omegaconf import OmegaConf
from pandas import DataFrame
# metrics
from sklearn.metrics import (confusion_matrix, f1_score, matthews_corrcoef,
                             precision_score, recall_score, roc_auc_score)
# pytorch geometric tools
from torch_geometric.loader import DataLoader as PygDataLoader
from tqdm import tqdm

# asep
from asep.data.asepv1_dataset import AsEPv1Dataset
from asep.preprocess import create_pair_graph_data, pyg_data_to_batch
from asep.train_model import (create_asepv1_dataloaders, create_asepv1_dataset,
                              create_embedding_config, create_model)
from asep.utils import (convert_to_json_serializable, dec_log_func, log_args,
                        log_dict)

# ==================== Configuration ====================
# Types
PathLike = Union[str, Path]
PDBDataFrame = DataFrame
AllAtomDataFrame = DataFrame
AdjMatrix = np.ndarray
BinaryMask = np.ndarray

# Paths
BASE = Path(__file__).resolve().parent  # => walle

# baseline model
ModelCheckpoint = BASE / "assets" / "ckpts" / "woven-rain-26" / "rank_0-epoch_185.pt"
ModelConfig = BASE / "assets" / "config" / "woven-rain-26.yaml"


# ==================== Function ====================
# helper: save results
def _save_json(args: Namespace, results: Dict[str, Any]) -> None:
    n = results["job_id"]
    outdir = args.outdir / n
    outdir.mkdir(parents=True, exist_ok=True)
    with open(fp := (outdir / f"{n}.json"), "w") as f:
        json.dump(results, f, default=convert_to_json_serializable)
    logger.info(f"Results saved to {fp.as_posix()}")


# config
def parse_config(config_path: Path) -> Dict:
    return OmegaConf.load(config_path)


# evaluate: core function
def eval_core(y_pred: np.ndarray, y_true: np.ndarray) -> Dict[str, Any]:
    # assert y_true and y_pred have the same shape
    assert y_true.shape == y_pred.shape

    # calculate confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.flatten()

    mcc = matthews_corrcoef(y_true=y_true, y_pred=y_pred)
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    # in case of all 0s or 1s
    if len(np.unique(y_true)) == 1:
        logger.warning(f"y_true is all {np.unique(y_true)[0]}, setting roc_auc to 0.")
        roc_auc = 0.0
    else:
        roc_auc = roc_auc_score(y_true, y_pred)

    metrics = {
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "mcc": mcc,
        "roc_auc": roc_auc,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }
    return metrics


# evaluate: epitope residue prediction
def dec_evaluate_epitope_residue_prediction(func: Callable) -> Callable:
    """
    decorator to evaluate epitope residue prediction

    Args:
        func (Callable): a function that returns a metric dict

    Returns:
        Callable: a function that returns a metric dict
    """

    def evaluate_epitope_residue_prediction(
        y_pred: np.ndarray, y_true: np.ndarray
    ) -> Dict[str, Dict[str, Any]]:
        metrics = {"metrics_epitope_pred": func(y_pred=y_pred, y_true=y_true)}
        # customization
        logger.info(f"----- metrics: epitope prediction -----")
        log_dict(metrics["metrics_epitope_pred"])
        return metrics

    return evaluate_epitope_residue_prediction


# evaluate: validate bipartite graph edge reconstruction
def dec_evaluate_bipartite_graph_reconstruction(func: Callable) -> Callable:
    def evaluate_bipartite_graph_reconstruction(
        edge_index_bg_pred: np.ndarray, edge_index_bg_true: np.ndarray
    ) -> Dict[str, Dict[str, Any]]:
        metrics = {
            "metrics_bi_adj": func(y_pred=edge_index_bg_pred, y_true=edge_index_bg_true)
        }
        # customization
        logger.info(f"----- metrics: bipartite graph reconstruction -----")
        log_dict(metrics["metrics_bi_adj"])
        return metrics

    return evaluate_bipartite_graph_reconstruction


# main: load pretrained model
@dec_log_func
def from_pretrained_model(
    model_checkpoint: PathLike, model_config: Dict[str, Any]
) -> Tuple[nn.Module, Dict[str, Any]]:
    # build model
    model = create_model(config := parse_config(model_config))

    checkpoint = torch.load(model_checkpoint, map_location="cpu")
    model.load_state_dict(checkpoint["model_state_dict"])

    return model, config


# main: inference on a new AbDb structure
@dec_log_func
def inference_abdb_structure(num_edge_cutoff: Optional[float | int] = None):
    args = cli()
    log_args(args)

    # --------------------------------------------------------------------------
    # Preprocessing
    # process input to pair graph data in a batch for inference
    # --------------------------------------------------------------------------
    pair_data = create_pair_graph_data(args)
    batch = pyg_data_to_batch(pair_data)

    # --------------------------------------------------------------------------
    # Inference
    # load model from pretrained checkpoint and run inference
    # --------------------------------------------------------------------------
    model, config = from_pretrained_model(
        model_checkpoint=args.pretrained_model_ckpt,
        model_config=args.pretrained_model_config,
    )

    device = torch.device(
        "cuda" if torch.cuda.is_available() and not args.cpu else "cpu"
    )
    model.to(device)
    model.eval()

    # inference
    with torch.no_grad():
        pred = model(batch)

    results = {
        "job_id": args.job_id,
        "edge_index_bg_pred": (p := pred["edge_index_bg_pred"][0]),
        "epitope_pred": p.sum(axis=0).detach().cpu().numpy() > num_edge_cutoff,
    }

    # --------------------------------------------------------------------------
    # Evaluate (--is_complex only)
    # 1. bipartite graph edge reconstruction;
    # 2. epitope node prediction
    # --------------------------------------------------------------------------
    if args.is_complex:
        # pred and true edge_index_bg
        t = pred["edge_index_bg_true"][0].clone().detach().cpu().numpy()
        p = pred["edge_index_bg_pred"][0].clone().detach().cpu().numpy()

        # print t, p shape
        logger.info(f"edge_index_bg_true: {t.shape}")
        logger.info(f"edge_index_bg_pred: {p.shape}")

        # evaluate 1. epitope node prediction
        y_true = t.sum(axis=0) > 0
        y_pred = p.sum(axis=0) > num_edge_cutoff
        node_metrics = dec_evaluate_epitope_residue_prediction(eval_core)(
            y_pred=y_pred, y_true=y_true
        )
        results.update(node_metrics)

        # evaluate 2. bipartite graph edge reconstruction;
        try:
            edge_cutoff = config["hparams"]["edge_cutoff"]
        except KeyError as e:
            logger.warning(f"KeyError: {e}, setting edge_cutoff to 0.5")
            edge_cutoff = 0.5
        p_binary = (p > edge_cutoff).astype(int)
        bi_adj_metrics = dec_evaluate_bipartite_graph_reconstruction(eval_core)(
            edge_index_bg_pred=p_binary.flatten(), edge_index_bg_true=t.flatten()
        )
        results.update(bi_adj_metrics)

    # save results
    _save_json(args, results)


# main: load asepv1 dataset
@dec_log_func
def make_asepv1_dataloaders(
    config: Dict[str, Any],
    return_dataset: bool = False,
    split_method: Optional[str] = None,
) -> Tuple[PygDataLoader, PygDataLoader, PygDataLoader]:
    """
    Make dataloaders for the AsEPv1 dataset

    Args:
        root (str, optional): Root path to the dataset. Defaults to None.
        train_batch_size (int, optional): Batch size for training. Defaults to 32.
        val_batch_size (int, optional): Batch size for validation. Defaults to 32.
        test_batch_size (int, optional): Batch size for testing. Defaults to 1.
        return_dataset (bool, optional): Whether to return the dataset. Defaults to False.
        dev (bool, optional): Whether to use dev mode. Defaults to False.

    Returns:
        Tuple[PygDataLoader, PygDataLoader, PygDataLoader]: train, val, test dataloaders
    """

    # [x]TODO: change the args to this,
    # refer to: asep/train_model.py#L406 for embedding_config definition
    # refer to: asep/train_model.py#L412 for dataloader creation, this requires two functions:
    # (1) create_asepv1_dataset; (2) create_asepv1_dataloaders
    # both are defined in asep/train_model.py
    embedding_config = create_embedding_config(dataset_config=config["dataset"])
    asepv1_dataset = create_asepv1_dataset(
        root=config["dataset"]["root"],
        name=config["dataset"]["name"],
        embedding_config=embedding_config,
    )

    train_loader, val_loader, test_loader = create_asepv1_dataloaders(
        asepv1_dataset=asepv1_dataset,
        wandb_run=None,
        config=config,
        split_idx=config["dataset"]["split_idx"],
        split_method=split_method or config["dataset"]["split_method"] or "epitope_ratio",
        dev=False,
    )
    if not return_dataset:
        return train_loader, val_loader, test_loader
    split_idx = config["dataset"]["split_idx"] or asepv1_dataset.get_idx_split(
        split_method=config["dataset"]["split_method"]
    )
    train_set = asepv1_dataset[split_idx["train"]]
    val_set = asepv1_dataset[split_idx["val"]]
    test_set = asepv1_dataset[split_idx["test"]]
    return train_set, val_set, test_set, train_loader, val_loader, test_loader


# input
def cli() -> Namespace:
    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawTextHelpFormatter,
        description="Inference from preprocess module",
        epilog=textwrap.dedent(
            text="""
            python evaluate_on_walle.py \\
                -o ./metrics \\
                --pretrained_model_config assets/config/woven-rain-26.yaml \\
                --pretrained_model_ckpt assets/ckpts/woven-rain-26/rank_0-epoch_185.pt
            python evaluate_on_walle.py \\
                -o ./metrics_val \\
                --set val \\
                --log_level DEBUG \\
                --pretrained_model_config assets/config/woven-rain-26.yaml \\
                --pretrained_model_ckpt assets/ckpts/woven-rain-26/rank_0-epoch_185.pt
            python evaluate_on_walle.py \\
                -o ./metrics_train \\
                --set train \\
                --pretrained_model_config assets/config/woven-rain-26.yaml \\
                --pretrained_model_ckpt assets/ckpts/woven-rain-26/rank_0-epoch_185.pt
            """
        ),
    )
    # outdir, default to cwd.
    parser.add_argument(
        "-o",
        "--outdir",
        type=Path,
        required=False,
        default=Path.cwd(),
        help="output directory, default: cwd.",
    )
    # pretrained model
    parser.add_argument(
        "-c",
        "--pretrained_model_config",
        type=Path,
        required=True,
        help="path to the pretrained model config",
    )
    parser.add_argument(
        "-m",
        "--pretrained_model_ckpt",
        type=Path,
        required=True,
        help="path to the pretrained model checkpoint",
    )
    # add a flag to set loguru logger level to DEBUG / INFO / WARNING / ERROR / CRITICAL
    parser.add_argument(
        "--log_level",
        type=str,
        required=False,
        default="INFO",
        help="log level, default: INFO",
    )
    # which set to evaluate on, default to test set
    parser.add_argument(
        "--set",
        type=str,
        required=False,
        default="test",
        help="which set to evaluate on, default: test set",
    )
    # try gpu
    parser.add_argument(
        "--cpu",
        action="store_true",
        help="use cpu, default: True",
    )
    args = parser.parse_args()
    return args


# ==================== Main ====================
if __name__ == "__main__":
    args = cli()

    # ----------------------------------------
    # main: load model and config
    # ----------------------------------------
    model, config = from_pretrained_model(
        model_checkpoint=args.pretrained_model_ckpt,
        model_config=args.pretrained_model_config,
    )
    device = torch.device(
        "cuda" if torch.cuda.is_available() and not args.cpu else "cpu"
    )
    model.to(device)
    model.eval()

    # param
    num_edge_cutoff = 3
    try:
        num_edge_cutoff = config["hparams"]["num_edge_cutoff"]
        logger.info(f"num_edge_cutoff: set to {num_edge_cutoff}")
    except KeyError as e:
        logger.warning(
            f"KeyError: {e},\n num_edge_cutoff not found in the configuration, setting it to 3"
        )

    # ----------------------------------------
    # load dataset
    # ----------------------------------------
    # [x] TODO: replace to accept args from config; refer to the asep/train_model.py#L402-L418
    # [x] TODO:correct the code for BLOSUM62/one-hot/esm2only encoding use case, currently it only accepts igfold/esm2 for ab ag encoding
    # turn config into a primitive dictionary
    config = OmegaConf.to_container(config, resolve=True)
    # force *_batch_size to 1
    config['hparams']['train_batch_size'] = 1
    config['hparams']['val_batch_size'] = 1
    config['hparams']['test_batch_size'] = 1
    print(f'{config["dataset"]["split_method"]=}')
    (train_loader, val_loader, test_loader) = make_asepv1_dataloaders(
        config=config,
        return_dataset=False,
        split_method=config["dataset"]["split_method"],
    )

    eval_loader = test_loader
    if args.set == "val":
        eval_loader = val_loader
    elif args.set == "train":
        eval_loader = train_loader
    # main: inference on test set (170 samples)
    for batch in tqdm(eval_loader, total=len(eval_loader)):
        with torch.no_grad():
            pred = model(batch)

        results = {
            "job_id": batch.abdbid[0],
            "edge_index_bg_pred": (p := pred["edge_index_bg_pred"][0]),
            "epitope_pred": p.sum(axis=0).detach().cpu().numpy() > num_edge_cutoff,
        }

        # pred and true edge_index_bg
        t = pred["edge_index_bg_true"][0].clone().detach().cpu().numpy()
        p = pred["edge_index_bg_pred"][0].clone().detach().cpu().numpy()

        # log t and p shape
        logger.info(f"edge_index_bg_true: {t.shape}")
        logger.info(f"edge_index_bg_pred: {p.shape}")

        # evaluate 1. epitope node prediction, t and p shape (Nb, Ng)
        y_true = t.sum(axis=0) > 0
        y_pred = p.sum(axis=0) > num_edge_cutoff
        logger.info(f"y_true: {y_true.shape}")
        logger.info(f"y_pred: {y_pred.shape}")
        node_metrics = dec_evaluate_epitope_residue_prediction(eval_core)(
            y_pred=y_pred, y_true=y_true
        )
        results.update(node_metrics)

        # evaluate 2. bipartite graph edge reconstruction;
        try:
            edge_cutoff = config["hparams"]["edge_cutoff"]
        except KeyError as e:
            logger.warning(f"KeyError: {e}, setting edge_cutoff to 0.5")
            edge_cutoff = 0.5
        p_binary = (p > edge_cutoff).astype(int)
        bi_adj_metrics = dec_evaluate_bipartite_graph_reconstruction(eval_core)(
            edge_index_bg_pred=p_binary.flatten(), edge_index_bg_true=t.flatten()
        )
        results.update(bi_adj_metrics)

        # save results
        _save_json(args, results)
