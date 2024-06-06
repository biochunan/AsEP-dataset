"""
Running inference code using the isolated preprocess module
"""

import argparse
# cli
import ast  # type=ast.literal_eval
import json
import textwrap
from argparse import Namespace
from pathlib import Path
from typing import Any, Callable, Dict, Tuple, Union

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

# custom
from asep.preprocess import create_pair_graph_data, pyg_data_to_batch
from asep.train_model import create_model
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
    outdir = args.outdir / args.job_id
    outdir.mkdir(parents=True, exist_ok=True)
    with open(fp := (outdir / f"{args.job_id}.json"), "w") as f:
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

    mcc = matthews_corrcoef(y_true, y_pred)
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
        egde_index_bg_pred: np.ndarray, egde_index_bg_true: np.ndarray
    ) -> Dict[str, Dict[str, Any]]:
        metrics = {
            "metrics_bi_adj": func(y_pred=egde_index_bg_pred, y_true=egde_index_bg_true)
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
def inference_abdb_structure():
    args = cli()
    log_args(args)

    # --------------------------------------------------------------------------
    # Preprocessing
    # process input to pair graph data in a batch for inference
    # --------------------------------------------------------------------------
    pair_data = create_pair_graph_data(
        job_id=args.job_id,
        ab_structure=args.ab_structure,
        ag_structure=args.ag_structure,
        ab_chain_id=args.ab_chain_id,
        ag_chain_id=args.ag_chain_id,
        esm2_ckpt_path=args.esm2_ckpt_host_path,
        ab_seqres=args.ab_seqres,
        ag_seqres=args.ag_seqres,
        is_complex=args.is_complex,
        log_level=args.log_level,
    )
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
        "epitope_pred": p.sum(axis=1).detach().cpu().numpy() > 3,
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

        # evaluate 1. bipartite graph edge reconstruction;
        y_true = t.sum(axis=1) > 0
        y_pred = p.sum(axis=1) > 3
        node_metrics = dec_evaluate_epitope_residue_prediction(eval_core)(
            y_pred=y_pred, y_true=y_true
        )
        results.update(node_metrics)

        # evaluate 2. epitope node prediction
        try:
            edge_cutoff = config["hparams"]["edge_cutoff"]
        except KeyError as e:
            logger.warning(f"KeyError: {e}, setting edge_cutoff to 0.5")
            edge_cutoff = 0.5
        p_binary = (p > edge_cutoff).astype(int)
        bi_adj_metrics = dec_evaluate_bipartite_graph_reconstruction(eval_core)(
            egde_index_bg_pred=p_binary.flatten(), egde_index_bg_true=t.flatten()
        )
        results.update(bi_adj_metrics)

    # save results
    _save_json(args, results)


# input
def cli() -> Namespace:
    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawTextHelpFormatter,
        description="Inference from preprocess module",
        epilog=textwrap.dedent(
            text="""
            python inference.py \\
                -n pdb3kr3_0P \\
                -sa data/pdbs/pdb3kr3_0P.pdb \\
                -sg data/pdbs/pdb3kr3_0P.pdb \\
                -ca H L \\
                -cg D \\
                -o test \\
                --is_complex \\
                --pretrained_model_config assets/config/woven-rain-26.yaml \\
                --pretrained_model_ckpt assets/ckpts/woven-rain-26/rank_0-epoch_185.pt \\
                --esm2_ckpt_host_path $HOME/.cache/torch/hub/checkpoints
            """
        ),
    )
    # job_id a custom string to identify the inference job
    parser.add_argument(
        "-n",
        "--job_id",
        type=str,
        required=False,
        default="complex",
        help="job id for the inference job, default: complex",
    )
    # antibody and antigen strudctures
    parser.add_argument(
        "-sa",
        "--ab_structure",
        type=Path,
        required=True,
        help="path to the ab structure",
    )
    parser.add_argument(
        "-sg",
        "--ag_structure",
        type=Path,
        required=True,
        help="path to the ag structure",
    )
    # antibody and antigen SEQRES sequence files (optional)
    parser.add_argument(
        "-qa",
        "--ab_seqres",
        type=ast.literal_eval,
        required=False,
        default=None,
        help="path to the ab seqres fasta file, if not provided, use the SEQRES sequence from the PDB file",
    )
    parser.add_argument(
        "-qg",
        "--ag_seqres",
        type=ast.literal_eval,
        required=False,
        default=None,
        help="path to the ag seqres fasta file, if not provided, use the SEQRES sequence from the PDB file",
    )
    # antibody and antigen chain ids (optional)
    parser.add_argument(
        "-ca",
        "--ab_chain_id",
        type=str,
        required=True,
        nargs="*",
        default=["H", "L"],
        help="ab chain id, if not provided, use default chain id H and L",
    )
    parser.add_argument(
        "-cg",
        "--ag_chain_id",
        type=str,
        required=True,
        nargs="*",
        default=["A"],
        help="ag chain id, if not provided, use default chain id A",
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
    # add a flag to provide esm2 checkpoint path default to $HOME/.cache/torch/hub/checkpoints
    parser.add_argument(
        "-e",
        "--esm2_ckpt_host_path",
        type=Path,
        required=False,
        default=None,
        help=textwrap.dedent(
            """
        Host path to the esm2 checkpoint folder,
        e.g. $HOME/.cache/torch/hub/checkpoints
        """
        ),
    )
    # pretrained model
    parser.add_argument(
        "--pretrained_model_config",
        type=Path,
        required=True,
        help="path to the pretrained model config",
    )
    parser.add_argument(
        "--pretrained_model_ckpt",
        type=Path,
        required=True,
        help="path to the pretrained model checkpoint",
    )

    # whether the input is a complex structure, if the input is a complex, will calculate the interface residues
    parser.add_argument(
        "--is_complex",
        action="store_true",
        help="whether the input is a complex structure, if the input is a complex, will calculate the interface residues",
    )
    # add a flag to set loguru logger level to DEBUG / INFO / WARNING / ERROR / CRITICAL
    parser.add_argument(
        "--log_level",
        type=str,
        required=False,
        default="INFO",
        help="log level, default: INFO",
    )
    # dev options, including whether the user is using a devcontainer or not
    parser.add_argument(
        "--devcontainer",
        action="store_true",
        help=textwrap.dedent(
            """
            whether the user is using a devcontainer or not,
            This is required by docker facilities, which require host path to run docker containers
            """
        ),
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
    inference_abdb_structure()
