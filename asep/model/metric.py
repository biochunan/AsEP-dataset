from typing import Dict

import torch
from torch import Tensor
from torcheval.metrics import BinaryAUPRC, BinaryConfusionMatrix
from torcheval.metrics.functional import binary_auprc
from torchmetrics.functional import matthews_corrcoef


def cal_edge_index_bg_auprc(
    edge_index_bg_pred: Tensor,
    edge_index_bg_true: Tensor,
    edge_cutoff: float = 0.5,
) -> Tensor:
    with torch.no_grad():
        t = edge_index_bg_true.reshape(-1).long().cpu()
        p = (edge_index_bg_pred > edge_cutoff).reshape(-1).long().cpu()

        return binary_auprc(p, t)


def cal_epitope_node_auprc(
    edge_index_bg_pred: Tensor,
    edge_index_bg_true: Tensor,
    num_edge_cutoff: int,  # used to determine epitope residue from edges,
) -> Tensor:
    with torch.no_grad():
        # get epitope idx
        t = (edge_index_bg_true.sum(dim=0) > 0).reshape(-1).long()
        p = (edge_index_bg_pred.sum(dim=0) > num_edge_cutoff).reshape(-1).long()

        return binary_auprc(p, t)


def cal_edge_index_bg_metrics(
    edge_index_bg_pred: Tensor,
    edge_index_bg_true: Tensor,
    edge_cutoff: float = 0.5,
) -> Dict:
    with torch.no_grad():
        t = edge_index_bg_true.reshape(-1).long().cpu()
        p = edge_index_bg_pred.reshape(-1).cpu()
        # auprc
        auprc = BinaryAUPRC().update(input=p, target=t).compute()
        # confusion matrix
        tn, fp, fn, tp = (
            BinaryConfusionMatrix(threshold=edge_cutoff)
            .update(input=p, target=t)
            .compute()
            .reshape(-1)
        )
        # option 1: manually calculate mcc
        mcc = (tp * tn - fp * fn) / (
            torch.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)) + 1e-7
        )
        # # option 2: use torchmetrics
        # p_ = (p > edge_cutoff).long()
        # mcc: Tensor = matthews_corrcoef(preds=p_, target=t, num_classes=2, task="binary")

        return {"tn": tn, "fp": fp, "fn": fn, "tp": tp, "auprc": auprc, "mcc": mcc}


def cal_epitope_node_metrics(
    edge_index_bg_pred: Tensor,
    edge_index_bg_true: Tensor,
    num_edge_cutoff: int,  # used to determine epitope residue from edges,
):
    with torch.no_grad():
        # get epitope idx
        t = (edge_index_bg_true.sum(dim=0) > 0).reshape(-1).long().cpu()
        p = (edge_index_bg_pred.sum(dim=0) > num_edge_cutoff).reshape(-1).long().cpu()


        # auprc
        auprc = BinaryAUPRC().update(input=p, target=t).compute()
        # confusion matrix
        tn, fp, fn, tp = (
            BinaryConfusionMatrix().update(input=p, target=t).compute().reshape(-1)
        )
        # option 1: manually calculate mcc
        mcc = (tp * tn - fp * fn) / torch.sqrt(
            (tp + fp) * (tp + fn) * (tn + fp) * (tn + fn) + 1e-7
        )
        # # option 2: use torchmetrics
        # mcc: Tensor = matthews_corrcoef(preds=p, target=t, num_classes=2)
        # print(p)
        # print(t)
        # print(mcc)

        return {"tn": tn, "fp": fp, "fn": fn, "tp": tp, "auprc": auprc, "mcc": mcc}
