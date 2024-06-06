from typing import Optional, Union

import torch
import torch.nn.functional as F
from torch import Tensor

# --------------------
# loss factory function
# --------------------
'''
Loss function configuration format:
{
    "loss": [
        {
            "name": "badj_rec_loss",  # F.binary_cross_entropy
            "w": 1.0,
            "kwargs": {}
        },
        {
            "name": "bipartite_edge_loss",
            "w": 0.0003942821556421417,
            "kwargs": {"thr": 40}
        },
        {
            "name": "l2",
            "w": 1e-2,
            "kwargs": {}
        }
    ]
}
each loss term is a dict, with keys:
- name: (str) loss function name
- w: (float) weight for this loss term
- kwargs: (dict) kwargs for this loss function
'''


def edge_index_bg_sum_loss(
    edge_index_bg_pred: Tensor,
    thr: Optional[float] = None,
) -> Tensor:
    """ Calculate the bipartite-edge loss
    Minimize this loss encourages the model to reconstruct the bipartite graph
    with the same number of average edges as in the ground-truth graphs

    Input is a single AbAg graph pair
    Args:
        edge_index_bg_pred (Tensor): predicted bipartite adjacency, shape (Nb, Ng)
        thr: (float) threshold for the number of edges in the reconstructed bipartite graph
    Returns:
        loss (Tensor): sum of the difference between the number of edges in the reconstructed bipartite graph
            and the average over the ground-truth bipartite graphs
    """
    thr = 40 if thr is None else thr
    return torch.abs(torch.sum(edge_index_bg_pred) - thr)


def edge_index_bg_rec_loss(
    edge_index_bg_pred: Tensor,
    edge_index_bg_true: Tensor,
    weight_tensor: Union[Tensor, float],
    reduction: str='none'
) -> Tensor:
    """ Calculate interface edge reconstruction loss

    Input is a single AbAg graph pair
    Args:
        edge_index_bg_pred: (Tensor) reconstructed bipartite adjacency matrix, shape (Nb, Ng), float  between 0, 1
        edge_index_bg_true: (Tensor) ground-truth  bipartite adjacency matrix, shape (Nb, Ng), binary 0/1
        weight_tensor: (Tensor) for balancing pos/neg samples, shape (Nb*Ng,)
    """
    device = edge_index_bg_pred.device

    # if weight_tensor is a float or a scalar Tensor, i.e. positive edge weight,
    # convert it to a tensor of the same shape as edge_index_bg_true
    if isinstance(weight_tensor, Union[int, float]):
        weight_tensor = torch.tensor(weight_tensor, device=device)
    if isinstance(weight_tensor, Tensor) and weight_tensor.ndim == 0:
        weight_tensor = edge_index_bg_true * weight_tensor
        weight_tensor[edge_index_bg_true == 0] = 1
        weight_tensor = weight_tensor.float().to(device)

    try:
        assert edge_index_bg_pred.reshape(-1).shape == edge_index_bg_true.reshape(-1).shape == weight_tensor.reshape(-1).shape
    except AssertionError as e:
        raise ValueError(
            "The following shapes should be the same but received:\n"
            f"{edge_index_bg_pred.reshape(-1).shape=}\n"
            f"{edge_index_bg_true.reshape(-1).shape=}\n"
            f"{weight_tensor.reshape(-1).shape=}\n"
        ) from e

    return F.binary_cross_entropy(
        edge_index_bg_pred.reshape(-1),  # shape (b*g, )
        edge_index_bg_true.reshape(-1),  # shape (b*g, )
        weight=weight_tensor.reshape(-1),
        reduction=reduction,
    )

