"""
Implement version 1 of the AsEP model
"""

# logging
import logging
# basic
import os
import os.path as osp
import re
import sys
from pathlib import Path
from pprint import pprint
from typing import (Any, Callable, Dict, Iterable, List, Mapping, Optional,
                    Set, Tuple, Union)

import numpy as np
import pandas as pd
# torch tools
import torch
import torch.nn as nn
import torch.nn.functional as F
# pyg tools
import torch_geometric as tg
import torch_geometric.transforms as T
import torch_scatter as ts
from omegaconf import DictConfig, OmegaConf
from torch import Tensor
from torch_geometric.data import Batch as PygBatch
from torch_geometric.data import Data as PygData
from torch_geometric.data import Dataset as PygDataset
from torch_geometric.datasets import TUDataset
from torch_geometric.loader import DataLoader as PygDataLoader
from torch_geometric.nn import GATConv, GCNConv
from torch_geometric.utils import to_dense_adj, to_dense_batch, to_undirected

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s {%(pathname)s:%(lineno)d} [%(threadName)s] [%(levelname)s] %(name)s - %(message)s",
    datefmt="%H:%M:%S",
)
# custom
from asep.data.asepv1_dataset import AsEPv1Dataset


class PyGAbAgIntGAE(nn.Module):
    def __init__(
        self,
        input_ab_dim: int,  # input dims
        input_ag_dim: int,  # input dims
        dim_list: List[int],  # dims (length = len(act_list) + 1)
        act_list: List[str],  # acts
        decoder: Optional[Dict] = None,  # layer type
        try_gpu: bool = True,  # use gpu
        input_ab_act: str = "relu",  # input activation
        input_ag_act: str = "relu",  # input activation
    ):
        super().__init__()
        decoder = (
            {
                "name": "inner_prod",
            }
            if decoder is None
            else decoder
        )
        self.device = torch.device(
            "cuda" if try_gpu and torch.cuda.is_available() else "cpu"
        )

        # add to hparams
        self.hparams = {
            "input_ab_dim": input_ab_dim,
            "input_ag_dim": input_ag_dim,
            "dim_list": dim_list,
            "act_list": act_list,
            "decoder": decoder,
        }
        self._args_sanity_check()

        # encoder
        _default_conv_kwargs = {"normalize": True}  # DO NOT set cache to True
        self.B_encoder_block = self._create_a_encoder_block(
            node_feat_name="x_b",
            edge_index_name="edge_index_b",
            input_dim=input_ab_dim,
            input_act=input_ab_act,
            dim_list=dim_list,
            act_list=act_list,
            gcn_kwargs=_default_conv_kwargs,
        ).to(self.device)
        self.G_encoder_block = self._create_a_encoder_block(
            node_feat_name="x_g",
            edge_index_name="edge_index_g",
            input_dim=input_ag_dim,
            input_act=input_ag_act,
            dim_list=dim_list,
            act_list=act_list,
            gcn_kwargs=_default_conv_kwargs,
        ).to(self.device)

        # decoder attr placeholder
        self.decoder = self.decoder_factory(self.hparams["decoder"])
        self._dc_func: Callable = self.decoder_func_factory(self.hparams["decoder"])

    def _args_sanity_check(self):
        # 1. if dim_list or act_list is provided, assert dim_list length is equal to act_list length + 1
        if self.hparams["dim_list"] is not None or self.hparams["act_list"] is not None:
            try:
                assert (
                    len(self.hparams["dim_list"]) == len(self.hparams["act_list"]) + 1
                ), (
                    f"dim_list length must be equal to act_list length + 1, "
                    f"got dim_list {self.hparams['dim_list']} and act_list {self.hparams['act_list']}"
                )
            except AssertionError as e:
                raise ValueError(
                    "dim_list length must be equal to act_list length + 1, "
                ) from e
        # 2. if decoder is provided, assert decoder name is in ['inner_prod', 'fc', 'bilinear']
        if self.hparams["decoder"] is not None:
            try:
                assert isinstance(self.hparams["decoder"], Union[dict, DictConfig])
            except AssertionError as e:
                raise TypeError(
                    f"decoder must be a dict, got {self.hparams['decoder']}"
                ) from e
            try:
                assert self.hparams["decoder"]["name"] in (
                    "inner_prod",
                    "fc",
                    "bilinear",
                )
            except AssertionError as e:
                raise ValueError(
                    f"decoder {self.hparams['decoder']['name']} not supported, "
                    "please choose from ['inner_prod', 'fc', 'bilinear']"
                ) from e

    def _create_a_encoder_block(
        self,
        node_feat_name: str,
        edge_index_name: str,
        input_dim: int,
        input_act: str,
        dim_list: List[int],
        act_list: List[str],
        gcn_kwargs: Dict[str, Any],
    ):
        def _create_gcn_layer(
            i: int, j: int, in_channels: int, out_channels: int
        ) -> GCNConv:
            if i == 0:
                mapping = f"{node_feat_name}, {edge_index_name} -> {node_feat_name}_{j}"
            else:
                mapping = (
                    f"{node_feat_name}_{i}, {edge_index_name} -> {node_feat_name}_{j}"
                )
            # print(mapping)

            return (
                GCNConv(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    **gcn_kwargs,
                ),
                mapping,
            )

        def _create_act_layer(act_name: Optional[str]) -> nn.Module:
            # assert act_name is either None or str
            assert act_name is None or isinstance(
                act_name, str
            ), f"act_name must be None or str, got {act_name}"

            if act_name is None:
                # return identity
                return (nn.Identity(),)
            elif act_name.lower() == "relu":
                return (nn.ReLU(inplace=True),)
            elif act_name.lower() == "leakyrelu":
                return (nn.LeakyReLU(inplace=True),)
            else:
                raise ValueError(
                    f"activation {act_name} not supported, please choose from ['relu', 'leakyrelu', None]"
                )

        modules = [
            _create_gcn_layer(0, 1, input_dim, dim_list[0]),
            _create_act_layer(input_act),
        ]

        for i in range(len(dim_list) - 1):
            modules.extend(
                [
                    _create_gcn_layer(
                        i + 1, i + 2, dim_list[i], dim_list[i + 1]
                    ),  # i+1 increment due to the input layer
                    _create_act_layer(act_list[i]),
                ]
            )

        return tg.nn.Sequential(
            input_args=f"{node_feat_name}, {edge_index_name}", modules=modules
        )

    def _init_fc_decoder(self, decoder) -> nn.Sequential:
        bias: bool = decoder["bias"]
        dp: Optional[float] = decoder["dropout"]

        dc = nn.ModuleList()

        # dropout
        if dp is not None:
            dc.append(nn.Dropout(dp))
        # fc linear
        dc.append(
            nn.Linear(
                in_features=self.hparams["dim_list"][-1] * 2, out_features=1, bias=bias
            )
        )
        # make it a sequential
        dc = nn.Sequential(*dc)

        return dc

    def encode(self, batch: PygBatch) -> Tuple[Tensor, Tensor]:
        """
        Args:
            batch: (PygBatch) batched data returned by PyG DataLoader
        Returns:
            B_z: (Tensor) shape (Nb, C)
            G_z: (Tensor) shape (Ng, C)
        """
        B_z = self.B_encoder_block(batch.x_b, batch.edge_index_b)  # (Nb, C)
        G_z = self.G_encoder_block(batch.x_g, batch.edge_index_g)  # (Ng, C)

        return B_z, G_z

    def decoder_factory(
        self, decoder_dict: Dict[str, str]
    ) -> Union[nn.Module, nn.Parameter, None]:
        name = decoder_dict["name"]

        if name == "bilinear":
            init_method = decoder_dict.get("init_method", "kaiming_normal_")
            decoder = nn.Parameter(
                data=torch.empty(
                    self.hparams["dim_list"][-1], self.hparams["dim_list"][-1]
                ),
                requires_grad=True,
            )
            torch.nn.init.__dict__[init_method](decoder)
            return decoder

        elif name == "fc":
            return self._init_fc_decoder(decoder_dict)

        elif name == "inner_prod":
            return

    def decoder_func_factory(self, decoder_dict: Dict[str, str]) -> Callable:
        name = decoder_dict["name"]

        if name == "bilinear":
            return lambda b_z, g_z: b_z @ self.decoder @ g_z.t()

        elif name == "fc":

            def _fc_runner(b_z: Tensor, g_z: Tensor) -> Tensor:
                """
                # (Nb, Ng, C*2)) -> (Nb, Ng, 1)
                # h = torch.cat([
                #         z_ab.unsqueeze(1).tile((1, z_ag.size(0), 1)),  # (Nb, 1, C) -> (Nb, Ng, C)
                #         z_ag.unsqueeze(0).tile((z_ab.size(0), 1, 1)),  # (1, Ng, C) -> (Nb, Ng, C)
                #     ], dim=-1)
                """
                h = torch.cat(
                    [
                        b_z.unsqueeze(1).expand(
                            -1, g_z.size(0), -1
                        ),  # (Nb, 1, C) -> (Nb, Ng, C)
                        g_z.unsqueeze(0).expand(
                            b_z.size(0), -1, -1
                        ),  # (1, Ng, C) -> (Nb, Ng, C)
                    ],
                    dim=-1,
                )
                # (Nb, Ng, C*2) -> (Nb, Ng, 1)
                h = self.decoder(h)
                return h.squeeze(-1)  # (Nb, Ng, 1) -> (Nb, Ng)

            return _fc_runner

        elif name == "inner_prod":
            return lambda b_z, g_z: b_z @ g_z.t()

    def decode(
        self, B_z: Tensor, G_z: Tensor, batch: PygBatch
    ) -> Tuple[Tensor, Tensor]:
        """
        Inner Product Decoder

        Args:
            z_ab: (Tensor)  shape (Nb, dim_latent)
            z_ag: (Tensor)  shape (Ng, dim_latent)

        Returns:
            A_reconstruct: (Tensor) shape (B, G)
                reconstructed bipartite adjacency matrix
        """
        # move batch to device
        batch = batch.to(self.device)

        edge_index_bg_pred = []
        edge_index_bg_true = []

        # dense bipartite edge index
        edge_index_bg_dense = torch.zeros(batch.x_b.shape[0], batch.x_g.shape[0]).to(
            self.device
        )
        edge_index_bg_dense[batch.edge_index_bg[0], batch.edge_index_bg[1]] = 1

        # get graph sizes (number of nodes) in the batch, used to slice the dense bipartite edge index
        node2graph_idx = torch.stack(
            [
                torch.cumsum(
                    torch.cat(
                        [
                            torch.zeros(1).long().to(self.device),
                            batch.x_b_batch.bincount(),
                        ]
                    ),
                    dim=0,
                ),  # (Nb+1, ) CDR     nodes
                torch.cumsum(
                    torch.cat(
                        [
                            torch.zeros(1).long().to(self.device),
                            batch.x_g_batch.bincount(),
                        ]
                    ),
                    dim=0,
                ),  # (Ng+1, ) antigen nodes
            ],
            dim=0,
        )

        for i in range(batch.num_graphs):
            edge_index_bg_pred.append(
                F.sigmoid(
                    self._dc_func(
                        b_z=B_z[batch.x_b_batch == i], g_z=G_z[batch.x_g_batch == i]
                    )
                )
            )  # Tensor (Nb, Ng)
            edge_index_bg_true.append(
                edge_index_bg_dense[
                    node2graph_idx[0, i] : node2graph_idx[0, i + 1],
                    node2graph_idx[1, i] : node2graph_idx[1, i + 1],
                ]
            )  # Tensor (Nb, Ng)

        return edge_index_bg_pred, edge_index_bg_true

    def forward(self, batch: PygBatch) -> Dict[str, Union[int, Tensor]]:
        # device
        batch = batch.to(self.device)
        # encode
        z_ab, z_ag = self.encode(batch)  # (Nb, C), (Ng, C)
        # decode
        edge_index_bg_pred, edge_index_bg_true = self.decode(z_ab, z_ag, batch)

        return {
            "abdbid": batch.abdbid,  # List[str]
            "edge_index_bg_pred": edge_index_bg_pred,  # List[Tensor (Nb, Ng)]
            "edge_index_bg_true": edge_index_bg_true,  # List[Tensor (Nb, Ng)]
        }


# a linear version of the model
class LinearAbAgIntGAE(nn.Module):
    def __init__(
        self,
        input_ab_dim: int,  # input dims
        input_ag_dim: int,  # input dims
        dim_list: List[int],  # dims (length = len(act_list) + 1)
        act_list: List[str],  # acts
        decoder: Optional[Dict] = None,  # layer type
        try_gpu: bool = True,  # use gpu
        input_ab_act: str = "relu",  # input activation
        input_ag_act: str = "relu",  # input activation
    ):
        super().__init__()
        decoder = (
            {
                "name": "inner_prod",
            }
            if decoder is None
            else decoder
        )
        self.device = torch.device(
            "cuda" if try_gpu and torch.cuda.is_available() else "cpu"
        )

        # add to hparams
        self.hparams = {
            "input_ab_dim": input_ab_dim,
            "input_ag_dim": input_ag_dim,
            "dim_list": dim_list,
            "act_list": act_list,
            "decoder": decoder,
        }
        self._args_sanity_check()

        # encoder
        self.B_encoder_block = self._create_a_encoder_block(
            node_feat_name="x_b",
            input_dim=input_ab_dim,
            input_act=input_ab_act,
            dim_list=dim_list,
            act_list=act_list,
        ).to(self.device)
        self.G_encoder_block = self._create_a_encoder_block(
            node_feat_name="x_g",
            input_dim=input_ag_dim,
            input_act=input_ag_act,
            dim_list=dim_list,
            act_list=act_list,
        ).to(self.device)

        # decoder attr placeholder
        self.decoder = self.decoder_factory(self.hparams["decoder"])
        self._dc_func: Callable = self.decoder_func_factory(self.hparams["decoder"])

    def _args_sanity_check(self):
        # 1. if dim_list or act_list is provided, assert dim_list length is equal to act_list length + 1
        if self.hparams["dim_list"] is not None or self.hparams["act_list"] is not None:
            try:
                assert (
                    len(self.hparams["dim_list"]) == len(self.hparams["act_list"]) + 1
                ), (
                    f"dim_list length must be equal to act_list length + 1, "
                    f"got dim_list {self.hparams['dim_list']} and act_list {self.hparams['act_list']}"
                )
            except AssertionError as e:
                raise ValueError(
                    "dim_list length must be equal to act_list length + 1, "
                ) from e
        # 2. if decoder is provided, assert decoder name is in ['inner_prod', 'fc', 'bilinear']
        if self.hparams["decoder"] is not None:
            try:
                assert isinstance(self.hparams["decoder"], Union[dict, DictConfig])
            except AssertionError as e:
                raise TypeError(
                    f"decoder must be a dict, got {self.hparams['decoder']}"
                ) from e
            try:
                assert self.hparams["decoder"]["name"] in (
                    "inner_prod",
                    "fc",
                    "bilinear",
                )
            except AssertionError as e:
                raise ValueError(
                    f"decoder {self.hparams['decoder']['name']} not supported, "
                    "please choose from ['inner_prod', 'fc', 'bilinear']"
                ) from e

    def _create_a_encoder_block(
        self,
        node_feat_name: str,
        input_dim: int,
        input_act: str,
        dim_list: List[int],
        act_list: List[str],
    ):
        def _create_linear_layer(i: int, in_channels: int, out_channels: int) -> tuple:
            if i == 0:
                mapping = f"{node_feat_name} -> {node_feat_name}_{i+1}"
            else:
                mapping = f"{node_feat_name}_{i} -> {node_feat_name}_{i+1}"
            # print(mapping)

            return (
                nn.Linear(in_channels, out_channels),
                mapping,
            )

        def _create_act_layer(act_name: Optional[str]) -> nn.Module:
            # assert act_name is either None or str
            assert act_name is None or isinstance(
                act_name, str
            ), f"act_name must be None or str, got {act_name}"

            if act_name is None:
                return (nn.Identity(),)
            elif act_name.lower() == "relu":
                return (nn.ReLU(inplace=True),)
            elif act_name.lower() == "leakyrelu":
                return (nn.LeakyReLU(inplace=True),)
            else:
                raise ValueError(
                    f"activation {act_name} not supported, please choose from ['relu', 'leakyrelu', None]"
                )

        modules = [
            _create_linear_layer(0, input_dim, dim_list[0]),  # First layer
            _create_act_layer(input_act),
        ]

        for i in range(len(dim_list) - 1):  # Additional layers
            modules.extend(
                [
                    _create_linear_layer(
                        i + 1, dim_list[i], dim_list[i + 1]
                    ),  # i+1 increment due to the input layer
                    _create_act_layer(act_list[i]),
                ]
            )

        return tg.nn.Sequential(input_args=f"{node_feat_name}", modules=modules)

    def _init_fc_decoder(self, decoder) -> nn.Sequential:
        bias: bool = decoder["bias"]
        dp: Optional[float] = decoder["dropout"]

        dc = nn.ModuleList()

        # dropout
        if dp is not None:
            dc.append(nn.Dropout(dp))
        # fc linear
        dc.append(
            nn.Linear(
                in_features=self.hparams["dim_list"][-1] * 2, out_features=1, bias=bias
            )
        )
        # make it a sequential
        dc = nn.Sequential(*dc)

        return dc

    def encode(self, batch: PygBatch) -> Tuple[Tensor, Tensor]:
        """
        Args:
            batch: (PygBatch) batched data returned by PyG DataLoader
        Returns:
            B_z: (Tensor) shape (Nb, C)
            G_z: (Tensor) shape (Ng, C)
        """
        batch = batch.to(self.device)
        B_z = self.B_encoder_block(batch.x_b)  # , batch.edge_index_b)  # (Nb, C)
        G_z = self.G_encoder_block(batch.x_g)  # , batch.edge_index_g)  # (Ng, C)

        return B_z, G_z

    def decoder_factory(
        self, decoder_dict: Dict[str, str]
    ) -> Union[nn.Module, nn.Parameter, None]:
        name = decoder_dict["name"]

        if name == "bilinear":
            init_method = decoder_dict.get("init_method", "kaiming_normal_")
            decoder = nn.Parameter(
                data=torch.empty(
                    self.hparams["dim_list"][-1], self.hparams["dim_list"][-1]
                ),
                requires_grad=True,
            )
            torch.nn.init.__dict__[init_method](decoder)
            return decoder

        elif name == "fc":
            return self._init_fc_decoder(decoder_dict)

        elif name == "inner_prod":
            return

    def decoder_func_factory(self, decoder_dict: Dict[str, str]) -> Callable:
        name = decoder_dict["name"]

        if name == "bilinear":
            return lambda b_z, g_z: b_z @ self.decoder @ g_z.t()

        elif name == "fc":

            def _fc_runner(b_z: Tensor, g_z: Tensor) -> Tensor:
                """
                # (Nb, Ng, C*2)) -> (Nb, Ng, 1)
                # h = torch.cat([
                #         z_ab.unsqueeze(1).tile((1, z_ag.size(0), 1)),  # (Nb, 1, C) -> (Nb, Ng, C)
                #         z_ag.unsqueeze(0).tile((z_ab.size(0), 1, 1)),  # (1, Ng, C) -> (Nb, Ng, C)
                #     ], dim=-1)
                """
                h = torch.cat(
                    [
                        b_z.unsqueeze(1).expand(
                            -1, g_z.size(0), -1
                        ),  # (Nb, 1, C) -> (Nb, Ng, C)
                        g_z.unsqueeze(0).expand(
                            b_z.size(0), -1, -1
                        ),  # (1, Ng, C) -> (Nb, Ng, C)
                    ],
                    dim=-1,
                )
                # (Nb, Ng, C*2) -> (Nb, Ng, 1)
                h = self.decoder(h)
                return h.squeeze(-1)  # (Nb, Ng, 1) -> (Nb, Ng)

            return _fc_runner

        elif name == "inner_prod":
            return lambda b_z, g_z: b_z @ g_z.t()

    def decode(
        self, B_z: Tensor, G_z: Tensor, batch: PygBatch
    ) -> Tuple[Tensor, Tensor]:
        """
        Inner Product Decoder

        Args:
            z_ab: (Tensor)  shape (Nb, dim_latent)
            z_ag: (Tensor)  shape (Ng, dim_latent)

        Returns:
            A_reconstruct: (Tensor) shape (B, G)
                reconstructed bipartite adjacency matrix
        """
        # move batch to device
        batch = batch.to(self.device)

        edge_index_bg_pred = []
        edge_index_bg_true = []

        # dense bipartite edge index
        edge_index_bg_dense = torch.zeros(batch.x_b.shape[0], batch.x_g.shape[0]).to(
            self.device
        )
        edge_index_bg_dense[batch.edge_index_bg[0], batch.edge_index_bg[1]] = 1

        # get graph sizes (number of nodes) in the batch, used to slice the dense bipartite edge index
        node2graph_idx = torch.stack(
            [
                torch.cumsum(
                    torch.cat(
                        [
                            torch.zeros(1).long().to(self.device),
                            batch.x_b_batch.bincount(),
                        ]
                    ),
                    dim=0,
                ),  # (Nb+1, ) CDR     nodes
                torch.cumsum(
                    torch.cat(
                        [
                            torch.zeros(1).long().to(self.device),
                            batch.x_g_batch.bincount(),
                        ]
                    ),
                    dim=0,
                ),  # (Ng+1, ) antigen nodes
            ],
            dim=0,
        )

        for i in range(batch.num_graphs):
            edge_index_bg_pred.append(
                F.sigmoid(
                    self._dc_func(
                        b_z=B_z[batch.x_b_batch == i], g_z=G_z[batch.x_g_batch == i]
                    )
                )
            )  # Tensor (Nb, Ng)
            edge_index_bg_true.append(
                edge_index_bg_dense[
                    node2graph_idx[0, i] : node2graph_idx[0, i + 1],
                    node2graph_idx[1, i] : node2graph_idx[1, i + 1],
                ]
            )  # Tensor (Nb, Ng)

        return edge_index_bg_pred, edge_index_bg_true

    def forward(self, batch: PygBatch) -> Dict[str, Union[int, Tensor]]:
        # device
        batch = batch.to(self.device)
        # encode
        z_ab, z_ag = self.encode(batch)  # (Nb, C), (Ng, C)
        # decode
        edge_index_bg_pred, edge_index_bg_true = self.decode(z_ab, z_ag, batch)

        return {
            "abdbid": batch.abdbid,  # List[str]
            "edge_index_bg_pred": edge_index_bg_pred,  # List[Tensor (Nb, Ng)]
            "edge_index_bg_true": edge_index_bg_true,  # List[Tensor (Nb, Ng)]
        }
