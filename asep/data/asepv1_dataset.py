# basic
import os
import os.path as osp
import pickle
import shutil
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
# torch tools
import torch
import torch.nn.functional as F
# pyg tools
import torch_geometric
from torch import Tensor
from torch_geometric.data import Data as PygData
from torch_geometric.data import InMemoryDataset as PygInMemoryDataset
from torcheval.metrics.functional import binary_auprc
from tqdm import tqdm

# asep tools
from asep.data.embedding_config import EmbeddingConfig

# ==================== Function ====================


# create a PairData object that inherits PyGData object,
# attrs:
# Ab and Ag node features      => x_b and x_g
# Ab and Ag inner graph edges  => edge_index_b and edge_index_g
# Ab-Ag bipartite edges        => edge_index_bg
class PairData(PygData):
    def __inc__(self, key: str, value: Any, *args, **kwargs) -> Any:
        if key == "edge_index_b":
            return self.x_b.size(0)
        if key == "edge_index_g":
            return self.x_g.size(0)
        if key == "edge_index_bg":
            return torch.tensor([[self.x_b.size(0)], [self.x_g.size(0)]])
        return super().__inc__(key, value, *args, **kwargs)


class AsEPv1Dataset(PygInMemoryDataset):
    # v1.2.0
    url = "https://drive.google.com/file/d/1g7U78c6FUhjqUPO6kiFocAApD0zTLhrM/view?usp=sharing"
    folder_name = "asepv1.2.0"

    def __init__(
        self,
        root: str,
        name: str,
        transform=None,
        pre_transform=None,
        pre_filter=None,
        # embedding setup
        embedding_config: EmbeddingConfig = None,
    ):
        """
        Args:
            root (str): Root directory where the dataset should be saved
            name (str): The name of the dataset
            transform (callable, optional): A function/transform that takes in an
                :obj:`torch_geometric.data.Data` object and returns a transformed
                version. The data object will be transformed before every access.
                (default: :obj:`None`)
            pre_transform (callable, optional): A function/transform that takes in
                an :obj:`torch_geometric.data.Data` object and returns a
                transformed version. The data object will be transformed before
                being saved to disk. (default: :obj:`None`)
            pre_filter (callable, optional): A function that takes in an
                :obj:`torch_geometric.data.Data` object and returns a boolean
                value, indicating whether the data object should be included in the
                final dataset. (default: :obj:`None`)
            embedding_config (EmbeddingConfig, optional): A dictionary containing
            the embedding configuration for the antibody and antigen graphs.
        """
        self.name = name
        self.emb_config = embedding_config or EmbeddingConfig()
        super().__init__(root, transform, pre_transform, pre_filter)
        nft = self.emb_config.node_feat_type
        if nft == "one_hot":
            self.data, self.slices = torch.load(self.processed_paths[0])
        elif nft == "pre_cal":
            self.data, self.slices = torch.load(self.processed_paths[1])
        elif nft == "custom":
            self.data, self.slices = torch.load(self.processed_paths[2])
        else:
            raise NotImplementedError(
                f"node_feat_type={nft} is not implemented, valid options are: ['one_hot', 'pre_cal']"
            )

    def __repr__(self) -> str:
        return f"{self.name}({len(self)})"

    @property
    def raw_dir(self) -> str:
        """return the raw directory"""
        return osp.join(self.root, self.name, "raw")

    @property
    def processed_dir(self) -> str:
        """return the processed directory"""
        return osp.join(self.root, self.name, "processed")

    @property
    def raw_file_names(self) -> List[str]:
        """Raw file names. Used to check if files exist before downloading."""
        return [
            osp.join(self.raw_dir, "asepv1-AbDb-IDs.txt"),
            osp.join(self.raw_dir, "asepv1_interim_graphs"),
            osp.join(self.raw_dir, "structures.tar.gz"),
            osp.join(self.root, self.name, "split", "split_dict.pt"),
        ]

    @property
    def processed_file_names(self) -> str:
        """
        Return the processed file names.
        These paths were used to save the processed data.
        """
        return ["one_hot.pt", "pre_cal.pt", "custom_emb.pt"]

    def download(self):
        # if files already exist, skip download
        if all(osp.exists(i) for i in self.raw_file_names):
            print(f"Files already exist in {self.raw_dir}, skip download.")
            return

        try:
            import gdown
        except ImportError as e:
            raise ImportError("gdown is not installed. Run `pip install gdown`.") from e

        output = osp.join(self.root, self.name, "raw", "asepv1.zip")
        gdown.download(url=self.url, output=output, quiet=False, fuzzy=True)

        # 1. unzip asepv1.zip to asepv1 folder
        import zipfile

        with zipfile.ZipFile(output, "r") as f:
            f.extractall(path=osp.dirname(output))  # created a folder named `asepv1`

        base = osp.join(osp.dirname(output), self.folder_name)
        # 2. decompress asepv1_interim_graphs.tar.gz to asepv1_interim_graphs folder
        import tarfile

        with tarfile.open(osp.join(base, "asepv1_interim_graphs.tar.gz"), "r:gz") as f:
            f.extractall(path=self.raw_dir)
        # 3. move asepv1-grphs.txt to raw folder
        shutil.move(
            osp.join(base, "asepv1-AbDb-IDs.txt"),
            osp.join(self.raw_dir, "asepv1-AbDb-IDs.txt"),
        )
        # 4, move split_dict.pt to split folder
        split_dir = osp.join(self.root, self.name, "split")
        os.makedirs(split_dir, exist_ok=True)
        shutil.move(
            osp.join(base, "split_dict.pt"), osp.join(split_dir, "split_dict.pt")
        )
        # 5. move structures.tar.gz to raw folder
        shutil.move(
            osp.join(base, "structures.tar.gz"),
            osp.join(self.raw_dir, "structures.tar.gz"),
        )
        # 6. clean up
        os.unlink(output)
        shutil.rmtree(base)

    def process(self):
        # read abdbid_list from asepv1-graphs.txt
        with open(osp.join(self.raw_dir, "asepv1-AbDb-IDs.txt"), "r") as f:
            abdbid_list = f.read().splitlines()

        # Read data into huge `Data` list.
        data_list: List[PairData] = [
            convert_one_interim_to_pyg_pair_data(
                abdbid=abdbid,
                interim_graph_dir=osp.join(self.raw_dir, "asepv1_interim_graphs"),
                embedding_config=self.emb_config,
            )
            for abdbid in tqdm(abdbid_list, unit="graph_pair")
        ]

        if self.pre_filter is not None:
            data_list = [data for data in data_list if self.pre_filter(data)]

        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]

        data, slices = self.collate(data_list)
        if self.emb_config.node_feat_type == "one_hot":
            torch.save((data, slices), self.processed_paths[0])
        elif self.emb_config.node_feat_type == "pre_cal":
            torch.save((data, slices), self.processed_paths[1])
        elif self.emb_config.node_feat_type == "custom":
            torch.save((data, slices), self.processed_paths[2])

    def get_idx_split(self, split_method: str = None) -> Dict[str, Tensor]:
        """
        Get the split indices for the dataset.

        Args:
            split_method (str, optional): The method to split the dataset.
                Either 'epitope_ratio' or 'epitope_group'.
                Defaults to 'epitope_ratio'.

        Returns:
            split_dict: dict with keys ['train', 'valid', 'test'],
                each value is a 1D tensor of graph indices
        """
        split_method = split_method or "epitope_ratio"
        assert split_method in {
            "epitope_ratio",
            "epitope_group",
        }, f"split_method={split_method} is not supported, valid options are: ['epitope_ratio', 'epitope_group']"
        return torch.load(osp.join(self.root, self.name, "split", "split_dict.pt"))[
            split_method
        ]

    def get_idx_random_split(self, seed: Optional[int] = None) -> Dict[str, Tensor]:
        """return a random split of the dataset"""
        if seed is not None:
            torch.manual_seed(seed)
        idx = torch.randperm(len(self))
        a, b, _ = 1384, 170, 170
        return {
            "train": idx[:a],  # 1384
            "valid": idx[a : a + b],  # 170
            "test": idx[a + b :],  # 170
        }


class AsEPv1Evaluator:
    def __init__(self) -> None:
        """
        Evaluator for the AsEPv1 dataset.
        Metric is AUC-PRC (Area Under the Precision-Recall Curve).
        """
        self.bprc = binary_auprc

    def eval(self, input_dict: Dict[str, Tensor]):
        """
        Args:
            input_dict (Dict[str, Tensor]): Dictionary with keys ['y_pred', 'y_true']
                y_true: logits Tensor (float) of shape (num_nodes,)
                y_pred: binary Tensor (int)   of shape (num_nodes,)
        """
        assert "y_pred" in input_dict, "input_dict must have key 'y_pred'"
        assert "y_true" in input_dict, "input_dict must have key 'y_true'"
        y_pred, y_true = input_dict["y_pred"], input_dict["y_true"]

        # check data type
        try:
            assert isinstance(
                y_pred, torch.Tensor
            ), f"y_pred must be a torch.Tensor, got {type(y_pred)}"
            assert isinstance(
                y_true, torch.Tensor
            ), f"y_true must be a torch.Tensor, got {type(y_true)}"
        except AssertionError as e:
            raise TypeError(
                f"y_pred and y_true must be torch.Tensor, got {type(y_pred)} and {type(y_true)}"
            ) from e

        # check shape
        try:
            assert y_true.shape == y_pred.shape
        except AssertionError as e:
            raise ValueError(
                f"y_true and y_pred must have the same shape, got {y_true.shape} and {y_pred.shape}"
            ) from e

        # check ndim
        try:
            assert y_true.ndim == 1
            assert y_pred.ndim == 1
        except AssertionError as e:
            raise ValueError(
                f"y_true and y_pred must have ndim=1, got {y_true.ndim} and {y_pred.ndim}"
            ) from e

        return {"auc-prc": self.bprc(y_pred, y_true)}


# ----------helper functions ----------
def sparse_matrix_to_edge_index(sp_mat, to_undirected: bool = False) -> Tensor:
    r, c = sp_mat.nonzero()
    E = torch.tensor(np.stack([r, c], axis=0), dtype=torch.long)
    if to_undirected:
        E = torch_geometric.utils.to_undirected(E)
    return E


def aa_seq_to_one_hot_tensor(aa_seq: Sequence[str]) -> Tensor:
    aa2int = {r: i for i, r in enumerate("ACDEFGHIKLMNPQRSTVWY")}
    one_hot = F.one_hot(
        torch.tensor([aa2int[i] for i in aa_seq]), num_classes=len(aa2int)
    ).type(torch.float32)
    return one_hot


# ---------- wrapper functions ----------
def load_interim_graph_pkl(abdbid: str, interim_graph_dir: str) -> Dict[str, Any]:
    pkl_fp = osp.join(interim_graph_dir, f"{abdbid}.pkl")
    with open(pkl_fp, "rb") as f:
        return pickle.load(f)


def load_interim_graph_pt(abdbid: str, interim_graph_dir: str) -> Dict[str, Any]:
    """
    A helper function to load interim graph data from a .pt file.

    Args:
        abdbid (str): AbDb Id of the graph pair
        interim_graph_dir (str): Path to the directory where the interim graphs are stored

    Returns:
        Dict[str, Any]: A dictionary containing the interim graph data
        Schema:
        {
            'Nb': int,
            'Ng': int,
            'abdbid': str,
            'seqers': {
                'ab': chain_label(str) -> seqres(str),
                'ag': chain_label(str) -> seqres(str),
            },
            'mapping': {
                'ab': 'seqres2cdr': binary array
                'ag': 'seqres2surf': binary array
            },
            'embedding': {
                'ab': {
                    'igfold': torch.Tensor,  e.g. [N, 512] where N is the number of nodes i.e. CDR residues
                    'esm2': torch.Tensor  # e.g. [N, 480] where N is the number of nodes i.e. CDR residues
                },
                'ag': {
                    'esm2': torch.Tensor e.g. [N, 480] where N is the number of nodes i.e. surface residues
                }
            },
            'edges': {
                'ab': torch_sparse COO tensor,
                'ag': torch_sparse COO tensor,
                'bipartite': torch_sparse COO tensor
            },
            'stats': {
                'cdr': int  # number of CDR nodes
                'surf': int # number of surface nodes
                'epitope': int # number of epitope nodes
                'epitope2surf_ratio': float # epitope to surface ratio
            }

        }
    """
    pt_fp = osp.join(interim_graph_dir, f"{abdbid}.pt")
    return torch.load(pt_fp)


def validate_custom_embedding_method(embedding_config: Dict[str, Any]) -> None:
    """
    Validate custom embedding method.

    Args:
        embedding_config (Dict[str, Any]): A dictionary containing the embedding
        configuration for the antibody and antigen graphs.
    """
    for func in [
        embedding_config["ab"]["custom_embedding_method"],
        embedding_config["ag"]["custom_embedding_method"],
    ]:
        assert isinstance(func, Callable), f"custom_embedding_method must be a callable"
        # 3.1 assert the callable has the correct signature,
        # that it takes sequence as input and
        # output a torch.Tensor with shape [L, D] where
        # - L is the length of the sequence
        # - D is the embedding dimension
        dummy_input = "ACDEFGHIKLMNPQRSTVWY"
        try:
            emb = func(dummy_input)
        except Exception as e:
            raise ValueError(
                f"custom_embedding_method must take a sequence as input and output a torch.Tensor, got error: {e}"
            )
        try:
            assert emb.ndim == 2
        except AssertionError as e:
            raise ValueError(
                f"custom_embedding_method must output a torch.Tensor with shape [L, D], got {emb.shape}"
            ) from e
        try:
            assert emb.shape[0] == len(dummy_input)
        except AssertionError as e:
            raise ValueError(
                f"custom_embedding_method must output a torch.Tensor with the first dimension equal to the length of the input sequence, got {emb.shape[0]} != {len(dummy_input)}"
            ) from e


def get_node_feat_from_interim_graph_data(
    interim_graph_data: Dict[str, Any], embedding_config: EmbeddingConfig
) -> Tuple[Tensor, Tensor]:
    """
    Get node features from interim graph data, or compute node features using a custom method if provided.

    Args:
        interim_graph_data (Dict[str, Any]): Interim graph data loaded from a .pt file
        embedding_config (Dict[str, Any]): A dictionary containing the embedding
            configuration for the antibody and antigen graphs.
    Raises:
        NotImplementedError: if node_feat_type is not 'pre_cal' or 'one_hot'

    Returns:
        Tuple[Tensor, Tensor]: A tuple of two tensors representing the node features for the antibody and antigen graphs respectively.
    """
    # cerate a short hand for the embedding config
    cfg = embedding_config

    # ----------------------------------------
    # antibody
    # ----------------------------------------
    if (t := cfg.node_feat_type) != "custom":
        if t == "pre_cal":
            x_b = interim_graph_data["embedding"]["ab"][cfg.ab.embedding_model]
        elif t == "one_hot":
            concat_seq = "{}{}".format(
                interim_graph_data["seqres"]["ab"]["H"],
                interim_graph_data["seqres"]["ab"]["L"],
            )
            x_b = aa_seq_to_one_hot_tensor(aa_seq=concat_seq)
    else:
        x_b = torch.cat(
            [
                cfg.ab.custom_embedding_method(interim_graph_data["seqres"]["ab"]["H"]),
                cfg.ab.custom_embedding_method(interim_graph_data["seqres"]["ab"]["L"]),
            ],
            dim=0,
        )

    # ----------------------------------------
    # antigen
    # ----------------------------------------
    if (t := cfg.node_feat_type) != "custom":
        if t == "pre_cal":
            x_g = interim_graph_data["embedding"]["ag"][cfg.ag.embedding_model]
        elif t == "one_hot":
            x_g = aa_seq_to_one_hot_tensor(
                aa_seq=list(interim_graph_data["seqres"]["ag"].values())[0]
            )
    else:
        x_g = cfg.ag.custom_embedding_method(
            list(interim_graph_data["seqres"]["ag"].values())[0]
        )

    # TODO: should we also provide the full version?
    # use the binary vector to slice the embeddings
    x_b = x_b[(interim_graph_data["mapping"]['ab']['seqres2cdr']).astype(bool)]
    x_g = x_g[(interim_graph_data["mapping"]['ag']['seqres2surf']).astype(bool)]

    return x_b, x_g


def convert_one_interim_to_pyg_pair_data(
    abdbid: str, interim_graph_dir: str, embedding_config: Dict[str, Any]
) -> PairData:
    """
    Convert one interim graph to PyG PairData object.

    Args:
        abdbid (str): AbDb Id of the graph pair
        interim_graph_dir (str): Directory where the interim graphs are stored
        embedding_config (Dict[str, Any]): A dictionary containing the embedding
            configuration for the antibody and antigen graphs.
            See `validate_embedding_config` for schema.

    Raises:
        NotImplementedError: _description_

    Returns:
        PairData: _description_
    """
    data = load_interim_graph_pt(abdbid, interim_graph_dir=interim_graph_dir)

    # node feat
    x_b, x_g = get_node_feat_from_interim_graph_data(
        interim_graph_data=data, embedding_config=embedding_config
    )

    # inner and bipartite graph edges
    edge_index_b = data["edges"]["ab"].coalesce().indices()
    edge_index_g = data["edges"]["ag"].coalesce().indices()
    edge_index_bg = data["edges"]["bipartite"].coalesce().indices()

    # node interface label
    y_b = torch.zeros(x_b.size(0), dtype=torch.long)
    y_g = torch.zeros(x_g.size(0), dtype=torch.long)
    y_b[edge_index_bg[0].unique(sorted=True)] = 1
    y_g[edge_index_bg[1].unique(sorted=True)] = 1

    # to PairData
    pair_data = PairData(
        x_b=x_b,
        x_g=x_g,
        edge_index_b=edge_index_b,
        edge_index_g=edge_index_g,
        abdbid=abdbid,
        edge_index_bg=edge_index_bg,
        y_b=y_b,
        y_g=y_g,
    )

    return pair_data


# ==================== Main ====================
if __name__ == "__main__":
    emb_config = EmbeddingConfig(
        node_feat_type="pre_cal",
        ab={"embedding_model": "igfold"},
        ag={"embedding_model": "esm2"},
    )
    asepv1_dataset = AsEPv1Dataset(
        root="/mnt/data",
        name="asep",
        # embedding setup
        embedding_config=emb_config,
    )
    # epitope ratio split
    split_idx = asepv1_dataset.get_idx_split(split_method="epitope_ratio")
    train_set = asepv1_dataset[split_idx["train"]]
    valid_set = asepv1_dataset[split_idx["val"]]
    test_set = asepv1_dataset[split_idx["test"]]
    # epitope group split
    split_idx = asepv1_dataset.get_idx_split(split_method="epitope_group")
    train_set = asepv1_dataset[split_idx["train"]]
    test_set = asepv1_dataset[split_idx["test"]]
