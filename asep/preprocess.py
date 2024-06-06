"""
Inference code:
Input:
1. load a checkpoint file
2. Ab structure: (Mandatory)
3. Ag structure: (Mandatory)
4. Ab SEQRES sequence: (Optional) if not provided, use the SEQRES sequence from the PDB file
5. Ag SEQRES sequence: (Optional) if not provided, use the SEQRES sequence from the PDB file
6. ab chain id: (Optional) if not provided, use default chain id H and L
7. ag chain id: (Optional) if not provided, use default chain id A
Processing:
1. build model
2. load checkpoint
Run:
1. run inference
Output:
1. save the predicted epitope to a file
"""

import argparse
# cli
import ast  # type=ast.literal_eval
# basic
import re
import shutil
import sys
import tempfile
from argparse import Namespace
from collections import OrderedDict
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
# torch tools
import torch
# Biopython
from Bio import SeqIO
from Bio.PDB import PDBIO, PDBParser, Polypeptide, Select
from graphein.protein.config import DSSPConfig, ProteinGraphConfig
# graphein
from graphein.protein.features.nodes import rsa
from graphein.protein.graphs import construct_graph
from loguru import logger
from networkx import Graph
from omegaconf import DictConfig, OmegaConf
from pandas import DataFrame
from scipy.spatial.distance import pdist, squareform
# pyg tools
from torch_geometric.data import Batch as PygBatch
from torch_geometric.loader import DataLoader as PygDataLoader
from torch_geometric.utils import dense_to_sparse

# custom
from asep.data.graph_pair import PairData
from asep.docker_utils import (run_esm2_docker_container,
                               run_igfold_docker_container)
from asep.utils import (Resi2CDR, calculate_dist_mat, dec_log_func,
                        extract_seqres_from_pdb, insert_cryst1_line_to_pdb,
                        log_args, run_seqres2atmseq)

# ==================== Configuration ====================
# Types
PathLike = Union[str, Path]
PDBDataFrame = DataFrame
AllAtomDataFrame = DataFrame
AdjMatrix = np.ndarray
BinaryMask = np.ndarray

# Paths
BASE = Path(__file__).resolve().parent  # => walle

# structure processing
GRAPHEIN_CFG = ProteinGraphConfig(
    **{
        "granularity": "centroids",  # "atom", "CA", "centroids"
        "insertions": True,
        "edge_construction_functions": [],
        "dssp_config": DSSPConfig(executable=shutil.which("mkdssp")),
        "graph_metadata_functions": [rsa],
    }
)
RSA_THR = 0.0  # surface residue Relative solvent-accessible Surface Area threshold
CDR_DEF = "ABM"  # CDR definition
DIST_THR = 4.5  # distance threshold for contacting residues


# ==================== Function ====================
# ==================== #
#     sanity check     #
# ==================== #
# confirm SEQRES exists in the file
def _is_SEQRES_in_pdb_file(fp: Path) -> bool:
    with open(fp) as f:
        for line in f:
            if line.startswith("SEQRES"):
                return True
    return False


# assert fasta header format
def _assert_fasta_header_format(fasta_file) -> None:
    """
    Asserts that the header of each sequence in a FASTA file follows the format:
    ><seqid>|<chain_id>

    :param fasta_file: Path to the FASTA file
    """
    # Define the pattern for the header
    header_pattern = re.compile(r"^>.+\|[^\|]{1}$")

    # Parse the FASTA file
    with open(fasta_file, "r") as handle:
        for record in SeqIO.parse(handle, "fasta"):
            header = f">{record.id}"
            if not header_pattern.match(header):
                raise ValueError(f"Invalid header format: {header}")


# assert SEQRES exists in a file
def _assert_SEQRES_in_pdb_file(pdb_file: PathLike) -> None:
    try:
        assert _is_SEQRES_in_pdb_file(pdb_file)
    except AssertionError as e:
        raise ValueError(f"No SEQRES found in {pdb_file}")


# ==================== #
#       configs        #
# ==================== #
def parse_config(config_path: Path) -> Dict:
    return OmegaConf.load(config_path)


# ==================== #
#       graphs         #
# ==================== #
# construct graph (Graphein)
def read_pdb_as_graph(
    pdb_path: Path, graphein_cfg: ProteinGraphConfig, chains: Optional[List[str]] = None
) -> Graph:
    """
    Copy raw pdb file to a temporary directory,
    insert "CRYST1" line to the pdb file,
    then construct graph from the new pdb file.
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        pdb_fp = tmpdir / f"{pdb_path.stem}_cryst1.pdb"
        shutil.copy(pdb_path, pdb_fp)
        if chains:
            structure = PDBParser(QUIET=True).get_structure(id="pdb", file=pdb_fp)

            class ExtractChains(Select):
                def __init__(self, chain_id_to_extract):
                    self.chain_id_to_extract = chain_id_to_extract

                def accept_chain(self, chain):
                    return chain.get_id() in self.chain_id_to_extract

            io = PDBIO()
            io.set_structure(structure)
            io.save(pdb_fp.as_posix(), select=ExtractChains(chains))

        pdb_fp = insert_cryst1_line_to_pdb(pdb_fp)
        g = construct_graph(config=graphein_cfg, path=pdb_fp, verbose=False)

    return g


# decorator for process_graph
def dec_process_ab_graph(func: Callable, cdr_def: Optional[str] = None) -> Callable:
    """
    A decorator that adds `cdr` (str, H1, H2, ...) and `is_cdr` (bool) columns to the output of `func`.
    `func` must return a DataFrame with `node_id` column.

    Args:
        func (Callable): a function (func: `process_graph`) that takes a Graph as input and returns a DataFrame with `node_id` column.
        cdr_def (Optional[str], optional): CDR definition. Defaults to 'ABM'.

    Returns:
        Returns:
        Callable: the decorated function
    """
    cdr_def = cdr_def or "ABM"

    def wrapper(G: Graph, *args, **kwargs) -> Tuple[DataFrame, DataFrame]:
        atom_df = func(G, *args, **kwargs)
        # [x]TODO: add `cdr`, `is_cdr` column
        atom_df["cdr"] = atom_df.apply(
            lambda x: Resi2CDR[cdr_def].get(f"{x.chain_id}{x.residue_number}", ""),
            axis=1,
        )
        atom_df["is_cdr"] = atom_df.cdr.apply(lambda x: x != "")
        # [x] TODO: Add a step to force chain order of H, L
        atom_df = pd.concat(
            [
                atom_df.query(f'chain_id == "{c}"')
                for c in ["H", "L"]
                if c in atom_df.chain_id.unique()
            ]
        ).reset_index(drop=True)
        cdr_df = atom_df[atom_df["is_cdr"]].reset_index(drop=True).copy()
        return atom_df, cdr_df

    return wrapper


# decorator for process_graph
def dec_process_ag_graph(func: Callable, rsa_thr: float = 0.2) -> Callable:
    """
    A decorator that adds `is_surface` column to the output of `func`.
    `func` must return a DataFrame with `node_id` column.

    Args:
        func (Callable): a function (func: `process_graph`) that takes a Graph as input and returns a DataFrame with `node_id` column.

        rsa_thr (float, optional): Relative solvent-accessible Surface Area threshold. Defaults to 0.2.

    Returns:
        Callable: the decorated function
    """

    def wrapper(G: Graph) -> Tuple[DataFrame, DataFrame]:
        atom_df = func(G)
        # [x] TODO: add `is_surface` column
        dssp_df = G.graph["dssp_df"]
        atom_df["is_surface"] = atom_df.node_id.apply(
            lambda x: x in dssp_df.query(f"rsa > {rsa_thr}").index.values
        )
        surf_df = atom_df[atom_df["is_surface"]].reset_index(drop=True).copy()
        return atom_df, surf_df

    return wrapper


# process graph read from Graphein
def process_graph(G: Graph) -> DataFrame:
    # G, chains = G_ag.copy(), ['C']  # debug only
    # G, chains = G_ab.copy(), ['H', 'L']  # debug only
    atom_df = G.graph["raw_pdb_df"]
    # remove HETATM
    atom_df = atom_df[atom_df["record_name"] == "ATOM"].reset_index(drop=True).copy()
    return atom_df


# generate bipartite edge adjacency matrix
def generate_bipartite_adj_matrix_placeholder(
    df1: AllAtomDataFrame, df2: AllAtomDataFrame
) -> AdjMatrix:
    """
    This is generating a PLACEHOLDER residue contact adjacency matrix between two ALL-ATOM structure DataFrames.
    It has the shape of (N, M), where N is the number of residues in df1, M is the number of residues in df2.

    Args:
        df1 (PDBDataFrame): ALL-ATOM structure DataFrame 1
        df2 (PDBDataFrame): ALL-ATOM structure DataFrame 2

    Returns:
        np.ndarray: Bipartite adjacency matrix, shape (N, M)
            N: number of residues in df1
            M: number of residues in df2
    """
    df1_nr = df1.drop_duplicates("node_id").reset_index(drop=True)
    df2_nr = df2.drop_duplicates("node_id").reset_index(drop=True)
    adj = np.zeros((len(df1_nr), len(df2_nr)), dtype=np.int8)

    return adj


# generate bipartite edge adjacency matrix
def generate_bipartite_adj_matrix(
    df1: AllAtomDataFrame, df2: AllAtomDataFrame, dist_thr: Union[float, int]
) -> AdjMatrix:
    """
    This is calculating non-hydrogen atom distances between the two input ALL-ATOM structure DataFrames.
    Find contact residues using distance cutoff of 4.5 Angstroms between non-hydrogen atoms. This requires all-atom structure DataFrames.
    Then reduce the ATOM-pair to RESIDUE-pair, and generate a bipartite adjacency matrix.

    Args:
        df1 (PDBDataFrame): ALL-ATOM structure DataFrame 1
        df2 (PDBDataFrame): ALL-ATOM structure DataFrame 2
        dist_thr (Union[float, int]): Distance threshold for contacting residues in Angstroms

    Returns:
        np.ndarray: Bipartite adjacency matrix, shape (N, M)
            N: number of residues in df1
            M: number of residues in df2
    """
    df1_nr = df1.drop_duplicates("node_id").reset_index(drop=True)
    df2_nr = df2.drop_duplicates("node_id").reset_index(drop=True)
    adj = np.zeros((len(df1_nr), len(df2_nr)), dtype=np.int8)

    df1_nr_node_id_to_idx: Dict[str, int] = {
        node_id: idx for idx, node_id in enumerate(df1_nr.node_id)
    }
    df2_nr_node_id_to_idx: Dict[str, int] = {
        node_id: idx for idx, node_id in enumerate(df2_nr.node_id)
    }

    dist_mat = calculate_dist_mat(df1=df1, df2=df2)
    rows, cols = np.where(
        (dist_mat < dist_thr) & (dist_mat > 0)
    )  # >0 => exclude self-loop

    for r, c in zip(df1.iloc[rows, :].node_id, df2.iloc[cols, :].node_id):
        n1, n2 = df1_nr_node_id_to_idx[r], df2_nr_node_id_to_idx[c]
        adj[n1, n2] = 1
    return adj


# generate intra-graph adjacency matrix
def generate_intra_graph_adj(
    df: AllAtomDataFrame, coord_columns: Optional[List[str]]
) -> AdjMatrix:
    """
    Generate intra-graph adjacency matrix from an ALL-ATOM structure DataFrame.
    Assume column `node_id` (Returned by Graphein) exists in the DataFrame.

    Args:
        df (AllAtomDataFrame): all-atom structure DataFrame
        coord_columns (Optional[List[str]]): coordinates columns. Defaults to ["x_coord", "y_coord", "z_coord"].

    Returns:
        AdjMatrix: intra-graph adjacency matrix, shape (N, N)
            where N is the number of residues in the structure
    """
    coord_columns = coord_columns or ["x_coord", "y_coord", "z_coord"]
    x = squareform(pdist(df[coord_columns].values))

    df_nr = df.drop_duplicates("node_id").reset_index(drop=True)
    df_nr_node_id_to_idx: Dict[str, int] = {
        node_id: idx for idx, node_id in enumerate(df_nr.node_id)
    }
    n_residue = len(df_nr)
    adj = np.zeros((n_residue, n_residue)).astype(np.int8)

    rows, cols = np.where((x < DIST_THR) & (x > 0))  # >0 => exclude self-loop
    for r, c in zip(df.iloc[rows, :].node_id, df.iloc[cols, :].node_id):
        n1, n2 = df_nr_node_id_to_idx[r], df_nr_node_id_to_idx[c]
        adj[n1, n2] = 1

    return adj


# map seqres -> atmseq -> surf_mask
def map_seqres_to_atmseq_with_downstream_mask(
    seqres2atmseq_mask: Dict[str, Any], target_mask: np.ndarray
) -> AdjMatrix:
    """
    Map SEQRES to ATMSEQ then to Surface residue mask.

    Args:
        seqres2atmseq_mask (Dict[str, Any]): a dictionary with keys: 'seqres', 'atmseq', 'mask'. This is mapping from SEQRES to ATMSEQ.
            All three str values have the same length L.
        target_mask (np.ndarray): e.g. surface residue mask, this is mapping from ATMSEQ to a downstream residue mask.
            The length of this array must equal to the length of 'atmseq' (exclude '-') in `seqres2atmseq_mask`.

    Returns:
        np.ndarray: a binary mask mapping SEQRES to Surface residues, shape (L, )
    """
    assert len(target_mask) == len(seqres2atmseq_mask["seq"]["atmseq"].replace("-", ""))
    seqres2target_mask, i = [], 0
    for c in seqres2atmseq_mask["seq"]["mask"]:
        if (
            c == "1" or c == 1 or c is True
        ):  # residue exists in the structure with a surface mask
            seqres2target_mask.append(target_mask[i])
            i += 1
        elif c == "0" or c == 0 or c is False:  # missing residue
            seqres2target_mask.append(0)
    return np.array(seqres2target_mask)


# input
def cli() -> Namespace:
    parser = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter)
    # job_id a custom string to identify the inference job
    parser.add_argument(
        "--job_id",
        type=str,
        required=False,
        default="complex",
        help="job id for the inference job, default: complex",
    )

    # antibody and antigen strudctures
    parser.add_argument(
        "--ab_structure", type=Path, required=True, help="path to the ab structure"
    )
    parser.add_argument(
        "--ag_structure", type=Path, required=True, help="path to the ag structure"
    )

    # antibody and antigen SEQRES sequence files (optional)
    parser.add_argument(
        "--ab_seqres",
        type=ast.literal_eval,
        required=False,
        default=None,
        help="path to the ab seqres fasta file, if not provided, use the SEQRES sequence from the PDB file",
    )
    parser.add_argument(
        "--ag_seqres",
        type=ast.literal_eval,
        required=False,
        default=None,
        help="path to the ag seqres fasta file, if not provided, use the SEQRES sequence from the PDB file",
    )
    # antibody and antigen chain ids (optional)
    parser.add_argument(
        "--ab_chain_id",
        type=str,
        required=True,
        nargs="*",
        default=["H", "L"],
        help="ab chain id, if not provided, use default chain id H and L",
    )
    parser.add_argument(
        "--ag_chain_id",
        type=str,
        required=True,
        nargs="*",
        default=["A"],
        help="ag chain id, if not provided, use default chain id A",
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

    args = parser.parse_args()
    return args


# helper: convert a list of PairData to a PygDataLoader
def pyg_data_to_loader(pair_data_list: List[PairData]) -> PygDataLoader:
    """Convert a list of PairData to a PygDataLoader"""
    loader = PygDataLoader(
        dataset=pair_data_list,
        batch_size=1,
        follow_batch=[
            "x_b",
            "x_g",
            "y_b",
            "y_g",
            "edge_index_b",
            "edge_index_g",
            "edge_index_bg",
        ],
    )
    return loader


# helper: convert a PairData or a list of PairData to a PygBatch
def pyg_data_to_batch(pair_data: Union[PairData, List[PairData]]) -> PygBatch:
    """Convert a PairData or a list of PairData to a PygBatch"""
    if isinstance(pair_data, PairData):
        data_list = [pair_data]
    elif isinstance(pair_data, list):
        data_list = pair_data
    batch = PygBatch.from_data_list(
        data_list=data_list,
        follow_batch=[
            "x_b",
            "x_g",
            "y_b",
            "y_g",
            "edge_index_b",
            "edge_index_g",
            "edge_index_bg",
        ],
    )
    return batch


# main: process input to pair graph data in a batch for inference
# def main(args: Union[Dict[str, Any], DictConfig, Namespace]) -> PairData:
def main(
    *,
    ab_structure: Path,
    ag_structure: Path,
    ab_chain_id: List[str],
    ag_chain_id: List[str],
    esm2_ckpt_path: Path,
    job_id: str = "AbAgComplex",
    ab_seqres: Path = None,
    ag_seqres: Path = None,
    is_complex: bool = False,
    log_level: str = "DEBUG",
) -> PairData:
    # if isinstance(args, Dict):
    #     args = OmegaConf.create(args, resolve=True)
    # # log args
    # log_args(args)
    log_args(
        {
            "job_id": job_id,
            "ab_structure": ab_structure,
            "ag_structure": ag_structure,
            "ab_chain_id": ab_chain_id,
            "ag_chain_id": ag_chain_id,
            "esm2_ckpt_path": esm2_ckpt_path,
            "ab_seqres": ab_seqres,
            "ag_seqres": ag_seqres,
            "is_complex": is_complex,
            "log_level": log_level,
        }
    )

    # set logger level to INFO
    logger.remove()  # Remove the default handler
    logger.add(sys.stdout, level=log_level)

    abid = ab_structure.stem
    agid = ag_structure.stem

    # sanity check: SEQRES availability
    if ab_seqres is None:
        try:
            _assert_SEQRES_in_pdb_file(pdb_file=ab_structure)
        except ValueError as e:
            raise ValueError(
                f"No SEQRES found in {ab_structure} and no `ab_seqres` provided"
            ) from e
    if ag_seqres is None:
        try:
            _assert_SEQRES_in_pdb_file(pdb_file=ag_structure)
        except ValueError as e:
            raise ValueError(
                f"No SEQRES found in {ag_structure} and no `ag_seqres` provided"
            ) from e

    # sanity check: fasta header format
    if ab_seqres:
        _assert_fasta_header_format(ab_seqres)
    if ag_seqres:
        _assert_fasta_header_format(ag_seqres)

    # structure graphs
    G_ab = read_pdb_as_graph(ab_structure, GRAPHEIN_CFG, chains=ab_chain_id)
    G_ag = read_pdb_as_graph(ag_structure, GRAPHEIN_CFG, chains=ag_chain_id)

    # structure DataFrames
    ag_atom_df, surf_atom_df = dec_process_ag_graph(process_graph, rsa_thr=RSA_THR)(
        G_ag
    )
    ab_atom_df, cdr_atom_df = dec_process_ab_graph(process_graph, cdr_def=CDR_DEF)(G_ab)

    # **************************** #
    #            EDGES             #
    # **************************** #
    logger.info("Generating edges...")
    # [x] TODO: edge_index_bg
    if is_complex:
        biparite_adj: AdjMatrix = generate_bipartite_adj_matrix(
            df1=cdr_atom_df, df2=surf_atom_df, dist_thr=DIST_THR
        )
    else:
        biparite_adj: AdjMatrix = generate_bipartite_adj_matrix_placeholder(
            df1=cdr_atom_df, df2=surf_atom_df
        )

    # [x] TODO: y_b, y_g
    y_b: BinaryMask = np.zeros(biparite_adj.shape[0]).astype(np.int8)
    y_g: BinaryMask = np.zeros(biparite_adj.shape[1]).astype(np.int8)
    if is_complex:
        y_b[np.where(biparite_adj.sum(axis=1) > 0)] = 1
        y_g[np.where(biparite_adj.sum(axis=0) > 0)] = 1

    # [x] TODO: edge_index_b
    adj_b: AdjMatrix = generate_intra_graph_adj(df=cdr_atom_df, coord_columns=None)

    # [x] TODO: edge_index_g
    adj_g: AdjMatrix = generate_intra_graph_adj(df=surf_atom_df, coord_columns=None)

    logger.info("Done generating edges...")
    # **************************** #
    #            SEQRES            #
    # **************************** #
    # [x] TODO: SEQRES
    if ab_seqres is None:
        seqres = extract_seqres_from_pdb(ab_structure)
    else:
        seqres = {
            r.id.rsplit("|")[1]: str(r.seq) for r in SeqIO.parse(ab_seqres, "fasta")
        }
    ab_seqres = {k: v for k, v in seqres.items() if k in ab_chain_id}
    ab_seqres = OrderedDict({x: ab_seqres[x] for x in "HL" if x in ab_seqres})

    if ag_seqres is None:
        seqres = extract_seqres_from_pdb(ag_structure)
    else:
        seqres = {
            r.id.rsplit("|")[1]: str(r.seq) for r in SeqIO.parse(ag_seqres, "fasta")
        }
    ag_seqres = {k: v for k, v in seqres.items() if k in ag_chain_id}

    # **************************** #
    # EMBEDDING: DOCKER CONTAINERS #
    # **************************** #
    logger.info("Generating embeddings...")
    logger.info("Running IgFold Docker container...")
    # [x] TODO: Docker
    # [x] TODO: SEQRES embeddings
    igfold_emb, _ = dec_log_func(run_igfold_docker_container)(
        antibody_name=abid, antibody_seqres=ab_seqres
    )
    ab_emb = igfold_emb["tensors"]["bert_embs"].squeeze()
    logger.info("Done running IgFold Docker container...")

    logger.info("Running ESM2 Docker container...")
    ag_emb, _ = dec_log_func(run_esm2_docker_container)(
        antigen_name=agid,
        antigen_seqres=ag_seqres,
        model_name="esm2_t12_35M_UR50D",
        esm2_ckpt_host_path=esm2_ckpt_path,
    )
    logger.info("Done running ESM2 Docker container...")

    logger.info("Done generating embeddings")

    # **************************** #
    #         SEQRES2NODES         #
    # **************************** #
    # [x] TODO: AG SEQRES2SURF mapping
    ag_atmseq = "".join(
        ag_atom_df.drop_duplicates("node_id").residue_name.apply(
            lambda x: Polypeptide.three_to_one(x)
        )
    )
    seqres2atmseq_mask = run_seqres2atmseq(
        seqres=list(ag_seqres.values())[0], atmseq=ag_atmseq
    )
    surf_mask = ag_atom_df.drop_duplicates("node_id").is_surface.values.astype(int)
    seqres2surf_mask = map_seqres_to_atmseq_with_downstream_mask(
        seqres2atmseq_mask=seqres2atmseq_mask, target_mask=surf_mask
    )
    x_g = ag_emb[seqres2surf_mask == 1, :]

    # [x] TODO: AB SEQRES2CDR mapping
    seqres = "".join(ab_seqres.values())
    atmseq = "".join(
        ab_atom_df.drop_duplicates("node_id").residue_name.apply(
            lambda x: Polypeptide.three_to_one(x)
        )
    )
    seqres2atmseq_mask = run_seqres2atmseq(seqres=seqres, atmseq=atmseq)
    atmseq2cdr_mask = ab_atom_df.drop_duplicates("node_id").is_cdr.values.astype(int)
    seqres2cdr_mask = map_seqres_to_atmseq_with_downstream_mask(
        seqres2atmseq_mask=seqres2atmseq_mask, target_mask=atmseq2cdr_mask
    )
    seqres2cdr_mask = np.array(seqres2cdr_mask).astype(bool)
    x_b = ab_emb[seqres2cdr_mask == 1, :]

    # **************************** #
    #  BUILD GRAPH REPRESENTATION  #
    # **************************** #
    edge_index_b, edge_index_b_edge_values = dense_to_sparse(torch.from_numpy(adj_b))
    edge_index_g, edge_index_g_edge_values = dense_to_sparse(torch.from_numpy(adj_g))
    edge_index_bg, edge_index_bg_edge_values = dense_to_sparse(
        torch.from_numpy(biparite_adj)
    )
    pair_data = PairData(
        abdbid=job_id,
        x_b=x_b,
        x_g=x_g,
        edge_index_b=edge_index_b,
        edge_index_g=edge_index_g,
        edge_index_bg=edge_index_bg,
        y_b=y_b,
        y_g=y_g,
    )

    return pair_data

# a function name pointing to the main function
create_pair_graph_data = main


# ==================== Main ====================
if __name__ == "__main__":
    args = cli()
    pair_data = main(
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
    logger.info(pair_data)
