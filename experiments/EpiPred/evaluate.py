import argparse
import shutil
import tempfile
import textwrap
from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd
import torch
from biopandas.pdb import PandasPdb
from loguru import logger
from scipy.spatial.distance import cdist
from sklearn.metrics import (f1_score, matthews_corrcoef, precision_score,
                             recall_score, roc_auc_score)
from tqdm import tqdm

# ==================== Configuration ====================
BASE = Path(__file__).resolve().parent
PDBS = BASE.joinpath("epipred-input", "pdbs")
EPIPRED_OUT = BASE.joinpath("epipred-output")
TASK_LIST = BASE.joinpath("assets", "ag-chain-ids.txt")

AATABLE = {
    "ALA": "A",
    "ARG": "R",
    "ASN": "N",
    "ASP": "D",
    "CYS": "C",
    "GLU": "E",
    "GLN": "Q",
    "GLY": "G",
    "HIS": "H",
    "ILE": "I",
    "LEU": "L",
    "LYS": "K",
    "MET": "M",
    "PHE": "F",
    "PRO": "P",
    "SER": "S",
    "THR": "T",
    "TRP": "W",
    "TYR": "Y",
    "VAL": "V",
}


# ==================== Function ====================
def read_pdb_non_std_extension_as_ppdb(pdb_fp: Path) -> PandasPdb:
    """
    Read pdb file with non-standard file extension
    and return a PandasPdb object.

    Raise:
        RuntimeError: if error occurs when reading the file.
    """
    ppdb = None
    try:
        with tempfile.TemporaryDirectory() as tmpdir:
            # copy the file to a temp dir with .pdb extension
            tmp_fp = Path(tmpdir) / f"{pdb_fp.stem}.pdb"
            shutil.copy(pdb_fp, tmp_fp)
            # read the file
            ppdb = PandasPdb().read_pdb(str(tmp_fp))
            return ppdb
    except Exception as e:
        raise RuntimeError(f"Error when reading file: {pdb_fp}\nError info: {e}")


def _load_abdb_pdb_as_df(pdb_fp: Path) -> pd.DataFrame:
    """
    Parse a pdb file and return a atom_df with node_id added.

    Args:
        pdb_fp (Path): pdb file path

    Returns:
        df: pdb structure DataFrame with node_id added
    """
    # parse it as df
    if pdb_fp.suffix != ".pdb":
        ppdb = read_pdb_non_std_extension_as_ppdb(pdb_fp)
    else:
        ppdb = PandasPdb().read_pdb(str(pdb_fp))

    # convert to dataframe
    atom_df = ppdb.df["ATOM"]
    # add node_id in the format of [chain_id]:[residue_name]:[residue_number]:[insertion]
    atom_df["node_id"] = (
        atom_df["chain_id"]
        + ":"
        + atom_df["residue_name"]
        + ":"
        + atom_df["residue_number"].astype(str)
        + ":"
        + atom_df["insertion"]
    )
    # remove the tailing space and colon in the node_id if insertion is empty
    atom_df["node_id"] = atom_df["node_id"].str.replace(r":\s*$", "", regex=True)

    return ppdb.df["ATOM"]


def _load_pred_epitope(
    abdbid: str, ag_chain: str, top_k: int = 1
) -> Dict[int, Dict[str, int]]:
    """
    Load the predicted epitope from EpiPred output.
    And convert to 0-based residue index.

    Args:
        abdbid (str): abdbid e.g. 5f9o_0P
        top_k (int, optional): load top k predictions. Defaults to 1.

    Returns:
        Dict[int, Dict[str, int]]: a dict of top_k predictions,
            and each prediction is a dict of node_idx
    """
    pdb_code = abdbid.split("_")[0]
    pred_dir = EPIPRED_OUT / f"{pdb_code}_{ag_chain}" / "epitope_predictions"

    # a function to parse one prediction file
    def _parse_one(file: Path) -> pd.DataFrame:
        return pd.read_table(file, header=None, names=["resi", "chain"], sep="\s+")
        # ----------------------------------------
        # resi is the raw residue number read from
        # the raw pdb file, need to map it to the
        # renumbered residue index
        # ----------------------------------------

    # number of returned decoys
    n_decoys = len(list(pred_dir.glob("original_*.txt")))
    if n_decoys == 0:
        raise ValueError(f"{abdbid} has no decoys.")
    elif n_decoys < top_k:
        logger.warning(
            f"{abdbid} has less than {top_k} decoys, only {n_decoys} will be evaluated."
        )
        top_k = n_decoys

    # parse the top_k epitope
    result = {}
    for i in range(top_k):
        fp = pred_dir.joinpath(f"original_{i}.txt")
        df = _parse_one(fp)
        result[i] = {"raw_resi": df["resi"].to_numpy()}

    return result, top_k


# abdbid, ag_chain, top_k = '6wzl_1P','F', 5
def main(abdbid: str, ag_chain: str, top_k: int = 1) -> Dict:
    # raw file
    atom_df = _load_abdb_pdb_as_df(PDBS.joinpath(f"pdb{abdbid}.pdb"))

    # split into ab and ag
    ab_df = atom_df.query('chain_id in ["H", "L"]').reset_index(drop=True)
    ag_df = atom_df.query('chain_id not in ["H", "L"]').reset_index(drop=True)

    # distance mat
    dist_mat = cdist(
        ab_df[["x_coord", "y_coord", "z_coord"]],
        ag_df[["x_coord", "y_coord", "z_coord"]],
    )
    _, cols = np.where(dist_mat < 4.5)
    epitope_nodes = ag_df.iloc[cols, :].node_id.unique()

    # ------------------------------------------------------------
    # EpiPred
    # In EpiPred, residues are renumbered starting from 1
    # 1. convert it to 0-based
    #
    # For ground-truth:
    # 1. edge_df: use atom_df to map to renumbered resi (0-based)
    # ------------------------------------------------------------

    # dicts to convert between node_id and idx
    # here idx is 0-based
    idx2nodeId: Dict[int, str] = (
        ag_df.drop_duplicates(["node_id"]).reset_index(drop=True)["node_id"].to_dict()
    )
    # remove AA_3 from node_id e.g. A:TYR:32 -> A:32
    idx2nodeId = {k: v for k, v in idx2nodeId.items()}
    nodeId2idx = {v: k for k, v in idx2nodeId.items()}
    # epitope residue idx
    # idx is renumbered resi, 0-based
    epitope: Dict[str, int] = {k: nodeId2idx[k] for k in epitope_nodes}
    # sort by values
    epitope = dict(sorted(epitope.items(), key=lambda x: x[1]))

    # ground truth
    true_idx = list(epitope.values())
    true_lab = torch.zeros(len(idx2nodeId))
    true_lab[true_idx] = 1

    # top_k pred epitope residue idx
    result, _top_k = _load_pred_epitope(abdbid, ag_chain=ag_chain, top_k=top_k)

    # compare result length and top_k, some entries may have less than top_k decoys
    if _top_k != top_k:
        top_k = _top_k

    # e.g. for i = 0
    d = {
        k.split(":")[-1]: v for k, v in nodeId2idx.items()
    }  # e.g. {'32': 0, '33': 1, ...}
    for k, v in result.items():
        result[k]["renum_resi"] = np.array([d[str(x)] for x in v["raw_resi"]])

    # metrics
    metrics = dict(abdbid=abdbid)

    for i in range(top_k):
        # pred idx
        pred_idx = np.array(result[i]["renum_resi"])
        pred_lab = torch.zeros(len(idx2nodeId))
        pred_lab[pred_idx] = 1

        # calculate metrics using sklearn
        metrics[f"rank{i+1}"] = dict(
            mcc=matthews_corrcoef(true_lab, pred_lab),
            aucroc=roc_auc_score(true_lab, pred_lab),
            precision=precision_score(true_lab, pred_lab),
            recall=recall_score(true_lab, pred_lab),
            f1=f1_score(true_lab, pred_lab),
        )

    return metrics


# ==================== Main ====================
if __name__ == "__main__":
    ag_chain_df = pd.read_table(
        TASK_LIST,
        header=None,
        names=["abdbid", "job_id", "ag_chain"],
        sep=",",
    )

    # iterate over the rows of ag_chain-df[['abdbid', 'ag_chain']]
    for i, row in tqdm(ag_chain_df.iterrows(), total=len(ag_chain_df)):
        try:
            metrics = main(abdbid=row["abdbid"], ag_chain=row["ag_chain"], top_k=3)
            # save metrics
            import json

            with open(
                BASE / "metrics" / f"{row['abdbid']}_{row['ag_chain']}.json", "w"
            ) as f:
                json.dump(metrics, f, indent=4)
        except Exception as e:
            print(
                f"Error when processing abdbid: {row['abdbid']}, ag_chain: {row['ag_chain']}:\n{e}"
            )
            continue
