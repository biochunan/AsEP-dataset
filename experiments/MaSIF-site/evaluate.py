# basic
import json
from pathlib import Path
from typing import Dict, List

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
INSTRUCT = BASE.joinpath("masif_output", "data_preparation", "01-benchmark_pdbs")
MAPPEDRES = BASE.joinpath("mapped_residues")
ABDB = Path("~/Dataset/AbDb/abdb_newdata").expanduser()
FAILED = BASE.joinpath("failed.txt")
AGCHAIN = BASE.joinpath("ag-chain-ids.txt")

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
def _load_abdb_pdb_as_df(pdb_fp: Path) -> pd.DataFrame:
    """
    Parse a pdb file and return a atom_df with node_id added.

    Args:
        pdb_fp (Path): pdb file path

    Returns:
        df: pdb structure DataFrame with node_id added
    """
    # if pdb_fp suffix is '.mar' make a temp directory copy it in and renamed as suffix .pdb
    if pdb_fp.suffix == ".mar":
        import shutil
        import tempfile

        with tempfile.TemporaryDirectory() as tmpdirname:
            tmp_fp = Path(tmpdirname) / "tmp.pdb"
            shutil.copy(pdb_fp, tmp_fp)
            pdb_fp = tmp_fp
            # parse it as df
            ppdb = PandasPdb().read_pdb(str(pdb_fp))
    else:
        # parse it as df
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


# abdbid, ag_chain, top_k = '6wzl_1P','F', 5
def main(abdbid: str, job_id: str) -> Dict:
    # abdbid, job_id, ag_chain = ag_chain_df.iloc[0].to_list()
    # raw file
    atom_df_raw = _load_abdb_pdb_as_df(ABDB.joinpath(f"pdb{abdbid}.mar"))
    atom_df_in = _load_abdb_pdb_as_df(INSTRUCT.joinpath(f"{job_id}.pdb"))

    # ------------------------------------------------------------
    # 0. Determine the ground-truth epitope residues
    # using a distance cutoff
    # ------------------------------------------------------------
    # split into ab and ag
    ab_df = atom_df_raw.query('chain_id in ["H", "L"]').reset_index(drop=True)
    ag_df = atom_df_raw.query('chain_id not in ["H", "L"]').reset_index(drop=True)
    # distance mat
    dist_mat = cdist(
        ab_df[["x_coord", "y_coord", "z_coord"]],
        ag_df[["x_coord", "y_coord", "z_coord"]],
    )
    _, cols = np.where(dist_mat < 4.5)
    epitope_nodes = ag_df.iloc[cols, :].node_id.unique()

    # ------------------------------------------------------------
    # MaSIF-site
    # Extract predicted residue IDs from the
    # mapped residue txt files
    # ------------------------------------------------------------

    def _load_pred_mapped_epitop_residue_ids(job_id: str) -> List[str]:
        fp = MAPPEDRES.joinpath(f"{job_id}.txt")
        # extract the residue ids
        with open(fp, "r") as f:
            # skip lines until the line starts with "Residue IDs:"
            for line in f:
                if line.startswith("Residue IDs:"):
                    break
            # read the residue ids
            pred_epi_res = [l.strip() for l in f]
        return pred_epi_res

    # predicted epitope residues
    pred_epitope_nodes = _load_pred_mapped_epitop_residue_ids(job_id)

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
    epitope = dict(sorted(epitope.items(), key=lambda x: x[1]))  # sort by values

    # ground truth labels
    true_idx = list(epitope.values())
    true_lab = torch.zeros(len(idx2nodeId))
    true_lab[true_idx] = 1

    # pred labels
    pred_epitope: Dict[str, int] = {k: nodeId2idx[k] for k in pred_epitope_nodes}
    pred_epitope = dict(
        sorted(pred_epitope.items(), key=lambda x: x[1])
    )  # sort by values
    pred_idx = list(pred_epitope.values())
    pred_lab = torch.zeros(len(idx2nodeId))
    pred_lab[pred_idx] = 1

    # calculate metrics using sklearn
    metrics = dict(
        abdbid=abdbid,
        mcc=matthews_corrcoef(true_lab, pred_lab),
        aucroc=roc_auc_score(true_lab, pred_lab),
        precision=precision_score(true_lab, pred_lab, zero_division=0),
        recall=recall_score(true_lab, pred_lab),
        f1=f1_score(true_lab, pred_lab),
        metadata=dict(
            abdbid=abdbid,
            job_id=job_id,
            ag_chain=job_id.split("_")[-1],
            epitope=epitope,
            pred_epitope=pred_epitope,
            true_idx=true_idx,
            pred_idx=pred_idx,
            true_lab=true_lab.type(torch.int).tolist(),
            pred_lab=pred_lab.type(torch.int).tolist(),
        ),
    )

    return metrics


# ==================== Main ====================
if __name__ == "__main__":
    ag_chain_df = pd.read_table(
        AGCHAIN,
        header=None,
        names=["abdbid", "job_id"],
        sep=",",
    )

    # job_ids on which MaSIF-site failed
    failed = pd.read_table(FAILED, header=None, names=["job_id"], sep=",")
    failed["job_id"] = failed["job_id"].str.replace(r"\.txt$", "", regex=True)

    # remove them from job_ids
    ag_chain_df = ag_chain_df[~ag_chain_df["job_id"].isin(failed["job_id"])]

    # add ag_chain and abdbid
    ag_chain_df["ag_chain"] = ag_chain_df["job_id"].apply(lambda x: x.split("_")[-1])

    # iterate over the rows of ag_chain-df[['abdbid', 'ag_chain']]
    for i, row in tqdm(ag_chain_df.iterrows(), total=len(ag_chain_df)):
        try:
            metrics = main(abdbid=row["abdbid"], job_id=row["job_id"])
            # save metrics
            with open(
                BASE / "metrics" / f"{row['abdbid']}_{row['ag_chain']}.json", "w"
            ) as f:
                json.dump(metrics, f, indent=4)
        except Exception as e:
            print(
                f"Error when processing abdbid: {row['abdbid']}, ag_chain: {row['ag_chain']}:\n{e}"
            )
            continue
