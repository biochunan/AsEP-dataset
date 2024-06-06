"""
Input:
- Ground truth AbAg complex structure (AbDb file)
- ESMFold predicted complex structure (ESMFold file)
"""

import json
import shutil
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from Bio import SeqIO
from biopandas.pdb import PandasPdb
from loguru import logger
from scipy.spatial.distance import cdist
from sklearn.metrics import (f1_score, matthews_corrcoef, precision_score,
                             recall_score, roc_auc_score)
from tqdm import tqdm
from utils import run_align_clustalomega

# ==================== Configuration ====================
BASE = Path("/workspaces/Antibody-specific-epitope-prediction-2.0/experiments/esmfold")
ABDB = Path("/AbDb")
PRED_STRUCT = BASE / "output"
INPUTFASTA = BASE / "assets" / "walle1723.fasta"
OUTPUT = BASE / "output"
CLUSTAL_OMEGA_EXECUTABLE = shutil.which("clustalo")


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


def renumber_residue_number(df_in: pd.DataFrame) -> pd.DataFrame:
    """
    Renumber residue_number for each chain to 0-based indexing.
    And update node_id accordingly.
    """
    df = df_in.copy()
    # add a placeholder column for renumbered residue id and node_id
    df["new_residue_number"] = np.nan
    # add a placeholder column for renumbered node_id using empty str
    df["node_id_renumbered"] = ""
    for chain_id in df["chain_id"].unique():
        # get the chain
        df_chain = df.query("chain_id == @chain_id").reset_index(drop=True).copy()
        # get residue_name, node_id, idx
        _df = df_chain.drop_duplicates(["node_id"]).reset_index(drop=True)
        residue_name = _df["residue_name"].to_list()
        node_id = _df["node_id"].to_list()
        idx = _df.index.to_list()
        # create mapping node_id -> idx
        nodeId2idx = {n: i for n, i in zip(node_id, idx)}
        nodeId2newNodeId = {
            n: f"{chain_id}:{rn}:{i}" for n, i, rn in zip(node_id, idx, residue_name)
        }
        # update new_residue_number
        df.loc[df["chain_id"] == chain_id, "residue_number_renumbered"] = (
            df.loc[df["chain_id"] == chain_id, "node_id"].map(nodeId2idx).to_list()
        )
        # update node_id_renumbered
        df.loc[df["chain_id"] == chain_id, "node_id_renumbered"] = (
            df.loc[df["chain_id"] == chain_id, "node_id"]
            .map(nodeId2newNodeId)
            .to_list()
        )
    # curate data type
    df["residue_number_renumbered"] = df["residue_number_renumbered"].astype(int)
    return df


def find_interacting_residues(
    complex_df: pd.DataFrame,
    chain_group_A: List[str],
    chain_group_B: Optional[List[str]] = None,
    dist_cutoff: float = 4.5,
) -> Tuple[List[str], List[str]]:
    """
    Find interacting residues between chain_group_A and chain_group_B using a distance cutoff.

    Args:
        complex_df (pd.DataFrame): atom dataframe of the complex
        chain_group_A (List[str]): list of chain ids in `chain_group_A`
        chain_group_B (Optional[List[str]], optional): list of chain ids in `chain_group_B`. Defaults to None.
            if None, use the rest of the chains in complex_df
        dist_cutoff (float, optional): residue-residue contact distance threshold between any non-hydrogen atom pairs
            between a pair of residues in `chain_group_A` and `chain_group_B`. Defaults to 4.5.

    Returns:
        Tuple[List[str], List[str]]: _description_
    """
    # if chain_group_B is not specified, use the rest of the chains in complex_df
    if chain_group_B is None:
        chain_group_B = set(complex_df.chain_id.unique()) - set(chain_group_A)

    # split complex_df into chain_group_A and chain_group_B
    binder_A = complex_df.query(f"chain_id in {chain_group_A}").reset_index(drop=True)
    binder_B = complex_df.query(f"chain_id in {chain_group_B}").reset_index(drop=True)

    # distance mat
    dist_mat = cdist(
        binder_A[["x_coord", "y_coord", "z_coord"]],
        binder_B[["x_coord", "y_coord", "z_coord"]],
    )
    rows, cols = np.where(dist_mat < dist_cutoff)

    # get the interface residues
    interface_A: List[str] = binder_A.iloc[rows, :]
    interface_B: List[str] = binder_B.iloc[cols, :]

    return interface_A, interface_B


def main(abdbid: str, data: pd.DataFrame) -> Dict[str, Any]:
    # true complex
    atom_df_true = _load_abdb_pdb_as_df(ABDB.joinpath(f"pdb{abdbid}.mar"))
    # get pred file name
    fn = (
        data.query("abdbid == @abdbid")
        .apply(lambda x: f"{x['abdbid']}|H:L:{x['ag_chain']}", axis=1)
        .to_list()[0]
    )
    atom_df_pred = _load_abdb_pdb_as_df(OUTPUT.joinpath(f"{fn}.pdb"))

    # --------------------------------------------------------------------------
    # CHAIN MAPPING
    # In ESMFold, it automatically renames the chain_id to A, B, C, ...
    # This is directly mapping the order of input chains separated by ":"
    # i.e. in our case, this is simply mapping H:L:ag_chain -> A:B:C
    # uncomment the following lines to check
    # --------------------------------------------------------------------------
    """
    abdbid = '4f9p_1P'
    def _get_seq(chain_id: str):
        return ''.join(atom_df_pred_renum.query(f'chain_id == "{chain_id}"').drop_duplicates(['new_node_id']).reset_index(drop=True)['residue_name'].map(AATABLE))
    seqs = {chain_id: _get_seq(chain_id) for chain_id in atom_df_pred_renum['chain_id'].unique()}
    _d = data.query('abdbid == @abdbid')
    _d['h_seq'].values[0]  == seqs['A']  # => True
    _d['l_seq'].values[0]  == seqs['B']  # => True
    _d['ag_seq'].values[0] == seqs['C']  # => True
    """
    ag_chain = data.query("abdbid == @abdbid").ag_chain.values[0]
    # update chain id in atom_df_pred
    chain_map = {"A": "H", "B": "L", "C": ag_chain}
    atom_df_pred["chain_id"] = atom_df_pred["chain_id"].map(chain_map)

    # --------------------------------------------------------------------------
    # replace `residue_number` in `node_id` by renumbered residue id
    # renumber each chain separately
    # input: atom_df_pred
    # output: atom_df_pred_renum
    # --------------------------------------------------------------------------
    atom_df_pred_renumbered = renumber_residue_number(df_in=atom_df_pred)
    atom_df_true_renumbered = renumber_residue_number(df_in=atom_df_true)

    # ------------------------------------------------------------
    # 0. Determine the ground-truth & predicted epitope residues
    # using a distance cutoff
    # ------------------------------------------------------------
    paratope_true_df, epitope_true_df = find_interacting_residues(
        complex_df=atom_df_true_renumbered,
        chain_group_A=["H", "L"],
        chain_group_B=[ag_chain],
        dist_cutoff=4.5,
    )
    paratope_pred_df, epitope_pred_df = find_interacting_residues(
        complex_df=atom_df_pred_renumbered,
        chain_group_A=["H", "L"],
        chain_group_B=[ag_chain],
        dist_cutoff=4.5,
    )

    # to node names
    paratope_true = paratope_true_df.node_id_renumbered.unique()
    epitope_true = epitope_true_df.node_id_renumbered.unique()
    paratope_pred = paratope_pred_df.node_id_renumbered.unique()
    epitope_pred = epitope_pred_df.node_id_renumbered.unique()
    """
    e.g.
    paratope_true: ['L:ASN:30', 'L:TYR:49', 'L:TRP:93', 'H:THR:29', 'H:SER:30']
    epitope_true : ['A:ASP:72', 'A:ARG:71', 'A:LYS:53', 'A:ASP:51', 'A:THR:32']
    paratope_pred: ['H:PRO:40', 'H:GLY:41', 'H:SER:90', 'L:VAL:8']
    epitope_pred : ['A:SER:7', 'A:LEU:4', 'A:PHE:25']
    """

    # --------------------------------------------------------------------------
    # Mappings
    # Goal: map both atmseq and seqres
    # 1. mapping epitopes to ATOMSEQ indices
    # 2. mapping ATOMSEQ indices to SEQERS indices
    # NOTE: epitope_pred is already in SEQRES indices,
    # NOTE: because its input is the SEQRES sequence
    # --------------------------------------------------------------------------
    epitope_pred_seqres_idx = [int(x.split(":")[-1]) for x in epitope_pred]
    """
    e.g. [7, 4, 25]
    """

    # TODO [chunan]: perform same mapping for paratope
    # 1.1 align ATOMSEQ to SEQRES
    def _get_seqres2atmseq_mask(
        data: pd.DataFrame, abdbid: str
    ) -> Tuple[List[int], str, str]:
        seqres = data[data.abdbid == abdbid].ag_seq.values[0]
        ag_chain = data[data.abdbid == abdbid].ag_chain.values[0]

        atmseq = (
            atom_df_true_renumbered.query("chain_id in @ag_chain")
            .drop_duplicates(["node_id"])
            .reset_index(drop=True)["residue_name"]
            .to_list()
        )
        atmseq = "".join([AATABLE[x] for x in atmseq])  # to 1-letter AA code
        aln = run_align_clustalomega(
            clustal_omega_executable=CLUSTAL_OMEGA_EXECUTABLE,
            seq1=seqres,
            seq2=atmseq,
        )
        assert "-" not in str(aln[0].seq), "Error: seqres contains dash"
        aln1 = str(aln[1].seq)  # atmseq in aln may contain "-"
        seqres2atmseq = [
            1 if i != "-" else 0 for i in aln1
        ]  # 1 => in atmseq; 0 => not in atmseq
        assert len(seqres2atmseq) == len(seqres)

        return seqres2atmseq, atmseq, seqres

    seqres2atmseq, atmseq, seqres = _get_seqres2atmseq_mask(data=data, abdbid=abdbid)
    """
    print(''.join(map(str, seqres2atmseq)))
    '1111111111....11111111100000011111111111111111111111111'
    """

    # --------------------------------------------------------------------------
    # 1.2 map ATOMSEQ indices to SEQRES indices
    # seqres pos: 0  1  2  3  4  5  6  7  8  9 ...
    # aln mask  : 1  1  1  1  0  0  0  1  1  1
    # atmseq pos: 0  1  2  3  -  -  -  4  5  6 ...
    # '-' means missing residue in ATMSEQ and skipped in the alignment
    # --------------------------------------------------------------------------
    ai, si = 0, 0  # ai => atmseq index; si => seqres index
    ai2siMapping = {}  # ATOMSEQ index -> SEQRES index
    for col_idx, state in enumerate(seqres2atmseq):
        # match, increment both indices
        if state == 1:
            ai2siMapping[ai] = si
            ai += 1
            si += 1
        # missing residue in ATMSEQ, only increment seqres index
        elif state == 0:
            si += 1

    # 2. mapping epitope residues to ATMSEQ indices for epitope_true
    epitope_true_atomseq_idx = [int(x.split(":")[-1]) for x in epitope_true]
    epitope_true_seqres_idx = [ai2siMapping[i] for i in epitope_true_atomseq_idx]
    """
    `epitope_true_seqres_idx` are the true epitope residues in SEQRES indices
    e.g. [72, 71, 53, 51, 32, 31, 29, 52]
    """

    # SEQRES length
    L = len(data.query("abdbid == @abdbid").ag_seq.values[0])

    # ground truth labels
    true_lab = np.zeros(L, dtype=np.int16)
    true_lab[epitope_true_seqres_idx] = 1
    pred_lab = np.zeros(L, dtype=np.int16)
    pred_lab[epitope_pred_seqres_idx] = 1

    # --------------------------------------------------------------------------
    # 3. CALCULATE METRICS
    # Because the input of ESMFold is SEQRES sequence,
    # We use the SEQRES indices to calculate metrics
    # --------------------------------------------------------------------------
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
            ag_chain=ag_chain,
            seqres=seqres,
            atmseq=atmseq,
            seqres2atmseq="".join(map(str, seqres2atmseq)),
            epitope_true_atomseq_idx=epitope_true_atomseq_idx,
            epitope_true_seqres_idx=epitope_true_seqres_idx,
            epitope_pred_seqres_idx=epitope_pred_seqres_idx,
            true_lab=true_lab.tolist(),
            pred_lab=pred_lab.tolist(),
        ),
    )

    return metrics


# ==================== Main ====================
if __name__ == "__main__":
    data = []
    with open(INPUTFASTA, "r") as f:
        for record in SeqIO.parse(f, "fasta"):
            abdbid, chains = record.id.split("|")
            ag_chain = chains.split(":")[-1]
            h_seq, l_seq, ag_seq = record.seq.split(":")
            # to str
            h_seq, l_seq, ag_seq = str(h_seq), str(l_seq), str(ag_seq)
            data.append(
                [
                    abdbid,
                    ag_chain,
                    len(h_seq),
                    len(l_seq),
                    len(ag_seq),
                    h_seq,
                    l_seq,
                    ag_seq,
                ]
            )
    data = pd.DataFrame(
        data,
        columns=[
            "abdbid",
            "ag_chain",
            "h_len",
            "l_len",
            "ag_len",
            "h_seq",
            "l_seq",
            "ag_seq",
        ],
    )
    data.to_csv(BASE / "analysis" / "walle1723.csv", index=False)

    # ESMFold predicted structures i.e. successed jobs
    pred_structs_fps = list(PRED_STRUCT.glob("*.pdb"))
    successed_jobs = [fp.stem.split("|")[0] for fp in pred_structs_fps]

    # --------------------------------------------------------------------------
    # NOTE: ESMFold log protein length is the sum of chain length of the
    # H, L, and Ag chains, plus the 2 special tokens - <cls> and <eos>.
    # i.e. the sum of `data`'s h_len + l_en + ag_len + 2 == length in `failed_jobs`
    # --------------------------------------------------------------------------
    failed_jobs = []
    # find abdbid in data but not in successed_jobs
    for abdbid in data.abdbid.values:
        if abdbid not in successed_jobs:
            failed_jobs.append(abdbid)

    logger.info(f"Total number of jobs: {len(data)}")
    logger.info(f"Number of successed jobs: {len(successed_jobs)}")
    logger.info(f"Number of failed jobs: {len(failed_jobs)}")

    # --------------------------------------------------------------------------
    # ITERATE OVER THE SUCCESSFUL JOBS
    # --------------------------------------------------------------------------
    # abdbid = successed_jobs[0]  # => 4f9p_1P
    # metrics = main(abdbid=abdbid, data=data)
    for abdbid in tqdm(successed_jobs):
        try:
            metrics = main(abdbid=abdbid, data=data)
            # save metrics
            with open(BASE / "metrics" / f"{abdbid}.json", "w") as f:
                json.dump(metrics, f, indent=4)
        except Exception as e:
            print(f"Error when processing abdbid: {abdbid}:\n{e}")
            continue
